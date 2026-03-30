from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
import traceback
import json
import re
import sys
import time
from pickle import TRUE
from typing import Any

import torch
from pypdf import PdfReader
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    class tqdm:  # type: ignore[no-redef]
        def __init__(self, iterable=None, total=None, desc="", unit=""):
            self.iterable = iterable
            self.total = total
            self.desc = desc

        def __iter__(self):
            if self.iterable is None:
                return iter(())
            return iter(self.iterable)

        def set_description(self, desc):
            self.desc = desc

        def update(self, _=1):
            return None

        def close(self):
            return None


MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
BASE_DIR = Path(r"C:\Users\Marcel Wagner\PycharmProjects\PythonProject1")
TEACHING_MATERIAL_DIR = BASE_DIR / "Teaching_Material"
OUTPUT_DIR = BASE_DIR / "Mistral" / "output"
LOG_DIR = OUTPUT_DIR / "logs"

TEMPERATURE = 0.2
QUESTION_COUNT_PER_LEVEL = 1
MAX_CONTEXT_CHARS = 6000
MAX_NEW_TOKENS = 700
MAX_GENERATION_ATTEMPTS = 3
MIN_TEMPERATURE = 0.01

BLOOM_LEVELS = {
    "1": "Wissen",
    "2": "Verstehen",
    "3": "Anwenden",
    "4": "Analysieren",
    "5": "Bewerten",
    "6": "Erstellen",
}


class GenerationError(Exception):
    def __init__(self, message: str, bloom_level: str, last_raw_response: str):
        super().__init__(message)
        self.bloom_level = bloom_level
        self.last_raw_response = last_raw_response


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


def get_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=get_dtype(),
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    return tokenizer, model


def collect_pdf_paths(material_dir: Path) -> list[Path]:
    pdf_paths = sorted(material_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"Keine PDF-Dateien gefunden in: {material_dir}")
    return pdf_paths


def prompt_for_pdf_selection(pdf_paths: list[Path]) -> Path:
    print("\nVerfuegbare PDFs:")
    for index, pdf_path in enumerate(pdf_paths, start=1):
        print(f"  {index}.) {pdf_path.stem}")

    while True:
        user_input = input(
            "\nWelche PDF soll verarbeitet werden? Bitte die Nummer eingeben: "
        ).strip()
        if not user_input:
            print("Bitte eine gueltige Auswahl eingeben.")
            continue

        if user_input.isdigit():
            selected_index = int(user_input)
            if 1 <= selected_index <= len(pdf_paths):
                return pdf_paths[selected_index - 1]

        print("Auswahl nicht gefunden. Bitte erneut eingeben.")


def prompt_for_temperature(default_temperature: float) -> float:
    while True:
        user_input = input(
            f"Temperatur eingeben (Enter fuer Standard {default_temperature}): "
        ).strip()
        if not user_input:
            return default_temperature

        try:
            temperature = float(user_input.replace(",", "."))
        except ValueError:
            print("Ungueltiger Zahlenwert. Beispiel: 0.2")
            continue

        if temperature <= 0:
            print("Die Temperatur muss groesser als 0 sein.")
            continue

        return temperature


def prompt_for_output_filename(selected_pdf: Path) -> Path:
    default_filename = f"quiz_questions_{selected_pdf.stem}.json"

    while True:
        user_input = input(
            f"Name der Output-Datei eingeben (Enter fuer {default_filename}): "
        ).strip()

        filename = user_input or default_filename
        if not filename.lower().endswith(".json"):
            filename = f"{filename}.json"

        if any(character in filename for character in '<>:"/\\|?*'):
            print("Der Dateiname enthaelt ungueltige Zeichen. Bitte erneut eingeben.")
            continue

        return OUTPUT_DIR / filename


def build_console_log_path(selected_pdf: Path, output_file: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return LOG_DIR / f"console_{selected_pdf.stem}_{output_file.stem}_{timestamp}.log"


@contextmanager
def console_logging(log_path: Path):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    with log_path.open("a", encoding="utf-8") as log_file:
        tee_stdout = TeeStream(original_stdout, log_file)
        tee_stderr = TeeStream(original_stderr, log_file)
        sys.stdout = tee_stdout
        sys.stderr = tee_stderr
        try:
            yield
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def prompt_to_continue() -> bool:
    while True:
        user_input = input(
            "\nWeitere PDF verarbeiten? [J/n/q]: "
        ).strip().casefold()
        if user_input in {"", "j", "ja", "y", "yes"}:
            return True
        if user_input in {"n", "nein", "q", "quit", "exit"}:
            return False
        print("Bitte 'j' oder 'n' eingeben.")


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    minutes, remaining_seconds = divmod(total_seconds, 60)
    hours, remaining_minutes = divmod(minutes, 60)

    if hours:
        return f"{hours}h {remaining_minutes:02d}m {remaining_seconds:02d}s"
    if minutes:
        return f"{minutes}m {remaining_seconds:02d}s"
    return f"{remaining_seconds}s"


def get_resource_snapshot() -> str:
    resource_parts = []

    if psutil is not None:
        process = psutil.Process()
        rss_bytes = process.memory_info().rss
        resource_parts.append(f"RAM Prozess: {rss_bytes / (1024 ** 3):.2f} GB")

        virtual_memory = psutil.virtual_memory()
        resource_parts.append(
            f"RAM System: {virtual_memory.percent:.0f}% belegt "
            f"({virtual_memory.available / (1024 ** 3):.2f} GB frei)"
        )

    if torch.cuda.is_available():
        device_index = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(device_index) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(device_index) / (1024 ** 3)
        total = torch.cuda.get_device_properties(device_index).total_memory / (1024 ** 3)
        resource_parts.append(
            f"GPU {device_index}: {allocated:.2f}/{total:.2f} GB belegt, "
            f"{reserved:.2f} GB reserviert"
        )
    else:
        resource_parts.append("GPU: nicht aktiv")

    return " | ".join(resource_parts)


def print_resource_snapshot(prefix: str):
    print(f"{prefix}{get_resource_snapshot()}")


def write_error_log(
    error: Exception,
    selected_pdf: Path,
    bloom_level: str | None,
    temperature: float,
    last_raw_response: str = "",
):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bloom_suffix = f"_bloom_{bloom_level}" if bloom_level is not None else ""
    log_path = LOG_DIR / f"error_{selected_pdf.stem}{bloom_suffix}_{timestamp}.log"

    log_lines = [
        f"Zeit: {datetime.now().isoformat()}",
        f"PDF: {selected_pdf.name}",
        f"Bloom-Stufe: {bloom_level or 'unbekannt'}",
        f"Temperatur: {temperature}",
        f"Fehlertyp: {type(error).__name__}",
        f"Fehlermeldung: {error}",
        "",
        f"Ressourcen: {get_resource_snapshot()}",
        "",
        "Traceback:",
        traceback.format_exc(),
    ]

    if last_raw_response:
        log_lines.extend(
            [
                "",
                "Letzte Modellantwort:",
                last_raw_response,
            ]
        )

    log_path.write_text("\n".join(log_lines), encoding="utf-8")
    print(f"Fehlerprotokoll gespeichert unter: {log_path}")


def extract_candidate_json_text(raw_text: str) -> str:
    cleaned_text = raw_text.strip()
    cleaned_text = cleaned_text.replace("```json", "").replace("```JSON", "").replace("```", "")
    cleaned_text = cleaned_text.replace("“", '"').replace("”", '"').replace("’", "'")

    start = cleaned_text.find("{")
    end = cleaned_text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Kein JSON-Objekt in der Modellantwort gefunden.")

    candidate = cleaned_text[start:end + 1]
    candidate = re.sub(r",(\s*[}\]])", r"\1", candidate)
    return candidate


def build_retry_prompt(base_prompt: str, attempt: int, error_message: str) -> str:
    return (
        f"{base_prompt}\n\n"
        f"Wiederholung Versuch {attempt}. "
        "Die vorherige Antwort war nicht valide. "
        "Gib jetzt nur ein einziges vollstaendiges JSON-Objekt aus. "
        "Kein einleitender Text. Kein abschliessender Text. Kein Markdown. "
        "Beginne direkt mit { und ende direkt mit }. "
        f"Letzter Validierungsfehler: {error_message}"
    )


def normalize_text(text: str) -> str:
    text = text.replace("\u00ad", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_text_pages(pdf_path: Path) -> tuple[list[dict[str, Any]], int, int]:
    pages: list[dict[str, Any]] = []
    skipped_pages = 0
    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)

    for page_number, page in enumerate(reader.pages, start=1):
        text = normalize_text(page.extract_text() or "")
        if not text:
            skipped_pages += 1
            continue

        pages.append(
            {
                "page": page_number,
                "text": text,
                "method": "text",
            }
        )

    return pages, skipped_pages, total_pages


def read_pdf_as_chapter(pdf_path: Path) -> dict[str, Any]:
    pages, skipped_page_count, total_pages = extract_text_pages(pdf_path)

    if not pages:
        raise ValueError(
            f"Aus {pdf_path.name} konnte kein Text gelesen werden. "
            "Die PDF enthaelt vermutlich keinen extrahierbaren Text."
        )

    chapter_name = pdf_path.stem
    return {
        "chapter_name": chapter_name,
        "source_pdf": pdf_path.name,
        "total_pages": total_pages,
        "text_page_count": len(pages),
        "skipped_page_count": skipped_page_count,
        "pages": pages,
    }


def build_chapter_context(chapter: dict[str, Any], max_chars: int) -> tuple[str, list[dict[str, Any]]]:
    selected_pages = []
    snippets = []
    total_chars = 0

    for page in chapter["pages"]:
        snippet = f"[Quelle: {chapter['source_pdf']}, Seite {page['page']}, Methode {page['method']}]\n{page['text']}"
        if total_chars + len(snippet) > max_chars and snippets:
            break
        snippets.append(snippet)
        selected_pages.append(
            {
                "datei": chapter["source_pdf"],
                "seite": page["page"],
                "methode": page["method"],
            }
        )
        total_chars += len(snippet)

    return "\n\n".join(snippets), selected_pages


def build_prompt(chapter_name: str, source_pdf: str, bloom_level: str, context_text: str) -> str:
    bloom_label = BLOOM_LEVELS[bloom_level]
    bloom_specific_instruction = (
        '7) Fuer Bloom-Stufe 6 ("Erstellen") formuliere die Frage als Auswahl zwischen mehreren moeglichen '
        'Entwuerfen, Konzepten, Vorgehensplaenen oder Loesungsansaetzen.\n'
        '8) Geprueft wird, welcher Vorschlag unter den gegebenen Zielen und Randbedingungen am besten geeignet ist.\n'
        '9) Die Lernenden muessen die beste konstruktive Option auswaehlen, nicht selbst frei etwas erstellen.\n'
        '10) Die falschen Optionen sollen plausible, aber im Kontext weniger geeignete Entwuerfe oder Vorgehensweisen sein.\n'
        if bloom_level == "6"
        else ""
    )
    return f"""
Du bist eine erfahrene Hochschullehrperson und erstellst pruefungsnahe Multiple-Choice-Fragen fuer Lehrmaterial.

Aufgabe:
Erzeuge fuer das Kapitel "{chapter_name}" aus der Datei "{source_pdf}" genau {QUESTION_COUNT_PER_LEVEL} Multiple-Choice-Fragen zur Bloom-Stufe {bloom_level} ({bloom_label}).

Fachliche Regeln:
1) Verwende ausschliesslich Informationen aus dem bereitgestellten Lehrtext.
2) Erfinde keine Inhalte, Fakten, Begriffe oder Seitenangaben.
3) Die Fragen muessen fachlich korrekt, klar und praezise auf Deutsch formuliert sein.
4) Jede Frage muss genau 4 Antwortoptionen haben: A, B, C, D.
5) Genau eine Antwort darf korrekt sein.
6) Jede Frage muss eindeutig zur Bloom-Stufe {bloom_level} ({bloom_label}) passen.
{bloom_specific_instruction}11) Jede Frage soll sich auf das Kapitel "{chapter_name}" beziehen.
12) Gib fuer jede Frage die Quelle als Dateiname und Seitenzahl an.

Formatregeln:
1) Gib ausschliesslich gueltiges JSON zurueck.
2) Gib kein Markdown und keine Erklaerungen aus.
3) Gib genau ein einziges JSON-Objekt aus.
4) Die Antwort muss direkt mit {{ beginnen und direkt mit }} enden.
5) Gib vor oder nach dem JSON keinen weiteren Text aus.
6) Das JSON muss exakt dieses Format haben:
{{
  "chapter_name": "{chapter_name}",
  "bloom_level": "{bloom_level}",
  "bloom_name": "{bloom_label}",
  "questions": [
    {{
      "question": "string",
      "options": {{
        "A": "string",
        "B": "string",
        "C": "string",
        "D": "string"
      }},
      "correct_answer": "A",
      "source": "Dateiname.pdf, Seite 3"
    }}
  ]
}}

Es muessen genau {QUESTION_COUNT_PER_LEVEL} Fragen im Array "questions" enthalten sein.

Lehrtext:
{context_text}
""".strip()


def generate_model_output(tokenizer, model, prompt: str, temperature: float) -> str:
    messages = [{"role": "user", "content": prompt}]
    model_inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
    )

    if isinstance(model_inputs, torch.Tensor):
        input_ids = model_inputs.to(model.device)
        attention_mask = torch.ones_like(input_ids)
    else:
        model_inputs = model_inputs.to(model.device)
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

    with torch.inference_mode():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][input_ids.shape[1]:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def extract_json_payload(raw_text: str) -> dict[str, Any]:
    candidate = extract_candidate_json_text(raw_text)
    return json.loads(candidate)


def normalize_question(question_data: dict[str, Any]) -> dict[str, Any]:
    options = question_data.get("options", {})
    correct_answer = str(question_data.get("correct_answer", "")).strip().upper()
    if correct_answer not in {"A", "B", "C", "D"}:
        raise ValueError("Ungueltige korrekte Antwort in der Modellausgabe.")

    normalized = {
        "question": normalize_text(str(question_data["question"])),
        "options": {
            "A": normalize_text(str(options["A"])),
            "B": normalize_text(str(options["B"])),
            "C": normalize_text(str(options["C"])),
            "D": normalize_text(str(options["D"])),
        },
        "correct_answer": correct_answer,
        "source": normalize_text(str(question_data["source"])),
    }
    return normalized


def validate_generation_payload(payload: dict[str, Any], chapter_name: str, bloom_level: str) -> dict[str, Any]:
    questions = payload.get("questions")
    if not isinstance(questions, list):
        raise ValueError("Das Feld 'questions' fehlt oder ist kein Array.")
    if len(questions) != QUESTION_COUNT_PER_LEVEL:
        raise ValueError(
            f"Es wurden nicht genau {QUESTION_COUNT_PER_LEVEL} Fragen erzeugt: {len(questions)}"
        )

    normalized_questions = [normalize_question(question) for question in questions]

    return {
        "chapter_name": chapter_name,
        "bloom_level": bloom_level,
        "bloom_name": BLOOM_LEVELS[bloom_level],
        "questions": normalized_questions,
    }


def generate_questions_for_level(
    tokenizer,
    model,
    chapter: dict[str, Any],
    bloom_level: str,
    temperature: float,
) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
    context_text, used_context = build_chapter_context(chapter, MAX_CONTEXT_CHARS)
    base_prompt = build_prompt(chapter["chapter_name"], chapter["source_pdf"], bloom_level, context_text)
    last_raw_response = ""
    last_error = ""

    for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
        prompt = base_prompt if attempt == 1 else build_retry_prompt(base_prompt, attempt, last_error)
        print(
            f"  Versuch {attempt}/{MAX_GENERATION_ATTEMPTS} "
            f"mit Temperatur {temperature:.3f}"
        )
        last_raw_response = generate_model_output(tokenizer, model, prompt, temperature)
        print("  Rohantwort des Modells:")
        print(last_raw_response[:4000])
        if len(last_raw_response) > 4000:
            print("  ... [Rohantwort gekuerzt]")
        print("  --- Ende Rohantwort ---")
        try:
            payload = extract_json_payload(last_raw_response)
            validated = validate_generation_payload(payload, chapter["chapter_name"], bloom_level)
            return validated, last_raw_response, used_context
        except Exception as error:
            last_error = f"Versuch {attempt}: {error}"
            print(f"  Antwort ungueltig: {error}")

    raise GenerationError(
        f"Keine valide Modellantwort fuer Kapitel '{chapter['chapter_name']}' "
        f"und Bloom-Stufe {bloom_level}: {last_error}",
        bloom_level=bloom_level,
        last_raw_response=last_raw_response,
    )


def print_generation_summary(result: dict[str, Any]):
    print("\n--- Generierungsuebersicht ---\n")
    for chapter in result["chapters"]:
        print(f"Kapitel: {chapter['chapter_name']} ({chapter['source_pdf']})")
        for level in chapter["bloom_levels"]:
            if "questions" in level:
                print(
                    f"  Bloom-Stufe {level['bloom_level']} ({level['bloom_name']}): "
                    f"{len(level['questions'])} Fragen"
                )
            else:
                print(
                    f"  Bloom-Stufe {level['bloom_level']} ({level['bloom_name']}): "
                    f"FEHLER - {level.get('error', 'Unbekannter Fehler')}"
                )
        print()


def save_output(result: dict[str, Any], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)


def process_selected_pdf(
    tokenizer,
    model,
    selected_pdf: Path,
    selected_temperature: float,
    output_file: Path,
    console_log_path: Path,
):
    chapter = read_pdf_as_chapter(selected_pdf)
    total_steps = len(BLOOM_LEVELS)
    progress_bar = tqdm(total=total_steps, desc="Generierung", unit="stufe")
    total_start_time = time.perf_counter()

    final_result: dict[str, Any] = {
        "model": MODEL_ID,
        "temperature": selected_temperature,
        "chapters": [],
    }

    try:
        print(f"\nAusgewaehlte PDF: {selected_pdf.name}")
        print(f"Temperatur: {selected_temperature}")
        print(f"Ausgabedatei: {output_file}")
        print(f"Konsolenlog: {console_log_path}")
        print(
            f"Seiten gesamt: {chapter['total_pages']} | "
            f"Seiten mit Text: {chapter['text_page_count']} | "
            f"Uebersprungene Seiten: {chapter['skipped_page_count']}"
        )
        print_resource_snapshot("Ressourcen vor Start: ")

        chapter_result = {
            "chapter_name": chapter["chapter_name"],
            "source_pdf": chapter["source_pdf"],
            "bloom_levels": [],
        }
        final_result["chapters"].append(chapter_result)

        for bloom_level, bloom_name in BLOOM_LEVELS.items():
            progress_bar.set_description(
                f"{chapter['chapter_name']} | Bloom {bloom_level} ({bloom_name})"
            )
            completed_steps = len(chapter_result["bloom_levels"])
            elapsed_before_step = time.perf_counter() - total_start_time
            if completed_steps > 0:
                average_seconds = elapsed_before_step / completed_steps
                remaining_steps = total_steps - completed_steps
                estimated_remaining = average_seconds * remaining_steps
                print(
                    f"\nStarte Bloom-Stufe {bloom_level} ({bloom_name}) | "
                    f"bisher: {format_duration(elapsed_before_step)} | "
                    f"Rest geschaetzt: {format_duration(estimated_remaining)}"
                )
            else:
                print(f"\nStarte Bloom-Stufe {bloom_level} ({bloom_name})")

            print_resource_snapshot("  Ressourcen: ")
            level_start_time = time.perf_counter()
            try:
                questions_payload, raw_response, used_context = generate_questions_for_level(
                    tokenizer=tokenizer,
                    model=model,
                    chapter=chapter,
                    bloom_level=bloom_level,
                    temperature=selected_temperature,
                )
                level_duration = time.perf_counter() - level_start_time
                chapter_result["bloom_levels"].append(
                    {
                        "bloom_level": questions_payload["bloom_level"],
                        "bloom_name": questions_payload["bloom_name"],
                        "questions": questions_payload["questions"],
                        "used_context": used_context,
                        "raw_response": raw_response,
                    }
                )
                progress_bar.update(1)
                total_elapsed = time.perf_counter() - total_start_time
                average_seconds = total_elapsed / len(chapter_result["bloom_levels"])
                remaining_steps = total_steps - len(chapter_result["bloom_levels"])
                estimated_remaining = average_seconds * remaining_steps
                print(
                    f"Fertig: Bloom-Stufe {bloom_level} ({bloom_name}) in {format_duration(level_duration)} | "
                    f"gesamt: {format_duration(total_elapsed)} | "
                    f"Rest geschaetzt: {format_duration(estimated_remaining)}"
                )
                print_resource_snapshot("  Ressourcen nach Stufe: ")
            except GenerationError as error:
                chapter_result["bloom_levels"].append(
                    {
                        "bloom_level": bloom_level,
                        "bloom_name": bloom_name,
                        "error": str(error),
                    }
                )
                progress_bar.update(1)
                write_error_log(
                    error=error,
                    selected_pdf=selected_pdf,
                    bloom_level=error.bloom_level,
                    temperature=selected_temperature,
                    last_raw_response=error.last_raw_response,
                )
                print(f"Bloom-Stufe {bloom_level} fehlgeschlagen: {error}")
            except Exception as error:
                chapter_result["bloom_levels"].append(
                    {
                        "bloom_level": bloom_level,
                        "bloom_name": bloom_name,
                        "error": str(error),
                    }
                )
                progress_bar.update(1)
                write_error_log(
                    error=error,
                    selected_pdf=selected_pdf,
                    bloom_level=bloom_level,
                    temperature=selected_temperature,
                )
                print(f"Bloom-Stufe {bloom_level} fehlgeschlagen: {error}")
            finally:
                save_output(final_result, output_file)
    finally:
        progress_bar.close()

    print_generation_summary(final_result)
    print(f"Gespeichert unter: {output_file}")


def main():
    print(f"Lehrmaterial-Ordner: {TEACHING_MATERIAL_DIR}")
    print("Lade Modell und Tokenizer einmalig ...")
    tokenizer, model = load_model_and_tokenizer()
    print_resource_snapshot("Ressourcen nach Modell-Laden: ")

    while True:
        pdf_paths = collect_pdf_paths(TEACHING_MATERIAL_DIR)
        selected_pdf = prompt_for_pdf_selection(pdf_paths)
        selected_temperature = prompt_for_temperature(TEMPERATURE)
        output_file = prompt_for_output_filename(selected_pdf)
        console_log_path = build_console_log_path(selected_pdf, output_file)

        try:
            with console_logging(console_log_path):
                print(f"Konsolenausgabe wird gespeichert unter: {console_log_path}")
                process_selected_pdf(
                    tokenizer=tokenizer,
                    model=model,
                    selected_pdf=selected_pdf,
                    selected_temperature=selected_temperature,
                    output_file=output_file,
                    console_log_path=console_log_path,
                )
        except Exception as error:
            write_error_log(
                error=error,
                selected_pdf=selected_pdf,
                bloom_level=None,
                temperature=selected_temperature,
            )
            print(f"Verarbeitung fehlgeschlagen: {error}")

        if not prompt_to_continue():
            break


if __name__ == "__main__":
    main()
