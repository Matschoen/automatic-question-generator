import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

BASE_DIR = Path(r"/")
TEACHING_MATERIAL_DIR = BASE_DIR / "Teaching_Material"
OUTPUT_DIR = BASE_DIR / "OpenAI" / "output"
ENV_FILE = BASE_DIR / "OpenAI" / ".env"

QUESTION_COUNT_PER_LEVEL = 1
MAX_OUTPUT_TOKENS = 1000

BLOOM_LEVELS = {
    "1": "Wissen",
    "2": "Verstehen",
    "3": "Anwenden",
    "4": "Analysieren",
    "5": "Bewerten",
    "6": "Erstellen"
}


def load_openai_client() -> OpenAI:
    load_dotenv(ENV_FILE)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            f"OPENAI_API_KEY ist nicht gesetzt. Bitte in {ENV_FILE} oder als Umgebungsvariable hinterlegen."
        )
    return OpenAI(api_key=api_key)


def get_model_id() -> str:
    return os.getenv("OPENAI_MODEL", "gpt-4o-mini")


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
        user_input = input("\nWelche PDF soll verarbeitet werden? Bitte Nummer eingeben: ").strip()
        if user_input.isdigit():
            selected_index = int(user_input)
            if 1 <= selected_index <= len(pdf_paths):
                return pdf_paths[selected_index - 1]
        print("Ungueltige Auswahl. Bitte erneut eingeben.")


def prompt_for_temperature(default_temperature: float = 0.2) -> float:
    while True:
        user_input = input(
            f"Temperatur eingeben (Enter fuer Standard {default_temperature}): "
        ).strip()

        if not user_input:
            return default_temperature

        try:
            temperature = float(user_input.replace(",", "."))
        except ValueError:
            print("Bitte einen gueltigen Zahlenwert eingeben, z. B. 0.2")
            continue

        if temperature < 0:
            print("Die Temperatur darf nicht negativ sein.")
            continue

        return temperature


def prompt_to_continue() -> bool:
    while True:
        user_input = input("\nNeue Fragen generieren? [J/n]: ").strip().casefold()
        if user_input in {"", "j", "ja", "y", "yes"}:
            return True
        if user_input in {"n", "nein", "q", "quit", "exit"}:
            return False
        print("Bitte 'j' oder 'n' eingeben.")


def prompt_for_output_filename(selected_pdf: Path) -> Path:
    default_filename = f"quiz_questions_{selected_pdf.stem}.json"

    while True:
        user_input = input(
            f"Name der JSON-Datei eingeben (Enter fuer {default_filename}): "
        ).strip()

        filename = user_input or default_filename
        if not filename.lower().endswith(".json"):
            filename = f"{filename}.json"

        if any(character in filename for character in '<>:"/\\|?*'):
            print("Der Dateiname enthaelt ungueltige Zeichen. Bitte erneut eingeben.")
            continue

        return OUTPUT_DIR / filename


def read_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages: list[str] = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(f"[Seite {page_number}]\n{text}")

    if not pages:
        raise ValueError(f"Aus {pdf_path.name} konnte kein Text extrahiert werden.")

    return "\n\n".join(pages)


def build_prompt(pdf_name: str, bloom_level: str, bloom_name: str, pdf_text: str) -> str:
    bloom_extra = ""
    if bloom_level == "6":
        bloom_extra = (
            'Fuer Bloom-Stufe 6 ("Erstellen") soll die Frage so gestellt werden, '
            "dass zwischen mehreren sinnvollen Entwuerfen, Konzepten oder Vorgehensweisen "
            "die beste Option ausgewaehlt werden muss."
        )

    return f"""
Du bist eine erfahrene Hochschullehrperson und erstellst pruefungsnahe Multiple-Choice-Fragen auf Deutsch.

Aufgabe:
Erzeuge genau {QUESTION_COUNT_PER_LEVEL} Multiple-Choice-Frage(n) zur Bloom-Stufe {bloom_level} ({bloom_name})
auf Basis des bereitgestellten Teaching Materials aus der Datei "{pdf_name}".

Wichtige Regeln:
1) Verwende ausschliesslich Informationen aus dem bereitgestellten PDF-Text.
2) Erfinde keine Inhalte.
3) Jede Frage muss genau 4 Antwortoptionen haben: A, B, C, D.
4) Es duerfen eine bis vier Antworten korrekt sein.
5) Mindestens eine und hoechstens vier Antworten muessen korrekt sein.
6) Speichere alle korrekten Antworten als Liste von Buchstaben unter "correct_answers".
7) Die Frage muss klar zur Bloom-Stufe {bloom_level} ({bloom_name}) passen.
8) Formuliere kurz, klar und praezise.
9) Gib als source den Dateinamen und eine passende Seitenzahl an.
10) Gib nur gueltiges JSON zurueck, ohne Markdown, ohne Einleitung, ohne Erklaerung.
11) Gib genau ein JSON-Objekt zurueck.
12) Speichere die richtige Antwort als Buchstabe unter "correct_answer".

{bloom_extra}

Das JSON muss exakt dieses Format haben:
{{
  "bloom_level": "{bloom_level}",
  "bloom_name": "{bloom_name}",
  "questions": [
    {{
      "question": "string",
      "options": {{
        "A": "string",
        "B": "string",
        "C": "string",
        "D": "string"
      }},
      "correct_answers": ["A", "C"]
      "source": "{pdf_name}, Seite 1"
    }}
  ]
}}

Teaching Material:
{pdf_text}
""".strip()


def generate_questions_for_level(
    client: OpenAI,
    pdf_name: str,
    pdf_text: str,
    bloom_level: str,
    bloom_name: str,
    temperature: float,
) -> dict:
    prompt = build_prompt(pdf_name, bloom_level, bloom_name, pdf_text)

    response = client.responses.create(
        model=get_model_id(),
        input=prompt,
        temperature=temperature,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        text={"format": {"type": "json_object"}},
    )

    output_text = getattr(response, "output_text", "").strip()
    if not output_text:
        raise ValueError(f"Keine Antwort vom Modell fuer Bloom-Stufe {bloom_level}.")

    payload = json.loads(output_text)

    if "questions" not in payload or not isinstance(payload["questions"], list):
        raise ValueError(f"Ungueltige JSON-Antwort fuer Bloom-Stufe {bloom_level}.")

    return payload


def save_output(result: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)


def run_generation(client: OpenAI) -> None:
    pdf_paths = collect_pdf_paths(TEACHING_MATERIAL_DIR)
    selected_pdf = prompt_for_pdf_selection(pdf_paths)
    temperature = prompt_for_temperature()
    output_file = prompt_for_output_filename(selected_pdf)

    print(f"\nLese PDF: {selected_pdf.name}")
    pdf_text = read_pdf_text(selected_pdf)

    result = {
        "source_pdf": selected_pdf.name,
        "model": get_model_id(),
        "temperature": temperature,
        "questions_by_bloom": [],
    }

    for bloom_level, bloom_name in BLOOM_LEVELS.items():
        print(f"Generiere Fragen fuer Bloom-Stufe {bloom_level} ({bloom_name}) ...")
        payload = generate_questions_for_level(
            client=client,
            pdf_name=selected_pdf.name,
            pdf_text=pdf_text,
            bloom_level=bloom_level,
            bloom_name=bloom_name,
            temperature=temperature,
        )
        result["questions_by_bloom"].append(payload)

    save_output(result, output_file)

    print("\nFertig. JSON gespeichert unter:")
    print(output_file)


def main() -> None:
    client = load_openai_client()

    while True:
        run_generation(client)
        if not prompt_to_continue():
            break


if __name__ == "__main__":
    main()
