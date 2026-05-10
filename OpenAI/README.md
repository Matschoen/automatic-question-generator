# OpenAI Question Generator

Dieses Skript erzeugt Multiple-Choice-Fragen aus PDF-basiertem Lehrmaterial mit der OpenAI-API.

Die Fragen werden für alle sechs Bloom-Stufen erzeugt und als JSON-Datei gespeichert.

## Datei

Der Einstiegspunkt ist:

- `question_generator_openai.py`

## Voraussetzungen

Benötigte Pakete in der virtuellen Umgebung:

```powershell
pip install openai python-dotenv pypdf
```

## API-Key

Der OpenAI-Key wird aus einer `.env`-Datei im `OpenAI`-Ordner geladen.

Datei:

- `.env`

Beispielinhalt:

```env
OPENAI_API_KEY=dein_openai_api_key
OPENAI_MODEL=gpt-4o-mini
```

Hinweise:

- `OPENAI_API_KEY` ist erforderlich.
- `OPENAI_MODEL` ist optional. Standard ist `gpt-4o-mini`.
- Die Datei `OpenAI/.env` ist per Root-`.gitignore` vom Commit ausgeschlossen.

## Ordnerstruktur

```text
PythonProject1/
|- OpenAI/
|  |- .env
|  |- output/
|  |- question_generator_openai.py
|  |- README.md
|- Teaching_Material/
|  |- Kapitel3.pdf
|  |- Kapitel5.pdf
|  |- Kapitel7.pdf
```

## Start

Aus dem Projektverzeichnis:

```powershell
.venv\Scripts\python.exe .\OpenAI\question_generator_openai.py
```

## Ablauf

Beim Start fragt das Skript nacheinander:

1. Welche PDF verarbeitet werden soll
2. Welche Temperatur verwendet werden soll
3. Wie die JSON-Datei heissen soll

Danach:

- wird der Text aus der ausgewaehlten PDF gelesen
- fuer jede Bloom-Stufe eine Frage erzeugt
- alles in einer JSON-Datei gespeichert
- gefragt, ob direkt weitere Fragen erzeugt werden sollen

## Bloom-Stufen

Das Skript verwendet diese sechs Stufen:

1. Wissen
2. Verstehen
3. Anwenden
4. Analysieren
5. Bewerten
6. Erstellen

Fuer Stufe 6 wird die Aufgabe als Auswahl der besten Loesung, des besten Entwurfs oder des geeignetsten Vorgehens formuliert.

## Ausgabeformat

Die Ausgabe wird als JSON in `OpenAI/output/` gespeichert.

Beispielstruktur:

```json
{
  "source_pdf": "Kapitel3.pdf",
  "model": "gpt-4o-mini",
  "temperature": 0.2,
  "questions_by_bloom": [
    {
      "bloom_level": "1",
      "bloom_name": "Wissen",
      "questions": [
        {
          "question": "....",
          "options": {
            "A": "....",
            "B": "....",
            "C": "....",
            "D": "...."
          },
          "correct_answer": "B",
          "source": "Kapitel3.pdf, Seite 2"
        }
      ]
    }
  ]
}
```

## Konfiguration

Wichtige Konstanten in `question_generator_openai.py`:

- `QUESTION_COUNT_PER_LEVEL = 1`
- `MAX_OUTPUT_TOKENS = 1000`

Die Modellwahl erfolgt ueber:

- `OPENAI_MODEL` in `OpenAI/.env`

Die Temperatur wird zur Laufzeit interaktiv abgefragt und direkt an die OpenAI-API uebergeben.

## Grenzen

- Das Skript nutzt den aus der PDF extrahierten Text direkt und ohne OCR-Fallback.
- Wenn eine PDF keinen extrahierbaren Text enthaelt, bricht der Lauf mit einem Fehler ab.
- Das Skript prueft aktuell nicht auf doppelte oder sehr aehnliche Fragen.
- Die fachliche Richtigkeit der erzeugten Fragen sollte manuell kontrolliert werden.
