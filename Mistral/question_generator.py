from transformers import AutoTokenizer, AutoModelForCausalLM
from pypdf import PdfReader
import torch
import json
import os

# Modell laden
model_id = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.float32,
    device_map="auto"
)

# PDF lesen
pdf_path = input("Pfad zur Datei: ").strip().strip('"')
reader = PdfReader(pdf_path)

pdf_text = ""
for page in reader.pages:
    text = page.extract_text()
    if text:
        pdf_text += text + "\n\n"

# Eingabeparameter
bloom_level = input("Bloom-Level (1-6): ")
temperature = 0.2
text_chunk = pdf_text

# Prompt
prompt = f"""
System role:
You are an experienced university instructor who creates high-quality exam questions
based on the six cognitive levels of Bloom's Taxonomy.

Bloom levels:
1 Remember – recall facts
2 Understand – explain concepts
3 Apply – use knowledge in a situation
4 Analyze – identify relationships
5 Evaluate – justify decisions
6 Create – produce new ideas

Task:
Generate one quiz question for the following teaching text that corresponds exactly
to the specified Bloom level. The question must test understanding at that cognitive level.

Requirements:
1) Precise and clear academic language
2) Explicit assignment to a Bloom level (1–6)
3) The Bloom level must be explicitly stated
4) Maximum 40 words per question
5) Provide answer options A–D and mark the correct answer
6) Indicate from which section of the teaching text the question was derived

Input parameters:
Bloom-Level: {bloom_level}
Temperature: {temperature}
Teaching text:
{text_chunk}

Output format:
Bloom-Level: <level>
Question: <question text>
A) ...
B) ...
C) ...
D) ...
Correct answer: <letter>
Source in teaching text: <chapter or section>
"""

# Tokenisieren
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generieren
outputs = model.generate(
    **inputs,
    max_new_tokens=400,
    temperature=temperature,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

# Antwort dekodieren
generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
print(response)

# Output-Ordner erstellen
os.makedirs("output", exist_ok=True)

# In JSON speichern
with open("output/quiz.json", "w", encoding="utf-8") as f:
    json.dump({"quiz": response}, f, indent=2, ensure_ascii=False)


