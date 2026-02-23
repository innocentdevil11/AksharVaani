# llm_merge.py

import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


SYSTEM_PROMPT = """
You are AksharVaani, an AI teaching assistant for Indian classrooms.

Your task:
- Merge professor speech transcription with OCR-extracted board content
- Align formulas with explanations
- Remove repetition
- Fix grammar but preserve teaching tone
- Output structured, clean lecture notes

Rules:
1. DO NOT hallucinate new content
2. Preserve mathematical notation (LaTeX allowed)
3. Prefer clarity over verbosity
4. Use headings, bullets, and equations
5. Keep explanation student-friendly
"""


def merge_modalities(
    transcript_segments,
    text_ocr,
    formula_ocr,
    model="llama-3.1-8b-instant"
):
    """
    transcript_segments : list of {start, end, text}
    text_ocr           : string
    formula_ocr        : string
    """

    transcript_text = "\n".join(
        [f"[{s['start']:.2f}-{s['end']:.2f}] {s['text']}" for s in transcript_segments]
    )

    USER_PROMPT = f"""
### PROFESSOR SPEECH TRANSCRIPTION
{transcript_text}

---

### BOARD TEXT (OCR)
{text_ocr}

---

### BOARD FORMULAS (OCR)
{formula_ocr}

---

### TASK
Merge all content into **well-structured lecture notes**.

Output format:
- Topic title
- Concept explanations
- Inline formulas
- Bullet points where helpful
- No timestamps in final output
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT}
        ],
        temperature=0.2,
        max_tokens=1200
    )

    return response.choices[0].message.content
