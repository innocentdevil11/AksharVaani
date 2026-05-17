"""
LLM-based Math Formatting Module
=================================
Sends transcript text to a generative model (Gemini / Groq) for
STEM context analysis and equation formatting in LaTeX.

Usage:
    python -m aksharvaani.src.alignment.math_formatter --input transcript.txt
"""

import os
from typing import Optional

try:
    import google.generativeai as genai  # type: ignore
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False


def format_math_with_gemini(text: str, api_key: Optional[str] = None) -> str:
    """Use Google Gemini to clean and format STEM equations."""
    if not HAS_GENAI:
        raise ImportError("google-generativeai is required: pip install google-generativeai")

    key = api_key or os.getenv("GEMINI_API_KEY", "")
    if not key:
        raise ValueError("GEMINI_API_KEY not set")

    genai.configure(api_key=key)
    model = genai.GenerativeModel("gemini-2.5-pro")

    prompt = (
        "You are an expert math assistant.\n"
        "I will give you a transcript of spoken words (possibly messy).\n"
        "Tasks:\n"
        "1. Identify if the text is about STEM.\n"
        "2. Rewrite equations in LaTeX math format.\n"
        "3. Provide a cleaned version with equations properly formatted.\n"
        "4. If no math, return a cleaned subject summary.\n\n"
        f"Transcript:\n{text}"
    )

    response = model.generate_content(prompt)
    return response.text.strip()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Format STEM transcripts")
    parser.add_argument("--input", required=True, help="Transcript file")
    parser.add_argument("--output", default="mathified_output.txt")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()

    result = format_math_with_gemini(text)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(result)
    print(f"✅ Formatted output → {args.output}")
