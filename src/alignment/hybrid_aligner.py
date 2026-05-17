"""
Hybrid OCR-STT Alignment Module
================================
Cross-validates OCR output with STT transcripts using LLM reasoning.
Corrects equation noise by providing both text and speech context.

Usage:
    python -m aksharvaani.src.alignment.hybrid_aligner \
        --segments segments.json --transcript transcript.json --output aligned.json
"""

import json, os, re, logging
from typing import List, Dict, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def is_equation_like(text: str) -> bool:
    """Heuristic check for math content."""
    return bool(re.search(r"[=<>+\-*/^∫∑∂]", text)) or bool(
        re.search(r"(?:sin|cos|tan|log|ln|exp|sqrt)", text, re.IGNORECASE)
    )


def find_matching_transcript(segment: dict, transcript: List[dict]) -> Optional[str]:
    """Find the transcript chunk with the most variable overlap."""
    seg_vars = set(re.findall(r"[a-zA-Z]", segment.get("content", "")))
    best, best_score = None, 0
    for t in transcript:
        t_vars = set(re.findall(r"[a-zA-Z]", t.get("text", "")))
        overlap = len(seg_vars & t_vars)
        if overlap > best_score:
            best_score, best = overlap, t["text"]
    return best


def build_correction_prompt(ocr_text: str, transcript_text: str) -> str:
    return (
        "You are correcting OCR noise in STEM equations.\n\n"
        f"OCR text:\n{ocr_text}\n\n"
        f"Transcript context:\n{transcript_text}\n\n"
        "Rules:\n- Fix formatting errors only.\n- Do NOT invent new equations.\n"
        "- Preserve all variables.\n- If uncertain, return OCR text unchanged.\n"
        "- Return only the corrected equation, nothing else."
    )


def call_groq(prompt: str, model: str = "llama-3.3-70b-versatile") -> str:
    """Call Groq API for text correction."""
    try:
        from groq import Groq
        client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1, max_tokens=200,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"Groq call failed: {e}")
        return ""


def validate_equation(text: str) -> bool:
    """Try symbolic parsing with sympy."""
    try:
        import sympy
        sympy.sympify(text.split("=")[-1].strip())
        return True
    except Exception:
        return False


def align_segments(segments: List[dict], transcript: List[dict]) -> List[dict]:
    """Run hybrid alignment on all equation-like segments."""
    aligned = []
    for seg in segments:
        content = seg.get("content", "")
        if not is_equation_like(content):
            aligned.append(seg)
            continue

        match = find_matching_transcript(seg, transcript)
        if not match:
            aligned.append(seg)
            continue

        prompt = build_correction_prompt(content, match)
        corrected = call_groq(prompt)

        if corrected and corrected != content:
            valid = validate_equation(corrected)
            seg["aligned_content"] = corrected
            seg["validated"] = valid
            seg["original_content"] = content
            if valid:
                seg["content"] = corrected
        aligned.append(seg)

    return aligned


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hybrid OCR-STT alignment")
    parser.add_argument("--segments", required=True)
    parser.add_argument("--transcript", required=True)
    parser.add_argument("--output", default="aligned_segments.json")
    args = parser.parse_args()

    with open(args.segments) as f:
        segs = json.load(f)
    with open(args.transcript) as f:
        trans = json.load(f)

    result = align_segments(segs, trans)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"✅ Aligned {len(result)} segments → {args.output}")
