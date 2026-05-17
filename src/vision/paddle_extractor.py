"""
PaddleOCR STEM Equation Extractor
==================================
Production-grade OCR for handwritten mathematical equations using PaddleOCR v3
with regex-based equation pattern matching, normalisation, and confidence scoring.

Usage:
    python -m aksharvaani.src.vision.paddle_extractor Kinematics.png --output results.json
"""

import json
import time
import logging
import re
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
from PIL import Image
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class PaddleOCRExtractor:
    """PaddleOCR-based STEM equation extractor with pattern matching."""

    EQUATION_PATTERNS = {
        "standard_eq":  r"[a-zA-Z_]\s*(?:\([^)]*\))?\s*=\s*[^;.!?\n]*",
        "inequality":   r"[a-zA-Z_0-9]\s*[<>≤≥≠≈∝]\s*[^;.!?\n]*",
        "integral":     r"∫\s*[^d]*d[a-zA-Z]\s*(?:=|$)",
        "derivative":   r"(?:d|∂)[a-zA-Z0-9]\s*/\s*d[a-zA-Z]\s*[=+\-]\s*[^;.!?\n]*",
        "limit":        r"(?:lim|limit)\s*[^=]*=\s*[^;.!?\n]*",
        "function":     r"[a-zA-Z_]\s*\([^)]*\)\s*(?:=|:|:=)\s*[^;.!?\n]*",
        "matrix":       r"\[\s*[^]]*\s*\]",
    }

    def __init__(self, use_gpu: bool = False):
        from paddleocr import PaddleOCR

        logger.info("Loading PaddleOCR v3…")
        self.ocr = PaddleOCR(use_angle_cls=True, lang="en")
        self.use_gpu = use_gpu
        logger.info("✓ PaddleOCR initialised")

    # ── OCR Recognition ──────────────────────────────────────────────────

    def _ocr_recognise(self, image: Image.Image) -> List[Dict[str, Any]]:
        img_np = np.array(image)
        results = self.ocr.ocr(img_np)

        if not results or not results[0]:
            return []

        regions = []
        for page in results:
            if not page:
                continue
            for line in page:
                try:
                    bbox, (text, confidence) = line[0], (line[1][0], line[1][1])
                    text = str(text).strip()
                    if text:
                        regions.append({"text": text, "confidence": float(confidence), "bbox": bbox})
                except Exception:
                    continue

        return regions

    # ── Text Normalisation ───────────────────────────────────────────────

    @staticmethod
    def _normalise(text: str) -> str:
        fixes = {
            r"(?<![a-zA-Z])I(?![a-zA-Z])": "1",
            r"(?<![a-zA-Z])O(?![a-zA-Z=])": "0",
            r"\s*\^\s*": "^", r"\s*\*\s*": "*",
            r"\s*=\s*": "=", r"\s{2,}": " ",
        }
        for pat, rep in fixes.items():
            text = re.sub(pat, rep, text, flags=re.IGNORECASE)
        return text.strip()

    # ── Pattern Matching ─────────────────────────────────────────────────

    def _match_pattern(self, text: str):
        best = (False, "none", 0.0)
        for name, pat in self.EQUATION_PATTERNS.items():
            if re.search(pat, text, re.IGNORECASE):
                conf = 0.7 + (0.15 if "=" in text else 0) + (0.1 if re.search(r"[a-zA-Z_]\s*\(", text) else 0)
                if min(0.95, conf) > best[2]:
                    best = (True, name, min(0.95, conf))
        if not best[0] and any(c in text for c in "=<>+-*/^∫∑∂"):
            return (True, "general", 0.65)
        return best

    # ── Equation Extraction ──────────────────────────────────────────────

    def _extract_equations(self, ocr_results: List[Dict]) -> List[Dict]:
        equations = []
        for r in ocr_results:
            norm = self._normalise(r["text"])
            if len(norm) < 2:
                continue
            is_eq, ptype, pconf = self._match_pattern(norm)
            if is_eq:
                final_conf = r["confidence"] * 0.35 + pconf * 0.65
                equations.append({
                    "text": norm,
                    "confidence": round(min(0.99, final_conf), 3),
                    "pattern": ptype,
                })

        # Deduplicate
        unique = {}
        for eq in equations:
            key = eq["text"].lower()
            if key not in unique or eq["confidence"] > unique[key]["confidence"]:
                unique[key] = eq
        return sorted(unique.values(), key=lambda x: x["confidence"], reverse=True)

    # ── Public API ───────────────────────────────────────────────────────

    def extract(self, image_path: str) -> Dict[str, Any]:
        """Extract STEM equations from an image.

        Returns
        -------
        dict
            ``{model, equations, inference_time, total_detected, …}``
        """
        t0 = time.time()
        try:
            image = Image.open(image_path).convert("RGB")
            if max(image.size) > 1024:
                image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)

            ocr_results = self._ocr_recognise(image)
            equations = self._extract_equations(ocr_results)
            elapsed = time.time() - t0
            avg = round(np.mean([e["confidence"] for e in equations]), 3) if equations else 0

            return {
                "model": "PaddleOCR-v3",
                "equations": equations,
                "total_detected": len(equations),
                "inference_time": round(elapsed, 3),
                "avg_confidence": avg,
                "text_regions": len(ocr_results),
            }
        except Exception as e:
            return {"model": "PaddleOCR-v3", "error": str(e), "equations": [], "total_detected": 0,
                    "inference_time": round(time.time() - t0, 3)}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract STEM equations with PaddleOCR")
    parser.add_argument("image", help="Image path")
    parser.add_argument("--output", help="Output JSON path")
    args = parser.parse_args()

    ext = PaddleOCRExtractor()
    result = ext.extract(args.image)
    print(json.dumps(result, indent=2))

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
