"""
BLIP Vision-Language Equation Extractor
=======================================
Uses Salesforce BLIP (image captioning) with multi-prompt extraction
to understand and extract STEM equations from classroom board images.

Usage:
    python -m aksharvaani.src.vision.blip_extractor Kinematics.png --output results.json
"""

import torch
import json
import time
import logging
import re
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
from PIL import Image, ImageEnhance
from transformers import BlipProcessor, BlipForConditionalGeneration
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class BLIPExtractor:
    """BLIP-based vision-language STEM equation extractor."""

    PROMPTS = [
        "A handwritten mathematical equation or formula",
        "Mathematical expressions and equations",
        "Handwritten math problems and solutions",
        "Write out every mathematical expression and formula",
        "Physics and engineering equations",
    ]

    def __init__(self, device: str = None, model_size: str = "base"):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        model_name = f"Salesforce/blip-image-captioning-{model_size}"
        logger.info(f"Loading BLIP-{model_size} on {device}…")

        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
        ).to(device)
        self.model.eval()
        logger.info(f"✓ BLIP-{model_size} ready")

    @staticmethod
    def _enhance(image: Image.Image) -> Image.Image:
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = ImageEnhance.Contrast(image).enhance(1.2)
        image = ImageEnhance.Sharpness(image).enhance(1.5)
        if max(image.size) > 1024:
            image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        return image

    @staticmethod
    def _has_math(text: str) -> bool:
        if any(c in text for c in "=<>+-*/^∫∑∂"):
            return True
        if re.search(r"(?:sin|cos|tan|log|ln|exp|sqrt)", text, re.IGNORECASE):
            return True
        return False

    def extract(self, image_path: str) -> Dict[str, Any]:
        """Extract equations using multi-prompt BLIP captioning."""
        t0 = time.time()
        try:
            image = self._enhance(Image.open(image_path).convert("RGB"))
            equations = []

            with torch.no_grad():
                for prompt in self.PROMPTS:
                    inputs = self.processor(image, text=prompt, return_tensors="pt").to(self.device)
                    outputs = self.model.generate(**inputs, max_length=150, num_beams=3)
                    caption = self.processor.decode(outputs[0], skip_special_tokens=True)

                    if self._has_math(caption) and len(caption) > 3:
                        equations.append({"text": caption.strip(), "confidence": 0.7, "source": prompt})

            # Deduplicate
            unique = {}
            for eq in equations:
                key = eq["text"].lower()
                if key not in unique:
                    unique[key] = eq
            equations = list(unique.values())

            return {
                "model": "BLIP-Vision",
                "equations": equations,
                "total_detected": len(equations),
                "inference_time": round(time.time() - t0, 3),
            }
        except Exception as e:
            return {"model": "BLIP-Vision", "error": str(e), "equations": [], "total_detected": 0,
                    "inference_time": round(time.time() - t0, 3)}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BLIP equation extraction")
    parser.add_argument("image", help="Image path")
    parser.add_argument("--output", help="Output JSON")
    parser.add_argument("--device", choices=["cuda", "cpu"])
    args = parser.parse_args()

    ext = BLIPExtractor(device=args.device)
    result = ext.extract(args.image)
    print(json.dumps(result, indent=2))

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
