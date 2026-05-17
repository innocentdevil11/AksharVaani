"""
Multi-Model Extraction Benchmark Runner
========================================
Runs multiple OCR/VLM extractors and compiles comparative reports.

Usage:
    python -m aksharvaani.src.vision.benchmark image.png --models all
"""

import json, logging, importlib
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class MultiModelBenchmark:
    MODELS = {
        "paddle": {"name": "PaddleOCR-v3", "module": "aksharvaani.src.vision.paddle_extractor", "class": "PaddleOCRExtractor"},
        "blip": {"name": "BLIP-Vision", "module": "aksharvaani.src.vision.blip_extractor", "class": "BLIPExtractor", "kwargs": {"device": "cpu"}},
    }

    def __init__(self, image_path: str, output_dir: str = "benchmark_results"):
        self.image_path = image_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results: Dict[str, Any] = {}

    def run(self, model_keys: List[str] = None) -> Dict[str, Any]:
        if model_keys is None or "all" in (model_keys or []):
            model_keys = list(self.MODELS.keys())
        for key in model_keys:
            cfg = self.MODELS.get(key)
            if not cfg:
                continue
            try:
                mod = importlib.import_module(cfg["module"])
                cls = getattr(mod, cfg["class"])
                extractor = cls(**cfg.get("kwargs", {}))
                self.results[cfg["name"]] = extractor.extract(self.image_path)
                logger.info(f"✓ {cfg['name']}: {self.results[cfg['name']].get('total_detected', 0)} equations")
            except Exception as e:
                self.results[cfg["name"]] = {"error": str(e), "equations": [], "total_detected": 0}
                logger.error(f"✗ {cfg['name']}: {e}")
        return self.results

    def save(self) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = self.output_dir / f"benchmark_{ts}.json"
        with open(out, "w") as f:
            json.dump(self.results, f, indent=2)
        return out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark OCR models")
    parser.add_argument("image")
    parser.add_argument("--models", nargs="+", default=["all"])
    parser.add_argument("--output", default="benchmark_results")
    args = parser.parse_args()
    bench = MultiModelBenchmark(args.image, args.output)
    bench.run(args.models)
    bench.save()
