"""
Circuit / Diagram Analyser
==========================
OpenCV-based contour detection for circuit diagrams and STEM figures.
Detects components (resistors, capacitors, transistors) and connection
topology without requiring any ML model.

Usage:
    python -m aksharvaani.src.vision.diagram_analyser circuit.png
"""

import cv2
import json
import os
from typing import Dict, List, Any


class DiagramAnalyser:
    """Deterministic circuit-diagram analysis using OpenCV contour detection."""

    def analyse(self, image_path: str) -> Dict[str, Any]:
        """Detect components and connections in a diagram image.

        Returns
        -------
        dict
            ``{components, connections, total, description}``
        """
        abs_path = os.path.abspath(image_path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Image not found: {abs_path}")

        img = cv2.imread(abs_path)
        if img is None:
            raise ValueError(f"Cannot read image: {abs_path}")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        components = self._detect_components(gray)
        connections = self._detect_connections(gray, components)

        # Build textual description for accessibility
        description = self._build_description(components, connections)

        return {
            "components": components,
            "connections": connections,
            "total": len(components),
            "description": description,
        }

    # ── Component Detection ──────────────────────────────────────────────

    def _detect_components(self, gray) -> List[Dict]:
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        components = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100:
                continue

            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            x, y, w, h = cv2.boundingRect(cnt)

            if len(approx) == 4:
                aspect = w / h if h > 0 else 0
                if 3 < aspect < 5:
                    ctype, conf = "resistor", 0.85
                else:
                    ctype, conf = ("ic" if w > 40 else "capacitor"), 0.75
            elif len(approx) == 3:
                ctype, conf = "transistor", 0.70
            else:
                ctype, conf = "component", 0.65

            components.append({
                "type": ctype, "confidence": round(conf, 2),
                "position": [int(x), int(y), int(w), int(h)],
            })

        return components

    # ── Connection Detection ─────────────────────────────────────────────

    def _detect_connections(self, gray, components) -> List[Dict]:
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, 3.14 / 180, 50, minLineLength=30, maxLineGap=10)
        if lines is None:
            return []

        connections = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            start = self._nearest(x1, y1, components)
            end = self._nearest(x2, y2, components)
            if start is not None and end is not None and start != end:
                connections.append({"from": start, "to": end})
        return connections

    @staticmethod
    def _nearest(x, y, components, threshold=40):
        nearest, min_dist = None, threshold
        for i, c in enumerate(components):
            cx, cy, cw, ch = c["position"]
            dist = ((x - cx - cw / 2) ** 2 + (y - cy - ch / 2) ** 2) ** 0.5
            if dist < min_dist:
                min_dist, nearest = dist, i
        return nearest

    # ── Accessibility Description ────────────────────────────────────────

    @staticmethod
    def _build_description(components, connections) -> str:
        if not components:
            return "No components detected in the diagram."

        types = {}
        for c in components:
            types[c["type"]] = types.get(c["type"], 0) + 1

        parts = [f"{count} {name}{'s' if count > 1 else ''}" for name, count in types.items()]
        desc = f"Diagram contains {len(components)} components: {', '.join(parts)}."

        if connections:
            desc += f" {len(connections)} connection(s) detected between components."
        return desc


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python diagram_analyser.py <image_path>")
        sys.exit(1)

    analyser = DiagramAnalyser()
    result = analyser.analyse(sys.argv[1])
    print(json.dumps(result, indent=2))
