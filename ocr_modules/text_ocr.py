import cv2
from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang="en")

def run_text_ocr(image_paths):
    """
    image_paths: list[str]
    """
    results = []

    for img_path in image_paths:
        ocr_result = ocr.ocr(img_path)
        for line in ocr_result:
            for word in line:
                results.append(word[1][0])

    return "\n".join(results)
