import cv2
import os
import hashlib
from paddleocr import FormulaRecognition

# Load ONCE (important)
model = FormulaRecognition(
    model_name="PP-FormulaNet_plus-M"
)

MAX_WIDTH = 3000


def get_formula_model():
    global _formula_model
    if _formula_model is None:
        _formula_model = FormulaRecognition(
            model_name="PP-FormulaNet_plus-M"
        )
    return _formula_model

def img_hash(img):
    return hashlib.md5(img.tobytes()).hexdigest()


def looks_like_formula(text_hint=""):
    math_tokens = ['=', '+', '-', '\\', '∂', '∫', '^', '_']
    return any(tok in text_hint for tok in math_tokens)


def run_formula_ocr(image_paths, output_dir="formula_output"):

    os.makedirs(output_dir, exist_ok=True)

    seen = set()
    all_results = []

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Width cap (critical)
        if img.shape[1] > MAX_WIDTH:
            img = img[:, :MAX_WIDTH]

        h = img_hash(img)
        if h in seen:
            continue
        seen.add(h)

        # OPTIONAL: plug text OCR hint here later
        # if not looks_like_formula(text_hint):
        #     continue

        results = model.predict(
            input=img,
            batch_size=1
        )

        for res in results:
            res.save_to_img(save_path=output_dir)
            res.save_to_json(save_path=os.path.join(output_dir, "results.json"))
            all_results.append(res)

    # ---- FUTURE LLM MERGE (COMMENTED) ----
    # merged_output = llm_merge(text_results, formula_results)
    # return merged_output

    return all_results
