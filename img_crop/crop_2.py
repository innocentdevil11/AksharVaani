import cv2
import os


def preprocess_blackboard(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    _, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return binary, enhanced


def split_with_overlap(img, output_folder, overlap_px=200):
    h, w = img.shape[:2]
    mid_h, mid_w = h // 2, w // 2

    crops = {
        "top_left": img[0: mid_h + overlap_px, 0: mid_w + overlap_px],
        "top_right": img[0: mid_h + overlap_px, mid_w - overlap_px: w],
        "bottom_left": img[mid_h - overlap_px: h, 0: mid_w + overlap_px],
        "bottom_right": img[mid_h - overlap_px: h, mid_w - overlap_px: w],
    }

    os.makedirs(output_folder, exist_ok=True)

    paths = []
    for name, crop in crops.items():
        path = os.path.join(output_folder, f"{name}.jpg")
        cv2.imwrite(path, crop)
        paths.append(path)

    return paths


def split_into_rows(img, output_folder, row_height=140, overlap=50):
    h, w = img.shape[:2]
    os.makedirs(output_folder, exist_ok=True)

    paths = []
    for y in range(0, h, row_height - overlap):
        y_end = min(h, y + row_height)
        row = img[y:y_end, 0:w]
        path = os.path.join(output_folder, f"row_{y}_{y_end}.jpg")
        cv2.imwrite(path, row)
        paths.append(path)

    return paths


# ===========================
# 🔥 MAIN WRAPPER FUNCTION
# ===========================

def crop_blackboard(
    image_path,
    output_root="crop_output",
    overlap_px=200,
    row_height=140
):
    """
    Master cropping wrapper
    Returns list of image paths
    """

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    os.makedirs(output_root, exist_ok=True)

    binary, enhanced = preprocess_blackboard(img)

    cv2.imwrite(os.path.join(output_root, "binary.jpg"), binary)
    cv2.imwrite(os.path.join(output_root, "enhanced.jpg"), enhanced)

    quadrant_dir = os.path.join(output_root, "quadrants")
    row_dir = os.path.join(output_root, "rows")

    quadrant_paths = split_with_overlap(
        enhanced, quadrant_dir, overlap_px
    )
    row_paths = split_into_rows(
        enhanced, row_dir, row_height
    )

    return quadrant_paths,row_paths