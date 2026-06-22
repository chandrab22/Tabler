import os

import cv2

# --- CONFIGURATION FOR MODEL DOWNLOAD PATHS ---
# Set environment variables BEFORE importing paddleocr
project_root = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(project_root, "_models")
if not os.path.exists(models_dir):
    os.makedirs(models_dir, exist_ok=True)

os.environ["PADDLE_HOME"] = models_dir
os.environ["PADDLEX_HOME"] = models_dir
os.environ["PADDLEOCR_DICT_PATH"] = models_dir
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
if "OCR_CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["OCR_CUDA_VISIBLE_DEVICES"]
else:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

print(f"ocr_reader: Models directory set to {models_dir}")

from config import (
    CLASS_NAME_COLUMN,
    CLASS_NAME_COLUMN_HEADER,
    CLASS_NAME_ROW,
    CLASS_NAME_ROW_HEADER,
    CLASS_NAME_SPLITTED_COLUMN,
    CLASS_NAME_SPANNING_COLUMN_CELL,
    CLASS_NAME_SPANNING_COLUMN_HEADER,
    CLASS_NAME_SPANNING_ROW_CELL,
    OCR_CROP_MARGIN_MIN,
    OCR_CROP_MARGIN_X_RATIO,
    OCR_CROP_MARGIN_Y_RATIO,
    OCR_ENABLE_MKLDNN,
    OCR_LANG,
    OCR_TEXT_DETECTION_MODEL_NAME,
    OCR_TEXT_RECOGNITION_MODEL_NAME,
    OCR_USE_ANGLE_CLS,
    OCR_USE_DOC_ORIENTATION,
    OCR_USE_DOC_UNWARPING,
    OCR_USE_TEXTLINE_ORIENTATION,
)

print("ocr_reader: Module loaded.")

_ocr_instance = None
_RELEVANT_OCR_CLASSES = {
    CLASS_NAME_COLUMN,
    CLASS_NAME_ROW,
    CLASS_NAME_COLUMN_HEADER,
    CLASS_NAME_ROW_HEADER,
    CLASS_NAME_SPANNING_COLUMN_HEADER,
    CLASS_NAME_SPLITTED_COLUMN,
    CLASS_NAME_SPANNING_ROW_CELL,
    CLASS_NAME_SPANNING_COLUMN_CELL,
}


def _normalize_class_name(name):
    return str(name or "").replace(" ", "_").lower()


def _prediction_to_xyxy(pred):
    return [
        pred["x"] - pred["width"] / 2,
        pred["y"] - pred["height"] / 2,
        pred["x"] + pred["width"] / 2,
        pred["y"] + pred["height"] / 2,
    ]


def _shift_polygon(poly, dx, dy):
    shifted = []
    for pt in poly:
        shifted.append([float(pt[0]) + dx, float(pt[1]) + dy])
    return shifted


def get_ocr_crop_box(predictions, image_source):
    if not predictions:
        return None

    image = None
    if isinstance(image_source, str):
        image = cv2.imread(image_source)
        if image is None:
            return None
        img_h, img_w = image.shape[:2]
    elif hasattr(image_source, "shape"):
        img_h, img_w = image_source.shape[:2]
    else:
        try:
            img_h, img_w = image_source[:2]
        except Exception:
            return None

    relevant_boxes = []
    for pred in predictions:
        if not isinstance(pred, dict):
            continue
        if _normalize_class_name(pred.get("class")) not in _RELEVANT_OCR_CLASSES:
            continue
        try:
            relevant_boxes.append(_prediction_to_xyxy(pred))
        except KeyError:
            continue

    if not relevant_boxes:
        return None

    margin_x = max(OCR_CROP_MARGIN_MIN, int(img_w * OCR_CROP_MARGIN_X_RATIO))
    margin_y = max(OCR_CROP_MARGIN_MIN, int(img_h * OCR_CROP_MARGIN_Y_RATIO))

    x1 = max(0, int(min(box[0] for box in relevant_boxes) - margin_x))
    y1 = max(0, int(min(box[1] for box in relevant_boxes) - margin_y))
    x2 = min(int(img_w), int(max(box[2] for box in relevant_boxes) + margin_x))
    y2 = min(int(img_h), int(max(box[3] for box in relevant_boxes) + margin_y))

    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def get_ocr_instance():
    global _ocr_instance
    if _ocr_instance is None:
        print("Initializing PaddleOCR instance...")
        try:
            from paddleocr import PaddleOCR

            try:
                _ocr_instance = PaddleOCR(
                    lang=OCR_LANG,
                    text_detection_model_name=OCR_TEXT_DETECTION_MODEL_NAME,
                    text_recognition_model_name=OCR_TEXT_RECOGNITION_MODEL_NAME,
                    enable_mkldnn=OCR_ENABLE_MKLDNN,
                    use_doc_orientation_classify=OCR_USE_DOC_ORIENTATION,
                    use_doc_unwarping=OCR_USE_DOC_UNWARPING,
                    use_textline_orientation=OCR_USE_TEXTLINE_ORIENTATION,
                )
            except Exception as init_err:
                err_msg = str(init_err)
                if any(
                    token in err_msg
                    for token in (
                        "Unknown argument",
                        "unexpected keyword argument",
                        "use_doc_orientation_classify",
                        "use_doc_unwarping",
                        "use_textline_orientation",
                    )
                ):
                    print(
                        "PaddleOCR init failed with new-style args "
                        f"({err_msg}), retrying with legacy args..."
                    )
                    _ocr_instance = PaddleOCR(
                        use_angle_cls=OCR_USE_ANGLE_CLS,
                        lang=OCR_LANG,
                        text_detection_model_name=OCR_TEXT_DETECTION_MODEL_NAME,
                        text_recognition_model_name=OCR_TEXT_RECOGNITION_MODEL_NAME,
                        enable_mkldnn=OCR_ENABLE_MKLDNN,
                    )
                else:
                    raise
            print("PaddleOCR initialized.")
        except ImportError:
            print("PaddleOCR module not found.")
            _ocr_instance = None
        except Exception as e:
            print(f"PaddleOCR init failed: {e}")
            _ocr_instance = None
        except SystemExit:
            print("PaddleOCR init caused SystemExit. Skipping OCR.")
            _ocr_instance = None

    return _ocr_instance


def read_text(image_path, crop_box=None):
    """
    Reads text from an image using PaddleOCR and returns a list of dictionaries.
    Each dictionary contains 'text' and 'box' (list of [x, y] coordinates).
    """
    ocr = get_ocr_instance()
    if ocr is None:
        print("OCR not available.")
        return []
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return []

    ocr_input = image_path
    offset_x = 0
    offset_y = 0
    if crop_box is not None:
        image = cv2.imread(image_path)
        if image is not None:
            img_h, img_w = image.shape[:2]
            x1 = max(0, min(int(crop_box[0]), img_w))
            y1 = max(0, min(int(crop_box[1]), img_h))
            x2 = max(x1, min(int(crop_box[2]), img_w))
            y2 = max(y1, min(int(crop_box[3]), img_h))
            crop = image[y1:y2, x1:x2]
            if crop.size:
                ocr_input = crop
                offset_x = x1
                offset_y = y1

    try:
        try:
            result = ocr.ocr(ocr_input, cls=OCR_USE_ANGLE_CLS)
        except TypeError:
            result = ocr.ocr(ocr_input)

        ocr_data = []
        if not result:
            return []

        res = result[0]

        if hasattr(res, "keys") and "dt_polys" in res and "rec_texts" in res:
            polys = res["dt_polys"]
            texts = res["rec_texts"]
            scores = res.get("rec_scores", [1.0] * len(texts))

            for poly, text, conf in zip(polys, texts, scores):
                text_str = str(text).strip()
                if not text_str:
                    continue
                if hasattr(poly, "tolist"):
                    poly = poly.tolist()
                if offset_x or offset_y:
                    poly = _shift_polygon(poly, offset_x, offset_y)
                ocr_data.append(
                    {
                        "text": text_str,
                        "box": poly,
                        "confidence": float(conf),
                    }
                )
        elif isinstance(res, list):
            lines = res if isinstance(res, list) and (len(res) == 0 or isinstance(res[0], list)) else result
            for line in lines:
                if line is None or len(line) < 2:
                    continue
                box = line[0]
                if hasattr(box, "tolist"):
                    box = box.tolist()
                if offset_x or offset_y:
                    box = _shift_polygon(box, offset_x, offset_y)

                text_info = line[1]
                if isinstance(text_info, (tuple, list)) and len(text_info) >= 2:
                    text_str = str(text_info[0]).strip()
                    conf = float(text_info[1])
                else:
                    text_str = str(text_info).strip()
                    conf = 1.0
                if text_str:
                    ocr_data.append(
                        {
                            "text": text_str,
                            "box": box,
                            "confidence": conf,
                        }
                    )

        return ocr_data
    except Exception as e:
        import traceback

        print(f"Error during OCR: {e}")
        traceback.print_exc()
        return []


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        test_img = sys.argv[1]
        data = read_text(test_img)
        print(f"Found {len(data)} text items.")
        for item in data[:5]:
            print(f"Text: {item['text']} at {item['box']}")
