from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import cv2


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = ROOT / "input"
DEFAULT_OUTPUT_DIR = ROOT / "output"


def _set_csv_env(output_dir: Path) -> None:
    os.environ["PIPELINE_PROFILE"] = "local_45_46"
    os.environ["ENABLE_GENERAL_COLUMN_CLASSES"] = "1"
    os.environ["ROBOFLOW_INFER_COORD_SPACE"] = "original"
    os.environ["KEEP_PREPROCESSED_IMAGES"] = "1"
    os.environ.setdefault("OCR_CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("ENABLE_ROW_SEPARATOR_SPLIT", "1")
    os.environ.setdefault("ENABLE_LINE_BASED_SPAN_CELL_RECOVERY", "1")
    os.environ["OUTPUT_DIR"] = str(output_dir)
    os.environ["PREPROCESSED_DIR"] = str(output_dir / "preprocessed")


def _transform_predictions_to_preprocessed_space(
    predictions: dict[str, Any],
    original_image_path: Path,
    preprocessed_image_path: Path,
) -> dict[str, Any]:
    if not predictions or not predictions.get("predictions"):
        return predictions

    original = cv2.imread(str(original_image_path))
    preprocessed = cv2.imread(str(preprocessed_image_path))
    if original is None or preprocessed is None:
        return predictions

    orig_h, orig_w = original.shape[:2]
    pre_h, pre_w = preprocessed.shape[:2]
    if orig_h <= 0 or orig_w <= 0 or pre_h <= 0 or pre_w <= 0:
        return predictions

    scale = min(pre_w / float(orig_w), pre_h / float(orig_h))
    new_w = orig_w * scale
    new_h = orig_h * scale
    x_offset = (pre_w - new_w) / 2.0
    y_offset = (pre_h - new_h) / 2.0

    transformed = []
    for pred in predictions.get("predictions", []):
        clone = dict(pred)
        clone["x"] = float(pred.get("x", 0.0)) * scale + x_offset
        clone["y"] = float(pred.get("y", 0.0)) * scale + y_offset
        clone["width"] = float(pred.get("width", 0.0)) * scale
        clone["height"] = float(pred.get("height", 0.0)) * scale
        transformed.append(clone)

    payload = dict(predictions)
    payload["predictions"] = transformed
    image_meta = dict(payload.get("image", {})) if isinstance(payload.get("image"), dict) else {}
    image_meta["width"] = pre_w
    image_meta["height"] = pre_h
    payload["image"] = image_meta
    return payload


def _iter_images(input_dir: Path, single_image: str | None) -> list[Path]:
    supported = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    images = [
        path
        for path in sorted(input_dir.iterdir())
        if path.is_file() and path.suffix.lower() in supported
    ]
    if single_image:
        images = [path for path in images if single_image in path.name]
    return images


def run(input_dir: Path, output_dir: Path, single_image: str | None = None) -> int:
    _set_csv_env(output_dir)
    sys.path.insert(0, str(ROOT))

    from csv_extractor import build_table_structure, save_csv
    from image_utils import preprocess_image
    from ocr_reader import get_ocr_instance, read_text

    raw_dir = output_dir / "raw_predictions"
    csv_dir = output_dir / "csv"
    pred_dir = output_dir / "predictions_preprocessed"
    ocr_dir = output_dir / "ocr"
    pre_dir = output_dir / "preprocessed"
    for directory in (csv_dir, pred_dir, ocr_dir, pre_dir):
        directory.mkdir(parents=True, exist_ok=True)

    images = _iter_images(input_dir, single_image)
    if not images:
        print(f"No images found in {input_dir}")
        return 1

    get_ocr_instance()
    success = 0
    for image_path in images:
        print(f"csv {image_path.name}", flush=True)
        raw_path = raw_dir / f"{image_path.stem}.json"
        if not raw_path.exists():
            print(f"  missing raw predictions: {raw_path}")
            continue

        pre_path = pre_dir / f"pre_{image_path.name}"
        preprocess_image(str(image_path), str(pre_path))
        raw_payload = json.loads(raw_path.read_text(encoding="utf-8"))
        transformed = _transform_predictions_to_preprocessed_space(
            raw_payload,
            image_path,
            pre_path,
        )
        pred_path = pred_dir / f"{image_path.stem}_predictions_preprocessed.json"
        pred_path.write_text(json.dumps(transformed, indent=2), encoding="utf-8")

        # Best tested OCR setting: full preprocessed image, not detection-derived crop.
        ocr_data = read_text(str(pre_path), crop_box=None)
        ocr_path = ocr_dir / f"{image_path.stem}_ocr.json"
        ocr_path.write_text(json.dumps(ocr_data, indent=2, default=str), encoding="utf-8")

        table = build_table_structure(
            transformed,
            ocr_data,
            structure_image_path=str(pre_path),
        )
        csv_path = csv_dir / f"{image_path.stem}.csv"
        save_csv(table, str(csv_path))
        print(f"  wrote {csv_path}", flush=True)
        success += 1
    print(f"CSV complete: {success}/{len(images)}")
    return 0 if success else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Build CSV from local raw predictions.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--single-image", default=os.getenv("SINGLE_IMAGE"))
    args = parser.parse_args()
    return run(args.input_dir, args.output_dir, args.single_image)


if __name__ == "__main__":
    raise SystemExit(main())
