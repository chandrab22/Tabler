from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import torch
from PIL import Image
from config import MODEL_COLUMN_ID, MODEL_GENERAL_ID
from inference_models.models.rfdetr.rfdetr_object_detection_onnx import (
    RFDetrForObjectDetectionONNX,
)


ROOT = Path(__file__).resolve().parent
ARTIFACT_ROOT = ROOT / "model_artifacts" / "tabler"
GENERAL_VERSION = MODEL_GENERAL_ID.rsplit("/", 1)[-1]
COLUMN_VERSION = MODEL_COLUMN_ID.rsplit("/", 1)[-1]
DEFAULT_INPUT_DIR = ROOT / "input"
DEFAULT_OUTPUT_DIR = ROOT / "output"
CONFIDENCE = float(os.getenv("MODEL_CONFIDENCE", "0.01"))
RELAUNCH_GUARD_ENV = "TABLER_LOCAL_INFERENCE_RELAUNCHED"

INFERENCE_CONFIG = {
    "image_pre_processing": {"auto-orient": {"enabled": True}},
    "network_input": {
        "training_input_size": {"height": 1280, "width": 1280},
        "dynamic_spatial_size_supported": False,
        "color_mode": "rgb",
        "resize_mode": "letterbox",
        "padding_value": 255,
        "input_channels": 3,
        "scaling_factor": 255,
        "normalization": [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
        "dataset_version_resize_dimensions": {"height": "1280", "width": "1280"},
    },
    "class_names_operations": [
        {"type": "class_name_removal", "class_name": "background_class83422"}
    ],
}


_DLL_DIRECTORY_HANDLES: list[Any] = []


def _with_overlay_pythonpath(env: dict[str, str]) -> dict[str, str]:
    updated = env.copy()
    overlay = os.getenv("INFERENCE_OVERLAY_PATH")
    overlay_path = Path(overlay) if overlay else None
    if overlay_path and overlay_path.exists():
        current = updated.get("PYTHONPATH", "")
        updated["PYTHONPATH"] = (
            str(overlay_path)
            if not current
            else str(overlay_path) + os.pathsep + current
        )
    return updated


def _preferred_inference_python() -> Path | None:
    explicit = os.getenv("INFERENCE_PYTHON")
    if explicit:
        explicit_path = Path(explicit)
        if explicit_path.exists():
            return explicit_path
    return None


def _maybe_relaunch_with_tested_python() -> int | None:
    if os.getenv(RELAUNCH_GUARD_ENV) == "1":
        return None

    preferred_python = _preferred_inference_python()
    if preferred_python is None:
        return None

    current_python = Path(sys.executable).resolve()
    try:
        same_python = current_python.samefile(preferred_python)
    except FileNotFoundError:
        same_python = current_python == preferred_python.resolve()
    if same_python:
        return None

    env = _with_overlay_pythonpath(os.environ)
    env[RELAUNCH_GUARD_ENV] = "1"
    cmd = [str(preferred_python), str(Path(__file__).resolve()), *sys.argv[1:]]
    print(
        "[local_inference] Relaunching with tested TensorRT Python:",
        preferred_python,
        flush=True,
    )
    return subprocess.run(cmd, env=env).returncode


def _prepare_windows_gpu_dll_paths() -> None:
    """Make pip-installed CUDA/TensorRT DLLs visible to ONNX Runtime on Windows."""
    if os.name != "nt":
        return
    site_roots = [Path(p) for p in sys.path if p.lower().endswith("site-packages")]
    candidates: list[Path] = []
    for site in site_roots:
        candidates.extend(
            [
                site / "onnxruntime" / "capi",
                site / "tensorrt_libs",
                site / "nvidia" / "cublas" / "bin",
                site / "nvidia" / "cuda_runtime" / "bin",
                site / "nvidia" / "cuda_nvrtc" / "bin",
                site / "torch" / "lib",
                site / "torchvision",
            ]
        )
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        path = str(candidate)
        try:
            _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(path))
        except (AttributeError, OSError):
            pass
        os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")


def _ensure_package(version: str, output_dir: Path) -> Path:
    src = ARTIFACT_ROOT / version
    if not (src / "weights.onnx").exists():
        raise FileNotFoundError(f"Missing model artifact: {src / 'weights.onnx'}")
    if not (src / "class_names.txt").exists():
        raise FileNotFoundError(f"Missing class names: {src / 'class_names.txt'}")

    package = output_dir / "runtime_model_packages" / version
    package.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / "weights.onnx", package / "weights.onnx")
    shutil.copy2(src / "class_names.txt", package / "class_names.txt")
    (package / "inference_config.json").write_text(
        json.dumps(INFERENCE_CONFIG, indent=2), encoding="utf-8"
    )
    return package


def _providers_for_package(package: Path) -> list[Any]:
    cache_path = package / "trt_engine_cache"
    cache_path.mkdir(parents=True, exist_ok=True)
    return [
        (
            "TensorrtExecutionProvider",
            {
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": str(cache_path),
                "trt_fp16_enable": True,
                "trt_max_workspace_size": 1 << 30,
            },
        ),
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]


def _load_models(output_dir: Path) -> dict[str, Any]:
    _prepare_windows_gpu_dll_paths()
    models: dict[str, Any] = {}
    for version in (GENERAL_VERSION, COLUMN_VERSION):
        package = _ensure_package(version, output_dir)
        models[version] = RFDetrForObjectDetectionONNX.from_pretrained(
            str(package),
            onnx_execution_providers=_providers_for_package(package),
            default_onnx_trt_options=False,
            device=torch.device("cpu"),
        )
        providers = models[version]._session.get_providers()
        print(f"loaded model {version}: providers={providers}", flush=True)
        if os.getenv("REQUIRE_TENSORRT", "1") == "1" and "TensorrtExecutionProvider" not in providers:
            raise RuntimeError(
                f"TensorRT provider is required for best UI match, but model {version} "
                f"loaded providers {providers}. Fix CUDA/TensorRT DLL paths or run with "
                "the tested inference Python environment."
            )
    return models


def _jpeg_q92_sub2(image_path: Path, output_dir: Path) -> Path:
    variants_dir = output_dir / "image_variants"
    variants_dir.mkdir(parents=True, exist_ok=True)
    out_path = variants_dir / f"{image_path.stem}_jpeg_q92_sub2.jpg"
    with Image.open(image_path) as image:
        image.convert("RGB").save(
            out_path,
            "JPEG",
            quality=92,
            subsampling=2,
            optimize=False,
        )
    return out_path


def _prediction_payload(model: Any, image_path: Path, source_model: str) -> dict[str, Any]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"cv2 could not read {image_path}")

    detections = model.infer(image, confidence=CONFIDENCE, input_color_format="bgr")[0]
    predictions = []
    for xyxy, confidence, class_id in zip(
        detections.xyxy.tolist(),
        detections.confidence.tolist(),
        detections.class_id.tolist(),
    ):
        x1, y1, x2, y2 = [float(v) for v in xyxy]
        class_id = int(class_id)
        predictions.append(
            {
                "x": (x1 + x2) / 2.0,
                "y": (y1 + y2) / 2.0,
                "width": x2 - x1,
                "height": y2 - y1,
                "confidence": float(confidence),
                "class": model.class_names[class_id],
                "class_id": class_id,
                "source_model": source_model,
            }
        )
    predictions.sort(key=lambda pred: pred["confidence"], reverse=True)
    h, w = image.shape[:2]
    return {"predictions": predictions, "image": {"width": w, "height": h}}


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
    raw_dir = output_dir / "raw_predictions"
    raw_dir.mkdir(parents=True, exist_ok=True)
    models = _load_models(output_dir)
    images = _iter_images(input_dir, single_image)
    if not images:
        print(f"No images found in {input_dir}")
        return 1

    runtime = {
        "confidence": CONFIDENCE,
        "image_variant": "jpeg_quality_92_subsampling_2",
        "models": {},
    }
    for version, model in models.items():
        runtime["models"][version] = {
            "actual_providers": model._session.get_providers(),
            "artifact_dir": str(ARTIFACT_ROOT / version),
        }

    for image_path in images:
        print(f"infer {image_path.name}", flush=True)
        infer_image = _jpeg_q92_sub2(image_path, output_dir)
        payload_general = _prediction_payload(models[GENERAL_VERSION], infer_image, "general")
        print(f"  model {GENERAL_VERSION} predictions: {len(payload_general['predictions'])}", flush=True)
        payload_column = _prediction_payload(models[COLUMN_VERSION], infer_image, "column")
        print(f"  model {COLUMN_VERSION} predictions: {len(payload_column['predictions'])}", flush=True)
        combined = {
            "predictions": payload_general["predictions"] + payload_column["predictions"],
            "image": payload_general.get("image", {}),
            "local_runtime": runtime,
        }
        (raw_dir / f"{image_path.stem}.json").write_text(
            json.dumps(combined, indent=2),
            encoding="utf-8",
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run best local ONNX/TRT inference.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--single-image", default=os.getenv("SINGLE_IMAGE"))
    args = parser.parse_args()
    relaunch_rc = _maybe_relaunch_with_tested_python()
    if relaunch_rc is not None:
        return relaunch_rc
    return run(args.input_dir, args.output_dir, args.single_image)


if __name__ == "__main__":
    raise SystemExit(main())
