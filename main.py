from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = ROOT / "input"
DEFAULT_OUTPUT_DIR = ROOT / "output"


def _run(cmd: list[str], env: dict[str, str]) -> int:
    print(" ".join(f'"{part}"' if " " in part else part for part in cmd), flush=True)
    return subprocess.run(cmd, env=env).returncode


def _default_inference_python() -> Path:
    explicit = os.getenv("INFERENCE_PYTHON")
    if explicit:
        return Path(explicit)
    return Path(sys.executable)


def _inference_env(base_env: dict[str, str]) -> dict[str, str]:
    env = base_env.copy()
    overlay = os.getenv("INFERENCE_OVERLAY_PATH")
    overlay_path = Path(overlay) if overlay else None
    if overlay_path and overlay_path.exists():
        current = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            str(overlay_path)
            if not current
            else str(overlay_path) + os.pathsep + current
        )
    return env


def main() -> int:
    parser = argparse.ArgumentParser(description="Production-style local ONNX/TensorRT CSV pipeline.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--single-image", default=os.getenv("SINGLE_IMAGE"))
    parser.add_argument(
        "--inference-python",
        type=Path,
        default=_default_inference_python(),
        help="Python executable containing inference-models + ONNX Runtime GPU + TensorRT.",
    )
    parser.add_argument("--skip-inference", action="store_true")
    args = parser.parse_args()

    env = os.environ.copy()
    env.setdefault("PIPELINE_PROFILE", "local_45_46")
    env.setdefault("OCR_CUDA_VISIBLE_DEVICES", "0")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_inference:
        infer_cmd = [
            str(args.inference_python),
            str(ROOT / "local_inference.py"),
            "--input-dir",
            str(args.input_dir),
            "--output-dir",
            str(args.output_dir),
        ]
        if args.single_image:
            infer_cmd += ["--single-image", args.single_image]
        rc = _run(infer_cmd, _inference_env(env))
        if rc != 0:
            return rc

    csv_cmd = [
        sys.executable,
        str(ROOT / "build_csv.py"),
        "--input-dir",
        str(args.input_dir),
        "--output-dir",
        str(args.output_dir),
    ]
    if args.single_image:
        csv_cmd += ["--single-image", args.single_image]
    return _run(csv_cmd, env)


if __name__ == "__main__":
    raise SystemExit(main())
