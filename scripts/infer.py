#!/usr/bin/env python3
"""
Command-Line Interface for CIFAR-10 Model Inference.

Supports single-image and batch/directory inference across FP32, FP16, and INT8
precisions using the unified InferenceEngine.

Usage Examples:
    # Single image inference on MPS (or CPU):
    python scripts/infer.py --image tests/test_data/sample.png --model resnet18_cifar --precision fp16 --device mps --top-k 5

    # Directory batch inference with bounded memory:
    python scripts/infer.py --dir path/to/images/ --model mobilenet_v3_small_cifar --precision int8 --device cpu --batch-size 32

    # Save output to machine-readable JSON:
    python scripts/infer.py --image sample.png --model resnet18_cifar --output result.json
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image

from edge_vision.inference import (
    InferenceEngine,
    InferenceError,
    InvalidInputError,
    DeviceCompatibilityError,
    ModelNotFoundError,
    UnsupportedModelError,
    UnsupportedPrecisionError,
)

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def open_image_safely(path: Path) -> Image.Image:
    """
    Safely opens and decodes an image file using Pillow, guaranteeing proper
    resource cleanup and full pixel loading into RGB format.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {path}")

    try:
        with Image.open(path) as img:
            rgb_img = img.convert("RGB")
            rgb_img.load()
            return rgb_img
    except Exception as err:
        raise ValueError(f"Failed to open or decode image at '{path}': {err}") from err


def format_single_prediction_json(
    image_path: Path,
    output: Any,
) -> Dict[str, Any]:
    """Formats SingleImageInferenceOutput into a clean JSON-serializable dictionary."""
    pred = output.prediction
    timing = output.timing
    return {
        "status": "success",
        "image_path": str(image_path.resolve()),
        "prediction": {
            "predicted_class": pred.predicted_class,
            "class_index": pred.class_index,
            "confidence": pred.confidence,
            "probabilities": pred.probabilities,
            "logits": pred.logits,
            "top_k": [
                {"rank": i + 1, "class_name": c, "confidence": conf}
                for i, (c, conf) in enumerate(pred.top_k)
            ],
        },
        "timing_ms": {
            "preprocess_ms": timing.preprocess_ms,
            "forward_ms": timing.forward_ms,
            "postprocess_ms": timing.postprocess_ms,
            "total_ms": timing.total_ms,
        },
        "metadata": output.metadata,
    }


def print_single_human_report(image_path: Path, output: Any, stream=sys.stdout) -> None:
    """Prints formatted, human-readable inference results to the specified stream."""
    pred = output.prediction
    timing = output.timing
    meta = output.metadata

    stream.write("\n" + "=" * 70 + "\n")
    stream.write("Edge Vision Reliability Lab — Inference Result\n")
    stream.write("=" * 70 + "\n")
    stream.write(f"Target Image:      {image_path.resolve()}\n")
    stream.write(f"Model:             {meta['model_name']} ({meta.get('parameter_count', 0):,} params)\n")
    stream.write(f"Precision:         {meta['precision'].upper()} ({meta.get('model_dtype', 'unknown')})\n")
    stream.write(f"Device & Backend:  {meta['device']} [{meta['backend']}]\n")
    stream.write(f"Artifact Path:     {meta['artifact_path']}\n")
    stream.write(f"Artifact Size:     {meta.get('artifact_size_bytes', 0):,} bytes\n")
    if "device_warning" in meta:
        stream.write(f"Device Warning:    {meta['device_warning']}\n")
    stream.write("-" * 70 + "\n")
    stream.write("Prediction:\n")
    stream.write(f"  Predicted Class: {pred.predicted_class} (index: {pred.class_index})\n")
    stream.write(f"  Confidence:      {pred.confidence * 100:.2f}%\n\n")

    stream.write(f"Top-{len(pred.top_k)} Predictions:\n")
    for i, (c, conf) in enumerate(pred.top_k, 1):
        bar = "█" * int(conf * 25)
        stream.write(f"  #{i:<2} {c:<14} {conf * 100:>6.2f}%  {bar}\n")

    stream.write("-" * 70 + "\n")
    stream.write("Latency Breakdown (Wall-Clock):\n")
    stream.write(f"  Preprocessing:   {timing.preprocess_ms:>8.3f} ms\n")
    stream.write(f"  Forward Pass:    {timing.forward_ms:>8.3f} ms\n")
    stream.write(f"  Postprocessing:  {timing.postprocess_ms:>8.3f} ms\n")
    stream.write(f"  Total Latency:   {timing.total_ms:>8.3f} ms")
    if timing.total_ms > 0:
        fps = 1000.0 / timing.total_ms
        stream.write(f"  ({fps:.1f} fps)")
    stream.write("\n" + "=" * 70 + "\n\n")


def run_single_inference(args: argparse.Namespace, engine: InferenceEngine) -> Dict[str, Any]:
    """Executes single-image inference workflow."""
    img_path = Path(args.image)
    pil_img = open_image_safely(img_path)
    output = engine.predict_image(pil_img, resize=args.resize, top_k=args.top_k)

    payload = format_single_prediction_json(img_path, output)

    if not args.json:
        print_single_human_report(img_path, output, stream=sys.stdout)
    else:
        # Diagnostic messages go to stderr if JSON is written to stdout
        sys.stderr.write(f"[INFO] Processed single image: {img_path.name}\n")

    return payload


def run_directory_inference(args: argparse.Namespace, engine: InferenceEngine) -> Dict[str, Any]:
    """Executes bounded batch directory inference workflow."""
    dir_path = Path(args.dir)
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Directory path does not exist or is not a directory: {dir_path}")

    # Discover and sort candidate image files deterministically
    all_files = sorted(
        [p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS],
        key=lambda x: x.name,
    )

    if not all_files:
        exts_str = ", ".join(sorted(SUPPORTED_IMAGE_EXTENSIONS))
        raise ValueError(f"No supported image files ({exts_str}) found in directory: {dir_path}")

    batch_size = max(1, args.batch_size)
    total_images = len(all_files)
    file_records: List[Dict[str, Any]] = []

    total_preprocess_ms = 0.0
    total_forward_ms = 0.0
    total_postprocess_ms = 0.0
    wall_start = time.perf_counter()

    if not args.json:
        sys.stdout.write(f"Found {total_images} supported image(s) in {dir_path.resolve()}\n")
        sys.stdout.write(f"Processing in bounded batches of size {batch_size}...\n")

    # Process in bounded chunks to prevent arbitrary memory growth
    for chunk_idx in range(0, total_images, batch_size):
        chunk_paths = all_files[chunk_idx : chunk_idx + batch_size]
        valid_images: List[Image.Image] = []
        valid_paths: List[Path] = []

        for p in chunk_paths:
            try:
                img = open_image_safely(p)
                valid_images.append(img)
                valid_paths.append(p)
            except Exception as err:
                if args.on_error == "stop":
                    raise ValueError(f"Aborting on unreadable image '{p.name}': {err}") from err
                else:  # "record"
                    file_records.append({
                        "file_name": p.name,
                        "file_path": str(p.resolve()),
                        "status": "error",
                        "error": str(err),
                    })

        if valid_images:
            batch_output = engine.predict_batch(
                valid_images,
                resize=args.resize,
                top_k=args.top_k,
            )

            # Accumulate timing metrics
            total_preprocess_ms += batch_output.timing.preprocess_ms
            total_forward_ms += batch_output.timing.forward_ms
            total_postprocess_ms += batch_output.timing.postprocess_ms

            # Format per-file predictions
            for p, pred in zip(valid_paths, batch_output.predictions):
                file_records.append({
                    "file_name": p.name,
                    "file_path": str(p.resolve()),
                    "status": "success",
                    "predicted_class": pred.predicted_class,
                    "class_index": pred.class_index,
                    "confidence": pred.confidence,
                    "probabilities": pred.probabilities,
                    "top_k": [
                        {"rank": i + 1, "class_name": c, "confidence": conf}
                        for i, (c, conf) in enumerate(pred.top_k)
                    ],
                })

    wall_duration_s = time.perf_counter() - wall_start
    successful_count = sum(1 for r in file_records if r["status"] == "success")
    error_count = total_images - successful_count
    overall_throughput = (successful_count / wall_duration_s) if wall_duration_s > 0 else 0.0

    summary = {
        "status": "completed",
        "directory": str(dir_path.resolve()),
        "total_images": total_images,
        "processed_successfully": successful_count,
        "errors_encountered": error_count,
        "wall_time_seconds": round(wall_duration_s, 3),
        "overall_throughput_fps": round(overall_throughput, 1),
        "accumulated_engine_timing_ms": {
            "preprocess_ms": round(total_preprocess_ms, 3),
            "forward_ms": round(total_forward_ms, 3),
            "postprocess_ms": round(total_postprocess_ms, 3),
            "total_engine_ms": round(total_preprocess_ms + total_forward_ms + total_postprocess_ms, 3),
        },
        "metadata": engine.get_metadata(),
        "results": file_records,
    }

    if not args.json:
        sys.stdout.write("\n" + "=" * 70 + "\n")
        sys.stdout.write("Batch Directory Inference Summary\n")
        sys.stdout.write("=" * 70 + "\n")
        sys.stdout.write(f"Directory:         {dir_path.resolve()}\n")
        sys.stdout.write(f"Total Images:      {total_images}\n")
        sys.stdout.write(f"Successful:        {successful_count}\n")
        sys.stdout.write(f"Errors:            {error_count}\n")
        sys.stdout.write(f"Wall-Clock Time:   {wall_duration_s:.3f} s\n")
        sys.stdout.write(f"Overall Throughput:{overall_throughput:>8.1f} fps\n")
        sys.stdout.write("-" * 70 + "\n")
        sys.stdout.write(f"{'File Name':<28} {'Predicted Class':<16} {'Confidence':<12} {'Status'}\n")
        sys.stdout.write("-" * 70 + "\n")
        for rec in file_records[:20]:
            if rec["status"] == "success":
                conf_str = f"{rec['confidence'] * 100:.1f}%"
                sys.stdout.write(f"{rec['file_name']:<28} {rec['predicted_class']:<16} {conf_str:<12} OK\n")
            else:
                sys.stdout.write(f"{rec['file_name']:<28} {'[ERROR]':<16} {'-':<12} FAILED\n")
        if len(file_records) > 20:
            sys.stdout.write(f"... and {len(file_records) - 20} more items (see JSON output for full list).\n")
        sys.stdout.write("=" * 70 + "\n\n")

    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    """Builds the argument parser for the inference CLI."""
    parser = argparse.ArgumentParser(
        description="Edge Vision Reliability Lab — Unified Inference CLI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Mutually exclusive input modes
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--image",
        type=str,
        help="Path to a single image file (PNG, JPEG, BMP, WebP, etc.).",
    )
    input_group.add_argument(
        "--dir",
        type=str,
        help="Path to a directory of images for batch inference.",
    )

    # Model and precision options
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18_cifar",
        help="Model architecture identifier ('resnet18_cifar' or 'mobilenet_v3_small_cifar').",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        choices=["fp32", "fp16", "int8"],
        help="Inference precision variant.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "mps"],
        help="Target execution compute device.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional custom checkpoint file path overriding default registry location.",
    )

    # Preprocessing and ranking options
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of highest-confidence predictions to return (1-10).",
    )
    parser.add_argument(
        "--resize",
        action="store_true",
        default=False,
        help="Explicitly enable bilinear resizing of non-32x32 images to (32, 32).",
    )

    # Batch / directory options
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Bounded batch size for directory inference.",
    )
    parser.add_argument(
        "--on-error",
        type=str,
        default="record",
        choices=["record", "stop"],
        help="Policy for unreadable images in directory mode: 'record' logs per-image error, 'stop' aborts immediately.",
    )

    # Output options
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional destination path to write machine-readable JSON results.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output raw JSON to stdout (human status messages are routed to stderr).",
    )

    return parser


def main() -> int:
    """CLI entrypoint."""
    parser = build_arg_parser()
    args = parser.parse_args()

    # Load inference engine with clean error interception
    try:
        engine = InferenceEngine.load(
            model_name=args.model,
            precision=args.precision,
            device=args.device,
            checkpoint_path=args.checkpoint,
        )
    except (InferenceError, ModelNotFoundError, UnsupportedModelError, UnsupportedPrecisionError, DeviceCompatibilityError) as err:
        sys.stderr.write(f"Error initializing model: {err}\n")
        return 1
    except Exception as err:
        sys.stderr.write(f"Unexpected error loading model: {err}\n")
        return 1

    # Execute inference
    try:
        if args.image:
            results_payload = run_single_inference(args, engine)
        else:
            results_payload = run_directory_inference(args, engine)
    except (InvalidInputError, FileNotFoundError, NotADirectoryError, ValueError) as err:
        sys.stderr.write(f"Error: {err}\n")
        return 1
    except Exception as err:
        sys.stderr.write(f"Unexpected inference error: {err}\n")
        return 1

    # Output JSON handling
    if args.json:
        sys.stdout.write(json.dumps(results_payload, indent=2) + "\n")

    if args.output:
        out_path = Path(args.output)
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results_payload, f, indent=2)
            if not args.json:
                sys.stdout.write(f"Saved inference results JSON to: {out_path.resolve()}\n")
        except Exception as err:
            sys.stderr.write(f"Error saving output JSON to '{args.output}': {err}\n")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
