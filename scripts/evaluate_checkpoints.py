"""
Standalone checkpoint evaluation script for Edge Vision Reliability Lab.
Evaluates a trained model checkpoint against an unaugmented evaluation dataset partition,
computing sample-weighted loss, top-1 accuracy, per-class metrics, and confusion matrix.
"""

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import sys
from typing import Any, Dict
import torch
import torch.nn as nn
import torchvision

from edge_vision.data.dataset import CIFAR10_CLASSES
from edge_vision.evaluation.evaluator import (
    evaluate_model,
    get_evaluation_dataloader,
    load_checkpoint_for_evaluation,
    resolve_evaluation_device,
    save_evaluation_report,
)


def run_evaluation(
    checkpoint_path: str,
    data_config_path: str = "configs/cifar10.yaml",
    split: str = "test",
    batch_size: int = 64,
    device_name: str = "auto",
    output_path: str = None,
) -> Dict[str, Any]:
    """
    Executes standalone checkpoint evaluation and returns the structured report dictionary.
    """
    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.is_file():
        raise FileNotFoundError(f"Checkpoint file not found at: {checkpoint_path}")

    device = resolve_evaluation_device(device_name)

    # 1. Load checkpoint and reconstruct model strictly
    model, checkpoint_payload = load_checkpoint_for_evaluation(ckpt_file, device=device)

    # 2. Acquire evaluation DataLoader with unaugmented transforms
    dataloader = get_evaluation_dataloader(
        data_config_path=data_config_path,
        split=split,
        batch_size=batch_size,
        num_workers=0,
    )

    # 3. Evaluate under torch.inference_mode()
    criterion = nn.CrossEntropyLoss()
    metrics = evaluate_model(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        device=device,
        class_names=CIFAR10_CLASSES,
    )

    # 4. Construct comprehensive structured report
    run_id = checkpoint_payload.get("run_id", "unnamed_run")
    rel_ckpt_path = os.path.relpath(ckpt_file, start=os.getcwd()) if ckpt_file.is_absolute() else str(ckpt_file)

    report_payload: Dict[str, Any] = {
        "run_id": run_id,
        "model_architecture": checkpoint_payload.get("model_architecture"),
        "checkpoint": {
            "path": rel_ckpt_path,
            "filename": ckpt_file.name,
            "epoch": checkpoint_payload.get("epoch"),
            "best_metric_value": checkpoint_payload.get("best_metric_value"),
            "best_epoch": checkpoint_payload.get("best_epoch"),
        },
        "training_metadata": {
            "seed": checkpoint_payload.get("seed"),
            "dataset_manifest_checksum": checkpoint_payload.get("dataset_manifest_checksum"),
            "training_timestamp": checkpoint_payload.get("timestamp"),
        },
        "evaluation_dataset": {
            "name": "cifar10",
            "split": split,
            "sample_count": metrics.total_samples,
            "data_config_path": data_config_path,
        },
        "metrics": metrics.to_dict(),
        "evaluation_environment": {
            "device": str(device),
            "device_requested": device_name,
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }

    # 5. Determine destination path and save atomically
    if output_path is not None:
        final_dest = Path(output_path)
    else:
        final_dest = Path("reports") / "experiment_results" / run_id / "evaluation" / f"evaluation_{split}.json"

    saved_path = save_evaluation_report(report_payload, final_dest)
    report_payload["saved_report_path"] = str(saved_path)

    return report_payload


def print_summary_table(report: Dict[str, Any]) -> None:
    """Prints a clean human-readable summary table of the evaluation results."""
    m = report["metrics"]
    ckpt = report["checkpoint"]
    eval_ds = report["evaluation_dataset"]

    print("\n" + "=" * 78)
    print(f"EVALUATION RESULTS: {report['model_architecture']} on {eval_ds['name'].upper()} ({eval_ds['split'].upper()} SPLIT)")
    print("=" * 78)
    print(f"  Run ID:            {report['run_id']}")
    print(f"  Checkpoint:        {ckpt['filename']} (Epoch {ckpt['epoch']})")
    print(f"  Device:            {report['evaluation_environment']['device']}")
    print(f"  Evaluated Samples: {m['total_samples']:,}")
    print(f"  Top-1 Accuracy:    {m['accuracy']:.2%} ({m['correct_predictions']:,} / {m['total_samples']:,})")
    print(f"  Cross-Entropy Loss: {m['loss']:.4f}")
    print("-" * 78)
    print("PER-CLASS ACCURACY BREAKDOWN:")
    print(f"  {'Class Name':<15} | {'Accuracy':<10} | {'Correct':<10} | {'Total':<10}")
    print("  " + "-" * 55)
    for cls_name, acc in m["per_class_accuracy"].items():
        cor = m["per_class_correct"][cls_name]
        tot = m["per_class_counts"][cls_name]
        print(f"  {cls_name:<15} | {acc:>8.2%}   | {cor:>8}   | {tot:>8}")
    print("=" * 78)
    print(f"Structured JSON report saved atomically to: {report['saved_report_path']}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Standalone evaluation of model checkpoints on CIFAR-10 evaluation partitions."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the checkpoint file (.pt) to evaluate.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "val"],
        help="Dataset split to evaluate on: 'test' (default) or 'val'.",
    )
    parser.add_argument(
        "--data-config",
        type=str,
        default="configs/cifar10.yaml",
        help="Path to CIFAR-10 dataset YAML config.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for evaluation (default: 64).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "mps"],
        help="Device to run evaluation on (default: auto).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Custom output file path for evaluation JSON report.",
    )

    args = parser.parse_args()

    report = run_evaluation(
        checkpoint_path=args.checkpoint,
        data_config_path=args.data_config,
        split=args.split,
        batch_size=args.batch_size,
        device_name=args.device,
        output_path=args.output,
    )
    print_summary_table(report)


if __name__ == "__main__":
    main()
