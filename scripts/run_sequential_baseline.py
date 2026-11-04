"""
Orchestrates sequential 100-epoch baseline training for ResNet-18 and MobileNetV3-Small on Apple Silicon MPS,
followed by held-out test set evaluation using the clean evaluation infrastructure (Task 2.3B).
"""

import argparse
from pathlib import Path
import shutil
import sys
import time

# Ensure repository root and scripts directory are in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import psutil
import torch

from edge_vision.training.checkpoint import CheckpointManager
from evaluate_checkpoints import print_summary_table, run_evaluation
from train_baseline import inspect_resources, run_training


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sequential full 100-epoch baseline training and evaluation."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Epochs per architecture (default: 100).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        choices=["mps", "cpu"],
        help="Device to train and evaluate on (default: mps).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Enable checkpoint resumption if latest checkpoint exists.",
    )
    args = parser.parse_args()

    overall_start = time.perf_counter()

    print("=" * 80)
    print("EDGE VISION RELIABILITY LAB — TASK 2.3B: FULL 100-EPOCH BASELINE EXPERIMENTS")
    print("=" * 80)
    print(f"Device:           {args.device}")
    print(f"Epochs per model: {args.epochs}")
    print(f"Seed:             {args.seed}")
    print(f"Resume enabled:   {args.resume}")
    print("Execution mode:   STRICTLY SEQUENTIAL (Never concurrent)")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Initial Pre-flight Verification
    # -------------------------------------------------------------------------
    inspect_resources()

    if args.device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS requested but torch.backends.mps.is_available() is False. "
                "Per experimental requirements, do not silently switch to CPU."
            )

    # -------------------------------------------------------------------------
    # Experiment 1: ResNet-18 Baseline (100 Epochs)
    # -------------------------------------------------------------------------
    print("\n" + "#" * 80)
    print("STAGE 1/2: TRAINING RESNET-18 (exp_baseline_resnet18_seed42)")
    print("#" * 80)

    r18_args = argparse.Namespace(
        architecture="resnet18_cifar",
        experiment_id="exp_baseline_resnet18_seed42",
        config="configs/train_baseline.yaml",
        epochs=args.epochs,
        seed=args.seed,
        device=args.device,
        resume=args.resume,
    )
    r18_start = time.perf_counter()
    r18_train_res = run_training(r18_args)
    r18_duration = time.perf_counter() - r18_start

    # Flush device memory cache between models
    if hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    print("\n[Resource Check] Cleared MPS cache post ResNet-18 training.")
    inspect_resources()

    # -------------------------------------------------------------------------
    # Experiment 2: MobileNetV3-Small Baseline (100 Epochs)
    # -------------------------------------------------------------------------
    print("\n" + "#" * 80)
    print("STAGE 2/2: TRAINING MOBILENETV3-SMALL (exp_baseline_mobilenet_v3_small_seed42)")
    print("#" * 80)

    mb3_args = argparse.Namespace(
        architecture="mobilenet_v3_small_cifar",
        experiment_id="exp_baseline_mobilenet_v3_small_seed42",
        config="configs/train_baseline.yaml",
        epochs=args.epochs,
        seed=args.seed,
        device=args.device,
        resume=args.resume,
    )
    mb3_start = time.perf_counter()
    mb3_train_res = run_training(mb3_args)
    mb3_duration = time.perf_counter() - mb3_start

    if hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    print("\n[Resource Check] Cleared MPS cache post MobileNetV3 training.")
    inspect_resources()

    # -------------------------------------------------------------------------
    # Evaluation: Held-out Test Set (10,000 samples)
    # -------------------------------------------------------------------------
    print("\n" + "#" * 80)
    print("STAGE 3: HELD-OUT TEST SET EVALUATION (10,000 SAMPLES)")
    print("#" * 80)

    # 1. ResNet-18 Test Evaluation
    r18_best_ckpt = Path("experiments") / "exp_baseline_resnet18_seed42" / "checkpoints" / "best_checkpoint.pt"
    r18_test_report_dest = Path("experiments") / "exp_baseline_resnet18_seed42" / "evaluation_test.json"
    print(f"\nEvaluating ResNet-18 best checkpoint: {r18_best_ckpt}")
    r18_eval_report = run_evaluation(
        checkpoint_path=str(r18_best_ckpt),
        data_config_path="configs/cifar10.yaml",
        split="test",
        batch_size=64,
        device_name=args.device,
        output_path=str(r18_test_report_dest),
    )
    print_summary_table(r18_eval_report)

    # Copy to reports/experiment_results for historical preservation
    reports_r18_dest = Path("reports") / "experiment_results" / "eval_test_resnet18.json"
    reports_r18_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(r18_test_report_dest, reports_r18_dest)

    # 2. MobileNetV3-Small Test Evaluation
    mb3_best_ckpt = Path("experiments") / "exp_baseline_mobilenet_v3_small_seed42" / "checkpoints" / "best_checkpoint.pt"
    mb3_test_report_dest = Path("experiments") / "exp_baseline_mobilenet_v3_small_seed42" / "evaluation_test.json"
    print(f"\nEvaluating MobileNetV3-Small best checkpoint: {mb3_best_ckpt}")
    mb3_eval_report = run_evaluation(
        checkpoint_path=str(mb3_best_ckpt),
        data_config_path="configs/cifar10.yaml",
        split="test",
        batch_size=64,
        device_name=args.device,
        output_path=str(mb3_test_report_dest),
    )
    print_summary_table(mb3_eval_report)

    # Copy to reports/experiment_results for historical preservation
    reports_mb3_dest = Path("reports") / "experiment_results" / "eval_test_mobilenet.json"
    reports_mb3_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(mb3_test_report_dest, reports_mb3_dest)

    total_time = time.perf_counter() - overall_start

    print("\n" + "=" * 80)
    print("TASK 2.3B COMPLETE — ALL BASELINE EXPERIMENTS & TEST EVALUATIONS FINISHED")
    print("=" * 80)
    print(f"Total Wall Time:           {total_time / 3600:.2f} hours ({total_time / 60:.1f} minutes)")
    print(f"ResNet-18 Run Time:        {r18_duration / 3600:.2f} hours ({r18_duration / 60:.1f} minutes)")
    print(f"MobileNetV3 Run Time:      {mb3_duration / 3600:.2f} hours ({mb3_duration / 60:.1f} minutes)")
    print("-" * 80)
    print(f"ResNet-18 Best Val Acc:    {r18_train_res['best_val_acc']:.2%} (Epoch {r18_train_res['best_epoch']})")
    print(f"ResNet-18 Test Acc:        {r18_eval_report['metrics']['accuracy']:.2%} (Loss: {r18_eval_report['metrics']['loss']:.4f})")
    print("-" * 80)
    print(f"MobileNetV3 Best Val Acc:  {mb3_train_res['best_val_acc']:.2%} (Epoch {mb3_train_res['best_epoch']})")
    print(f"MobileNetV3 Test Acc:      {mb3_eval_report['metrics']['accuracy']:.2%} (Loss: {mb3_eval_report['metrics']['loss']:.4f})")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
