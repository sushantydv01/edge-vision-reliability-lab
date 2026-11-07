"""
Evaluates INT8 models on the 10,000 clean CIFAR-10 test set images and compares
per-class metrics, confusion matrices, and prediction concordance against FP32 baselines.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from edge_vision.data.dataset import CIFAR10_CLASSES
from edge_vision.evaluation.evaluator import get_evaluation_dataloader, load_checkpoint_for_evaluation
from edge_vision.evaluation.metrics import EvaluationTracker


def evaluate_clean_pair(
    fp32_ckpt_path: str,
    int8_ckpt_path: str,
    model_name: str,
    test_loader,
    criterion,
    device: torch.device,
) -> dict:
    print(f"\n[*] Evaluating clean test set for {model_name}...")

    # Load models
    fp32_model, _ = load_checkpoint_for_evaluation(fp32_ckpt_path, device=device, precision="fp32")
    int8_model, int8_meta = load_checkpoint_for_evaluation(int8_ckpt_path, device=device, precision="int8")

    fp32_model.eval()
    int8_model.eval()

    tracker_fp32 = EvaluationTracker(class_names=CIFAR10_CLASSES)
    tracker_int8 = EvaluationTracker(class_names=CIFAR10_CLASSES)

    all_preds_fp32 = []
    all_preds_int8 = []
    all_targets = []

    with torch.inference_mode():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            out_fp32 = fp32_model(inputs)
            out_int8 = int8_model(inputs)

            loss_fp32 = criterion(out_fp32, targets)
            loss_int8 = criterion(out_int8.float(), targets)

            b_size = inputs.size(0)
            tracker_fp32.update(loss_fp32.item(), out_fp32, targets, b_size)
            tracker_int8.update(loss_int8.item(), out_int8, targets, b_size)

            all_preds_fp32.append(torch.argmax(out_fp32, dim=1).cpu().numpy())
            all_preds_int8.append(torch.argmax(out_int8, dim=1).cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    preds_fp32 = np.concatenate(all_preds_fp32)
    preds_int8 = np.concatenate(all_preds_int8)
    targets_np = np.concatenate(all_targets)

    metrics_fp32 = tracker_fp32.compute()
    metrics_int8 = tracker_int8.compute()

    concordant = int((preds_fp32 == preds_int8).sum())
    total = len(preds_fp32)
    concordance_rate = concordant / total
    changed_preds = int(total - concordant)

    acc_32 = metrics_fp32.accuracy * 100
    acc_int8 = metrics_int8.accuracy * 100
    delta_pp = acc_int8 - acc_32

    print(f"    FP32 Accuracy: {acc_32:.2f}% | Loss: {metrics_fp32.loss:.4f} | Correct: {metrics_fp32.correct_predictions}/{total}")
    print(f"    INT8 Accuracy: {acc_int8:.2f}% | Loss: {metrics_int8.loss:.4f} | Correct: {metrics_int8.correct_predictions}/{total}")
    print(f"    Accuracy delta: {delta_pp:+.2f} pp")
    print(f"    Concordance: {concordance_rate*100:.2f}% ({concordant}/{total}) | Changed predictions: {changed_preds}")

    # Per-class delta
    per_class_delta = {}
    for c in CIFAR10_CLASSES:
        p32 = metrics_fp32.per_class_accuracy[c] * 100
        pint8 = metrics_int8.per_class_accuracy[c] * 100
        per_class_delta[c] = round(pint8 - p32, 2)

    return {
        "model_architecture": model_name,
        "total_samples": total,
        "fp32_accuracy": round(acc_32, 2),
        "int8_accuracy": round(acc_int8, 2),
        "accuracy_diff_pp": round(delta_pp, 2),
        "fp32_loss": round(metrics_fp32.loss, 4),
        "int8_loss": round(metrics_int8.loss, 4),
        "fp32_correct": metrics_fp32.correct_predictions,
        "int8_correct": metrics_int8.correct_predictions,
        "concordant_samples": concordant,
        "changed_predictions": changed_preds,
        "concordance_pct": round(concordance_rate * 100, 2),
        "per_class_fp32": {k: round(v * 100, 2) for k, v in metrics_fp32.per_class_accuracy.items()},
        "per_class_int8": {k: round(v * 100, 2) for k, v in metrics_int8.per_class_accuracy.items()},
        "per_class_delta_pp": per_class_delta,
        "confusion_matrix_int8": metrics_int8.confusion_matrix,
    }


def main():
    device = torch.device("cpu")
    print(f"[*] Running clean test set evaluation on {device}...")

    test_loader = get_evaluation_dataloader(data_config_path="configs/cifar10.yaml", split="test", batch_size=64)
    criterion = nn.CrossEntropyLoss()

    models = [
        ("resnet18_cifar", "experiments/checkpoints_inference_fp32/resnet18_cifar_fp32.pt", "experiments/checkpoints_int8/resnet18_cifar_int8.pt"),
        ("mobilenet_v3_small_cifar", "experiments/checkpoints_inference_fp32/mobilenet_v3_small_cifar_fp32.pt", "experiments/checkpoints_int8/mobilenet_v3_small_cifar_int8.pt"),
    ]

    results = {}
    for m_arch, p_32, p_8 in models:
        results[m_arch] = evaluate_clean_pair(p_32, p_8, m_arch, test_loader, criterion, device)

    out_file = Path("reports/experiment_results/clean_test_int8_comparison.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[+] Saved clean test comparison to {out_file}")


if __name__ == "__main__":
    main()

