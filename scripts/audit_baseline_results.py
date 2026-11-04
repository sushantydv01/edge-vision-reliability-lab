"""
Independent numerical audit script for baseline experiment artifacts (Task 2.3B).
Verifies CSV histories, summary JSONs, checkpoints, and evaluation reports.
"""

import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from edge_vision.models import get_model
from edge_vision.models.metadata import get_model_summary


def audit_experiment(
    exp_dir: str,
    eval_json_path: str,
    expected_arch: str,
    expected_exp_id: str,
) -> dict:
    print("=" * 80)
    print(f"AUDITING EXPERIMENT: {exp_dir}")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. CSV History Verification
    # -------------------------------------------------------------------------
    csv_path = Path(exp_dir) / "metrics_history.csv"
    assert csv_path.is_file(), f"Missing CSV at {csv_path}"
    df = pd.read_csv(csv_path)

    total_epochs = len(df)
    print(f"1. Epoch Count: {total_epochs}")
    assert total_epochs == 100, f"Expected 100 epochs, got {total_epochs}"
    assert list(df["epoch"]) == list(range(1, 101)), "Epoch index sequence is non-consecutive"

    # Best validation accuracy & corresponding epoch
    idx_max_val_acc = df["val_acc"].idxmax()
    best_val_acc = float(df.loc[idx_max_val_acc, "val_acc"])
    best_val_acc_epoch = int(df.loc[idx_max_val_acc, "epoch"])

    # Best validation loss & corresponding epoch
    idx_min_val_loss = df["val_loss"].idxmin()
    best_val_loss = float(df.loc[idx_min_val_loss, "val_loss"])
    best_val_loss_epoch = int(df.loc[idx_min_val_loss, "epoch"])

    # Final epoch metrics (Epoch 100)
    final_row = df.iloc[-1]
    final_train_loss = float(final_row["train_loss"])
    final_train_acc = float(final_row["train_acc"])
    final_val_loss = float(final_row["val_loss"])
    final_val_acc = float(final_row["val_acc"])

    # Learning rate progression
    lr_start = float(df.loc[0, "learning_rate"])
    lr_mid = float(df.loc[49, "learning_rate"])
    lr_end = float(df.loc[99, "learning_rate"])

    # Epoch duration metrics
    total_epoch_duration = float(df["epoch_duration_sec"].sum())
    mean_epoch_duration = float(df["epoch_duration_sec"].mean())

    print(f"2. Best Val Accuracy:   {best_val_acc:.4f} ({best_val_acc*100:.2f}%) at Epoch {best_val_acc_epoch}")
    print(f"3. Best Val Loss:       {best_val_loss:.4f} at Epoch {best_val_loss_epoch}")
    print(f"4. Final Train Loss:    {final_train_loss:.6f} | Train Acc: {final_train_acc*100:.2f}%")
    print(f"5. Final Val Loss:      {final_val_loss:.6f} | Val Acc:   {final_val_acc*100:.2f}%")
    print(f"6. LR Progression:      Start={lr_start:.6e}, Ep50={lr_mid:.6e}, Ep100={lr_end:.6e}")
    print(f"7. Epoch Duration Sum:  {total_epoch_duration:.2f}s ({total_epoch_duration/3600:.2f} hours)")
    print(f"8. Mean Epoch Duration: {mean_epoch_duration:.2f}s per epoch")

    # -------------------------------------------------------------------------
    # 2. Experiment Summary JSON Verification
    # -------------------------------------------------------------------------
    summary_path = Path(exp_dir) / "experiment_summary.json"
    assert summary_path.is_file(), f"Missing summary JSON at {summary_path}"
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    print("\n--- Summary JSON Cross-Check ---")
    sum_exp_id = summary.get("experiment_id")
    sum_arch = summary.get("model_architecture")
    sum_best_ep = summary.get("best_metrics", {}).get("best_epoch")
    sum_best_acc = summary.get("best_metrics", {}).get("best_val_acc")
    sum_best_loss = summary.get("best_metrics", {}).get("best_val_loss")
    sum_wall_time = summary.get("timestamps", {}).get("total_duration_seconds")

    print(f"  summary experiment_id:    {sum_exp_id}")
    print(f"  summary architecture:     {sum_arch}")
    print(f"  summary best_epoch:       {sum_best_ep}")
    print(f"  summary best_val_acc:     {sum_best_acc:.4f}")
    print(f"  summary best_val_loss:    {sum_best_loss:.4f}")
    print(f"  summary total wall time:  {sum_wall_time:.2f}s ({sum_wall_time/3600:.2f} hours)")

    assert sum_exp_id == expected_exp_id, f"Summary exp_id {sum_exp_id} != {expected_exp_id}"
    assert sum_arch == expected_arch, f"Summary arch {sum_arch} != {expected_arch}"
    assert sum_best_ep == best_val_acc_epoch, f"Summary best_epoch {sum_best_ep} != {best_val_acc_epoch}"
    assert abs(sum_best_acc - best_val_acc) < 1e-5, f"Summary best_acc mismatch"

    # -------------------------------------------------------------------------
    # 3. Checkpoint Artifacts & Parameter Audit
    # -------------------------------------------------------------------------
    ckpt_dir = Path(exp_dir) / "checkpoints"
    best_ckpt_path = ckpt_dir / "best_checkpoint.pt"
    latest_ckpt_path = ckpt_dir / "latest_checkpoint.pt"
    assert best_ckpt_path.is_file(), f"Missing best checkpoint at {best_ckpt_path}"
    assert latest_ckpt_path.is_file(), f"Missing latest checkpoint at {latest_ckpt_path}"

    best_ckpt_size = best_ckpt_path.stat().st_size
    latest_ckpt_size = latest_ckpt_path.stat().st_size
    print(f"\n--- Checkpoint Audit ---")
    print(f"  Best Checkpoint Size:   {best_ckpt_size:,} bytes ({best_ckpt_size / (1024**2):.2f} MB)")
    print(f"  Latest Checkpoint Size: {latest_ckpt_size:,} bytes ({latest_ckpt_size / (1024**2):.2f} MB)")

    best_payload = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
    latest_payload = torch.load(latest_ckpt_path, map_location="cpu", weights_only=False)

    ckpt_epoch = best_payload.get("epoch")
    ckpt_best_epoch = best_payload.get("best_epoch")
    ckpt_best_metric = best_payload.get("best_metric_value")
    ckpt_arch = best_payload.get("model_architecture")
    ckpt_seed = best_payload.get("seed")

    print(f"  Payload epoch:            {ckpt_epoch}")
    print(f"  Payload best_epoch:       {ckpt_best_epoch}")
    print(f"  Payload best_metric:      {ckpt_best_metric:.4f}")
    print(f"  Payload architecture:     {ckpt_arch}")
    print(f"  Payload seed:             {ckpt_seed}")
    print(f"  Latest payload epoch:     {latest_payload.get('epoch')}")

    assert ckpt_epoch == best_val_acc_epoch, f"best_checkpoint saved epoch {ckpt_epoch} != {best_val_acc_epoch}"
    assert ckpt_best_epoch == best_val_acc_epoch, f"best_checkpoint recorded best_epoch {ckpt_best_epoch} != {best_val_acc_epoch}"
    assert abs(ckpt_best_metric - best_val_acc) < 1e-5, f"best_checkpoint metric mismatch"
    assert ckpt_arch == expected_arch, f"Checkpoint arch mismatch: {ckpt_arch}"
    assert ckpt_seed == 42, f"Checkpoint seed mismatch: {ckpt_seed}"
    assert latest_payload.get("epoch") == 100, f"Latest checkpoint epoch {latest_payload.get('epoch')} != 100"

    # Count parameters directly via model constructor
    model = get_model(expected_arch, num_classes=10)
    model.load_state_dict(best_payload["model_state_dict"], strict=True)
    summary_meta = get_model_summary(model)
    total_params = summary_meta["total_parameters"]
    trainable_params = summary_meta["trainable_parameters"]
    fp32_mb = summary_meta["parameter_mb"]
    fp32_mib = summary_meta["parameter_mib"]
    print(f"  Model Parameter Count:    {total_params:,} total ({trainable_params:,} trainable)")
    print(f"  Theoretical FP32 Memory:  {fp32_mb:.2f} MB ({fp32_mib:.2f} MiB)")

    # -------------------------------------------------------------------------
    # 4. Held-out Test Evaluation Report Verification
    # -------------------------------------------------------------------------
    eval_file = Path(eval_json_path)
    assert eval_file.is_file(), f"Missing evaluation report at {eval_file}"
    with open(eval_file, "r", encoding="utf-8") as f:
        eval_report = json.load(f)

    print(f"\n--- Held-Out Test Evaluation Audit ---")
    rep_run_id = eval_report.get("run_id")
    rep_arch = eval_report.get("model_architecture")
    rep_ckpt_epoch = eval_report.get("checkpoint", {}).get("epoch")
    rep_split = eval_report.get("evaluation_dataset", {}).get("split")
    rep_sample_count = eval_report.get("evaluation_dataset", {}).get("sample_count")

    print(f"  Report run_id:            {rep_run_id}")
    print(f"  Report architecture:      {rep_arch}")
    print(f"  Report checkpoint epoch:  {rep_ckpt_epoch}")
    print(f"  Report split:             {rep_split}")
    print(f"  Report sample count:      {rep_sample_count:,}")

    assert rep_run_id == expected_exp_id
    assert rep_arch == expected_arch
    assert rep_ckpt_epoch == best_val_acc_epoch
    assert rep_split == "test"
    assert rep_sample_count == 10000

    m = eval_report["metrics"]
    test_acc = float(m["accuracy"])
    test_loss = float(m["loss"])
    total_samples = int(m["total_samples"])
    correct_preds = int(m["correct_predictions"])
    cm = np.array(m["confusion_matrix"])

    print(f"  Test Accuracy:            {test_acc:.4f} ({test_acc * 100:.2f}%)")
    print(f"  Test Loss:                {test_loss:.4f}")
    print(f"  Correct Predictions:      {correct_preds:,} / {total_samples:,}")
    print(f"  Confusion Matrix Shape:   {cm.shape}")
    print(f"  Confusion Matrix Sum:     {cm.sum():,}")
    print(f"  Confusion Matrix Trace:   {np.trace(cm):,}")

    assert total_samples == 10000
    assert cm.shape == (10, 10)
    assert cm.sum() == total_samples, f"Matrix sum {cm.sum()} != {total_samples}"
    assert np.trace(cm) == correct_preds, f"Matrix trace {np.trace(cm)} != correct_preds {correct_preds}"
    assert abs((correct_preds / total_samples) - test_acc) < 1e-6, "Accuracy != correct / total"

    # Per-class assertions
    print(f"\n--- Per-Class Accuracy Audit ---")
    for cls_name, cls_acc in m["per_class_accuracy"].items():
        cnt = int(m["per_class_counts"][cls_name])
        cor = int(m["per_class_correct"][cls_name])
        calc_acc = cor / cnt
        assert cnt == 1000, f"Class {cls_name} count {cnt} != 1000"
        assert abs(calc_acc - cls_acc) < 1e-6, f"Class {cls_name} accuracy mismatch"
        print(f"  {cls_name:<12}: {cls_acc*100:>6.2f}% ({cor:>4}/{cnt})")

    print("\n>> All mathematical and integrity assertions PASSED for this experiment.")
    print("=" * 80 + "\n")

    return {
        "architecture": expected_arch,
        "experiment_id": expected_exp_id,
        "total_epochs": total_epochs,
        "best_val_acc": best_val_acc,
        "best_val_epoch": best_val_acc_epoch,
        "best_val_loss": best_val_loss,
        "best_val_loss_epoch": best_val_loss_epoch,
        "final_train_loss": final_train_loss,
        "final_train_acc": final_train_acc,
        "final_val_loss": final_val_loss,
        "final_val_acc": final_val_acc,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "correct_preds": correct_preds,
        "total_samples": total_samples,
        "wall_time_seconds": sum_wall_time,
        "total_epoch_duration": total_epoch_duration,
        "mean_epoch_duration": mean_epoch_duration,
        "best_checkpoint_size": best_ckpt_size,
        "param_count": total_params,
    }


def main() -> None:
    print("STARTING INDEPENDENT AUDIT OF TASK 2.3B EXPERIMENT ARTIFACTS\n")

    r18_audit = audit_experiment(
        exp_dir="experiments/exp_baseline_resnet18_seed42",
        eval_json_path="experiments/exp_baseline_resnet18_seed42/evaluation_test.json",
        expected_arch="resnet18_cifar",
        expected_exp_id="exp_baseline_resnet18_seed42",
    )

    mb3_audit = audit_experiment(
        exp_dir="experiments/exp_baseline_mobilenet_v3_small_seed42",
        eval_json_path="experiments/exp_baseline_mobilenet_v3_small_seed42/evaluation_test.json",
        expected_arch="mobilenet_v3_small_cifar",
        expected_exp_id="exp_baseline_mobilenet_v3_small_seed42",
    )

    print("AUDIT SUMMARY COMPARISON:")
    print("-" * 65)
    print(f"{'Metric':<30} | {'ResNet-18':<15} | {'MobileNetV3-Small':<15}")
    print("-" * 65)
    print(f"{'Best Val Accuracy':<30} | {r18_audit['best_val_acc']*100:>13.2f}% | {mb3_audit['best_val_acc']*100:>13.2f}%")
    print(f"{'Best Val Epoch':<30} | {r18_audit['best_val_epoch']:>14} | {mb3_audit['best_val_epoch']:>14}")
    print(f"{'Best Val Loss':<30} | {r18_audit['best_val_loss']:>14.4f} | {mb3_audit['best_val_loss']:>14.4f}")
    print(f"{'Test Accuracy':<30} | {r18_audit['test_acc']*100:>13.2f}% | {mb3_audit['test_acc']*100:>13.2f}%")
    print(f"{'Test Loss':<30} | {r18_audit['test_loss']:>14.4f} | {mb3_audit['test_loss']:>14.4f}")
    print(f"{'Final Train Loss':<30} | {r18_audit['final_train_loss']:>14.6f} | {mb3_audit['final_train_loss']:>14.6f}")
    print(f"{'Final Train Acc':<30} | {r18_audit['final_train_acc']*100:>13.2f}% | {mb3_audit['final_train_acc']*100:>13.2f}%")
    print(f"{'Final Val Loss':<30} | {r18_audit['final_val_loss']:>14.6f} | {mb3_audit['final_val_loss']:>14.6f}")
    print(f"{'Final Val Acc':<30} | {r18_audit['final_val_acc']*100:>13.2f}% | {mb3_audit['final_val_acc']*100:>13.2f}%")
    print(f"{'Mean Epoch Duration':<30} | {r18_audit['mean_epoch_duration']:>13.2f}s | {mb3_audit['mean_epoch_duration']:>13.2f}s")
    print(f"{'Total Wall Time':<30} | {r18_audit['wall_time_seconds']/3600:>12.2f}h | {mb3_audit['wall_time_seconds']/3600:>12.2f}h")
    print(f"{'Parameters':<30} | {r18_audit['param_count']:>14,} | {mb3_audit['param_count']:>14,}")
    print(f"{'Checkpoint Size (Disk)':<30} | {r18_audit['best_checkpoint_size']/(1024**2):>12.2f}MB | {mb3_audit['best_checkpoint_size']/(1024**2):>12.2f}MB")
    print("-" * 65)


if __name__ == "__main__":
    main()
