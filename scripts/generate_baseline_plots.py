"""
Generates learning curves and analytical visualizations for Task 2.3B baseline experiments.
Uses actual 100-epoch CSV histories from experiments/.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def setup_style() -> None:
    plt.rcParams.update({
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "axes.edgecolor": "#cccccc",
        "axes.linewidth": 1.0,
        "grid.color": "#e5e5e5",
        "grid.linestyle": "--",
        "grid.alpha": 0.7,
        "figure.autolayout": True,
    })


def plot_loss(
    df: pd.DataFrame,
    title: str,
    output_path: Path,
    model_name: str,
    exp_id: str,
    best_epoch: int,
    best_loss: float,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.plot(df["epoch"], df["train_loss"], label="Train Loss (Sample-Weighted)", color="#1f77b4", linewidth=2.0)
    ax.plot(df["epoch"], df["val_loss"], label="Val Loss (Canonical Eval)", color="#ff7f0e", linewidth=2.0)

    # Highlight best validation loss epoch
    ax.scatter([best_epoch], [best_loss], color="#d62728", s=60, zorder=5, label=f"Best Val Epoch ({best_epoch}: {best_loss:.4f})")
    ax.annotate(
        f"Epoch {best_epoch}\nLoss: {best_loss:.4f}",
        xy=(best_epoch, best_loss),
        xytext=(best_epoch - 18, best_loss + 0.15),
        arrowprops=dict(facecolor="#d62728", shrink=0.08, width=1.5, headwidth=6),
        fontsize=9,
        fontweight="bold",
        color="#d62728",
    )

    ax.set_title(f"{title}\n[{exp_id}]", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=11)
    ax.set_xlim(1, 100)
    ax.set_ylim(bottom=0.0)
    ax.grid(True)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=10)

    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_accuracy(
    df: pd.DataFrame,
    title: str,
    output_path: Path,
    model_name: str,
    exp_id: str,
    best_epoch: int,
    best_acc: float,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    train_acc_pct = df["train_acc"] * 100.0
    val_acc_pct = df["val_acc"] * 100.0

    ax.plot(df["epoch"], train_acc_pct, label="Train Accuracy (Augmented)", color="#1f77b4", linewidth=2.0)
    ax.plot(df["epoch"], val_acc_pct, label="Val Accuracy (Canonical)", color="#2ca02c", linewidth=2.0)

    # Highlight best validation accuracy epoch
    best_acc_pct = best_acc * 100.0
    ax.scatter([best_epoch], [best_acc_pct], color="#d62728", s=60, zorder=5, label=f"Best Val Epoch ({best_epoch}: {best_acc_pct:.2f}%)")
    ax.annotate(
        f"Epoch {best_epoch}\nAcc: {best_acc_pct:.2f}%",
        xy=(best_epoch, best_acc_pct),
        xytext=(best_epoch - 22, best_acc_pct - 12.0),
        arrowprops=dict(facecolor="#d62728", shrink=0.08, width=1.5, headwidth=6),
        fontsize=9,
        fontweight="bold",
        color="#d62728",
    )

    ax.set_title(f"{title}\n[{exp_id}]", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=11)
    ax.set_xlim(1, 100)
    ax.set_ylim(20.0, 102.0)
    ax.grid(True)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=10, loc="lower right")

    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_learning_rate(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    ax.plot(df["epoch"], df["learning_rate"], label="Cosine Annealing LR", color="#9467bd", linewidth=2.2)

    ax.set_title(
        "Learning Rate Schedule: CosineAnnealingLR\n[Initial lr=0.05, T_max=100, eta_min=0.0001]",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Learning Rate", fontsize=11)
    ax.set_xlim(1, 100)
    ax.set_ylim(0.0, 0.052)
    ax.grid(True)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=10)

    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_validation_comparison(
    df_r18: pd.DataFrame,
    df_mb3: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

    r18_val_pct = df_r18["val_acc"] * 100.0
    mb3_val_pct = df_mb3["val_acc"] * 100.0

    ax.plot(df_r18["epoch"], r18_val_pct, label="ResNet-18 (Peak Val: 95.50% at Ep 97, Test: 94.78%)", color="#1f77b4", linewidth=2.2)
    ax.plot(df_mb3["epoch"], mb3_val_pct, label="MobileNetV3-Small (Peak Val: 86.86% at Ep 98, Test: 86.51%)", color="#ff7f0e", linewidth=2.2)

    # Shaded accuracy gap area
    ax.fill_between(df_r18["epoch"], r18_val_pct, mb3_val_pct, color="#cccccc", alpha=0.3, label="Controlled Baseline Accuracy Gap")

    ax.set_title(
        "Validation Accuracy Comparison: ResNet-18 vs. MobileNetV3-Small\n[Controlled SGD Baseline, Seed 42, 100 Epochs on Apple Silicon MPS]",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Validation Top-1 Accuracy (%)", fontsize=11)
    ax.set_xlim(1, 100)
    ax.set_ylim(25.0, 100.0)
    ax.grid(True)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=9.5, loc="lower right")

    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_per_class_comparison(
    r18_eval_path: Path,
    mb3_eval_path: Path,
    output_path: Path,
) -> None:
    with open(r18_eval_path, "r", encoding="utf-8") as f:
        r18_rep = json.load(f)
    with open(mb3_eval_path, "r", encoding="utf-8") as f:
        mb3_rep = json.load(f)

    classes = list(r18_rep["metrics"]["per_class_accuracy"].keys())
    r18_accs = [r18_rep["metrics"]["per_class_accuracy"][c] * 100.0 for c in classes]
    mb3_accs = [mb3_rep["metrics"]["per_class_accuracy"][c] * 100.0 for c in classes]

    x = np.arange(len(classes))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    rects1 = ax.bar(x - width / 2, r18_accs, width, label=f"ResNet-18 (Overall: {r18_rep['metrics']['accuracy']*100:.2f}%)", color="#1f77b4")
    rects2 = ax.bar(x + width / 2, mb3_accs, width, label=f"MobileNetV3-Small (Overall: {mb3_rep['metrics']['accuracy']*100:.2f}%)", color="#ff7f0e")

    ax.set_ylabel("Clean Test Accuracy (%)", fontsize=11)
    ax.set_title(
        "Per-Class Held-Out Test Accuracy Breakdown (10,000 CIFAR-10 Test Samples)\n[ResNet-18 vs. MobileNetV3-Small]",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in classes], fontsize=10, rotation=25)
    ax.set_ylim(60.0, 102.0)
    ax.grid(True, axis="y")
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=10, loc="lower right")

    # Add accuracy delta labels above bars
    for i, (r_acc, m_acc) in enumerate(zip(r18_accs, mb3_accs)):
        delta = m_acc - r_acc
        color = "#d62728" if delta < -10 else "#555555"
        ax.text(x[i] + width / 2, m_acc + 1.2, f"{delta:+.1f}%", ha="center", va="bottom", fontsize=7.5, color=color, fontweight="bold")

    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path}")


def main() -> None:
    setup_style()
    figures_dir = Path("reports") / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    r18_csv = Path("experiments") / "exp_baseline_resnet18_seed42" / "metrics_history.csv"
    mb3_csv = Path("experiments") / "exp_baseline_mobilenet_v3_small_seed42" / "metrics_history.csv"

    df_r18 = pd.read_csv(r18_csv)
    df_mb3 = pd.read_csv(mb3_csv)

    # 1. ResNet-18 Loss
    plot_loss(
        df=df_r18,
        title="ResNet-18 Baseline: Training vs. Validation Loss",
        output_path=figures_dir / "baseline_resnet18_loss.png",
        model_name="resnet18_cifar",
        exp_id="exp_baseline_resnet18_seed42",
        best_epoch=97,
        best_loss=float(df_r18.loc[96, "val_loss"]),
    )

    # 2. ResNet-18 Accuracy
    plot_accuracy(
        df=df_r18,
        title="ResNet-18 Baseline: Training vs. Validation Accuracy",
        output_path=figures_dir / "baseline_resnet18_accuracy.png",
        model_name="resnet18_cifar",
        exp_id="exp_baseline_resnet18_seed42",
        best_epoch=97,
        best_acc=float(df_r18.loc[96, "val_acc"]),
    )

    # 3. MobileNetV3 Loss
    plot_loss(
        df=df_mb3,
        title="MobileNetV3-Small Baseline: Training vs. Validation Loss",
        output_path=figures_dir / "baseline_mobilenet_loss.png",
        model_name="mobilenet_v3_small_cifar",
        exp_id="exp_baseline_mobilenet_v3_small_seed42",
        best_epoch=98,
        best_loss=float(df_mb3.loc[97, "val_loss"]),
    )

    # 4. MobileNetV3 Accuracy
    plot_accuracy(
        df=df_mb3,
        title="MobileNetV3-Small Baseline: Training vs. Validation Accuracy",
        output_path=figures_dir / "baseline_mobilenet_accuracy.png",
        model_name="mobilenet_v3_small_cifar",
        exp_id="exp_baseline_mobilenet_v3_small_seed42",
        best_epoch=98,
        best_acc=float(df_mb3.loc[97, "val_acc"]),
    )

    # 5. Learning Rate
    plot_learning_rate(
        df=df_r18,
        output_path=figures_dir / "baseline_learning_rate.png",
    )

    # 6. Validation Accuracy Comparison
    plot_validation_comparison(
        df_r18=df_r18,
        df_mb3=df_mb3,
        output_path=figures_dir / "baseline_validation_comparison.png",
    )

    # 7. Per-Class Test Accuracy Comparison
    plot_per_class_comparison(
        r18_eval_path=Path("experiments") / "exp_baseline_resnet18_seed42" / "evaluation_test.json",
        mb3_eval_path=Path("experiments") / "exp_baseline_mobilenet_v3_small_seed42" / "evaluation_test.json",
        output_path=figures_dir / "baseline_per_class_test_accuracy.png",
    )

    print("\nAll baseline figures generated successfully in reports/figures/")


if __name__ == "__main__":
    main()
