#!/usr/bin/env python3
"""
Generates publication-quality robustness visualization figures for Task 4.2.
Reads actual experiment results from experiments/robustness_fp32_seed42/results.csv
and results.json, and outputs figures to experiments/robustness_fp32_seed42/plots/.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

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


def plot_accuracy_vs_severity(df: pd.DataFrame, out_path: Path) -> None:
    """Plots accuracy vs severity (0=clean to 5) in a 2x3 grid across all 6 corruptions."""
    corruptions = ["brightness", "contrast", "gaussian_blur", "gaussian_noise", "jpeg_compression", "motion_blur"]
    clean_df = df[df["condition"] == "clean"]
    clean_accs = {
        "resnet18_cifar": clean_df[clean_df["model"] == "resnet18_cifar"]["accuracy"].iloc[0] * 100.0,
        "mobilenet_v3_small_cifar": clean_df[clean_df["model"] == "mobilenet_v3_small_cifar"]["accuracy"].iloc[0] * 100.0,
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=300, sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, corr in enumerate(corruptions):
        ax = axes[idx]
        corr_df = df[df["corruption"] == corr]

        # ResNet18 curve (sev 0 to 5)
        res_sub = corr_df[corr_df["model"] == "resnet18_cifar"].sort_values("severity")
        sevs_res = [0] + res_sub["severity"].tolist()
        accs_res = [clean_accs["resnet18_cifar"]] + (res_sub["accuracy"] * 100.0).tolist()

        # MobileNet curve (sev 0 to 5)
        mob_sub = corr_df[corr_df["model"] == "mobilenet_v3_small_cifar"].sort_values("severity")
        sevs_mob = [0] + mob_sub["severity"].tolist()
        accs_mob = [clean_accs["mobilenet_v3_small_cifar"]] + (mob_sub["accuracy"] * 100.0).tolist()

        ax.plot(sevs_res, accs_res, "o-", color="#1f77b4", linewidth=2.2, markersize=6, label="ResNet-18")
        ax.plot(sevs_mob, accs_mob, "s--", color="#ff7f0e", linewidth=2.2, markersize=6, label="MobileNetV3-Small")

        # Crossover indicator if MobileNet outperforms ResNet
        crossovers = [s for s, r, m in zip(sevs_res, accs_res, accs_mob) if m > r]
        if crossovers:
            for s in crossovers:
                m_val = accs_mob[s]
                ax.scatter([s], [m_val], color="#2ca02c", s=80, zorder=5)

        title = corr.replace("_", " ").title()
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        ax.set_xticks([0, 1, 2, 3, 4, 5])
        ax.set_xticklabels(["Clean (0)", "1", "2", "3", "4", "5"], fontsize=10)
        ax.set_ylim(10.0, 100.0)
        ax.grid(True)
        if idx % 3 == 0:
            ax.set_ylabel("Top-1 Accuracy (%)", fontsize=11)
        if idx >= 3:
            ax.set_xlabel("Severity Level", fontsize=11)
        if idx == 0:
            ax.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=10)

    fig.suptitle("FP32 CIFAR-10 Accuracy vs. Corruption Severity by Degradation Type", fontsize=15, fontweight="bold", y=1.02)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_accuracy_drop_vs_severity(df: pd.DataFrame, out_path: Path) -> None:
    """Plots accuracy degradation (percentage points drop) vs severity (1 to 5) across corruptions."""
    corruptions = ["brightness", "contrast", "gaussian_blur", "gaussian_noise", "jpeg_compression", "motion_blur"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=300, sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, corr in enumerate(corruptions):
        ax = axes[idx]
        corr_df = df[df["corruption"] == corr]

        res_sub = corr_df[corr_df["model"] == "resnet18_cifar"].sort_values("severity")
        mob_sub = corr_df[corr_df["model"] == "mobilenet_v3_small_cifar"].sort_values("severity")

        ax.plot(res_sub["severity"], res_sub["accuracy_drop_pp"], "o-", color="#1f77b4", linewidth=2.2, markersize=6, label="ResNet-18")
        ax.plot(mob_sub["severity"], mob_sub["accuracy_drop_pp"], "s--", color="#ff7f0e", linewidth=2.2, markersize=6, label="MobileNetV3-Small")

        # Zero reference line
        ax.axhline(0.0, color="#888888", linestyle=":", linewidth=1.0)

        title = corr.replace("_", " ").title()
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_ylim(-5.0, 75.0)
        ax.grid(True)
        if idx % 3 == 0:
            ax.set_ylabel("Accuracy Drop (pp)", fontsize=11)
        if idx >= 3:
            ax.set_xlabel("Severity Level", fontsize=11)
        if idx == 0:
            ax.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=10)

    fig.suptitle("Predictive Accuracy Degradation (Percentage Points Drop) vs. Severity", fontsize=15, fontweight="bold", y=1.02)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_per_corruption_comparison(df: pd.DataFrame, summary_data: dict, out_path: Path) -> None:
    """Grouped bar chart of mean corrupted accuracy and clean accuracy across corruptions."""
    corruptions = ["clean", "brightness", "contrast", "gaussian_blur", "gaussian_noise", "jpeg_compression", "motion_blur"]
    display_names = ["Clean Baseline", "Brightness", "Contrast", "Gaussian Blur", "Gaussian Noise", "JPEG Comp.", "Motion Blur"]

    res_accs = []
    mob_accs = []

    res_sum = summary_data["model_summaries"]["resnet18_cifar"]
    mob_sum = summary_data["model_summaries"]["mobilenet_v3_small_cifar"]

    res_accs.append(res_sum["clean_accuracy"] * 100.0)
    mob_accs.append(mob_sum["clean_accuracy"] * 100.0)

    for c in corruptions[1:]:
        res_accs.append(res_sum["per_corruption_summary"][c]["mean_accuracy"] * 100.0)
        mob_accs.append(mob_sum["per_corruption_summary"][c]["mean_accuracy"] * 100.0)

    x = np.arange(len(corruptions))
    width = 0.36

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    rects1 = ax.bar(x - width / 2, res_accs, width, label="ResNet-18 (FP32)", color="#1f77b4", edgecolor="#0e4e7b")
    rects2 = ax.bar(x + width / 2, mob_accs, width, label="MobileNetV3-Small (FP32)", color="#ff7f0e", edgecolor="#b85600")

    # Add numeric labels on bars
    for r in rects1:
        h = r.get_height()
        ax.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    for r in rects2:
        h = r.get_height()
        ax.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_title("Mean Predictive Accuracy Across Clean & Corruption Degradation Types", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, fontsize=10, rotation=15)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=10, loc="upper right")

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_overall_severity_degradation(summary_data: dict, out_path: Path) -> None:
    """Plots overall mean accuracy as severity increases from 0 (clean) to 5."""
    res_sum = summary_data["model_summaries"]["resnet18_cifar"]
    mob_sum = summary_data["model_summaries"]["mobilenet_v3_small_cifar"]

    sevs = [0, 1, 2, 3, 4, 5]
    res_vals = [res_sum["clean_accuracy"] * 100.0] + [res_sum["per_severity_summary"][str(s)]["mean_accuracy"] * 100.0 for s in range(1, 6)]
    mob_vals = [mob_sum["clean_accuracy"] * 100.0] + [mob_sum["per_severity_summary"][str(s)]["mean_accuracy"] * 100.0 for s in range(1, 6)]

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    ax.plot(sevs, res_vals, "o-", color="#1f77b4", linewidth=2.5, markersize=8, label=f"ResNet-18 (Mean: {res_sum['mean_corrupted_accuracy']*100:.2f}%)")
    ax.plot(sevs, mob_vals, "s--", color="#ff7f0e", linewidth=2.5, markersize=8, label=f"MobileNetV3-Small (Mean: {mob_sum['mean_corrupted_accuracy']*100:.2f}%)")

    # Annotate points
    for s, r, m in zip(sevs, res_vals, mob_vals):
        ax.annotate(f"{r:.1f}%", xy=(s, r), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=9, fontweight="bold", color="#1f77b4")
        ax.annotate(f"{m:.1f}%", xy=(s, m), xytext=(0, -14), textcoords="offset points", ha="center", fontsize=9, fontweight="bold", color="#ff7f0e")

    ax.set_title("Overall Mean Accuracy Degradation by Severity Level (Across All Corruptions)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Severity Level", fontsize=11)
    ax.set_ylabel("Mean Top-1 Accuracy (%)", fontsize=11)
    ax.set_xticks(sevs)
    ax.set_xticklabels(["0 (Clean)", "1", "2", "3", "4", "5"], fontsize=10)
    ax.set_ylim(20, 100)
    ax.grid(True)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=10)

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    setup_style()
    exp_dir = Path("experiments/robustness_fp32_seed42")
    if not exp_dir.is_dir():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    plots_dir = exp_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(exp_dir / "results.csv")
    with open(exp_dir / "summary.json", "r", encoding="utf-8") as f:
        summary_data = json.load(f)

    plot_accuracy_vs_severity(df, plots_dir / "accuracy_vs_severity.png")
    plot_accuracy_drop_vs_severity(df, plots_dir / "accuracy_drop_vs_severity.png")
    plot_per_corruption_comparison(df, summary_data, plots_dir / "per_corruption_comparison.png")
    plot_overall_severity_degradation(summary_data, plots_dir / "overall_severity_degradation.png")

    # Also copy to reports/figures/robustness/ for project documentation
    reports_plots_dir = Path("reports/figures/robustness")
    reports_plots_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    for p in plots_dir.glob("*.png"):
        shutil.copy(p, reports_plots_dir / p.name)
    print(f"All figures copied to: {reports_plots_dir}")


if __name__ == "__main__":
    main()
