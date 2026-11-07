#!/usr/bin/env python3
"""
Publication-Quality Visualization Suite for Task 5.4 Cross-Precision Analysis.
Generates 8 figures comparing FP32, FP16, and INT8 across ResNet-18 and MobileNetV3-Small:
1. Artifact size by architecture and precision
2. Clean test accuracy comparison
3. Mean corrupted accuracy comparison
4. Accuracy versus severity for all precisions (2x3 grid)
5. Per-corruption accuracy comparison across precisions
6. CPU batch-1 latency comparison across thread counts
7. CPU batch-64 throughput comparison across thread counts
8. Efficiency-reliability Pareto trade-off curves (2x2 grid)
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from edge_vision.analysis import (
    MODEL_METADATA,
    build_clean_accuracy_table,
    build_cpu_performance_table,
    build_model_size_table,
    build_pareto_analysis,
    build_per_corruption_table,
    build_per_severity_table,
    build_robustness_summary_table,
    load_cross_precision_data,
)


def setup_matplotlib_style() -> None:
    """Sets publication-quality styling parameters."""
    plt.rcParams.update({
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "axes.edgecolor": "#cccccc",
        "axes.linewidth": 1.0,
        "grid.color": "#e8e8e8",
        "grid.linestyle": "--",
        "grid.alpha": 0.8,
        "legend.frameon": True,
        "legend.edgecolor": "#cccccc",
        "figure.autolayout": True,
    })


def plot_01_artifact_size(size_data: List[Dict], out_path: Path) -> None:
    """Figure 1: Artifact size comparison with compression ratios."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    models = ["resnet18_cifar", "mobilenet_v3_small_cifar"]
    precisions = ["FP32", "FP16", "INT8"]
    colors = {
        ("resnet18_cifar", "FP32"): "#1f77b4",
        ("resnet18_cifar", "FP16"): "#4b9cd3",
        ("resnet18_cifar", "INT8"): "#084081",
        ("mobilenet_v3_small_cifar", "FP32"): "#ff7f0e",
        ("mobilenet_v3_small_cifar", "FP16"): "#fdae6b",
        ("mobilenet_v3_small_cifar", "INT8"): "#a63603",
    }

    # Grouped bars
    x = np.arange(len(models))
    width = 0.25

    for i, prec in enumerate(precisions):
        sizes_mb = []
        labels = []
        bar_colors = []
        for m in models:
            match = [r for r in size_data if r["model"] == m and r["precision"] == prec][0]
            sizes_mb.append(match["serialized_mb"])
            bar_colors.append(colors[(m, prec)])
            if prec == "FP32":
                labels.append(f"{match['serialized_mb']:.1f} MB\n(1.00×)")
            else:
                labels.append(f"{match['serialized_mb']:.1f} MB\n({match['compression_ratio']:.2f}×)")

        rects = ax.bar(x + (i - 1) * width, sizes_mb, width, label=prec, color=bar_colors, edgecolor="#333333", linewidth=0.8)

        # Bar value labels
        for rect, lbl in zip(rects, labels):
            h = rect.get_height()
            ax.annotate(lbl,
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_ylabel("Serialized Inference Checkpoint Size (MB / 10^6 bytes)", fontsize=11, fontweight="bold")
    ax.set_title("Model Artifact Size and Compression Ratios Across Precisions", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(["ResNet-18\n(11.17M params)", "MobileNetV3-Small\n(1.53M params)"], fontsize=11, fontweight="bold")
    ax.set_ylim(0, 52)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Custom legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#333333", alpha=0.9),
        plt.Rectangle((0, 0), 1, 1, color="#666666", alpha=0.7),
        plt.Rectangle((0, 0), 1, 1, color="#999999", alpha=0.5),
    ]
    ax.legend(handles, ["FP32 Baseline", "FP16 Precision", "INT8 Static PTQ"], loc="upper right", fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_02_clean_accuracy(clean_data: List[Dict], out_path: Path) -> None:
    """Figure 2: Clean test accuracy comparison."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    models = ["resnet18_cifar", "mobilenet_v3_small_cifar"]
    precisions = ["FP32", "FP16", "INT8"]
    colors = {
        ("resnet18_cifar", "FP32"): "#1f77b4",
        ("resnet18_cifar", "FP16"): "#4b9cd3",
        ("resnet18_cifar", "INT8"): "#084081",
        ("mobilenet_v3_small_cifar", "FP32"): "#ff7f0e",
        ("mobilenet_v3_small_cifar", "FP16"): "#fdae6b",
        ("mobilenet_v3_small_cifar", "INT8"): "#a63603",
    }

    x = np.arange(len(models))
    width = 0.25

    for i, prec in enumerate(precisions):
        accs = []
        labels = []
        bar_colors = []
        for m in models:
            match = [r for r in clean_data if r["model"] == m and r["precision"] == prec][0]
            accs.append(match["clean_accuracy"])
            bar_colors.append(colors[(m, prec)])
            if prec == "FP32":
                labels.append(f"{match['clean_accuracy']:.2f}%")
            else:
                diff = match["accuracy_diff_pp"]
                sign = "+" if diff >= 0 else ""
                labels.append(f"{match['clean_accuracy']:.2f}%\n({sign}{diff:.2f} pp)")

        rects = ax.bar(x + (i - 1) * width, accs, width, label=prec, color=bar_colors, edgecolor="#333333", linewidth=0.8)

        for rect, lbl in zip(rects, labels):
            h = rect.get_height()
            ax.annotate(lbl,
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_ylabel("Clean Test Accuracy (%) [10,000 samples]", fontsize=11, fontweight="bold")
    ax.set_title("Clean CIFAR-10 Test Accuracy Comparison Across Precisions", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(["ResNet-18", "MobileNetV3-Small"], fontsize=11, fontweight="bold")
    ax.set_ylim(80, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#1f77b4"),
        plt.Rectangle((0, 0), 1, 1, color="#4b9cd3"),
        plt.Rectangle((0, 0), 1, 1, color="#084081"),
    ]
    ax.legend(handles, ["FP32", "FP16", "INT8"], loc="lower right", fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_03_corrupted_accuracy(rob_data: List[Dict], out_path: Path) -> None:
    """Figure 3: Mean corrupted accuracy comparison."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    models = ["resnet18_cifar", "mobilenet_v3_small_cifar"]
    precisions = ["FP32", "FP16", "INT8"]
    colors = {
        ("resnet18_cifar", "FP32"): "#1f77b4",
        ("resnet18_cifar", "FP16"): "#4b9cd3",
        ("resnet18_cifar", "INT8"): "#084081",
        ("mobilenet_v3_small_cifar", "FP32"): "#ff7f0e",
        ("mobilenet_v3_small_cifar", "FP16"): "#fdae6b",
        ("mobilenet_v3_small_cifar", "INT8"): "#a63603",
    }

    x = np.arange(len(models))
    width = 0.25

    for i, prec in enumerate(precisions):
        accs = []
        labels = []
        bar_colors = []
        for m in models:
            match = [r for r in rob_data if r["model"] == m and r["precision"] == prec][0]
            accs.append(match["mean_corrupted_accuracy"])
            bar_colors.append(colors[(m, prec)])
            labels.append(f"{match['mean_corrupted_accuracy']:.2f}%\n(drop: -{match['mean_accuracy_drop_pp']:.1f} pp)")

        rects = ax.bar(x + (i - 1) * width, accs, width, label=prec, color=bar_colors, edgecolor="#333333", linewidth=0.8)

        for rect, lbl in zip(rects, labels):
            h = rect.get_height()
            ax.annotate(lbl,
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_ylabel("Mean Corrupted Accuracy (%) [30 conditions]", fontsize=11, fontweight="bold")
    ax.set_title("Mean Out-of-Distribution Corrupted Accuracy Across Precisions", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(["ResNet-18", "MobileNetV3-Small"], fontsize=11, fontweight="bold")
    ax.set_ylim(45, 75)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#1f77b4"),
        plt.Rectangle((0, 0), 1, 1, color="#4b9cd3"),
        plt.Rectangle((0, 0), 1, 1, color="#084081"),
    ]
    ax.legend(handles, ["FP32", "FP16", "INT8"], loc="lower right", fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_04_accuracy_vs_severity(dfs: Dict[str, pd.DataFrame], out_path: Path) -> None:
    """Figure 4: Accuracy vs severity across all 6 corruptions and 3 precisions."""
    corruptions = ["brightness", "contrast", "gaussian_blur", "gaussian_noise", "jpeg_compression", "motion_blur"]
    titles = [
        "Brightness Darkening",
        "Contrast Attenuation",
        "Gaussian Blur",
        "Gaussian Noise",
        "JPEG Compression",
        "Motion Blur",
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=300, sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, (corr, title) in enumerate(zip(corruptions, titles)):
        ax = axes[idx]

        # ResNet18 curves
        for prec, style, col in [("fp32", "-", "#1f77b4"), ("fp16", "--", "#4b9cd3"), ("int8", ":", "#084081")]:
            df = dfs[prec]
            clean_acc = df[(df["model"] == "resnet18_cifar") & (df["condition"] == "clean")]["accuracy"].iloc[0] * 100.0
            sub = df[(df["model"] == "resnet18_cifar") & (df["corruption"] == corr)].sort_values("severity")
            sevs = [0] + sub["severity"].tolist()
            accs = [clean_acc] + (sub["accuracy"] * 100.0).tolist()
            ax.plot(sevs, accs, style, color=col, linewidth=1.8, marker="o" if prec == "fp32" else None,
                    markersize=4, label=f"ResNet-18 {prec.upper()}")

        # MobileNetV3 curves
        for prec, style, col in [("fp32", "-", "#ff7f0e"), ("fp16", "--", "#fdae6b"), ("int8", ":", "#a63603")]:
            df = dfs[prec]
            clean_acc = df[(df["model"] == "mobilenet_v3_small_cifar") & (df["condition"] == "clean")]["accuracy"].iloc[0] * 100.0
            sub = df[(df["model"] == "mobilenet_v3_small_cifar") & (df["corruption"] == corr)].sort_values("severity")
            sevs = [0] + sub["severity"].tolist()
            accs = [clean_acc] + (sub["accuracy"] * 100.0).tolist()
            ax.plot(sevs, accs, style, color=col, linewidth=1.8, marker="s" if prec == "fp32" else None,
                    markersize=4, label=f"MobileNet {prec.upper()}")

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks(range(6))
        ax.set_xticklabels(["Clean", "1", "2", "3", "4", "5"])
        ax.grid(True, linestyle="--", alpha=0.6)
        if idx % 3 == 0:
            ax.set_ylabel("Accuracy (%)", fontsize=10, fontweight="bold")
        if idx >= 3:
            ax.set_xlabel("Severity Level", fontsize=10, fontweight="bold")

    # Global legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.02), fontsize=10)

    fig.suptitle("Predictive Accuracy vs. Corruption Severity Across Precisions (FP32, FP16, INT8)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_05_per_corruption(corr_data: List[Dict], out_path: Path) -> None:
    """Figure 5: Per-corruption accuracy comparison across precisions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=300, sharey=True)

    corruptions = ["brightness", "contrast", "gaussian_blur", "gaussian_noise", "jpeg_compression", "motion_blur"]
    clean_names = ["Brightness", "Contrast", "Gaussian Blur", "Gaussian Noise", "JPEG", "Motion Blur"]

    for ax, model, m_title in [(ax1, "resnet18_cifar", "ResNet-18"), (ax2, "mobilenet_v3_small_cifar", "MobileNetV3-Small")]:
        sub = [r for r in corr_data if r["model"] == model]
        x = np.arange(len(corruptions))
        width = 0.25

        fp32_accs = [r["fp32_acc"] for r in sub]
        fp16_accs = [r["fp16_acc"] for r in sub]
        int8_accs = [r["int8_acc"] for r in sub]

        c_fp32 = "#1f77b4" if "resnet" in model else "#ff7f0e"
        c_fp16 = "#4b9cd3" if "resnet" in model else "#fdae6b"
        c_int8 = "#084081" if "resnet" in model else "#a63603"

        ax.bar(x - width, fp32_accs, width, label="FP32", color=c_fp32, edgecolor="#333", linewidth=0.6)
        ax.bar(x, fp16_accs, width, label="FP16", color=c_fp16, edgecolor="#333", linewidth=0.6)
        rects = ax.bar(x + width, int8_accs, width, label="INT8", color=c_int8, edgecolor="#333", linewidth=0.6)

        # Annotate INT8 vs FP32 delta
        for i, (r, diff) in enumerate(zip(rects, [s["int8_vs_fp32_diff_pp"] for s in sub])):
            sign = "+" if diff >= 0 else ""
            col = "#006600" if diff >= 0 else "#cc0000"
            ax.annotate(f"{sign}{diff:.1f}",
                        xy=(r.get_x() + r.get_width() / 2, r.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7.5, fontweight="bold", color=col)

        ax.set_title(f"{m_title}: Per-Corruption Accuracy Across Precisions", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(clean_names, rotation=25, ha="right", fontsize=9.5, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        ax.set_ylim(30, 95)
        ax.legend(loc="upper right", fontsize=9.5)

    ax1.set_ylabel("Mean Corrupted Accuracy (%)", fontsize=11, fontweight="bold")

    plt.suptitle("Per-Corruption Vulnerability and Cross-Precision Deltas (INT8 - FP32 in pp)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_06_cpu_batch1_latency(cpu_data: List[Dict], out_path: Path) -> None:
    """Figure 6: CPU Batch-1 Latency comparison across thread configurations."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    # Filter out FP16 (which is N/A on CPU)
    models = [("resnet18_cifar", ax1, "ResNet-18"), ("mobilenet_v3_small_cifar", ax2, "MobileNetV3-Small")]
    threads = [1, 2, 4]
    x = np.arange(len(threads))
    width = 0.28

    for m_key, ax, m_title in models:
        eager_fp32 = [r["batch_1_mean_ms"] for r in cpu_data if r["model"] == m_key and r["precision_key"] == "eager_fp32"]
        ts_fp32 = [r["batch_1_mean_ms"] for r in cpu_data if r["model"] == m_key and r["precision_key"] == "torchscript_fp32"]
        ts_int8 = [r["batch_1_mean_ms"] for r in cpu_data if r["model"] == m_key and r["precision_key"] == "torchscript_int8"]

        c_fp32 = "#1f77b4" if "resnet" in m_key else "#ff7f0e"
        c_ts = "#4b9cd3" if "resnet" in m_key else "#fdae6b"
        c_int8 = "#084081" if "resnet" in m_key else "#a63603"

        rects1 = ax.bar(x - width, eager_fp32, width, label="Eager FP32", color=c_fp32, edgecolor="#333", linewidth=0.6)
        rects2 = ax.bar(x, ts_fp32, width, label="TorchScript FP32", color=c_ts, edgecolor="#333", linewidth=0.6)
        rects3 = ax.bar(x + width, ts_int8, width, label="TorchScript INT8", color=c_int8, edgecolor="#333", linewidth=0.6)

        # Label values
        for rects in [rects1, rects2, rects3]:
            for rect in rects:
                h = rect.get_height()
                ax.annotate(f"{h:.2f}",
                            xy=(rect.get_x() + rect.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8, fontweight="bold")

        ax.set_title(f"{m_title}: Batch-1 Latency Across Threads", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(["1 Thread", "2 Threads", "4 Threads"], fontsize=10, fontweight="bold")
        ax.set_ylabel("Inference Latency (ms)", fontsize=11, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        ax.legend(loc="upper right", fontsize=9.5)

    plt.suptitle("CPU Batch-1 Forward Pass Latency by Thread Configuration (Apple Silicon M3)\n*FP16 omitted: FP16 CPU latency not benchmarked (FP16 measured on MPS)",
                 fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_07_cpu_batch64_throughput(cpu_data: List[Dict], out_path: Path) -> None:
    """Figure 7: CPU Batch-64 Throughput comparison across thread configurations."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    models = [("resnet18_cifar", ax1, "ResNet-18"), ("mobilenet_v3_small_cifar", ax2, "MobileNetV3-Small")]
    threads = [1, 2, 4]
    x = np.arange(len(threads))
    width = 0.28

    for m_key, ax, m_title in models:
        eager_fp32 = [r["batch_64_throughput_img_s"] for r in cpu_data if r["model"] == m_key and r["precision_key"] == "eager_fp32"]
        ts_fp32 = [r["batch_64_throughput_img_s"] for r in cpu_data if r["model"] == m_key and r["precision_key"] == "torchscript_fp32"]
        ts_int8 = [r["batch_64_throughput_img_s"] for r in cpu_data if r["model"] == m_key and r["precision_key"] == "torchscript_int8"]

        c_fp32 = "#1f77b4" if "resnet" in m_key else "#ff7f0e"
        c_ts = "#4b9cd3" if "resnet" in m_key else "#fdae6b"
        c_int8 = "#084081" if "resnet" in m_key else "#a63603"

        rects1 = ax.bar(x - width, eager_fp32, width, label="Eager FP32", color=c_fp32, edgecolor="#333", linewidth=0.6)
        rects2 = ax.bar(x, ts_fp32, width, label="TorchScript FP32", color=c_ts, edgecolor="#333", linewidth=0.6)
        rects3 = ax.bar(x + width, ts_int8, width, label="TorchScript INT8", color=c_int8, edgecolor="#333", linewidth=0.6)

        for rects in [rects1, rects2, rects3]:
            for rect in rects:
                h = rect.get_height()
                lbl = f"{h:,.0f}" if h >= 1000 else f"{h:.1f}"
                ax.annotate(lbl,
                            xy=(rect.get_x() + rect.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8, fontweight="bold")

        ax.set_title(f"{m_title}: Batch-64 Throughput Across Threads", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(["1 Thread", "2 Threads", "4 Threads"], fontsize=10, fontweight="bold")
        ax.set_ylabel("Throughput (images / sec)", fontsize=11, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        ax.legend(loc="upper left", fontsize=9.5)

    plt.suptitle("CPU Batch-64 Inference Throughput by Thread Configuration (Apple Silicon M3)\n*FP16 omitted: FP16 CPU latency not benchmarked (FP16 measured on MPS)",
                 fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_08_pareto_frontiers(pareto_data: Dict[str, Any], out_path: Path) -> None:
    """Figure 8: 2x2 Efficiency-Reliability Pareto Trade-off Curves."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 13), dpi=300)
    ((ax1, ax2), (ax3, ax4)) = axes

    # Styling helper
    point_styles = {
        "resnet18_cifar_FP32": {"color": "#1f77b4", "marker": "o", "label": "ResNet-18 FP32"},
        "resnet18_cifar_FP16": {"color": "#4b9cd3", "marker": "^", "label": "ResNet-18 FP16"},
        "resnet18_cifar_INT8": {"color": "#084081", "marker": "s", "label": "ResNet-18 INT8"},
        "mobilenet_v3_small_cifar_FP32": {"color": "#ff7f0e", "marker": "o", "label": "MobileNetV3 FP32"},
        "mobilenet_v3_small_cifar_FP16": {"color": "#fdae6b", "marker": "^", "label": "MobileNetV3 FP16"},
        "mobilenet_v3_small_cifar_INT8": {"color": "#a63603", "marker": "s", "label": "MobileNetV3 INT8"},
    }

    # Differentiated annotation offsets to prevent label collisions
    offsets_a = {
        "mobilenet_v3_small_cifar_INT8": {"xytext": (-10, -16), "ha": "center"},
        "mobilenet_v3_small_cifar_FP16": {"xytext": (0, 9), "ha": "center"},
        "mobilenet_v3_small_cifar_FP32": {"xytext": (10, -5), "ha": "left"},
        "resnet18_cifar_INT8": {"xytext": (-5, 9), "ha": "center"},
        "resnet18_cifar_FP16": {"xytext": (0, 9), "ha": "center"},
        "resnet18_cifar_FP32": {"xytext": (0, 9), "ha": "center"},
    }

    offsets_b = {
        "mobilenet_v3_small_cifar_INT8": {"xytext": (0, 9), "ha": "center"},
        "mobilenet_v3_small_cifar_FP16": {"xytext": (-10, -16), "ha": "center"},
        "mobilenet_v3_small_cifar_FP32": {"xytext": (10, -5), "ha": "left"},
        "resnet18_cifar_INT8": {"xytext": (-5, 9), "ha": "center"},
        "resnet18_cifar_FP16": {"xytext": (0, 9), "ha": "center"},
        "resnet18_cifar_FP32": {"xytext": (0, 9), "ha": "center"},
    }

    offsets_c = {
        "mobilenet_v3_small_cifar_INT8": {"xytext": (10, 4), "ha": "left"},
        "resnet18_cifar_INT8": {"xytext": (-2, 12), "ha": "center"},
        "resnet18_cifar_FP32": {"xytext": (12, -12), "ha": "left"},
        "mobilenet_v3_small_cifar_FP32": {"xytext": (-10, 8), "ha": "right"},
    }

    offsets_d = {
        "mobilenet_v3_small_cifar_INT8": {"xytext": (10, 4), "ha": "left"},
        "resnet18_cifar_INT8": {"xytext": (-2, 12), "ha": "center"},
        "resnet18_cifar_FP32": {"xytext": (12, -12), "ha": "left"},
        "mobilenet_v3_small_cifar_FP32": {"xytext": (-10, 8), "ha": "right"},
    }

    # Subplot A: Size vs Clean Accuracy
    ax = ax1
    p_res = pareto_data["pareto_size_vs_clean"]
    all_pts = p_res["all_items"]
    frontier_pts = p_res["frontier"]

    for pt in all_pts:
        cid = f"{pt['model']}_{pt['precision']}"
        sty = point_styles[cid]
        edge = "green" if pt["is_pareto_efficient"] else "gray"
        lw = 2.5 if pt["is_pareto_efficient"] else 1.0
        ax.scatter(pt["size_mb"], pt["clean_accuracy"], s=160, color=sty["color"],
                   marker=sty["marker"], edgecolors=edge, linewidth=lw, zorder=5)
        pos = offsets_a.get(cid, {"xytext": (6, 4), "ha": "left"})
        ax.annotate(sty["label"], (pt["size_mb"], pt["clean_accuracy"]),
                    xytext=pos["xytext"], textcoords="offset points", fontsize=8.5,
                    fontweight="bold", ha=pos["ha"])

    # Connect frontier
    fx = [p["size_mb"] for p in frontier_pts]
    fy = [p["clean_accuracy"] for p in frontier_pts]
    ax.plot(fx, fy, "--", color="#2ca02c", linewidth=2.0, alpha=0.85, label="Pareto Frontier", zorder=3)
    ax.set_xlabel("Model Checkpoint Size (MB) [Lower is Better]", fontsize=10, fontweight="bold")
    ax.set_ylabel("Clean Test Accuracy (%) [Higher is Better]", fontsize=10, fontweight="bold")
    ax.set_title("(A) Model Size vs. Clean Test Accuracy", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower right", fontsize=9)

    # Subplot B: Size vs Corrupted Accuracy
    ax = ax2
    p_res = pareto_data["pareto_size_vs_corrupted"]
    all_pts = p_res["all_items"]
    frontier_pts = p_res["frontier"]

    for pt in all_pts:
        cid = f"{pt['model']}_{pt['precision']}"
        sty = point_styles[cid]
        edge = "green" if pt["is_pareto_efficient"] else "gray"
        lw = 2.5 if pt["is_pareto_efficient"] else 1.0
        ax.scatter(pt["size_mb"], pt["mean_corrupted_accuracy"], s=160, color=sty["color"],
                   marker=sty["marker"], edgecolors=edge, linewidth=lw, zorder=5)
        pos = offsets_b.get(cid, {"xytext": (6, 4), "ha": "left"})
        ax.annotate(sty["label"], (pt["size_mb"], pt["mean_corrupted_accuracy"]),
                    xytext=pos["xytext"], textcoords="offset points", fontsize=8.5,
                    fontweight="bold", ha=pos["ha"])

    fx = [p["size_mb"] for p in frontier_pts]
    fy = [p["mean_corrupted_accuracy"] for p in frontier_pts]
    ax.plot(fx, fy, "--", color="#2ca02c", linewidth=2.0, alpha=0.85, label="Pareto Frontier", zorder=3)
    ax.set_xlabel("Model Checkpoint Size (MB) [Lower is Better]", fontsize=10, fontweight="bold")
    ax.set_ylabel("Mean Corrupted Accuracy (%) [Higher is Better]", fontsize=10, fontweight="bold")
    ax.set_title("(B) Model Size vs. Mean Corrupted Accuracy", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower right", fontsize=9)

    # Subplot C: CPU Latency (th=4) vs Clean Accuracy
    ax = ax3
    p_res = pareto_data["pareto_cpu_latency_th4_vs_clean"]
    all_pts = p_res["all_items"]
    frontier_pts = p_res["frontier"]

    for pt in all_pts:
        cid = f"{pt['model']}_{pt['precision']}"
        sty = point_styles[cid]
        edge = "green" if pt["is_pareto_efficient"] else "gray"
        lw = 2.5 if pt["is_pareto_efficient"] else 1.0
        ax.scatter(pt["latency_th4_b1_ms"], pt["clean_accuracy"], s=160, color=sty["color"],
                   marker=sty["marker"], edgecolors=edge, linewidth=lw, zorder=5)
        pos = offsets_c.get(cid, {"xytext": (6, 4), "ha": "left"})
        ax.annotate(sty["label"], (pt["latency_th4_b1_ms"], pt["clean_accuracy"]),
                    xytext=pos["xytext"], textcoords="offset points", fontsize=8.5,
                    fontweight="bold", ha=pos["ha"])

    fx = [p["latency_th4_b1_ms"] for p in frontier_pts]
    fy = [p["clean_accuracy"] for p in frontier_pts]
    ax.plot(fx, fy, "--", color="#2ca02c", linewidth=2.0, alpha=0.85, label="Pareto Frontier", zorder=3)
    ax.set_xlabel("CPU Batch-1 Latency at 4 Threads (ms) [Lower is Better]", fontsize=10, fontweight="bold")
    ax.set_ylabel("Clean Test Accuracy (%) [Higher is Better]", fontsize=10, fontweight="bold")
    ax.set_title("(C) CPU Latency (4 Threads) vs. Clean Test Accuracy", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="center right", fontsize=9)

    # Subplot D: CPU Latency (th=4) vs Corrupted Accuracy
    ax = ax4
    p_res = pareto_data["pareto_cpu_latency_th4_vs_corrupted"]
    all_pts = p_res["all_items"]
    frontier_pts = p_res["frontier"]

    for pt in all_pts:
        cid = f"{pt['model']}_{pt['precision']}"
        sty = point_styles[cid]
        edge = "green" if pt["is_pareto_efficient"] else "gray"
        lw = 2.5 if pt["is_pareto_efficient"] else 1.0
        ax.scatter(pt["latency_th4_b1_ms"], pt["mean_corrupted_accuracy"], s=160, color=sty["color"],
                   marker=sty["marker"], edgecolors=edge, linewidth=lw, zorder=5)
        pos = offsets_d.get(cid, {"xytext": (6, 4), "ha": "left"})
        ax.annotate(sty["label"], (pt["latency_th4_b1_ms"], pt["mean_corrupted_accuracy"]),
                    xytext=pos["xytext"], textcoords="offset points", fontsize=8.5,
                    fontweight="bold", ha=pos["ha"])

    fx = [p["latency_th4_b1_ms"] for p in frontier_pts]
    fy = [p["mean_corrupted_accuracy"] for p in frontier_pts]
    ax.plot(fx, fy, "--", color="#2ca02c", linewidth=2.0, alpha=0.85, label="Pareto Frontier", zorder=3)
    ax.set_xlabel("CPU Batch-1 Latency at 4 Threads (ms) [Lower is Better]", fontsize=10, fontweight="bold")
    ax.set_ylabel("Mean Corrupted Accuracy (%) [Higher is Better]", fontsize=10, fontweight="bold")
    ax.set_title("(D) CPU Latency (4 Threads) vs. Mean Corrupted Accuracy", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="center right", fontsize=9)

    plt.suptitle("Cross-Architecture Efficiency–Reliability Pareto Frontiers Across Precisions\n(Green outlines and dashed lines designate non-dominated cross-architecture configurations; CPU benchmarks restricted to CPU FP32 & INT8)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def main() -> None:
    results_dir = Path("reports/experiment_results")
    experiments_dir = Path("experiments")
    out_dir = Path("reports/figures/cross_precision")
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_matplotlib_style()
    data = load_cross_precision_data(results_dir, experiments_dir)

    size_tab = build_model_size_table(data["checkpoint_sizes"])
    clean_tab = build_clean_accuracy_table(data["summaries"], data["clean_fp16_comp"], data["clean_int8_comp"])
    rob_tab = build_robustness_summary_table(data["summaries"])
    corr_tab = build_per_corruption_table(data["summaries"])
    sev_tab = build_per_severity_table(data["summaries"])
    cpu_tab = build_cpu_performance_table(data["cpu_bench"])
    pareto_res = build_pareto_analysis(size_tab, rob_tab, data["cpu_bench"])

    print("Generating Figure 1: Artifact size comparison...")
    plot_01_artifact_size(size_tab, out_dir / "01_artifact_size_comparison.png")

    print("Generating Figure 2: Clean accuracy comparison...")
    plot_02_clean_accuracy(clean_tab, out_dir / "02_clean_accuracy_comparison.png")

    print("Generating Figure 3: Mean corrupted accuracy comparison...")
    plot_03_corrupted_accuracy(rob_tab, out_dir / "03_mean_corrupted_accuracy_comparison.png")

    print("Generating Figure 4: Accuracy vs severity grid...")
    plot_04_accuracy_vs_severity(data["dfs"], out_dir / "04_accuracy_vs_severity.png")

    print("Generating Figure 5: Per-corruption accuracy comparison...")
    plot_05_per_corruption(corr_tab, out_dir / "05_per_corruption_comparison.png")

    print("Generating Figure 6: CPU batch-1 latency comparison...")
    plot_06_cpu_batch1_latency(cpu_tab, out_dir / "06_cpu_batch1_latency.png")

    print("Generating Figure 7: CPU batch-64 throughput comparison...")
    plot_07_cpu_batch64_throughput(cpu_tab, out_dir / "07_cpu_batch64_throughput.png")

    print("Generating Figure 8: Efficiency-reliability Pareto curves...")
    plot_08_pareto_frontiers(pareto_res, out_dir / "08_efficiency_reliability_pareto.png")

    # Save consolidated JSON reports
    summary_export = {
        "metadata": {
            "title": "Cross-Precision Efficiency-Reliability Summary",
            "models": ["resnet18_cifar", "mobilenet_v3_small_cifar"],
            "precisions": ["FP32", "FP16", "INT8"],
            "total_conditions_per_model": 31,
            "test_samples_per_condition": 10000,
        },
        "model_sizes": size_tab,
        "clean_accuracy": clean_tab,
        "robustness_summary": rob_tab,
        "per_corruption_summary": corr_tab,
        "per_severity_summary": sev_tab,
        "cpu_microbenchmark_summary": cpu_tab,
        "pareto_frontiers": pareto_res,
    }

    with open(results_dir / "cross_precision_summary.json", "w") as f:
        json.dump(summary_export, f, indent=2)

    # Save flat metrics CSV for easy tabular consumption
    flat_rows = []
    for cfg in pareto_res["all_configs"]:
        m = cfg["model"]
        p = cfg["precision_key"]
        s_match = [r for r in size_tab if r["model"] == m and r["precision_key"] == p][0]
        c_match = [r for r in clean_tab if r["model"] == m and r["precision_key"] == p][0]
        r_match = [r for r in rob_tab if r["model"] == m and r["precision_key"] == p][0]

        flat_rows.append({
            "model": m,
            "display_name": cfg["display_name"],
            "precision": cfg["precision"],
            "parameters": s_match["parameters"],
            "idealized_param_payload_mb": s_match["idealized_param_payload_mb"],
            "serialized_bytes": s_match["serialized_bytes"],
            "serialized_mb": s_match["serialized_mb"],
            "compression_ratio": s_match["compression_ratio"],
            "clean_accuracy": c_match["clean_accuracy"],
            "clean_loss": c_match["cross_entropy_loss"],
            "accuracy_diff_vs_fp32_pp": c_match["accuracy_diff_pp"],
            "concordance_pct": c_match["concordance_pct"],
            "mean_corrupted_accuracy": r_match["mean_corrupted_accuracy"],
            "mean_accuracy_drop_pp": r_match["mean_accuracy_drop_pp"],
            "mean_relative_accuracy_drop": r_match["mean_relative_accuracy_drop"],
            "cpu_lat_th1_b1_ms": cfg["latency_th1_b1_ms"],
            "cpu_lat_th2_b1_ms": cfg["latency_th2_b1_ms"],
            "cpu_lat_th4_b1_ms": cfg["latency_th4_b1_ms"],
            "cpu_thpt_th1_b64_img_s": cfg["throughput_th1_b64_img_s"],
            "cpu_thpt_th2_b64_img_s": cfg["throughput_th2_b64_img_s"],
            "cpu_thpt_th4_b64_img_s": cfg["throughput_th4_b64_img_s"],
        })

    pd.DataFrame(flat_rows).to_csv(results_dir / "cross_precision_metrics.csv", index=False)
    print(f"\n[+] Successfully generated all 8 figures in {out_dir}")
    print(f"[+] Saved consolidated artifacts to {results_dir / 'cross_precision_summary.json'} and {results_dir / 'cross_precision_metrics.csv'}")


if __name__ == "__main__":
    main()
