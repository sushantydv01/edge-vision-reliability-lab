"""
Cross-precision efficiency and reliability analysis module.
Consolidates and validates results across FP32, FP16, and INT8.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from edge_vision.analysis.pareto import compute_pareto_frontier


# Authoritative model parameter metadata established in Task 2.1 & 5.2B
MODEL_METADATA = {
    "resnet18_cifar": {
        "display_name": "ResNet-18",
        "total_parameters": 11_173_962,
        "trainable_parameters": 11_173_962,
        "buffers": 9_620,
        "state_dict_elements": 11_183_582,
        "theoretical_param_bytes": {
            "fp32": 11_173_962 * 4,  # 44,695,848 bytes
            "fp16": 11_173_962 * 2,  # 22,347,924 bytes
            "int8": 11_173_962 * 1,  # 11,173,962 bytes
        },
    },
    "mobilenet_v3_small_cifar": {
        "display_name": "MobileNetV3-Small",
        "total_parameters": 1_528_106,
        "trainable_parameters": 1_528_106,
        "buffers": 12_146,
        "state_dict_elements": 1_540_252,
        "theoretical_param_bytes": {
            "fp32": 1_528_106 * 4,  # 6,112,424 bytes
            "fp16": 1_528_106 * 2,  # 3,056,212 bytes
            "int8": 1_528_106 * 1,  # 1_528_106 bytes
        },
    },
}


def load_cross_precision_data(
    results_dir: Path,
    experiments_dir: Path,
) -> Dict[str, Any]:
    """
    Loads and validates all experimental source-of-truth artifacts.
    """
    results_dir = Path(results_dir)
    experiments_dir = Path(experiments_dir)

    # 1. Load CSVs
    dfs = {
        "fp32": pd.read_csv(results_dir / "robustness_fp32_results.csv"),
        "fp16": pd.read_csv(results_dir / "robustness_fp16_results.csv"),
        "int8": pd.read_csv(results_dir / "robustness_int8_results.csv"),
    }

    # Validate condition counts
    for prec, df in dfs.items():
        if len(df) != 62:
            raise ValueError(f"Expected 62 rows in {prec} CSV, got {len(df)}")
        for model in ["resnet18_cifar", "mobilenet_v3_small_cifar"]:
            m_df = df[df["model"] == model]
            if len(m_df) != 31:
                raise ValueError(f"Expected 31 conditions for {model} in {prec}, got {len(m_df)}")
            if not (m_df["sample_count"] == 10000).all():
                raise ValueError(f"Condition sample count mismatch in {prec} {model}")

    # 2. Load summaries
    summaries = {}
    for prec in ["fp32", "fp16", "int8"]:
        with open(results_dir / f"robustness_{prec}_summary.json") as f:
            summaries[prec] = json.load(f)

    # 3. Load clean comparison details
    with open(results_dir / "clean_test_fp16_comparison.json") as f:
        clean_fp16_comp = json.load(f)
    with open(results_dir / "clean_test_int8_comparison.json") as f:
        clean_int8_comp = json.load(f)

    # 4. Load CPU benchmark results
    with open(results_dir / "cpu_inference_benchmark.json") as f:
        cpu_bench = json.load(f)

    # 5. Checkpoint sizes on disk
    checkpoint_paths = {
        ("resnet18_cifar", "fp32"): experiments_dir / "checkpoints_inference_fp32" / "resnet18_cifar_fp32.pt",
        ("resnet18_cifar", "fp16"): experiments_dir / "checkpoints_fp16" / "resnet18_cifar_fp16.pt",
        ("resnet18_cifar", "int8"): experiments_dir / "checkpoints_int8" / "resnet18_cifar_int8.pt",
        ("mobilenet_v3_small_cifar", "fp32"): experiments_dir / "checkpoints_inference_fp32" / "mobilenet_v3_small_cifar_fp32.pt",
        ("mobilenet_v3_small_cifar", "fp16"): experiments_dir / "checkpoints_fp16" / "mobilenet_v3_small_cifar_fp16.pt",
        ("mobilenet_v3_small_cifar", "int8"): experiments_dir / "checkpoints_int8" / "mobilenet_v3_small_cifar_int8.pt",
    }

    checkpoint_sizes = {}
    for key, path in checkpoint_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint artifact missing: {path}")
        checkpoint_sizes[key] = path.stat().st_size

    return {
        "dfs": dfs,
        "summaries": summaries,
        "clean_fp16_comp": clean_fp16_comp,
        "clean_int8_comp": clean_int8_comp,
        "cpu_bench": cpu_bench,
        "checkpoint_sizes": checkpoint_sizes,
    }


def build_model_size_table(checkpoint_sizes: Dict[Tuple[str, str], int]) -> List[Dict[str, Any]]:
    """
    Constructs model size and compression table.
    """
    rows = []
    for model in ["resnet18_cifar", "mobilenet_v3_small_cifar"]:
        meta = MODEL_METADATA[model]
        fp32_size = checkpoint_sizes[(model, "fp32")]

        for prec in ["fp32", "fp16", "int8"]:
            sz_bytes = checkpoint_sizes[(model, prec)]
            sz_mb = sz_bytes / 1_000_000
            sz_mib = sz_bytes / (1024 * 1024)
            theo_bytes = meta["theoretical_param_bytes"][prec]
            theo_mb = theo_bytes / 1_000_000
            theo_mib = theo_bytes / (1024 * 1024)

            comp_ratio = fp32_size / sz_bytes
            reduction_pct = (1.0 - sz_bytes / fp32_size) * 100.0

            rows.append({
                "model": model,
                "display_name": meta["display_name"],
                "precision": prec.upper(),
                "precision_key": prec,
                "parameters": meta["total_parameters"],
                "state_dict_elements": meta["state_dict_elements"],
                "idealized_param_payload_bytes": theo_bytes,
                "idealized_param_payload_mb": round(theo_mb, 2),
                "idealized_param_payload_mib": round(theo_mib, 2),
                "theoretical_param_bytes": theo_bytes,
                "theoretical_param_mb": round(theo_mb, 2),
                "theoretical_param_mib": round(theo_mib, 2),
                "serialized_bytes": sz_bytes,
                "serialized_mb": round(sz_mb, 2),
                "serialized_mib": round(sz_mib, 2),
                "compression_ratio": round(comp_ratio, 3),
                "size_reduction_pct": round(reduction_pct, 2),
            })
    return rows


def build_clean_accuracy_table(
    summaries: Dict[str, Any],
    clean_fp16_comp: Dict[str, Any],
    clean_int8_comp: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Constructs clean test predictive reliability table.
    """
    rows = []
    for model in ["resnet18_cifar", "mobilenet_v3_small_cifar"]:
        meta = MODEL_METADATA[model]
        fp32_acc = summaries["fp32"]["model_summaries"][model]["clean_accuracy"] * 100.0
        fp32_loss = clean_fp16_comp[model]["fp32_loss"]
        fp32_correct = clean_fp16_comp[model]["fp32_correct"]

        for prec in ["fp32", "fp16", "int8"]:
            clean_acc = summaries[prec]["model_summaries"][model]["clean_accuracy"] * 100.0
            diff_pp = clean_acc - fp32_acc

            if prec == "fp32":
                loss = fp32_loss
                correct = fp32_correct
                incorrect = 10_000 - correct
                concordance = 100.0
                changed = 0
            elif prec == "fp16":
                loss = clean_fp16_comp[model]["fp16_loss"]
                correct = clean_fp16_comp[model]["fp16_correct"]
                incorrect = 10_000 - correct
                concordance = clean_fp16_comp[model]["concordance_pct"]
                changed = clean_fp16_comp[model]["changed_predictions"]
            else:  # int8
                loss = clean_int8_comp[model]["int8_loss"]
                correct = clean_int8_comp[model]["int8_correct"]
                incorrect = 10_000 - correct
                concordance = clean_int8_comp[model]["concordance_pct"]
                changed = clean_int8_comp[model]["changed_predictions"]

            rows.append({
                "model": model,
                "display_name": meta["display_name"],
                "precision": prec.upper(),
                "precision_key": prec,
                "clean_accuracy": round(clean_acc, 2),
                "cross_entropy_loss": round(loss, 4),
                "correct_samples": correct,
                "incorrect_samples": incorrect,
                "accuracy_diff_pp": round(diff_pp, 2),
                "concordance_pct": round(concordance, 2),
                "changed_predictions": changed,
            })
    return rows


def build_robustness_summary_table(summaries: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Constructs overall corruption robustness table.
    """
    rows = []
    for model in ["resnet18_cifar", "mobilenet_v3_small_cifar"]:
        meta = MODEL_METADATA[model]
        fp32_mean_corr = summaries["fp32"]["model_summaries"][model]["mean_corrupted_accuracy"] * 100.0

        for prec in ["fp32", "fp16", "int8"]:
            m_sum = summaries[prec]["model_summaries"][model]
            clean_acc = m_sum["clean_accuracy"] * 100.0
            mean_corr_acc = m_sum["mean_corrupted_accuracy"] * 100.0
            mean_drop_pp = m_sum["mean_accuracy_drop_pp"]
            rel_drop = m_sum.get("mean_relative_accuracy_drop")
            if rel_drop is None:
                rel_drop = m_sum.get("relative_mCE")

            corr_diff_vs_fp32 = mean_corr_acc - fp32_mean_corr

            rows.append({
                "model": model,
                "display_name": meta["display_name"],
                "precision": prec.upper(),
                "precision_key": prec,
                "clean_accuracy": round(clean_acc, 2),
                "mean_corrupted_accuracy": round(mean_corr_acc, 2),
                "mean_accuracy_drop_pp": round(mean_drop_pp, 3),
                "mean_relative_accuracy_drop": round(rel_drop, 4),
                "corrupted_accuracy_diff_vs_fp32_pp": round(corr_diff_vs_fp32, 2),
            })
    return rows


def build_per_corruption_table(summaries: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Constructs per-corruption accuracy comparison table across precisions.
    """
    corruptions = ["brightness", "contrast", "gaussian_blur", "gaussian_noise", "jpeg_compression", "motion_blur"]
    rows = []

    for model in ["resnet18_cifar", "mobilenet_v3_small_cifar"]:
        meta = MODEL_METADATA[model]
        for corr in corruptions:
            row = {
                "model": model,
                "display_name": meta["display_name"],
                "corruption": corr,
            }
            for prec in ["fp32", "fp16", "int8"]:
                c_data = summaries[prec]["model_summaries"][model]["per_corruption_summary"][corr]
                row[f"{prec}_acc"] = round(c_data["mean_accuracy"] * 100.0, 2)
                row[f"{prec}_drop_pp"] = round(c_data["mean_accuracy_drop_pp"], 2)

            row["int8_vs_fp32_diff_pp"] = round(row["int8_acc"] - row["fp32_acc"], 2)
            row["int8_vs_fp16_diff_pp"] = round(row["int8_acc"] - row["fp16_acc"], 2)
            rows.append(row)
    return rows


def build_per_severity_table(summaries: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Constructs per-severity accuracy comparison table across precisions.
    """
    rows = []
    for model in ["resnet18_cifar", "mobilenet_v3_small_cifar"]:
        meta = MODEL_METADATA[model]
        for sev in range(1, 6):
            s_str = str(sev)
            row = {
                "model": model,
                "display_name": meta["display_name"],
                "severity": sev,
            }
            for prec in ["fp32", "fp16", "int8"]:
                s_data = summaries[prec]["model_summaries"][model]["per_severity_summary"][s_str]
                row[f"{prec}_acc"] = round(s_data["mean_accuracy"] * 100.0, 2)
                row[f"{prec}_drop_pp"] = round(s_data["mean_accuracy_drop_pp"], 2)

            row["int8_vs_fp32_diff_pp"] = round(row["int8_acc"] - row["fp32_acc"], 2)
            row["int8_vs_fp16_diff_pp"] = round(row["int8_acc"] - row["fp16_acc"], 2)
            rows.append(row)
    return rows


def build_cpu_performance_table(cpu_bench: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extracts Task 5.3C isolated-process microbenchmark results.
    """
    rows = []
    threads_list = [1, 2, 4]

    for th in threads_list:
        th_key = f"th{th}"
        th_cfg = cpu_bench["configurations"][th_key]

        for model in ["resnet18_cifar", "mobilenet_v3_small_cifar"]:
            meta = MODEL_METADATA[model]

            # FP32 Eager
            eager_fp32 = th_cfg["models"][model]["eager_fp32"]
            rows.append({
                "model": model,
                "display_name": meta["display_name"],
                "precision": "FP32 (Eager)",
                "precision_key": "eager_fp32",
                "threads": th,
                "batch_1_mean_ms": round(eager_fp32["batch_1"]["mean_latency_ms"], 3),
                "batch_1_median_ms": round(eager_fp32["batch_1"]["median_latency_ms"], 3),
                "batch_1_p95_ms": round(eager_fp32["batch_1"]["p95_latency_ms"], 3),
                "batch_64_mean_ms": round(eager_fp32["batch_64"]["mean_latency_ms"], 3),
                "batch_64_throughput_img_s": round(eager_fp32["batch_64"]["throughput_samples_per_sec"], 1),
            })

            # FP32 TorchScript
            ts_fp32 = th_cfg["models"][model]["torchscript_fp32"]
            rows.append({
                "model": model,
                "display_name": meta["display_name"],
                "precision": "FP32 (TorchScript)",
                "precision_key": "torchscript_fp32",
                "threads": th,
                "batch_1_mean_ms": round(ts_fp32["batch_1"]["mean_latency_ms"], 3),
                "batch_1_median_ms": round(ts_fp32["batch_1"]["median_latency_ms"], 3),
                "batch_1_p95_ms": round(ts_fp32["batch_1"]["p95_latency_ms"], 3),
                "batch_64_mean_ms": round(ts_fp32["batch_64"]["mean_latency_ms"], 3),
                "batch_64_throughput_img_s": round(ts_fp32["batch_64"]["throughput_samples_per_sec"], 1),
            })

            # FP16 (Not available on CPU)
            rows.append({
                "model": model,
                "display_name": meta["display_name"],
                "precision": "FP16",
                "precision_key": "fp16",
                "threads": th,
                "batch_1_mean_ms": None,
                "batch_1_median_ms": None,
                "batch_1_p95_ms": None,
                "batch_64_mean_ms": None,
                "batch_64_throughput_img_s": None,
                "notes": "N/A (FP16 CPU latency not benchmarked; FP16 measured on MPS)",
            })

            # INT8 TorchScript
            ts_int8 = th_cfg["models"][model]["torchscript_int8"]
            rows.append({
                "model": model,
                "display_name": meta["display_name"],
                "precision": "INT8 (TorchScript)",
                "precision_key": "torchscript_int8",
                "threads": th,
                "batch_1_mean_ms": round(ts_int8["batch_1"]["mean_latency_ms"], 3),
                "batch_1_median_ms": round(ts_int8["batch_1"]["median_latency_ms"], 3),
                "batch_1_p95_ms": round(ts_int8["batch_1"]["p95_latency_ms"], 3),
                "batch_64_mean_ms": round(ts_int8["batch_64"]["mean_latency_ms"], 3),
                "batch_64_throughput_img_s": round(ts_int8["batch_64"]["throughput_samples_per_sec"], 1),
            })

    return rows


def build_pareto_analysis(
    size_table: List[Dict[str, Any]],
    robustness_table: List[Dict[str, Any]],
    cpu_bench: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Computes Pareto frontiers across key trade-off dimensions.
    """
    # Build combined configuration list
    configs = []
    size_map = {(r["model"], r["precision_key"]): r["serialized_mb"] for r in size_table}
    clean_map = {(r["model"], r["precision_key"]): r["clean_accuracy"] for r in robustness_table}
    corr_map = {(r["model"], r["precision_key"]): r["mean_corrupted_accuracy"] for r in robustness_table}
    drop_map = {(r["model"], r["precision_key"]): r["mean_accuracy_drop_pp"] for r in robustness_table}

    th4_cfg = cpu_bench["configurations"]["th4"]["models"]
    th2_cfg = cpu_bench["configurations"]["th2"]["models"]
    th1_cfg = cpu_bench["configurations"]["th1"]["models"]

    for model in ["resnet18_cifar", "mobilenet_v3_small_cifar"]:
        for prec in ["fp32", "fp16", "int8"]:
            size_mb = size_map[(model, prec)]
            clean_acc = clean_map[(model, prec)]
            corr_acc = corr_map[(model, prec)]
            drop_pp = drop_map[(model, prec)]

            if prec == "fp16":
                lat_th4_b1 = None
                lat_th2_b1 = None
                lat_th1_b1 = None
                thpt_th4_b64 = None
                thpt_th2_b64 = None
                thpt_th1_b64 = None
            elif prec == "fp32":
                lat_th4_b1 = th4_cfg[model]["torchscript_fp32"]["batch_1"]["mean_latency_ms"]
                lat_th2_b1 = th2_cfg[model]["torchscript_fp32"]["batch_1"]["mean_latency_ms"]
                lat_th1_b1 = th1_cfg[model]["torchscript_fp32"]["batch_1"]["mean_latency_ms"]
                thpt_th4_b64 = th4_cfg[model]["torchscript_fp32"]["batch_64"]["throughput_samples_per_sec"]
                thpt_th2_b64 = th2_cfg[model]["torchscript_fp32"]["batch_64"]["throughput_samples_per_sec"]
                thpt_th1_b64 = th1_cfg[model]["torchscript_fp32"]["batch_64"]["throughput_samples_per_sec"]
            else:  # int8
                lat_th4_b1 = th4_cfg[model]["torchscript_int8"]["batch_1"]["mean_latency_ms"]
                lat_th2_b1 = th2_cfg[model]["torchscript_int8"]["batch_1"]["mean_latency_ms"]
                lat_th1_b1 = th1_cfg[model]["torchscript_int8"]["batch_1"]["mean_latency_ms"]
                thpt_th4_b64 = th4_cfg[model]["torchscript_int8"]["batch_64"]["throughput_samples_per_sec"]
                thpt_th2_b64 = th2_cfg[model]["torchscript_int8"]["batch_64"]["throughput_samples_per_sec"]
                thpt_th1_b64 = th1_cfg[model]["torchscript_int8"]["batch_64"]["throughput_samples_per_sec"]

            configs.append({
                "config_id": f"{model}_{prec}",
                "model": model,
                "display_name": MODEL_METADATA[model]["display_name"],
                "precision": prec.upper(),
                "precision_key": prec,
                "size_mb": size_mb,
                "clean_accuracy": clean_acc,
                "mean_corrupted_accuracy": corr_acc,
                "mean_accuracy_drop_pp": drop_pp,
                "latency_th4_b1_ms": lat_th4_b1,
                "latency_th2_b1_ms": lat_th2_b1,
                "latency_th1_b1_ms": lat_th1_b1,
                "throughput_th4_b64_img_s": thpt_th4_b64,
                "throughput_th2_b64_img_s": thpt_th2_b64,
                "throughput_th1_b64_img_s": thpt_th1_b64,
            })

    # Trade-off 1: Size (min) vs Clean Accuracy (max)
    size_clean_res = compute_pareto_frontier(
        configs,
        [("size_mb", "min"), ("clean_accuracy", "max")],
    )

    # Trade-off 2: Size (min) vs Mean Corrupted Accuracy (max)
    size_corr_res = compute_pareto_frontier(
        configs,
        [("size_mb", "min"), ("mean_corrupted_accuracy", "max")],
    )

    # Filter for CPU-evaluated configs (FP32 & INT8)
    cpu_configs = [c for c in configs if c["latency_th4_b1_ms"] is not None]

    # Trade-off 3: CPU Latency th=4 (min) vs Clean Accuracy (max)
    lat4_clean_res = compute_pareto_frontier(
        cpu_configs,
        [("latency_th4_b1_ms", "min"), ("clean_accuracy", "max")],
    )

    # Trade-off 4: CPU Latency th=4 (min) vs Mean Corrupted Accuracy (max)
    lat4_corr_res = compute_pareto_frontier(
        cpu_configs,
        [("latency_th4_b1_ms", "min"), ("mean_corrupted_accuracy", "max")],
    )

    # Trade-off 5: CPU Latency th=2 (min) vs Clean Accuracy (max)
    lat2_clean_res = compute_pareto_frontier(
        cpu_configs,
        [("latency_th2_b1_ms", "min"), ("clean_accuracy", "max")],
    )

    # Trade-off 6: CPU Latency th=2 (min) vs Mean Corrupted Accuracy (max)
    lat2_corr_res = compute_pareto_frontier(
        cpu_configs,
        [("latency_th2_b1_ms", "min"), ("mean_corrupted_accuracy", "max")],
    )

    # Trade-off 7: CPU Latency th=1 (min) vs Clean Accuracy (max)
    lat1_clean_res = compute_pareto_frontier(
        cpu_configs,
        [("latency_th1_b1_ms", "min"), ("clean_accuracy", "max")],
    )

    # Trade-off 8: CPU Latency th=1 (min) vs Mean Corrupted Accuracy (max)
    lat1_corr_res = compute_pareto_frontier(
        cpu_configs,
        [("latency_th1_b1_ms", "min"), ("mean_corrupted_accuracy", "max")],
    )

    # Within-architecture Pareto calculations
    res_configs = [c for c in configs if c["model"] == "resnet18_cifar"]
    mob_configs = [c for c in configs if c["model"] == "mobilenet_v3_small_cifar"]
    res_cpu_configs = [c for c in cpu_configs if c["model"] == "resnet18_cifar"]
    mob_cpu_configs = [c for c in cpu_configs if c["model"] == "mobilenet_v3_small_cifar"]

    within_arch = {
        "resnet18_cifar": {
            "pareto_size_vs_clean": compute_pareto_frontier(res_configs, [("size_mb", "min"), ("clean_accuracy", "max")]),
            "pareto_size_vs_corrupted": compute_pareto_frontier(res_configs, [("size_mb", "min"), ("mean_corrupted_accuracy", "max")]),
            "pareto_cpu_latency_th4_vs_clean": compute_pareto_frontier(res_cpu_configs, [("latency_th4_b1_ms", "min"), ("clean_accuracy", "max")]),
            "pareto_cpu_latency_th4_vs_corrupted": compute_pareto_frontier(res_cpu_configs, [("latency_th4_b1_ms", "min"), ("mean_corrupted_accuracy", "max")]),
            "pareto_cpu_latency_th2_vs_clean": compute_pareto_frontier(res_cpu_configs, [("latency_th2_b1_ms", "min"), ("clean_accuracy", "max")]),
            "pareto_cpu_latency_th2_vs_corrupted": compute_pareto_frontier(res_cpu_configs, [("latency_th2_b1_ms", "min"), ("mean_corrupted_accuracy", "max")]),
            "pareto_cpu_latency_th1_vs_clean": compute_pareto_frontier(res_cpu_configs, [("latency_th1_b1_ms", "min"), ("clean_accuracy", "max")]),
            "pareto_cpu_latency_th1_vs_corrupted": compute_pareto_frontier(res_cpu_configs, [("latency_th1_b1_ms", "min"), ("mean_corrupted_accuracy", "max")]),
        },
        "mobilenet_v3_small_cifar": {
            "pareto_size_vs_clean": compute_pareto_frontier(mob_configs, [("size_mb", "min"), ("clean_accuracy", "max")]),
            "pareto_size_vs_corrupted": compute_pareto_frontier(mob_configs, [("size_mb", "min"), ("mean_corrupted_accuracy", "max")]),
            "pareto_cpu_latency_th4_vs_clean": compute_pareto_frontier(mob_cpu_configs, [("latency_th4_b1_ms", "min"), ("clean_accuracy", "max")]),
            "pareto_cpu_latency_th4_vs_corrupted": compute_pareto_frontier(mob_cpu_configs, [("latency_th4_b1_ms", "min"), ("mean_corrupted_accuracy", "max")]),
            "pareto_cpu_latency_th2_vs_clean": compute_pareto_frontier(mob_cpu_configs, [("latency_th2_b1_ms", "min"), ("clean_accuracy", "max")]),
            "pareto_cpu_latency_th2_vs_corrupted": compute_pareto_frontier(mob_cpu_configs, [("latency_th2_b1_ms", "min"), ("mean_corrupted_accuracy", "max")]),
            "pareto_cpu_latency_th1_vs_clean": compute_pareto_frontier(mob_cpu_configs, [("latency_th1_b1_ms", "min"), ("clean_accuracy", "max")]),
            "pareto_cpu_latency_th1_vs_corrupted": compute_pareto_frontier(mob_cpu_configs, [("latency_th1_b1_ms", "min"), ("mean_corrupted_accuracy", "max")]),
        },
    }

    return {
        "all_configs": configs,
        "cross_architecture": {
            "pareto_size_vs_clean": size_clean_res,
            "pareto_size_vs_corrupted": size_corr_res,
            "pareto_cpu_latency_th4_vs_clean": lat4_clean_res,
            "pareto_cpu_latency_th4_vs_corrupted": lat4_corr_res,
            "pareto_cpu_latency_th2_vs_clean": lat2_clean_res,
            "pareto_cpu_latency_th2_vs_corrupted": lat2_corr_res,
            "pareto_cpu_latency_th1_vs_clean": lat1_clean_res,
            "pareto_cpu_latency_th1_vs_corrupted": lat1_corr_res,
        },
        "within_architecture": within_arch,
        "pareto_size_vs_clean": size_clean_res,
        "pareto_size_vs_corrupted": size_corr_res,
        "pareto_cpu_latency_th4_vs_clean": lat4_clean_res,
        "pareto_cpu_latency_th4_vs_corrupted": lat4_corr_res,
        "pareto_cpu_latency_th2_vs_clean": lat2_clean_res,
        "pareto_cpu_latency_th2_vs_corrupted": lat2_corr_res,
        "pareto_cpu_latency_th1_vs_clean": lat1_clean_res,
        "pareto_cpu_latency_th1_vs_corrupted": lat1_corr_res,
    }
