#!/usr/bin/env python3
"""
Generates the typed JSON research dataset for the frontend overview dashboard
directly from canonical experiment artifacts.
"""

import json
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd


def generate_research_data() -> None:
    project_root = Path(__file__).resolve().parent.parent
    summary_path = project_root / "reports/experiment_results/cross_precision_summary.json"
    benchmark_path = project_root / "experiments/inference_benchmarks/run_m4_empirical/benchmark_summary.csv"
    output_path = project_root / "frontend/src/data/researchData.json"

    with open(summary_path) as f:
        summary = json.load(f)

    df_bench = pd.read_csv(benchmark_path)

    # 1. Project Metadata
    meta = {
        "title": "Edge Vision Reliability Lab",
        "objective": (
            "How much can we compress a computer vision model before it becomes unreliable "
            "under realistic image degradation, and what trade-offs exist between model size, "
            "inference latency, resource consumption, and predictive reliability?"
        ),
        "dataset": "CIFAR-10 (32×32)",
        "hardware": "Apple M3 (8-core CPU: 4P + 4E, 8 GB Unified Memory)",
        "os": "macOS 27.0 (Darwin arm64)",
        "software": "Python 3.12.16, PyTorch 2.15.0, Torchvision 0.29.2",
        "architectures_count": 2,
        "precisions_count": 3,
        "total_configurations": 6,
        "corruptions_count": 6,
        "severity_levels_count": 5,
        "conditions_per_model": 31,  # 1 clean + 30 corrupted
        "samples_per_condition": 10000,
        "samples_per_model": 310000,
        "forward_passes_per_tier": 620000,
        "total_project_predictions": 1860000,
        "evaluation_label": "1,860,000 model-image prediction evaluations",
        "evaluation_description": "1,860,000 model-image prediction evaluations (6 model configurations × 31 conditions × 10,000 test images)",
        "seed": 42,
    }

    # 2. Baseline Models Comparison (ResNet-18 vs MobileNetV3-Small)
    # Extract clean accuracy and corrupted accuracy for FP32
    resnet_fp32_clean = next(c for c in summary["clean_accuracy"] if c["model"] == "resnet18_cifar" and c["precision_key"] == "fp32")
    mobilenet_fp32_clean = next(c for c in summary["clean_accuracy"] if c["model"] == "mobilenet_v3_small_cifar" and c["precision_key"] == "fp32")
    
    resnet_fp32_size = next(s for s in summary["model_sizes"] if s["model"] == "resnet18_cifar" and s["precision_key"] == "fp32")
    mobilenet_fp32_size = next(s for s in summary["model_sizes"] if s["model"] == "mobilenet_v3_small_cifar" and s["precision_key"] == "fp32")

    resnet_fp32_rob = next(r for r in summary["robustness_summary"] if r["model"] == "resnet18_cifar" and r["precision_key"] == "fp32")
    mobilenet_fp32_rob = next(r for r in summary["robustness_summary"] if r["model"] == "mobilenet_v3_small_cifar" and r["precision_key"] == "fp32")

    baseline_comparison = {
        "resnet18": {
            "model_id": "resnet18_cifar",
            "name": "ResNet-18",
            "description": "Dense residual baseline with standard residual blocks and skip connections",
            "parameters": resnet_fp32_size["parameters"],
            "state_dict_elements": resnet_fp32_size["state_dict_elements"],
            "fp32_serialized_bytes": resnet_fp32_size["serialized_bytes"],
            "fp32_serialized_mb": resnet_fp32_size["serialized_mb"],
            "clean_accuracy": resnet_fp32_clean["clean_accuracy"],
            "cross_entropy_loss": resnet_fp32_clean["cross_entropy_loss"],
            "mean_corrupted_accuracy": resnet_fp32_rob["mean_corrupted_accuracy"],
            "mean_accuracy_drop_pp": resnet_fp32_rob["mean_accuracy_drop_pp"],
            "mean_relative_accuracy_drop": resnet_fp32_rob["mean_relative_accuracy_drop"],
        },
        "mobilenet_v3_small": {
            "model_id": "mobilenet_v3_small_cifar",
            "name": "MobileNetV3-Small",
            "description": "Compact mobile architecture with depthwise separable convolutions and squeeze-and-excitation blocks",
            "parameters": mobilenet_fp32_size["parameters"],
            "state_dict_elements": mobilenet_fp32_size["state_dict_elements"],
            "fp32_serialized_bytes": mobilenet_fp32_size["serialized_bytes"],
            "fp32_serialized_mb": mobilenet_fp32_size["serialized_mb"],
            "clean_accuracy": mobilenet_fp32_clean["clean_accuracy"],
            "cross_entropy_loss": mobilenet_fp32_clean["cross_entropy_loss"],
            "mean_corrupted_accuracy": mobilenet_fp32_rob["mean_corrupted_accuracy"],
            "mean_accuracy_drop_pp": mobilenet_fp32_rob["mean_accuracy_drop_pp"],
            "mean_relative_accuracy_drop": mobilenet_fp32_rob["mean_relative_accuracy_drop"],
        },
        "comparison_ratios": {
            "parameter_ratio": round(resnet_fp32_size["parameters"] / mobilenet_fp32_size["parameters"], 2),  # 7.31x
            "size_ratio": round(resnet_fp32_size["serialized_bytes"] / mobilenet_fp32_size["serialized_bytes"], 2),  # 7.16x
            "clean_accuracy_diff_pp": round(resnet_fp32_clean["clean_accuracy"] - mobilenet_fp32_clean["clean_accuracy"], 2),  # +8.27 pp
            "corrupted_accuracy_diff_pp": round(resnet_fp32_rob["mean_corrupted_accuracy"] - mobilenet_fp32_rob["mean_corrupted_accuracy"], 2),  # +6.90 pp
        }
    }

    # 3. Precision Variants Matrix (6 configurations)
    precision_variants = []
    for model_id in ["resnet18_cifar", "mobilenet_v3_small_cifar"]:
        for prec_key in ["fp32", "fp16", "int8"]:
            size_entry = next(s for s in summary["model_sizes"] if s["model"] == model_id and s["precision_key"] == prec_key)
            clean_entry = next(c for c in summary["clean_accuracy"] if c["model"] == model_id and c["precision_key"] == prec_key)
            rob_entry = next(r for r in summary["robustness_summary"] if r["model"] == model_id and r["precision_key"] == prec_key)

            runtime = "PyTorch Eager" if prec_key in ["fp32", "fp16"] else "TorchScript (QNNPACK)"
            eval_device = "Apple Silicon MPS" if prec_key == "fp16" else ("CPU strictly" if prec_key == "int8" else "CPU, MPS")

            precision_variants.append({
                "config_id": f"{model_id}_{prec_key}",
                "model_id": model_id,
                "display_name": size_entry["display_name"],
                "precision": size_entry["precision"],
                "precision_key": prec_key,
                "parameters": size_entry["parameters"],
                "theoretical_param_bytes": size_entry["theoretical_param_bytes"],
                "serialized_bytes": size_entry["serialized_bytes"],
                "serialized_mb": size_entry["serialized_mb"],
                "clean_accuracy": clean_entry["clean_accuracy"],
                "cross_entropy_loss": clean_entry["cross_entropy_loss"],
                "concordance_pct": clean_entry["concordance_pct"],
                "changed_predictions": clean_entry["changed_predictions"],
                "mean_corrupted_accuracy": rob_entry["mean_corrupted_accuracy"],
                "mean_accuracy_drop_pp": rob_entry["mean_accuracy_drop_pp"],
                "compression_ratio": size_entry["compression_ratio"],
                "size_reduction_pct": size_entry["size_reduction_pct"],
                "runtime_backend": runtime,
                "evaluation_device": eval_device,
            })

    # 4. Severity Curves (Clean + Severity 1-5)
    # Severity 0 is clean accuracy
    severity_curves = [
        {
            "severity": 0,
            "label": "Clean",
            "resnet_fp32": 94.78,
            "resnet_fp16": 94.78,
            "resnet_int8": 94.81,
            "mobilenet_fp32": 86.51,
            "mobilenet_fp16": 86.49,
            "mobilenet_int8": 86.18,
        }
    ]

    for sev in range(1, 6):
        r_sev = next(s for s in summary["per_severity_summary"] if s["model"] == "resnet18_cifar" and s["severity"] == sev)
        m_sev = next(s for s in summary["per_severity_summary"] if s["model"] == "mobilenet_v3_small_cifar" and s["severity"] == sev)
        severity_curves.append({
            "severity": sev,
            "label": f"Sev {sev}",
            "resnet_fp32": r_sev["fp32_acc"],
            "resnet_fp16": r_sev["fp16_acc"],
            "resnet_int8": r_sev["int8_acc"],
            "mobilenet_fp32": m_sev["fp32_acc"],
            "mobilenet_fp16": m_sev["fp16_acc"],
            "mobilenet_int8": m_sev["int8_acc"],
        })

    # 5. Per-Corruption Data
    corruption_names = {
        "brightness": ("Photometric", "Brightness (Darkening)"),
        "contrast": ("Photometric", "Contrast Attenuation"),
        "gaussian_blur": ("Blur", "Gaussian Blur"),
        "motion_blur": ("Blur", "Horizontal Motion Blur"),
        "gaussian_noise": ("Noise", "Gaussian Noise"),
        "jpeg_compression": ("Digital", "JPEG Compression"),
    }

    per_corruption_data = []
    for corr_key, (category, name) in corruption_names.items():
        r_corr = next(c for c in summary["per_corruption_summary"] if c["model"] == "resnet18_cifar" and c["corruption"] == corr_key)
        m_corr = next(c for c in summary["per_corruption_summary"] if c["model"] == "mobilenet_v3_small_cifar" and c["corruption"] == corr_key)

        per_corruption_data.append({
            "key": corr_key,
            "name": name,
            "category": category,
            "resnet_fp32": r_corr["fp32_acc"],
            "resnet_fp16": r_corr["fp16_acc"],
            "resnet_int8": r_corr["int8_acc"],
            "mobilenet_fp32": m_corr["fp32_acc"],
            "mobilenet_fp16": m_corr["fp16_acc"],
            "mobilenet_int8": m_corr["int8_acc"],
            "resnet_drop_pp": r_corr["fp32_drop_pp"],
            "mobilenet_drop_pp": m_corr["fp32_drop_pp"],
            "resnet_advantage_pp": round(r_corr["fp32_acc"] - m_corr["fp32_acc"], 2),
            "is_crossover": m_corr["fp32_acc"] > r_corr["fp32_acc"],
        })

    # 6. Efficiency Highlights from benchmark_summary.csv
    # MobileNet INT8 CPU b64
    m_int8_cpu_b64 = df_bench[
        (df_bench["model"] == "mobilenet_v3_small_cifar") &
        (df_bench["precision"] == "int8") &
        (df_bench["device"] == "cpu") &
        (df_bench["batch_size"] == 64)
    ].iloc[0]

    # MobileNet FP16 MPS b64
    m_fp16_mps_b64 = df_bench[
        (df_bench["model"] == "mobilenet_v3_small_cifar") &
        (df_bench["precision"] == "fp16") &
        (df_bench["device"] == "mps") &
        (df_bench["batch_size"] == 64)
    ].iloc[0]

    # MobileNet INT8 CPU b1
    m_int8_cpu_b1 = df_bench[
        (df_bench["model"] == "mobilenet_v3_small_cifar") &
        (df_bench["precision"] == "int8") &
        (df_bench["device"] == "cpu") &
        (df_bench["batch_size"] == 1)
    ].iloc[0]

    # MobileNet FP32 CPU b1
    m_fp32_cpu_b1 = df_bench[
        (df_bench["model"] == "mobilenet_v3_small_cifar") &
        (df_bench["precision"] == "fp32") &
        (df_bench["device"] == "cpu") &
        (df_bench["batch_size"] == 1)
    ].iloc[0]

    # ResNet INT8 CPU b1
    r_int8_cpu_b1 = df_bench[
        (df_bench["model"] == "resnet18_cifar") &
        (df_bench["precision"] == "int8") &
        (df_bench["device"] == "cpu") &
        (df_bench["batch_size"] == 1)
    ].iloc[0]

    # ResNet FP32 CPU b1
    r_fp32_cpu_b1 = df_bench[
        (df_bench["model"] == "resnet18_cifar") &
        (df_bench["precision"] == "fp32") &
        (df_bench["device"] == "cpu") &
        (df_bench["batch_size"] == 1)
    ].iloc[0]

    # ResNet FP16 MPS b64
    r_fp16_mps_b64 = df_bench[
        (df_bench["model"] == "resnet18_cifar") &
        (df_bench["precision"] == "fp16") &
        (df_bench["device"] == "mps") &
        (df_bench["batch_size"] == 64)
    ].iloc[0]

    efficiency_highlights = {
        "mobilenet_int8_cpu_b64": {
            "label": "Peak CPU Edge Throughput",
            "model": "MobileNetV3-Small",
            "precision": "INT8",
            "device": "CPU (4 threads)",
            "backend": "TorchScript (QNNPACK)",
            "batch_size": 64,
            "total_throughput_fps": float(m_int8_cpu_b64["throughput_fps"]),  # 7754.8
            "forward_throughput_fps": float(m_int8_cpu_b64["forward_only_throughput_fps"]),  # 11531.5
            "median_total_ms": float(m_int8_cpu_b64["median_total_ms"]),  # 8.253
            "median_forward_ms": float(m_int8_cpu_b64["median_forward_ms"]),  # 5.550
        },
        "mobilenet_fp16_mps_b64": {
            "label": "Peak Apple Silicon MPS Throughput",
            "model": "MobileNetV3-Small",
            "precision": "FP16",
            "device": "MPS (Metal)",
            "backend": "PyTorch Eager",
            "batch_size": 64,
            "total_throughput_fps": float(m_fp16_mps_b64["throughput_fps"]),  # 5986.3
            "forward_throughput_fps": float(m_fp16_mps_b64["forward_only_throughput_fps"]),  # 8408.9
            "median_total_ms": float(m_fp16_mps_b64["median_total_ms"]),  # 10.691
            "median_forward_ms": float(m_fp16_mps_b64["median_forward_ms"]),  # 7.611
        },
        "mobilenet_cpu_b1_speedup": {
            "label": "Single-Image CPU Quantization Speedup",
            "model": "MobileNetV3-Small",
            "device": "CPU (Apple M3: 4 threads default vs 1 thread isolated)",
            "comparison_type": "Deployment-path comparison (FP32 Eager vs INT8 TorchScript/QNNPACK)",
            "batch_size": 1,
            "fp32_median_total_ms": float(m_fp32_cpu_b1["median_total_ms"]),  # 25.821 ms
            "int8_median_total_ms": float(m_int8_cpu_b1["median_total_ms"]),  # 0.575 ms
            "speedup_factor": round(float(m_fp32_cpu_b1["median_total_ms"]) / float(m_int8_cpu_b1["median_total_ms"]), 1),  # 44.9x
            "fp32_single_thread_ms": 3.78,
            "int8_single_thread_ms": 0.41,
            "single_thread_speedup_factor": 9.1,
            "scope_note": "Specific to this Apple M3 CPU benchmark; deployment-path comparison, not pure precision-only scaling.",
        },
        "resnet_fp16_mps_b64": {
            "label": "ResNet MPS Batch Throughput",
            "model": "ResNet-18",
            "precision": "FP16",
            "device": "MPS (Metal)",
            "backend": "PyTorch Eager",
            "batch_size": 64,
            "total_throughput_fps": float(r_fp16_mps_b64["throughput_fps"]),  # 2128.8
            "median_total_ms": float(r_fp16_mps_b64["median_total_ms"]),  # 30.064
        },
    }

    # 7. Key Research Findings (Empirical Observational Phrasing)
    key_findings = [
        {
            "id": 1,
            "title": "In Evaluated Conditions, FP16 Maintained Predictive Accuracy with ~50% Storage Reduction",
            "badge": "Precision Invariant",
            "summary": (
                "In the evaluated benchmark conditions, half-precision conversion achieved a 2.00× serialized reduction for ResNet-18 (44.8 MB → 22.4 MB) "
                "and 1.97× for MobileNetV3-Small (6.2 MB → 3.2 MB) without substantial accuracy loss. "
                "Measured ResNet-18 clean accuracy was unchanged at 94.78% (99.99% concordance; 1 prediction change out of 10,000), "
                "while MobileNetV3-Small changed by 0.02 pp (86.51% → 86.49%, 99.94% concordance)."
            ),
            "evidence": "Clean accuracy concordance: ResNet 99.99%, MobileNet 99.94%.",
        },
        {
            "id": 2,
            "title": "Measurements Show Architecture-Dependent INT8 Quantization Trade-offs",
            "badge": "PTQ Asymmetry",
            "summary": (
                "In the evaluated conditions, post-training static quantization (INT8) exhibited asymmetric effects across architectures. "
                "Dense ResNet-18 measured a slight clean accuracy delta of +0.03 pp (94.78% → 94.81%, 99.64% concordance), whereas "
                "inverted-residual MobileNetV3-Small measured a -0.33 pp delta (86.51% → 86.18%, 95.47% concordance). "
                "In CPU latency benchmarks on the Apple M3, the MobileNet batch-1 deployment-path comparison (FP32 Eager vs INT8 TorchScript/QNNPACK) "
                "showed a 44.9× latency ratio under default 4-thread execution (25.821 ms vs 0.575 ms) and a 9.1× ratio under isolated single-thread "
                "execution (3.78 ms vs 0.41 ms), reflecting that INT8 latency varied less across tested thread configurations while measured FP32 "
                "latency was substantially higher under the default four-thread configuration. For ResNet-18, the measured speedup was 1.12× "
                "(3.950 ms vs 3.528 ms) under default threading. The benchmark establishes these empirical differences for these deployment paths "
                "on this hardware, but does not independently identify underlying causes or generalize to other devices."
            ),
            "evidence": "Measured INT8 clean acc delta: ResNet +0.03 pp, MobileNet -0.33 pp; CPU deployment-path ratio: MobileNet 44.9× (4-th) / 9.1× (1-th).",
        },
        {
            "id": 3,
            "title": "Observed Robustness Crossover Under JPEG Compression",
            "badge": "Robustness Crossover",
            "summary": (
                "Under the evaluated JPEG compression transformations, MobileNetV3-Small recorded higher mean corrupted accuracy "
                "than ResNet-18 (69.66% vs 67.83%, a +1.83 pp difference), despite having 7.3× fewer parameters and "
                "an 8.27 pp lower clean accuracy baseline. While the measurements establish this empirical crossover under the "
                "tested JPEG compression suite, they do not identify the specific architectural factors responsible for the difference."
            ),
            "evidence": "Observed JPEG mean corrupted accuracy: MobileNet 69.66% vs ResNet 67.83%.",
        },
        {
            "id": 4,
            "title": "Measurements Show Largest Accuracy Degradation Under Additive Gaussian Noise",
            "badge": "Worst-Case Corruption",
            "summary": (
                "Across the six evaluated synthetic corruption types, additive Gaussian noise was accompanied by the largest observed accuracy "
                "reduction for both architectures, with measured drops of 49.32 pp on ResNet-18 (from 94.78% clean to 45.46% mean corrupted) "
                "and 43.97 pp on MobileNetV3-Small (from 86.51% clean to 42.54% mean corrupted). In contrast, under photometric brightness darkening "
                "(factors [0.30, 0.90]), measured accuracy drops were smallest among tested conditions (ResNet dropped 4.12 pp to 90.66%; "
                "MobileNet dropped 9.83 pp to 76.68%). In these experiments, measured accuracy degradation was greatest under additive Gaussian noise "
                "and least severe under brightness darkening across both architectures."
            ),
            "evidence": "Measured Gaussian noise drops: ResNet 49.32 pp, MobileNet 43.97 pp; brightness drops: ResNet 4.12 pp, MobileNet 9.83 pp.",
        },
        {
            "id": 5,
            "title": "Empirical Pareto Frontier on Size-Robustness Plane Consists of INT8 Variants",
            "badge": "Pareto Optimal",
            "summary": (
                "Under multi-objective evaluation across serialized artifact size versus mean corrupted accuracy across all 30 conditions, "
                "the empirical Pareto frontier in this benchmark is formed by the INT8 variants. MobileNetV3-Small INT8 occupied the compact edge boundary "
                "(1.94 MB artifact size, 59.01% mean corrupted accuracy), while ResNet-18 INT8 defined the higher-accuracy boundary "
                "(11.30 MB artifact size, 65.76% mean corrupted accuracy). Under these benchmark metrics, neither FP32 nor FP16 variants provided a superior "
                "combination of smaller disk footprint and higher mean corrupted accuracy."
            ),
            "evidence": "Empirical size vs corrupted accuracy frontier: MobileNet INT8 (1.94 MB, 59.01%) and ResNet INT8 (11.30 MB, 65.76%).",
        },
    ]

    # 8. Methodology Limitations
    methodology_limitations = [
        {
            "title": "Custom Synthetic Suite vs Official CIFAR-10-C",
            "detail": (
                "The 6 custom synthetic transformations (Gaussian noise, Gaussian blur, horizontal motion blur, "
                "brightness darkening, contrast attenuation, and JPEG compression) do not constitute the official "
                "CIFAR-10-C 15-corruption benchmark. Metrics reflect perturbation stability under controlled mathematical "
                "transformations and cannot be equated with official mean Corruption Error (mCE)."
            ),
        },
        {
            "title": "1D Horizontal Motion Blur",
            "detail": (
                "Motion blur is evaluated using linear 1D horizontal convolution kernels without directional variance, "
                "camera trajectory modeling, or rolling shutter artifacts."
            ),
        },
        {
            "title": "Darkening-Only Photometric Shift",
            "detail": (
                "The brightness transformation applies scalar intensity attenuation (factors in range [0.30, 0.90]: 0.90, 0.75, 0.60, 0.45, 0.30), "
                "modeling underexposure and illumination loss without simulating high-key lighting, glare, or sensor saturation."
            ),
        },
        {
            "title": "Resolution & Generalization Scope",
            "detail": (
                "All evaluations operate on CIFAR-10's 32×32 spatial resolution with a single fixed training seed (42). "
                "Results demonstrate algorithmic sensitivity to resolution-scaled degradations and do not assert real-world "
                "out-of-distribution generalization to megapixel industrial camera sensors or autonomous driving feeds."
            ),
        },
        {
            "title": "Device & Precision Runtime Disparity",
            "detail": (
                "INT8 inference executes strictly on CPU via the QNNPACK backend, while FP16 was evaluated on Apple Silicon MPS. "
                "Hardware acceleration characteristics and memory topologies differ fundamentally across these execution targets."
            ),
        },
        {
            "title": "Single-Image CPU Threading & Hardware Scope",
            "detail": (
                "The 44.9× (default 4 threads) and 9.1× (isolated 1 thread) batch-1 comparisons for MobileNetV3-Small "
                "evaluate specific deployment paths (FP32 PyTorch Eager vs INT8 TorchScript/QNNPACK) on Apple M3 CPU. "
                "The benchmark establishes these empirical latency differences but does not independently identify "
                "their underlying runtime or microarchitectural causes. Results are specific to this Apple M3 benchmark "
                "and should not be generalized to other devices or microarchitectures."
            ),
        },
    ]

    # 9. Benchmark Records (Full 40 Empirical Runs)
    benchmark_records = []
    for _, row in df_bench.iterrows():
        benchmark_records.append({
            "model": row["model"],
            "precision": row["precision"],
            "device": row["device"],
            "backend": row["backend"],
            "batch_size": int(row["batch_size"]),
            "throughput_fps": float(row["throughput_fps"]),
            "median_total_ms": float(row["median_total_ms"]),
            "mean_total_ms": float(row["mean_total_ms"]),
            "p95_total_ms": float(row["p95_total_ms"]),
            "std_total_ms": float(row["std_total_ms"]),
            "per_image_ms": float(row["per_image_ms"]),
            "median_forward_ms": float(row["median_forward_ms"]),
            "forward_only_throughput_fps": float(row["forward_only_throughput_fps"]),
            "load_time_ms": float(row["load_time_ms"]),
            "is_diagnostic": bool(row["is_diagnostic"]),
            "note": str(row["note"]) if pd.notna(row["note"]) else "",
        })

    # 10. Pareto Frontiers from cross_precision_summary.json
    pareto_frontiers = summary.get("pareto_frontiers", {})

    data = {
        "metadata": meta,
        "baselineComparison": baseline_comparison,
        "precisionVariants": precision_variants,
        "severityCurves": severity_curves,
        "perCorruptionData": per_corruption_data,
        "efficiencyHighlights": efficiency_highlights,
        "keyFindings": key_findings,
        "methodologyLimitations": methodology_limitations,
        "benchmarkRecords": benchmark_records,
        "paretoFrontiers": pareto_frontiers,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Generated {output_path} successfully ({output_path.stat().st_size} bytes).")


if __name__ == "__main__":
    generate_research_data()
