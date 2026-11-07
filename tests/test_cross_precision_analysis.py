"""
Unit tests for Cross-Precision Efficiency-Reliability Analysis and Pareto Frontier calculations.
"""

import hashlib
from pathlib import Path
import pytest

from edge_vision.analysis import (
    MODEL_METADATA,
    build_clean_accuracy_table,
    build_cpu_performance_table,
    build_model_size_table,
    build_pareto_analysis,
    build_per_corruption_table,
    build_per_severity_table,
    build_robustness_summary_table,
    compute_pareto_frontier,
    is_pareto_dominant,
    load_cross_precision_data,
)


@pytest.fixture
def analysis_data():
    results_dir = Path("reports/experiment_results")
    experiments_dir = Path("experiments")
    return load_cross_precision_data(results_dir, experiments_dir)


def test_data_pipeline_integrity(analysis_data):
    """Verifies that all 3 precision levels contain valid 31-condition data with 10k samples each."""
    dfs = analysis_data["dfs"]
    assert set(dfs.keys()) == {"fp32", "fp16", "int8"}

    for prec, df in dfs.items():
        assert len(df) == 62
        for model in ["resnet18_cifar", "mobilenet_v3_small_cifar"]:
            m_df = df[df["model"] == model]
            assert len(m_df) == 31
            assert (m_df["sample_count"] == 10_000).all()
            assert (m_df["correct"] + m_df["incorrect"] == 10_000).all()


def test_model_size_and_compression_calculations(analysis_data):
    """Verifies theoretical parameter storage vs serialized artifact sizes and compression ratios."""
    sizes = analysis_data["checkpoint_sizes"]
    size_tab = build_model_size_table(sizes)
    assert len(size_tab) == 6

    # Verify ResNet-18
    res_fp32 = [r for r in size_tab if r["model"] == "resnet18_cifar" and r["precision"] == "FP32"][0]
    res_fp16 = [r for r in size_tab if r["model"] == "resnet18_cifar" and r["precision"] == "FP16"][0]
    res_int8 = [r for r in size_tab if r["model"] == "resnet18_cifar" and r["precision"] == "INT8"][0]

    assert res_fp32["parameters"] == 11_173_962
    assert res_fp32["idealized_param_payload_bytes"] == 44_695_848
    assert res_fp16["idealized_param_payload_bytes"] == 22_347_924
    assert res_int8["idealized_param_payload_bytes"] == 11_173_962
    assert res_fp32["theoretical_param_bytes"] == 44_695_848
    assert res_fp16["theoretical_param_bytes"] == 22_347_924
    assert res_int8["theoretical_param_bytes"] == 11_173_962

    assert res_fp16["compression_ratio"] == pytest.approx(1.998, rel=1e-2)
    assert res_int8["compression_ratio"] == pytest.approx(3.964, rel=1e-2)
    assert res_fp16["size_reduction_pct"] == pytest.approx(49.95, rel=1e-2)
    assert res_int8["size_reduction_pct"] == pytest.approx(74.77, rel=1e-2)

    # Verify MobileNetV3-Small
    mob_fp32 = [r for r in size_tab if r["model"] == "mobilenet_v3_small_cifar" and r["precision"] == "FP32"][0]
    mob_fp16 = [r for r in size_tab if r["model"] == "mobilenet_v3_small_cifar" and r["precision"] == "FP16"][0]
    mob_int8 = [r for r in size_tab if r["model"] == "mobilenet_v3_small_cifar" and r["precision"] == "INT8"][0]

    assert mob_fp32["parameters"] == 1_528_106
    assert mob_fp32["idealized_param_payload_bytes"] == 6_112_424
    assert mob_fp16["idealized_param_payload_bytes"] == 3_056_212
    assert mob_int8["idealized_param_payload_bytes"] == 1_528_106
    assert mob_fp32["theoretical_param_bytes"] == 6_112_424
    assert mob_fp16["theoretical_param_bytes"] == 3_056_212
    assert mob_int8["theoretical_param_bytes"] == 1_528_106

    assert mob_fp16["compression_ratio"] == pytest.approx(1.972, rel=1e-2)
    assert mob_int8["compression_ratio"] == pytest.approx(3.218, rel=1e-2)
    assert mob_fp16["size_reduction_pct"] == pytest.approx(49.29, rel=1e-2)
    assert mob_int8["size_reduction_pct"] == pytest.approx(68.92, rel=1e-2)


def test_clean_accuracy_concordance(analysis_data):
    """Verifies clean test accuracy, difference relative to FP32, and concordance."""
    clean_tab = build_clean_accuracy_table(
        analysis_data["summaries"],
        analysis_data["clean_fp16_comp"],
        analysis_data["clean_int8_comp"],
    )
    assert len(clean_tab) == 6

    # ResNet-18
    res_rows = {r["precision"]: r for r in clean_tab if r["model"] == "resnet18_cifar"}
    assert res_rows["FP32"]["clean_accuracy"] == 94.78
    assert res_rows["FP16"]["clean_accuracy"] == 94.78
    assert res_rows["INT8"]["clean_accuracy"] == 94.81
    assert res_rows["FP16"]["accuracy_diff_pp"] == 0.00
    assert res_rows["INT8"]["accuracy_diff_pp"] == 0.03
    assert res_rows["FP16"]["concordance_pct"] == 99.99
    assert res_rows["INT8"]["concordance_pct"] == 99.64

    # MobileNetV3-Small
    mob_rows = {r["precision"]: r for r in clean_tab if r["model"] == "mobilenet_v3_small_cifar"}
    assert mob_rows["FP32"]["clean_accuracy"] == 86.51
    assert mob_rows["FP16"]["clean_accuracy"] == 86.49
    assert mob_rows["INT8"]["clean_accuracy"] == 86.18
    assert mob_rows["FP16"]["accuracy_diff_pp"] == -0.02
    assert mob_rows["INT8"]["accuracy_diff_pp"] == -0.33
    assert mob_rows["FP16"]["concordance_pct"] == 99.94
    assert mob_rows["INT8"]["concordance_pct"] == 95.47


def test_robustness_aggregation_and_negative_drop_preservation(analysis_data):
    """Verifies mean corrupted accuracy, accuracy drops, and negative drop preservation."""
    rob_tab = build_robustness_summary_table(analysis_data["summaries"])
    assert len(rob_tab) == 6

    # Verify that negative drops exist in raw corrupted conditions (e.g. brightness sev 1)
    for prec, df in analysis_data["dfs"].items():
        corrupt_only = df[df["condition"] != "clean"]
        assert (corrupt_only["accuracy_drop_pp"] < 0).any(), f"Expected negative drops in {prec}"

    # Verify overall means match expected values
    res_int8 = [r for r in rob_tab if r["model"] == "resnet18_cifar" and r["precision"] == "INT8"][0]
    assert res_int8["mean_corrupted_accuracy"] == 65.76
    assert res_int8["mean_accuracy_drop_pp"] == 29.052

    mob_int8 = [r for r in rob_tab if r["model"] == "mobilenet_v3_small_cifar" and r["precision"] == "INT8"][0]
    assert mob_int8["mean_corrupted_accuracy"] == 59.01
    assert mob_int8["mean_accuracy_drop_pp"] == 27.167


def test_pareto_dominance_logic():
    """Unit tests for the Pareto dominance evaluator."""
    # Minimize size, maximize acc
    objs = [("size", "min"), ("acc", "max")]

    # A is strictly smaller and more accurate -> A dominates B
    a = {"size": 10, "acc": 95}
    b = {"size": 20, "acc": 90}
    assert is_pareto_dominant(a, b, objs) is True
    assert is_pareto_dominant(b, a, objs) is False

    # Trade-off: A is smaller, but B is more accurate -> neither dominates
    c = {"size": 10, "acc": 85}
    d = {"size": 20, "acc": 95}
    assert is_pareto_dominant(c, d, objs) is False
    assert is_pareto_dominant(d, c, objs) is False

    # Equal on all objectives -> neither strictly dominates
    e = {"size": 10, "acc": 90}
    f = {"size": 10, "acc": 90}
    assert is_pareto_dominant(e, f, objs) is False

    # Missing value handling
    g = {"size": None, "acc": 90}
    assert is_pareto_dominant(g, a, objs) is False
    assert is_pareto_dominant(a, g, objs) is False

    # 4-objective mixed direction test: (size: min, lat: min, clean: max, corr: max)
    mixed_objs = [("size", "min"), ("lat", "min"), ("clean", "max"), ("corr", "max")]
    m1 = {"size": 10, "lat": 1.0, "clean": 95.0, "corr": 70.0}
    m2 = {"size": 15, "lat": 1.5, "clean": 90.0, "corr": 65.0}  # Strictly worse on all 4
    m3 = {"size": 8, "lat": 2.0, "clean": 96.0, "corr": 72.0}   # Trade-off: smaller and more accurate, but higher latency
    assert is_pareto_dominant(m1, m2, mixed_objs) is True
    assert is_pareto_dominant(m2, m1, mixed_objs) is False
    assert is_pareto_dominant(m1, m3, mixed_objs) is False
    assert is_pareto_dominant(m3, m1, mixed_objs) is False


def test_pareto_frontier_computation(analysis_data):
    """Verifies Pareto frontiers for size vs clean and size vs corrupted accuracy."""
    size_tab = build_model_size_table(analysis_data["checkpoint_sizes"])
    rob_tab = build_robustness_summary_table(analysis_data["summaries"])
    pareto_res = build_pareto_analysis(size_tab, rob_tab, analysis_data["cpu_bench"])

    # On Size vs Corrupted Accuracy, only INT8 models should be on the Pareto frontier
    # because INT8 achieves higher corrupted accuracy than FP32/FP16 on both architectures while being smallest
    f_size_corr = [f"{x['model']}_{x['precision']}" for x in pareto_res["pareto_size_vs_corrupted"]["frontier"]]
    assert set(f_size_corr) == {
        "mobilenet_v3_small_cifar_INT8",
        "resnet18_cifar_INT8",
    }

    # On Size vs Clean Accuracy:
    # ResNet-18 INT8 dominates ResNet-18 FP32 and FP16 (smaller and higher accuracy)
    # MobileNetV3 INT8, FP16, FP32 form a trade-off chain (FP32 has +0.02pp over FP16, FP16 has +0.31pp over INT8)
    f_size_clean = [f"{x['model']}_{x['precision']}" for x in pareto_res["pareto_size_vs_clean"]["frontier"]]
    assert "resnet18_cifar_INT8" in f_size_clean
    assert "resnet18_cifar_FP32" not in f_size_clean
    assert "resnet18_cifar_FP16" not in f_size_clean

    # Verify Within-Architecture Pareto Frontiers:
    # ResNet-18 Size vs Clean: only INT8 is non-dominated
    res_clean_frontier = [f"{x['model']}_{x['precision']}" for x in pareto_res["within_architecture"]["resnet18_cifar"]["pareto_size_vs_clean"]["frontier"]]
    assert res_clean_frontier == ["resnet18_cifar_INT8"]

    # ResNet-18 1-thread CPU Latency vs Clean: both FP32 and INT8 are non-dominated (trade-off between 4.4ms and 10.9ms)
    res_lat1_frontier = [f"{x['model']}_{x['precision']}" for x in pareto_res["within_architecture"]["resnet18_cifar"]["pareto_cpu_latency_th1_vs_clean"]["frontier"]]
    assert set(res_lat1_frontier) == {"resnet18_cifar_FP32", "resnet18_cifar_INT8"}

    # ResNet-18 2-thread CPU Latency vs Clean: both FP32 and INT8 are non-dominated (3.916ms vs 5.709ms)
    res_lat2_frontier = [f"{x['model']}_{x['precision']}" for x in pareto_res["within_architecture"]["resnet18_cifar"]["pareto_cpu_latency_th2_vs_clean"]["frontier"]]
    assert set(res_lat2_frontier) == {"resnet18_cifar_FP32", "resnet18_cifar_INT8"}

    # ResNet-18 4-thread CPU Latency vs Clean: INT8 strictly dominates FP32 (3.084ms < 3.477ms, 94.81% > 94.78%)
    res_lat4_frontier = [f"{x['model']}_{x['precision']}" for x in pareto_res["within_architecture"]["resnet18_cifar"]["pareto_cpu_latency_th4_vs_clean"]["frontier"]]
    assert res_lat4_frontier == ["resnet18_cifar_INT8"]

    # MobileNetV3 Size vs Clean: all 3 (INT8, FP16, FP32) are non-dominated
    mob_clean_frontier = [f"{x['model']}_{x['precision']}" for x in pareto_res["within_architecture"]["mobilenet_v3_small_cifar"]["pareto_size_vs_clean"]["frontier"]]
    assert set(mob_clean_frontier) == {"mobilenet_v3_small_cifar_INT8", "mobilenet_v3_small_cifar_FP16", "mobilenet_v3_small_cifar_FP32"}

    # Cross-Architecture CPU Latency 1-thread vs Clean: all 4 CPU models are non-dominated
    cross_lat1_clean = [f"{x['model']}_{x['precision']}" for x in pareto_res["cross_architecture"]["pareto_cpu_latency_th1_vs_clean"]["frontier"]]
    assert set(cross_lat1_clean) == {
        "mobilenet_v3_small_cifar_INT8",
        "mobilenet_v3_small_cifar_FP32",
        "resnet18_cifar_FP32",
        "resnet18_cifar_INT8",
    }

    # Cross-Architecture CPU Latency 2-thread vs Clean: MobileNet FP32 (17.02ms, 86.51%) is dominated by ResNet FP32 (3.92ms, 94.78%)
    cross_lat2_clean = [f"{x['model']}_{x['precision']}" for x in pareto_res["cross_architecture"]["pareto_cpu_latency_th2_vs_clean"]["frontier"]]
    assert set(cross_lat2_clean) == {
        "mobilenet_v3_small_cifar_INT8",
        "resnet18_cifar_FP32",
        "resnet18_cifar_INT8",
    }

    # Cross-Architecture CPU Latency 4-thread vs Clean: MobileNet INT8 and ResNet INT8
    cross_lat4_clean = [f"{x['model']}_{x['precision']}" for x in pareto_res["cross_architecture"]["pareto_cpu_latency_th4_vs_clean"]["frontier"]]
    assert set(cross_lat4_clean) == {
        "mobilenet_v3_small_cifar_INT8",
        "resnet18_cifar_INT8",
    }


def test_artifact_exact_byte_counts(analysis_data):
    """Verifies that actual serialized inference checkpoint byte counts match verified values."""
    sizes = analysis_data["checkpoint_sizes"]
    expected_exact_bytes = {
        ("resnet18_cifar", "fp32"): 44_775_947,
        ("resnet18_cifar", "fp16"): 22_408_843,
        ("resnet18_cifar", "int8"): 11_296_669,
        ("mobilenet_v3_small_cifar", "fp32"): 6_249_979,
        ("mobilenet_v3_small_cifar", "fp16"): 3_169_467,
        ("mobilenet_v3_small_cifar", "int8"): 1_942_499,
    }
    for key, expected_bytes in expected_exact_bytes.items():
        assert sizes[key] == expected_bytes, f"Byte mismatch for {key}: expected {expected_bytes}, got {sizes[key]}"


def test_historical_fp32_baseline_integrity():
    """Confirms that historical baseline FP32 results remain byte-for-byte unchanged."""
    expected_hashes = {
        "experiments/exp_baseline_resnet18_seed42/checkpoints/best_checkpoint.pt": "56a238e67ad34321895d88da43ea182141a0355bf15bee94ca2dc51ccdedabff",
        "experiments/exp_baseline_mobilenet_v3_small_seed42/checkpoints/best_checkpoint.pt": "7243571c4d769b0a489ae3fd7392f407b80609de9e3c81f60284e21afca581c7",
        "reports/experiment_results/robustness_fp32_results.csv": "9b0927b8f53665fc844da0d535e0fe098813e9fb848f027c6fd50cd9203aa008",
        "reports/experiment_results/robustness_fp32_summary.json": "2de0c05a44edbcd361f7ab4cf08f9ea9bf5051c52ec41a102268ecf792b1fc87",
        "reports/experiment_results/robustness_fp32_metadata.json": "b2747d7f60ae12df82f580e72b220c220abcc35918535c7acfd98f81f7377899",
        "reports/experiment_results/robustness_fp32_detailed_results.json": "4a773c9a9df39e1a07f6f560473ea50c6b0a395ac6aa4eb4c1d916ec18e98566",
    }

    for path_str, exp_hash in expected_hashes.items():
        with open(path_str, "rb") as f:
            actual_hash = hashlib.sha256(f.read()).hexdigest()
        assert actual_hash == exp_hash, f"Hash mismatch for {path_str}!"

