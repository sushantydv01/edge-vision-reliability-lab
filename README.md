# Edge Vision Reliability Lab

A reproducible experimental platform for investigating how model compression affects computer vision reliability under realistic image corruptions.

## Research Question

> **How does model compression (FP16 precision reduction and INT8 post-training quantization) impact the out-of-distribution robustness and corruption reliability of compact edge vision architectures compared to standard convolutional baselines?**

When deploying models to resource-constrained edge environments, compression techniques (such as quantization) are essential to satisfy latency, memory, and energy budgets. However, edge systems frequently operate in noisy, dynamic physical environments (e.g., weather degradation, sensor noise, compression artifacts). This project quantifies the trade-offs between computational efficiency and corruption reliability.

---

## Project Scope

- **Architectures**:
  - `MobileNetV3-Small` (compact, inverted residual edge architecture)
  - `ResNet18` (canonical residual baseline)
- **Precision / Compression Variants**:
  - FP32 baseline
  - FP16 (where supported/applicable)
  - INT8 Post-Training Quantization (PTQ)
- **Dataset**: CIFAR-10 & CIFAR-10-C style perturbations (common realistic image corruptions across noise, blur, weather, and digital categories at multiple severity levels).
- **Target Metrics**:
  - **Reliability**: Clean accuracy, corruption error (mCE / corrupted accuracy across severities), degradation delta relative to FP32 baseline.
  - **Efficiency**: Model file size / parameter memory footprint, inference latency (mean, median, p95/p99), throughput (samples/sec), peak working memory.
- **Hardware Profile**:
  - Edge target constraint: Apple M3 (8 GB unified memory), strict memory limits.
  - Primary compute target: CPU (conservative and fully reproducible baseline); optional Apple MPS acceleration evaluation.
  - No dependency on cloud GPUs or external proprietary APIs.

---

## Planned Milestones

| Milestone | Title | Objective |
| :--- | :--- | :--- |
| **0** | **Environment & Project Baseline** | Workspace verification, Git tracking, Python 3.12 virtual environment setup, strict pinning. |
| **1** | **Data & Corruption Pipeline** | CIFAR-10 data loader with clean baseline and standardized synthetic image corruption suite. |
| **2** | **Model Baselines (FP32)** | Modular model definitions for MobileNetV3-Small and ResNet18; baseline weights validation. |
| **3** | **Compression & Quantization** | FP16 conversion and INT8 Post-Training Quantization (PTQ) pipelines with calibration routines. |
| **4** | **Benchmarking Engine** | Deterministic profiling harness measuring latency, memory usage, throughput, and robustness metrics. |
| **5** | **Analysis & Reporting** | Comparative evaluation reports, Pareto frontier visualizations (efficiency vs reliability), and experimental findings.

---

## Project Structure

```text
edge-vision-reliability-lab/
├── configs/                  # Experiment configuration files (YAML/JSON)
├── data/
│   ├── raw/                  # Original downloaded datasets (e.g. CIFAR-10)
│   └── processed/            # Preprocessed datasets or cached corruption batches
├── models/                   # Serialized checkpoints and quantized model artifacts
├── artifacts/                # Profile traces, intermediate exports, logs
├── reports/
│   ├── figures/              # Generated plots and visualization figures
│   └── experiment_results/   # Structured metric logs (CSV/JSON/Parquet)
├── scripts/                  # Standalone execution scripts (run_benchmarks, train, quantize)
├── src/
│   └── edge_vision/          # Reusable Python package modules
├── tests/                    # Unit, regression, and pipeline integration tests
├── .gitignore                # Git ignore configuration
├── PROJECT_STATUS.md         # Current state and completed tasks log
└── README.md                 # Project overview and documentation
```

---

## Getting Started

1. **Prerequisites**:
   - Python 3.12 (installed at `/opt/homebrew/bin/python3.12`)
   - Apple Silicon Mac (M-series, e.g., M3)

2. **Setup Virtual Environment & Install Dependencies**:
   ```bash
   /opt/homebrew/bin/python3.12 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements-lock.txt
   ```

3. **Run Environment & Hardware Verification**:
   ```bash
   python scripts/verify_env.py
   ```

