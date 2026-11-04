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
| **2** | **Model Baselines (FP32)** | Modular model definitions for MobileNetV3-Small and ResNet18; training engine and clean evaluation infrastructure. |
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

4. **CIFAR-10 Dataset & Pipeline Inspection**:
   ```bash
   python scripts/inspect_cifar10.py
   ```
   Inspects raw dataset integrity, validates train/val/test splits, tests batch tensor dimensions, and generates a visual sample grid at `reports/figures/cifar10_samples.png`.

5. **Run Test Suite**:
   ```bash
   pytest -v
   ```

---

## Model Architecture Specifications

- **ResNet-18 CIFAR Adaptation** (`resnet18_cifar`):
  - Canonical $7\times 7$ (stride 2) stem convolution replaced with $3\times 3$ (stride 1, padding 1, no bias) convolution to preserve $32\times 32$ spatial resolution.
  - Initial maxpool replaced with `nn.Identity()`.
  - Linear classification head replaced with `nn.Linear(512, num_classes)`.
  - Parameters: 11,173,962 (FP32 parameter memory: 44.70 MB).
- **MobileNetV3-Small CIFAR Adaptation** (`mobilenet_v3_small_cifar`):
  - Stem convolution modified to stride 1 (`features[0][0].stride = (1, 1)`), avoiding spatial collapse to $1\times 1$ at stage 9 and preventing single-element `BatchNorm2d` training failures with batch size 1 ($N \times H \times W = 4 > 1$).
  - All 11 inverted residual blocks, internal squeeze-and-excitation modules, and pointwise convolution are preserved.
  - Final projection in `classifier[3]` replaced with `nn.Linear(1024, num_classes)`.
  - Parameters: 1,528,106 (FP32 parameter memory: 6.11 MB).
- **Model Factory & Registry**:
  - `get_model(name, num_classes=10)` dynamically instantiates registered architectures with explicit input validation.

---

## Training Infrastructure & Reproducibility

- **Trainer Interface & Test-Set Isolation**:
  - `Trainer` accepts strictly `train_loader` and `val_loader`.
  - `test_loader` is completely excluded from the training API to eliminate test-set leakage.
  - Checkpoint selection and early stopping decisions are governed solely by validation metrics.
- **Reproducibility & RNG State Management**:
  - `set_seed(seed)` initializes Python `random`, NumPy, PyTorch CPU, and Apple Silicon MPS.
  - `get_rng_states()` and `set_rng_states()` capture and restore the complete PRNG state, including the dedicated `torch.Generator` assigned to the training DataLoader.
  - Guarantees exact bitwise reproducibility upon epoch-boundary checkpoint resume.
- **Atomic Checkpoint Management**:
  - Checkpoints are saved to process-unique temporary files (`.tmp.<pid>`) and atomically replaced via `os.replace` to prevent corrupted checkpoints upon process interruption.
  - Stores full `model.state_dict()` (including BatchNorm running statistics), optimizer state, scheduler state, RNG states, and configuration metadata.
  - Serialized checkpoint sizes with SGD momentum: ResNet-18 is 89.49 MB; MobileNetV3-Small is 12.40 MB.
- **Hardware Acceleration**:
  - Supports `cpu`, `mps`, and `auto` device selection.
  - Hardware smoke testing confirmed Apple Silicon MPS provides $1.3\times$ to $2.1\times$ faster training throughput and $6.5\times$ to $7.8\times$ faster validation throughput than CPU while using modest unified memory footprint (<300 MB process RSS).

---

## Standalone Checkpoint Evaluation

The standalone evaluation infrastructure (`src/edge_vision/evaluation/`) provides decoupled, hermetic evaluation of trained model checkpoints without requiring the training engine:

- **Strict Model Reconstruction**: `load_checkpoint_for_evaluation()` automatically parses checkpoint architecture metadata and loads model weights with `strict=True`.
- **Pure Unaugmented Transforms**: Evaluates strictly with canonical CIFAR-10 normalization, bypassing random cropping and horizontal flipping.
- **Inference Mode & Model Freeze**: Enforces `model.eval()`, parameter gradient disabling (`requires_grad=False`), and `torch.inference_mode()`.
- **Comprehensive Metrics**: Computes sample-weighted cross-entropy loss, top-1 accuracy, per-class counts and accuracy, and a complete $10 \times 10$ confusion matrix, enforcing mathematical consistency assertions ($\sum C = N$, $\text{trace}(C) = N_{\text{correct}}$).
- **Atomic Serialization**: Persists evaluation results atomically to JSON via temporary files and `os.replace`.

### Running Standalone Evaluation

```bash
# Evaluate a checkpoint on the validation split
python scripts/evaluate_checkpoints.py \
  --checkpoint experiments/exp_pilot_resnet18/checkpoints/best_checkpoint.pt \
  --split val \
  --device auto \
  --batch-size 64 \
  --output reports/experiment_results/eval_val_resnet18.json

# Evaluate on the held-out test split (reserved for final benchmark evaluation)
python scripts/evaluate_checkpoints.py \
  --checkpoint experiments/exp_pilot_resnet18/checkpoints/best_checkpoint.pt \
  --split test \
  --device auto \
  --batch-size 64 \
  --output reports/experiment_results/eval_test_resnet18.json
```

---

## Data Pipeline Specifications

- **Dataset**: Official CIFAR-10 (torchvision implementation, stored under `data/raw/`).
- **Data Partitioning**:
  - **Training Set**: 45,000 images (derived deterministically with seed `42`).
  - **Validation Set**: 5,000 images (non-overlapping subset for hyperparameter/checkpoint evaluation).
  - **Test Set**: 10,000 images (strictly held-out benchmark partition).
- **Split Generation & Reproducibility**:
  - Deterministically generated via `torch.Generator().manual_seed(seed)` and `torch.randperm(50000)`.
  - Partition disjointness, length constraints, and 100% dataset coverage are strictly validated at runtime.
  - Verification manifest persisted at [configs/cifar10_split_manifest.json](file:///Users/sushantydv01/Desktop/edge-vision-reliability-lab/configs/cifar10_split_manifest.json) (<1 KB) storing SHA-256 partition checksums, PyTorch version, algorithm tags, and boundary verification samples.
  - If the requested seed differs from the manifest, the pipeline automatically regenerates the split for the requested seed rather than silently reusing stale indices.
  - If a checksum mismatch occurs for the configured seed (e.g. across major PyTorch releases), the pipeline raises an explicit error and requires intentional manifest regeneration via `force_regenerate=True`.
- **Canonical Normalization**:
  - Mean: `(0.4914, 0.4822, 0.4465)`
  - Standard Deviation: `(0.2470, 0.2435, 0.2616)`
- **Memory-Conscious Architecture**:
  - `TransformedSubset` implements a lightweight indexed view with on-the-fly transforms over a shared underlying dataset object. The 153.6 MB raw image array is loaded once and shared between train and validation splits without memory duplication.
  - Default batch size: `64`.
  - Default worker count: `0` (avoiding process-forking memory overhead on macOS).
  - Pinned memory: Disabled by default for conservative host memory footprint.

- **Current Reproducibility Limitations**:
  - **PyTorch PRNG Versioning**: PyTorch does not guarantee bit-for-bit identical `torch.randperm` permutations across future major versions. Checksum verification in `get_or_create_split_indices` detects any PRNG drift and requires explicit intentional regeneration (`force_regenerate=True`).
  - **Worker Concurrency**: Determinism is strictly verified with `num_workers=0`. When multi-worker training is introduced in subsequent milestones, deterministic `worker_init_fn` seeding will be evaluated.
