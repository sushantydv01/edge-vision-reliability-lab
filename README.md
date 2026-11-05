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
  - **Reliability**: Clean accuracy, accuracy under synthetic image corruptions, absolute and relative degradation deltas across severities.
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
| **2** | **Model Baselines (FP32)** | Modular model definitions for MobileNetV3-Small and ResNet18; 100-epoch baseline training and clean test evaluation (Completed). |
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

## Baseline Model Performance (100-Epoch FP32 Experiments)

Both architectures were trained from scratch for 100 epochs under an identical controlled baseline recipe (`configs/train_baseline.yaml`: SGD, lr 0.05, momentum 0.9, weight decay $5\times 10^{-4}$, CosineAnnealingLR, batch size 64, seed 42) on Apple Silicon MPS:

| Architecture | Experiment ID | Parameters | Checkpoint Size (Disk) | Peak Val Acc (Epoch) | Min Val Loss (Epoch) | Test Acc (10,000 samples) | Test Loss | MPS Wall Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ResNet-18** (`resnet18_cifar`) | `exp_baseline_resnet18_seed42` | 11,173,962 | 89.51 MB (85.36 MiB) | **95.50%** (Ep 97) | 0.1814 (Ep 98) | **94.78%** | 0.2003 | 2.64 hours |
| **MobileNetV3-Small** (`mobilenet_v3_small_cifar`) | `exp_baseline_mobilenet_v3_small_seed42` | 1,528,106 | 12.42 MB (11.85 MiB) | **86.86%** (Ep 98) | 0.3649 (Ep 98) | **86.51%** | 0.3896 | 0.76 hours |

- **Generalization Metrics**:
  - ResNet-18 exhibited a training-to-validation gap of 4.54% (99.96% train vs 95.42% val at Ep 100) and a validation-to-test difference of 0.72% (95.50% vs 94.78%).
  - MobileNetV3-Small exhibited a training-to-validation gap of 2.76% (89.42% train vs 86.66% val at Ep 100) and a validation-to-test difference of 0.35% (86.86% vs 86.51%).
- **Per-Class Dynamics**: Both networks performed highest on rigid vehicle categories (>91% test accuracy) and lowest on fine-textured animal categories, with cat/dog confusion accounting for the majority of classification errors (cats: 88.8% for ResNet-18 vs 70.4% for MobileNetV3-Small).
- **Scope & Methodological Limitations**: Results represent an initial single random seed (seed 42) on low-resolution $32\times 32$ CIFAR-10 imagery. Full multi-seed variance will be measured in subsequent benchmarks.
- **Learning Curves & Visualizations**: Detailed loss, accuracy, and comparison plots are available under `reports/figures/`:
  - `baseline_resnet18_loss.png` & `baseline_resnet18_accuracy.png`
  - `baseline_mobilenet_loss.png` & `baseline_mobilenet_accuracy.png`
  - `baseline_learning_rate.png`
  - `baseline_validation_comparison.png` & `baseline_per_class_test_accuracy.png`

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
  - Serialized checkpoint sizes with SGD momentum: ResNet-18 is 89.51 MB (85.36 MiB); MobileNetV3-Small is 12.42 MB (11.85 MiB).
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

## Deterministic Synthetic Image Corruption Suite

The corruption engine (`src/edge_vision/corruptions/`) provides a clean, modular, and mathematically reproducible platform for synthesizing environmental and digital image degradations to evaluate out-of-distribution robustness.

> **Research Scope**: This is a custom, deterministic synthetic corruption suite inspired by common corruption benchmarks (such as Hendrycks & Dietterich 2019) and tailored for CIFAR-10 resolution ($32\times 32\times 3$). It is an independent experimental implementation, not an official CIFAR-10-C release.

### Supported Corruptions & Implemented Severity Levels

Each corruption defines five progressive severity levels (1 to 5) centralized in [`configs/corruptions.yaml`](configs/corruptions.yaml):

| Corruption Name | Parameter | Severity 1 | Severity 2 | Severity 3 | Severity 4 | Severity 5 | Description & Operational Details |
| :--- | :--- | ---:| ---:| ---:| ---:| ---:| :--- |
| `gaussian_noise` | $\sigma_{\text{norm}}$ ($[0, 1]$) | 0.04 | 0.06 | 0.08 | 0.10 | 0.12 | Additive zero-mean Gaussian noise scaled by 255.0 to uint8 pixel range. |
| `gaussian_blur` | $\sigma$ (pixels) | 0.3 | 0.5 | 0.7 | 0.9 | 1.1 | Spatial 2D Gaussian filter (`scipy.ndimage.gaussian_filter`, reflect mode, per-channel). |
| `motion_blur` | $K$ (pixels) | 3 | 5 | 7 | 9 | 11 | Horizontal 1D linear motion blur ($1/K$ averaging kernel, reflect mode). |
| `brightness` | factor | 0.90 | 0.75 | 0.60 | 0.45 | 0.30 | Multiplicative darkening ($I' = \text{clip}(\text{round}(I \times \text{factor}))$, progressive darkening). |
| `contrast` | factor | 0.80 | 0.60 | 0.40 | 0.20 | 0.10 | Linear scaling around spatial channel mean ($I' = \mu_c + \text{factor} \times (I - \mu_c)$). |
| `jpeg_compression` | quality | 80 | 60 | 40 | 20 | 10 | In-memory JPEG compression and decompression artifacts via PIL and `io.BytesIO`. |

### Architectural Design & Invariants

1. **Input / Output Contract**:
   - Accepts raw RGB images as NumPy arrays of shape $(H, W, 3)$ with `uint8` values in $[0, 255]$.
   - Validates shape, channels ($C=3$), dtype (`uint8`), and severity level ($1 \le \text{severity} \le 5$).
   - Returns a freshly allocated NumPy array of identical shape and dtype with valid pixels in $[0, 255]$.
   - Strictly non-destructive: original input arrays are never modified in-place.
2. **Pre-Normalization Execution**:
   - Corruptions are executed on raw uint8 pixel arrays *before* torchvision `ToTensor` (conversion to $[0, 1]$ float) and `Normalize` (zero-mean standardization).
3. **Deterministic Seeding & Common Random Numbers (CRN)**:
   - For stochastic operations (`gaussian_noise`), seeds are computed from `(seed, sample_idx, corruption_name)` via SHA-256 digest:
     ```python
     token = f"{seed}:{sample_idx}:{corruption_name}"
     digest = hashlib.sha256(token.encode("utf-8")).digest()
     derived_seed = struct.unpack(">Q", digest[:8])[0]
     ```
   - **Common Random Numbers (CRN)**: Under the default CRN design, a standardized noise field $\xi \sim \mathcal{N}(0, 1)$ is sampled once per image sample and scaled proportionally by $\sigma = \text{std\_norm} \times 255.0$ across severity levels. This guarantees *ceteris paribus* severity comparisons: increasing severity scales the magnitude of the exact same spatial noise realization rather than introducing random spatial variance. Independent streams per severity can be selected via `use_common_random_numbers=False`.
   - Eliminates dependence on Python's randomized process `hash()` seed, DataLoader worker scheduling, batch sizes, or shuffle ordering. Cross-process determinism was empirically verified via isolated Python subprocesses with varying `PYTHONHASHSEED`.
   - NumPy's global random state (`np.random.get_state()`) is never altered and has zero effect on corruption outputs.
4. **Lightweight Dataset Adapter (`CorruptedDataset`)**:
   - Wraps torchvision `CIFAR10`, `TransformedSubset`, or `torch.utils.data.Subset` instances without copying or duplicating the underlying 153.6 MB raw dataset in host memory.
   - Recursively resolves canonical dataset sample indices to ensure bitwise identical noise realization for identical samples across re-evaluations and multi-model benchmarking.
   - Fails explicitly with an informative `TypeError` for unsupported datasets (e.g., `TensorDataset` or pre-transformed tensors).
5. **Configuration Single Source of Truth**:
   - [`configs/corruptions.yaml`](configs/corruptions.yaml) is the single authoritative source of truth for severity parameters.
   - Python code dynamically populates `CORRUPTION_REGISTRY` without duplicate hardcoded parameter tables.

### Public API Usage Example

```python
import numpy as np
from edge_vision.corruptions import (
    apply_corruption,
    list_corruptions,
    CorruptedDataset,
    get_corrupted_dataloader,
)
from edge_vision.data.transforms import get_cifar10_transforms
import torchvision.datasets as datasets

# 1. Inspect supported operations
print(list_corruptions())
# ['brightness', 'contrast', 'gaussian_blur', 'gaussian_noise', 'jpeg_compression', 'motion_blur']

# 2. Corrupt a single raw CIFAR-10 image array (32x32x3 uint8)
raw_img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
corrupted_img = apply_corruption(
    image=raw_img,
    name="gaussian_noise",
    severity=3,
    seed=42,
    sample_idx=100,
)
assert corrupted_img.shape == (32, 32, 3) and corrupted_img.dtype == np.uint8

# 3. Create a Corrupted DataLoader over the CIFAR-10 test set
test_dataset = datasets.CIFAR10(root="data/raw", train=False, download=False, transform=None)
eval_transform = get_cifar10_transforms(train=False, augment=False)

corrupted_loader = get_corrupted_dataloader(
    base_dataset=test_dataset,
    corruption_name="motion_blur",
    severity=4,
    transform=eval_transform,
    batch_size=64,
)
```

### Limitations & Roadmap

- **Motion Blur Kernel**: Initial version implements pure horizontal linear motion blur (orientation 0 degrees); arbitrary angle kernels can be added if justified.
- **Illumination Range**: Initial brightness suite tests progressive darkening ($0.90 \to 0.30$) rather than bilateral overexposure.
- **Weather Corruptions**: Excluded from initial suite; reserved for future evaluation milestones if physical domain transfer requires them.

---

## FP32 Robustness Evaluation Engine

The robustness evaluation engine (`src/edge_vision/evaluation/robustness.py`) evaluates model predictive reliability across clean and synthetically corrupted image conditions.

### Evaluation Workflow & Guarantees
- **Shared-Batch Presentation**: Both candidate models (`resnet18_cifar` and `mobilenet_v3_small_cifar`) are evaluated on the exact same in-memory tensor batch sequentially, eliminating input divergence across architectures.
- **31 Evaluation Conditions**: Clean test set + 6 corruptions $\times$ 5 severities = 31 conditions (62 evaluations, 620,000 total predictions across the 10,000 test set images).
- **Metric Definitions**:
  - Absolute Accuracy Drop ($\text{pp}$): $(\text{clean\_acc} - \text{corrupted\_acc}) \times 100.0$
  - Relative Accuracy Drop: $(\text{clean\_acc} - \text{corrupted\_acc}) / \text{clean\_acc}$
  - Aggregates: Arithmetic mean corrupted accuracy and mean drop (overall, per-corruption, per-severity).
  - Timing Semantics (`condition_wall_time_seconds`): Condition wall-clock time measures the complete shared evaluation loop, including DataLoader iteration, corruption generation, preprocessing, device transfer, both model forward passes, and metric tracking. It is not an isolated per-model inference latency measurement (dedicated latency profiling is reserved for Milestone 4).
- **Atomic Outputs & Overwrite Protection**:
  - Preserves `metadata.json`, `results.csv`, `results.json`, and `summary.json` via atomic process-unique temporary files and `os.replace`.
  - Rejects execution if the output directory already contains completed benchmark artifacts unless `--overwrite` is explicitly specified.

### CLI Usage

```bash
# Run smoke test (128 samples, clean + gaussian noise sev 1)
python scripts/run_robustness_benchmark.py --smoke-test --device auto

# Run custom subset evaluation
python scripts/run_robustness_benchmark.py \
  --models resnet18_cifar mobilenet_v3_small_cifar \
  --corruptions gaussian_noise motion_blur \
  --severities 1 3 5 \
  --max-samples 1000 \
  --device auto

# Execute full 620,000-prediction benchmark (requires --full-run; add --overwrite if re-running)
python scripts/run_robustness_benchmark.py \
  --full-run \
  --device auto \
  --output-dir experiments/robustness_fp32_seed42

# Generate analytical robustness plots
python scripts/generate_robustness_plots.py
```

### FP32 Robustness Baseline Results (Seed 42)

Evaluated across 10,000 held-out CIFAR-10 test images under clean and 30 corrupted conditions (6 corruptions $\times$ 5 severities, 620,000 total predictions):

| Metric | ResNet-18 (FP32) | MobileNetV3-Small (FP32) | Delta ($\Delta = \text{ResNet} - \text{MobileNet}$) |
| :--- | ---:| ---:| ---:|
| Clean Accuracy | **94.78%** | 86.51% | +8.27 pp |
| Mean Corrupted Accuracy | **65.73%** | 58.83% | +6.90 pp |
| Mean Absolute Accuracy Drop | 29.05 pp | **27.68 pp** | +1.37 pp (ResNet drops slightly more in abs pp) |
| Mean Relative Accuracy Drop | **30.65%** | 32.00% | -1.35% (MobileNet degrades slightly more relatively) |
| Worst Corruption Type | `gaussian_noise` (45.46%) | `motion_blur` (41.47%) | Architecture-dependent vulnerability |
| Worst Single Condition | `gaussian_noise_sev5` (22.79%) | `contrast_sev5` (20.50%) | Catastrophic degradation under extreme noise/contrast |

#### Key Empirical Insights
- **Robustness Metric Duality**: ResNet-18 exhibits higher clean and overall corrupted accuracy, but MobileNetV3-Small exhibits lower absolute accuracy drop in percentage points (27.68 pp vs. 29.05 pp). Conclusions regarding which model is "more robust" depend strictly on whether absolute drop or relative degradation is evaluated.
- **JPEG Compression Crossover**: In this benchmark, MobileNetV3-Small achieved higher empirical accuracy than ResNet-18 under severe JPEG compression (quality 40: 73.91% vs. 73.38%; quality 20: 65.45% vs. 59.25%; quality 10: 50.65% vs. 40.61%). The specific architectural mechanisms behind this difference have not been isolated experimentally and remain an open research question.
- **Experimental Scope & Limitations**:
  - Image inputs are $32\times 32$ CIFAR-10 resolution; findings should not be assumed to hold for higher-resolution domains.
  - The corruption engine is an independent custom synthetic suite, not official CIFAR-10-C.
  - Motion blur is horizontal-only; brightness is darkening-only.
  - Evaluations reflect a single fixed seed (seed 42); multi-seed variance has not been evaluated.
  - Synthetic degradations do not establish real-world camera lens or environmental degradation performance.

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
  - Verification manifest persisted at [`configs/cifar10_split_manifest.json`](configs/cifar10_split_manifest.json) (<1 KB) storing SHA-256 partition checksums, PyTorch version, algorithm tags, and boundary verification samples.
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
