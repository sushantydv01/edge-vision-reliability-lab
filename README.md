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

---

## Inference & Benchmarking (Milestone 4)

A unified inference subsystem and benchmarking harness for executing and profiling all six CIFAR-10 model/precision variants.

### Environment & Hardware Assumptions
- **Python**: 3.12 (virtual environment at `.venv`)
- **PyTorch**: 2.5+ with Torchvision and Pillow installed
- **Hardware Profile**: Apple Silicon Mac (M3, 8 GB unified memory); CPU and MPS supported

### Supported Model & Device Configurations

| Architecture | Precision | Backend | Supported Devices | Operational Notes |
| :--- | :--- | :--- | :--- | :--- |
| `resnet18_cifar` | `fp32` | PyTorch Eager | CPU, MPS | Canonical baseline |
| `resnet18_cifar` | `fp16` | PyTorch Eager | MPS (preferred), CPU | Observed substantial CPU slowdown compared to FP32 |
| `resnet18_cifar` | `int8` | TorchScript | CPU only | Strictly CPU-bound via QNNPACK backend |
| `mobilenet_v3_small_cifar` | `fp32` | PyTorch Eager | CPU, MPS | Inverted residual baseline |
| `mobilenet_v3_small_cifar` | `fp16` | PyTorch Eager | MPS (preferred), CPU | Observed substantial CPU slowdown compared to FP32 |
| `mobilenet_v3_small_cifar` | `int8` | TorchScript | CPU only | Strictly CPU-bound via QNNPACK backend |

> [!NOTE]
> INT8 models cannot run on Apple Silicon MPS because PyTorch does not implement quantized operators on MPS. Attempting to select `mps` for `int8` raises an explicit `DeviceCompatibilityError`.

### CLI Usage: Single Image Inference

```bash
# Basic single image inference on Apple Silicon MPS
python scripts/infer.py --image path/to/image.png --model resnet18_cifar --precision fp16 --device mps --top-k 5

# Enable bilinear resizing for non-32x32 images
python scripts/infer.py --image path/to/high_res.jpg --model mobilenet_v3_small_cifar --precision fp32 --resize

# Output machine-readable JSON (human messages routed to stderr)
python scripts/infer.py --image path/to/image.png --model resnet18_cifar --json --output result.json
```

### CLI Usage: Batch Directory Inference

Processes image directories in deterministic sorted order with bounded batching to prevent unbounded memory consumption:

```bash
# Process directory of images in batches of 32
python scripts/infer.py --dir path/to/images/ --model resnet18_cifar --precision fp32 --batch-size 32

# With explicit error recording policy (continues on unreadable files and logs them)
python scripts/infer.py --dir path/to/images/ --model mobilenet_v3_small_cifar --precision int8 --on-error record --output batch_results.json
```

### Resize Behavior
- **Default (`--resize` omitted)**: Strictly rejects images whose spatial dimensions are not $32\times 32$, raising an explicit `InvalidInputError`.
- **Enabled (`--resize` supplied)**: Applies deterministic bilinear interpolation with antialiasing to scale arbitrary image dimensions to $32\times 32$ before canonical CIFAR-10 normalization.

### Benchmark Runner

Reproducibly profiles model load latency, steady-state batch throughput, and component latency breakdowns:

```bash
# Quick sanity smoke test (batch sizes 1 and 8, 2 repetitions)
python scripts/benchmark_inference.py --smoke-test --output-dir experiments/inference_benchmarks/smoke_run

# Comprehensive benchmark across all 6 model/precision variants and batch sizes (1, 8, 32, 64)
python scripts/benchmark_inference.py \
  --models resnet18_cifar mobilenet_v3_small_cifar \
  --precisions fp32 fp16 int8 \
  --batch-sizes 1,8,32,64 \
  --warmup 5 \
  --repetitions 20 \
  --output-dir experiments/inference_benchmarks/run_full
```

#### Benchmark Output Structure
Results are saved to:
- `benchmark_summary.csv`: Tabular CSV with median, mean, p95, per-image latency, total engine throughput (FPS), forward-path latency, forward-only throughput (FPS), and load times.
- `benchmark_results.json`: Comprehensive machine-readable report containing platform/PyTorch versions, Git commit hash, artifact checksums, random seeds, and raw measurements.

### Benchmark Methodology & Performance Analysis

#### 1. CPU Threading Configuration & Sensitivity
- **Runtime Defaults**: Task 4.4 benchmark runs were conducted under unpinned PyTorch runtime defaults on Apple Silicon M3 (`torch.get_num_threads() == 4`, `torch.get_num_interop_threads() == 8`). Thread counts were not explicitly pinned or restricted by the benchmark runner.
- **Thread Sensitivity**: CPU inference performance—most notably single-image (batch-1) latency for lightweight architectures like MobileNetV3-Small—is highly sensitive to thread configuration. On lightweight architectures with small tensor footprints, multi-threading overhead can dominate execution time at small batch sizes.

#### 2. Comparisons with Prior Benchmarks (Task 5.3C)
- **Harness Differences**: Prior CPU benchmarks in Task 5.3C executed in isolated child processes with explicitly pinned thread counts (`threads=1, 2, 4`), whereas Task 4.4 used a single shared process under default threading (`threads=4`, `interop_threads=8`).
- **Consistency**: Task 4.4's MobileNetV3-Small FP32 batch-1 forward latency (~25.7 ms) is consistent with Task 5.3C's 4-thread result (~24.2 ms), rather than its 1-thread result (~4.0 ms).
- **Non-Interchangeability**: Thread count is not conclusively proven as the sole cause of every difference across benchmark suites. Process isolation, harness warm-up, timing scopes, memory allocator behavior, and background host activity also differ; measurements from different harnesses are not directly interchangeable.

#### 3. Empirical Observations vs. Causal Hypotheses
- **Thread-Pool Overhead**: Thread-pool synchronization and scheduling overhead are plausible hypotheses for multi-threaded batch-1 latency degradation on small networks, but they remain hypotheses rather than definitively proven hardware mechanisms.
- **CPU FP16 Slowdown**: ResNet-18 FP16 on CPU exhibits severe throughput degradation (~13.6 FPS at batch 64 compared to ~207.8 FPS for FP32). This is documented strictly as an empirical observation under PyTorch 2.5 on Apple Silicon ARM64 CPU. Hypotheses such as missing vector GEMM kernels or scalar software emulation are unverified without low-level hardware performance counter traces and are not asserted as established causes.
- **Separation of Concerns**: In all project documentation, measured empirical findings are kept strictly separated from explanatory hypotheses.

#### 4. Throughput & Latency Interpretation
- **Total Engine Throughput vs. Forward-Only Throughput**: Headline throughput figures reported in the benchmark (`throughput_fps`) reflect **total engine throughput** calculated from median total latency (`median_total_ms`), not forward-only execution:
  - For MobileNetV3-Small INT8 at batch 64 on Apple M3 CPU (4 intra-op threads, QNNPACK), median total engine latency is 8.253 ms, resulting in headline total engine throughput of **7,754.8 images/sec**.
  - Total engine latency encompasses the entire in-memory inference pipeline: image preprocessing (`preprocess_batch_images`: tensor conversion, NCHW permutation, scaling, and CIFAR-10 normalization), forward-path execution, device synchronization (where applicable), CPU transfer, and prediction formatting (softmax and top-$k$ extraction).
  - By contrast, **forward-only throughput** (`forward_only_throughput_fps`) is derived from **forward-path latency** (`median_forward_ms`): this measures model forward execution plus applicable input tensor casting/device transfer (`_prepare_tensor_for_forward`) and pipeline synchronization (`torch.mps.synchronize()`). For MobileNetV3-Small INT8 at batch 64, median forward-path latency is **5.550 ms**, yielding forward-only throughput of **11,531.5 images/sec**.
  - Similarly, for MobileNetV3-Small FP16 on MPS at batch 64, total engine throughput is **5,986.3 images/sec** (median total latency 10.691 ms), while forward-only throughput reaches **8,408.9 images/sec** (median forward-path latency **7.611 ms**). For ResNet-18 FP16 on MPS at batch 64, total engine throughput is **2,128.8 images/sec** (30.064 ms), while forward-only throughput is **2,357.7 images/sec** (median forward-path latency **27.145 ms**).
- **In-Memory Input vs. Pre-Allocated Tensors**: The benchmark operates on in-memory synthetic image batches (NumPy `uint8` arrays of shape $(B, 32, 32, 3)$) rather than pre-allocated normalized tensors, thereby exercising the complete preprocessing, execution, and postprocessing pipeline of the engine.
- **Exclusion of Disk and Camera I/O**: Both total engine throughput and forward-only throughput are strictly **in-memory benchmarks**. Neither metric represents end-to-end disk/camera pipeline throughput, where storage read/write, file decoding (Pillow/libjpeg from compressed JPEG/PNG bytes), and camera frame acquisition form the primary latency bottleneck.
- **INT8 Is Not Universally Faster**: Quantization does not universally outperform full precision across all hardware and models:
  - In Task 5.3C single-thread testing (`threads=1`, `batch_size=1`), ResNet-18 INT8 was substantially slower than FP32: **10.937 ms** median forward latency for INT8 vs. **4.805 ms** for eager FP32 (and **4.414 ms** for TorchScript FP32) across 100 timed runs in isolated child processes.
  - On Apple Silicon, FP32 on MPS (**1,372.4 images/sec** total engine throughput at batch 64, 46.634 ms) substantially outperforms INT8 on CPU (**333.3 images/sec** total engine throughput at batch 64, 192.020 ms).
  - INT8 speedups are contingent on operator fusion, backend implementation (QNNPACK), batch size, and target hardware architecture.

#### 5. Sample-Size Limitation on Tail Latency (p95)
- **Statistical Limitation**: The benchmark runner computes and reports 95th-percentile (p95) latency via linear interpolation across 10 repetitions per configuration. However, with a sample size of $N = 10$, this metric is sample-limited and cannot be interpreted as a high-confidence tail-latency or SLA guarantee.
- **Future Formal Profiling**: Formal tail-latency characterization or SLA profiling should execute at least 30 to 100 repetitions under strict OS-level isolation.

### REST Inference API (Task 4.5)

A high-performance local REST API built with FastAPI that exposes the unified inference engine and artifact registry to client applications.

#### Architecture & Design
- **Dedicated Package**: Located in [`src/edge_vision/api/`](file:///Users/sushantydv01/Desktop/edge-vision-reliability-lab/src/edge_vision/api/).
- **Decoupled Application Factory**: `create_app()` instantiates and configures the FastAPI application independently of the server runner, enabling isolated unit testing without socket binding.
- **Model Lifecycle & Thread Safety (`ModelManager`)**:
  - Models are loaded into memory once and reused across requests.
  - Model switching (`POST /models/load`) is atomic: the target model is loaded and verified in a temporary container before acquiring a thread lock (`threading.RLock`) to swap the resident engine pointer. If loading fails, the active model remains resident and unharmed.
  - Concurrent inference and switching are serialized via reentrant locking, preventing race conditions or corrupted internal tensor states.
- **Error Sanitization**: Error messages strip internal host filesystem paths and tracebacks to prevent information disclosure.
- **Zero Duplicate Registries**: Directly leverages `ArtifactRegistry` and `InferenceEngine` abstractions.

#### Server Launch

```bash
# Launch server bound to localhost on port 8000
uvicorn edge_vision.api.main:app --host 127.0.0.1 --port 8000

# Or run entrypoint directly
python -m edge_vision.api.main --host 127.0.0.1 --port 8000
```

> [!NOTE]
> The server binds strictly to localhost (`127.0.0.1`) by default for local development and demonstration. Interactive OpenAPI documentation is accessible at `http://127.0.0.1:8000/docs`.

#### Endpoints & Example Curl Commands

##### 1. Service Health (`GET /health`)
Does not trigger expensive model loading.

```bash
curl -s http://127.0.0.1:8000/health
```

Example response (model loaded):
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "resnet18_cifar",
  "precision": "fp32",
  "device": "cpu",
  "backend": "pytorch_eager"
}
```

##### 2. List Supported Models (`GET /models`)
Queries `ArtifactRegistry` without loading checkpoints.

```bash
curl -s http://127.0.0.1:8000/models
```

##### 3. Get Active Model Metadata (`GET /models/current`)

```bash
curl -s http://127.0.0.1:8000/models/current
```

##### 4. Load or Switch Model (`POST /models/load`)

```bash
# Load MobileNetV3-Small INT8 on CPU
curl -s -X POST http://127.0.0.1:8000/models/load \
  -H "Content-Type: application/json" \
  -d '{"model_name": "mobilenet_v3_small_cifar", "precision": "int8", "device": "cpu"}'

# Load ResNet-18 FP16 on Apple Silicon MPS
curl -s -X POST http://127.0.0.1:8000/models/load \
  -H "Content-Type: application/json" \
  -d '{"model_name": "resnet18_cifar", "precision": "fp16", "device": "mps"}'
```

##### 5. Single Image Prediction (`POST /predict`)
Uploads a single image via `multipart/form-data`. Supports optional `resize` and `top_k` query parameters.

```bash
# Single image inference with top-3 predictions and automatic resizing enabled
curl -s -X POST "http://127.0.0.1:8000/predict?resize=true&top_k=3" \
  -F "file=@path/to/sample.png"
```

Example response:
```json
{
  "status": "success",
  "filename": "sample.png",
  "prediction": {
    "predicted_class": "frog",
    "class_index": 6,
    "confidence": 0.8924,
    "probabilities": { "airplane": 0.0012, "frog": 0.8924, "deer": 0.0541 },
    "logits": [ -2.14, 5.82, 0.41, 0.12, -0.98, -1.22, 6.74, 1.25, -3.11, -0.55 ],
    "top_k": [
      { "rank": 1, "class_name": "frog", "confidence": 0.8924 },
      { "rank": 2, "class_name": "deer", "confidence": 0.0541 },
      { "rank": 3, "class_name": "bird", "confidence": 0.0215 }
    ]
  },
  "timing_ms": {
    "preprocess_ms": 1.245,
    "forward_ms": 4.882,
    "postprocess_ms": 0.154,
    "total_ms": 6.281
  },
  "model_name": "mobilenet_v3_small_cifar",
  "precision": "int8",
  "device": "cpu"
}
```

##### 6. Bounded Batch Prediction (`POST /predict/batch`)
Uploads multiple images in one multipart request, strictly preserving input ordering.

```bash
curl -s -X POST "http://127.0.0.1:8000/predict/batch?resize=true&top_k=5" \
  -F "files=@image1.png" \
  -F "files=@image2.png" \
  -F "files=@image3.png"
```

#### Upload Constraints & Safety Limits
- **Maximum File Size**: 10 MB per image upload (`MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024`).
- **Bounded Streaming Uploads**: File content is read asynchronously in 64 KB chunks (`CHUNK_SIZE_BYTES = 64 * 1024`). Reading halts immediately once the 10 MB ceiling is breached to prevent buffer accumulation.
- **Maximum Spatial Dimensions**: Rejects images where either spatial dimension exceeds 2048 pixels (`MAX_IMAGE_DIMENSION = 2048`) or is non-positive before allocating complete raster arrays.
- **Aggregate Batch Ceiling**: Bounded to 64 MB total across all files in a single batch request (`MAX_AGGREGATE_BATCH_BYTES = 64 * 1024 * 1024`), enforced per-chunk during ingestion.
- **Bounded Batch Count**: Up to 64 images per batch request (`MAX_BATCH_SIZE = 64`). Requests exceeding 64 images are rejected with HTTP 400.
- **Supported Formats**: PNG, JPG, JPEG, BMP, WEBP, TIFF.
- **Resource Management**: File handles are closed asynchronously in `finally:` blocks. Image tensors and PIL objects are unreferenced immediately after forward execution to prevent heap bloat.
- **Spooling Note**: Bounded chunk reading protects application heap memory against excessive allocations; however, Starlette's multipart parser spools requests before endpoint execution. Complete HTTP wire-level protection against oversized payloads requires an edge reverse proxy.

### Key Operational Limitations
1. **Device Boundaries**: INT8 post-training quantized artifacts execute exclusively on CPU via QNNPACK; Apple Silicon MPS does not support PyTorch quantized operators.
2. **CPU FP16 Performance**: In PyTorch on ARM64 CPU, FP16 inference exhibits substantial empirical slowdown compared to FP32. For accelerated FP16 inference, Apple Silicon MPS is strongly recommended.
3. **Platform Portability**: CPU benchmark results measured on Apple Silicon ARM64 reflect Apple M3 microarchitectural characteristics and do not generalize to x86 or server processors.
