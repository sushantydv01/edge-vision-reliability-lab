# Edge Vision Reliability Lab - Project Status

## 1. System & Environment Baseline

- **Date Initialized**: 2026-11-03
- **Host Hardware**: Apple MacBook Pro (Apple M3)
- **Memory**: 8 GB Unified Memory (Available at inspection: ~1.23 GB, 84.7% utilized)
- **Storage**: ~161.90 GB available SSD space (out of 460.43 GB total)
- **OS**: macOS 27.0 (Darwin 27.0.0, arm64)
- **Python**: Python 3.12.16 (virtual environment at `.venv/bin/python`)
- **Git**: Git 2.47.0 (Tracking branch `main`)
- **Package Manager**: Homebrew 7.0.12 (host) / pip 26.2.2 (.venv)

### Core Dependency Manifest (Locked)
- `torch`: 2.15.0
- `torchvision`: 0.29.2
- `numpy`: 2.5.4
- `scipy`: 1.18.1
- `pandas`: 3.1.0
- `Pillow`: 13.0.0
- `PyYAML`: 6.0.3
- `tqdm`: 4.70.1
- `pytest`: 9.2.0
- `psutil`: 7.2.2
- `matplotlib`: 3.11.2
- `plotly`: 7.1.0
- `edge-vision-reliability-lab`: 0.1.0 (editable install from `src/edge_vision`)

---

## 2. Milestone Progress Tracker

### Milestone 0: Environment & Project Baseline
- [x] Workspace verification and environment inspection.
- [x] Initial directory skeleton creation.
- [x] Modern `.gitignore` configured for ML experiments and data exclusion.
- [x] Local Git repository initialized on branch `main`.
- [x] Concise `README.md` and `PROJECT_STATUS.md` created.
- [x] Python 3.12 virtual environment creation (`.venv`).
- [x] Dependency specification (`pyproject.toml` and reproducible `requirements-lock.txt`).
- [x] Verification script executed: CPU and Apple MPS tensor execution confirmed.

### Milestone 1: Data & Corruption Pipeline
- [x] Clean CIFAR-10 data loader with deterministic splits and lightweight view architecture (Task 2).
- [ ] Modular corruption pipeline (synthesizing CIFAR-10-C style perturbations: noise, blur, weather, digital) (Task 3).
- [ ] Robustness evaluation suite across corruption severities (severity 1-5).

### Milestone 2: Baseline Models (FP32)
- [x] Model architecture adaptations and factory (`MobileNetV3-Small`, `ResNet18`) (Task 2.1).
- [x] Baseline training infrastructure, reproducibility, and checkpointing (Task 2.2).
- [ ] Clean baseline accuracy verification on test split (Task 2.3).

### Milestone 3: Compression & Quantization
- [ ] FP16 conversion pipeline.
- [ ] INT8 Post-Training Quantization (PTQ) pipeline (PyTorch quantization; evaluate ONNX/CoreML in dedicated step).
- [ ] Quantization calibration dataset pipeline.
- [ ] Validation of quantized model output shapes and numeric sanity.

### Milestone 4: Benchmarking Engine
- [ ] Latency and throughput profiler (warmup runs, deterministic timing on CPU).
- [ ] Peak memory and disk size profiler.
- [ ] Robustness evaluator across corruption severities (severity 1-5).
- [ ] Structured metric output logging (CSV / JSON).

### Milestone 5: Analysis & Reporting
- [ ] Clean vs Corrupted accuracy delta analysis.
- [ ] Pareto frontier visualization (latency/size vs accuracy/robustness).
- [ ] Comprehensive final report and engineering recommendations.

---

## 3. Execution Devices & Engineering Limitations

1. **Primary Compute Device (CPU)**:
   - CPU is the default baseline for deterministic inference benchmarking, latency/throughput profiling, and INT8 quantization execution.
   - Verified basic tensor math and matmul on CPU with PyTorch 2.15.0.

2. **Secondary Acceleration (Apple Silicon MPS)**:
   - `torch.backends.mps.is_available()` and `is_built()` verified as `True`.
   - Small matrix multiplication verified on MPS device.
   - **Key Limitation**: MPS support in PyTorch currently does not natively support many INT8 quantized tensor operations or PyTorch FX quantization backends (which target `fbgemm` or `qnnpack` on CPU). Therefore, all quantization workflows and standard comparisons must run on CPU, with MPS evaluated only for supported FP32/FP16 pipelines.

3. **Memory Budget & Pressure**:
   - Host machine has 8 GB total unified memory, with ~1.23 GB free at test time.
   - Batch sizes must remain conservative (e.g. 32-64).
   - In-memory dataset caching must be bounded; batch iterators should stream data to avoid host memory exhaustion.

4. **Package Isolation**:
   - Explicitly built on Python 3.12.16 using `.venv`. System Python 3.14 is completely bypassed.

---

## 4. CIFAR-10 Data Pipeline Verification & Integrity Results

- **Acquisition Source**: Official CIFAR-10 dataset (MD5 `c58f30108f718f92721af3b95e74349a` verified against torchvision).
- **Data Integrity**:
  - Training Set: 50,000 images, shape `(50000, 32, 32, 3)`, dtype `uint8`.
  - Test Set: 10,000 images, shape `(10000, 32, 32, 3)`, dtype `uint8`.
  - Classes: Exactly 10 balanced classes (5,000 training images per class), labels 0 to 9.
- **Split Specification & Reproducibility**:
  - Deterministic seed: `42`.
  - Training Partition: 45,000 samples.
  - Validation Partition: 5,000 samples.
  - Partition disjointness and completeness: 100% verified (0 overlap, 50,000 union).
  - Compact manifest: [configs/cifar10_split_manifest.json](file:///Users/sushantydv01/Desktop/edge-vision-reliability-lab/configs/cifar10_split_manifest.json) (795 bytes) recording metadata, algorithm (`torch.randperm_v1`), PyTorch version (`2.15.0`), SHA-256 partition checksums, and boundary sample index checks.
  - Stale manifest protection: Changing seed automatically regenerates the split rather than reusing stale indices.
  - Checksum mismatch detection: Detects PRNG drift across environments or library upgrades and requires explicit regeneration via `force_regenerate=True`.
- **Memory Architecture**:
  - `TransformedSubset` implements a lightweight indexed view with on-the-fly transforms over a single shared underlying `base_dataset` instance. The 153.6 MB raw NumPy array is loaded once in memory and never duplicated across splits.
- **DataLoader Defaults**:
  - Batch size: `64` (Train: 704 batches, Val: 79 batches, Test: 157 batches).
  - Multi-worker count: `0` (conservative host memory policy).
  - Normalization: Mean `(0.4914, 0.4822, 0.4465)`, Std `(0.2470, 0.2435, 0.2616)`.
- **Artifacts Generated**:
  - Visual class grid: `reports/figures/cifar10_samples.png`.
- **Automated Tests**:
  - `pytest -v`: 23 passed (16 data pipeline unit tests, 4 dataset integration tests, 3 environment smoke tests).
- **Next Milestone Task**:
  - Milestone 2, Task 2.3: Full Baseline Training & Clean Test-Set Evaluation.

---

## 5. CIFAR-10 Model Architecture Layer Verification & Results (Task 2.1)

- **Adapted Architectures**:
  - `resnet18_cifar`:
    - Replaced ImageNet 7x7 stride-2 conv with 3x3 stride-1 pad-1 conv (no bias).
    - Replaced downsampling maxpool with `nn.Identity()`.
    - Replaced 1000-class fc head with `nn.Linear(512, num_classes)`.
    - Parameters: 11,173,962 (FP32 memory: 44.70 MB / 42.63 MiB).
    - Spatial progression: 32x32 -> 32x32 (stem, layer1) -> 16x16 (layer2) -> 8x8 (layer3) -> 4x4 (layer4) -> 1x1 (avgpool).
  - `mobilenet_v3_small_cifar`:
    - Adjusted stem conv stride from 2 to 1 (`features[0][0].stride = (1, 1)`).
    - Preserved all 11 inverted residual blocks and pre-pooling pointwise conv.
    - Replaced 1000-class projection in classifier with `nn.Linear(1024, num_classes)`.
    - Parameters: 1,528,106 (FP32 memory: 6.11 MB / 5.83 MiB).
    - Spatial progression: 32x32 -> 32x32 (features[0]) -> 16x16 (features[1]) -> 8x8 (features[2-3]) -> 4x4 (features[4-8]) -> 2x2 (features[9-12]) -> 1x1 (avgpool).
    - Prevents spatial collapse to 1x1 at stage 9, ensuring BatchNorm2d is stable in training mode with batch size 1 ($N \times H \times W = 4 > 1$).
- **Model Factory & Metadata**:
  - Factory: `get_model(name, num_classes=10)` with strict registry validation.
  - Metadata: Exact parameter counting (`total`, `trainable`, `non_trainable`) and memory footprint calculation (`bytes`, `megabytes_decimal`, `mebibytes_binary`).
- **Automated Verification**:
  - 22 dedicated model unit & integration tests covering structural modifications, forward shapes across batch sizes (1, 2, 64), train/eval mode stability with $B=1$, raw logit properties, intermediate stage shape locks, exact parameter/memory metrics, factory dispatch, seed reproducibility, and Apple Silicon MPS forward execution.

---

## 6. FP32 Baseline Training Engine & Empirical Validation (Task 2.2)

- **Engine Modules Implemented**:
  - `edge_vision.training.config`: Type-safe dataclass configurations with strict validation.
  - `edge_vision.training.seeding`: Multi-runtime PRNG seeding (`random`, `numpy`, `torch` CPU/MPS) and complete state capture/restoration including `DataLoader` generator state.
  - `edge_vision.training.metrics`: Sample-weighted loss aggregation and top-1 accuracy accumulator (`MetricTracker`, `EpochMetrics`).
  - `edge_vision.training.checkpoint`: Atomic checkpoint persistence (`_save_atomic`), resume validation, and `best_checkpoint.pt` / `latest_checkpoint.pt` tracking.
  - `edge_vision.training.experiment`: Structured file-based logging (`metrics_history.csv` and `experiment_summary.json`).
  - `edge_vision.training.data`: Training DataLoader factory enforcing training-only augmentation (RandomCrop pad=4, RandomHorizontalFlip p=0.5), unaugmented validation transform, and strict test-set isolation (never returns `test_loader`).
  - `edge_vision.training.trainer`: Training and validation loops, `zero_grad(set_to_none=True)`, gradient clipping, scheduler stepping, accelerator synchronization, and test-set isolation assertions.
- **Hardware Smoke Test (Pilot A)**:
  - Verified CPU and Apple Silicon MPS execution across both models.
  - Checkpoint file sizes with SGD momentum: ResNet-18 is 89.49 MB; MobileNetV3-Small is 12.40 MB.
  - MPS demonstrated $1.3\times$ to $2.1\times$ faster training throughput and $6.5\times$ to $7.8\times$ faster validation throughput than CPU.
- **5-Epoch Convergence Pilot Results (Pilot B on MPS)**:
  - **`resnet18_cifar`**:
    - Train Loss: `1.6564` (Ep 1) $\to$ `0.6347` (Ep 5)
    - Train Accuracy: `39.43%` $\to$ `78.00%`
    - Val Loss: `1.2251` $\to$ `0.7441`
    - Val Accuracy: `55.68%` $\to$ **`75.36%`** (Peak at Epoch 4)
    - Throughput: ~450–476 samples/second; Epoch duration: ~97–103 seconds.
  - **`mobilenet_v3_small_cifar`**:
    - Train Loss: `1.7123` (Ep 1) $\to$ `1.0166` (Ep 5)
    - Train Accuracy: `35.15%` $\to$ `63.83%`
    - Val Loss: `1.8426` $\to$ `1.0874`
    - Val Accuracy: `30.34%` $\to$ **`60.82%`** (Peak at Epoch 5)
    - Throughput: ~1,170–1,695 samples/second; Epoch duration: ~27–39 seconds.
  - **Convergence Verdict**: Monotonic downward training loss progression, smooth gradient updates, no NaN/Inf occurrences, and healthy validation trajectories confirm the training loop is ready for full-schedule baseline experiments.
- **Automated Verification**:
  - `pytest -v`: 67 passed out of 67 tests (23 data/env + 22 models + 22 training engine tests).
  - Validated bitwise epoch-boundary resume equivalence, sample-weighted loss aggregation, and test-set isolation.
