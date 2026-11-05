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
- [x] Deterministic synthetic image corruption suite (Task 3): 6 corruption operations (Gaussian noise, Gaussian blur, motion blur, brightness, contrast, JPEG compression) with 5 severity levels, SHA-256 seed derivation, pure uint8 operations, registry dispatch, and lightweight CorruptedDataset adapter.
- [x] Robustness evaluation engine across corruption severities (Task 4): Shared-batch identical tensor presentation, dual-model streaming evaluation, sample-weighted metric tracking, degradation formulas, and atomic multi-format reporting.

### Milestone 2: Baseline Models (FP32)
- [x] Model architecture adaptations and factory (`MobileNetV3-Small`, `ResNet18`) (Task 2.1).
- [x] Baseline training infrastructure, reproducibility, and checkpointing (Task 2.2).
- [x] Standalone clean evaluation infrastructure, audit corrections, and baseline configuration readiness (Task 2.3A).
- [x] Controlled full baseline training (100 epochs) and clean test-set evaluation (Task 2.3B).

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
  - Compact manifest: [`configs/cifar10_split_manifest.json`](configs/cifar10_split_manifest.json) (795 bytes) recording metadata, algorithm (`torch.randperm_v1`), PyTorch version (`2.15.0`), SHA-256 partition checksums, and boundary sample index checks.
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

---

## 7. Standalone Clean Evaluation Infrastructure & Full-Training Readiness (Task 2.3A)

- **Evaluation Architecture & Modules Implemented**:
  - `edge_vision.evaluation.metrics`:
    - `EvaluationMetrics`: Structured dataclass storing loss, overall top-1 accuracy, per-class counts, per-class accuracy, and a $10 \times 10$ confusion matrix.
    - `EvaluationTracker`: Online sample-weighted cross-entropy loss aggregator, multi-class prediction accumulator, and confusion matrix builder.
    - Mathematical assertions: Enforces $\sum_{i,j} C_{i,j} = N$ and $\text{trace}(C) = N_{\text{correct}}$.
  - `edge_vision.evaluation.evaluator`:
    - `resolve_evaluation_device()`: Hardware selector prioritizing Apple Silicon MPS over CPU.
    - `load_checkpoint_for_evaluation()`: Reconstructs model architecture from checkpoint metadata or explicit parameters, verifies model type, and enforces strict weight loading (`strict=True`).
    - `get_evaluation_dataloader()`: Builds an isolated DataLoader using strictly unaugmented transforms (canonical normalization only, no random crops or flips).
    - `evaluate_model()`: Runs evaluation loop under `model.eval()`, frozen weights (`requires_grad=False`), and `torch.inference_mode()`.
    - `save_evaluation_report()`: Atomically serializes evaluation results to JSON (`.tmp.<pid>` renamed via `os.replace`).
  - `scripts/evaluate_checkpoints.py`:
    - Standalone CLI runner supporting `--checkpoint`, `--split` (`test` or `val`), `--device` (`auto`, `cpu`, `mps`), `--batch-size`, and `--output`.
    - Produces both formatted CLI summary tables and structured JSON reports.
- **Automated Verification**:
  - 11 dedicated evaluation tests in `tests/test_evaluation.py` covering:
    - Empty tracker edge cases.
    - Multi-batch sample-weighted loss and confusion matrix mathematical invariants.
    - Strict checkpoint state dict loading and missing/mismatched key rejection.
    - Eval mode and parameter freeze enforcement.
    - Bitwise CPU vs MPS prediction equivalence on synthetic batches.
    - Atomic JSON serialization and schema validity.
    - Test-set isolation (refusal to evaluate unapproved partitions).
  - Full project test suite: **78 passed out of 78 tests** in ~13.5 seconds.
- **Audit & Dossier Corrections**:
  - Completed technical audit corrections in `reports/PROJECT_TECHNICAL_REVIEW.md`:
    1. Zero-copy semantics: Clarified that `TransformedSubset` shares the underlying 153.6 MB NumPy array, while PIL conversions and tensor transforms allocate dynamically per sample.
    2. PIL bottleneck: Labeled single-threaded PIL overhead as an unverified hypothesis pending formal cProfile/PyTorch profiler traces.
    3. MobileNet convergence: Clarified that 5 epochs are insufficient to determine asymptotic convergence causes and confirmed that the controlled baseline recipe mandates identical hyperparameters.
    4. INT8 hardware acceleration: Clarified that while PyTorch's native integer quantization backends (`fbgemm`/`qnnpack`) run in software on CPU, external runtimes (CoreML on Apple Neural Engine, ONNX Runtime, TensorRT) provide dedicated INT8 edge acceleration.
    5. CIFAR-10-C taxonomy: Clarified the distinction between official pre-generated CIFAR-10-C benchmark datasets and programmatic in-repo corruptions following the CIFAR-10-C taxonomy.
    6. `src/__init__.py`: Noted as a redundant container marker, harmless for setuptools packaging.
    7. Professional tone: Aligned documentation tone with objective, evidence-based engineering standards.
- **Controlled Baseline Configuration Readiness**:
  - Verified `configs/train_baseline.yaml`:
    - SGD optimizer: lr=0.05, momentum=0.9, nesterov=false, weight_decay=0.0005.
    - Scheduler: CosineAnnealingLR (T_max=100, eta_min=0.0001, warmup_epochs=0).
    - Loss: CrossEntropyLoss.
    - Augmentation: RandomCrop (size=32, padding=4), RandomHorizontalFlip (p=0.5).
    - Dataloader: batch_size=64, num_workers=0, pin_memory=false.
    - Checkpointing: save_best=true (by val_acc), save_latest=true.
- **Empirical Resource & Runtime Projections for Full 100-Epoch Training**:
  - **`resnet18_cifar`**:
    - Measured per-epoch duration (MPS): ~100.6 seconds.
    - Estimated 100-epoch training time: **~2.8 hours** (168 minutes).
    - Checkpoint storage footprint: 89.49 MB per checkpoint (2 checkpoints = 178.98 MB total).
  - **`mobilenet_v3_small_cifar`**:
    - Measured per-epoch duration (MPS): ~34.9 seconds.
    - Estimated 100-epoch training time: **~58 minutes**.
    - Checkpoint storage footprint: 12.40 MB per checkpoint (2 checkpoints = 24.80 MB total).
  - **Total Sequential Run Time**: **~3.8 hours**.
  - **Total Disk Space Needed**: **< 250 MB**.
  - **Execution Recommendation**: Run sequentially on Apple Silicon MPS to maintain low host memory footprint and avoid macOS unified memory swap thrashing.

---

## 8. Controlled FP32 Baseline Experiments & Independent Audit (Task 2.3B)

- **Execution Environment**:
  - Host: Apple MacBook Pro (Apple M3, 8 GB Unified Memory, macOS 27.0).
  - Runtime: Python 3.12.16, PyTorch 2.15.0, Torchvision 0.29.2.
  - Device: Apple Silicon MPS (`torch.backends.mps.is_available() == True`).
  - Execution Policy: Strictly sequential training (ResNet-18 followed by MPS cache clearance and MobileNetV3-Small).
  - Total Sequential Run Time: **3.40 hours** (203.7 minutes total wall-clock).

- **Controlled Baseline Recipe (`configs/train_baseline.yaml`)**:
  - Seed: `42` (PRNG initialized across Python, NumPy, PyTorch CPU, and MPS).
  - Epochs: `100` per architecture.
  - Batch size: `64` ($N_{\text{train\_batches}}=704$, $N_{\text{val\_batches}}=79$, $N_{\text{test\_batches}}=157$).
  - Optimizer: SGD ($lr=0.05$, $\text{momentum}=0.9$, $\text{nesterov}=\text{false}$, $\text{weight\_decay}=5\times 10^{-4}$).
  - Scheduler: CosineAnnealingLR ($T_{\max}=100$, $\eta_{\min}=0.0001$, $\text{warmup}=0$).
  - Augmentation: RandomCrop ($32\times 32$, padding 4) + RandomHorizontalFlip ($p=0.5$).
  - Loss: CrossEntropyLoss (sample-weighted batch aggregation).

### Measured Experimental Results & Independent Numerical Audit

| Experimental Metric | ResNet-18 CIFAR Adaptation (`resnet18_cifar`) | MobileNetV3-Small CIFAR Adaptation (`mobilenet_v3_small_cifar`) | Relative Delta / Ratio |
| :--- | :--- | :--- | :--- |
| **Experiment Run ID** | `exp_baseline_resnet18_seed42` | `exp_baseline_mobilenet_v3_small_seed42` | Paired baseline |
| **Model Parameters** | 11,173,962 (all trainable) | 1,528,106 (all trainable) | **7.31× fewer parameters** |
| **Theoretical FP32 Weights** | 44.70 MB (42.63 MiB) | 6.11 MB (5.83 MiB) | **7.32× smaller weights** |
| **Checkpoint Size on Disk** | 89.51 MB (85.36 MiB; 89,510,229 B) | 12.42 MB (11.85 MiB; 12,420,537 B) | **7.21× smaller checkpoint** |
| **Completed Epochs** | 100 / 100 (complete sequence) | 100 / 100 (complete sequence) | 100% completed |
| **Mean Epoch Duration** | 94.82 seconds (~499 samples/sec) | 27.10 seconds (~1,720 samples/sec) | **3.50× faster per epoch** |
| **Total Wall-Clock Time** | 2.64 hours (9,494.81s) | 0.76 hours (2,719.32s) | **3.49× faster overall** |
| **Best Validation Accuracy** | **95.50%** (4,775 / 5,000) | **86.86%** (4,343 / 5,000) | **-8.64% validation gap** |
| **Best Validation Epoch** | **Epoch 97** | **Epoch 98** | Late-stage stabilization |
| **Best Validation Loss** | 0.1814 (Ep 98); 0.1825 (at Ep 97) | 0.3649 (at Ep 98) | +0.1824 |
| **Final Train Loss (Ep 100)**| 0.002944 (Train Acc: 99.96%) | 0.307454 (Train Acc: 89.42%) | — |
| **Final Val Loss (Ep 100)** | 0.187413 (Val Acc: 95.42%) | 0.366877 (Val Acc: 86.66%) | — |
| **Held-Out Test Accuracy** | **94.78%** (9,478 / 10,000) | **86.51%** (8,651 / 10,000) | **-8.27% test gap** |
| **Held-Out Test Loss** | **0.2003** | **0.3896** | +0.1893 |
| **Train $\to$ Val Gap (Ep 100)**| **4.54%** (99.96% vs 95.42%) | **2.76%** (89.42% vs 86.66%) | Capacity-driven memorization |
| **Val $\to$ Test Gap (Best)** | **0.72%** (95.50% vs 94.78%) | **0.35%** (86.86% vs 86.51%) | Strong split consistency |

### Key Analytical Findings & Scientific Interpretations

1. **Late-Stage Stabilization vs. Asymptotic Claims**:
   - Both architectures demonstrated smooth convergence transitioning to late-stage stabilization after Epoch 80 as the learning rate decayed below $10^{-3}$ towards $\eta_{\min} = 10^{-4}$.
   - Validation loss and accuracy stabilized in narrow bands across the final 10 epochs (ResNet-18: 95.10%–95.50%; MobileNetV3: 86.30%–86.86%).
2. **Generalization Disparities & Overfitting Dynamics**:
   - **ResNet-18**: Training loss decreased monotonically to $0.0029$ with training accuracy reaching $99.96\%$ by Epoch 100, while validation accuracy stabilized at $95.42\%$ (peak $95.50\%$). This resulted in an observed **training-to-validation generalization gap of $4.54\%$** ($4.46\%$ at peak), reflecting significant empirical fitting of the 45,000-sample training set. The **validation-to-test performance difference was $0.72\%$** ($95.50\% \to 94.78\%$), demonstrating tight alignment between the validation partition and the held-out test distribution.
   - **MobileNetV3-Small**: The training and validation trajectories tracked each other consistently across the 100 epochs without displaying obvious late-stage divergence in loss or accuracy. At Epoch 100, training accuracy reached $89.42\%$ (train loss $0.3075$) while validation accuracy reached $86.66\%$ (val loss $0.3669$), representing an observed **training-to-validation generalization gap of $2.76\%$** ($2.56\%$ at the Epoch 98 peak). The **validation-to-test performance difference was $0.35\%$** ($86.86\% \to 86.51\%$). While these observations indicate stable generalization under the chosen regularization (RandomCrop, HorizontalFlip, weight decay $5\times 10^{-4}$), absence of overfitting cannot be claimed conclusively based on low validation-to-test delta alone, as capacity limits under the shared SGD recipe also bounded training set fitting.
3. **Per-Class Vulnerability & Failure Modes**:
   - Both networks performed best on rigid man-made transportation classes: `automobile` (97.8% vs 93.1%), `ship` (97.4% vs 92.3%), and `truck` (96.2% vs 91.9%).
   - The largest performance drop occurred on fine-textured natural categories, particularly **`cat`**:
     - ResNet-18 cat accuracy: 88.80% (55 cats misclassified as dogs).
     - MobileNetV3-Small cat accuracy: **70.40%** (124 cats misclassified as dogs).
     - Parameter-efficient depthwise separable convolutions exhibited disproportionate difficulty distinguishing fine fur and limb geometry between feline and canine classes.
4. **Memory & Footprint Distinctions**:
   - **Parameter Memory**: Theoretical FP32 weight tensors occupy 44.70 MB (ResNet-18) vs 6.11 MB (MobileNetV3).
   - **Serialized Checkpoint File Size**: Checkpoints on disk store model weights, optimizer momentum buffers (which duplicate weight shapes in FP32), scheduler states, and PRNG states, totaling 89.51 MB (85.36 MiB) for ResNet-18 and 12.42 MB (11.85 MiB) for MobileNetV3.
   - **Runtime Working Memory**: Process RSS remained bounded at 240–298 MB for ResNet-18 and 440–512 MB for MobileNetV3 on Apple Silicon MPS, with zero swap thrashing.
5. **Methodological Caveats & Limitations**:
   - **Single Seed**: Results reflect an initial single random seed (seed 42). While PRNG seeds are deterministically verified, multi-seed trials (e.g. seeds 42, 123, 999) are required to establish statistical error bars.
   - **Resolution Scale**: Measured on $32\times 32$ CIFAR-10 images; findings do not directly generalize to high-resolution edge camera inputs without multi-scale receptive field analysis.
   - **Optimizer Sensitivity Hypothesis**: The 8.27% clean test accuracy gap under the shared SGD recipe is an empirical observation. Whether MobileNetV3-Small's convergence is constrained by SGD weight decay or would improve under AdamW with warmup remains an unverified hypothesis to be investigated in exploratory Track 2 studies.

### Generated Artifacts & Visualizations

- **Learning Curve Figures** (`reports/figures/`):
  - `baseline_resnet18_loss.png`: Training vs. validation loss for ResNet-18 across 100 epochs.
  - `baseline_resnet18_accuracy.png`: Training vs. validation accuracy for ResNet-18 across 100 epochs.
  - `baseline_mobilenet_loss.png`: Training vs. validation loss for MobileNetV3-Small across 100 epochs.
  - `baseline_mobilenet_accuracy.png`: Training vs. validation accuracy for MobileNetV3-Small across 100 epochs.
  - `baseline_learning_rate.png`: CosineAnnealingLR progression from 0.05 to 0.0001 over 100 epochs.
  - `baseline_validation_comparison.png`: Direct epoch-by-epoch validation accuracy comparison between both architectures.
  - `baseline_per_class_test_accuracy.png`: Per-class test accuracy comparison across all 10 CIFAR-10 categories.
- **Checkpoints**:
  - `experiments/exp_baseline_resnet18_seed42/checkpoints/best_checkpoint.pt` (Epoch 97)
  - `experiments/exp_baseline_mobilenet_v3_small_seed42/checkpoints/best_checkpoint.pt` (Epoch 98)
- **Evaluation Reports**:
  - `experiments/exp_baseline_resnet18_seed42/evaluation_test.json`
  - `experiments/exp_baseline_mobilenet_v3_small_seed42/evaluation_test.json`
  - `reports/experiment_results/eval_test_resnet18.json`
  - `reports/experiment_results/eval_test_mobilenet.json`
- **Execution & Audit Scripts**:
  - `scripts/train_baseline.py`: Standalone CLI training script.
  - `scripts/run_sequential_baseline.py`: Sequential pipeline runner.
  - `scripts/generate_baseline_plots.py`: Visualization generator.
  - `scripts/audit_baseline_results.py`: Independent numerical audit script.

---

## 8. Milestone 1, Task 3: Deterministic Synthetic Image Corruption Suite

### Overview & Scientific Objective
To evaluate edge model robustness under controlled realistic environmental perturbations, a custom deterministic corruption engine was implemented in `src/edge_vision/corruptions/`. The engine generates reproducible image degradations inspired by common benchmark taxonomies (e.g., Hendrycks & Dietterich 2019) across 6 corruption types and 5 severity levels.

> **Important Scientific Scope**: This is an independent, custom deterministic synthetic image corruption suite tailored for CIFAR-10 resolution ($32\times 32\times 3$), not an official CIFAR-10-C release. Model robustness evaluation against these corruptions will be executed in subsequent benchmarking tasks.

### Supported Corruptions & Implemented Severity Levels

| Corruption Name | Parameter | Severity 1 | Severity 2 | Severity 3 | Severity 4 | Severity 5 | Description & Operational Details |
| :--- | :--- | ---:| ---:| ---:| ---:| ---:| :--- |
| `gaussian_noise` | $\sigma_{\text{norm}}$ ($[0, 1]$) | 0.04 | 0.06 | 0.08 | 0.10 | 0.12 | Zero-mean additive Gaussian noise; scaled by 255.0 to uint8 pixel range. |
| `gaussian_blur` | $\sigma$ (pixels) | 0.3 | 0.5 | 0.7 | 0.9 | 1.1 | Spatial 2D Gaussian filter (`scipy.ndimage.gaussian_filter`, reflect mode, per-channel). |
| `motion_blur` | $K$ (pixels) | 3 | 5 | 7 | 9 | 11 | Horizontal 1D linear motion blur ($1/K$ averaging kernel, reflect mode). |
| `brightness` | factor | 0.90 | 0.75 | 0.60 | 0.45 | 0.30 | Multiplicative darkening ($I' = \text{clip}(\text{round}(I \times \text{factor}))$, progressive darkening). |
| `contrast` | factor | 0.80 | 0.60 | 0.40 | 0.20 | 0.10 | Linear scaling around spatial channel mean ($I' = \mu_c + \text{factor} \times (I - \mu_c)$). |
| `jpeg_compression` | quality | 80 | 60 | 40 | 20 | 10 | In-memory JPEG encoding/decoding via PIL and `io.BytesIO` (no disk writes). |

### Deterministic Randomness & Common Random Numbers (CRN)
- **Seed Derivation**: For stochastic corruptions (`gaussian_noise`), local `np.random.Generator` seeds are computed deterministically via SHA-256 cryptographic digest:
  $$\text{seed}_{\text{u64}} = \text{struct.unpack}(">Q", \text{SHA256}(f"\{\text{seed}\}:\{\text{sample\_idx}\}:\{\text{corruption\_name}\}")[:8])[0]$$
- **Common Random Numbers (CRN)**: A standardized Gaussian noise field $\xi \sim \mathcal{N}(0, 1)$ is sampled once per sample and scaled proportionally by $\sigma = \text{std\_norm} \times 255.0$ across severities 1 to 5. This provides *ceteris paribus* severity scaling where perturbation magnitude grows monotonically without altering the underlying spatial pattern.
- **Empirical Guarantees**:
  - Independent of Python's randomized process `hash()` seed (empirically verified across subprocesses with varying `PYTHONHASHSEED`).
  - Independent of DataLoader worker scheduling, batch sizes, and dataset shuffle ordering.
  - Isolated from global NumPy random state (`np.random.get_state()` is never mutated, and output is immune to global seeds).
  - Bitwise reproducible across platforms, processes, and repeated runs.

### Architectural Invariants & Memory Safety
1. **Pre-Normalization Execution**: Corruptions execute on raw uint8 RGB arrays ($H \times W \times 3$, $[0, 255]$) *before* torchvision `ToTensor` (scaling to $[0, 1]$) and `Normalize` (standardization).
2. **Non-Destructive Operations**: Raw input image arrays are never modified in-place. All operations return freshly allocated uint8 NumPy arrays.
3. **Lightweight Dataset Adapter (`CorruptedDataset`)**:
   - Recursively unwraps torchvision `CIFAR10`, `TransformedSubset`, or `torch.utils.data.Subset` views to resolve root canonical indices without duplicating image arrays in host memory.
   - Strictly validates raw uint8 storage and rejects unsupported or pre-transformed datasets with an informative `TypeError`.
   - Integrates seamlessly with PyTorch `DataLoader`.
4. **Authoritative Single Source of Truth**:
   - `configs/corruptions.yaml` is the single source of truth for severity parameters. The runtime registry is built dynamically from YAML, eliminating duplicate parameter tables in Python.

### Verification & Test Suite
- **New Tests Added**: 37 focused unit and integration tests in `tests/test_corruptions.py`.
- **Total Test Suite**: 115 tests passing in 12.45s across all test modules (0 failures, 0 skipped).
- **Verified Invariants**: Input shape/dtype validation, error handling, seed idempotence, CRN proportional scaling, subprocess hash isolation, zero global RNG mutation, multi-image monotonic severity degradation, horizontal motion blur directional invariance, contrast constant-image invariance, pre-normalization execution, and nested subset resolution.

---

## 9. Milestone 1, Task 4: FP32 Robustness Evaluation Engine

### Overview & Scientific Objective
To evaluate the out-of-distribution predictive reliability of the trained baseline models (`resnet18_cifar` and `mobilenet_v3_small_cifar`), a reproducible robustness evaluation engine was implemented in `src/edge_vision/evaluation/robustness.py` and fronted by CLI script `scripts/run_robustness_benchmark.py`.

The benchmark measures accuracy degradation across the clean held-out CIFAR-10 test set and all 30 corruption conditions (6 corruptions $\times$ 5 severity levels), totaling 31 conditions per model (62 model-condition evaluations, 620,000 total inference predictions across 10,000 test samples).

### Key Architectural Invariants & Scientific Guarantees
1. **Shared-Batch Identical Input Presentation**:
   - To guarantee perfectly paired comparisons, both models are loaded into unified memory simultaneously (~141 MB total RSS; models take ~51 MB parameter memory).
   - In each condition, each batch `(batch_inputs, batch_targets)` is fetched once from the condition DataLoader, transferred to the active device (`mps` / `cpu`), and passed sequentially to ResNet-18 and MobileNetV3-Small in the same forward loop under `torch.inference_mode()`.
   - Empirically verified in `tests/test_robustness.py::test_shared_batch_input_identity_across_models` via an `IdentityCheckingModel` recording SHA-256 tensor byte digests across forward passes (0 byte discrepancies).
2. **Explicit Degradation Formulas (No Negative Clipping)**:
   - Absolute drop (percentage points): $\text{accuracy\_drop\_pp} = (\text{clean\_acc} - \text{corrupted\_acc}) \times 100.0$
   - Relative accuracy drop: $\text{relative\_accuracy\_drop} = (\text{clean\_acc} - \text{corrupted\_acc}) / \text{clean\_acc}$
   - Degradation metrics explicitly preserve negative values (e.g. if a model performs marginally better on low-severity brightness).
3. **Comprehensive Aggregate Summaries**:
   - Unweighted arithmetic mean corrupted accuracy and mean accuracy drop (overall across all 30 corruptions, per-corruption across 5 severities, and per-severity across 6 corruptions).
4. **Synchronized Condition Wall-Clock Timing (`condition_wall_time_seconds`)**:
   - Condition wall-clock time measures the complete shared evaluation loop, including DataLoader iteration, corruption generation, preprocessing, device transfer, both model forward passes, and metric tracking. It is not an isolated per-model inference latency measurement (isolated latency profiling is reserved for Milestone 4). Device synchronization (`torch.mps.synchronize()`) is enforced before and after the batch evaluation loop.
5. **Atomic Multi-Format Persistence**:
   - Saves `metadata.json` (SHA-256 hashes of checkpoints and config, environment versions, timestamps), `results.csv` (1 row per model-condition evaluation), `results.json` (per-condition metrics, confusion matrices, class accuracies), and `summary.json` (comparative aggregates).
6. **Safety Interlocks & Overwrite Protection**:
   - The CLI requires an explicit `--smoke-test`, `--max-samples <N>`, or `--full-run` flag to guard against accidental long-running 620,000-prediction runs during exploratory work.
   - Output directory overwrite protection: Validates before checkpoint loading or model evaluation whether the target output directory already contains completed benchmark artifacts (`metadata.json`, `results.csv`, `results.json`, `summary.json`). Halts immediately with a clear error unless `--overwrite` is explicitly passed.

### Smoke Test Validation (Stage C)
- **Configuration**: Evaluated both models on clean test set and `gaussian_noise` (severity 1) over a 128-sample subset using seed 42 on MPS device.
- **Results**:
  - `clean`: ResNet-18 Acc 96.88% (Loss 0.1332); MobileNetV3-Small Acc 87.50% (Loss 0.3934).
  - `gaussian_noise_sev1`: ResNet-18 Acc 82.03% (-14.84 pp, rel 15.32%, Loss 0.7172); MobileNetV3-Small Acc 74.22% (-13.28 pp, rel 15.18%, Loss 0.7228).
  - Checkpoints verified via SHA-256 hashes matching baseline audit. Output files `metadata.json`, `results.csv`, `results.json`, `summary.json` written atomically and verified.

### Verification & Test Suite
- **New Tests Added**: 15 unit and integration tests in `tests/test_robustness.py`.
- **Total Test Suite**: 130 tests passing in 12.82s across all test modules (0 failures, 0 skipped).
- **Verified Invariants**: Shared-batch input byte identity, degradation mathematics, sample limits, multi-level summary calculations, atomic persistence, condition wall-clock timing semantics, and output-directory overwrite protection.

---

## 10. Milestone 1, Task 4.2: Full FP32 Robustness Benchmark Results

### Benchmark Execution Summary
- **Execution Date**: 2026-11-05 (03:38:33 to 03:42:34 UTC)
- **Total Runtime**: 240.98 seconds (~4.0 minutes) on Apple Silicon M3 (MPS)
- **Evaluations**: 62 model-condition evaluations across 31 conditions (1 clean + 30 corruptions)
- **Total Predictions**: 620,000 predictions over 10,000 held-out CIFAR-10 test samples
- **Artifacts**: `experiments/robustness_fp32_seed42/` (`metadata.json`, `results.csv`, `results.json`, `summary.json`, `plots/`)

### Main Model Comparison

| Metric | ResNet-18 (FP32) | MobileNetV3-Small (FP32) | Architecture Delta (ResNet - MobileNet) |
| :--- | ---:| ---:| ---:|
| **Clean Test Accuracy** | 94.78% | 86.51% | +8.27 pp |
| **Mean Corrupted Accuracy** | 65.73% | 58.83% | +6.90 pp |
| **Mean Absolute Accuracy Drop** | 29.05 pp | 27.68 pp | +1.37 pp (ResNet drops more in abs pp) |
| **Mean Relative Accuracy Drop** | 30.65% | 32.00% | -1.35% (MobileNet degrades slightly more relatively) |
| **Worst Corruption Type** | `gaussian_noise` (45.46%) | `motion_blur` (41.47%) | Architecture-dependent vulnerability |
| **Worst Single Condition** | `gaussian_noise_sev5` (22.79%) | `contrast_sev5` (20.50%) | Both models suffer catastrophic failure |

### Per-Corruption Degradation Profiles

| Corruption Type | ResNet-18 Mean Acc | ResNet-18 Drop (pp) | ResNet-18 Rel Drop | MobileNetV3 Mean Acc | MobileNetV3 Drop (pp) | MobileNetV3 Rel Drop | Leader Model |
| :--- | ---:| ---:| ---:| ---:| ---:| ---:| :--- |
| **Brightness** | 90.66% | 4.12 pp | 4.35% | 76.68% | 9.83 pp | 11.37% | ResNet-18 (+13.98 pp) |
| **Contrast** | 72.06% | 22.72 pp | 23.97% | 56.49% | 30.02 pp | 34.70% | ResNet-18 (+15.57 pp) |
| **Gaussian Blur** | 71.11% | 23.67 pp | 24.97% | 66.12% | 20.39 pp | 23.57% | ResNet-18 (+4.99 pp) |
| **Gaussian Noise** | 45.46% | 49.32 pp | 52.04% | 42.54% | 43.97 pp | 50.82% | ResNet-18 (+2.92 pp) |
| **JPEG Compression** | 67.83% | 26.95 pp | 28.44% | 69.66% | 16.85 pp | 19.48% | **MobileNetV3-Small (+1.83 pp)** |
| **Motion Blur** | 47.27% | 47.51 pp | 50.13% | 41.47% | 45.04 pp | 52.07% | ResNet-18 (+5.80 pp) |

### Key Scientific Findings & Interpretation
1. **Empirical Robustness Trade-offs**:
   - ResNet-18 achieves higher clean accuracy (94.78% vs. 86.51%) and higher overall mean corrupted accuracy (65.73% vs. 58.83%).
   - However, absolute degradation (percentage points drop) is lower for MobileNetV3-Small (27.68 pp vs. 29.05 pp), whereas relative degradation (Mean Relative Accuracy Drop) is lower for ResNet-18 (30.65% vs. 32.00%). Robustness rankings depend on whether absolute or relative metrics are considered.
2. **JPEG Compression Crossover**:
   - Despite an 8.27 pp lower clean baseline, MobileNetV3-Small demonstrates higher empirical resilience to severe JPEG compression artifacts than ResNet-18:
     - Quality 40 (sev 3): MobileNetV3 73.91% vs. ResNet-18 73.38% (+0.53 pp)
     - Quality 20 (sev 4): MobileNetV3 65.45% vs. ResNet-18 59.25% (+6.20 pp)
     - Quality 10 (sev 5): MobileNetV3 50.65% vs. ResNet-18 40.61% (+10.04 pp)
   - *Scientific Note*: The observed crossover indicates an architecture-dependent difference in sensitivity to high-frequency block and quantization artifacts. The specific causal mechanisms (e.g. receptive field profiles, normalization, or activation dynamics) have not been experimentally isolated in this benchmark and remain hypotheses for future investigation.
3. **Preserved Negative Drop on Mild Blur**:
   - On `gaussian_blur` severity 1 ($\sigma = 0.3$), MobileNetV3-Small accuracy slightly improved from 86.51% (clean) to 86.54% (-0.03 pp drop, preserved without clipping).
4. **Catastrophic Degradation Under High-Frequency Perturbations**:
   - Additive Gaussian noise and linear motion blur cause the most severe degradation for both architectures, reducing Top-1 accuracy to 21–25% at severity 5.

### Experimental Scope & Limitations
1. **Low-Resolution Domain**: CIFAR-10 imagery ($32\times 32\times 3$) possesses low spatial redundancy; image degradations that eliminate fine details disproportionately impact small-scale feature extraction compared to higher-resolution domains (e.g. ImageNet at $224\times 224$).
2. **Custom Synthetic Benchmark**: This benchmark is an independent, custom synthetic corruption suite inspired by common taxonomies, not the official pre-generated CIFAR-10-C distribution. The reported degradation metric is Mean Relative Accuracy Drop, not standardized mCE.
3. **Parametric Constraints**: Motion blur is evaluated exclusively along the horizontal axis ($0^\circ$), and brightness is evaluated exclusively as progressive darkening ($0.90 \to 0.30$).
4. **Single-Seed Evaluation**: Evaluations were conducted on seed 42 checkpoints; multi-seed variance and statistical significance across training runs have not been measured.
5. **Domain Transfer**: Synthetic mathematical corruptions do not replicate physical lens aberrations, atmospheric weather, or analog sensor noise. Findings should not be extrapolated directly to safety-critical production deployments without domain-specific validation.

- **Next Milestone**:
  - Milestone 3: Compression & Quantization Engine (FP16 conversion and PyTorch INT8 PTQ).
