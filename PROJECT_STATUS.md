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

- **Next Milestone**:
  - Milestone 1, Task 3: Deterministic Synthetic Image Corruption Pipeline following the CIFAR-10-C taxonomy.
