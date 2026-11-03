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
- [ ] Clean CIFAR-10 data loader with deterministic splits and caching.
- [ ] Modular corruption pipeline (synthesizing CIFAR-10-C style perturbations: noise, blur, weather, digital).
- [ ] Unit tests for data loading and perturbation deterministic seeds.

### Milestone 2: Baseline Models (FP32)
- [ ] Model architecture wrappers (`MobileNetV3-Small`, `ResNet18`).
- [ ] Baseline pre-trained / trained weights management.
- [ ] Clean baseline accuracy verification on test split.

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
