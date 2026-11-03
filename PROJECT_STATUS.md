# Edge Vision Reliability Lab - Project Status

## 1. System & Environment Baseline

- **Date Initialized**: 2026-11-03
- **Host Hardware**: Apple MacBook Pro (Apple M3)
- **Memory**: 8 GB Unified Memory (conservative memory policy required)
- **Storage**: ~163 GB available SSD space
- **OS**: macOS 27.0 (Darwin 27.0.0, arm64)
- **Python**: Python 3.12.16 (`/opt/homebrew/bin/python3.12`)
- **Git**: Git 2.47.0 (Initialized locally with branch `main`)
- **Package Manager**: Homebrew 7.0.12

---

## 2. Milestone Progress Tracker

### Milestone 0: Environment & Project Baseline
- [x] Workspace verification and environment inspection.
- [x] Initial directory skeleton creation.
- [x] Modern `.gitignore` configured for ML experiments and data exclusion.
- [x] Local Git repository initialized on branch `main`.
- [x] Concise `README.md` and `PROJECT_STATUS.md` created.
- [ ] Python 3.12 virtual environment creation (`.venv`).
- [ ] Dependency specification (`pyproject.toml` / `requirements.txt`) with strict version pinning.
- [ ] Smoke test confirming PyTorch CPU & MPS device availability.

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
- [ ] INT8 Post-Training Quantization (PTQ) pipeline (PyTorch quantization / ONNX Runtime / Core ML if applicable).
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

## 3. Engineering Decisions & Constraints

1. **Python Version**: Explicitly locked to Python 3.12 (`/opt/homebrew/bin/python3.12`). Python 3.14 was detected on the host system but is skipped due to library ecosystem compatibility (e.g., PyTorch, torchvision, NumPy).
2. **Memory Constraints**: The host machine has 8 GB unified memory. All data batching, evaluation scripts, and calibration sets must be stream-oriented or conservatively batched to avoid memory pressure or swap thrashing.
3. **Compute Devices**: Standard baseline target is CPU for deterministic reproducibility across platforms. Apple Silicon MPS (Metal Performance Shaders) will be assessed as a secondary acceleration target once the CPU baseline is confirmed.
4. **Packaging Structure**: Adopted standard `src/edge_vision/` package layout to facilitate reproducible local imports and editable installs without sys.path hacks.
