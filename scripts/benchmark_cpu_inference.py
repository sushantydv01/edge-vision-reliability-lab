"""
Isolated-Process CPU Inference Latency and Throughput Benchmark.

Measures forward-pass inference time strictly in isolated OS processes per configuration
to prevent inter-run thread pool pollution or OpenMP worker state retention.
Evaluates:
  - Models: ResNet-18, MobileNetV3-Small
  - Precisions: Eager FP32, TorchScript FP32, TorchScript INT8
  - Batch sizes: 1, 64
  - Intra-op thread counts: 1, 2, 4
  - Repetitions: 2 independent OS process runs per configuration to assess stability.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List

import numpy as np
import torch

from edge_vision.evaluation.evaluator import load_checkpoint_for_evaluation


CHECKPOINTS = {
    "fp32": {
        "resnet18_cifar": "experiments/checkpoints_inference_fp32/resnet18_cifar_fp32.pt",
        "mobilenet_v3_small_cifar": "experiments/checkpoints_inference_fp32/mobilenet_v3_small_cifar_fp32.pt",
    },
    "int8": {
        "resnet18_cifar": "experiments/checkpoints_int8/resnet18_cifar_int8.pt",
        "mobilenet_v3_small_cifar": "experiments/checkpoints_int8/mobilenet_v3_small_cifar_int8.pt",
    },
}


def run_worker(
    model_name: str,
    variant: str,
    batch_size: int,
    threads: int,
    repetition: int,
    out_file: str,
) -> None:
    """
    Executes in a fresh child process.
    Configures threads once at start, loads model, preallocates deterministic input,
    and records forward pass timings.
    """
    # 1. Configure thread settings strictly before any model execution
    torch.set_num_threads(threads)
    actual_intra_op = torch.get_num_threads()
    actual_inter_op = torch.get_num_interop_threads()

    device = torch.device("cpu")
    torch.backends.quantized.engine = "qnnpack"

    # 2. Load model
    if variant == "eager_fp32":
        model, _ = load_checkpoint_for_evaluation(CHECKPOINTS["fp32"][model_name], device=device, precision="fp32")
        model.eval()
    elif variant == "torchscript_fp32":
        eager_model, _ = load_checkpoint_for_evaluation(CHECKPOINTS["fp32"][model_name], device=device, precision="fp32")
        eager_model.eval()
        dummy_trace = torch.randn(1, 3, 32, 32, device=device)
        with torch.inference_mode():
            model = torch.jit.trace(eager_model, dummy_trace)
    elif variant == "torchscript_int8":
        model, _ = load_checkpoint_for_evaluation(CHECKPOINTS["int8"][model_name], device=device, precision="int8")
        model.eval()
    else:
        raise ValueError(f"Unknown variant: {variant}")

    # 3. Deterministic contiguous input preallocation
    torch.manual_seed(42)
    input_tensor = torch.randn(batch_size, 3, 32, 32, dtype=torch.float32, device=device).contiguous()

    # 4. Standardized warm-up and timed iterations
    warmup_runs = 20 if batch_size == 1 else 10
    timed_runs = 100 if batch_size == 1 else 30

    with torch.inference_mode():
        for _ in range(warmup_runs):
            _ = model(input_tensor)

    latencies: List[float] = []
    with torch.inference_mode():
        for _ in range(timed_runs):
            t0 = time.perf_counter()
            _ = model(input_tensor)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)  # ms

    lat_arr = np.array(latencies)
    mean_ms = float(np.mean(lat_arr))
    median_ms = float(np.median(lat_arr))
    std_ms = float(np.std(lat_arr))
    p95_ms = float(np.percentile(lat_arr, 95))
    p99_ms = float(np.percentile(lat_arr, 99))
    throughput = float((batch_size * 1000.0) / mean_ms)

    payload = {
        "model_name": model_name,
        "variant": variant,
        "batch_size": batch_size,
        "configured_threads": threads,
        "actual_intra_op_threads": actual_intra_op,
        "actual_inter_op_threads": actual_inter_op,
        "repetition": repetition,
        "pid": os.getpid(),
        "warmup_runs": warmup_runs,
        "timed_runs": timed_runs,
        "mean_latency_ms": round(mean_ms, 3),
        "median_latency_ms": round(median_ms, 3),
        "std_latency_ms": round(std_ms, 3),
        "p95_latency_ms": round(p95_ms, 3),
        "p99_latency_ms": round(p99_ms, 3),
        "throughput_samples_per_sec": round(throughput, 1),
        "raw_latencies_ms": [round(x, 4) for x in latencies],
    }

    out_p = Path(out_file)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        json.dump(payload, f, indent=2)


def orchestrate_benchmark(
    output_file: str,
    repetitions: int = 2,
    raw_dir: str = "experiments/cpu_benchmark_isolated_runs",
) -> None:
    """
    Orchestrates the isolated benchmark matrix sequentially, launching a fresh child
    process for each run. Aggregates results across repetitions and generates the final JSON.
    """
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)

    thread_configs = [1, 2, 4]
    models = ["mobilenet_v3_small_cifar", "resnet18_cifar"]
    variants = ["eager_fp32", "torchscript_fp32", "torchscript_int8"]
    batch_sizes = [1, 64]

    total_runs = len(thread_configs) * len(models) * len(variants) * len(batch_sizes) * repetitions
    run_idx = 0

    print("================================================================================")
    print("STARTING ISOLATED-PROCESS CPU INFERENCE BENCHMARK")
    print(f"Total isolated worker processes: {total_runs} ({repetitions} repetitions per configuration)")
    print(f"PyTorch version: {torch.__version__} | Platform: {os.uname().machine}")
    print("================================================================================")

    worker_outputs: Dict[str, Any] = {}

    for rep in range(1, repetitions + 1):
        print(f"\n>>> Starting Repetition {rep} of {repetitions} <<<")
        for threads in thread_configs:
            for model in models:
                for variant in variants:
                    for batch_size in batch_sizes:
                        run_idx += 1
                        out_json = raw_path / f"{model}_{variant}_th{threads}_b{batch_size}_rep{rep}.json"

                        cmd = [
                            sys.executable,
                            __file__,
                            "--worker",
                            "--model", model,
                            "--variant", variant,
                            "--batch-size", str(batch_size),
                            "--threads", str(threads),
                            "--repetition", str(rep),
                            "--worker-out", str(out_json),
                        ]

                        print(f"[{run_idx:02d}/{total_runs}] Running {model} | {variant} | threads={threads} | bs={batch_size:2d} (rep {rep})...", end="", flush=True)

                        t0 = time.time()
                        proc = subprocess.run(cmd, capture_output=True, text=True)
                        elapsed = time.time() - t0

                        if proc.returncode != 0:
                            print(f" FAILED (exit code {proc.returncode})")
                            print("STDERR:\n", proc.stderr)
                            sys.exit(proc.returncode)

                        with open(out_json, "r") as f:
                            run_data = json.load(f)

                        print(f" Mean {run_data['mean_latency_ms']:6.2f} ms | Throughput {run_data['throughput_samples_per_sec']:7.1f} img/s ({elapsed:4.1f}s)")

                        key = f"{model}__{variant}__th{threads}__b{batch_size}"
                        if key not in worker_outputs:
                            worker_outputs[key] = []
                        worker_outputs[key].append(run_data)

    print("\n[+] All isolated worker processes completed successfully.")
    print("[*] Aggregating multi-run statistics and computing configuration-specific speedups...")

    # Aggregate results
    final_results: Dict[str, Any] = {
        "metadata": {
            "device": "cpu",
            "cpu_architecture": os.uname().machine,
            "quantized_engine": "qnnpack",
            "pytorch_version": torch.__version__,
            "benchmark_timestamp": datetime.now(timezone.utc).isoformat(),
            "repetitions": repetitions,
            "isolation_method": "Fresh OS child process for every individual run",
            "notes": (
                "Isolated-process CPU benchmark comparing Eager FP32, TorchScript FP32, and TorchScript INT8. "
                "Each configuration ran in a fresh child process to eliminate OpenMP worker thread state retention. "
                "Measurements are reported across 1, 2, and 4 threads, with run-to-run variation across independent processes."
            ),
        },
        "configurations": {},
        "comparative_analysis": {},
    }

    for key, reps in worker_outputs.items():
        model, variant, th_str, b_str = key.split("__")
        threads = int(th_str.replace("th", ""))
        batch_size = int(b_str.replace("b", ""))

        means = [r["mean_latency_ms"] for r in reps]
        medians = [r["median_latency_ms"] for r in reps]
        throughputs = [r["throughput_samples_per_sec"] for r in reps]
        p95s = [r["p95_latency_ms"] for r in reps]

        rep_diff_ms = abs(means[0] - means[1]) if len(means) >= 2 else 0.0
        rep_diff_pct = (rep_diff_ms / np.mean(means) * 100.0) if np.mean(means) > 0 else 0.0

        if th_str not in final_results["configurations"]:
            final_results["configurations"][th_str] = {"threads": threads, "models": {}}
        if model not in final_results["configurations"][th_str]["models"]:
            final_results["configurations"][th_str]["models"][model] = {}
        if variant not in final_results["configurations"][th_str]["models"][model]:
            final_results["configurations"][th_str]["models"][model][variant] = {}

        final_results["configurations"][th_str]["models"][model][variant][f"batch_{batch_size}"] = {
            "mean_latency_ms": round(float(np.mean(means)), 3),
            "median_latency_ms": round(float(np.mean(medians)), 3),
            "p95_latency_ms": round(float(np.mean(p95s)), 3),
            "throughput_samples_per_sec": round(float(np.mean(throughputs)), 1),
            "repetition_means_ms": [round(m, 3) for m in means],
            "repetition_difference_ms": round(rep_diff_ms, 3),
            "repetition_variation_pct": round(rep_diff_pct, 2),
            "individual_runs": [
                {
                    "repetition": r["repetition"],
                    "pid": r["pid"],
                    "mean_latency_ms": r["mean_latency_ms"],
                    "median_latency_ms": r["median_latency_ms"],
                    "p95_latency_ms": r["p95_latency_ms"],
                    "throughput_samples_per_sec": r["throughput_samples_per_sec"],
                }
                for r in reps
            ],
        }

    # Compute comparative speedups and serialization overheads
    for th_str in final_results["configurations"].keys():
        threads = final_results["configurations"][th_str]["threads"]
        final_results["comparative_analysis"][th_str] = {}

        for model in models:
            final_results["comparative_analysis"][th_str][model] = {}
            for b in [1, 64]:
                b_key = f"batch_{b}"
                m_cfg = final_results["configurations"][th_str]["models"][model]

                eager_mean = m_cfg["eager_fp32"][b_key]["mean_latency_ms"]
                ts_fp32_mean = m_cfg["torchscript_fp32"][b_key]["mean_latency_ms"]
                ts_int8_mean = m_cfg["torchscript_int8"][b_key]["mean_latency_ms"]

                eager_thru = m_cfg["eager_fp32"][b_key]["throughput_samples_per_sec"]
                ts_fp32_thru = m_cfg["torchscript_fp32"][b_key]["throughput_samples_per_sec"]
                ts_int8_thru = m_cfg["torchscript_int8"][b_key]["throughput_samples_per_sec"]

                final_results["comparative_analysis"][th_str][model][b_key] = {
                    "eager_fp32_mean_ms": eager_mean,
                    "torchscript_fp32_mean_ms": ts_fp32_mean,
                    "torchscript_int8_mean_ms": ts_int8_mean,
                    "torchscript_vs_eager_ratio": round(eager_mean / ts_fp32_mean, 3),
                    "int8_vs_torchscript_fp32_speedup": round(ts_fp32_mean / ts_int8_mean, 3),
                    "int8_vs_eager_fp32_speedup": round(eager_mean / ts_int8_mean, 3),
                    "int8_throughput_improvement_vs_torchscript_fp32": round(ts_int8_thru / ts_fp32_thru, 3),
                    "int8_throughput_improvement_vs_eager_fp32": round(ts_int8_thru / eager_thru, 3),
                }

    out_p = Path(output_file)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        json.dump(final_results, f, indent=2)

    print(f"\n[+] Saved final isolated CPU benchmark results to {out_p}")


def main():
    parser = argparse.ArgumentParser(description="Isolated-Process CPU Inference Benchmark.")
    parser.add_argument("--worker", action="store_true", help="Internal flag: run single isolated benchmark worker.")
    parser.add_argument("--model", type=str, choices=["resnet18_cifar", "mobilenet_v3_small_cifar"])
    parser.add_argument("--variant", type=str, choices=["eager_fp32", "torchscript_fp32", "torchscript_int8"])
    parser.add_argument("--batch-size", type=int, choices=[1, 64])
    parser.add_argument("--threads", type=int, choices=[1, 2, 4])
    parser.add_argument("--repetition", type=int, default=1)
    parser.add_argument("--worker-out", type=str)

    parser.add_argument("--repetitions", type=int, default=2, help="Number of repetitions per configuration.")
    parser.add_argument("--output-file", type=str, default="reports/experiment_results/cpu_inference_benchmark.json")

    args = parser.parse_args()

    if args.worker:
        if not (args.model and args.variant and args.batch_size and args.threads and args.worker_out):
            parser.error("--worker requires --model, --variant, --batch-size, --threads, and --worker-out.")
        run_worker(
            model_name=args.model,
            variant=args.variant,
            batch_size=args.batch_size,
            threads=args.threads,
            repetition=args.repetition,
            out_file=args.worker_out,
        )
    else:
        orchestrate_benchmark(output_file=args.output_file, repetitions=args.repetitions)


if __name__ == "__main__":
    main()
