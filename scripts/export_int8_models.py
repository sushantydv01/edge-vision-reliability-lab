"""
Script to calibrate and export standalone INT8 TorchScript models using QNNPACK and PyTorch FX.
"""

from pathlib import Path
import sys
import torch

from edge_vision.evaluation.evaluator import load_checkpoint_for_evaluation
from edge_vision.evaluation.robustness import compute_file_sha256
from edge_vision.quantization.ptq import (
    export_int8_checkpoint,
    get_calibration_dataloader,
    quantize_model_ptq,
)

DEFAULT_CHECKPOINTS = {
    "resnet18_cifar": "experiments/exp_baseline_resnet18_seed42/checkpoints/best_checkpoint.pt",
    "mobilenet_v3_small_cifar": "experiments/exp_baseline_mobilenet_v3_small_seed42/checkpoints/best_checkpoint.pt",
}


def main() -> None:
    print("[*] Starting INT8 Post-Training Quantization (PTQ) calibration & export...")
    torch.backends.quantized.engine = "qnnpack"

    calib_samples = 512
    calib_seed = 42
    print(f"[*] Acquiring deterministic calibration DataLoader ({calib_samples} samples from training split, seed={calib_seed})...")
    calib_loader = get_calibration_dataloader(
        data_config_path="configs/cifar10.yaml",
        num_samples=calib_samples,
        batch_size=64,
        seed=calib_seed,
    )
    print(f"    Calibration batches: {len(calib_loader)} (total images: {calib_samples})")

    out_dir = Path("experiments/checkpoints_int8")
    out_dir.mkdir(parents=True, exist_ok=True)

    for m_arch, ckpt_str in DEFAULT_CHECKPOINTS.items():
        ckpt_path = Path(ckpt_str)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Source FP32 checkpoint missing: {ckpt_path}")

        ckpt_sha = compute_file_sha256(ckpt_path)
        print(f"\n[*] Processing architecture: {m_arch}")
        print(f"    Source checkpoint: {ckpt_path} (SHA-256: {ckpt_sha[:12]}...)")

        # Load float32 source model
        fp32_model, raw_ckpt = load_checkpoint_for_evaluation(ckpt_path, device=torch.device("cpu"), precision="fp32")

        # Quantize via PyTorch FX PTQ
        print("    Running PyTorch FX preparation, calibration, and conversion...")
        traced_int8, op_inventory = quantize_model_ptq(
            model=fp32_model,
            model_architecture=m_arch,
            calibration_dataloader=calib_loader,
            device="cpu",
        )

        out_pt = out_dir / f"{m_arch}_int8.pt"
        exported_path = export_int8_checkpoint(
            traced_model=traced_int8,
            model_architecture=m_arch,
            output_path=out_pt,
            source_checkpoint_path=ckpt_path,
            source_checkpoint_sha256=ckpt_sha,
            calibration_samples=calib_samples,
            operator_inventory=op_inventory,
        )

        sz_bytes = exported_path.stat().st_size
        print(f"    Exported standalone artifact: {exported_path}")
        print(f"    File size: {sz_bytes:,} bytes ({sz_bytes / 1e6:.4f} MB, {sz_bytes / (1024*1024):.4f} MiB)")
        print(f"    Quantized modules count: {sum(op_inventory.get('modules', {}).values())}")
        print(f"    Quantized functions count: {sum(op_inventory.get('functions', {}).values())}")

    print("\n[+] INT8 model export complete.")


if __name__ == "__main__":
    main()

