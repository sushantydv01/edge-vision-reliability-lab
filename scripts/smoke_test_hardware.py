"""
Bounded hardware smoke test runner for CPU and Apple Silicon MPS backends.
Runs a bounded test (5 training batches, 2 validation batches) to measure timing,
throughput, memory footprints, and checkpoint serialization without expensive training.
"""

from pathlib import Path
import shutil
import tempfile
import time
import psutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from edge_vision.models import get_model
from edge_vision.training.checkpoint import CheckpointManager
from edge_vision.training.config import TrainingConfig, load_training_config
from edge_vision.training.data import get_training_dataloaders
from edge_vision.training.experiment import ExperimentLogger
from edge_vision.training.seeding import set_seed
from edge_vision.training.trainer import Trainer


def run_bounded_smoke_test(
    model_name: str,
    device_name: str,
    num_train_batches: int = 5,
    num_val_batches: int = 2,
    batch_size: int = 64,
) -> dict:
    set_seed(42)
    device = torch.device(device_name)

    cfg = load_training_config("configs/train_baseline.yaml")
    cfg.model_architecture = model_name
    cfg.dataloader.batch_size = batch_size
    cfg.device = device_name

    # Load dataloaders
    train_loader_full, val_loader_full, gen = get_training_dataloaders(cfg)

    # Subsample loaders to bounded batch count
    train_indices = list(range(num_train_batches * batch_size))
    val_indices = list(range(num_val_batches * batch_size))

    train_sub = Subset(train_loader_full.dataset, train_indices)
    val_sub = Subset(val_loader_full.dataset, val_indices)

    train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_sub, batch_size=batch_size, shuffle=False)

    model = get_model(model_name, num_classes=10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    tmp_dir = tempfile.mkdtemp(prefix=f"smoke_{model_name}_{device_name}_")
    ckpt_mgr = CheckpointManager(tmp_dir)
    logger = ExperimentLogger(tmp_dir, f"smoke_{model_name}_{device_name}")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        config=cfg,
        checkpoint_manager=ckpt_mgr,
        experiment_logger=logger,
        device=device,
        loader_generator=gen,
    )

    proc = psutil.Process()
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()

    rss_before = proc.memory_info().rss / 1_000_000
    mps_before = torch.mps.current_allocated_memory() if device.type == "mps" else None

    # Step 1: Train epoch (5 batches)
    t0 = time.perf_counter()
    train_metrics = trainer.train_epoch(1)
    train_duration = time.perf_counter() - t0

    # Step 2: Val epoch (2 batches)
    t1 = time.perf_counter()
    val_metrics = trainer.validate_epoch(1)
    val_duration = time.perf_counter() - t1

    rss_after = proc.memory_info().rss / 1_000_000
    mps_after = torch.mps.current_allocated_memory() if device.type == "mps" else None
    mps_driver = torch.mps.driver_allocated_memory() if device.type == "mps" else None

    # Step 3: Checkpoint save
    ckpt_path, best_path = ckpt_mgr.save_epoch(
        epoch=1,
        model=trainer.model,
        optimizer=trainer.optimizer,
        scheduler=trainer.scheduler,
        metrics={"train_loss": train_metrics.loss, "val_loss": val_metrics.loss, "val_acc": val_metrics.accuracy},
        config=cfg.to_dict(),
        rng_states={},
    )
    latest_size_bytes = ckpt_path.stat().st_size
    best_size_bytes = best_path.stat().st_size if best_path else 0

    # Step 4: Checkpoint load & resume validation
    loaded = ckpt_mgr.load_checkpoint(ckpt_path)
    CheckpointManager.validate_checkpoint_compatibility(loaded, cfg.to_dict())

    # Step 5: Log summary
    logger.write_summary(
        config_dict=cfg.to_dict(),
        best_epoch=1,
        best_val_acc=val_metrics.accuracy,
        best_val_loss=val_metrics.loss,
        final_train_loss=train_metrics.loss,
        final_train_acc=train_metrics.accuracy,
        final_val_loss=val_metrics.loss,
        final_val_acc=val_metrics.accuracy,
        device_str=str(device),
        status="completed",
    )

    # Cleanup temp dir
    shutil.rmtree(tmp_dir, ignore_errors=True)

    return {
        "model": model_name,
        "device": device_name,
        "train_loss": train_metrics.loss,
        "train_acc": train_metrics.accuracy,
        "train_duration_sec": train_duration,
        "train_samples_per_sec": (num_train_batches * batch_size) / train_duration,
        "val_loss": val_metrics.loss,
        "val_acc": val_metrics.accuracy,
        "val_duration_sec": val_duration,
        "rss_before_mb": rss_before,
        "rss_after_mb": rss_after,
        "mps_allocated_mb": (mps_after / 1_000_000) if mps_after is not None else None,
        "mps_driver_mb": (mps_driver / 1_000_000) if mps_driver is not None else None,
        "latest_checkpoint_bytes": latest_size_bytes,
        "best_checkpoint_bytes": best_size_bytes,
        "system_avail_gb": vm.available / 1_000_000_000,
        "swap_used_gb": swap.used / 1_000_000_000,
    }


def main():
    print("=" * 80)
    print("RUNNING BOUNDED HARDWARE SMOKE TEST (CPU & MPS)")
    print("=" * 80)

    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")

    models = ["resnet18_cifar", "mobilenet_v3_small_cifar"]

    for model_name in models:
        for dev in devices:
            print(f"\n--- Testing Model: {model_name} on Device: {dev} ---")
            res = run_bounded_smoke_test(model_name, dev)
            print(f"  Train Loss: {res['train_loss']:.4f} | Train Acc: {res['train_acc']:.2%}")
            print(f"  Val Loss:   {res['val_loss']:.4f} | Val Acc:   {res['val_acc']:.2%}")
            print(f"  Train Time: {res['train_duration_sec']:.3f}s ({res['train_samples_per_sec']:.1f} samples/s)")
            print(f"  Val Time:   {res['val_duration_sec']:.3f}s")
            print(f"  Process RSS: {res['rss_after_mb']:.2f} MB (change: {res['rss_after_mb'] - res['rss_before_mb']:+.2f} MB)")
            if res['mps_allocated_mb'] is not None:
                print(f"  MPS Allocated: {res['mps_allocated_mb']:.2f} MB | Driver: {res['mps_driver_mb']:.2f} MB")
            print(f"  Checkpoint Size: {res['latest_checkpoint_bytes']:,} bytes ({res['latest_checkpoint_bytes']/1_000_000:.2f} MB)")
            print(f"  System Avail: {res['system_avail_gb']:.2f} GB | Swap Used: {res['swap_used_gb']:.2f} GB")


if __name__ == "__main__":
    main()
