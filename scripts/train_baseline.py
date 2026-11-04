"""
Full 100-epoch baseline training runner for CIFAR-10 architectures on Apple Silicon MPS.
Task 2.3B: Controlled FP32 Baseline Training (seed 42).
"""

import argparse
from pathlib import Path
import shutil
import sys

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import psutil
import torch
import torch.nn as nn

from edge_vision.models import get_model
from edge_vision.training.checkpoint import CheckpointManager
from edge_vision.training.config import load_training_config
from edge_vision.training.data import get_training_dataloaders
from edge_vision.training.experiment import ExperimentLogger
from edge_vision.training.seeding import set_rng_states, set_seed
from edge_vision.training.trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 100-epoch controlled baseline training on CIFAR-10."
    )
    parser.add_argument(
        "--architecture",
        type=str,
        required=True,
        choices=["resnet18_cifar", "mobilenet_v3_small_cifar"],
        help="Model architecture to train.",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="Unique experiment run ID (default: exp_baseline_<arch_short>_seed<seed>).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_baseline.yaml",
        help="Path to baseline training configuration YAML.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Total training epochs (default: 100).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        choices=["mps", "cpu"],
        help="Execution device (default: mps).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint if one exists in the experiment directory.",
    )
    return parser.parse_args()


def inspect_resources() -> None:
    """Logs initial host hardware, disk, memory, and swap status."""
    disk = shutil.disk_usage(".")
    disk_free_gb = disk.free / (1024**3)
    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()

    print("-" * 70)
    print("HOST RESOURCE BASELINE")
    print(f"  Disk Free:  {disk_free_gb:.2f} GB (Total: {disk.total / (1024**3):.2f} GB)")
    print(f"  RAM Free:   {mem.available / (1024**3):.2f} GB / {mem.total / (1024**3):.2f} GB ({mem.percent}% used)")
    print(f"  Swap Used:  {swap.used / (1024**3):.2f} GB / {swap.total / (1024**3):.2f} GB")
    print("-" * 70)

    if disk_free_gb < 1.0:
        raise RuntimeError(f"Insufficient disk space ({disk_free_gb:.2f} GB free). Aborting.")


def run_training(args: argparse.Namespace) -> dict:
    # 1. Device check
    if args.device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS device requested but torch.backends.mps.is_available() is False. "
                "Per experimental requirements, do not silently switch to CPU."
            )
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # 2. Inspect host resources
    inspect_resources()

    # 3. Determine experiment ID
    if args.experiment_id:
        exp_id = args.experiment_id
    else:
        short_name = "resnet18" if "resnet" in args.architecture else "mobilenet_v3_small"
        exp_id = f"exp_baseline_{short_name}_seed{args.seed}"

    # 4. Load baseline configuration without altering hyperparameters
    cfg = load_training_config(args.config)
    cfg.experiment_id = exp_id
    cfg.model_architecture = args.architecture
    cfg.epochs = args.epochs
    cfg.seed = args.seed
    cfg.device = args.device
    cfg.output_dir = "experiments"

    exp_dir = Path(cfg.output_dir) / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = exp_dir / cfg.checkpoint.checkpoint_dir

    print(f"\n{'='*75}")
    print(f"BASELINE EXPERIMENT: {exp_id}")
    print(f"  Architecture:  {cfg.model_architecture}")
    print(f"  Total Epochs:  {cfg.epochs}")
    print(f"  Batch Size:    {cfg.dataloader.batch_size}")
    print(f"  Optimizer:     {cfg.optimizer.name.upper()} (lr={cfg.optimizer.learning_rate}, momentum={cfg.optimizer.momentum}, wd={cfg.optimizer.weight_decay})")
    print(f"  Scheduler:     {cfg.scheduler.name} (T_max={cfg.scheduler.t_max}, eta_min={cfg.scheduler.eta_min})")
    print(f"  Device:        {device} (FP32)")
    print(f"  Seed:          {cfg.seed}")
    print(f"  Output Dir:    {exp_dir}")
    print(f"{'='*75}\n")

    # 5. Set seed and create data loaders
    set_seed(cfg.seed)
    train_loader, val_loader, loader_gen = get_training_dataloaders(cfg)

    # 6. Instantiate model, optimizer, scheduler, criterion
    model = get_model(cfg.model_architecture, num_classes=10)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.optimizer.learning_rate,
        momentum=cfg.optimizer.momentum,
        nesterov=cfg.optimizer.nesterov,
        weight_decay=cfg.optimizer.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.scheduler.t_max,
        eta_min=cfg.scheduler.eta_min,
    )
    criterion = nn.CrossEntropyLoss()

    ckpt_mgr = CheckpointManager(
        ckpt_dir,
        metric_for_best=cfg.checkpoint.metric_for_best,
        mode=cfg.checkpoint.mode,
    )
    logger = ExperimentLogger(exp_dir, cfg.experiment_id)

    # 7. Checkpoint resume logic
    start_epoch = 1
    latest_ckpt_path = ckpt_dir / "latest_checkpoint.pt"
    if args.resume and latest_ckpt_path.is_file():
        print(f"Resuming from existing checkpoint: {latest_ckpt_path}")
        payload = ckpt_mgr.load_checkpoint(latest_ckpt_path)
        ckpt_mgr.validate_checkpoint_compatibility(payload, cfg.to_dict())

        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        if payload.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(payload["scheduler_state_dict"])
        if "rng_states" in payload and payload["rng_states"] is not None:
            set_rng_states(payload["rng_states"], loader_generator=loader_gen)

        ckpt_mgr.best_metric_value = payload.get("best_metric_value", float("-inf"))
        ckpt_mgr.best_epoch = payload.get("best_epoch", 0)
        ckpt_mgr._best_val_loss = payload.get("metrics", {}).get("val_loss", float("inf"))

        completed_epoch = payload["epoch"]
        start_epoch = completed_epoch + 1
        print(
            f"  Loaded state from epoch {completed_epoch}. "
            f"Resuming at epoch {start_epoch}/{cfg.epochs} "
            f"(Previous best val_acc: {ckpt_mgr.best_metric_value:.2%}, epoch {ckpt_mgr.best_epoch}).\n"
        )
        if start_epoch > cfg.epochs:
            print(f"Training already completed ({completed_epoch} >= {cfg.epochs}). Nothing to run.")
            return {
                "total_epochs": cfg.epochs,
                "best_epoch": ckpt_mgr.best_epoch,
                "best_val_acc": ckpt_mgr.best_metric_value,
                "best_val_loss": getattr(ckpt_mgr, "_best_val_loss", 0.0),
            }

    # 8. Instantiate Trainer and execute fit
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=cfg,
        checkpoint_manager=ckpt_mgr,
        experiment_logger=logger,
        device=device,
        loader_generator=loader_gen,
    )

    results = trainer.fit(
        epochs=cfg.epochs,
        start_epoch=start_epoch,
        verbose=True,
    )

    print(f"\n{'='*75}")
    print(f"EXPERIMENT COMPLETED: {exp_id}")
    print(f"  Total Epochs:      {results['total_epochs']}")
    print(f"  Best Val Accuracy: {results['best_val_acc']:.2%} (Epoch {results['best_epoch']})")
    print(f"  Best Val Loss:     {results['best_val_loss']:.4f}")
    print(f"  Final Train Loss:  {results['final_train_loss']:.4f} | Train Acc: {results['final_train_acc']:.2%}")
    print(f"  Final Val Loss:    {results['final_val_loss']:.4f} | Val Acc:   {results['final_val_acc']:.2%}")
    print(f"  Best Checkpoint:   {ckpt_mgr.best_checkpoint_path}")
    print(f"  Latest Checkpoint: {ckpt_mgr.latest_checkpoint_path}")
    print(f"{'='*75}\n")

    return results


def main() -> None:
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
