"""
Five-epoch convergence pilot runner for ResNet-18 and MobileNetV3-Small CIFAR-10 baselines.
Uses full 45,000 training and 5,000 validation samples on Apple Silicon MPS backend.
"""

from pathlib import Path
import psutil
import torch
import torch.nn as nn

from edge_vision.models import get_model
from edge_vision.training.checkpoint import CheckpointManager
from edge_vision.training.config import TrainingConfig, load_training_config
from edge_vision.training.data import get_training_dataloaders
from edge_vision.training.experiment import ExperimentLogger
from edge_vision.training.seeding import set_seed
from edge_vision.training.trainer import Trainer


def run_pilot(model_name: str, epochs: int = 5) -> dict:
    set_seed(42)
    device = torch.device("mps")

    cfg = load_training_config("configs/train_baseline.yaml")
    cfg.experiment_id = f"pilot_{model_name}"
    cfg.model_architecture = model_name
    cfg.epochs = epochs
    cfg.device = "mps"
    cfg.output_dir = "experiments"

    exp_dir = Path("experiments") / cfg.experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, generator = get_training_dataloaders(cfg)

    model = get_model(model_name, num_classes=10)
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

    ckpt_mgr = CheckpointManager(exp_dir)
    logger = ExperimentLogger(exp_dir, cfg.experiment_id)

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
        loader_generator=generator,
    )

    print(f"\n{'='*70}\nSTARTING {epochs}-EPOCH CONVERGENCE PILOT: {model_name} on MPS\n{'='*70}")
    results = trainer.fit(epochs=epochs, start_epoch=1)

    print(f"\nPILOT COMPLETE: {model_name}")
    print(f"  Best Val Accuracy: {results['best_val_acc']:.2%} (Epoch {results['best_epoch']})")
    print(f"  Best Val Loss:     {results['best_val_loss']:.4f}")
    print(f"  Final Train Loss:  {results['final_train_loss']:.4f} | Final Train Acc: {results['final_train_acc']:.2%}")
    print(f"  Final Val Loss:    {results['final_val_loss']:.4f} | Final Val Acc:   {results['final_val_acc']:.2%}")

    return results


def main():
    print("=" * 80)
    print("RUNNING 5-EPOCH CONVERGENCE PILOTS ON APPLE SILICON MPS")
    print("=" * 80)

    # 1. ResNet-18 Pilot
    res_r18 = run_pilot("resnet18_cifar", epochs=5)

    # Clean cache between runs
    if hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()

    # 2. MobileNetV3-Small Pilot
    res_mb3 = run_pilot("mobilenet_v3_small_cifar", epochs=5)


if __name__ == "__main__":
    main()
