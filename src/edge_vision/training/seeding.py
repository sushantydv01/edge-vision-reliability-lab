"""
Reproducibility, pseudo-random number generator seeding, and RNG state management.
"""

import random
from typing import Any, Dict, Optional
import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    Sets random seeds across Python, NumPy, PyTorch CPU, and hardware backends (MPS/CUDA).

    Parameters:
        seed: Non-negative integer random seed.
        deterministic: If True, configures PyTorch to prefer deterministic algorithms
                       (with warn_only=True to avoid runtime crashes on ops without deterministic implementations).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if torch.backends.mps.is_available() and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def get_rng_states(loader_generator: Optional[torch.Generator] = None) -> Dict[str, Any]:
    """
    Captures complete pseudo-random number generator states for epoch-boundary recovery.

    Parameters:
        loader_generator: Optional torch.Generator used by DataLoader for reproducible shuffling.

    Returns:
        Dictionary containing serialized RNG states across all active runtimes.
    """
    states: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        states["torch_cuda"] = torch.cuda.get_rng_state_all()

    if torch.backends.mps.is_available() and hasattr(torch.mps, "get_rng_state"):
        try:
            states["torch_mps"] = torch.mps.get_rng_state()
        except Exception:
            states["torch_mps"] = None

    if loader_generator is not None:
        states["loader_generator"] = loader_generator.get_state()

    return states


def set_rng_states(
    states: Dict[str, Any],
    loader_generator: Optional[torch.Generator] = None,
) -> None:
    """
    Restores captured pseudo-random number generator states.

    Parameters:
        states: Dictionary of RNG states previously captured by get_rng_states().
        loader_generator: Optional torch.Generator instance to restore.
    """
    if "python" in states and states["python"] is not None:
        random.setstate(states["python"])

    if "numpy" in states and states["numpy"] is not None:
        np.random.set_state(states["numpy"])

    if "torch_cpu" in states and states["torch_cpu"] is not None:
        torch.set_rng_state(states["torch_cpu"])

    if (
        "torch_cuda" in states
        and states["torch_cuda"] is not None
        and torch.cuda.is_available()
    ):
        torch.cuda.set_rng_state_all(states["torch_cuda"])

    if (
        "torch_mps" in states
        and states["torch_mps"] is not None
        and torch.backends.mps.is_available()
        and hasattr(torch.mps, "set_rng_state")
    ):
        try:
            torch.mps.set_rng_state(states["torch_mps"])
        except Exception:
            pass

    if (
        "loader_generator" in states
        and states["loader_generator"] is not None
        and loader_generator is not None
    ):
        loader_generator.set_state(states["loader_generator"])
