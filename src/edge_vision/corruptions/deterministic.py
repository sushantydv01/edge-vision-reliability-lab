"""
Deterministic seed derivation for stochastic image corruptions.
"""

import hashlib
import struct
from typing import Optional


def derive_deterministic_seed(
    seed: int,
    sample_idx: int,
    corruption_name: str,
    severity: Optional[int] = None,
) -> int:
    """
    Derives a deterministic, 64-bit unsigned integer seed for a local NumPy random generator.

    Guarantees:
      - Independent of Python's randomized hash() process seed (uses SHA-256 cryptographic digest).
      - Independent of DataLoader worker scheduling and batch iteration order.
      - Independent across corruption types and sample indices.
      - Identical output across platforms and Python versions for identical inputs.
      - Never mutates NumPy's global random state.

    Common Random Numbers (CRN) Design:
      When severity is omitted (None), the seed depends only on (seed, sample_idx, corruption_name).
      This enables generating a standardized random noise field that can be scaled proportionally
      across severity levels (Common Random Numbers). When severity is provided, the seed is
      severity-specific, generating independent noise streams per severity level.

    Args:
        seed: Base experiment random seed (non-negative integer).
        sample_idx: Canonical index of the image sample in the dataset (non-negative integer).
        corruption_name: Name of the corruption operation (non-empty string).
        severity: Optional severity level (integer between 1 and 5 inclusive, or None for CRN).

    Returns:
        64-bit unsigned integer suitable for np.random.default_rng(derived_seed).

    Raises:
        ValueError: If seed, sample_idx, or severity are invalid.
        TypeError: If argument types are incorrect.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"Seed must be a non-negative integer, got {seed!r}")
    if isinstance(sample_idx, bool) or not isinstance(sample_idx, int) or sample_idx < 0:
        raise ValueError(f"Sample index must be a non-negative integer, got {sample_idx!r}")
    if not isinstance(corruption_name, str) or not corruption_name.strip():
        raise ValueError(f"Corruption name must be a non-empty string, got {corruption_name!r}")

    if severity is not None:
        if isinstance(severity, bool) or not isinstance(severity, int) or severity < 1 or severity > 5:
            raise ValueError(f"Severity must be an integer in range [1, 5], got {severity!r}")
        token = f"{seed}:{sample_idx}:{corruption_name.strip()}:{severity}"
    else:
        token = f"{seed}:{sample_idx}:{corruption_name.strip()}"

    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return struct.unpack(">Q", digest[:8])[0]
