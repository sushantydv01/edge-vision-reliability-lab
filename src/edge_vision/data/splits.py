"""
Deterministic train/validation splitting, compact manifest management, and checksum verification.
"""

import hashlib
import json
import os
from pathlib import Path
import re
import struct
from typing import Dict, List, Optional, Sequence, Tuple
import torch

SUPPORTED_ALGORITHMS = ("torch.randperm_v1",)
HEX_64_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def compute_indices_checksum(indices: Sequence[int]) -> str:
    """
    Compute a deterministic SHA-256 checksum for an integer index sequence
    using fixed-width 32-bit unsigned little-endian binary serialization (<I).
    Guarantees platform-independent, byte-order-controlled hashing.

    Args:
        indices: Sequence of non-negative integer indices.

    Returns:
        64-character lowercase hex SHA-256 digest string.
    """
    byte_buffer = struct.pack(f"<{len(indices)}I", *indices)
    return hashlib.sha256(byte_buffer).hexdigest()


def validate_split_indices(
    train_indices: Sequence[int],
    val_indices: Sequence[int],
    total_samples: int = 50000,
) -> None:
    """
    Strictly validate train and validation partition properties:
    lengths, disjointness, index ranges, uniqueness, and full coverage.
    """
    if not isinstance(train_indices, (list, tuple)) or not isinstance(val_indices, (list, tuple)):
        raise TypeError("Train and validation indices must be lists or tuples of integers.")

    train_len = len(train_indices)
    val_len = len(val_indices)

    if train_len + val_len != total_samples:
        raise ValueError(
            f"Index counts ({train_len} + {val_len} = {train_len + val_len}) "
            f"do not match total expected samples ({total_samples})."
        )

    set_train = set(train_indices)
    set_val = set(val_indices)

    if len(set_train) != train_len:
        raise ValueError(f"Training indices contain {train_len - len(set_train)} duplicate entries.")

    if len(set_val) != val_len:
        raise ValueError(f"Validation indices contain {val_len - len(set_val)} duplicate entries.")

    overlap = set_train.intersection(set_val)
    if len(overlap) > 0:
        raise ValueError(f"Train and validation splits overlap by {len(overlap)} samples: {list(overlap)[:5]}...")

    union_set = set_train.union(set_val)
    expected_set = set(range(total_samples))
    if union_set != expected_set:
        missing = expected_set - union_set
        raise ValueError(f"Split indices do not completely cover the dataset (missing {len(missing)} samples).")


def validate_manifest_schema(manifest: Dict) -> None:
    """
    Strictly validate that a loaded manifest conforms to the required schema,
    types, value ranges, and checksum formatting.
    """
    if not isinstance(manifest, dict):
        raise ValueError("Manifest payload must be a JSON object (dictionary).")

    required_fields = {
        "dataset": str,
        "total_samples": int,
        "train_size": int,
        "val_size": int,
        "seed": int,
        "algorithm": str,
        "pytorch_version": str,
        "train_indices_checksum": str,
        "val_indices_checksum": str,
    }

    for field, expected_type in required_fields.items():
        if field not in manifest:
            raise ValueError(f"Incompatible split manifest; missing required field '{field}'.")
        val = manifest[field]
        # In Python, bool is a subclass of int, so disallow bools for int fields
        if expected_type is int and isinstance(val, bool):
            raise ValueError(f"Field '{field}' must be an integer, not a boolean.")
        if not isinstance(val, expected_type):
            raise ValueError(
                f"Field '{field}' has invalid type {type(val).__name__}; expected {expected_type.__name__}."
            )

    total = manifest["total_samples"]
    train_size = manifest["train_size"]
    val_size = manifest["val_size"]

    if total <= 0 or train_size <= 0 or val_size <= 0:
        raise ValueError(f"Sizes must be strictly positive (got total={total}, train={train_size}, val={val_size}).")

    if train_size + val_size != total:
        raise ValueError(
            f"Inconsistent split metadata: train_size ({train_size}) + val_size ({val_size}) != total ({total})."
        )

    if manifest["algorithm"] not in SUPPORTED_ALGORITHMS:
        raise ValueError(
            f"Unsupported algorithm '{manifest['algorithm']}'; supported: {list(SUPPORTED_ALGORITHMS)}."
        )

    for cksum_field in ("train_indices_checksum", "val_indices_checksum"):
        cksum_val = manifest[cksum_field]
        if not HEX_64_PATTERN.match(cksum_val):
            raise ValueError(
                f"Field '{cksum_field}' must be a valid 64-character lowercase hex SHA-256 digest (got '{cksum_val}')."
            )


def generate_train_val_split_indices(
    total_samples: int = 50000,
    train_size: int = 45000,
    val_size: int = 5000,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """
    Generate deterministic non-overlapping train and validation indices using PyTorch CPU generator.

    Args:
        total_samples: Total number of samples in the base training set (e.g. 50000).
        train_size: Number of samples in training subset (e.g. 45000).
        val_size: Number of samples in validation subset (e.g. 5000).
        seed: Random seed for deterministic permutation.

    Returns:
        Tuple of (train_indices, val_indices).
    """
    if train_size + val_size != total_samples:
        raise ValueError(
            f"train_size ({train_size}) + val_size ({val_size}) must equal total_samples ({total_samples})"
        )

    generator = torch.Generator().manual_seed(seed)
    shuffled_indices = torch.randperm(total_samples, generator=generator).tolist()

    train_indices = shuffled_indices[:train_size]
    val_indices = shuffled_indices[train_size:]

    validate_split_indices(train_indices, val_indices, total_samples=total_samples)
    return train_indices, val_indices


def save_split_manifest(
    train_indices: Sequence[int],
    val_indices: Sequence[int],
    manifest_path: str = "configs/cifar10_split_manifest.json",
    seed: int = 42,
    algorithm: str = "torch.randperm_v1",
    metadata: Optional[Dict] = None,
) -> Dict:
    """
    Atomically save a compact reproducibility manifest containing split metadata and SHA-256 checksums.
    Avoids committing oversized 50k integer lists to Git.

    Uses atomic replacement via a temporary file in the same directory to prevent corrupted/partial writes.
    """
    validate_split_indices(train_indices, val_indices, total_samples=len(train_indices) + len(val_indices))

    train_cksum = compute_indices_checksum(train_indices)
    val_cksum = compute_indices_checksum(val_indices)

    payload = {
        "dataset": "cifar10",
        "total_samples": len(train_indices) + len(val_indices),
        "train_size": len(train_indices),
        "val_size": len(val_indices),
        "seed": seed,
        "algorithm": algorithm,
        "pytorch_version": torch.__version__,
        "train_indices_checksum": train_cksum,
        "val_indices_checksum": val_cksum,
        "sample_verification": {
            "first_5_train_indices": list(train_indices[:5]),
            "last_5_train_indices": list(train_indices[-5:]),
            "first_5_val_indices": list(val_indices[:5]),
            "last_5_val_indices": list(val_indices[-5:]),
        },
    }
    if metadata:
        payload["metadata"] = metadata

    validate_manifest_schema(payload)

    out_path = Path(manifest_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Atomic write pattern: write to tmp file in same directory, flush, fsync, and replace atomically
    tmp_path = out_path.with_suffix(f".tmp.{os.getpid()}")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, out_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    return payload


def load_split_manifest(
    manifest_path: str = "configs/cifar10_split_manifest.json",
) -> Dict:
    """
    Load, parse, and strictly validate a compact split manifest.
    """
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"Split manifest not found at: {manifest_path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Malformed JSON in split manifest at {manifest_path}: {e}")

    validate_manifest_schema(data)
    return data


def get_or_create_split_indices(
    manifest_path: str = "configs/cifar10_split_manifest.json",
    total_samples: int = 50000,
    train_size: int = 45000,
    val_size: int = 5000,
    seed: int = 42,
    force_regenerate: bool = False,
    verify_checksum: bool = True,
) -> Tuple[List[int], List[int]]:
    """
    Deterministically retrieve or generate train/val split indices.

    - Reuses existing manifest only if the seed, total samples, and split sizes match.
    - If the requested seed differs, generates the split for the requested seed.
    - Verifies generated index checksums against manifest if manifest exists and matches seed.
    - If force_regenerate=True, regenerates the split and overwrites the manifest atomically.
    """
    path = Path(manifest_path)

    # 1. Force regeneration or missing manifest
    if force_regenerate or not path.is_file():
        train_idx, val_idx = generate_train_val_split_indices(
            total_samples=total_samples,
            train_size=train_size,
            val_size=val_size,
            seed=seed,
        )
        save_split_manifest(
            train_indices=train_idx,
            val_indices=val_idx,
            manifest_path=manifest_path,
            seed=seed,
        )
        return train_idx, val_idx

    # 2. Manifest exists: parse and check metadata compatibility
    manifest = load_split_manifest(manifest_path)
    manifest_seed = manifest["seed"]
    manifest_train_size = manifest["train_size"]
    manifest_val_size = manifest["val_size"]
    manifest_total = manifest["total_samples"]

    metadata_matches = (
        manifest_seed == seed
        and manifest_train_size == train_size
        and manifest_val_size == val_size
        and manifest_total == total_samples
    )

    if not metadata_matches:
        # Configuration mismatch: requested seed or split parameters differ from manifest.
        # Regenerate deterministically for requested parameters without silently returning stale split.
        train_idx, val_idx = generate_train_val_split_indices(
            total_samples=total_samples,
            train_size=train_size,
            val_size=val_size,
            seed=seed,
        )
        return train_idx, val_idx

    # 3. Metadata matches: generate indices for this seed and verify checksum
    train_idx, val_idx = generate_train_val_split_indices(
        total_samples=total_samples,
        train_size=train_size,
        val_size=val_size,
        seed=seed,
    )

    if verify_checksum:
        actual_train_cksum = compute_indices_checksum(train_idx)
        actual_val_cksum = compute_indices_checksum(val_idx)

        expected_train_cksum = manifest["train_indices_checksum"]
        expected_val_cksum = manifest["val_indices_checksum"]

        if actual_train_cksum != expected_train_cksum or actual_val_cksum != expected_val_cksum:
            raise RuntimeError(
                f"Checksum mismatch for split seed {seed} against manifest {manifest_path}.\n"
                f"Expected train checksum: {expected_train_cksum}\n"
                f"Actual train checksum:   {actual_train_cksum}\n"
                f"Expected val checksum:   {expected_val_cksum}\n"
                f"Actual val checksum:     {actual_val_cksum}\n"
                f"PyTorch version used in manifest: {manifest.get('pytorch_version')} "
                f"(current: {torch.__version__}).\n"
                f"To intentionally update the manifest for the current environment, set force_regenerate=True."
            )

    return train_idx, val_idx
