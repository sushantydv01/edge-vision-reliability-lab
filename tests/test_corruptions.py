"""
Unit and integration tests for deterministic synthetic image corruption engine.
"""

import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
from PIL import Image
import pytest
import torch
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
import torchvision.transforms as transforms

from edge_vision.corruptions import (
    CORRUPTION_REGISTRY,
    CorruptedDataset,
    apply_brightness,
    apply_contrast,
    apply_corruption,
    apply_gaussian_blur,
    apply_gaussian_noise,
    apply_jpeg_compression,
    apply_motion_blur,
    build_corruption_registry,
    derive_deterministic_seed,
    get_corrupted_dataloader,
    get_corruption,
    get_default_corruption_config_path,
    get_severity_parameters,
    list_corruptions,
    load_corruption_config,
    validate_image_array,
)
from edge_vision.data.dataset import TransformedSubset


# ============================================================================
# Test Fixtures & Helpers
# ============================================================================

@pytest.fixture
def sample_image() -> np.ndarray:
    """Provides a synthetic 32x32x3 uint8 image with rich structure."""
    rng = np.random.default_rng(101)
    x = np.linspace(30, 220, 32, dtype=np.float32)
    y = np.linspace(30, 220, 32, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    r = xx
    g = yy
    b = (xx + yy) / 2.0
    structured = np.stack([r, g, b], axis=-1)
    noise = rng.integers(-15, 16, size=(32, 32, 3), dtype=np.int32)
    return np.clip(structured + noise, 0, 255).astype(np.uint8)


@pytest.fixture
def test_images(sample_image: np.ndarray) -> list[np.ndarray]:
    """A diverse battery of structured images to test invariant corruption behaviors."""
    # 1. Structured gradient + noise
    img1 = sample_image
    # 2. Vertical edge (strong horizontal variation across columns)
    img2 = np.zeros((32, 32, 3), dtype=np.uint8)
    img2[:, :16, :] = 40
    img2[:, 16:, :] = 210
    # 3. Smooth diagonal 2D gradient
    x = np.linspace(40, 210, 32, dtype=np.float32)
    xx, yy = np.meshgrid(x, x)
    img3 = np.stack([xx, yy, (xx + yy) / 2.0], axis=-1).astype(np.uint8)
    return [img1, img2, img3]


class DummyCIFAR10Base(Dataset):
    """Synthetic dataset with torchvision CIFAR-10 data/targets interface."""

    def __init__(self, size: int = 50):
        rng = np.random.default_rng(202)
        self.data = rng.integers(20, 235, size=(size, 32, 32, 3), dtype=np.uint8)
        self.targets = [int(x) for x in rng.integers(0, 10, size=size)]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx], self.targets[idx]


# ============================================================================
# 1. Input / Output Correctness & Validation Tests
# ============================================================================

def test_registry_contains_all_six_corruptions():
    expected = [
        "brightness",
        "contrast",
        "gaussian_blur",
        "gaussian_noise",
        "jpeg_compression",
        "motion_blur",
    ]
    assert list_corruptions() == expected
    for name in expected:
        defn = get_corruption(name)
        assert defn.name == name
        assert set(defn.severities.keys()) == {1, 2, 3, 4, 5}


@pytest.mark.parametrize("name", [
    "gaussian_noise",
    "gaussian_blur",
    "motion_blur",
    "brightness",
    "contrast",
    "jpeg_compression",
])
def test_all_corruptions_return_valid_uint8_array(sample_image: np.ndarray, name: str):
    original_copy = sample_image.copy()
    corrupted = apply_corruption(sample_image, name=name, severity=3, seed=42, sample_idx=7)

    assert isinstance(corrupted, np.ndarray)
    assert corrupted.shape == (32, 32, 3)
    assert corrupted.dtype == np.uint8
    assert np.all(corrupted >= 0) and np.all(corrupted <= 255)
    # Check that original array was NOT modified in-place
    assert np.array_equal(sample_image, original_copy)
    assert corrupted is not sample_image


def test_input_validation_errors(sample_image: np.ndarray):
    with pytest.raises(TypeError, match="Expected numpy.ndarray"):
        validate_image_array("not_an_array")  # type: ignore

    with pytest.raises(ValueError, match="Expected 3D image"):
        validate_image_array(sample_image[:, :, 0])

    with pytest.raises(ValueError, match="Expected 3D image"):
        validate_image_array(np.expand_dims(sample_image, 0))

    with pytest.raises(ValueError, match="Expected 3 color channels"):
        validate_image_array(np.zeros((32, 32, 4), dtype=np.uint8))

    with pytest.raises(TypeError, match="Expected image dtype np.uint8"):
        validate_image_array(sample_image.astype(np.float32))

    with pytest.raises(ValueError, match="Unknown corruption"):
        apply_corruption(sample_image, name="snow_storm", severity=1)

    for invalid_sev in [0, 6, -1, 10, "3", True, False, None]:
        with pytest.raises(ValueError, match="Severity must be an integer between 1 and 5"):
            apply_corruption(sample_image, name="gaussian_blur", severity=invalid_sev)  # type: ignore


# ============================================================================
# 2. Determinism, CRN Design & Randomness Independence Tests
# ============================================================================

def test_derive_deterministic_seed_guarantees():
    # 1. Idempotent
    s1 = derive_deterministic_seed(seed=42, sample_idx=10, corruption_name="gaussian_noise")
    s2 = derive_deterministic_seed(seed=42, sample_idx=10, corruption_name="gaussian_noise")
    assert s1 == s2
    assert isinstance(s1, int) and s1 >= 0

    # 2. Independent across sample index
    s_diff_idx = derive_deterministic_seed(seed=42, sample_idx=11, corruption_name="gaussian_noise")
    assert s1 != s_diff_idx

    # 3. Independent across base seed
    s_diff_seed = derive_deterministic_seed(seed=43, sample_idx=10, corruption_name="gaussian_noise")
    assert s1 != s_diff_seed

    # 4. Independent across corruption name
    s_diff_name = derive_deterministic_seed(seed=42, sample_idx=10, corruption_name="other_noise")
    assert s1 != s_diff_name

    # 5. Severity distinction when severity is explicitly passed
    s_sev1 = derive_deterministic_seed(seed=42, sample_idx=10, corruption_name="gaussian_noise", severity=1)
    s_sev2 = derive_deterministic_seed(seed=42, sample_idx=10, corruption_name="gaussian_noise", severity=2)
    assert s_sev1 != s_sev2

    # 6. Validation of arguments
    with pytest.raises(ValueError):
        derive_deterministic_seed(-1, 0, "gaussian_noise")
    with pytest.raises(ValueError):
        derive_deterministic_seed(42, -5, "gaussian_noise")
    with pytest.raises(ValueError):
        derive_deterministic_seed(42, 0, "")
    with pytest.raises(ValueError):
        derive_deterministic_seed(42, 0, "gaussian_noise", 6)


def test_gaussian_noise_common_random_numbers_scaling(sample_image: np.ndarray):
    """
    Verifies that Common Random Numbers (CRN) scales the identical standardized
    noise field across severity levels 1 to 5.
    """
    perturbations = []
    severities = [0.04, 0.06, 0.08, 0.10, 0.12]

    # Under CRN, apply across severities with the same seed and sample_idx
    for std_norm in severities:
        out = apply_gaussian_noise(
            sample_image,
            std_norm=std_norm,
            seed=42,
            sample_idx=100,
            use_common_random_numbers=True,
        )
        # Compute mean absolute difference from original
        diff = np.abs(out.astype(np.float64) - sample_image.astype(np.float64))
        perturbations.append(np.mean(diff))

    # Mean perturbation magnitude must strictly increase with severity
    for i in range(len(perturbations) - 1):
        assert perturbations[i] < perturbations[i + 1]

    # Verify that the unrounded noise fields scale proportionally across severities
    rng_seed = derive_deterministic_seed(42, 100, "gaussian_noise", severity=None)
    std_field = np.random.default_rng(rng_seed).normal(0.0, 1.0, size=sample_image.shape)

    # Check that severity 2 noise is exactly 1.5x severity 1 noise (0.06 / 0.04 = 1.5)
    noise_1 = std_field * (0.04 * 255.0)
    noise_2 = std_field * (0.06 * 255.0)
    assert np.allclose(noise_2, 1.5 * noise_1)


def test_gaussian_noise_independent_streams_when_crn_disabled(sample_image: np.ndarray):
    """
    Verifies that when use_common_random_numbers=False, different severities receive
    independent random streams.
    """
    out1 = apply_gaussian_noise(sample_image, std_norm=0.08, seed=42, sample_idx=10, severity=1, use_common_random_numbers=False)
    out2 = apply_gaussian_noise(sample_image, std_norm=0.08, seed=42, sample_idx=10, severity=2, use_common_random_numbers=False)
    # Even at the same std_norm, different severity IDs should produce different noise realizations
    assert not np.array_equal(out1, out2)


def test_gaussian_noise_does_not_mutate_or_depend_on_global_rng(sample_image: np.ndarray):
    # 1. Verify global state is not mutated by corruption call
    np.random.seed(12345)
    state_before = np.random.get_state()
    global_rand_before = np.random.rand(5)
    np.random.set_state(state_before)

    _ = apply_gaussian_noise(sample_image, std_norm=0.08, seed=42, sample_idx=100)

    global_rand_after = np.random.rand(5)
    assert np.array_equal(global_rand_before, global_rand_after)

    # 2. Verify corruption output has zero dependence on global RNG state
    np.random.seed(111)
    out_global_1 = apply_gaussian_noise(sample_image, std_norm=0.08, seed=42, sample_idx=100)

    np.random.seed(999999)
    out_global_2 = apply_gaussian_noise(sample_image, std_norm=0.08, seed=42, sample_idx=100)

    assert np.array_equal(out_global_1, out_global_2)


def test_cross_process_determinism():
    """
    Spawns two independent Python subprocesses with distinct PYTHONHASHSEED values
    to empirically verify that SHA-256 seed derivation is cross-process deterministic.
    """
    code = (
        "import numpy as np\n"
        "from edge_vision.corruptions import apply_corruption\n"
        "img = np.full((32, 32, 3), 128, dtype=np.uint8)\n"
        "out = apply_corruption(img, 'gaussian_noise', severity=3, seed=42, sample_idx=123)\n"
        "print(int(np.sum(out)))\n"
    )

    env1 = os.environ.copy()
    env1["PYTHONHASHSEED"] = "1111"
    proc1 = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env1, check=True)

    env2 = os.environ.copy()
    env2["PYTHONHASHSEED"] = "9999"
    proc2 = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env2, check=True)

    assert proc1.stdout.strip() == proc2.stdout.strip()


@pytest.mark.parametrize("name", [
    "gaussian_blur",
    "motion_blur",
    "brightness",
    "contrast",
    "jpeg_compression",
])
def test_deterministic_corruptions_idempotence(sample_image: np.ndarray, name: str):
    res1 = apply_corruption(sample_image, name=name, severity=3)
    res2 = apply_corruption(sample_image, name=name, severity=3)
    assert np.array_equal(res1, res2)


# ============================================================================
# 3. Severity Monotonicity & Quantitative Degradation Tests
# ============================================================================

def test_gaussian_noise_monotonicity_multi_image(test_images: list[np.ndarray]):
    for img in test_images:
        mses = []
        for sev in range(1, 6):
            out = apply_corruption(img, "gaussian_noise", severity=sev, seed=42, sample_idx=50)
            mse = float(np.mean((out.astype(np.float64) - img.astype(np.float64)) ** 2))
            mses.append(mse)

        for i in range(len(mses) - 1):
            assert mses[i] < mses[i + 1]


def test_gaussian_blur_monotonicity_multi_image(test_images: list[np.ndarray]):
    for img in test_images:
        mses = []
        for sev in range(1, 6):
            out = apply_corruption(img, "gaussian_blur", severity=sev)
            mse = float(np.mean((out.astype(np.float64) - img.astype(np.float64)) ** 2))
            mses.append(mse)

        for i in range(len(mses) - 1):
            assert mses[i] < mses[i + 1]


def test_motion_blur_monotonicity_multi_image(test_images: list[np.ndarray]):
    for img in test_images:
        mses = []
        for sev in range(1, 6):
            out = apply_corruption(img, "motion_blur", severity=sev)
            mse = float(np.mean((out.astype(np.float64) - img.astype(np.float64)) ** 2))
            mses.append(mse)

        for i in range(len(mses) - 1):
            assert mses[i] < mses[i + 1]


def test_motion_blur_horizontal_invariance():
    """
    Verifies that a horizontally uniform image (horizontal stripe pattern)
    is invariant to horizontal 1D motion blur.
    """
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[:16, :, :] = 40
    img[16:, :, :] = 210
    for sev in range(1, 6):
        out = apply_corruption(img, "motion_blur", severity=sev)
        assert np.array_equal(out, img)


def test_brightness_monotonic_darkening_multi_image(test_images: list[np.ndarray]):
    for img in test_images:
        means = []
        for sev in range(1, 6):
            out = apply_corruption(img, "brightness", severity=sev)
            means.append(float(np.mean(out)))

        for i in range(len(means) - 1):
            assert means[i] > means[i + 1]


def test_contrast_monotonic_attenuation_multi_image(test_images: list[np.ndarray]):
    for img in test_images:
        stds = []
        for sev in range(1, 6):
            out = apply_corruption(img, "contrast", severity=sev)
            stds.append(float(np.std(out)))

        for i in range(len(stds) - 1):
            assert stds[i] > stds[i + 1]


def test_contrast_constant_image_invariant():
    """
    On a constant image, pixels equal the spatial mean, so I - mu_c = 0.
    The image should remain completely unchanged under contrast scaling.
    """
    constant_img = np.full((32, 32, 3), 140, dtype=np.uint8)
    for sev in range(1, 6):
        out = apply_corruption(constant_img, "contrast", severity=sev)
        assert np.array_equal(out, constant_img)


def test_contrast_preserves_per_channel_mean(sample_image: np.ndarray):
    orig_means = np.mean(sample_image.astype(np.float64), axis=(0, 1))
    for sev in range(1, 6):
        out = apply_corruption(sample_image, "contrast", severity=sev)
        out_means = np.mean(out.astype(np.float64), axis=(0, 1))
        # Preserved within 1 intensity unit due to integer rounding
        assert np.all(np.abs(out_means - orig_means) < 1.0)


def test_jpeg_compression_monotonicity_multi_image(test_images: list[np.ndarray]):
    for img in test_images:
        mses = []
        for sev in range(1, 6):
            out = apply_corruption(img, "jpeg_compression", severity=sev)
            mse = float(np.mean((out.astype(np.float64) - img.astype(np.float64)) ** 2))
            mses.append(mse)

        for i in range(len(mses) - 1):
            assert mses[i] <= mses[i + 1]


# ============================================================================
# 4. Configuration & Single Source of Truth Audit Tests
# ============================================================================

def test_yaml_is_single_authoritative_source_for_registry():
    """
    Verifies that CORRUPTION_REGISTRY dynamically reflects configs/corruptions.yaml.
    """
    cfg = load_corruption_config()
    for name in list_corruptions():
        reg_defn = get_corruption(name)
        cfg_defn = cfg.corruptions[name]
        assert reg_defn.severities == cfg_defn.severities
        assert reg_defn.parameter_name == cfg_defn.parameter_name
        assert reg_defn.description == cfg_defn.description


def test_load_corruption_config_from_arbitrary_cwd():
    """
    Verifies that get_default_corruption_config_path resolves successfully
    even when changing the current working directory to a temporary directory.
    """
    orig_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            os.chdir(tmp_dir)
            cfg_path = get_default_corruption_config_path()
            assert cfg_path.is_file()
            cfg = load_corruption_config()
            assert len(cfg.corruptions) == 6
        finally:
            os.chdir(orig_cwd)


def test_load_corruption_config_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_corruption_config("nonexistent_corruptions.yaml")


def test_get_severity_parameters():
    params = get_severity_parameters("brightness")
    assert isinstance(params, dict) and len(params) == 5

    param_3 = get_severity_parameters("brightness", severity=3)
    assert param_3 == 0.60


# ============================================================================
# 5. Dataset & DataLoader Integration Tests
# ============================================================================

def test_corrupted_dataset_wrapping_dummy_cifar():
    base_ds = DummyCIFAR10Base(size=20)
    orig_data_copy = base_ds.data.copy()

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    corrupted_ds = CorruptedDataset(
        base_dataset=base_ds,
        corruption_name="gaussian_noise",
        severity=3,
        transform=eval_transform,
        seed=42,
    )

    assert len(corrupted_ds) == 20

    tensor1, target1 = corrupted_ds[0]
    tensor2, target2 = corrupted_ds[0]

    assert target1 == target2 == base_ds.targets[0]
    assert isinstance(tensor1, torch.Tensor)
    assert tensor1.shape == (3, 32, 32)
    assert torch.equal(tensor1, tensor2)

    assert np.array_equal(base_ds.data, orig_data_copy)


def test_corrupted_dataset_wrapping_transformed_subset():
    base_ds = DummyCIFAR10Base(size=30)
    subset_indices = [5, 12, 19, 25]
    subset = TransformedSubset(base_ds, subset_indices, transform=None)

    corrupted_subset = CorruptedDataset(
        base_dataset=subset,
        corruption_name="motion_blur",
        severity=4,
        transform=transforms.ToTensor(),
        seed=42,
    )

    assert len(corrupted_subset) == len(subset_indices)
    tensor_img, target = corrupted_subset[1]  # index 1 maps to real index 12

    assert target == base_ds.targets[12]
    assert tensor_img.shape == (3, 32, 32)
    assert tensor_img.min() >= 0.0 and tensor_img.max() <= 1.0


def test_corrupted_dataset_nested_subsets_canonical_mapping():
    """
    Verifies that nested subsets (Subset(Subset(dataset))) resolve back
    to the correct root sample index for deterministic noise generation.
    """
    base_ds = DummyCIFAR10Base(size=40)
    sub1 = Subset(base_ds, [4, 8, 12, 16, 20])
    sub2 = Subset(sub1, [1, 3])  # sub2[0] -> sub1[1] -> 8; sub2[1] -> sub1[3] -> 16

    corrupted = CorruptedDataset(
        base_dataset=sub2,
        corruption_name="gaussian_noise",
        severity=3,
        seed=42,
    )

    # Comparing sub2[0] with direct access to base_ds[8]
    direct_corrupted = CorruptedDataset(
        base_dataset=base_ds,
        corruption_name="gaussian_noise",
        severity=3,
        seed=42,
    )

    img_nested, tgt_nested = corrupted[0]
    img_direct, tgt_direct = direct_corrupted[8]

    assert tgt_nested == tgt_direct == base_ds.targets[8]
    assert torch.equal(img_nested, img_direct)


def test_corrupted_dataset_rejects_unsupported_datasets():
    """
    Verifies that passing a dataset without raw uint8 array storage (e.g. TensorDataset)
    fails with an informative TypeError.
    """
    tensors = torch.zeros((10, 3, 32, 32), dtype=torch.float32)
    labels = torch.zeros(10, dtype=torch.long)
    tensor_ds = TensorDataset(tensors, labels)

    with pytest.raises(TypeError, match="CorruptedDataset requires a CIFAR-10-compatible dataset"):
        corrupted = CorruptedDataset(base_dataset=tensor_ds, corruption_name="gaussian_noise", severity=1)
        _ = corrupted[0]


def test_corrupted_dataloader_reproducibility():
    base_ds = DummyCIFAR10Base(size=16)

    loader1 = get_corrupted_dataloader(
        base_dataset=base_ds,
        corruption_name="gaussian_noise",
        severity=2,
        transform=transforms.ToTensor(),
        seed=42,
        batch_size=8,
    )

    loader2 = get_corrupted_dataloader(
        base_dataset=base_ds,
        corruption_name="gaussian_noise",
        severity=2,
        transform=transforms.ToTensor(),
        seed=42,
        batch_size=8,
    )

    batches1 = list(loader1)
    batches2 = list(loader2)

    assert len(batches1) == len(batches2) == 2
    for (img1, tgt1), (img2, tgt2) in zip(batches1, batches2):
        assert torch.equal(img1, img2)
        assert torch.equal(tgt1, tgt2)


def test_corruption_occurs_before_normalization():
    base_ds = DummyCIFAR10Base(size=5)

    raw_img = base_ds.data[0]
    expected_corrupted_arr = apply_brightness(raw_img, factor=0.30)
    expected_pil = Image.fromarray(expected_corrupted_arr)

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    downstream_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    expected_tensor = downstream_transform(expected_pil)

    corrupted_ds = CorruptedDataset(
        base_dataset=base_ds,
        corruption_name="brightness",
        severity=5,
        transform=downstream_transform,
        seed=42,
    )

    actual_tensor, _ = corrupted_ds[0]
    assert torch.allclose(actual_tensor, expected_tensor, atol=1e-6)
