"""
Pure image corruption functions for the Edge Vision Reliability Lab.

All corruption functions operate on raw RGB NumPy arrays in HWC uint8 format [0, 255],
preserve image dimensions, avoid in-place modification of inputs, and return new uint8 arrays.
"""

import io
from typing import Optional
import numpy as np
from PIL import Image
import scipy.ndimage

from edge_vision.corruptions.deterministic import derive_deterministic_seed


def validate_image_array(image: np.ndarray) -> None:
    """
    Validates that input is a valid 3D RGB NumPy array with dtype uint8 in HWC format.

    Args:
        image: Array to inspect.

    Raises:
        TypeError: If input is not a NumPy array or dtype is not uint8.
        ValueError: If ndim != 3 or channel dimension != 3.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(image).__name__}")
    if image.ndim != 3:
        raise ValueError(f"Expected 3D image array of shape (H, W, C), got ndim={image.ndim} with shape {image.shape}")
    if image.shape[2] != 3:
        raise ValueError(f"Expected 3 color channels in HWC format (shape[2] == 3), got shape {image.shape}")
    if image.dtype != np.uint8:
        raise TypeError(f"Expected image dtype np.uint8, got {image.dtype}")


def apply_gaussian_noise(
    image: np.ndarray,
    std_norm: float,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
    sample_idx: Optional[int] = None,
    severity: int = 1,
    use_common_random_numbers: bool = True,
) -> np.ndarray:
    """
    Applies zero-mean additive Gaussian noise to a raw RGB image.

    Standard deviation is specified in normalized pixel units [0, 1] and scaled to
    [0, 255] for application to uint8 image pixels.

    Deterministic Randomness & Common Random Numbers (CRN):
      If `rng` is None, a local NumPy Generator is instantiated with a seed derived via
      SHA-256 digest. Under the default CRN design (use_common_random_numbers=True), the
      seed is derived strictly from (seed, sample_idx, 'gaussian_noise'), producing an
      identical standardized Gaussian noise field xi ~ N(0, 1) across all severity levels.
      The noise added is then scaled by std_norm * 255.0. This ensures strict ceteris paribus
      severity comparison: increasing severity scales the magnitude of the exact same spatial
      noise realization rather than introducing random spatial variance.
      If use_common_random_numbers=False, severity is included in the seed derivation to produce
      independent random noise streams per severity level.
      NumPy's global random state is never altered.

    Args:
        image: RGB image array of shape (H, W, 3) and dtype uint8.
        std_norm: Standard deviation in normalized pixel range [0, 1].
        rng: Optional pre-configured NumPy random generator.
        seed: Base random seed (defaults to 42 if rng is None).
        sample_idx: Sample index in dataset (defaults to 0 if rng is None).
        severity: Severity level (1-5), used in seed derivation when CRN is disabled.
        use_common_random_numbers: Whether to scale a standardized noise field across severities.

    Returns:
        New corrupted RGB image array of shape (H, W, 3) and dtype uint8.
    """
    validate_image_array(image)
    if std_norm < 0:
        raise ValueError(f"Gaussian noise std must be non-negative, got {std_norm}")

    scale_255 = float(std_norm) * 255.0

    if rng is None:
        effective_seed = 42 if seed is None else seed
        effective_idx = 0 if sample_idx is None else sample_idx

        if use_common_random_numbers:
            # Derive seed strictly from sample identity, generating common standardized noise
            derived = derive_deterministic_seed(
                seed=effective_seed,
                sample_idx=effective_idx,
                corruption_name="gaussian_noise",
                severity=None,
            )
            local_rng = np.random.default_rng(derived)
            standard_noise = local_rng.normal(loc=0.0, scale=1.0, size=image.shape)
            noise = standard_noise * scale_255
        else:
            derived = derive_deterministic_seed(
                seed=effective_seed,
                sample_idx=effective_idx,
                corruption_name="gaussian_noise",
                severity=severity,
            )
            local_rng = np.random.default_rng(derived)
            noise = local_rng.normal(loc=0.0, scale=scale_255, size=image.shape)
    else:
        noise = rng.normal(loc=0.0, scale=scale_255, size=image.shape)

    corrupted = np.clip(np.round(image.astype(np.float32) + noise), 0, 255).astype(np.uint8)
    return corrupted


def apply_gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """
    Applies 2D Gaussian blur filter to an RGB image.

    Uses `scipy.ndimage.gaussian_filter` with spatial standard deviation `sigma`
    along height (axis 0) and width (axis 1), sigma=0 along channel (axis 2),
    and reflection mode at image boundaries.

    Args:
        image: RGB image array of shape (H, W, 3) and dtype uint8.
        sigma: Standard deviation of Gaussian blur kernel in pixels.

    Returns:
        New blurred RGB image array of shape (H, W, 3) and dtype uint8.
    """
    validate_image_array(image)
    if sigma < 0:
        raise ValueError(f"Gaussian blur sigma must be non-negative, got {sigma}")
    if sigma == 0:
        return image.copy()

    # Apply spatial blur per channel independently (sigma=0 along color channel axis)
    filtered = scipy.ndimage.gaussian_filter(
        image.astype(np.float32),
        sigma=(float(sigma), float(sigma), 0.0),
        mode="reflect",
    )
    return np.clip(np.round(filtered), 0, 255).astype(np.uint8)


def apply_motion_blur(image: np.ndarray, kernel_length: int) -> np.ndarray:
    """
    Applies horizontal linear motion blur to an RGB image.

    Simulates horizontal camera or object motion during exposure using a 1D uniform
    averaging kernel of length `kernel_length` convolved along the width axis (axis 1)
    with reflection boundary conditions.

    Limitation note:
      This initial version implements pure horizontal motion blur (orientation 0 degrees).
      Arbitrary angle motion kernels may be added in future iterations if required.

    Args:
        image: RGB image array of shape (H, W, 3) and dtype uint8.
        kernel_length: Length of 1D horizontal averaging kernel (positive integer).

    Returns:
        New motion-blurred RGB image array of shape (H, W, 3) and dtype uint8.
    """
    validate_image_array(image)
    if isinstance(kernel_length, bool) or not isinstance(kernel_length, int) or kernel_length < 1:
        raise ValueError(f"Motion blur kernel length must be a positive integer, got {kernel_length!r}")
    if kernel_length == 1:
        return image.copy()

    weights = np.ones(kernel_length, dtype=np.float32) / float(kernel_length)
    filtered = scipy.ndimage.convolve1d(
        image.astype(np.float32),
        weights=weights,
        axis=1,
        mode="reflect",
    )
    return np.clip(np.round(filtered), 0, 255).astype(np.uint8)


def apply_brightness(image: np.ndarray, factor: float) -> np.ndarray:
    """
    Applies multiplicative brightness adjustment to an RGB image.

    Formula:
      I_out = clip(round(I * factor), 0, 255)

    Limitation note:
      This initial suite focuses on progressive illumination darkening (factors in [0.90, 0.30]).

    Args:
        image: RGB image array of shape (H, W, 3) and dtype uint8.
        factor: Multiplicative illumination factor (non-negative float).

    Returns:
        New brightness-adjusted RGB image array of shape (H, W, 3) and dtype uint8.
    """
    validate_image_array(image)
    if factor < 0:
        raise ValueError(f"Brightness factor must be non-negative, got {factor}")

    adjusted = image.astype(np.float32) * float(factor)
    return np.clip(np.round(adjusted), 0, 255).astype(np.uint8)


def apply_contrast(image: np.ndarray, factor: float) -> np.ndarray:
    """
    Applies linear contrast scaling around the per-channel spatial mean of an RGB image.

    Formula:
      For each channel c:
        mu_c = (1 / (H * W)) * sum_{i,j} I_{i,j,c}
        I'_{i,j,c} = clip(round(mu_c + factor * (I_{i,j,c} - mu_c)), 0, 255)

    Preserves the overall average illumination of each color channel while attenuating
    pixel variations around the channel mean as factor decreases from 1.0 toward 0.0.
    For spatially uniform/constant images, pixel values equal the mean, so the image is unchanged.

    Args:
        image: RGB image array of shape (H, W, 3) and dtype uint8.
        factor: Multiplicative contrast factor (non-negative float).

    Returns:
        New contrast-adjusted RGB image array of shape (H, W, 3) and dtype uint8.
    """
    validate_image_array(image)
    if factor < 0:
        raise ValueError(f"Contrast factor must be non-negative, got {factor}")

    img_f = image.astype(np.float32)
    # Spatial mean per channel: shape (1, 1, 3)
    channel_mean = np.mean(img_f, axis=(0, 1), keepdims=True)
    adjusted = channel_mean + float(factor) * (img_f - channel_mean)
    return np.clip(np.round(adjusted), 0, 255).astype(np.uint8)


def apply_jpeg_compression(image: np.ndarray, quality: int) -> np.ndarray:
    """
    Applies JPEG compression and decompression artifacts using in-memory encoding.

    Encodes the raw RGB image to JPEG in memory via PIL and io.BytesIO at the specified
    quality, then immediately decompresses it back to an RGB NumPy array.
    No temporary files are created on disk.

    Args:
        image: RGB image array of shape (H, W, 3) and dtype uint8.
        quality: JPEG compression quality setting (integer in [1, 100]).

    Returns:
        New compressed/decompressed RGB image array of shape (H, W, 3) and dtype uint8.
    """
    validate_image_array(image)
    if isinstance(quality, bool) or not isinstance(quality, int) or quality < 1 or quality > 100:
        raise ValueError(f"JPEG quality must be an integer between 1 and 100, got {quality!r}")

    pil_img = Image.fromarray(image, mode="RGB")
    buf = io.BytesIO()
    try:
        pil_img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        decompressed = Image.open(buf)
        if decompressed.mode != "RGB":
            decompressed = decompressed.convert("RGB")
        result = np.array(decompressed, dtype=np.uint8)
    finally:
        buf.close()

    return result
