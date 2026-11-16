"""
Standalone deterministic image preprocessing pipeline for CIFAR-10 inference.
"""

from typing import Any, List, Optional, Sequence, Tuple, Union
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as F

from edge_vision.data.dataset import CIFAR10_CLASSES


class ImagePreprocessor:
    """
    Validates, converts, resizes, and normalizes image inputs for CIFAR-10 model inference.

    Supports:
        - PIL Images (RGB, L, RGBA)
        - NumPy arrays (HWC layout, uint8 or float32/float64 in [0.0, 1.0])
        - PyTorch tensors (CHW layout for single image, NCHW for batch, uint8 or float in [0.0, 1.0])

    Contracts:
        - Input array and tensor data are never mutated.
        - Canonical CIFAR-10 normalization is applied:
            mean = [0.4914, 0.4822, 0.4465]
            std  = [0.2470, 0.2435, 0.2616]
        - Output tensors are contiguous torch.float32 tensors on CPU.
        - Resizing to 32x32 is performed ONLY when explicitly requested (resize=True).
          Images with mismatched spatial dimensions are rejected when resize=False.
    """

    DEFAULT_MEAN: Tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
    DEFAULT_STD: Tuple[float, float, float] = (0.2470, 0.2435, 0.2616)
    TARGET_SIZE: Tuple[int, int] = (32, 32)

    def __init__(
        self,
        mean: Sequence[float] = DEFAULT_MEAN,
        std: Sequence[float] = DEFAULT_STD,
        default_resize: bool = False,
    ) -> None:
        """
        Initializes the preprocessor with normalization parameters.

        Parameters:
            mean: Channel-wise mean values (R, G, B).
            std: Channel-wise standard deviation values (R, G, B).
            default_resize: Default setting for spatial resizing to (32, 32).
        """
        if len(mean) != 3 or len(std) != 3:
            raise ValueError(f"Normalization mean and std must each contain exactly 3 channel values, got {len(mean)} and {len(std)}.")

        for s in std:
            if s <= 0.0:
                raise ValueError(f"Standard deviation values must be strictly positive, got {s}.")

        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
        self.default_resize = default_resize

    def preprocess_image(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        resize: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        Preprocesses a single image into a normalized float32 tensor of shape (3, 32, 32).

        Parameters:
            image: PIL Image, HWC NumPy array, or CHW PyTorch tensor.
            resize: Whether to explicitly resize non-32x32 inputs to (32, 32).
                    If None, uses self.default_resize.

        Returns:
            Contiguous torch.float32 tensor of shape (3, 32, 32).
        """
        do_resize = self.default_resize if resize is None else resize
        tensor = self._process_single(image, do_resize)
        return tensor.contiguous()

    def preprocess_batch(
        self,
        images: Union[Sequence[Union[Image.Image, np.ndarray, torch.Tensor]], np.ndarray, torch.Tensor],
        resize: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        Preprocesses a batch of images into a normalized float32 tensor of shape (N, 3, 32, 32).

        Parameters:
            images: Sequence of individual image objects, 4D NHWC NumPy array, or 4D NCHW PyTorch tensor.
            resize: Whether to explicitly resize non-32x32 inputs to (32, 32).
                    If None, uses self.default_resize.

        Returns:
            Contiguous torch.float32 tensor of shape (N, 3, 32, 32).
        """
        do_resize = self.default_resize if resize is None else resize

        # Handle batched 4D NumPy array
        if isinstance(images, np.ndarray) and images.ndim == 4:
            return self._process_numpy_batch(images, do_resize)

        # Handle batched 4D PyTorch tensor
        if isinstance(images, torch.Tensor) and images.ndim == 4:
            return self._process_tensor_batch(images, do_resize)

        # Handle sequence of single images
        if not isinstance(images, (list, tuple)):
            raise TypeError(
                f"Unsupported batch input type '{type(images).__name__}'. "
                "Expected a list/tuple of images, a 4D NumPy array (N, H, W, C), or a 4D PyTorch tensor (N, C, H, W)."
            )

        if len(images) == 0:
            raise ValueError("Empty batch provided. Expected at least one image in the sequence.")

        processed_list = [self._process_single(img, do_resize) for img in images]
        batch_tensor = torch.stack(processed_list, dim=0)
        return batch_tensor.contiguous()

    # -------------------------------------------------------------------------
    # Internal Format Handlers
    # -------------------------------------------------------------------------

    def _process_single(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        resize: bool,
    ) -> torch.Tensor:
        """Processes any single image input into a (3, 32, 32) normalized tensor."""
        if isinstance(image, Image.Image):
            return self._process_pil(image, resize)
        elif isinstance(image, np.ndarray):
            return self._process_numpy_single(image, resize)
        elif isinstance(image, torch.Tensor):
            return self._process_tensor_single(image, resize)
        else:
            raise TypeError(
                f"Unsupported image type '{type(image).__name__}'. "
                "Supported types: PIL.Image.Image, numpy.ndarray, or torch.Tensor."
            )

    def _process_pil(self, img: Image.Image, resize: bool) -> torch.Tensor:
        """Converts and normalizes a PIL Image."""
        # 1. Deterministic RGB conversion
        if img.mode != "RGB":
            # For RGBA, L, CMYK, P etc., convert deterministically to RGB
            img = img.convert("RGB")

        w, h = img.size

        # 2. Resizing validation
        if (h, w) != self.TARGET_SIZE:
            if not resize:
                raise ValueError(
                    f"Expected PIL Image spatial dimensions 32x32, got {w}x{h} (width x height). "
                    "Set resize=True to enable explicit bilinear resizing to 32x32."
                )
            img = img.resize(self.TARGET_SIZE, resample=Image.Resampling.BILINEAR)

        # 3. Convert to float32 tensor scaled to [0.0, 1.0]
        raw_arr = np.array(img, copy=True)
        tensor = torch.from_numpy(raw_arr).permute(2, 0, 1).to(dtype=torch.float32) / 255.0

        # 4. Canonical normalization
        return (tensor - self.mean) / self.std

    def _process_numpy_single(self, arr: np.ndarray, resize: bool) -> torch.Tensor:
        """Converts and normalizes a single NumPy array in HWC layout."""
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"Expected numpy.ndarray, got {type(arr).__name__}")

        # 1. Dtype validation
        if arr.dtype == np.uint8:
            is_uint8 = True
        elif arr.dtype in (np.float32, np.float64):
            is_uint8 = False
        else:
            raise TypeError(
                f"Unsupported NumPy array dtype '{arr.dtype}'. "
                "Supported dtypes: uint8 in [0, 255] or float32/float64 in [0.0, 1.0]."
            )

        # 2. Non-finite values check
        if not is_uint8 and (np.isnan(arr).any() or np.isinf(arr).any()):
            raise ValueError("NumPy array contains NaN or infinite values.")

        # 3. Dimensions and layout check
        if arr.ndim == 2:
            # Grayscale (H, W) -> expand to (H, W, 1) -> (H, W, 3)
            h, w = arr.shape
            arr_rgb = np.repeat(arr[:, :, np.newaxis], 3, axis=-1)
        elif arr.ndim == 3:
            h, w, c = arr.shape
            # Ambiguity guard: check if caller passed CHW by mistake (e.g. (3, 32, 32))
            if arr.shape[0] in (1, 3, 4) and c not in (1, 3, 4):
                raise ValueError(
                    f"Ambiguous NumPy array layout with shape {arr.shape}. "
                    "NumPy inputs must use HWC layout (e.g. (32, 32, 3)), not CHW."
                )

            if c == 1:
                arr_rgb = np.repeat(arr, 3, axis=-1)
            elif c == 3:
                arr_rgb = arr
            elif c == 4:
                # RGBA -> discard alpha channel deterministically
                arr_rgb = arr[:, :, :3]
            else:
                raise ValueError(
                    f"Expected 1, 3, or 4 channels in HWC NumPy array, got shape {arr.shape}."
                )
        else:
            raise ValueError(
                f"Unsupported NumPy array dimensions: {arr.ndim}. "
                "Expected 2D (H, W) or 3D (H, W, C) for a single image."
            )

        # 4. Range check
        if not is_uint8:
            min_val = float(arr_rgb.min()) if arr_rgb.size > 0 else 0.0
            max_val = float(arr_rgb.max()) if arr_rgb.size > 0 else 0.0
            if min_val < 0.0 or max_val > 1.0:
                raise ValueError(
                    f"Floating-point NumPy array values must be in [0.0, 1.0], "
                    f"got min={min_val:.4f}, max={max_val:.4f}."
                )

        # 5. Convert to float32 tensor (CHW) in [0.0, 1.0] without in-place mutation
        tensor = torch.from_numpy(arr_rgb.copy()).permute(2, 0, 1).to(dtype=torch.float32)
        if is_uint8:
            tensor = tensor / 255.0

        # 6. Resizing validation
        _, curr_h, curr_w = tensor.shape
        if (curr_h, curr_w) != self.TARGET_SIZE:
            if not resize:
                raise ValueError(
                    f"Expected NumPy array spatial dimensions 32x32, got {curr_h}x{curr_w} (height x width). "
                    "Set resize=True to enable explicit bilinear resizing to 32x32."
                )
            tensor = F.resize(
                tensor,
                size=list(self.TARGET_SIZE),
                interpolation=F.InterpolationMode.BILINEAR,
                antialias=True,
            )

        # 7. Canonical normalization
        return (tensor - self.mean) / self.std

    def _process_tensor_single(self, tensor: torch.Tensor, resize: bool) -> torch.Tensor:
        """Converts and normalizes a single PyTorch tensor in CHW layout."""
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(tensor).__name__}")

        # 1. Dtype validation
        if tensor.dtype == torch.uint8:
            is_uint8 = True
        elif tensor.dtype in (torch.float32, torch.float64, torch.float16):
            is_uint8 = False
        else:
            raise TypeError(
                f"Unsupported PyTorch tensor dtype '{tensor.dtype}'. "
                "Supported dtypes: torch.uint8 in [0, 255] or float in [0.0, 1.0]."
            )

        # 2. Non-finite values check
        if not is_uint8 and (torch.isnan(tensor).any() or torch.isinf(tensor).any()):
            raise ValueError("PyTorch tensor contains NaN or infinite values.")

        # 3. Dimensions and layout check (must be CHW)
        if tensor.dim() != 3:
            raise ValueError(
                f"Unsupported PyTorch tensor dimensions: {tensor.dim()}. "
                "Expected 3D tensor with shape (C, H, W) for a single image."
            )

        c, h, w = tensor.shape

        # Ambiguity guard: check if caller passed HWC tensor by mistake (e.g. (32, 32, 3))
        if c not in (1, 3, 4) and w in (1, 3, 4):
            raise ValueError(
                f"Ambiguous PyTorch tensor layout with shape {tuple(tensor.shape)}. "
                "PyTorch inputs must use CHW layout (e.g. (3, 32, 32)), not HWC."
            )

        if c == 1:
            tensor_rgb = tensor.repeat(3, 1, 1)
        elif c == 3:
            tensor_rgb = tensor.clone()
        elif c == 4:
            tensor_rgb = tensor[:3, :, :].clone()
        else:
            raise ValueError(
                f"Expected PyTorch tensor with 1, 3, or 4 channels in CHW layout, got shape {tuple(tensor.shape)}."
            )

        # 4. Range check
        if not is_uint8:
            min_val = float(tensor_rgb.min().item()) if tensor_rgb.numel() > 0 else 0.0
            max_val = float(tensor_rgb.max().item()) if tensor_rgb.numel() > 0 else 0.0
            if min_val < 0.0 or max_val > 1.0:
                raise ValueError(
                    f"Floating-point PyTorch tensor values must be in [0.0, 1.0], "
                    f"got min={min_val:.4f}, max={max_val:.4f}."
                )

        # 5. Convert to float32 on CPU
        tensor_rgb = tensor_rgb.cpu().to(dtype=torch.float32)
        if is_uint8:
            tensor_rgb = tensor_rgb / 255.0

        # 6. Resizing validation
        _, curr_h, curr_w = tensor_rgb.shape
        if (curr_h, curr_w) != self.TARGET_SIZE:
            if not resize:
                raise ValueError(
                    f"Expected PyTorch tensor spatial dimensions 32x32, got {curr_h}x{curr_w} (height x width). "
                    "Set resize=True to enable explicit bilinear resizing to 32x32."
                )
            tensor_rgb = F.resize(
                tensor_rgb,
                size=list(self.TARGET_SIZE),
                interpolation=F.InterpolationMode.BILINEAR,
                antialias=True,
            )

        # 7. Canonical normalization
        return (tensor_rgb - self.mean) / self.std

    def _process_numpy_batch(self, batch: np.ndarray, resize: bool) -> torch.Tensor:
        """Converts and normalizes a 4D NumPy array of shape (N, H, W, C)."""
        if batch.ndim != 4:
            raise ValueError(f"Expected 4D NumPy array (N, H, W, C), got {batch.ndim}D array.")

        n, h, w, c = batch.shape
        if n == 0:
            raise ValueError("Empty NumPy batch provided with N=0.")

        # Ambiguity guard: check for NCHW NumPy array
        if batch.shape[1] in (1, 3, 4) and c not in (1, 3, 4):
            raise ValueError(
                f"Ambiguous NumPy batch layout with shape {batch.shape}. "
                "NumPy batch inputs must use NHWC layout (e.g. (N, 32, 32, 3)), not NCHW."
            )

        # Process each item deterministically
        processed = [self._process_numpy_single(batch[i], resize) for i in range(n)]
        return torch.stack(processed, dim=0).contiguous()

    def _process_tensor_batch(self, batch: torch.Tensor, resize: bool) -> torch.Tensor:
        """Converts and normalizes a 4D PyTorch tensor of shape (N, C, H, W)."""
        if batch.dim() != 4:
            raise ValueError(f"Expected 4D PyTorch tensor (N, C, H, W), got {batch.dim()}D tensor.")

        n, c, h, w = batch.shape
        if n == 0:
            raise ValueError("Empty PyTorch batch provided with N=0.")

        # Ambiguity guard: check for NHWC PyTorch tensor
        if c not in (1, 3, 4) and w in (1, 3, 4):
            raise ValueError(
                f"Ambiguous PyTorch batch layout with shape {tuple(batch.shape)}. "
                "PyTorch batch inputs must use NCHW layout (e.g. (N, 3, 32, 32)), not NHWC."
            )

        # Process each item deterministically
        processed = [self._process_tensor_single(batch[i], resize) for i in range(n)]
        return torch.stack(processed, dim=0).contiguous()
