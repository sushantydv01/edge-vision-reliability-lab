"""
Unified Inference Engine for CIFAR-10 computer vision models across precisions and devices.
"""

from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
from PIL import Image
import torch
import torch.nn as nn

from edge_vision.data.dataset import CIFAR10_CLASSES
from edge_vision.evaluation.evaluator import convert_model_to_fp16
from edge_vision.inference.exceptions import (
    DeviceCompatibilityError,
    InferenceError,
    InvalidInputError,
)
from edge_vision.inference.preprocessing import ImagePreprocessor
from edge_vision.inference.registry import ArtifactInfo, ArtifactRegistry
from edge_vision.inference.types import (
    BatchInferenceOutput,
    InferenceTiming,
    PredictionResult,
    SingleImageInferenceOutput,
)
from edge_vision.models import get_model


class InferenceEngine:
    """
    Unified high-level inference engine for executing ResNet-18 and MobileNetV3-Small
    across FP32, FP16, and INT8 precisions.

    Backends:
        - FP32: PyTorch Eager (nn.Module) on CPU or Apple Silicon MPS.
        - FP16: PyTorch Eager (nn.Module) on Apple Silicon MPS (or CPU with warning).
        - INT8: TorchScript (ScriptModule) on CPU via QNNPACK backend.
    """

    def __init__(
        self,
        model: Union[nn.Module, torch.jit.ScriptModule],
        artifact_info: ArtifactInfo,
        device: torch.device,
        backend: str,
        model_dtype: str,
        device_warning: Optional[str] = None,
    ) -> None:
        self._model = model
        self._artifact_info = artifact_info
        self._device = device
        self._backend = backend
        self._model_dtype = model_dtype
        self._device_warning = device_warning
        self._preprocessor = ImagePreprocessor(default_resize=False)

    @classmethod
    def load(
        cls,
        model_name: str,
        precision: str = "fp32",
        device: str = "auto",
        checkpoint_path: Optional[Union[str, Path]] = None,
    ) -> "InferenceEngine":
        """
        Loads and reconstructs an inference-ready model variant from its canonical artifact.

        Parameters:
            model_name: Architecture identifier (e.g. 'resnet18_cifar', 'mobilenet_v3_small_cifar').
            precision: Model precision ('fp32', 'fp16', or 'int8').
            device: Target compute device ('auto', 'cpu', or 'mps').
            checkpoint_path: Optional path to override the canonical artifact location.

        Returns:
            Configured InferenceEngine instance.

        Raises:
            UnsupportedModelError: If model_name is unrecognized.
            UnsupportedPrecisionError: If precision is unrecognized.
            ModelNotFoundError: If checkpoint artifact does not exist.
            DeviceCompatibilityError: If an incompatible precision/device combination is requested.
            InferenceError: If model deserialization or reconstruction fails.
        """
        # 1. Resolve artifact via registry
        info = ArtifactRegistry.resolve(
            model_name=model_name,
            precision=precision,
            checkpoint_path=checkpoint_path,
        )

        # 2. Resolve compute device and validate compatibility
        dev_req = device.strip().lower()
        if dev_req not in ("auto", "cpu", "mps"):
            raise DeviceCompatibilityError(
                f"Unsupported device '{device}'. Expected 'auto', 'cpu', or 'mps'."
            )

        device_warning: Optional[str] = None

        if info.precision == "int8":
            if dev_req == "mps":
                raise DeviceCompatibilityError(
                    "INT8 quantization backend (QNNPACK) is strictly supported on CPU. "
                    "Apple Silicon MPS does not support PyTorch quantized INT8 operators."
                )
            resolved_device = torch.device("cpu")
            backend = "torchscript"
            model_dtype = "qint8"
        elif info.precision == "fp16":
            backend = "pytorch_eager"
            model_dtype = "float16"
            if dev_req == "auto":
                if torch.backends.mps.is_available():
                    resolved_device = torch.device("mps")
                else:
                    resolved_device = torch.device("cpu")
                    device_warning = "CPU FP16 executed without hardware-accelerated vector GEMM."
            elif dev_req == "mps":
                if not torch.backends.mps.is_available():
                    raise DeviceCompatibilityError(
                        "MPS device requested for FP16, but torch.backends.mps.is_available() is False."
                    )
                resolved_device = torch.device("mps")
            else:  # cpu
                resolved_device = torch.device("cpu")
                device_warning = "CPU FP16 executed without hardware-accelerated vector GEMM."
        else:  # fp32
            backend = "pytorch_eager"
            model_dtype = "float32"
            if dev_req == "auto":
                if torch.backends.mps.is_available():
                    resolved_device = torch.device("mps")
                else:
                    resolved_device = torch.device("cpu")
            elif dev_req == "mps":
                if not torch.backends.mps.is_available():
                    raise DeviceCompatibilityError(
                        "MPS device requested for FP32, but torch.backends.mps.is_available() is False."
                    )
                resolved_device = torch.device("mps")
            else:  # cpu
                resolved_device = torch.device("cpu")

        # 3. Model reconstruction and checkpoint loading
        if info.precision == "int8":
            torch.backends.quantized.engine = "qnnpack"
            try:
                model = torch.jit.load(str(info.artifact_path), map_location="cpu")
            except Exception as err:
                raise InferenceError(
                    f"Failed to load TorchScript INT8 artifact at '{info.artifact_path}': {err}"
                ) from err
            model.eval()
        else:
            try:
                model = get_model(info.model_name, num_classes=10)
            except Exception as err:
                raise InferenceError(
                    f"Failed to instantiate model architecture '{info.model_name}': {err}"
                ) from err

            try:
                checkpoint = torch.load(info.artifact_path, map_location="cpu", weights_only=False)
            except Exception as err:
                raise InferenceError(
                    f"Failed to load checkpoint file at '{info.artifact_path}': {err}"
                ) from err

            if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
                raise InferenceError(
                    f"Malformed checkpoint at '{info.artifact_path}': missing 'model_state_dict'."
                )

            try:
                if info.precision == "fp16":
                    convert_model_to_fp16(model)
                    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
                else:
                    model.float()
                    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            except Exception as err:
                raise InferenceError(
                    f"Incompatible state dict for architecture '{info.model_name}' [{info.precision}] "
                    f"at '{info.artifact_path}': {err}"
                ) from err

            model.eval()
            model.to(resolved_device)

        return cls(
            model=model,
            artifact_info=info,
            device=resolved_device,
            backend=backend,
            model_dtype=model_dtype,
            device_warning=device_warning,
        )

    # -------------------------------------------------------------------------
    # Public Inference Methods
    # -------------------------------------------------------------------------

    def predict_image(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        resize: bool = False,
        top_k: int = 5,
    ) -> SingleImageInferenceOutput:
        """
        Executes prediction on a single image.

        Parameters:
            image: PIL Image, HWC NumPy array, or CHW PyTorch tensor.
            resize: Whether to explicitly resize non-32x32 inputs to (32, 32).
            top_k: Number of highest-confidence predictions to return (1 <= top_k <= 10).

        Returns:
            SingleImageInferenceOutput containing prediction, timing, and metadata.
        """
        self._validate_top_k(top_k)

        # 1. Preprocessing
        t0 = time.perf_counter()
        try:
            tensor = self._preprocessor.preprocess_image(image, resize=resize)
        except (ValueError, TypeError) as err:
            raise InvalidInputError(f"Image preprocessing failed: {err}") from err
        batch_tensor = tensor.unsqueeze(0)
        t1 = time.perf_counter()
        preprocess_ms = (t1 - t0) * 1000.0

        # 2. Forward execution
        t1 = time.perf_counter()
        batch_tensor = self._prepare_tensor_for_forward(batch_tensor)
        with torch.inference_mode():
            logits = self._model(batch_tensor)
        if self._device.type == "mps":
            torch.mps.synchronize()
        t2 = time.perf_counter()
        forward_ms = (t2 - t1) * 1000.0

        # 3. Postprocessing
        t2 = time.perf_counter()
        cpu_logits = logits.detach().to(device="cpu", dtype=torch.float32)
        cpu_probs = torch.softmax(cpu_logits, dim=-1)
        pred_res = self._format_prediction(cpu_logits[0], cpu_probs[0], top_k=top_k)
        t3 = time.perf_counter()
        postprocess_ms = (t3 - t2) * 1000.0

        r_preprocess = round(preprocess_ms, 3)
        r_forward = round(forward_ms, 3)
        r_postprocess = round(postprocess_ms, 3)
        r_total = round(r_preprocess + r_forward + r_postprocess, 3)

        timing = InferenceTiming(
            preprocess_ms=r_preprocess,
            forward_ms=r_forward,
            postprocess_ms=r_postprocess,
            total_ms=r_total,
        )

        return SingleImageInferenceOutput(
            prediction=pred_res,
            timing=timing,
            metadata=self.get_metadata(),
        )

    def predict_batch(
        self,
        images: Union[Sequence[Union[Image.Image, np.ndarray, torch.Tensor]], np.ndarray, torch.Tensor],
        resize: bool = False,
        top_k: int = 5,
    ) -> BatchInferenceOutput:
        """
        Executes prediction on a batch of images.

        Parameters:
            images: Sequence of images, 4D NHWC NumPy array, or 4D NCHW PyTorch tensor.
            resize: Whether to explicitly resize non-32x32 inputs to (32, 32).
            top_k: Number of highest-confidence predictions to return (1 <= top_k <= 10).

        Returns:
            BatchInferenceOutput containing prediction list, aggregated timing, and throughput.
        """
        self._validate_top_k(top_k)

        # 1. Preprocessing
        t0 = time.perf_counter()
        try:
            batch_tensor = self._preprocessor.preprocess_batch(images, resize=resize)
        except (ValueError, TypeError) as err:
            raise InvalidInputError(f"Batch preprocessing failed: {err}") from err
        batch_size = batch_tensor.size(0)
        t1 = time.perf_counter()
        preprocess_ms = (t1 - t0) * 1000.0

        # 2. Forward execution
        t1 = time.perf_counter()
        batch_tensor = self._prepare_tensor_for_forward(batch_tensor)
        with torch.inference_mode():
            logits = self._model(batch_tensor)
        if self._device.type == "mps":
            torch.mps.synchronize()
        t2 = time.perf_counter()
        forward_ms = (t2 - t1) * 1000.0

        # 3. Postprocessing
        t2 = time.perf_counter()
        cpu_logits = logits.detach().to(device="cpu", dtype=torch.float32)
        cpu_probs = torch.softmax(cpu_logits, dim=-1)
        predictions = [
            self._format_prediction(cpu_logits[i], cpu_probs[i], top_k=top_k)
            for i in range(batch_size)
        ]
        t3 = time.perf_counter()
        postprocess_ms = (t3 - t2) * 1000.0

        r_preprocess = round(preprocess_ms, 3)
        r_forward = round(forward_ms, 3)
        r_postprocess = round(postprocess_ms, 3)
        r_total = round(r_preprocess + r_forward + r_postprocess, 3)
        fps = (batch_size / (r_total / 1000.0)) if r_total > 0 else 0.0

        timing = InferenceTiming(
            preprocess_ms=r_preprocess,
            forward_ms=r_forward,
            postprocess_ms=r_postprocess,
            total_ms=r_total,
        )

        return BatchInferenceOutput(
            predictions=predictions,
            timing=timing,
            batch_size=batch_size,
            throughput_fps=round(fps, 1),
            metadata=self.get_metadata(),
        )

    def get_metadata(self) -> Dict[str, Any]:
        """
        Returns structured metadata describing the model, precision, runtime, and artifact.
        """
        meta: Dict[str, Any] = {
            "model_name": self._artifact_info.model_name,
            "precision": self._artifact_info.precision,
            "backend": self._backend,
            "device": str(self._device),
            "artifact_path": str(self._artifact_info.artifact_path),
            "artifact_size_bytes": self._artifact_info.file_size_bytes,
            "parameter_count": self._artifact_info.parameter_count,
            "input_dimensions": [3, 32, 32],
            "num_classes": len(CIFAR10_CLASSES),
            "class_names": list(CIFAR10_CLASSES),
            "model_dtype": self._model_dtype,
        }
        if self._artifact_info.companion_metadata:
            meta["quantization_metadata"] = self._artifact_info.companion_metadata
        if self._device_warning:
            meta["device_warning"] = self._device_warning
        return meta

    # -------------------------------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------------------------------

    def _prepare_tensor_for_forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Casts and transfers input tensor to the model's device and dtype."""
        if self._artifact_info.precision == "fp16":
            return tensor.to(device=self._device, dtype=torch.float16, non_blocking=True)
        elif self._artifact_info.precision == "int8":
            return tensor.to(device=torch.device("cpu"), dtype=torch.float32)
        else:
            return tensor.to(device=self._device, dtype=torch.float32, non_blocking=True)

    @staticmethod
    def _validate_top_k(top_k: int) -> None:
        """Validates that top_k is within allowed bounds [1, 10]."""
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1 or top_k > len(CIFAR10_CLASSES):
            raise InvalidInputError(
                f"top_k must be an integer between 1 and {len(CIFAR10_CLASSES)}, got {top_k!r}."
            )

    @staticmethod
    def _format_prediction(
        logit_vec: torch.Tensor,
        prob_vec: torch.Tensor,
        top_k: int,
    ) -> PredictionResult:
        """Constructs a PredictionResult dataclass instance from single sample vectors."""
        idx = int(torch.argmax(prob_vec).item())
        conf = float(prob_vec[idx].item())

        prob_dict = {
            CIFAR10_CLASSES[i]: round(float(prob_vec[i].item()), 6)
            for i in range(len(CIFAR10_CLASSES))
        }
        logit_list = [round(float(l.item()), 4) for l in logit_vec]

        topk_vals, topk_inds = torch.topk(prob_vec, k=top_k)
        topk_list = [
            (CIFAR10_CLASSES[int(i.item())], round(float(v.item()), 6))
            for v, i in zip(topk_vals, topk_inds)
        ]

        return PredictionResult(
            predicted_class=CIFAR10_CLASSES[idx],
            class_index=idx,
            confidence=conf,
            probabilities=prob_dict,
            logits=logit_list,
            top_k=topk_list,
        )
