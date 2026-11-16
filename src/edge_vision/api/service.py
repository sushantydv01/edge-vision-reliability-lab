"""
Service layer managing inference model lifecycle, thread safety, and image decoding.
"""

import io
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
from fastapi import UploadFile
from PIL import Image

from edge_vision.inference import (
    ArtifactRegistry,
    BatchInferenceOutput,
    DeviceCompatibilityError,
    InferenceEngine,
    InferenceError,
    InvalidInputError,
    ModelNotFoundError,
    SingleImageInferenceOutput,
    UnsupportedModelError,
    UnsupportedPrecisionError,
)

MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB per image
MAX_BATCH_SIZE = 64  # Bounded batch ceiling
MAX_AGGREGATE_BATCH_BYTES = 64 * 1024 * 1024  # 64 MB aggregate batch limit
MAX_IMAGE_DIMENSION = 2048  # Maximum allowed width or height in pixels
CHUNK_SIZE_BYTES = 64 * 1024  # 64 KB chunk size for streaming reads
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


async def read_bounded_upload(
    upload_file: UploadFile,
    max_bytes: int = MAX_IMAGE_SIZE_BYTES,
    current_aggregate_bytes: int = 0,
    max_aggregate_bytes: Optional[int] = None,
    filename: Optional[str] = None,
    chunk_size: int = CHUNK_SIZE_BYTES,
) -> bytes:
    """
    Reads an UploadFile in bounded chunks, immediately aborting if cumulative
    bytes exceed max_bytes or aggregate limit.

    Guarantees that no more than (limit + chunk_size) bytes are accumulated in memory.
    """
    fname = filename or upload_file.filename or "upload.png"
    chunks: List[bytes] = []
    total_bytes = 0

    while True:
        chunk = await upload_file.read(chunk_size)
        if not chunk:
            break
        total_bytes += len(chunk)
        if total_bytes > max_bytes:
            raise InvalidInputError(
                f"Uploaded file '{fname}' exceeds maximum allowed size of 10 MB ({total_bytes} bytes)."
            )
        if max_aggregate_bytes is not None and (current_aggregate_bytes + total_bytes) > max_aggregate_bytes:
            limit_mb = max_aggregate_bytes // (1024 * 1024)
            raise InvalidInputError(
                f"Batch upload exceeds maximum aggregate size limit of {limit_mb} MB."
            )
        chunks.append(chunk)

    return b"".join(chunks)


def decode_image_bytes(
    data: bytes,
    filename: str = "upload.png",
    max_dimension: int = MAX_IMAGE_DIMENSION,
) -> Image.Image:
    """
    Decodes raw image bytes into a loaded, detached RGB PIL Image.

    Guarantees that input buffers are validated, bounded in size and dimensions,
    and properly closed before raster decompression.
    """
    if not data or len(data) == 0:
        raise InvalidInputError(f"Uploaded file '{filename}' is empty (0 bytes).")

    if len(data) > MAX_IMAGE_SIZE_BYTES:
        raise InvalidInputError(
            f"Uploaded file '{filename}' exceeds maximum allowed size of 10 MB ({len(data)} bytes)."
        )

    ext = Path(filename).suffix.lower()
    if ext and ext not in ALLOWED_EXTENSIONS:
        raise InvalidInputError(
            f"File '{filename}' has an unsupported extension '{ext}'. "
            f"Allowed image extensions: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
        )

    try:
        with Image.open(io.BytesIO(data)) as raw_img:
            width, height = raw_img.size
            if width <= 0 or height <= 0:
                raise InvalidInputError(
                    f"Image '{filename}' has invalid non-positive dimensions ({width}x{height})."
                )
            if width > max_dimension or height > max_dimension:
                raise InvalidInputError(
                    f"Image '{filename}' dimensions ({width}x{height}) exceed maximum allowed dimension of {max_dimension} pixels."
                )
            rgb_img = raw_img.convert("RGB")
            rgb_img.load()
            return rgb_img
    except InvalidInputError:
        raise
    except Exception as err:
        raise InvalidInputError(f"Failed to decode image from '{filename}': {err}") from err


class ModelManager:
    """
    Thread-safe model lifecycle manager.

    Maintains the currently active InferenceEngine, synchronizes concurrent inference
    calls with model-switching operations, and queries the ArtifactRegistry.
    """

    def __init__(self, initial_engine: Optional[InferenceEngine] = None) -> None:
        self._engine: Optional[InferenceEngine] = initial_engine
        self._lock = threading.RLock()

    @property
    def is_loaded(self) -> bool:
        """Returns True if an inference engine is currently resident in memory."""
        with self._lock:
            return self._engine is not None

    def list_models(self) -> List[Dict[str, Any]]:
        """
        Queries the ArtifactRegistry to return all supported model configurations.
        Does not trigger model loading.
        """
        configs = ArtifactRegistry.list_supported_configurations()
        results = []
        for model_name, precision in configs:
            info = ArtifactRegistry.resolve(model_name, precision)
            preferred_dev = "cpu" if precision == "int8" else "mps"
            backends = ["torchscript"] if precision == "int8" else ["pytorch_eager"]
            results.append({
                "model_name": info.model_name,
                "precision": info.precision,
                "supported_backends": backends,
                "preferred_device": preferred_dev,
                "file_size_bytes": info.file_size_bytes,
                "parameter_count": info.parameter_count,
            })
        return results

    def get_current_model_info(self) -> Dict[str, Any]:
        """
        Returns structured metadata for the currently active model.
        Sanitizes filesystem paths to prevent leaking host directory structures.
        """
        with self._lock:
            if self._engine is None:
                return {
                    "loaded": False,
                    "model_name": None,
                    "precision": None,
                    "device": None,
                    "backend": None,
                    "parameter_count": None,
                    "metadata": None,
                }

            raw_meta = self._engine.get_metadata()
            sanitized_meta = dict(raw_meta)
            # Sanitize host file path to basename only
            if "artifact_path" in sanitized_meta:
                sanitized_meta["artifact_path"] = Path(sanitized_meta["artifact_path"]).name

            return {
                "loaded": True,
                "model_name": sanitized_meta.get("model_name"),
                "precision": sanitized_meta.get("precision"),
                "device": sanitized_meta.get("device"),
                "backend": sanitized_meta.get("backend"),
                "parameter_count": sanitized_meta.get("parameter_count"),
                "metadata": sanitized_meta,
            }

    def load_model(
        self,
        model_name: str,
        precision: str,
        device: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Loads and switches the active model.

        To guarantee zero downtime and eliminate half-initialized states, the new engine
        is constructed and validated completely BEFORE acquiring the lock to swap the reference.
        """
        t0 = time.perf_counter()

        # 1. Instantiate the new engine outside the lock so failures do not affect active state
        target_device = device if device is not None else "auto"
        new_engine = InferenceEngine.load(
            model_name=model_name,
            precision=precision,
            device=target_device,
        )
        t_load = round((time.perf_counter() - t0) * 1000.0, 3)

        # 2. Atomically swap active reference under lock
        with self._lock:
            self._engine = new_engine
            meta = self._engine.get_metadata()

        return {
            "status": "success",
            "message": f"Successfully loaded model '{model_name}' [{precision}] on device '{meta['device']}'.",
            "model_name": meta["model_name"],
            "precision": meta["precision"],
            "device": meta["device"],
            "backend": meta["backend"],
            "load_time_ms": t_load,
        }

    def predict_single(
        self,
        image: Image.Image,
        resize: bool = False,
        top_k: int = 5,
    ) -> SingleImageInferenceOutput:
        """
        Executes prediction on a single PIL Image under thread lock.
        """
        with self._lock:
            if self._engine is None:
                raise InferenceError(
                    "No model is currently loaded. Please load a model via POST /models/load first."
                )
            return self._engine.predict_image(image, resize=resize, top_k=top_k)

    def predict_batch(
        self,
        images: List[Image.Image],
        resize: bool = False,
        top_k: int = 5,
    ) -> BatchInferenceOutput:
        """
        Executes prediction on a bounded batch of PIL Images under thread lock.
        """
        if len(images) == 0:
            raise InvalidInputError("Batch prediction requires at least one image.")

        if len(images) > MAX_BATCH_SIZE:
            raise InvalidInputError(
                f"Batch size {len(images)} exceeds maximum allowed batch size of {MAX_BATCH_SIZE} images."
            )

        with self._lock:
            if self._engine is None:
                raise InferenceError(
                    "No model is currently loaded. Please load a model via POST /models/load first."
                )
            return self._engine.predict_batch(images, resize=resize, top_k=top_k)
