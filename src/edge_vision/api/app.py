"""
FastAPI application factory and route definitions for the inference REST API.
"""

import logging
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from edge_vision.api.schemas import (
    BatchItemPrediction,
    BatchPredictionResponse,
    CurrentModelResponse,
    ErrorResponse,
    HealthResponse,
    InferenceTimingSchema,
    ModelArtifactSummary,
    ModelLoadRequest,
    ModelLoadResponse,
    ModelsListResponse,
    PredictionDetail,
    SinglePredictionResponse,
    TopKPrediction,
)
from edge_vision.api.service import (
    MAX_AGGREGATE_BATCH_BYTES,
    MAX_BATCH_SIZE,
    MAX_IMAGE_SIZE_BYTES,
    ModelManager,
    decode_image_bytes,
    read_bounded_upload,
)
from edge_vision.inference import (
    DeviceCompatibilityError,
    InferenceError,
    InvalidInputError,
    ModelNotFoundError,
    PredictionResult,
    UnsupportedModelError,
    UnsupportedPrecisionError,
)

logger = logging.getLogger("edge_vision.api")


def sanitize_message(msg: str) -> str:
    """
    Strips internal filesystem paths and host-specific directory references
    to prevent information disclosure in error responses.
    """
    # Replace absolute unix paths matching /... with their file basenames
    clean = re.sub(r"(/[a-zA-Z0-9_\.\-]+)+/([a-zA-Z0-9_\.\-]+)", r"\2", str(msg))
    return clean


def format_prediction_detail(pred: PredictionResult) -> PredictionDetail:
    """Converts a domain PredictionResult into a Pydantic PredictionDetail schema."""
    top_k_list = [
        TopKPrediction(rank=idx + 1, class_name=c, confidence=conf)
        for idx, (c, conf) in enumerate(pred.top_k or [])
    ]
    return PredictionDetail(
        predicted_class=pred.predicted_class,
        class_index=pred.class_index,
        confidence=pred.confidence,
        probabilities=pred.probabilities,
        logits=pred.logits,
        top_k=top_k_list,
    )


def create_app(model_manager: Optional[ModelManager] = None) -> FastAPI:
    """
    Factory function creating and configuring the FastAPI inference application.

    Parameters:
        model_manager: Optional custom or pre-initialized ModelManager instance.
                       If None, a fresh ModelManager is created.
    """
    manager = model_manager if model_manager is not None else ModelManager()

    app = FastAPI(
        title="Edge Vision Reliability Lab — Inference REST API",
        description=(
            "Local REST API providing model lifecycle management, registry inspection, "
            "and single/batch inference across FP32, FP16, and INT8 CIFAR-10 vision models."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # -------------------------------------------------------------------------
    # Exception Handlers
    # -------------------------------------------------------------------------

    @app.exception_handler(ModelNotFoundError)
    async def model_not_found_handler(request: Request, exc: ModelNotFoundError):
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content=ErrorResponse(
                detail=sanitize_message(str(exc)),
                error_type="ModelNotFoundError",
            ).model_dump(),
        )

    @app.exception_handler(UnsupportedModelError)
    async def unsupported_model_handler(request: Request, exc: UnsupportedModelError):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(
                detail=sanitize_message(str(exc)),
                error_type="UnsupportedModelError",
            ).model_dump(),
        )

    @app.exception_handler(UnsupportedPrecisionError)
    async def unsupported_precision_handler(request: Request, exc: UnsupportedPrecisionError):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(
                detail=sanitize_message(str(exc)),
                error_type="UnsupportedPrecisionError",
            ).model_dump(),
        )

    @app.exception_handler(DeviceCompatibilityError)
    async def device_compat_handler(request: Request, exc: DeviceCompatibilityError):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(
                detail=sanitize_message(str(exc)),
                error_type="DeviceCompatibilityError",
            ).model_dump(),
        )

    @app.exception_handler(InvalidInputError)
    async def invalid_input_handler(request: Request, exc: InvalidInputError):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(
                detail=sanitize_message(str(exc)),
                error_type="InvalidInputError",
            ).model_dump(),
        )

    @app.exception_handler(InferenceError)
    async def general_inference_error_handler(request: Request, exc: InferenceError):
        # Unloaded model condition triggers 409 Conflict
        status_code = (
            status.HTTP_409_CONFLICT
            if "No model is currently loaded" in str(exc)
            else status.HTTP_400_BAD_REQUEST
        )
        return JSONResponse(
            status_code=status_code,
            content=ErrorResponse(
                detail=sanitize_message(str(exc)),
                error_type=exc.__class__.__name__,
            ).model_dump(),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "detail": sanitize_message(str(exc)),
                "error_type": "ValidationError",
            },
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled server exception: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                detail="An internal server error occurred.",
                error_type="InternalServerError",
            ).model_dump(),
        )

    # -------------------------------------------------------------------------
    # Route Endpoints
    # -------------------------------------------------------------------------

    @app.get(
        "/health",
        response_model=HealthResponse,
        summary="Service Health Check",
        description="Returns service availability and currently loaded model info without triggering model loading.",
    )
    async def get_health():
        info = manager.get_current_model_info()
        return HealthResponse(
            status="healthy",
            model_loaded=info["loaded"],
            model_name=info["model_name"],
            precision=info["precision"],
            device=info["device"],
            backend=info["backend"],
        )

    @app.get(
        "/models",
        response_model=ModelsListResponse,
        summary="List Supported Models",
        description="Returns all supported model architectures and precision configurations from the artifact registry.",
    )
    async def list_models():
        models_data = manager.list_models()
        return ModelsListResponse(
            models=[ModelArtifactSummary(**m) for m in models_data],
            total_count=len(models_data),
        )

    @app.get(
        "/models/current",
        response_model=CurrentModelResponse,
        summary="Active Model Metadata",
        description="Returns metadata for the currently active resident inference model.",
    )
    async def get_current_model():
        info = manager.get_current_model_info()
        return CurrentModelResponse(**info)

    @app.post(
        "/models/load",
        response_model=ModelLoadResponse,
        summary="Load or Switch Inference Model",
        description="Initializes and switches the resident model configuration. Validates precision and device compatibility.",
    )
    async def load_model(payload: ModelLoadRequest):
        res = manager.load_model(
            model_name=payload.model_name,
            precision=payload.precision,
            device=payload.device,
        )
        return ModelLoadResponse(**res)

    @app.post(
        "/predict",
        response_model=SinglePredictionResponse,
        summary="Single Image Prediction",
        description="Accepts a single image upload via multipart/form-data and returns classification predictions and timing.",
    )
    async def predict_single(
        request: Request,
        file: UploadFile = File(..., description="Uploaded image file (PNG, JPG, BMP, WEBP)."),
        resize: Optional[bool] = Query(None, description="Explicitly resize non-32x32 inputs to (32, 32)."),
        top_k: Optional[int] = Query(None, ge=1, le=10, description="Number of top predictions [1, 10]."),
    ):
        # Resolve query/form fallback for parameters
        form_data = await request.form()
        eff_resize = resize if resize is not None else form_data.get("resize", "false").lower() in ("true", "1")

        eff_top_k = top_k
        if eff_top_k is None:
            raw_top_k = form_data.get("top_k")
            if raw_top_k is not None:
                try:
                    eff_top_k = int(raw_top_k)
                except ValueError:
                    raise InvalidInputError(f"top_k must be an integer, got {raw_top_k!r}.")
            else:
                eff_top_k = 5

        if eff_top_k < 1 or eff_top_k > 10:
            raise InvalidInputError(f"top_k must be between 1 and 10, got {eff_top_k}.")

        # Read and validate image content in bounded chunks
        filename = file.filename or "upload.png"
        try:
            content = await read_bounded_upload(file, max_bytes=MAX_IMAGE_SIZE_BYTES, filename=filename)
            pil_img = decode_image_bytes(content, filename=filename)
        finally:
            await file.close()

        # Execute prediction through manager
        output = manager.predict_single(pil_img, resize=eff_resize, top_k=eff_top_k)

        pred_detail = format_prediction_detail(output.prediction)
        timing_schema = InferenceTimingSchema(
            preprocess_ms=output.timing.preprocess_ms,
            forward_ms=output.timing.forward_ms,
            postprocess_ms=output.timing.postprocess_ms,
            total_ms=output.timing.total_ms,
        )

        return SinglePredictionResponse(
            status="success",
            filename=filename,
            prediction=pred_detail,
            timing_ms=timing_schema,
            model_name=output.metadata["model_name"],
            precision=output.metadata["precision"],
            device=output.metadata["device"],
        )

    @app.post(
        "/predict/batch",
        response_model=BatchPredictionResponse,
        summary="Bounded Batch Prediction",
        description="Accepts multiple image uploads and returns ordered predictions with batch throughput metrics.",
    )
    async def predict_batch(
        request: Request,
        files: List[UploadFile] = File(..., description="List of image files to evaluate in batch."),
        resize: Optional[bool] = Query(None, description="Explicitly resize non-32x32 inputs to (32, 32)."),
        top_k: Optional[int] = Query(None, ge=1, le=10, description="Number of top predictions [1, 10]."),
    ):
        if not files or len(files) == 0:
            raise InvalidInputError("Batch prediction requires at least one image file.")

        if len(files) > MAX_BATCH_SIZE:
            raise InvalidInputError(
                f"Batch size {len(files)} exceeds maximum allowed batch size of {MAX_BATCH_SIZE} images."
            )

        # Resolve query/form fallback for parameters
        form_data = await request.form()
        eff_resize = resize if resize is not None else form_data.get("resize", "false").lower() in ("true", "1")

        eff_top_k = top_k
        if eff_top_k is None:
            raw_top_k = form_data.get("top_k")
            if raw_top_k is not None:
                try:
                    eff_top_k = int(raw_top_k)
                except ValueError:
                    raise InvalidInputError(f"top_k must be an integer, got {raw_top_k!r}.")
            else:
                eff_top_k = 5

        if eff_top_k < 1 or eff_top_k > 10:
            raise InvalidInputError(f"top_k must be between 1 and 10, got {eff_top_k}.")

        # Decode all images safely in memory while guaranteeing file handle cleanup
        # and bounded streaming upload reads (both per-file and aggregate)
        pil_images = []
        filenames = []
        aggregate_bytes = 0
        try:
            for idx, f in enumerate(files):
                fname = f.filename or f"image_{idx}.png"
                filenames.append(fname)
                content = await read_bounded_upload(
                    f,
                    max_bytes=MAX_IMAGE_SIZE_BYTES,
                    current_aggregate_bytes=aggregate_bytes,
                    max_aggregate_bytes=MAX_AGGREGATE_BATCH_BYTES,
                    filename=fname,
                )
                aggregate_bytes += len(content)
                img = decode_image_bytes(content, filename=fname)
                pil_images.append(img)
        finally:
            for f in files:
                await f.close()

        # Execute bounded batch prediction through manager
        output = manager.predict_batch(pil_images, resize=eff_resize, top_k=eff_top_k)

        # Assemble per-image prediction items preserving original upload ordering
        item_predictions = [
            BatchItemPrediction(
                filename=filenames[idx],
                index=idx,
                prediction=format_prediction_detail(p),
            )
            for idx, p in enumerate(output.predictions)
        ]

        timing_schema = InferenceTimingSchema(
            preprocess_ms=output.timing.preprocess_ms,
            forward_ms=output.timing.forward_ms,
            postprocess_ms=output.timing.postprocess_ms,
            total_ms=output.timing.total_ms,
        )

        return BatchPredictionResponse(
            status="success",
            batch_size=output.batch_size,
            predictions=item_predictions,
            timing_ms=timing_schema,
            throughput_fps=output.throughput_fps,
            model_name=output.metadata["model_name"],
            precision=output.metadata["precision"],
            device=output.metadata["device"],
        )

    return app
