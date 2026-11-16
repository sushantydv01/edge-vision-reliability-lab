/**
 * Strict TypeScript types for FastAPI Inference REST API.
 * Accurately mirrors Pydantic models in src/edge_vision/api/schemas.py.
 */

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  model_name: string | null;
  precision: string | null;
  device: string | null;
  backend: string | null;
}

export interface ModelArtifactSummary {
  model_name: string;
  precision: string;
  supported_backends: string[];
  preferred_device: string;
  file_size_bytes: number;
  parameter_count: number;
}

export interface ModelsListResponse {
  models: ModelArtifactSummary[];
  total_count: number;
}

export interface CurrentModelResponse {
  loaded: boolean;
  model_name: string | null;
  precision: string | null;
  device: string | null;
  backend: string | null;
  parameter_count: number | null;
  metadata: Record<string, unknown> | null;
}

export interface ModelLoadRequest {
  model_name: string;
  precision: string;
  device?: string;
}

export interface ModelLoadResponse {
  status: string;
  message: string;
  model_name: string;
  precision: string;
  device: string;
  backend: string;
  load_time_ms: number;
}

export interface TopKPrediction {
  rank: number;
  class_name: string;
  confidence: number;
}

export interface PredictionDetail {
  predicted_class: string;
  class_index: number;
  confidence: number;
  probabilities: Record<string, number>;
  logits: number[];
  top_k: TopKPrediction[];
}

export interface InferenceTiming {
  preprocess_ms: number;
  forward_ms: number;
  postprocess_ms: number;
  total_ms: number;
}

export interface SinglePredictionResponse {
  status: string;
  filename: string;
  prediction: PredictionDetail;
  timing_ms: InferenceTiming;
  model_name: string;
  precision: string;
  device: string;
}

export interface BatchItemPrediction {
  filename: string;
  index: number;
  prediction: PredictionDetail;
}

export interface BatchPredictionResponse {
  status: string;
  batch_size: number;
  predictions: BatchItemPrediction[];
  timing_ms: InferenceTiming;
  throughput_fps: number;
  model_name: string;
  precision: string;
  device: string;
}

export interface ApiErrorDetail {
  detail: string;
  error_type?: string;
}

export interface PredictOptions {
  resize?: boolean;
  top_k?: number;
}
