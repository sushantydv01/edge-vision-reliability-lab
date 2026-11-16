/**
 * Typed API Client for Edge Vision Reliability Lab REST Inference API.
 * Provides resilient, typed HTTP interactions matching FastAPI endpoints.
 */

import { getApiUrl, API_BASE_URL } from '@/config/api';
import type {
  HealthResponse,
  ModelsListResponse,
  CurrentModelResponse,
  ModelLoadRequest,
  ModelLoadResponse,
  SinglePredictionResponse,
  BatchPredictionResponse,
  PredictOptions,
} from './types';

/**
 * Standard structured API error carrying HTTP status and backend error category.
 */
export class ApiError extends Error {
  public readonly status: number;
  public readonly errorType?: string;

  constructor(message: string, status: number, errorType?: string) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.errorType = errorType;
    Object.setPrototypeOf(this, ApiError.prototype);
  }
}

/**
 * Parses and formats an HTTP error response from the backend.
 */
async function parseErrorResponse(response: Response): Promise<ApiError> {
  const status = response.status;
  try {
    const data = await response.json();
    if (data && typeof data === 'object') {
      if (typeof data.detail === 'string') {
        return new ApiError(data.detail, status, data.error_type);
      }
      if (Array.isArray(data.detail)) {
        // FastAPI / Pydantic validation error array
        const message = data.detail
          .map((err: { loc?: unknown[]; msg?: string }) => {
            const field = Array.isArray(err.loc) ? err.loc.join('.') : 'field';
            return `${field}: ${err.msg || 'invalid'}`;
          })
          .join('; ');
        return new ApiError(`Validation error: ${message}`, status, 'ValidationError');
      }
    }
  } catch {
    // Response was not JSON
  }

  const fallback = response.statusText || `Request failed with HTTP status ${status}`;
  return new ApiError(fallback, status);
}

/**
 * Catches fetch-level network exceptions and translates them to an ApiError.
 */
function handleNetworkError(err: unknown): never {
  if (err instanceof ApiError) {
    throw err;
  }
  if (err instanceof DOMException && err.name === 'AbortError') {
    throw new ApiError('Request aborted', 0, 'AbortError');
  }
  const message =
    err instanceof Error && err.message
      ? `Network error: ${err.message}. Ensure backend is running at ${API_BASE_URL}.`
      : `Cannot reach backend at ${API_BASE_URL}. Connection refused or server offline.`;
  throw new ApiError(message, 0, 'NetworkError');
}

/**
 * Checks backend service availability and loaded model status.
 */
export async function fetchHealth(signal?: AbortSignal): Promise<HealthResponse> {
  try {
    const res = await fetch(getApiUrl('health'), {
      method: 'GET',
      headers: { Accept: 'application/json' },
      signal,
    });
    if (!res.ok) throw await parseErrorResponse(res);
    return (await res.json()) as HealthResponse;
  } catch (err) {
    handleNetworkError(err);
  }
}

/**
 * Retrieves all supported model configurations in the model registry.
 */
export async function fetchAvailableModels(signal?: AbortSignal): Promise<ModelsListResponse> {
  try {
    const res = await fetch(getApiUrl('models'), {
      method: 'GET',
      headers: { Accept: 'application/json' },
      signal,
    });
    if (!res.ok) throw await parseErrorResponse(res);
    return (await res.json()) as ModelsListResponse;
  } catch (err) {
    handleNetworkError(err);
  }
}

/**
 * Retrieves metadata for the currently active inference model resident in memory.
 */
export async function fetchCurrentModel(signal?: AbortSignal): Promise<CurrentModelResponse> {
  try {
    const res = await fetch(getApiUrl('modelsCurrent'), {
      method: 'GET',
      headers: { Accept: 'application/json' },
      signal,
    });
    if (!res.ok) throw await parseErrorResponse(res);
    return (await res.json()) as CurrentModelResponse;
  } catch (err) {
    handleNetworkError(err);
  }
}

/**
 * Requests the backend to load and activate a specified model and precision.
 */
export async function loadModel(
  req: ModelLoadRequest,
  signal?: AbortSignal
): Promise<ModelLoadResponse> {
  try {
    const res = await fetch(getApiUrl('modelsLoad'), {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
      },
      body: JSON.stringify(req),
      signal,
    });
    if (!res.ok) throw await parseErrorResponse(res);
    return (await res.json()) as ModelLoadResponse;
  } catch (err) {
    handleNetworkError(err);
  }
}

/**
 * Executes inference on a single uploaded image file.
 * Uses multipart/form-data without manually overriding the boundary header.
 */
export async function predictImage(
  file: File,
  options?: PredictOptions,
  signal?: AbortSignal
): Promise<SinglePredictionResponse> {
  try {
    const formData = new FormData();
    formData.append('file', file, file.name);

    const queryParams = new URLSearchParams();
    if (options?.resize !== undefined) {
      queryParams.append('resize', String(options.resize));
    }
    if (options?.top_k !== undefined) {
      queryParams.append('top_k', String(options.top_k));
    }

    const baseEndpoint = getApiUrl('predict');
    const url = queryParams.toString() ? `${baseEndpoint}?${queryParams.toString()}` : baseEndpoint;

    const res = await fetch(url, {
      method: 'POST',
      headers: {
        Accept: 'application/json',
        // Note: Do NOT set Content-Type header here; browser will automatically
        // provide multipart/form-data with the correct boundary parameter.
      },
      body: formData,
      signal,
    });

    if (!res.ok) throw await parseErrorResponse(res);
    return (await res.json()) as SinglePredictionResponse;
  } catch (err) {
    handleNetworkError(err);
  }
}

/**
 * Executes inference on a bounded batch of uploaded image files (up to 64 images).
 */
export async function predictBatch(
  files: File[],
  options?: PredictOptions,
  signal?: AbortSignal
): Promise<BatchPredictionResponse> {
  try {
    const formData = new FormData();
    for (const file of files) {
      formData.append('files', file, file.name);
    }

    const queryParams = new URLSearchParams();
    if (options?.resize !== undefined) {
      queryParams.append('resize', String(options.resize));
    }
    if (options?.top_k !== undefined) {
      queryParams.append('top_k', String(options.top_k));
    }

    const baseEndpoint = getApiUrl('predictBatch');
    const url = queryParams.toString() ? `${baseEndpoint}?${queryParams.toString()}` : baseEndpoint;

    const res = await fetch(url, {
      method: 'POST',
      headers: {
        Accept: 'application/json',
      },
      body: formData,
      signal,
    });

    if (!res.ok) throw await parseErrorResponse(res);
    return (await res.json()) as BatchPredictionResponse;
  } catch (err) {
    handleNetworkError(err);
  }
}
