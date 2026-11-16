/**
 * Centralized API configuration for Edge Vision Reliability Lab.
 * Defines base URLs, endpoint routes, and system limits matching the FastAPI backend.
 */

export const API_BASE_URL: string =
  import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

export const API_ENDPOINTS = {
  health: '/health',
  models: '/models',
  modelsCurrent: '/models/current',
  modelsLoad: '/models/load',
  predict: '/predict',
  predictBatch: '/predict/batch',
} as const;

export const SYSTEM_LIMITS = {
  maxImageSizeBytes: 10 * 1024 * 1024, // 10 MB per image
  maxBatchSize: 64, // 64 images per batch request
  maxImageDimension: 2048, // 2048px maximum width or height
  maxAggregateBatchBytes: 64 * 1024 * 1024, // 64 MB total batch upload limit
  targetSpatialDimension: [32, 32] as const, // CIFAR-10 canonical spatial resolution
  supportedExtensions: ['.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tif', '.tiff'] as const,
} as const;

/**
 * Constructs a fully qualified URL for a given API endpoint.
 */
export function getApiUrl(endpoint: keyof typeof API_ENDPOINTS): string {
  const path = API_ENDPOINTS[endpoint];
  const base = API_BASE_URL.replace(/\/+$/, '');
  return `${base}${path}`;
}
