import { useState, useEffect, useCallback } from 'react';
import { Badge } from '@/components/ui/Badge';
import { AlertCircle } from 'lucide-react';

import {
  fetchHealth,
  fetchAvailableModels,
  fetchCurrentModel,
  predictImage,
} from '@/api';
import type {
  CurrentModelResponse,
  ModelArtifactSummary,
  ModelLoadResponse,
  SinglePredictionResponse,
} from '@/api';

import { ModelLifecycleControl } from '@/components/inference/ModelLifecycleControl';
import {
  ImageUploadWorkspace,
  ImageFileState,
} from '@/components/inference/ImageUploadWorkspace';
import { PredictionResultDisplay } from '@/components/inference/PredictionResultDisplay';
import {
  SessionInferenceHistory,
  HistoryRecord,
} from '@/components/inference/SessionInferenceHistory';
import { CrossModelCompareSection } from '@/components/inference/CrossModelCompareSection';
import { API_BASE_URL } from '@/config/api';

export function LiveInferencePage() {
  // Backend & model states
  const [isBackendConnected, setIsBackendConnected] = useState(false);
  const [currentModel, setCurrentModel] = useState<CurrentModelResponse | null>(null);
  const [availableModels, setAvailableModels] = useState<ModelArtifactSummary[]>([]);
  const [isLoadingModels, setIsLoadingModels] = useState(false);

  // Image & inference states
  const [activeImage, setActiveImage] = useState<ImageFileState | null>(null);
  const [activePrediction, setActivePrediction] = useState<SinglePredictionResponse | null>(null);
  const [isInferenceRunning, setIsInferenceRunning] = useState(false);
  const [inferenceError, setInferenceError] = useState<string | null>(null);

  // Global operation concurrency locks
  const [isModelLoading, setIsModelLoading] = useState(false);
  const [isCrossOperationRunning, setIsCrossOperationRunning] = useState(false);

  // In-memory session history
  const [history, setHistory] = useState<HistoryRecord[]>([]);
  const [activeHistoryId, setActiveHistoryId] = useState<string | undefined>(undefined);

  // Synchronize backend health and model status
  const refreshBackendStatus = useCallback(async () => {
    setIsLoadingModels(true);
    try {
      const health = await fetchHealth();
      setIsBackendConnected(health.status === 'healthy');

      // Fetch available models from registry
      const modelsList = await fetchAvailableModels();
      setAvailableModels(modelsList.models);

      // Fetch currently resident model
      const current = await fetchCurrentModel();
      setCurrentModel(current);
    } catch {
      setIsBackendConnected(false);
      setCurrentModel(null);
    } finally {
      setIsLoadingModels(false);
    }
  }, []);

  useEffect(() => {
    refreshBackendStatus();
  }, [refreshBackendStatus]);

  // Model load handler
  const handleModelLoaded = (loadRes: ModelLoadResponse) => {
    setCurrentModel({
      loaded: true,
      model_name: loadRes.model_name,
      precision: loadRes.precision,
      device: loadRes.device,
      backend: loadRes.backend,
      parameter_count:
        availableModels.find(
          (m) => m.model_name === loadRes.model_name && m.precision === loadRes.precision
        )?.parameter_count || null,
      metadata: null,
    });
  };

  // Inference execution handler
  const handleRunInference = async (options: { resize: boolean; top_k: number }) => {
    if (
      !activeImage ||
      !currentModel?.loaded ||
      isInferenceRunning ||
      isModelLoading ||
      isCrossOperationRunning
    ) {
      return;
    }

    setIsInferenceRunning(true);
    setInferenceError(null);

    try {
      const res = await predictImage(activeImage.file, options);
      setActivePrediction(res);

      // Append to in-memory session history
      const newRecord: HistoryRecord = {
        id: `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
        filename: res.filename,
        previewUrl: activeImage.previewUrl,
        modelName: res.model_name,
        precision: res.precision,
        device: res.device,
        predictedClass: res.prediction.predicted_class,
        confidence: res.prediction.confidence,
        totalMs: res.timing_ms.total_ms,
        timestamp: new Date().toLocaleTimeString(),
        fullResult: res,
      };

      setHistory((prev) => [newRecord, ...prev.slice(0, 19)]); // Store up to 20 recent runs
      setActiveHistoryId(newRecord.id);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Inference request failed';
      setInferenceError(message);
      // Clear active prediction on failure so stale results are never presented as new
      setActivePrediction(null);
    } finally {
      setIsInferenceRunning(false);
    }
  };

  const handleSelectHistoryRecord = (record: HistoryRecord) => {
    setActivePrediction(record.fullResult);
    setActiveHistoryId(record.id);
  };

  const handleClearHistory = () => {
    setHistory([]);
    setActiveHistoryId(undefined);
  };

  return (
    <div className="space-y-6 pb-12">
      {/* 1. Header Banner */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-2 border-b border-surface-border">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-xl sm:text-2xl font-bold text-text-primary tracking-tight">
              Live Inference Workstation
            </h1>
            <Badge variant="accent">Module 02</Badge>
          </div>
          <p className="text-xs sm:text-sm text-text-secondary mt-1">
            Interactive client for the local FastAPI inference engine, runtime model switching, and real-time classification.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <div
            className={`inline-flex items-center gap-2 px-2.5 py-1 rounded-full text-xs font-mono border ${
              isBackendConnected
                ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400'
                : 'bg-rose-500/10 border-rose-500/30 text-rose-400'
            }`}
          >
            <span
              className={`w-2 h-2 rounded-full ${
                isBackendConnected ? 'bg-emerald-400 animate-pulse' : 'bg-rose-400'
              }`}
            />
            <span>{isBackendConnected ? 'API Connected' : 'API Offline'}</span>
          </div>
        </div>
      </div>

      {/* Backend Offline Warning Banner */}
      {!isBackendConnected && (
        <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs space-y-1">
          <div className="flex items-center gap-2 font-bold text-rose-400">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span>Cannot reach FastAPI inference service at {API_BASE_URL}</span>
          </div>
          <p className="pl-6 leading-relaxed text-text-secondary">
            Ensure the local inference service is running by launching{' '}
            <code className="text-text-primary font-mono bg-surface px-1.5 py-0.5 rounded border border-surface-border">
              uvicorn edge_vision.api.main:app --host 127.0.0.1 --port 8000
            </code>
            . Click the "Refresh" button once the service is started.
          </p>
        </div>
      )}

      {/* 2. Model Lifecycle Controls */}
      <ModelLifecycleControl
        currentModel={currentModel}
        availableModels={availableModels}
        isLoadingModels={isLoadingModels}
        isBackendConnected={isBackendConnected}
        onModelLoaded={handleModelLoaded}
        onRefreshStatus={refreshBackendStatus}
        isOperationBusy={isInferenceRunning || isCrossOperationRunning}
        onModelLoadingChange={setIsModelLoading}
      />

      {/* 3. Main Workstation Grid: Input (5 cols) & Output (7 cols) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Image Upload & Preprocessing (5 cols) */}
        <div className="lg:col-span-5 space-y-4">
          <ImageUploadWorkspace
            activeImage={activeImage}
            onImageSelected={setActiveImage}
            onRunInference={handleRunInference}
            isInferenceRunning={isInferenceRunning}
            isModelLoaded={Boolean(currentModel?.loaded)}
            isBackendConnected={isBackendConnected}
            inferenceError={inferenceError}
            isOperationBusy={isModelLoading || isCrossOperationRunning}
          />
        </div>

        {/* Right Column: Prediction Results & Timing Breakdown (7 cols) */}
        <div className="lg:col-span-7 space-y-4">
          <PredictionResultDisplay
            result={activePrediction}
            isLoading={isInferenceRunning}
            activeImageName={activeImage?.file.name || null}
            currentResidentModel={currentModel}
          />
        </div>
      </div>

      {/* 4. Cross-Model Concordance & Latency Comparison */}
      <CrossModelCompareSection
        baseResult={activePrediction}
        activeImage={activeImage}
        onModelSwitched={handleModelLoaded}
        isBackendConnected={isBackendConnected}
        isOperationBusy={isModelLoading || isInferenceRunning}
        onCompareRunningChange={setIsCrossOperationRunning}
        currentResidentModel={currentModel}
      />

      {/* 5. In-Memory Session History Log */}
      <SessionInferenceHistory
        history={history}
        onSelectRecord={handleSelectHistoryRecord}
        onClearHistory={handleClearHistory}
        activeResultId={activeHistoryId}
      />
    </div>
  );
}
