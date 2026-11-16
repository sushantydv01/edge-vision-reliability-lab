import { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { Select } from '@/components/ui/Select';
import {
  Cpu,
  CheckCircle2,
  AlertTriangle,
  RefreshCw,
  HardDrive,
  Activity,
  Layers,
  Info,
} from 'lucide-react';
import { loadModel } from '@/api';
import type { CurrentModelResponse, ModelArtifactSummary, ModelLoadResponse } from '@/api';
import { formatNumber, formatMB } from '@/data';

interface ModelLifecycleControlProps {
  currentModel: CurrentModelResponse | null;
  availableModels: ModelArtifactSummary[];
  isLoadingModels: boolean;
  isBackendConnected: boolean;
  onModelLoaded: (response: ModelLoadResponse) => void;
  onRefreshStatus: () => void;
  isOperationBusy?: boolean;
  onModelLoadingChange?: (loading: boolean) => void;
}

export function ModelLifecycleControl({
  currentModel,
  availableModels,
  isLoadingModels,
  isBackendConnected,
  onModelLoaded,
  onRefreshStatus,
  isOperationBusy = false,
  onModelLoadingChange,
}: ModelLifecycleControlProps) {
  const [selectedModel, setSelectedModel] = useState('resnet18_cifar');
  const [selectedPrecision, setSelectedPrecision] = useState('fp32');
  const [selectedDevice, setSelectedDevice] = useState('mps');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [loadSuccess, setLoadSuccess] = useState<ModelLoadResponse | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  // Sync selected state with currentModel when it first loads or changes
  useEffect(() => {
    if (currentModel?.loaded && currentModel.model_name && currentModel.precision) {
      setSelectedModel(currentModel.model_name);
      setSelectedPrecision(currentModel.precision);
      if (currentModel.device) {
        setSelectedDevice(currentModel.device);
      }
    }
  }, [currentModel?.loaded, currentModel?.model_name, currentModel?.precision, currentModel?.device]);

  // Enforce device constraints: INT8 strictly requires CPU
  useEffect(() => {
    if (selectedPrecision === 'int8' && selectedDevice !== 'cpu') {
      setSelectedDevice('cpu');
    }
  }, [selectedPrecision, selectedDevice]);

  const handlePrecisionChange = (newPrecision: string) => {
    setSelectedPrecision(newPrecision);
    if (newPrecision === 'int8') {
      setSelectedDevice('cpu');
    }
  };

  const isCurrentlyLoadedSelected =
    Boolean(currentModel?.loaded) &&
    currentModel?.model_name === selectedModel &&
    currentModel?.precision === selectedPrecision &&
    currentModel?.device === selectedDevice;

  const handleLoadModel = async () => {
    if (!isBackendConnected || isSubmitting || isOperationBusy) return;

    setIsSubmitting(true);
    onModelLoadingChange?.(true);
    setLoadError(null);
    setLoadSuccess(null);

    try {
      const res = await loadModel({
        model_name: selectedModel,
        precision: selectedPrecision,
        device: selectedDevice,
      });
      setLoadSuccess(res);
      onModelLoaded(res);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to load model';
      setLoadError(message);
    } finally {
      setIsSubmitting(false);
      onModelLoadingChange?.(false);
    }
  };

  // Find metadata for the selected configuration from the registry
  const selectedSummary = availableModels.find(
    (m) => m.model_name === selectedModel && m.precision === selectedPrecision
  );

  return (
    <Card className="border-surface-border bg-surface/90">
      <CardHeader className="pb-3">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <div className="flex items-center gap-2">
              <Badge variant="accent" className="font-mono text-[10px]">
                Phase 1
              </Badge>
              <CardTitle className="text-sm font-bold text-text-primary">
                Model Lifecycle & Runtime Configuration
              </CardTitle>
            </div>
            <CardDescription className="text-xs mt-0.5">
              Query supported models, inspect in-memory residency, and atomically switch active engines.
            </CardDescription>
          </div>

          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={onRefreshStatus}
              disabled={isLoadingModels || isSubmitting || isOperationBusy}
              className="text-xs h-7 px-2"
              title={isOperationBusy ? 'Operation in progress' : 'Refresh backend status and model residency'}
            >
              <RefreshCw className={`w-3 h-3 mr-1 ${isLoadingModels ? 'animate-spin' : ''}`} />
              Refresh
            </Button>
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Active Residency Status Banner */}
        <div className="p-3 rounded-lg border bg-surface-secondary/50 border-surface-border">
          {currentModel?.loaded ? (
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs">
              <div className="flex items-center gap-2.5">
                <span className="relative flex h-2.5 w-2.5">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
                  <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-emerald-500" />
                </span>
                <div>
                  <span className="text-[10px] font-mono uppercase tracking-wider text-text-muted block">
                    Active In-Memory Engine
                  </span>
                  <span className="font-bold text-text-primary text-sm flex items-center gap-2">
                    {currentModel.model_name === 'resnet18_cifar' ? 'ResNet-18' : 'MobileNetV3-Small'}
                    <Badge variant="accent" className="text-[10px] py-0 px-1 font-mono uppercase">
                      {currentModel.precision}
                    </Badge>
                    <Badge variant="outline" className="text-[10px] py-0 px-1 font-mono uppercase">
                      {currentModel.device}
                    </Badge>
                  </span>
                </div>
              </div>

              <div className="flex flex-wrap items-center gap-3 text-[11px] text-text-secondary border-t sm:border-t-0 sm:border-l border-surface-border/60 pt-2 sm:pt-0 sm:pl-3">
                <div className="space-y-0.5">
                  <span className="text-text-muted text-[10px] block">Runtime Backend:</span>
                  <span className="font-mono text-text-primary">{currentModel.backend || 'pytorch_eager'}</span>
                </div>
                <div className="space-y-0.5">
                  <span className="text-text-muted text-[10px] block">Parameters:</span>
                  <span className="font-mono text-text-primary">
                    {currentModel.parameter_count ? formatNumber(currentModel.parameter_count) : '—'}
                  </span>
                </div>
              </div>
            </div>
          ) : (
            <div className="flex items-start gap-2.5 text-xs text-amber-400">
              <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
              <div>
                <span className="font-semibold block text-text-primary">
                  No Inference Model Resident in Memory
                </span>
                <span className="text-text-secondary text-[11px]">
                  Select an architecture and precision variant below and click{' '}
                  <strong className="text-text-primary font-medium">"Load Model"</strong> to initialize the engine.
                  Inference requests will return HTTP 409 Conflict until a model is loaded.
                </span>
              </div>
            </div>
          )}
        </div>

        {/* Model Selection Controls */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 items-end">
          <Select
            label="Model Architecture"
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            disabled={isSubmitting || isOperationBusy || !isBackendConnected}
            options={[
              { value: 'resnet18_cifar', label: 'ResNet-18 (Dense Residual)' },
              { value: 'mobilenet_v3_small_cifar', label: 'MobileNetV3-Small (Inverted Residual)' },
            ]}
          />

          <Select
            label="Precision Variant"
            value={selectedPrecision}
            onChange={(e) => handlePrecisionChange(e.target.value)}
            disabled={isSubmitting || isOperationBusy || !isBackendConnected}
            options={[
              { value: 'fp32', label: 'FP32 (PyTorch Eager)' },
              { value: 'fp16', label: 'FP16 (Half-Precision)' },
              { value: 'int8', label: 'INT8 (PTQ TorchScript)' },
            ]}
          />

          <Select
            label="Compute Device"
            value={selectedDevice}
            onChange={(e) => setSelectedDevice(e.target.value)}
            disabled={isSubmitting || isOperationBusy || !isBackendConnected || selectedPrecision === 'int8'}
            options={
              selectedPrecision === 'int8'
                ? [{ value: 'cpu', label: 'Host CPU (QNNPACK required)' }]
                : [
                    { value: 'mps', label: 'Apple Silicon MPS (Metal)' },
                    { value: 'cpu', label: 'Host CPU' },
                  ]
            }
          />

          <Button
            variant="default"
            onClick={handleLoadModel}
            disabled={isSubmitting || isOperationBusy || !isBackendConnected || isCurrentlyLoadedSelected}
            className="w-full text-xs font-semibold h-10"
            title={
              !isBackendConnected
                ? 'Backend service is offline'
                : isOperationBusy
                ? 'Another operation (prediction or comparison) is currently in progress'
                : isCurrentlyLoadedSelected
                ? 'This model configuration is already resident in memory'
                : 'Initialize and residentially load this model configuration'
            }
          >
            {isSubmitting ? (
              <>
                <RefreshCw className="w-3.5 h-3.5 mr-1.5 animate-spin" />
                Initializing Engine...
              </>
            ) : isCurrentlyLoadedSelected ? (
              <>
                <CheckCircle2 className="w-3.5 h-3.5 mr-1.5 text-emerald-400" />
                Active in Memory
              </>
            ) : (
              <>
                <Cpu className="w-3.5 h-3.5 mr-1.5 text-accent" />
                Load Model
              </>
            )}
          </Button>
        </div>

        {/* Selected Model Registry Metadata Card */}
        {selectedSummary && (
          <div className="flex flex-wrap items-center gap-4 text-[11px] text-text-muted bg-surface-secondary/30 px-3 py-2 rounded-md border border-surface-border/60">
            <span className="flex items-center gap-1 font-mono text-text-secondary">
              <HardDrive className="w-3 h-3 text-accent" />
              Artifact Size: {formatMB(selectedSummary.file_size_bytes)}
            </span>
            <span className="flex items-center gap-1 font-mono text-text-secondary">
              <Layers className="w-3 h-3 text-accent" />
              Parameters: {formatNumber(selectedSummary.parameter_count)}
            </span>
            <span className="flex items-center gap-1 font-mono text-text-secondary">
              <Activity className="w-3 h-3 text-accent" />
              Backend: {selectedSummary.supported_backends.join(', ')}
            </span>
            {selectedPrecision === 'int8' && (
              <span className="flex items-center gap-1 text-amber-400 text-[10px]">
                <Info className="w-3 h-3 shrink-0" />
                INT8 operators execute strictly on CPU via QNNPACK engine
              </span>
            )}
          </div>
        )}

        {/* Feedback Messages */}
        {loadSuccess && (
          <div className="p-2.5 rounded-lg bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-xs flex items-center justify-between">
            <div className="flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 shrink-0" />
              <span>{loadSuccess.message}</span>
            </div>
            <Badge variant="outline" className="font-mono text-[10px] text-emerald-300 border-emerald-500/40">
              Load Time: {loadSuccess.load_time_ms.toFixed(1)} ms
            </Badge>
          </div>
        )}

        {loadError && (
          <div className="p-2.5 rounded-lg bg-rose-500/10 border border-rose-500/30 text-rose-400 text-xs flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 shrink-0" />
            <div className="space-y-0.5">
              <strong className="block font-medium">Model Initialization Failed</strong>
              <span className="text-[11px] text-rose-300 leading-snug">{loadError}</span>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
