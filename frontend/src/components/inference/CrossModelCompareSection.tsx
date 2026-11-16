import { useState } from 'react';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { Select } from '@/components/ui/Select';
import {
  GitCompare,
  CheckCircle2,
  AlertTriangle,
  RefreshCw,
  Info,
  RotateCcw,
} from 'lucide-react';
import { loadModel, predictImage } from '@/api';
import type { SinglePredictionResponse, ModelLoadResponse, CurrentModelResponse } from '@/api';
import type { ImageFileState } from './ImageUploadWorkspace';

interface CrossModelCompareSectionProps {
  baseResult: SinglePredictionResponse | null;
  activeImage: ImageFileState | null;
  onModelSwitched: (loadRes: ModelLoadResponse) => void;
  isBackendConnected: boolean;
  isOperationBusy?: boolean;
  onCompareRunningChange?: (running: boolean) => void;
  currentResidentModel?: CurrentModelResponse | null;
}

export function CrossModelCompareSection({
  baseResult,
  activeImage,
  onModelSwitched,
  isBackendConnected,
  isOperationBusy = false,
  onCompareRunningChange,
  currentResidentModel,
}: CrossModelCompareSectionProps) {
  const [targetModel, setTargetModel] = useState('mobilenet_v3_small_cifar');
  const [targetPrecision, setTargetPrecision] = useState('int8');
  const [targetDevice, setTargetDevice] = useState('cpu');
  const [isComparing, setIsComparing] = useState(false);
  const [isRestoring, setIsRestoring] = useState(false);
  const [compareResult, setCompareResult] = useState<SinglePredictionResponse | null>(null);
  const [compareError, setCompareError] = useState<string | null>(null);

  if (!baseResult || !activeImage) {
    return null;
  }

  const handleRunComparison = async () => {
    if (!activeImage || isComparing || isRestoring || isOperationBusy || !isBackendConnected) return;

    setIsComparing(true);
    onCompareRunningChange?.(true);
    setCompareError(null);

    try {
      // 1. Genuinely load target model via backend lifecycle API
      const loadRes = await loadModel({
        model_name: targetModel,
        precision: targetPrecision,
        device: targetDevice,
      });
      onModelSwitched(loadRes);

      // 2. Run inference on target model using the exact same image and preprocessing top_k
      const predRes = await predictImage(activeImage.file, {
        resize: true,
        top_k: baseResult.prediction.top_k.length,
      });
      setCompareResult(predRes);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Comparison failed';
      setCompareError(message);
    } finally {
      setIsComparing(false);
      onCompareRunningChange?.(false);
    }
  };

  const handleRestoreBaseline = async () => {
    if (!isBackendConnected || isRestoring || isComparing || isOperationBusy) return;

    setIsRestoring(true);
    onCompareRunningChange?.(true);
    setCompareError(null);

    try {
      const loadRes = await loadModel({
        model_name: baseResult.model_name,
        precision: baseResult.precision,
        device: baseResult.device,
      });
      onModelSwitched(loadRes);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to restore baseline model';
      setCompareError(message);
    } finally {
      setIsRestoring(false);
      onCompareRunningChange?.(false);
    }
  };

  const isSameConfig =
    baseResult.model_name === targetModel &&
    baseResult.precision === targetPrecision &&
    baseResult.device === targetDevice;

  const isConcordant =
    compareResult &&
    baseResult.prediction.predicted_class === compareResult.prediction.predicted_class;

  const isConfigAResident = Boolean(
    currentResidentModel?.loaded &&
    currentResidentModel.model_name === baseResult.model_name &&
    currentResidentModel.precision === baseResult.precision &&
    currentResidentModel.device === baseResult.device
  );

  const isConfigBResident = Boolean(
    compareResult &&
    currentResidentModel?.loaded &&
    currentResidentModel.model_name === compareResult.model_name &&
    currentResidentModel.precision === compareResult.precision &&
    currentResidentModel.device === compareResult.device
  );

  const latencyRatio =
    compareResult && compareResult.timing_ms.total_ms > 0
      ? (baseResult.timing_ms.total_ms / compareResult.timing_ms.total_ms).toFixed(1)
      : null;

  return (
    <Card className="border-surface-border bg-surface/90">
      <CardHeader className="pb-3 border-b border-surface-border">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <div className="flex items-center gap-2">
              <GitCompare className="w-4 h-4 text-accent" />
              <CardTitle className="text-sm font-bold text-text-primary">
                Cross-Configuration Concordance & Latency Comparison
              </CardTitle>
            </div>
            <CardDescription className="text-xs mt-0.5">
              Evaluate the same input image across different architectures and precision tiers to observe output stability and latency differences.
            </CardDescription>
          </div>
          <Badge variant="outline" className="font-mono text-[10px]">
            Comparative Evaluation
          </Badge>
        </div>
      </CardHeader>

      <CardContent className="space-y-4 pt-4">
        {/* Comparison Selector Bar */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 items-end p-3 rounded-lg bg-surface-secondary/40 border border-surface-border">
          <Select
            label="Target Architecture"
            value={targetModel}
            onChange={(e) => setTargetModel(e.target.value)}
            disabled={isComparing || isRestoring}
            options={[
              { value: 'resnet18_cifar', label: 'ResNet-18' },
              { value: 'mobilenet_v3_small_cifar', label: 'MobileNetV3-Small' },
            ]}
          />

          <Select
            label="Target Precision"
            value={targetPrecision}
            onChange={(e) => {
              setTargetPrecision(e.target.value);
              if (e.target.value === 'int8') setTargetDevice('cpu');
            }}
            disabled={isComparing || isRestoring}
            options={[
              { value: 'fp32', label: 'FP32 (Eager)' },
              { value: 'fp16', label: 'FP16 (Half)' },
              { value: 'int8', label: 'INT8 (PTQ)' },
            ]}
          />

          <Select
            label="Compute Device"
            value={targetDevice}
            onChange={(e) => setTargetDevice(e.target.value)}
            disabled={isComparing || isRestoring || targetPrecision === 'int8'}
            options={
              targetPrecision === 'int8'
                ? [{ value: 'cpu', label: 'Host CPU (Required)' }]
                : [
                    { value: 'mps', label: 'Apple Silicon MPS' },
                    { value: 'cpu', label: 'Host CPU' },
                  ]
            }
          />

          <Button
            variant="default"
            onClick={handleRunComparison}
            disabled={isComparing || isRestoring || isOperationBusy || !isBackendConnected || isSameConfig}
            className="w-full text-xs font-semibold h-10"
            title={
              !isBackendConnected
                ? 'Backend service is offline'
                : isOperationBusy
                ? 'Another operation (prediction or model load) is currently in progress'
                : isSameConfig
                ? 'Target configuration is identical to the baseline model'
                : 'Load target model into memory and evaluate on the same image'
            }
          >
            {isComparing ? (
              <>
                <RefreshCw className="w-3.5 h-3.5 mr-1.5 animate-spin" />
                Comparing...
              </>
            ) : (
              <>
                <GitCompare className="w-3.5 h-3.5 mr-1.5 text-accent" />
                Run Cross-Eval
              </>
            )}
          </Button>
        </div>

        {/* Backend Lifecycle Transparency Banner */}
        <div className="p-2.5 rounded-lg bg-surface-secondary/30 border border-surface-border/60 text-[11px] text-text-secondary flex items-start gap-2">
          <Info className="w-3.5 h-3.5 text-accent shrink-0 mt-0.5" />
          <div className="space-y-0.5">
            <span className="font-semibold text-text-primary block">
              Backend Model Lifecycle Transparency:
            </span>
            <p className="leading-relaxed">
              The local inference engine maintains a single resident model in memory at any given time.
              Running comparative evaluation explicitly loads Configuration B via <code className="font-mono text-text-primary">POST /models/load</code>, making it the active engine for subsequent requests until switched.
            </p>
          </div>
        </div>

        {compareError && (
          <div className="p-2.5 rounded-lg bg-rose-500/10 border border-rose-500/30 text-rose-400 text-xs flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 shrink-0" />
            <span>{compareError}</span>
          </div>
        )}

        {/* Side-by-Side Comparison Results */}
        {compareResult && (
          <div className="space-y-3 pt-2">
            {/* Concordance Header */}
            <div className="p-3 rounded-lg border flex items-center justify-between text-xs bg-surface-secondary/30 border-surface-border">
              <div className="flex items-center gap-2">
                {isConcordant ? (
                  <>
                    <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                    <span className="font-semibold text-text-primary">
                      Prediction Concordance: Identical Class Prediction
                    </span>
                  </>
                ) : (
                  <>
                    <AlertTriangle className="w-4 h-4 text-amber-400" />
                    <span className="font-semibold text-text-primary">
                      Prediction Discrepancy: Differing Class Labels
                    </span>
                  </>
                )}
              </div>
              <Badge variant={isConcordant ? 'accent' : 'outline'} className="font-mono text-[10px]">
                {isConcordant ? 'Concordant' : 'Discordant'}
              </Badge>
            </div>

            {/* Comparison Cards Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Baseline Result Card */}
              <div className="p-3.5 rounded-xl border border-surface-border bg-surface-secondary/40 space-y-2.5 text-xs">
                <div className="flex items-center justify-between">
                  <span className="text-[10px] font-mono text-text-muted uppercase">Configuration A (Baseline)</span>
                  <div className="flex items-center gap-1">
                    <Badge variant="outline" className="font-mono text-[10px] uppercase">
                      {baseResult.model_name === 'resnet18_cifar' ? 'ResNet-18' : 'MobileNetV3'}
                    </Badge>
                    <Badge variant="accent" className="font-mono text-[10px] uppercase">
                      {baseResult.precision}
                    </Badge>
                    <Badge variant="purple" className="font-mono text-[10px] uppercase">
                      {baseResult.device}
                    </Badge>
                  </div>
                </div>

                <div className="flex items-baseline justify-between border-b border-surface-border/60 pb-2">
                  <div>
                    <span className="text-[10px] text-text-muted font-mono block">PREDICTED CLASS</span>
                    <span className="text-lg font-bold capitalize text-text-primary">
                      {baseResult.prediction.predicted_class}
                    </span>
                  </div>
                  <div className="text-right">
                    <span className="text-[10px] text-text-muted font-mono block">CONFIDENCE</span>
                    <span className="text-sm font-mono font-bold text-accent">
                      {(baseResult.prediction.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>

                <div className="flex justify-between items-center text-[11px] font-mono text-text-secondary">
                  <span>Measured Latency:</span>
                  <span className="text-text-primary font-bold">{baseResult.timing_ms.total_ms.toFixed(2)} ms</span>
                </div>

                <div className="pt-1 border-t border-surface-border/40 flex justify-end">
                  {isConfigAResident ? (
                    <span className="text-[10px] font-mono text-emerald-400 flex items-center gap-1">
                      <CheckCircle2 className="w-3 h-3" />
                      Resident in Memory
                    </span>
                  ) : (
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={handleRestoreBaseline}
                      disabled={isRestoring || isComparing || isOperationBusy || !isBackendConnected}
                      className="text-[11px] h-6 px-2 text-text-secondary hover:text-accent"
                      title={isOperationBusy ? 'Another operation is currently in progress' : 'Reload Configuration A into backend memory as active engine'}
                    >
                      <RotateCcw className="w-3 h-3 mr-1" />
                      {isRestoring ? 'Restoring Engine...' : 'Re-activate Engine A'}
                    </Button>
                  )}
                </div>
              </div>

              {/* Target Comparison Card */}
              <div className="p-3.5 rounded-xl border border-accent/40 bg-accent/5 space-y-2.5 text-xs">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-1.5">
                    <span className="text-[10px] font-mono text-text-muted uppercase">Configuration B (Target)</span>
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" title="Currently Resident in Memory" />
                  </div>
                  <div className="flex items-center gap-1">
                    <Badge variant="outline" className="font-mono text-[10px] uppercase">
                      {compareResult.model_name === 'resnet18_cifar' ? 'ResNet-18' : 'MobileNetV3'}
                    </Badge>
                    <Badge variant="accent" className="font-mono text-[10px] uppercase">
                      {compareResult.precision}
                    </Badge>
                    <Badge variant="purple" className="font-mono text-[10px] uppercase">
                      {compareResult.device}
                    </Badge>
                  </div>
                </div>

                <div className="flex items-baseline justify-between border-b border-surface-border/60 pb-2">
                  <div>
                    <span className="text-[10px] text-text-muted font-mono block">PREDICTED CLASS</span>
                    <span className="text-lg font-bold capitalize text-text-primary">
                      {compareResult.prediction.predicted_class}
                    </span>
                  </div>
                  <div className="text-right">
                    <span className="text-[10px] text-text-muted font-mono block">CONFIDENCE</span>
                    <span className="text-sm font-mono font-bold text-emerald-400">
                      {(compareResult.prediction.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>

                <div className="flex justify-between items-center text-[11px] font-mono text-text-secondary">
                  <span>Measured Latency:</span>
                  <span className="text-text-primary font-bold">
                    {compareResult.timing_ms.total_ms.toFixed(2)} ms
                  </span>
                </div>

                <div className="pt-1 border-t border-surface-border/40 flex justify-end">
                  {isConfigBResident ? (
                    <span className="text-[10px] font-mono text-emerald-400 flex items-center gap-1">
                      <CheckCircle2 className="w-3 h-3" />
                      Resident in Memory
                    </span>
                  ) : (
                    <span className="text-[10px] font-mono text-text-muted">
                      Non-Resident (Historical Eval)
                    </span>
                  )}
                </div>
              </div>
            </div>

            {/* Performance Interpretation Callout */}
            <div className="p-3.5 rounded-xl bg-surface-secondary/40 border border-surface-border space-y-1.5 text-xs">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-1 font-mono">
                <span className="text-text-secondary font-semibold">
                  Compound Deployment Latency Ratio (Config A / Config B):
                </span>
                <Badge variant="accent" className="font-mono text-xs font-bold w-fit">
                  {latencyRatio ? `${latencyRatio}× ${Number(latencyRatio) >= 1.0 ? 'faster' : 'slower'}` : '—'}
                </Badge>
              </div>
              <p className="text-[11px] text-text-muted leading-relaxed">
                <strong className="text-text-secondary font-medium">Methodological Note: </strong>
                This ratio reflects compound differences across two distinct end-to-end deployment configurations:
                Architecture ({baseResult.model_name === 'resnet18_cifar' ? 'ResNet-18' : 'MobileNetV3'} vs {compareResult.model_name === 'resnet18_cifar' ? 'ResNet-18' : 'MobileNetV3'}),
                Precision ({baseResult.precision.toUpperCase()} vs {compareResult.precision.toUpperCase()}), and
                Execution Device ({baseResult.device.toUpperCase()} vs {compareResult.device.toUpperCase()}).
                It represents an end-to-end system comparison and does not isolate the quantization effect of INT8 alone.
              </p>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
