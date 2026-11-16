import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { EmptyState } from '@/components/ui/EmptyState';
import {
  Sparkles,
  Zap,
  Clock,
  BarChart2,
  RefreshCw,
  Info,
} from 'lucide-react';
import type { SinglePredictionResponse, CurrentModelResponse } from '@/api';

interface PredictionResultDisplayProps {
  result: SinglePredictionResponse | null;
  isLoading: boolean;
  activeImageName: string | null;
  currentResidentModel?: CurrentModelResponse | null;
}

export function PredictionResultDisplay({
  result,
  isLoading,
  activeImageName,
  currentResidentModel,
}: PredictionResultDisplayProps) {
  if (isLoading) {
    return (
      <Card className="h-full flex flex-col justify-between border-surface-border bg-surface/90">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-sm font-bold text-text-primary">
                Inference Diagnostics & Classification
              </CardTitle>
              <CardDescription className="text-xs">
                Executing real-time model evaluation through active runtime engine
              </CardDescription>
            </div>
            <Badge variant="accent" className="font-mono text-[10px]">
              Processing
            </Badge>
          </div>
        </CardHeader>

        <CardContent className="flex-1 flex flex-col items-center justify-center py-16 space-y-4">
          <div className="p-4 rounded-full bg-accent/10 border border-accent/20 text-accent animate-pulse">
            <RefreshCw className="w-8 h-8 animate-spin" />
          </div>
          <div className="text-center space-y-1">
            <h4 className="text-sm font-bold text-text-primary">Evaluating Predictions on Hardware</h4>
            <p className="text-xs text-text-secondary max-w-xs">
              Streaming image buffer to memory, executing forward tensor transformations, and computing softmax probabilities...
            </p>
          </div>
        </CardContent>

        {/* Empty timing shell */}
        <div className="p-4 bg-surface-secondary/40 border-t border-surface-border grid grid-cols-2 sm:grid-cols-4 gap-2 text-center opacity-50">
          <div className="p-2 rounded bg-surface border border-surface-border">
            <span className="text-[10px] uppercase font-mono text-text-muted block">Preprocess</span>
            <span className="font-mono text-xs text-text-muted">... ms</span>
          </div>
          <div className="p-2 rounded bg-surface border border-surface-border">
            <span className="text-[10px] uppercase font-mono text-text-muted block">Forward</span>
            <span className="font-mono text-xs text-text-muted">... ms</span>
          </div>
          <div className="p-2 rounded bg-surface border border-surface-border">
            <span className="text-[10px] uppercase font-mono text-text-muted block">Postprocess</span>
            <span className="font-mono text-xs text-text-muted">... ms</span>
          </div>
          <div className="p-2 rounded bg-surface border border-surface-border">
            <span className="text-[10px] uppercase font-mono text-text-muted block">Total Latency</span>
            <span className="font-mono text-xs text-accent">... ms</span>
          </div>
        </div>
      </Card>
    );
  }

  if (!result) {
    return (
      <Card className="h-full flex flex-col justify-between border-surface-border bg-surface/90">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-sm font-bold text-text-primary">
                Inference Diagnostics & Classification
              </CardTitle>
              <CardDescription className="text-xs">
                Classification outputs, top-k probabilities, and component latency breakdown
              </CardDescription>
            </div>
            <Badge variant="outline" className="text-[10px]">
              Awaiting Input
            </Badge>
          </div>
        </CardHeader>

        <CardContent className="flex-1 flex flex-col justify-center py-12">
          <EmptyState
            icon={<Sparkles className="w-8 h-8 text-accent/70" />}
            title="No Active Prediction"
            description="Upload an image and trigger live prediction to observe classification confidence and timing metrics."
            action={
              <div className="text-[11px] font-mono text-text-muted bg-surface-secondary/60 px-3 py-1.5 rounded border border-surface-border">
                {activeImageName ? `Ready to evaluate: ${activeImageName}` : 'Ready for input image upload'}
              </div>
            }
          />
        </CardContent>

        <div className="p-4 bg-surface-secondary/40 border-t border-surface-border grid grid-cols-2 sm:grid-cols-4 gap-2 text-center text-text-muted text-xs font-mono">
          <div className="p-2 rounded bg-surface border border-surface-border">
            <span className="text-[10px] uppercase block text-text-muted">Preprocess</span>
            <span>— ms</span>
          </div>
          <div className="p-2 rounded bg-surface border border-surface-border">
            <span className="text-[10px] uppercase block text-text-muted">Forward</span>
            <span>— ms</span>
          </div>
          <div className="p-2 rounded bg-surface border border-surface-border">
            <span className="text-[10px] uppercase block text-text-muted">Postprocess</span>
            <span>— ms</span>
          </div>
          <div className="p-2 rounded bg-surface border border-surface-border">
            <span className="text-[10px] uppercase block text-text-muted">Total Latency</span>
            <span>— ms</span>
          </div>
        </div>
      </Card>
    );
  }

  const { prediction, timing_ms, model_name, precision, device, filename } = result;
  const isHighConfidence = prediction.confidence >= 0.8;
  const isModerateConfidence = prediction.confidence >= 0.5 && prediction.confidence < 0.8;

  const isResidentEngine = Boolean(
    currentResidentModel?.loaded &&
    currentResidentModel.model_name === model_name &&
    currentResidentModel.precision === precision &&
    currentResidentModel.device === device
  );

  // Format all 10 probabilities sorted descending
  const sortedProbabilities = Object.entries(prediction.probabilities || {}).sort(
    (a, b) => b[1] - a[1]
  );

  return (
    <Card className="h-full flex flex-col justify-between border-surface-border bg-surface/90">
      <div>
        <CardHeader className="pb-3 border-b border-surface-border">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <div>
              <div className="flex items-center gap-2">
                <Badge variant="accent" className="font-mono text-[10px]">
                  Output
                </Badge>
                <CardTitle className="text-sm font-bold text-text-primary">
                  Inference Diagnostics & Classification
                </CardTitle>
              </div>
              <CardDescription className="text-xs mt-0.5 truncate max-w-md">
                Evaluated file: <span className="text-text-primary font-mono">{filename}</span>
                {!isResidentEngine && currentResidentModel?.loaded && (
                  <span className="block text-[11px] text-amber-400/90 pt-0.5">
                    (Current resident engine: {currentResidentModel.model_name === 'resnet18_cifar' ? 'ResNet-18' : 'MobileNetV3-Small'} [{currentResidentModel.precision?.toUpperCase()} on {currentResidentModel.device?.toUpperCase()}])
                  </span>
                )}
              </CardDescription>
            </div>

            <div className="flex flex-wrap items-center gap-1.5 font-mono text-[11px]">
              <Badge variant="outline" className="capitalize">
                {model_name === 'resnet18_cifar' ? 'ResNet-18' : 'MobileNetV3-Small'}
              </Badge>
              <Badge variant="accent" className="uppercase">
                {precision}
              </Badge>
              <Badge variant="purple" className="uppercase">
                {device}
              </Badge>
              {isResidentEngine ? (
                <Badge variant="accent" className="text-[10px] bg-emerald-500/10 border-emerald-500/30 text-emerald-400 font-mono flex items-center gap-1">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                  Resident Engine
                </Badge>
              ) : (
                <Badge variant="outline" className="text-[10px] text-amber-400 border-amber-500/40 bg-amber-500/5 font-mono flex items-center gap-1" title="This prediction was generated by a model variant that is no longer resident in backend memory">
                  <Info className="w-3 h-3" />
                  Historical Eval
                </Badge>
              )}
            </div>
          </div>
        </CardHeader>

        <CardContent className="space-y-4 pt-4">
          {/* Top Prediction Banner */}
          <div className="p-4 rounded-xl bg-gradient-to-r from-surface-secondary/80 to-surface-secondary/40 border border-surface-border flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div className="space-y-1">
              <span className="text-[10px] font-mono uppercase tracking-wider text-text-muted block">
                Top-1 Classification Result
              </span>
              <div className="flex items-baseline gap-3">
                <h3 className="text-2xl font-black text-text-primary tracking-tight capitalize">
                  {prediction.predicted_class}
                </h3>
                <span className="text-xs font-mono text-text-muted">
                  Class Index: #{prediction.class_index}
                </span>
              </div>
            </div>

            <div className="text-right sm:border-l sm:border-surface-border/60 sm:pl-4">
              <span className="text-[10px] font-mono uppercase tracking-wider text-text-muted block">
                Prediction Confidence
              </span>
              <div
                className={`text-2xl font-black font-mono tracking-tight ${
                  isHighConfidence
                    ? 'text-emerald-400'
                    : isModerateConfidence
                    ? 'text-accent'
                    : 'text-amber-400'
                }`}
              >
                {(prediction.confidence * 100).toFixed(2)}%
              </div>
            </div>
          </div>

          {/* Top-K Predictions Bar Breakdown */}
          <div className="space-y-2">
            <div className="flex items-center justify-between text-xs">
              <span className="font-semibold text-text-primary flex items-center gap-1.5">
                <Zap className="w-3.5 h-3.5 text-accent" />
                Top-{prediction.top_k.length} Class Probabilities:
              </span>
              <span className="text-[10px] font-mono text-text-muted">Softmax Normalized</span>
            </div>

            <div className="space-y-1.5">
              {prediction.top_k.map((item) => {
                const pct = item.confidence * 100;
                const isTop = item.rank === 1;

                return (
                  <div
                    key={item.class_name}
                    className="p-2 rounded-lg bg-surface-secondary/30 border border-surface-border/60 text-xs space-y-1"
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <span className="font-mono text-[10px] text-text-muted w-4">
                          #{item.rank}
                        </span>
                        <span className={`font-semibold capitalize ${isTop ? 'text-accent' : 'text-text-primary'}`}>
                          {item.class_name}
                        </span>
                      </div>
                      <span className="font-mono text-xs font-bold text-text-primary">
                        {pct.toFixed(2)}%
                      </span>
                    </div>

                    <div className="w-full bg-surface-border h-1.5 rounded-full overflow-hidden">
                      <div
                        className={`h-full rounded-full transition-all duration-300 ${
                          isTop ? 'bg-accent' : 'bg-text-secondary/60'
                        }`}
                        style={{ width: `${Math.max(pct, 1)}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Complete 10-Class Probability Distribution Tray */}
          {sortedProbabilities.length > 0 && (
            <div className="space-y-1.5 pt-1">
              <span className="text-[11px] font-semibold text-text-secondary flex items-center gap-1">
                <BarChart2 className="w-3.5 h-3.5 text-accent" />
                All 10 CIFAR-10 Classes Distribution:
              </span>
              <div className="grid grid-cols-2 sm:grid-cols-5 gap-1.5 text-[10px] font-mono">
                {sortedProbabilities.map(([cls, prob]) => {
                  const isTop = cls === prediction.predicted_class;
                  return (
                    <div
                      key={cls}
                      className={`p-1.5 rounded border text-center ${
                        isTop
                          ? 'bg-accent/15 border-accent text-accent font-bold'
                          : 'bg-surface-secondary/40 border-surface-border text-text-secondary'
                      }`}
                    >
                      <span className="block capitalize truncate">{cls}</span>
                      <span className="text-text-primary">{(prob * 100).toFixed(1)}%</span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </CardContent>
      </div>

      {/* Latency Timing Breakdown Footer */}
      <div className="p-3.5 bg-surface-secondary/50 border-t border-surface-border">
        <div className="flex items-center justify-between pb-2 text-[11px] text-text-muted">
          <span className="flex items-center gap-1">
            <Clock className="w-3 h-3 text-accent" />
            Execution Latency Breakdown
          </span>
          <span className="font-mono text-text-secondary">
            Throughput: {(1000 / timing_ms.total_ms).toFixed(1)} img/s (single-image)
          </span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-center text-xs font-mono">
          <div className="p-2 rounded bg-surface border border-surface-border">
            <span className="text-[10px] uppercase text-text-muted block">Preprocess</span>
            <span className="text-text-primary font-semibold">{timing_ms.preprocess_ms.toFixed(2)} ms</span>
          </div>
          <div className="p-2 rounded bg-surface border border-surface-border">
            <span className="text-[10px] uppercase text-text-muted block">Forward Pass</span>
            <span className="text-text-primary font-semibold">{timing_ms.forward_ms.toFixed(2)} ms</span>
          </div>
          <div className="p-2 rounded bg-surface border border-surface-border">
            <span className="text-[10px] uppercase text-text-muted block">Postprocess</span>
            <span className="text-text-primary font-semibold">{timing_ms.postprocess_ms.toFixed(2)} ms</span>
          </div>
          <div className="p-2 rounded bg-surface border border-accent/40 bg-accent/5">
            <span className="text-[10px] uppercase text-accent font-bold block">Total Latency</span>
            <span className="text-accent font-bold">{timing_ms.total_ms.toFixed(2)} ms</span>
          </div>
        </div>
      </div>
    </Card>
  );
}
