import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { Select } from '@/components/ui/Select';
import { ArrowLeftRight, RotateCcw, Cpu, HardDrive, AlertCircle } from 'lucide-react';
import type { PrecisionVariant } from '@/data/types';
import { formatNumber, formatPercent, formatMB } from '@/data/formatters';

export interface SelectedConfig {
  model_id: string;
  precision: 'fp32' | 'fp16' | 'int8';
}

interface ComparisonControlsProps {
  configA: SelectedConfig;
  configB: SelectedConfig;
  onChangeA: (config: SelectedConfig) => void;
  onChangeB: (config: SelectedConfig) => void;
  onSwap: () => void;
  onReset: () => void;
  precisionVariants: PrecisionVariant[];
}

const ARCHITECTURE_OPTIONS = [
  { value: 'resnet18_cifar', label: 'ResNet-18 (Dense Residual)' },
  { value: 'mobilenet_v3_small_cifar', label: 'MobileNetV3-Small (Inverted Residual)' },
];

const PRECISION_OPTIONS = [
  { value: 'fp32', label: 'FP32 (PyTorch Eager)' },
  { value: 'fp16', label: 'FP16 (Half-Precision)' },
  { value: 'int8', label: 'INT8 (PTQ TorchScript)' },
];

export function ComparisonControls({
  configA,
  configB,
  onChangeA,
  onChangeB,
  onSwap,
  onReset,
  precisionVariants,
}: ComparisonControlsProps) {
  const variantA = precisionVariants.find(
    (v) => v.model_id === configA.model_id && v.precision_key === configA.precision
  );
  const variantB = precisionVariants.find(
    (v) => v.model_id === configB.model_id && v.precision_key === configB.precision
  );

  return (
    <Card className="border-surface-border bg-surface/90">
      <CardHeader className="pb-3 border-b border-surface-border">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <div className="flex items-center gap-2">
              <Badge variant="accent" className="font-mono text-[10px]">
                Section A
              </Badge>
              <CardTitle className="text-base font-bold text-text-primary">
                Interactive Model & Precision Comparison Controls
              </CardTitle>
            </div>
            <CardDescription className="text-xs mt-1">
              Select any two model-precision configurations to analyze cross-architecture and quantization trade-offs side by side.
            </CardDescription>
          </div>

          <div className="flex items-center gap-2 self-start sm:self-auto">
            <Button
              variant="outline"
              size="sm"
              onClick={onSwap}
              className="text-xs h-8 px-2.5"
              title="Swap Configuration A and B"
            >
              <ArrowLeftRight className="w-3.5 h-3.5 mr-1.5" />
              Swap
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={onReset}
              className="text-xs h-8 px-2.5 text-text-muted hover:text-text-primary"
              title="Reset to canonical baseline comparison (ResNet-18 FP32 vs MobileNetV3-Small FP32)"
            >
              <RotateCcw className="w-3.5 h-3.5 mr-1.5" />
              Reset Baseline
            </Button>
          </div>
        </div>
      </CardHeader>

      <CardContent className="pt-4 space-y-4">
        {/* Side-by-side selection grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {/* Configuration A Card */}
          <div className="p-4 rounded-xl border border-accent/30 bg-accent/5 space-y-3">
            <div className="flex items-center justify-between pb-2 border-b border-accent/20">
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-accent" />
                <span className="text-xs font-mono font-bold uppercase tracking-wider text-accent">
                  Configuration A (Baseline)
                </span>
              </div>
              <Badge variant="accent" className="font-mono text-[10px] uppercase">
                {configA.precision}
              </Badge>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <Select
                label="Architecture A"
                value={configA.model_id}
                onChange={(e) =>
                  onChangeA({ ...configA, model_id: e.target.value })
                }
                options={ARCHITECTURE_OPTIONS}
              />
              <Select
                label="Precision A"
                value={configA.precision}
                onChange={(e) =>
                  onChangeA({
                    ...configA,
                    precision: e.target.value as 'fp32' | 'fp16' | 'int8',
                  })
                }
                options={PRECISION_OPTIONS}
              />
            </div>

            {variantA && (
              <div className="pt-2 border-t border-accent/15 grid grid-cols-2 sm:grid-cols-4 gap-2 text-center font-mono">
                <div className="p-2 rounded bg-surface/80 border border-surface-border">
                  <span className="text-[10px] uppercase text-text-muted block">Parameters</span>
                  <span className="text-xs font-bold text-text-primary">
                    {formatNumber(variantA.parameters)}
                  </span>
                </div>
                <div className="p-2 rounded bg-surface/80 border border-surface-border">
                  <span className="text-[10px] uppercase text-text-muted block">Disk Size</span>
                  <span className="text-xs font-bold text-accent">
                    {formatMB(variantA.serialized_bytes, 2)}
                  </span>
                </div>
                <div className="p-2 rounded bg-surface/80 border border-surface-border">
                  <span className="text-[10px] uppercase text-text-muted block">Clean Acc.</span>
                  <span className="text-xs font-bold text-emerald-400">
                    {formatPercent(variantA.clean_accuracy)}
                  </span>
                </div>
                <div className="p-2 rounded bg-surface/80 border border-surface-border">
                  <span className="text-[10px] uppercase text-text-muted block">Corrupted Acc.</span>
                  <span className="text-xs font-bold text-text-primary">
                    {formatPercent(variantA.mean_corrupted_accuracy)}
                  </span>
                </div>
              </div>
            )}

            <div className="flex flex-wrap items-center gap-2 text-[11px] text-text-muted font-mono pt-1">
              <span className="inline-flex items-center gap-1">
                <Cpu className="w-3 h-3 text-text-secondary" />
                Backend: <strong className="text-text-primary">{variantA?.runtime_backend}</strong>
              </span>
              <span>•</span>
              <span className="inline-flex items-center gap-1">
                <HardDrive className="w-3 h-3 text-text-secondary" />
                Evaluated Target: <strong className="text-text-primary">{variantA?.evaluation_device}</strong>
              </span>
            </div>
          </div>

          {/* Configuration B Card */}
          <div className="p-4 rounded-xl border border-purple-500/30 bg-purple-500/5 space-y-3">
            <div className="flex items-center justify-between pb-2 border-b border-purple-500/20">
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-purple-400" />
                <span className="text-xs font-mono font-bold uppercase tracking-wider text-purple-400">
                  Configuration B (Comparison Target)
                </span>
              </div>
              <Badge variant="purple" className="font-mono text-[10px] uppercase">
                {configB.precision}
              </Badge>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <Select
                label="Architecture B"
                value={configB.model_id}
                onChange={(e) =>
                  onChangeB({ ...configB, model_id: e.target.value })
                }
                options={ARCHITECTURE_OPTIONS}
              />
              <Select
                label="Precision B"
                value={configB.precision}
                onChange={(e) =>
                  onChangeB({
                    ...configB,
                    precision: e.target.value as 'fp32' | 'fp16' | 'int8',
                  })
                }
                options={PRECISION_OPTIONS}
              />
            </div>

            {variantB && (
              <div className="pt-2 border-t border-purple-500/15 grid grid-cols-2 sm:grid-cols-4 gap-2 text-center font-mono">
                <div className="p-2 rounded bg-surface/80 border border-surface-border">
                  <span className="text-[10px] uppercase text-text-muted block">Parameters</span>
                  <span className="text-xs font-bold text-text-primary">
                    {formatNumber(variantB.parameters)}
                  </span>
                </div>
                <div className="p-2 rounded bg-surface/80 border border-surface-border">
                  <span className="text-[10px] uppercase text-text-muted block">Disk Size</span>
                  <span className="text-xs font-bold text-purple-400">
                    {formatMB(variantB.serialized_bytes, 2)}
                  </span>
                </div>
                <div className="p-2 rounded bg-surface/80 border border-surface-border">
                  <span className="text-[10px] uppercase text-text-muted block">Clean Acc.</span>
                  <span className="text-xs font-bold text-emerald-400">
                    {formatPercent(variantB.clean_accuracy)}
                  </span>
                </div>
                <div className="p-2 rounded bg-surface/80 border border-surface-border">
                  <span className="text-[10px] uppercase text-text-muted block">Corrupted Acc.</span>
                  <span className="text-xs font-bold text-text-primary">
                    {formatPercent(variantB.mean_corrupted_accuracy)}
                  </span>
                </div>
              </div>
            )}

            <div className="flex flex-wrap items-center gap-2 text-[11px] text-text-muted font-mono pt-1">
              <span className="inline-flex items-center gap-1">
                <Cpu className="w-3 h-3 text-text-secondary" />
                Backend: <strong className="text-text-primary">{variantB?.runtime_backend}</strong>
              </span>
              <span>•</span>
              <span className="inline-flex items-center gap-1">
                <HardDrive className="w-3 h-3 text-text-secondary" />
                Evaluated Target: <strong className="text-text-primary">{variantB?.evaluation_device}</strong>
              </span>
            </div>
          </div>
        </div>

        {/* Hardware Availability Callout */}
        <div className="p-3 rounded-lg bg-surface-secondary/40 border border-surface-border text-xs text-text-secondary flex items-start gap-2.5">
          <AlertCircle className="w-4 h-4 text-accent shrink-0 mt-0.5" />
          <div className="space-y-0.5 text-[11px] leading-relaxed">
            <span className="font-semibold text-text-primary block">Hardware & Runtime Target Compatibility Note:</span>
            <span>
              INT8 static quantization executes strictly on CPU via PyTorch's QNNPACK backend. MPS acceleration is evaluated exclusively for FP32 and FP16. If a specific hardware-precision pairing is absent from benchmark tables, it indicates no valid empirical measurement exists—measurements are never silently substituted across devices.
            </span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
