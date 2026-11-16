import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Info } from 'lucide-react';
import type { BaselineComparison } from '@/data/types';
import { formatNumber, formatPercent, formatRatio } from '@/data/formatters';

interface ArchitectureComparisonSectionProps {
  baselineComparison: BaselineComparison;
}

export function ArchitectureComparisonSection({
  baselineComparison,
}: ArchitectureComparisonSectionProps) {
  const { resnet18, mobilenet_v3_small, comparison_ratios } = baselineComparison;

  return (
    <Card className="border-surface-border bg-surface/90">
      <CardHeader className="pb-3 border-b border-surface-border">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <div className="flex items-center gap-2">
              <Badge variant="accent" className="font-mono text-[10px]">
                Section B
              </Badge>
              <CardTitle className="text-base font-bold text-text-primary">
                Architecture Comparison: ResNet-18 vs. MobileNetV3-Small
              </CardTitle>
            </div>
            <CardDescription className="text-xs mt-1">
              Structural topology, parameter efficiency, baseline clean accuracy, and synthetic degradation resilience.
            </CardDescription>
          </div>
          <Badge variant="outline" className="font-mono text-[11px] self-start sm:self-auto">
            FP32 Baseline Regimes
          </Badge>
        </div>
      </CardHeader>

      <CardContent className="pt-4 space-y-5">
        {/* Metric Comparison Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* ResNet-18 Profile */}
          <div className="p-4 rounded-xl border border-surface-border bg-surface-secondary/40 space-y-4">
            <div className="flex items-center justify-between pb-2 border-b border-surface-border">
              <div>
                <h3 className="text-sm font-bold text-text-primary">{resnet18.name}</h3>
                <p className="text-[11px] text-text-secondary mt-0.5">{resnet18.description}</p>
              </div>
              <Badge variant="default" className="font-mono text-[10px]">
                Dense Residual
              </Badge>
            </div>

            <div className="grid grid-cols-2 gap-3 font-mono text-xs">
              <div className="p-2.5 rounded bg-surface border border-surface-border space-y-1">
                <span className="text-[10px] uppercase text-text-muted block">Parameters</span>
                <span className="text-sm font-bold text-text-primary">
                  {formatNumber(resnet18.parameters)}
                </span>
                <span className="text-[10px] text-text-muted block">7.31× larger parameter pool</span>
              </div>

              <div className="p-2.5 rounded bg-surface border border-surface-border space-y-1">
                <span className="text-[10px] uppercase text-text-muted block">FP32 Serialized Size</span>
                <span className="text-sm font-bold text-text-primary">
                  {resnet18.fp32_serialized_mb.toFixed(2)} MB
                </span>
                <span className="text-[10px] text-text-muted block">7.16× larger disk payload</span>
              </div>

              <div className="p-2.5 rounded bg-surface border border-surface-border space-y-1">
                <span className="text-[10px] uppercase text-text-muted block">Clean Test Accuracy</span>
                <span className="text-sm font-bold text-emerald-400">
                  {formatPercent(resnet18.clean_accuracy)}
                </span>
                <span className="text-[10px] text-emerald-400/80 block">+8.27 pp higher baseline</span>
              </div>

              <div className="p-2.5 rounded bg-surface border border-surface-border space-y-1">
                <span className="text-[10px] uppercase text-text-muted block">Mean Corrupted Acc.</span>
                <span className="text-sm font-bold text-accent">
                  {formatPercent(resnet18.mean_corrupted_accuracy)}
                </span>
                <span className="text-[10px] text-accent/80 block">+6.90 pp higher mean</span>
              </div>
            </div>

            {/* Accuracy Drop Summary */}
            <div className="p-3 rounded-lg bg-surface/60 border border-surface-border text-xs space-y-1.5 font-mono">
              <div className="flex justify-between text-text-secondary">
                <span>Mean Absolute Accuracy Drop:</span>
                <strong className="text-rose-400">-{resnet18.mean_accuracy_drop_pp.toFixed(2)} pp</strong>
              </div>
              <div className="flex justify-between text-text-secondary">
                <span>Mean Relative Accuracy Drop:</span>
                <strong className="text-rose-400">{(resnet18.mean_relative_accuracy_drop * 100).toFixed(2)}%</strong>
              </div>
            </div>
          </div>

          {/* MobileNetV3-Small Profile */}
          <div className="p-4 rounded-xl border border-surface-border bg-surface-secondary/40 space-y-4">
            <div className="flex items-center justify-between pb-2 border-b border-surface-border">
              <div>
                <h3 className="text-sm font-bold text-text-primary">{mobilenet_v3_small.name}</h3>
                <p className="text-[11px] text-text-secondary mt-0.5">{mobilenet_v3_small.description}</p>
              </div>
              <Badge variant="purple" className="font-mono text-[10px]">
                Inverted Residual + SE
              </Badge>
            </div>

            <div className="grid grid-cols-2 gap-3 font-mono text-xs">
              <div className="p-2.5 rounded bg-surface border border-surface-border space-y-1">
                <span className="text-[10px] uppercase text-text-muted block">Parameters</span>
                <span className="text-sm font-bold text-text-primary">
                  {formatNumber(mobilenet_v3_small.parameters)}
                </span>
                <span className="text-[10px] text-text-muted block">86.3% fewer parameters</span>
              </div>

              <div className="p-2.5 rounded bg-surface border border-surface-border space-y-1">
                <span className="text-[10px] uppercase text-text-muted block">FP32 Serialized Size</span>
                <span className="text-sm font-bold text-purple-400">
                  {mobilenet_v3_small.fp32_serialized_mb.toFixed(2)} MB
                </span>
                <span className="text-[10px] text-purple-400/80 block">86.0% storage reduction</span>
              </div>

              <div className="p-2.5 rounded bg-surface border border-surface-border space-y-1">
                <span className="text-[10px] uppercase text-text-muted block">Clean Test Accuracy</span>
                <span className="text-sm font-bold text-text-primary">
                  {formatPercent(mobilenet_v3_small.clean_accuracy)}
                </span>
                <span className="text-[10px] text-text-muted block">86.51% held-out CIFAR-10</span>
              </div>

              <div className="p-2.5 rounded bg-surface border border-surface-border space-y-1">
                <span className="text-[10px] uppercase text-text-muted block">Mean Corrupted Acc.</span>
                <span className="text-sm font-bold text-text-primary">
                  {formatPercent(mobilenet_v3_small.mean_corrupted_accuracy)}
                </span>
                <span className="text-[10px] text-text-muted block">58.83% across 30 conditions</span>
              </div>
            </div>

            {/* Accuracy Drop Summary */}
            <div className="p-3 rounded-lg bg-surface/60 border border-surface-border text-xs space-y-1.5 font-mono">
              <div className="flex justify-between text-text-secondary">
                <span>Mean Absolute Accuracy Drop:</span>
                <strong className="text-rose-400">-{mobilenet_v3_small.mean_accuracy_drop_pp.toFixed(2)} pp</strong>
              </div>
              <div className="flex justify-between text-text-secondary">
                <span>Mean Relative Accuracy Drop:</span>
                <strong className="text-rose-400">{(mobilenet_v3_small.mean_relative_accuracy_drop * 100).toFixed(2)}%</strong>
              </div>
            </div>
          </div>
        </div>

        {/* Detailed Side-by-Side Comparison Table */}
        <div className="overflow-x-auto rounded-lg border border-surface-border">
          <table className="w-full text-left text-xs">
            <thead className="bg-surface-secondary/70 text-text-muted font-mono uppercase text-[10px] border-b border-surface-border">
              <tr>
                <th className="py-2.5 px-3">Metric Dimension</th>
                <th className="py-2.5 px-3 text-right">ResNet-18 (Dense)</th>
                <th className="py-2.5 px-3 text-right">MobileNetV3-Small (Compact)</th>
                <th className="py-2.5 px-3 text-right">Differential / Ratio</th>
                <th className="py-2.5 px-3">Interpretation</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-surface-border font-mono text-[11px]">
              <tr className="hover:bg-surface-hover/30">
                <td className="py-2.5 px-3 font-semibold text-text-primary">Trainable Parameters</td>
                <td className="py-2.5 px-3 text-right text-text-primary">{formatNumber(resnet18.parameters)}</td>
                <td className="py-2.5 px-3 text-right text-text-primary">{formatNumber(mobilenet_v3_small.parameters)}</td>
                <td className="py-2.5 px-3 text-right text-accent font-bold">{formatRatio(comparison_ratios.parameter_ratio)}</td>
                <td className="py-2.5 px-3 text-text-secondary font-sans text-xs">ResNet-18 has 7.31× more parameters for representation capacity.</td>
              </tr>
              <tr className="hover:bg-surface-hover/30">
                <td className="py-2.5 px-3 font-semibold text-text-primary">FP32 Serialized Size</td>
                <td className="py-2.5 px-3 text-right text-text-primary">{resnet18.fp32_serialized_mb.toFixed(2)} MB</td>
                <td className="py-2.5 px-3 text-right text-purple-400 font-bold">{mobilenet_v3_small.fp32_serialized_mb.toFixed(2)} MB</td>
                <td className="py-2.5 px-3 text-right text-accent font-bold">{formatRatio(comparison_ratios.size_ratio)}</td>
                <td className="py-2.5 px-3 text-text-secondary font-sans text-xs">MobileNetV3-Small serialized payload is 7.16× smaller on disk.</td>
              </tr>
              <tr className="hover:bg-surface-hover/30">
                <td className="py-2.5 px-3 font-semibold text-text-primary">Clean Test Accuracy</td>
                <td className="py-2.5 px-3 text-right text-emerald-400 font-bold">{formatPercent(resnet18.clean_accuracy)}</td>
                <td className="py-2.5 px-3 text-right text-text-primary">{formatPercent(mobilenet_v3_small.clean_accuracy)}</td>
                <td className="py-2.5 px-3 text-right text-emerald-400 font-bold">+{comparison_ratios.clean_accuracy_diff_pp.toFixed(2)} pp</td>
                <td className="py-2.5 px-3 text-text-secondary font-sans text-xs">ResNet-18 measured 8.27 pp higher baseline accuracy on held-out CIFAR-10.</td>
              </tr>
              <tr className="hover:bg-surface-hover/30">
                <td className="py-2.5 px-3 font-semibold text-text-primary">Mean Corrupted Accuracy</td>
                <td className="py-2.5 px-3 text-right text-emerald-400 font-bold">{formatPercent(resnet18.mean_corrupted_accuracy)}</td>
                <td className="py-2.5 px-3 text-right text-text-primary">{formatPercent(mobilenet_v3_small.mean_corrupted_accuracy)}</td>
                <td className="py-2.5 px-3 text-right text-emerald-400 font-bold">+{comparison_ratios.corrupted_accuracy_diff_pp.toFixed(2)} pp</td>
                <td className="py-2.5 px-3 text-text-secondary font-sans text-xs">ResNet-18 retained 6.90 pp higher accuracy averaged across all 30 conditions.</td>
              </tr>
              <tr className="hover:bg-surface-hover/30">
                <td className="py-2.5 px-3 font-semibold text-text-primary">Mean Absolute Drop</td>
                <td className="py-2.5 px-3 text-right text-rose-400">-{resnet18.mean_accuracy_drop_pp.toFixed(2)} pp</td>
                <td className="py-2.5 px-3 text-right text-rose-400">-{mobilenet_v3_small.mean_accuracy_drop_pp.toFixed(2)} pp</td>
                <td className="py-2.5 px-3 text-right text-text-muted">-1.37 pp</td>
                <td className="py-2.5 px-3 text-text-secondary font-sans text-xs">MobileNet measured a slightly smaller absolute drop because its baseline was lower.</td>
              </tr>
              <tr className="hover:bg-surface-hover/30">
                <td className="py-2.5 px-3 font-semibold text-text-primary">Mean Relative Drop</td>
                <td className="py-2.5 px-3 text-right text-text-primary">{(resnet18.mean_relative_accuracy_drop * 100).toFixed(2)}%</td>
                <td className="py-2.5 px-3 text-right text-text-primary">{(mobilenet_v3_small.mean_relative_accuracy_drop * 100).toFixed(2)}%</td>
                <td className="py-2.5 px-3 text-right text-text-muted">-1.35%</td>
                <td className="py-2.5 px-3 text-text-secondary font-sans text-xs">ResNet-18 incurred a slightly smaller relative degradation from its baseline.</td>
              </tr>
            </tbody>
          </table>
        </div>

        {/* Robustness Nuance Callout */}
        <div className="p-3.5 rounded-lg bg-surface-secondary/40 border border-surface-border text-xs text-text-secondary space-y-1.5">
          <div className="flex items-center gap-2 text-text-primary font-semibold">
            <Info className="w-4 h-4 text-accent shrink-0" />
            <span>Robustness Interpretation & Metric Meaning:</span>
          </div>
          <p className="text-[11px] leading-relaxed text-text-secondary">
            Neither model can be described as universally "more robust" in all dimensions. ResNet-18 achieves higher absolute clean and corrupted accuracy across 5 of the 6 corruption families. However, MobileNetV3-Small exhibits higher mean accuracy specifically under JPEG compression (69.66% vs 67.83%), and incurs a smaller absolute drop from its clean baseline. Absolute accuracy, absolute degradation (pp), and relative degradation (%) offer complementary perspectives on model stability under perturbation.
          </p>
        </div>
      </CardContent>
    </Card>
  );
}
