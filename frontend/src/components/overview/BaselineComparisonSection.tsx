import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { TrendingDown } from 'lucide-react';
import { researchData, formatNumber, formatPercent, formatMB, formatDrop } from '@/data';

export function BaselineComparisonSection() {
  const { baselineComparison } = researchData;
  const { resnet18, mobilenet_v3_small, comparison_ratios } = baselineComparison;

  return (
    <section className="space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-1 border-b border-surface-border">
        <div>
          <h2 className="text-lg font-bold text-text-primary tracking-tight">
            Baseline Architecture Comparison (FP32)
          </h2>
          <p className="text-xs text-text-secondary">
            Contrasting the dense residual baseline against the compact mobile inverted-residual architecture.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant="outline" className="font-mono text-[11px]">
            {comparison_ratios.parameter_ratio}× Parameter Gap
          </Badge>
          <Badge variant="outline" className="font-mono text-[11px]">
            {comparison_ratios.size_ratio}× Size Gap
          </Badge>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* ResNet-18 Card */}
        <Card className="border-accent/30 bg-surface/90 relative overflow-hidden">
          <div className="absolute top-0 right-0 h-1 w-full bg-accent" />
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <Badge variant="accent">Dense Residual Baseline</Badge>
              <span className="text-xs font-mono text-text-muted">resnet18_cifar</span>
            </div>
            <CardTitle className="text-base text-text-primary mt-1">
              {resnet18.name}
            </CardTitle>
            <CardDescription className="text-xs">
              {resnet18.description}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3 text-xs">
            <div className="grid grid-cols-2 gap-2 pt-1">
              <div className="p-2.5 rounded-lg bg-surface-secondary/50 border border-surface-border space-y-0.5">
                <span className="text-[10px] text-text-muted uppercase font-mono">Parameters</span>
                <div className="text-sm font-bold font-mono text-text-primary">
                  {formatNumber(resnet18.parameters)}
                </div>
                <div className="text-[10px] text-text-muted">11.18M state elements</div>
              </div>

              <div className="p-2.5 rounded-lg bg-surface-secondary/50 border border-surface-border space-y-0.5">
                <span className="text-[10px] text-text-muted uppercase font-mono">FP32 Checkpoint</span>
                <div className="text-sm font-bold font-mono text-text-primary">
                  {formatMB(resnet18.fp32_serialized_bytes)}
                </div>
                <div className="text-[10px] text-text-muted">{formatNumber(resnet18.fp32_serialized_bytes)} B</div>
              </div>

              <div className="p-2.5 rounded-lg bg-surface-secondary/50 border border-surface-border space-y-0.5">
                <span className="text-[10px] text-text-muted uppercase font-mono">Clean Test Acc.</span>
                <div className="text-sm font-bold font-mono text-emerald-400">
                  {formatPercent(resnet18.clean_accuracy)}
                </div>
                <div className="text-[10px] text-text-muted">Loss: {resnet18.cross_entropy_loss.toFixed(4)}</div>
              </div>

              <div className="p-2.5 rounded-lg bg-surface-secondary/50 border border-surface-border space-y-0.5">
                <span className="text-[10px] text-text-muted uppercase font-mono">Mean Corrupted Acc.</span>
                <div className="text-sm font-bold font-mono text-accent">
                  {formatPercent(resnet18.mean_corrupted_accuracy)}
                </div>
                <div className="text-[10px] text-rose-400 flex items-center gap-1">
                  <TrendingDown className="w-3 h-3" />
                  <span>{formatDrop(resnet18.mean_accuracy_drop_pp)}</span>
                </div>
              </div>
            </div>

            <div className="pt-2 border-t border-surface-border/60 text-[11px] text-text-secondary leading-relaxed">
              <strong>Architecture Profile:</strong> High capacity residual topology with dense skip connections.
              Maintains an <strong>8.27 pp higher clean baseline</strong> and an average <strong>6.90 pp advantage</strong> under
              environmental corruptions, at the cost of 7.3× more parameters and 7.2× larger storage footprint.
            </div>
          </CardContent>
        </Card>

        {/* MobileNetV3-Small Card */}
        <Card className="border-purple-500/30 bg-surface/90 relative overflow-hidden">
          <div className="absolute top-0 right-0 h-1 w-full bg-purple-500" />
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <Badge variant="purple">Compact Inverted Residual</Badge>
              <span className="text-xs font-mono text-text-muted">mobilenet_v3_small_cifar</span>
            </div>
            <CardTitle className="text-base text-text-primary mt-1">
              {mobilenet_v3_small.name}
            </CardTitle>
            <CardDescription className="text-xs">
              {mobilenet_v3_small.description}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3 text-xs">
            <div className="grid grid-cols-2 gap-2 pt-1">
              <div className="p-2.5 rounded-lg bg-surface-secondary/50 border border-surface-border space-y-0.5">
                <span className="text-[10px] text-text-muted uppercase font-mono">Parameters</span>
                <div className="text-sm font-bold font-mono text-text-primary">
                  {formatNumber(mobilenet_v3_small.parameters)}
                </div>
                <div className="text-[10px] text-text-muted">1.54M state elements</div>
              </div>

              <div className="p-2.5 rounded-lg bg-surface-secondary/50 border border-surface-border space-y-0.5">
                <span className="text-[10px] text-text-muted uppercase font-mono">FP32 Checkpoint</span>
                <div className="text-sm font-bold font-mono text-text-primary">
                  {formatMB(mobilenet_v3_small.fp32_serialized_bytes)}
                </div>
                <div className="text-[10px] text-text-muted">{formatNumber(mobilenet_v3_small.fp32_serialized_bytes)} B</div>
              </div>

              <div className="p-2.5 rounded-lg bg-surface-secondary/50 border border-surface-border space-y-0.5">
                <span className="text-[10px] text-text-muted uppercase font-mono">Clean Test Acc.</span>
                <div className="text-sm font-bold font-mono text-emerald-400">
                  {formatPercent(mobilenet_v3_small.clean_accuracy)}
                </div>
                <div className="text-[10px] text-text-muted">Loss: {mobilenet_v3_small.cross_entropy_loss.toFixed(4)}</div>
              </div>

              <div className="p-2.5 rounded-lg bg-surface-secondary/50 border border-surface-border space-y-0.5">
                <span className="text-[10px] text-text-muted uppercase font-mono">Mean Corrupted Acc.</span>
                <div className="text-sm font-bold font-mono text-purple-400">
                  {formatPercent(mobilenet_v3_small.mean_corrupted_accuracy)}
                </div>
                <div className="text-[10px] text-rose-400 flex items-center gap-1">
                  <TrendingDown className="w-3 h-3" />
                  <span>{formatDrop(mobilenet_v3_small.mean_accuracy_drop_pp)}</span>
                </div>
              </div>
            </div>

            <div className="pt-2 border-t border-surface-border/60 text-[11px] text-text-secondary leading-relaxed">
              <strong>Architecture Profile:</strong> Lightweight inverted bottleneck topology with depthwise separable layers.
              Achieves <strong>86.51% clean accuracy</strong> with only <strong>1.53M parameters</strong>. While mean corrupted accuracy
              is 58.83%, it exhibits lower absolute drop (27.68 pp) and outperforms ResNet-18 under JPEG compression (69.66% vs 67.83%).
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Comparative Analytical Interpretation Callout */}
      <div className="p-3.5 rounded-lg bg-surface-secondary/30 border border-surface-border text-xs text-text-secondary leading-relaxed">
        <strong className="text-text-primary font-medium">Comparative Analytical Finding:</strong> In the evaluated conditions, ResNet-18
        demonstrated higher absolute clean and average corrupted classification accuracy. However, neither architecture was universally more robust
        across all conditions: ResNet-18 showed a larger absolute drop under perturbations (29.05 pp vs 27.68 pp), and MobileNetV3-Small maintained
        higher accuracy under JPEG compression (69.66% vs 67.83%). The benchmark establishes this empirical crossover under the tested JPEG suite
        without identifying an underlying architectural cause.
      </div>
    </section>
  );
}
