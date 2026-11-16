import { useState, useMemo } from 'react';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  ZAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import { Award, Info, AlertCircle } from 'lucide-react';
import type { ParetoFrontiers, ParetoPoint } from '@/data/types';
import { formatPercent, formatMs } from '@/data/formatters';

interface ParetoFrontierSectionProps {
  paretoFrontiers: ParetoFrontiers;
}

type FrontierMode = 'size_vs_corrupted' | 'size_vs_clean' | 'latency_vs_corrupted';

export function ParetoFrontierSection({
  paretoFrontiers,
}: ParetoFrontierSectionProps) {
  const [frontierMode, setFrontierMode] = useState<FrontierMode>('size_vs_corrupted');
  const [archFilter, setArchFilter] = useState<'all' | 'resnet18_cifar' | 'mobilenet_v3_small_cifar'>('all');

  // Select the appropriate frontier dataset based on mode
  const currentFrontierSet = useMemo(() => {
    if (frontierMode === 'size_vs_clean') {
      return paretoFrontiers.pareto_size_vs_clean;
    }
    if (frontierMode === 'latency_vs_corrupted') {
      return (
        paretoFrontiers.pareto_cpu_latency_th4_vs_corrupted ||
        paretoFrontiers.pareto_size_vs_corrupted
      );
    }
    return paretoFrontiers.pareto_size_vs_corrupted;
  }, [paretoFrontiers, frontierMode]);

  // Dynamically compute Pareto efficiency on the active candidate set:
  // Candidate B dominates Candidate A if:
  // B.x <= A.x AND B.y >= A.y AND (B.x < A.x OR B.y > A.y)
  // where x is minimized (size or latency) and y is maximized (accuracy).
  const plotData = useMemo(() => {
    const rawItems = currentFrontierSet.all_items || [];
    const candidates = rawItems.filter((item) => {
      if (archFilter !== 'all' && item.model !== archFilter) return false;
      if (frontierMode === 'latency_vs_corrupted' && item.latency_th4_b1_ms === null) return false;
      return true;
    });

    // Compute coordinates for each candidate
    const mapped = candidates.map((item) => {
      let xVal = item.size_mb;
      let yVal = item.mean_corrupted_accuracy;

      if (frontierMode === 'size_vs_clean') {
        xVal = item.size_mb;
        yVal = item.clean_accuracy;
      } else if (frontierMode === 'latency_vs_corrupted') {
        xVal = item.latency_th4_b1_ms || 0;
        yVal = item.mean_corrupted_accuracy;
      }

      return {
        ...item,
        x: Number(xVal.toFixed(2)),
        y: Number(yVal.toFixed(2)),
        name: `${item.display_name} [${item.precision}]`,
      };
    });

    // Mathematically evaluate Pareto dominance among the active candidate set
    return mapped.map((itemA) => {
      const dominatingList: string[] = [];
      for (const itemB of mapped) {
        if (itemB.config_id === itemA.config_id) continue;
        if (
          itemB.x <= itemA.x &&
          itemB.y >= itemA.y &&
          (itemB.x < itemA.x || itemB.y > itemA.y)
        ) {
          dominatingList.push(itemB.name);
        }
      }

      const isEfficient = dominatingList.length === 0;

      return {
        ...itemA,
        is_pareto_efficient: isEfficient,
        dominated_by: dominatingList.length > 0 ? dominatingList : undefined,
      };
    });
  }, [currentFrontierSet, frontierMode, archFilter]);

  const frontierPoints = plotData.filter((p) => p.is_pareto_efficient);
  const dominatedPoints = plotData.filter((p) => !p.is_pareto_efficient);

  // X & Y Axis metadata
  const axisConfig = useMemo(() => {
    if (frontierMode === 'size_vs_clean') {
      return {
        xLabel: 'Serialized Artifact Size (MB)',
        xUnit: ' MB',
        xDomain: [0, 50],
        yLabel: 'Clean Test Accuracy (%)',
        yUnit: '%',
        yDomain: [80, 100],
      };
    }
    if (frontierMode === 'latency_vs_corrupted') {
      return {
        xLabel: 'CPU Latency (Apple M3, batch=1, 4-th) (ms)',
        xUnit: ' ms',
        xDomain: [0, 25],
        yLabel: 'Mean Corrupted Accuracy (%)',
        yUnit: '%',
        yDomain: [50, 70],
      };
    }
    return {
      xLabel: 'Serialized Artifact Size (MB)',
      xUnit: ' MB',
      xDomain: [0, 50],
      yLabel: 'Mean Corrupted Accuracy (%)',
      yUnit: '%',
      yDomain: [50, 70],
    };
  }, [frontierMode]);

  return (
    <Card className="border-surface-border bg-surface/90">
      <CardHeader className="pb-3 border-b border-surface-border">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <div className="flex items-center gap-2">
              <Badge variant="accent" className="font-mono text-[10px]">
                Section F
              </Badge>
              <CardTitle className="text-base font-bold text-text-primary">
                Multi-Objective Pareto Frontier: Accuracy vs. Efficiency Trade-offs
              </CardTitle>
            </div>
            <CardDescription className="text-xs mt-1">
              Non-dominated empirical boundary identifying configurations where no alternative delivers strictly superior accuracy at lower resource footprint.
            </CardDescription>
          </div>

          {/* Mode Selector Tabs */}
          <div className="flex flex-wrap items-center gap-1.5 self-start sm:self-auto bg-surface-secondary/70 p-1 rounded-lg border border-surface-border">
            <button
              onClick={() => setFrontierMode('size_vs_corrupted')}
              className={`px-2.5 py-1 text-xs font-mono rounded transition-colors ${
                frontierMode === 'size_vs_corrupted'
                  ? 'bg-accent/20 text-accent font-bold border border-accent/40'
                  : 'text-text-muted hover:text-text-primary'
              }`}
            >
              Size vs. Corrupted Acc.
            </button>
            <button
              onClick={() => setFrontierMode('size_vs_clean')}
              className={`px-2.5 py-1 text-xs font-mono rounded transition-colors ${
                frontierMode === 'size_vs_clean'
                  ? 'bg-accent/20 text-accent font-bold border border-accent/40'
                  : 'text-text-muted hover:text-text-primary'
              }`}
            >
              Size vs. Clean Acc.
            </button>
            <button
              onClick={() => setFrontierMode('latency_vs_corrupted')}
              className={`px-2.5 py-1 text-xs font-mono rounded transition-colors ${
                frontierMode === 'latency_vs_corrupted'
                  ? 'bg-accent/20 text-accent font-bold border border-accent/40'
                  : 'text-text-muted hover:text-text-primary'
              }`}
            >
              CPU Latency vs. Corrupted
            </button>
          </div>
        </div>
      </CardHeader>

      <CardContent className="pt-4 space-y-5">
        {/* Sub-Header Objectives Callout */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 p-3 rounded-lg bg-surface-secondary/30 border border-surface-border text-xs">
          <div className="flex items-center gap-3 font-mono">
            <span className="text-text-muted">
              X-Axis: <strong className="text-accent">Minimize</strong> ({axisConfig.xLabel})
            </span>
            <span>•</span>
            <span className="text-text-muted">
              Y-Axis: <strong className="text-emerald-400">Maximize</strong> ({axisConfig.yLabel})
            </span>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => setArchFilter('all')}
              className={`px-2 py-0.5 text-[11px] font-mono rounded ${
                archFilter === 'all'
                  ? 'bg-accent/20 text-accent font-bold'
                  : 'text-text-muted hover:text-text-primary'
              }`}
            >
              All
            </button>
            <button
              onClick={() => setArchFilter('resnet18_cifar')}
              className={`px-2 py-0.5 text-[11px] font-mono rounded ${
                archFilter === 'resnet18_cifar'
                  ? 'bg-accent/20 text-accent font-bold'
                  : 'text-text-muted hover:text-text-primary'
              }`}
            >
              ResNet-18
            </button>
            <button
              onClick={() => setArchFilter('mobilenet_v3_small_cifar')}
              className={`px-2 py-0.5 text-[11px] font-mono rounded ${
                archFilter === 'mobilenet_v3_small_cifar'
                  ? 'bg-accent/20 text-accent font-bold'
                  : 'text-text-muted hover:text-text-primary'
              }`}
            >
              MobileNetV3-Small
            </button>
          </div>
        </div>

        {/* CPU Latency Deployment Configuration Notice */}
        {frontierMode === 'latency_vs_corrupted' && (
          <div className="p-3 rounded-lg bg-amber-500/10 border border-amber-500/30 text-amber-300 text-xs flex items-start gap-2.5">
            <AlertCircle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
            <div className="space-y-0.5 text-[11px] leading-relaxed">
              <strong className="text-amber-200 block">Complete Deployment Pipeline Comparison:</strong>
              <span>
                The CPU latency frontier compares complete end-to-end deployment configurations on Apple M3 CPU (batch size 1, 4 intra-op threads, measured in isolated OS processes with 20 warmup runs and 100 timed iterations across 2 independent process repetitions). INT8 uses TorchScript with the QNNPACK runtime, while FP32 uses TorchScript graph operations. This comparison evaluates entire deployment pipelines rather than isolating numerical precision scaling in isolation. FP16 CPU measurements are omitted because half-precision execution was evaluated on Apple Silicon MPS.
              </span>
            </div>
          </div>
        )}

        {/* Interactive ScatterPlot */}
        <div className="h-80 w-full bg-surface-secondary/20 p-3 rounded-xl border border-surface-border">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 20, right: 30, left: 10, bottom: 25 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#262626" />
              <XAxis
                type="number"
                dataKey="x"
                name={axisConfig.xLabel}
                unit={axisConfig.xUnit}
                domain={axisConfig.xDomain}
                stroke="#737373"
                tick={{ fill: '#a3a3a3', fontSize: 11, fontFamily: 'monospace' }}
                label={{
                  value: axisConfig.xLabel,
                  position: 'insideBottom',
                  offset: -15,
                  fill: '#737373',
                  fontSize: 11,
                  fontFamily: 'monospace',
                }}
              />
              <YAxis
                type="number"
                dataKey="y"
                name={axisConfig.yLabel}
                unit={axisConfig.yUnit}
                domain={axisConfig.yDomain}
                stroke="#737373"
                tick={{ fill: '#a3a3a3', fontSize: 11, fontFamily: 'monospace' }}
                label={{
                  value: axisConfig.yLabel,
                  angle: -90,
                  position: 'insideLeft',
                  offset: 0,
                  fill: '#737373',
                  fontSize: 11,
                  fontFamily: 'monospace',
                }}
              />
              <ZAxis range={[120, 260]} />
              <RechartsTooltip
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    const data = payload[0].payload as ParetoPoint & {
                      x: number;
                      y: number;
                      is_pareto_efficient: boolean;
                      name: string;
                      dominated_by?: string[];
                    };
                    return (
                      <div className="bg-surface border border-surface-border p-3 rounded-lg shadow-xl text-xs font-mono space-y-1.5 min-w-[220px]">
                        <div className="flex items-center justify-between pb-1 border-b border-surface-border">
                          <strong className="text-text-primary">{data.name}</strong>
                          {data.is_pareto_efficient ? (
                            <Badge variant="accent" className="text-[9px]">
                              Pareto Optimal
                            </Badge>
                          ) : (
                            <Badge variant="outline" className="text-[9px] text-text-muted">
                              Dominated
                            </Badge>
                          )}
                        </div>
                        <div className="space-y-0.5 text-[11px]">
                          <span className="text-accent block">
                            Serialized Size: <strong>{data.size_mb.toFixed(2)} MB</strong>
                          </span>
                          <span className="text-emerald-400 block">
                            Clean Test Acc: <strong>{formatPercent(data.clean_accuracy)}</strong>
                          </span>
                          <span className="text-purple-400 block">
                            Mean Corrupted Acc: <strong>{formatPercent(data.mean_corrupted_accuracy)}</strong>
                          </span>
                          {data.latency_th4_b1_ms !== null && (
                            <span className="text-amber-400 block">
                              CPU Latency (b=1, 4-th): <strong>{formatMs(data.latency_th4_b1_ms)}</strong>
                            </span>
                          )}
                          {!data.is_pareto_efficient && data.dominated_by && (
                            <span className="text-rose-400 block text-[10px] pt-1 leading-snug">
                              Dominated by: {data.dominated_by.join(', ')}
                            </span>
                          )}
                        </div>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <Legend
                wrapperStyle={{ fontSize: '11px', fontFamily: 'monospace', paddingTop: '10px' }}
              />

              {/* Pareto Efficient Points */}
              <Scatter
                name="Pareto-Efficient Configurations"
                data={frontierPoints}
                fill="#10b981"
                shape="circle"
              >
                {frontierPoints.map((_, index) => (
                  <Cell
                    key={`frontier-${index}`}
                    fill="#10b981"
                    stroke="#34d399"
                    strokeWidth={2}
                  />
                ))}
              </Scatter>

              {/* Dominated Points */}
              <Scatter
                name="Dominated Configurations"
                data={dominatedPoints}
                fill="#525252"
                shape="cross"
              >
                {dominatedPoints.map((_, index) => (
                  <Cell
                    key={`dominated-${index}`}
                    fill="#525252"
                    stroke="#737373"
                    strokeWidth={1.5}
                  />
                ))}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        </div>

        {/* Pareto Frontier Summary Cards */}
        <div className="space-y-2">
          <span className="text-xs font-mono font-bold uppercase tracking-wider text-text-secondary flex items-center gap-1.5">
            <Award className="w-3.5 h-3.5 text-accent" />
            Configurations on Current Pareto Frontier ({frontierPoints.length}):
          </span>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {frontierPoints.map((item) => (
              <div
                key={item.config_id}
                className="p-3 rounded-lg border border-emerald-500/30 bg-emerald-500/5 space-y-2 font-mono text-xs"
              >
                <div className="flex items-center justify-between pb-1 border-b border-emerald-500/20">
                  <span className="font-bold text-text-primary">{item.display_name}</span>
                  <Badge variant="accent" className="text-[10px]">
                    {item.precision}
                  </Badge>
                </div>

                <div className="grid grid-cols-2 gap-2 text-[11px]">
                  <div className="bg-surface/80 p-1.5 rounded border border-surface-border">
                    <span className="text-[10px] text-text-muted uppercase block">
                      {frontierMode === 'latency_vs_corrupted' ? 'CPU Latency' : 'Disk Footprint'}
                    </span>
                    <strong className="text-accent">
                      {frontierMode === 'latency_vs_corrupted' && item.latency_th4_b1_ms !== null
                        ? formatMs(item.latency_th4_b1_ms)
                        : `${item.size_mb.toFixed(2)} MB`}
                    </strong>
                  </div>
                  <div className="bg-surface/80 p-1.5 rounded border border-surface-border">
                    <span className="text-[10px] text-text-muted uppercase block">
                      {frontierMode === 'size_vs_clean' ? 'Clean Acc' : 'Corrupted Acc'}
                    </span>
                    <strong className="text-emerald-400">
                      {formatPercent(
                        frontierMode === 'size_vs_clean'
                          ? item.clean_accuracy
                          : item.mean_corrupted_accuracy
                      )}
                    </strong>
                  </div>
                </div>

                <p className="text-[10px] text-text-secondary font-sans leading-tight">
                  {item.config_id === 'mobilenet_v3_small_cifar_int8'
                    ? 'Occupies the compact, low-resource boundary (1.94 MB storage / 0.48 ms CPU latency).'
                    : item.config_id === 'resnet18_cifar_int8'
                    ? 'Occupies the high-accuracy boundary (11.30 MB storage / 65.76% corrupted acc).'
                    : 'Non-dominated under multi-objective evaluation in this configuration pool.'}
                </p>
              </div>
            ))}
          </div>
        </div>

        {/* Pareto Methodology Caveat */}
        <div className="p-3.5 rounded-lg bg-surface-secondary/40 border border-surface-border text-xs text-text-secondary space-y-1.5">
          <div className="flex items-center gap-2 text-text-primary font-semibold">
            <Info className="w-4 h-4 text-accent shrink-0" />
            <span>Pareto Efficiency Methodology & Interpretation:</span>
          </div>
          <p className="text-[11px] leading-relaxed text-text-secondary">
            A model configuration is deemed <strong>Pareto-efficient</strong> if no other tested configuration simultaneously achieves both lower resource cost (smaller serialized file or lower latency) and higher predictive accuracy. Pareto efficiency does not declare a single "optimal" winner; rather, it identifies the mathematical trade-off frontier. Configurations off the frontier (dominated) are superseded by a frontier variant that offers strictly better performance along at least one dimension without degradation in the other.
          </p>
        </div>
      </CardContent>
    </Card>
  );
}
