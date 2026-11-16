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
} from 'recharts';
import { Award, Info } from 'lucide-react';
import type { ParetoFrontiers } from '@/data/types';
import { formatPercent, formatMs } from '@/data/formatters';

interface EfficiencyParetoSectionProps {
  paretoFrontiers: ParetoFrontiers;
}

type FrontierMode = 'size_vs_corrupted' | 'size_vs_clean' | 'latency_vs_corrupted';

export function EfficiencyParetoSection({
  paretoFrontiers,
}: EfficiencyParetoSectionProps) {
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
      let yVal =
        frontierMode === 'size_vs_clean' ? item.clean_accuracy : item.mean_corrupted_accuracy;

      if (frontierMode === 'latency_vs_corrupted') {
        xVal = item.latency_th4_b1_ms ?? 0;
        yVal = item.mean_corrupted_accuracy;
      }

      return {
        ...item,
        x: xVal,
        y: yVal,
      };
    });

    // Determine dynamic Pareto dominance
    return mapped.map((candidate) => {
      const isDominated = mapped.some((other) => {
        if (other.config_id === candidate.config_id) return false;
        // Dominance: other is strictly better on one and no worse on the other
        const noWorseX = other.x <= candidate.x;
        const noWorseY = other.y >= candidate.y;
        const strictlyBetter = (other.x < candidate.x && other.y >= candidate.y) ||
                               (other.x <= candidate.x && other.y > candidate.y);
        return noWorseX && noWorseY && strictlyBetter;
      });

      return {
        ...candidate,
        is_dynamically_efficient: !isDominated,
      };
    });
  }, [currentFrontierSet, archFilter, frontierMode]);

  // Split into frontier points and dominated points for distinct visual scatter series
  const frontierPoints = useMemo(
    () => plotData.filter((p) => p.is_dynamically_efficient),
    [plotData]
  );
  const dominatedPoints = useMemo(
    () => plotData.filter((p) => !p.is_dynamically_efficient),
    [plotData]
  );

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
                Multi-Objective Pareto Optimality Frontier
              </CardTitle>
            </div>
            <CardDescription className="text-xs mt-1">
              Non-dominated boundary identifying configurations where no alternative achieves both lower resource cost and strictly higher predictive accuracy.
            </CardDescription>
          </div>

          {/* Architecture Filter */}
          <div className="flex items-center gap-1.5 self-start sm:self-auto bg-surface-secondary/70 p-1 rounded-lg border border-surface-border">
            <button
              type="button"
              onClick={() => setArchFilter('all')}
              className={`px-2.5 py-1 text-xs font-mono rounded transition-colors ${
                archFilter === 'all'
                  ? 'bg-accent/20 text-accent font-bold border border-accent/40'
                  : 'text-text-muted hover:text-text-primary'
              }`}
            >
              All Models
            </button>
            <button
              type="button"
              onClick={() => setArchFilter('resnet18_cifar')}
              className={`px-2.5 py-1 text-xs font-mono rounded transition-colors ${
                archFilter === 'resnet18_cifar'
                  ? 'bg-accent/20 text-accent font-bold border border-accent/40'
                  : 'text-text-muted hover:text-text-primary'
              }`}
            >
              ResNet-18
            </button>
            <button
              type="button"
              onClick={() => setArchFilter('mobilenet_v3_small_cifar')}
              className={`px-2.5 py-1 text-xs font-mono rounded transition-colors ${
                archFilter === 'mobilenet_v3_small_cifar'
                  ? 'bg-accent/20 text-accent font-bold border border-accent/40'
                  : 'text-text-muted hover:text-text-primary'
              }`}
            >
              MobileNetV3
            </button>
          </div>
        </div>
      </CardHeader>

      <CardContent className="pt-4 space-y-5">
        {/* Frontier Mode Selection Tabs */}
        <div className="flex flex-wrap items-center gap-2 border-b border-surface-border pb-3">
          <span className="text-xs font-mono text-text-muted mr-1">Pareto Objective Pair:</span>
          <button
            type="button"
            onClick={() => setFrontierMode('size_vs_corrupted')}
            className={`px-3 py-1.5 text-xs font-mono rounded-lg transition-colors border ${
              frontierMode === 'size_vs_corrupted'
                ? 'bg-accent/20 text-accent font-bold border-accent/50'
                : 'bg-surface-secondary/40 text-text-muted border-surface-border hover:text-text-primary'
            }`}
          >
            1. Artifact Size vs. Mean Corrupted Acc
          </button>
          <button
            type="button"
            onClick={() => setFrontierMode('size_vs_clean')}
            className={`px-3 py-1.5 text-xs font-mono rounded-lg transition-colors border ${
              frontierMode === 'size_vs_clean'
                ? 'bg-accent/20 text-accent font-bold border-accent/50'
                : 'bg-surface-secondary/40 text-text-muted border-surface-border hover:text-text-primary'
            }`}
          >
            2. Artifact Size vs. Clean Accuracy
          </button>
          <button
            type="button"
            onClick={() => setFrontierMode('latency_vs_corrupted')}
            className={`px-3 py-1.5 text-xs font-mono rounded-lg transition-colors border ${
              frontierMode === 'latency_vs_corrupted'
                ? 'bg-accent/20 text-accent font-bold border-accent/50'
                : 'bg-surface-secondary/40 text-text-muted border-surface-border hover:text-text-primary'
            }`}
          >
            3. CPU Batch-1 Latency vs. Corrupted Acc
          </button>
        </div>

        {/* CPU Latency Deployment Notice */}
        {frontierMode === 'latency_vs_corrupted' && (
          <div className="p-3.5 rounded-lg bg-surface-secondary/40 border border-surface-border text-xs text-text-secondary flex items-start gap-2.5">
            <Info className="w-4 h-4 text-accent shrink-0 mt-0.5" />
            <p className="text-[11px] leading-relaxed">
              <strong>Benchmarking Protocol Specification:</strong> The CPU latency frontier compares complete end-to-end deployment configurations on Apple M3 CPU (batch size 1, 4 intra-op threads, measured in isolated OS processes with 20 warmup runs and 100 timed iterations across 2 independent process repetitions). INT8 uses TorchScript with QNNPACK runtime, while FP32 uses TorchScript graph operations. FP16 CPU measurements are omitted because half-precision execution was evaluated on Apple Silicon MPS.
            </p>
          </div>
        )}

        {/* Recharts 2D Scatter Chart */}
        <div className="h-80 w-full bg-surface-secondary/20 p-3 rounded-xl border border-surface-border">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 20, right: 30, left: 10, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#262626" />
              <XAxis
                dataKey="x"
                type="number"
                name={frontierMode === 'latency_vs_corrupted' ? 'Batch 1 Latency' : 'Serialized Size'}
                unit={frontierMode === 'latency_vs_corrupted' ? ' ms' : ' MB'}
                domain={frontierMode === 'latency_vs_corrupted' ? [0, 28] : [0, 48]}
                stroke="#737373"
                tick={{ fill: '#a3a3a3', fontSize: 11, fontFamily: 'monospace' }}
              />
              <YAxis
                dataKey="y"
                type="number"
                name="Accuracy"
                unit="%"
                domain={frontierMode === 'size_vs_clean' ? [80, 100] : [55, 70]}
                stroke="#737373"
                tick={{ fill: '#a3a3a3', fontSize: 11, fontFamily: 'monospace' }}
              />
              <ZAxis range={[90, 90]} />
              <RechartsTooltip
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    const data = payload[0].payload;
                    return (
                      <div className="bg-surface border border-surface-border p-3 rounded shadow-xl text-xs font-mono space-y-1.5">
                        <div className="flex items-center justify-between gap-3">
                          <span className="font-bold text-text-primary">{data.display_name} [{data.precision}]</span>
                          <Badge
                            variant={data.is_dynamically_efficient ? 'accent' : 'outline'}
                            className="text-[9px]"
                          >
                            {data.is_dynamically_efficient ? 'Pareto-Optimal' : 'Dominated'}
                          </Badge>
                        </div>
                        <div className="pt-1 space-y-0.5 text-[11px]">
                          <span className="text-text-muted block">
                            {frontierMode === 'latency_vs_corrupted'
                              ? `Batch 1 Latency: ${formatMs(data.x)}`
                              : `Serialized Size: ${data.x.toFixed(2)} MB`}
                          </span>
                          <span className="text-emerald-400 block">
                            Clean Accuracy: <strong>{formatPercent(data.clean_accuracy)}</strong>
                          </span>
                          <span className="text-rose-400 block">
                            Mean Corrupted Accuracy: <strong>{formatPercent(data.mean_corrupted_accuracy)}</strong>
                          </span>
                        </div>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <Legend wrapperStyle={{ fontSize: '11px', fontFamily: 'monospace', paddingTop: '10px' }} />
              <Scatter
                name="Pareto-Efficient Configurations"
                data={frontierPoints}
                fill="#10b981"
                shape="circle"
              />
              <Scatter
                name="Dominated Configurations"
                data={dominatedPoints}
                fill="#737373"
                shape="cross"
              />
            </ScatterChart>
          </ResponsiveContainer>
        </div>

        {/* Pareto Frontier Listing */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 font-mono text-xs">
          <div className="p-3.5 rounded-xl bg-emerald-500/5 border border-emerald-500/20 space-y-2">
            <span className="font-bold text-emerald-400 flex items-center gap-1.5">
              <Award className="w-3.5 h-3.5" />
              Configurations on Current Pareto Frontier ({frontierPoints.length}):
            </span>
            <div className="space-y-1.5">
              {frontierPoints.map((p) => (
                <div key={p.config_id} className="p-2 rounded bg-surface/80 border border-surface-border flex items-center justify-between">
                  <span className="font-semibold text-text-primary">{p.display_name} [{p.precision}]</span>
                  <div className="flex items-center gap-2 text-[11px]">
                    <span className="text-text-muted">{frontierMode === 'latency_vs_corrupted' ? formatMs(p.x) : `${p.x.toFixed(2)} MB`}</span>
                    <span className="text-emerald-400 font-bold">{formatPercent(p.y)}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="p-3.5 rounded-xl bg-surface-secondary/40 border border-surface-border space-y-2 font-sans text-text-secondary text-[11px] leading-relaxed">
            <span className="font-bold text-text-primary font-mono text-xs block">Mathematical Pareto Optimality Criteria:</span>
            <p>
              A candidate configuration is declared <strong>Pareto-efficient</strong> if no other tested candidate delivers equal or superior accuracy at lower resource footprint (file size or latency).
            </p>
            <p>
              Under size vs corrupted accuracy, <strong>MobileNetV3-Small INT8</strong> defines the minimum-storage boundary (1.94 MB, 59.01% corrupted acc) and <strong>ResNet-18 INT8</strong> defines the high-accuracy boundary (11.30 MB, 65.76% corrupted acc). FP32 and FP16 variants are dominated because INT8 achieves equal or slightly higher accuracy at substantially lower resource footprint.
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
