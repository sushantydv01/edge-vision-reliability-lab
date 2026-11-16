import { useState, useMemo } from 'react';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer,
  ZAxis,
} from 'recharts';
import { Scale, CheckCircle2 } from 'lucide-react';
import type { PrecisionVariant, BenchmarkRecord } from '@/data/types';
import { formatPercent, formatMs } from '@/data/formatters';

interface AccuracyEfficiencySectionProps {
  precisionVariants: PrecisionVariant[];
  benchmarkRecords: BenchmarkRecord[];
}

export function AccuracyEfficiencySection({
  precisionVariants,
  benchmarkRecords,
}: AccuracyEfficiencySectionProps) {
  const [accuracyTarget, setAccuracyTarget] = useState<'corrupted' | 'clean'>('corrupted');

  // Map each of the 6 configurations with their efficiency and accuracy metrics
  const tradeOffData = useMemo(() => {
    return precisionVariants.map((v) => {
      // Find matching batch 1 CPU record for latency comparison
      const cpuB1 = benchmarkRecords.find(
        (r) => r.model === v.model_id && r.precision === v.precision_key && r.device === 'cpu' && r.batch_size === 1
      );
      // Find matching batch 64 MPS record for peak throughput
      const mpsB64 = benchmarkRecords.find(
        (r) => r.model === v.model_id && r.precision === v.precision_key && r.device === 'mps' && r.batch_size === 64
      );
      // For INT8, peak throughput is CPU B64
      const cpuB64 = benchmarkRecords.find(
        (r) => r.model === v.model_id && r.precision === v.precision_key && r.device === 'cpu' && r.batch_size === 64
      );

      const peakThroughput = mpsB64 ? mpsB64.throughput_fps : cpuB64 ? cpuB64.throughput_fps : 0;
      const b1Latency = cpuB1 ? cpuB1.median_total_ms : 0;

      const accuracy =
        accuracyTarget === 'corrupted' ? v.mean_corrupted_accuracy : v.clean_accuracy;

      return {
        id: `${v.model_id}_${v.precision_key}`,
        name: `${v.display_name} [${v.precision}]`,
        shortName: `${v.display_name === 'ResNet-18' ? 'ResNet' : 'MobileNet'} ${v.precision}`,
        model: v.display_name,
        precision: v.precision,
        size_mb: v.serialized_mb,
        clean_acc: v.clean_accuracy,
        corrupted_acc: v.mean_corrupted_accuracy,
        accuracy,
        b1_latency_ms: b1Latency,
        peak_throughput_fps: peakThroughput,
        compression_ratio: v.compression_ratio,
        size_reduction_pct: v.size_reduction_pct,
      };
    });
  }, [precisionVariants, benchmarkRecords, accuracyTarget]);

  return (
    <Card className="border-surface-border bg-surface/90">
      <CardHeader className="pb-3 border-b border-surface-border">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <div className="flex items-center gap-2">
              <Badge variant="accent" className="font-mono text-[10px]">
                Section E
              </Badge>
              <CardTitle className="text-base font-bold text-text-primary">
                Multi-Objective Accuracy vs. Resource Footprint Trade-offs
              </CardTitle>
            </div>
            <CardDescription className="text-xs mt-1">
              Contrasting predictive reliability against serialized model capacity, single-image latency, and batch throughput without imposing arbitrary scalar weighting.
            </CardDescription>
          </div>

          {/* Accuracy Objective Toggle */}
          <div className="flex items-center gap-1.5 self-start sm:self-auto bg-surface-secondary/70 p-1 rounded-lg border border-surface-border">
            <button
              type="button"
              onClick={() => setAccuracyTarget('corrupted')}
              className={`px-2.5 py-1 text-xs font-mono rounded transition-colors ${
                accuracyTarget === 'corrupted'
                  ? 'bg-accent/20 text-accent font-bold border border-accent/40'
                  : 'text-text-muted hover:text-text-primary'
              }`}
            >
              Mean Corrupted Accuracy (Robustness)
            </button>
            <button
              type="button"
              onClick={() => setAccuracyTarget('clean')}
              className={`px-2.5 py-1 text-xs font-mono rounded transition-colors ${
                accuracyTarget === 'clean'
                  ? 'bg-emerald-500/20 text-emerald-400 font-bold border border-emerald-500/40'
                  : 'text-text-muted hover:text-text-primary'
              }`}
            >
              Clean Test Accuracy (Nominal)
            </button>
          </div>
        </div>
      </CardHeader>

      <CardContent className="pt-4 space-y-5">
        {/* Scatter Visualization: Storage Size (X) vs Accuracy (Y) */}
        <div className="h-80 w-full bg-surface-secondary/20 p-3 rounded-xl border border-surface-border">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 20, right: 30, left: 10, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#262626" />
              <XAxis
                dataKey="size_mb"
                type="number"
                name="Serialized Size"
                unit=" MB"
                domain={[0, 48]}
                stroke="#737373"
                tick={{ fill: '#a3a3a3', fontSize: 11, fontFamily: 'monospace' }}
              />
              <YAxis
                dataKey="accuracy"
                type="number"
                name="Accuracy"
                unit="%"
                domain={accuracyTarget === 'corrupted' ? [50, 70] : [80, 100]}
                stroke="#737373"
                tick={{ fill: '#a3a3a3', fontSize: 11, fontFamily: 'monospace' }}
              />
              <ZAxis range={[70, 70]} />
              <RechartsTooltip
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    const data = payload[0].payload;
                    return (
                      <div className="bg-surface border border-surface-border p-3 rounded shadow-xl text-xs font-mono space-y-1.5">
                        <span className="font-bold text-text-primary block">{data.name}</span>
                        <div className="pt-1 space-y-0.5 text-[11px]">
                          <span className="text-accent block">
                            Serialized Size: <strong>{data.size_mb.toFixed(2)} MB</strong>
                          </span>
                          <span className="text-emerald-400 block">
                            Clean Accuracy: <strong>{formatPercent(data.clean_acc)}</strong>
                          </span>
                          <span className="text-rose-400 block">
                            Mean Corrupted Accuracy: <strong>{formatPercent(data.corrupted_acc)}</strong>
                          </span>
                          <span className="text-purple-400 block">
                            Compression Ratio: <strong>{data.compression_ratio.toFixed(2)}×</strong>
                          </span>
                          <span className="text-text-muted block">
                            Batch 1 CPU Latency: <strong>{formatMs(data.b1_latency_ms)}</strong>
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
                name="ResNet-18 Configurations"
                data={tradeOffData.filter((d) => d.model === 'ResNet-18')}
                fill="#3b82f6"
              />
              <Scatter
                name="MobileNetV3-Small Configurations"
                data={tradeOffData.filter((d) => d.model === 'MobileNetV3-Small')}
                fill="#10b981"
              />
            </ScatterChart>
          </ResponsiveContainer>
        </div>

        {/* Trade-off Matrix Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 font-mono text-xs">
          {/* Trade-off 1: Architecture Scaling */}
          <div className="p-3.5 rounded-xl bg-surface-secondary/40 border border-surface-border space-y-2">
            <div className="flex items-center justify-between pb-1 border-b border-surface-border">
              <span className="font-bold text-text-primary flex items-center gap-1.5">
                <Scale className="w-3.5 h-3.5 text-accent" />
                Architectural Capacity Trade-off
              </span>
              <Badge variant="outline" className="text-accent text-[9px]">
                FP32 Baselines
              </Badge>
            </div>
            <div className="space-y-1.5 text-[11px] font-sans text-text-secondary leading-relaxed">
              <p>
                MobileNetV3-Small achieves an <strong>86.04% storage reduction</strong> (6.25 MB vs. 44.78 MB) and requires 7.31× fewer parameters than ResNet-18.
              </p>
              <p>
                However, this efficiency incurs an <strong>8.27 pp deficit in clean accuracy</strong> (86.51% vs. 94.78%) and a <strong>6.90 pp deficit in mean corrupted accuracy</strong> (58.83% vs. 65.73%).
              </p>
            </div>
          </div>

          {/* Trade-off 2: Quantization Efficiency */}
          <div className="p-3.5 rounded-xl bg-surface-secondary/40 border border-surface-border space-y-2">
            <div className="flex items-center justify-between pb-1 border-b border-surface-border">
              <span className="font-bold text-text-primary flex items-center gap-1.5">
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                Quantization Retention Trade-off
              </span>
              <Badge variant="outline" className="text-emerald-400 text-[9px]">
                INT8 Quantization
              </Badge>
            </div>
            <div className="space-y-1.5 text-[11px] font-sans text-text-secondary leading-relaxed">
              <p>
                For ResNet-18, INT8 post-training quantization reduces file size by <strong>74.77%</strong> (11.30 MB vs 44.78 MB) with zero accuracy degradation (clean accuracy +0.03 pp, corrupted accuracy +0.03 pp).
              </p>
              <p>
                For MobileNetV3-Small, INT8 achieves a <strong>68.92% storage reduction</strong> (1.94 MB) and up to 7,754.8 img/s batch throughput, measuring 59.01% corrupted accuracy (+0.19 pp over FP32).
              </p>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
