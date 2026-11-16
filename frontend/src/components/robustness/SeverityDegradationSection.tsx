import { useState, useMemo } from 'react';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import { Layers, Info } from 'lucide-react';
import type { RobustnessSuite } from '@/data/types';
import { formatPercent } from '@/data/formatters';
import type { RobustnessFilterState } from './RobustnessControls';

interface SeverityDegradationSectionProps {
  filters: RobustnessFilterState;
  suite: RobustnessSuite;
}

type SeverityViewMode = 'precision_compare' | 'arch_compare';

export function SeverityDegradationSection({
  filters,
  suite,
}: SeverityDegradationSectionProps) {
  const [viewMode, setViewMode] = useState<SeverityViewMode>('precision_compare');

  const activeModelName = filters.model_id === 'resnet18_cifar' ? 'ResNet-18' : 'MobileNetV3-Small';
  const otherModelId = filters.model_id === 'resnet18_cifar' ? 'mobilenet_v3_small_cifar' : 'resnet18_cifar';
  const otherModelName = otherModelId === 'resnet18_cifar' ? 'ResNet-18' : 'MobileNetV3-Small';

  // Selected corruption metadata
  const selectedCorrKey = filters.corruption === 'all' ? 'gaussian_noise' : filters.corruption;
  const corrMeta = suite.corruption_metadata[selectedCorrKey];

  // Helper to get accuracy at a specific severity for a given model, precision, and corruption
  const getAccuracy = (modelId: string, prec: 'fp32' | 'fp16' | 'int8', sev: number): number => {
    if (sev === 0) {
      return suite.clean_baselines[`${modelId}_${prec}`] ?? 0;
    }
    // Filter conditions matching this model, precision, and severity
    const matching = suite.detailed_conditions.filter(
      (c) =>
        c.model_id === modelId &&
        c.precision === prec &&
        c.severity === sev &&
        (filters.corruption === 'all' ? true : c.corruption === filters.corruption)
    );
    if (matching.length === 0) return 0;
    const avg = matching.reduce((sum, c) => sum + c.accuracy, 0) / matching.length;
    return Number(avg.toFixed(2));
  };

  // Build chart dataset from severity 0 (Clean) to 5
  const chartData = useMemo(() => {
    return [0, 1, 2, 3, 4, 5].map((sev) => {
      const label = sev === 0 ? 'Clean' : `Sev ${sev}`;
      const paramVal =
        sev === 0
          ? 'Nominal Input'
          : corrMeta
          ? `${corrMeta.param_symbol}=${corrMeta.param_values[sev - 1]}`
          : `Level ${sev}`;

      return {
        severity: sev,
        label,
        paramVal,
        // Precision comparison for active model
        active_fp32: getAccuracy(filters.model_id, 'fp32', sev),
        active_fp16: getAccuracy(filters.model_id, 'fp16', sev),
        active_int8: getAccuracy(filters.model_id, 'int8', sev),
        // Cross architecture comparison for active precision
        primary_model: getAccuracy(filters.model_id, filters.precision, sev),
        comparison_model: getAccuracy(otherModelId, filters.precision, sev),
      };
    });
  }, [filters, otherModelId, corrMeta]);

  const cleanBaseline = suite.clean_baselines[`${filters.model_id}_${filters.precision}`] ?? 94.78;

  return (
    <Card className="border-surface-border bg-surface/90">
      <CardHeader className="pb-3 border-b border-surface-border">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <div className="flex items-center gap-2">
              <Badge variant="accent" className="font-mono text-[10px]">
                Section C
              </Badge>
              <CardTitle className="text-base font-bold text-text-primary">
                Severity Degradation Trajectory (Clean → Severity 1–5)
              </CardTitle>
            </div>
            <CardDescription className="text-xs mt-1">
              Monotonic performance degradation across 5 progressive corruption severity levels relative to nominal clean test baseline.
            </CardDescription>
          </div>

          {/* View Mode Toggle */}
          <div className="flex items-center gap-1.5 self-start sm:self-auto bg-surface-secondary/70 p-1 rounded-lg border border-surface-border">
            <button
              onClick={() => setViewMode('precision_compare')}
              className={`px-2.5 py-1 text-xs font-mono rounded transition-colors ${
                viewMode === 'precision_compare'
                  ? 'bg-accent/20 text-accent font-bold border border-accent/40'
                  : 'text-text-muted hover:text-text-primary'
              }`}
            >
              Compare Precisions ({activeModelName})
            </button>
            <button
              onClick={() => setViewMode('arch_compare')}
              className={`px-2.5 py-1 text-xs font-mono rounded transition-colors ${
                viewMode === 'arch_compare'
                  ? 'bg-accent/20 text-accent font-bold border border-accent/40'
                  : 'text-text-muted hover:text-text-primary'
              }`}
            >
              Cross-Architecture ({filters.precision.toUpperCase()})
            </button>
          </div>
        </div>
      </CardHeader>

      <CardContent className="pt-4 space-y-5">
        {/* Scope Context Banner */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 p-3 rounded-lg bg-surface-secondary/40 border border-surface-border text-xs font-mono">
          <div className="flex items-center gap-2">
            <Layers className="w-3.5 h-3.5 text-accent" />
            <span className="text-text-muted">Domain Scope:</span>
            <strong className="text-text-primary">
              {filters.corruption === 'all'
                ? 'Mean Across All 6 Corruptions'
                : suite.corruption_metadata[filters.corruption]?.name}
            </strong>
          </div>
          <div className="text-[11px] text-text-muted">
            X-Axis: <strong className="text-text-primary">Severity (0 to 5)</strong> • Y-Axis:{' '}
            <strong className="text-emerald-400">Accuracy (%)</strong>
          </div>
        </div>

        {/* Recharts LineChart */}
        <div className="h-80 w-full bg-surface-secondary/20 p-3 rounded-xl border border-surface-border">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 15, right: 30, left: 0, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#262626" />
              <XAxis
                dataKey="label"
                stroke="#737373"
                tick={{ fill: '#a3a3a3', fontSize: 11, fontFamily: 'monospace' }}
              />
              <YAxis
                domain={[30, 100]}
                unit="%"
                stroke="#737373"
                tick={{ fill: '#a3a3a3', fontSize: 11, fontFamily: 'monospace' }}
              />
              <ReferenceLine
                y={cleanBaseline}
                stroke="#10b981"
                strokeDasharray="4 4"
                label={{
                  value: `Clean Baseline (${cleanBaseline.toFixed(2)}%)`,
                  fill: '#10b981',
                  fontSize: 10,
                  position: 'insideTopRight',
                  fontFamily: 'monospace',
                }}
              />
              <RechartsTooltip
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    const data = payload[0].payload;
                    return (
                      <div className="bg-surface border border-surface-border p-3 rounded-lg shadow-xl text-xs font-mono space-y-1.5 min-w-[200px]">
                        <div className="flex items-center justify-between pb-1 border-b border-surface-border">
                          <strong className="text-text-primary">{data.label}</strong>
                          <span className="text-[10px] text-text-muted">{data.paramVal}</span>
                        </div>
                        {payload.map((entry, idx) => (
                          <div key={idx} className="flex justify-between items-center text-[11px]">
                            <span style={{ color: entry.color }}>{entry.name}:</span>
                            <strong className="text-text-primary">{entry.value}%</strong>
                          </div>
                        ))}
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <Legend
                wrapperStyle={{ fontSize: '11px', fontFamily: 'monospace', paddingTop: '10px' }}
              />

              {viewMode === 'precision_compare' ? (
                <>
                  <Line
                    type="monotone"
                    dataKey="active_fp32"
                    name={`${activeModelName} FP32`}
                    stroke="#0284c7"
                    strokeWidth={2.5}
                    dot={{ fill: '#0284c7', r: 4 }}
                    activeDot={{ r: 6 }}
                  />
                  <Line
                    type="monotone"
                    dataKey="active_fp16"
                    name={`${activeModelName} FP16`}
                    stroke="#a855f7"
                    strokeWidth={2}
                    strokeDasharray="4 2"
                    dot={{ fill: '#a855f7', r: 3.5 }}
                  />
                  <Line
                    type="monotone"
                    dataKey="active_int8"
                    name={`${activeModelName} INT8`}
                    stroke="#10b981"
                    strokeWidth={2}
                    dot={{ fill: '#10b981', r: 3.5 }}
                  />
                </>
              ) : (
                <>
                  <Line
                    type="monotone"
                    dataKey="primary_model"
                    name={`${activeModelName} [${filters.precision.toUpperCase()}]`}
                    stroke="#0284c7"
                    strokeWidth={2.5}
                    dot={{ fill: '#0284c7', r: 4 }}
                    activeDot={{ r: 6 }}
                  />
                  <Line
                    type="monotone"
                    dataKey="comparison_model"
                    name={`${otherModelName} [${filters.precision.toUpperCase()}]`}
                    stroke="#a855f7"
                    strokeWidth={2.5}
                    dot={{ fill: '#a855f7', r: 4 }}
                    activeDot={{ r: 6 }}
                  />
                </>
              )}
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Severity Metrics Table */}
        <div className="overflow-x-auto rounded-lg border border-surface-border">
          <table className="w-full text-left text-xs">
            <thead className="bg-surface-secondary/70 text-text-muted font-mono uppercase text-[10px] border-b border-surface-border">
              <tr>
                <th className="py-2.5 px-3">Severity Level</th>
                <th className="py-2.5 px-3">Perturbation Parameter</th>
                <th className="py-2.5 px-3 text-right">FP32 Accuracy</th>
                <th className="py-2.5 px-3 text-right">FP16 Accuracy</th>
                <th className="py-2.5 px-3 text-right">INT8 Accuracy</th>
                <th className="py-2.5 px-3 text-right">Mean Drop vs Clean (pp)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-surface-border font-mono text-[11px]">
              {chartData.map((row) => {
                const dropPP =
                  row.severity === 0 ? 0 : Number((cleanBaseline - row.active_fp32).toFixed(2));

                return (
                  <tr key={row.severity} className="hover:bg-surface-hover/30">
                    <td className="py-2.5 px-3 font-semibold text-text-primary">
                      {row.label}
                    </td>
                    <td className="py-2.5 px-3 text-text-secondary text-[10px]">
                      {row.paramVal}
                    </td>
                    <td className="py-2.5 px-3 text-right text-text-primary font-bold">
                      {formatPercent(row.active_fp32)}
                    </td>
                    <td className="py-2.5 px-3 text-right text-purple-400">
                      {formatPercent(row.active_fp16)}
                    </td>
                    <td className="py-2.5 px-3 text-right text-emerald-400">
                      {formatPercent(row.active_int8)}
                    </td>
                    <td className="py-2.5 px-3 text-right font-bold text-rose-400">
                      {row.severity === 0 ? '0.00 pp' : `-${dropPP.toFixed(2)} pp`}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* Scientific Interpretation Note */}
        <div className="p-3.5 rounded-lg bg-surface-secondary/40 border border-surface-border text-xs text-text-secondary space-y-1.5">
          <div className="flex items-center gap-2 text-text-primary font-semibold">
            <Info className="w-4 h-4 text-accent shrink-0" />
            <span>Severity Trajectory Dynamics:</span>
          </div>
          <p className="text-[11px] leading-relaxed text-text-secondary">
            Across both architectures and all three precision regimes, measured degradation increases non-linearly with severity level. At mild severities (1–2), measured accuracy drops remain modest across most corruption types, while severities 4 and 5 exhibit substantially larger performance drops. Across the five severity levels, FP16 and INT8 quantized curves track closely with the FP32 trajectory, indicating that numerical precision reduction produced minimal difference in degradation trajectory in these experiments.
          </p>
        </div>
      </CardContent>
    </Card>
  );
}
