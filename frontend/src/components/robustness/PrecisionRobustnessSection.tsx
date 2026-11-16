import { useMemo } from 'react';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { Info, CheckCircle2 } from 'lucide-react';
import type { RobustnessSuite, PrecisionVariant } from '@/data/types';
import { formatPercent } from '@/data/formatters';
import type { RobustnessFilterState } from './RobustnessControls';

interface PrecisionRobustnessSectionProps {
  filters: RobustnessFilterState;
  suite: RobustnessSuite;
  precisionVariants: PrecisionVariant[];
}

export function PrecisionRobustnessSection({
  filters,
  suite,
  precisionVariants,
}: PrecisionRobustnessSectionProps) {
  const activeModelName = filters.model_id === 'resnet18_cifar' ? 'ResNet-18' : 'MobileNetV3-Small';

  // Compute stats for FP32, FP16, and INT8 under active filters
  const precisionComparison = useMemo(() => {
    const precisions: Array<'fp32' | 'fp16' | 'int8'> = ['fp32', 'fp16', 'int8'];

    // Get FP32 baseline clean and corrupted accuracy for delta computation
    const fp32Conditions = suite.detailed_conditions.filter((c) => {
      if (c.model_id !== filters.model_id) return false;
      if (c.precision !== 'fp32') return false;
      if (filters.category !== 'all' && c.category !== filters.category) return false;
      if (filters.corruption !== 'all' && c.corruption !== filters.corruption) return false;
      if (filters.severity !== 'all' && c.severity !== Number(filters.severity)) return false;
      return true;
    });

    const fp32AvgCorrupted =
      fp32Conditions.length > 0
        ? fp32Conditions.reduce((sum, c) => sum + c.accuracy, 0) / fp32Conditions.length
        : 0;

    return precisions.map((prec) => {
      const variant = precisionVariants.find(
        (v) => v.model_id === filters.model_id && v.precision_key === prec
      );

      const conditions = suite.detailed_conditions.filter((c) => {
        if (c.model_id !== filters.model_id) return false;
        if (c.precision !== prec) return false;
        if (filters.category !== 'all' && c.category !== filters.category) return false;
        if (filters.corruption !== 'all' && c.corruption !== filters.corruption) return false;
        if (filters.severity !== 'all' && c.severity !== Number(filters.severity)) return false;
        return true;
      });

      const avgCorrupted =
        conditions.length > 0
          ? conditions.reduce((sum, c) => sum + c.accuracy, 0) / conditions.length
          : 0;

      const avgDropPP =
        conditions.length > 0
          ? conditions.reduce((sum, c) => sum + c.accuracy_drop_pp, 0) / conditions.length
          : 0;

      const cleanAccuracy = suite.clean_baselines[`${filters.model_id}_${prec}`] ?? 0;
      const deltaVsFp32PP = avgCorrupted - fp32AvgCorrupted;

      return {
        precision: prec.toUpperCase(),
        precision_key: prec,
        runtime_backend: variant?.runtime_backend || (prec === 'int8' ? 'TorchScript (QNNPACK)' : 'PyTorch Eager'),
        clean_acc: Number(cleanAccuracy.toFixed(2)),
        corrupted_acc: Number(avgCorrupted.toFixed(2)),
        drop_pp: Number(avgDropPP.toFixed(2)),
        delta_vs_fp32_pp: Number(deltaVsFp32PP.toFixed(2)),
        concordance_pct: variant?.concordance_pct ?? 100,
        changed_preds: variant?.changed_predictions ?? 0,
        compression_ratio: variant?.compression_ratio ?? 1,
      };
    });
  }, [filters, suite, precisionVariants]);

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
                Precision Quantization Robustness: FP32 vs. FP16 vs. INT8
              </CardTitle>
            </div>
            <CardDescription className="text-xs mt-1">
              Evaluating whether numerical precision reduction (16-bit half precision and 8-bit post-training quantization) compromises predictive reliability under image degradation.
            </CardDescription>
          </div>

          <Badge variant="outline" className="font-mono text-xs self-start sm:self-auto">
            {activeModelName} Precision Regimes
          </Badge>
        </div>
      </CardHeader>

      <CardContent className="pt-4 space-y-5">
        {/* Precision Comparison Bar Chart */}
        <div className="h-64 w-full bg-surface-secondary/20 p-3 rounded-xl border border-surface-border">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={precisionComparison}
              margin={{ top: 15, right: 30, left: 0, bottom: 20 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#262626" />
              <XAxis
                dataKey="precision"
                stroke="#737373"
                tick={{ fill: '#a3a3a3', fontSize: 11, fontFamily: 'monospace' }}
              />
              <YAxis
                domain={[0, 100]}
                unit="%"
                stroke="#737373"
                tick={{ fill: '#a3a3a3', fontSize: 11, fontFamily: 'monospace' }}
              />
              <RechartsTooltip
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    const data = payload[0].payload;
                    return (
                      <div className="bg-surface border border-surface-border p-3 rounded shadow-xl text-xs font-mono space-y-1">
                        <strong className="text-text-primary block">{data.precision} Regime</strong>
                        <span className="text-[10px] text-text-muted block">Backend: {data.runtime_backend}</span>
                        <div className="pt-1 space-y-0.5 text-[11px]">
                          <span className="text-emerald-400 block">Clean Acc: {data.clean_acc}%</span>
                          <span className="text-accent block">Corrupted Acc: {data.corrupted_acc}%</span>
                          <span className="text-purple-400 block">
                            Delta vs FP32:{' '}
                            {data.delta_vs_fp32_pp >= 0
                              ? `+${data.delta_vs_fp32_pp} pp`
                              : `${data.delta_vs_fp32_pp} pp`}
                          </span>
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
              <Bar
                dataKey="clean_acc"
                name="Clean Test Accuracy (%)"
                fill="#10b981"
                radius={[4, 4, 0, 0]}
                barSize={28}
              />
              <Bar
                dataKey="corrupted_acc"
                name="Corrupted Accuracy (%)"
                fill="#0284c7"
                radius={[4, 4, 0, 0]}
                barSize={28}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Detailed Precision Stability Table */}
        <div className="overflow-x-auto rounded-lg border border-surface-border">
          <table className="w-full text-left text-xs">
            <thead className="bg-surface-secondary/70 text-text-muted font-mono uppercase text-[10px] border-b border-surface-border">
              <tr>
                <th className="py-2.5 px-3">Precision Regime</th>
                <th className="py-2.5 px-3">Runtime Backend</th>
                <th className="py-2.5 px-3 text-right">Compression</th>
                <th className="py-2.5 px-3 text-right">Clean Test Acc (%)</th>
                <th className="py-2.5 px-3 text-right">Corrupted Acc (%)</th>
                <th className="py-2.5 px-3 text-right" title="Percentage point delta relative to own FP32 corrupted accuracy">
                  Delta vs FP32 (pp)
                </th>
                <th className="py-2.5 px-3 text-right" title="Clean prediction concordance relative to FP32">
                  Concordance (%)
                </th>
                <th className="py-2.5 px-3">Reliability Verdict</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-surface-border font-mono text-[11px]">
              {precisionComparison.map((row) => (
                <tr key={row.precision_key} className="hover:bg-surface-hover/30">
                  <td className="py-2.5 px-3 font-semibold text-text-primary flex items-center gap-1.5">
                    <Badge
                      variant={
                        row.precision_key === 'fp32'
                          ? 'default'
                          : row.precision_key === 'fp16'
                          ? 'purple'
                          : 'accent'
                      }
                      className="text-[10px]"
                    >
                      {row.precision}
                    </Badge>
                  </td>
                  <td className="py-2.5 px-3 text-text-secondary text-[11px]">
                    {row.runtime_backend}
                  </td>
                  <td className="py-2.5 px-3 text-right text-text-muted">
                    {row.compression_ratio.toFixed(2)}×
                  </td>
                  <td className="py-2.5 px-3 text-right text-emerald-400 font-bold">
                    {formatPercent(row.clean_acc)}
                  </td>
                  <td className="py-2.5 px-3 text-right text-accent font-bold">
                    {formatPercent(row.corrupted_acc)}
                  </td>
                  <td className="py-2.5 px-3 text-right font-bold">
                    <span
                      className={
                        row.delta_vs_fp32_pp > 0
                          ? 'text-emerald-400'
                          : row.delta_vs_fp32_pp < 0
                          ? 'text-rose-400'
                          : 'text-text-muted'
                      }
                    >
                      {row.delta_vs_fp32_pp > 0
                        ? `+${row.delta_vs_fp32_pp.toFixed(2)} pp`
                        : `${row.delta_vs_fp32_pp.toFixed(2)} pp`}
                    </span>
                  </td>
                  <td className="py-2.5 px-3 text-right text-text-secondary">
                    {formatPercent(row.concordance_pct)}
                  </td>
                  <td className="py-2.5 px-3 font-sans text-[11px]">
                    {row.precision_key === 'fp32' ? (
                      <span className="text-text-muted">Canonical baseline</span>
                    ) : row.delta_vs_fp32_pp >= -0.1 ? (
                      <span className="text-emerald-400 inline-flex items-center gap-1">
                        <CheckCircle2 className="w-3 h-3 shrink-0" />
                        Negligible degradation (Parity preserved)
                      </span>
                    ) : (
                      <span className="text-amber-400">Mild perturbation divergence</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Quantization Robustness Takeaway Note */}
        <div className="p-3.5 rounded-lg bg-surface-secondary/40 border border-surface-border text-xs text-text-secondary space-y-1.5">
          <div className="flex items-center gap-2 text-text-primary font-semibold">
            <Info className="w-4 h-4 text-accent shrink-0" />
            <span>Quantization Impact on Out-of-Distribution Robustness:</span>
          </div>
          <p className="text-[11px] leading-relaxed text-text-secondary">
            Across the 30 evaluated synthetic degradation conditions, INT8 models measured slightly higher mean corrupted accuracy than their FP32 counterparts. Specifically, ResNet-18 INT8 measured a mean corrupted accuracy of <strong>65.76%</strong> compared to <strong>65.73%</strong> for FP32 (+0.03 pp difference), while MobileNetV3-Small INT8 measured <strong>59.01%</strong> compared to <strong>58.83%</strong> for FP32 (+0.19 pp difference). The benchmark documents these observed performance differences across the tested conditions, but does not establish an underlying causal mechanism such as regularization or improved generalization.
          </p>
        </div>
      </CardContent>
    </Card>
  );
}
