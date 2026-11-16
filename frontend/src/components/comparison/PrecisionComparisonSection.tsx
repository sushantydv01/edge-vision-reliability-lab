import { useState } from 'react';
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
import { HardDrive, Info } from 'lucide-react';
import type { PrecisionVariant } from '@/data/types';
import { formatPercent, formatRatio } from '@/data/formatters';

interface PrecisionComparisonSectionProps {
  precisionVariants: PrecisionVariant[];
}

export function PrecisionComparisonSection({
  precisionVariants,
}: PrecisionComparisonSectionProps) {
  const [selectedArchFilter, setSelectedArchFilter] = useState<'all' | 'resnet18_cifar' | 'mobilenet_v3_small_cifar'>('all');

  const filteredVariants = precisionVariants.filter((v) => {
    if (selectedArchFilter === 'all') return true;
    return v.model_id === selectedArchFilter;
  });

  // Chart data preparing compression vs clean vs corrupted accuracy
  const chartData = filteredVariants.map((v) => ({
    name: `${v.display_name === 'ResNet-18' ? 'ResNet' : 'MobileNet'} ${v.precision}`,
    model_id: v.model_id,
    precision: v.precision,
    serialized_mb: Number(v.serialized_mb.toFixed(2)),
    clean_accuracy: Number(v.clean_accuracy.toFixed(2)),
    corrupted_accuracy: Number(v.mean_corrupted_accuracy.toFixed(2)),
    compression_ratio: v.compression_ratio,
  }));

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
                Precision Quantization Comparison: FP32, FP16 & INT8
              </CardTitle>
            </div>
            <CardDescription className="text-xs mt-1">
              Evaluating storage footprint, compression ratio, prediction concordance, and accuracy retention under numerical precision reduction.
            </CardDescription>
          </div>

          {/* Filter Pills */}
          <div className="flex items-center gap-1.5 self-start sm:self-auto bg-surface-secondary/70 p-1 rounded-lg border border-surface-border">
            <button
              onClick={() => setSelectedArchFilter('all')}
              className={`px-2.5 py-1 text-xs font-mono rounded transition-colors ${
                selectedArchFilter === 'all'
                  ? 'bg-accent/20 text-accent font-bold border border-accent/40'
                  : 'text-text-muted hover:text-text-primary'
              }`}
            >
              All Variants (6)
            </button>
            <button
              onClick={() => setSelectedArchFilter('resnet18_cifar')}
              className={`px-2.5 py-1 text-xs font-mono rounded transition-colors ${
                selectedArchFilter === 'resnet18_cifar'
                  ? 'bg-accent/20 text-accent font-bold border border-accent/40'
                  : 'text-text-muted hover:text-text-primary'
              }`}
            >
              ResNet-18
            </button>
            <button
              onClick={() => setSelectedArchFilter('mobilenet_v3_small_cifar')}
              className={`px-2.5 py-1 text-xs font-mono rounded transition-colors ${
                selectedArchFilter === 'mobilenet_v3_small_cifar'
                  ? 'bg-accent/20 text-accent font-bold border border-accent/40'
                  : 'text-text-muted hover:text-text-primary'
              }`}
            >
              MobileNetV3-Small
            </button>
          </div>
        </div>
      </CardHeader>

      <CardContent className="pt-4 space-y-6">
        {/* Full Precision Comparison Table */}
        <div className="overflow-x-auto rounded-lg border border-surface-border">
          <table className="w-full text-left text-xs">
            <thead className="bg-surface-secondary/70 text-text-muted font-mono uppercase text-[10px] border-b border-surface-border">
              <tr>
                <th className="py-2.5 px-3">Architecture</th>
                <th className="py-2.5 px-3">Precision</th>
                <th className="py-2.5 px-3">Runtime Backend</th>
                <th className="py-2.5 px-3 text-right">Serialized Size (MB)</th>
                <th className="py-2.5 px-3 text-right" title="Calculated relative to the same architecture's FP32 baseline on disk">Compression Ratio</th>
                <th className="py-2.5 px-3 text-right">Clean Test Acc. (%)</th>
                <th className="py-2.5 px-3 text-right" title="Percentage point delta relative to the same architecture's clean FP32 accuracy">Delta vs FP32 (pp)</th>
                <th className="py-2.5 px-3 text-right" title="Exact prediction agreement percentage on 10,000 clean test samples relative to the same architecture's FP32 baseline">Concordance vs FP32 (%)</th>
                <th className="py-2.5 px-3 text-right" title="Number of changed class predictions out of 10,000 clean test samples relative to the same architecture's FP32 baseline">Changed Preds</th>
                <th className="py-2.5 px-3 text-right">Corrupted Acc. (%)</th>
                <th className="py-2.5 px-3 text-right">Mean Drop (pp)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-surface-border font-mono text-[11px]">
              {filteredVariants.map((v) => {
                const isBaseline = v.precision_key === 'fp32';
                const deltaVsFp32 = isBaseline
                  ? 0
                  : v.clean_accuracy -
                    (precisionVariants.find(
                      (b) => b.model_id === v.model_id && b.precision_key === 'fp32'
                    )?.clean_accuracy || v.clean_accuracy);

                return (
                  <tr key={v.config_id} className="hover:bg-surface-hover/30">
                    <td className="py-2.5 px-3 font-semibold text-text-primary">
                      {v.display_name}
                    </td>
                    <td className="py-2.5 px-3">
                      <Badge
                        variant={
                          v.precision === 'FP32'
                            ? 'default'
                            : v.precision === 'FP16'
                            ? 'purple'
                            : 'accent'
                        }
                        className="font-mono text-[10px]"
                      >
                        {v.precision}
                      </Badge>
                    </td>
                    <td className="py-2.5 px-3 text-text-secondary text-[11px]">
                      {v.runtime_backend}
                    </td>
                    <td className="py-2.5 px-3 text-right font-bold text-text-primary">
                      {v.serialized_mb.toFixed(2)} MB
                    </td>
                    <td className="py-2.5 px-3 text-right text-accent font-semibold">
                      {isBaseline ? '1.00× (Baseline)' : formatRatio(v.compression_ratio)}
                    </td>
                    <td className="py-2.5 px-3 text-right font-bold text-emerald-400">
                      {formatPercent(v.clean_accuracy)}
                    </td>
                    <td className="py-2.5 px-3 text-right">
                      {isBaseline ? (
                        <span className="text-text-muted">—</span>
                      ) : deltaVsFp32 > 0 ? (
                        <span className="text-emerald-400 font-bold">+{deltaVsFp32.toFixed(2)} pp</span>
                      ) : deltaVsFp32 < 0 ? (
                        <span className="text-rose-400 font-bold">{deltaVsFp32.toFixed(2)} pp</span>
                      ) : (
                        <span className="text-text-secondary">0.00 pp</span>
                      )}
                    </td>
                    <td className="py-2.5 px-3 text-right text-text-primary">
                      {isBaseline ? '100.0%' : formatPercent(v.concordance_pct)}
                    </td>
                    <td className="py-2.5 px-3 text-right text-text-secondary">
                      {isBaseline ? '0 / 10k' : `${v.changed_predictions} / 10k`}
                    </td>
                    <td className="py-2.5 px-3 text-right font-semibold text-text-primary">
                      {formatPercent(v.mean_corrupted_accuracy)}
                    </td>
                    <td className="py-2.5 px-3 text-right text-rose-400 font-semibold">
                      -{v.mean_accuracy_drop_pp.toFixed(2)} pp
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* Visual Chart: Storage Footprint vs Accuracy Retention */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <h4 className="text-xs font-mono uppercase font-bold text-text-secondary flex items-center gap-1.5">
              <HardDrive className="w-3.5 h-3.5 text-accent" />
              Serialized Disk Footprint (MB) vs Accuracy (%) Across Precision Variants
            </h4>
            <span className="text-[11px] text-text-muted font-mono">
              Evaluated on held-out CIFAR-10 test split
            </span>
          </div>

          <div className="h-64 w-full bg-surface-secondary/20 p-3 rounded-xl border border-surface-border">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#262626" />
                <XAxis
                  dataKey="name"
                  stroke="#737373"
                  tick={{ fill: '#a3a3a3', fontSize: 11, fontFamily: 'monospace' }}
                />
                <YAxis
                  yAxisId="left"
                  orientation="left"
                  stroke="#0284c7"
                  domain={[0, 50]}
                  unit=" MB"
                  tick={{ fill: '#38bdf8', fontSize: 11, fontFamily: 'monospace' }}
                />
                <YAxis
                  yAxisId="right"
                  orientation="right"
                  stroke="#10b981"
                  domain={[50, 100]}
                  unit="%"
                  tick={{ fill: '#34d399', fontSize: 11, fontFamily: 'monospace' }}
                />
                <RechartsTooltip
                  content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      const data = payload[0].payload;
                      return (
                        <div className="bg-surface border border-surface-border p-2.5 rounded shadow-xl text-xs font-mono space-y-1">
                          <span className="font-bold text-text-primary block">{data.name}</span>
                          <span className="text-accent block">
                            Serialized Disk Size: <strong>{data.serialized_mb} MB</strong> ({data.compression_ratio}× compression)
                          </span>
                          <span className="text-emerald-400 block">
                            Clean Test Acc: <strong>{data.clean_accuracy}%</strong>
                          </span>
                          <span className="text-purple-400 block">
                            Mean Corrupted Acc: <strong>{data.corrupted_accuracy}%</strong>
                          </span>
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
                  yAxisId="left"
                  dataKey="serialized_mb"
                  name="Serialized Size (MB)"
                  fill="#0284c7"
                  radius={[4, 4, 0, 0]}
                  barSize={20}
                />
                <Bar
                  yAxisId="right"
                  dataKey="clean_accuracy"
                  name="Clean Accuracy (%)"
                  fill="#10b981"
                  radius={[4, 4, 0, 0]}
                  barSize={20}
                />
                <Bar
                  yAxisId="right"
                  dataKey="corrupted_accuracy"
                  name="Mean Corrupted Accuracy (%)"
                  fill="#a855f7"
                  radius={[4, 4, 0, 0]}
                  barSize={20}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Quantization & Serialization Technical Distinction Note */}
        <div className="p-3.5 rounded-lg bg-surface-secondary/40 border border-surface-border text-xs text-text-secondary space-y-2">
          <div className="flex items-center gap-2 text-text-primary font-semibold">
            <Info className="w-4 h-4 text-accent shrink-0" />
            <span>Important Deployment & Quantization Discrepancies:</span>
          </div>
          <ul className="space-y-1.5 text-[11px] list-disc list-inside text-text-secondary leading-relaxed">
            <li>
              <strong>Serialized File Size vs. Theoretical Bit-Width:</strong> Reported artifact sizes reflect actual serialized files on disk. INT8 TorchScript archives contain static quantization scales, zero-point tables, schema metadata, and computational graph headers. Consequently, ResNet-18 INT8 measures 11.30 MB (3.96× reduction) and MobileNetV3-Small INT8 measures 1.94 MB (3.22× reduction), rather than exactly 4.00×.
            </li>
            <li>
              <strong>Execution Target Asymmetry:</strong> In this evaluation suite, INT8 static quantization executes strictly on CPU via PyTorch's QNNPACK backend. Conversely, FP16 half-precision execution was evaluated on Apple Silicon MPS (Metal Performance Shaders). Performance metrics must not be compared without taking execution hardware and backend runtime into account.
            </li>
            <li>
              <strong>Predictive Concordance:</strong> Prediction concordance measures exact label agreement against the FP32 baseline on all 10,000 clean test samples. ResNet-18 INT8 maintained 99.64% concordance (36 disagreements), while MobileNetV3-Small INT8 measured 95.47% concordance (453 disagreements), indicating greater sensitivity in depthwise separable inverted residual blocks.
            </li>
          </ul>
        </div>
      </CardContent>
    </Card>
  );
}
