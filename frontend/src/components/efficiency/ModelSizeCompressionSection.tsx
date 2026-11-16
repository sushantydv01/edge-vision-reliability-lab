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
import { Info } from 'lucide-react';
import type { PrecisionVariant } from '@/data/types';
import { formatRatio } from '@/data/formatters';

interface ModelSizeCompressionSectionProps {
  precisionVariants: PrecisionVariant[];
}

export function ModelSizeCompressionSection({
  precisionVariants,
}: ModelSizeCompressionSectionProps) {
  // Chart data comparing serialized size vs theoretical param payload
  const chartData = precisionVariants.map((v) => {
    const theoreticalMB = Number((v.theoretical_param_bytes / (1000 * 1000)).toFixed(2));
    const overheadMB = Number(Math.max(0, v.serialized_mb - theoreticalMB).toFixed(2));

    return {
      name: `${v.display_name === 'ResNet-18' ? 'ResNet' : 'MobileNet'} [${v.precision}]`,
      serialized_mb: v.serialized_mb,
      theoretical_mb: theoreticalMB,
      overhead_mb: overheadMB,
      compression_ratio: v.compression_ratio,
      size_reduction_pct: v.size_reduction_pct,
      parameters: v.parameters,
      model_id: v.model_id,
      precision: v.precision,
    };
  });

  return (
    <Card className="border-surface-border bg-surface/90">
      <CardHeader className="pb-3 border-b border-surface-border">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <div className="flex items-center gap-2">
              <Badge variant="accent" className="font-mono text-[10px]">
                Section D
              </Badge>
              <CardTitle className="text-base font-bold text-text-primary">
                Model Storage Footprint & Quantization Compression Matrix
              </CardTitle>
            </div>
            <CardDescription className="text-xs mt-1">
              Exact on-disk serialized byte counts, theoretical parameter footprints, and architecture-specific compression ratios across all six deployed configurations.
            </CardDescription>
          </div>

          <Badge variant="outline" className="font-mono text-xs self-start sm:self-auto">
            6 Deployed Checkpoints Evaluated
          </Badge>
        </div>
      </CardHeader>

      <CardContent className="pt-4 space-y-5">
        {/* Visual Bar Chart: Serialized Storage Footprint */}
        <div className="h-72 w-full bg-surface-secondary/20 p-3 rounded-xl border border-surface-border">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={chartData}
              margin={{ top: 15, right: 25, left: 10, bottom: 25 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#262626" />
              <XAxis
                dataKey="name"
                stroke="#737373"
                tick={{ fill: '#a3a3a3', fontSize: 10, fontFamily: 'monospace' }}
              />
              <YAxis
                stroke="#737373"
                unit=" MB"
                tick={{ fill: '#a3a3a3', fontSize: 11, fontFamily: 'monospace' }}
              />
              <RechartsTooltip
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    const data = payload[0].payload;
                    return (
                      <div className="bg-surface border border-surface-border p-3 rounded shadow-xl text-xs font-mono space-y-1">
                        <span className="font-bold text-text-primary block">{data.name}</span>
                        <div className="pt-1 space-y-0.5 text-[11px]">
                          <span className="text-purple-400 block">
                            Serialized File Size: <strong>{data.serialized_mb} MB</strong>
                          </span>
                          <span className="text-accent block">
                            Compression Ratio: <strong>{data.compression_ratio.toFixed(2)}×</strong> ({data.size_reduction_pct.toFixed(1)}% reduction)
                          </span>
                          <span className="text-text-muted block">
                            Theoretical Weight Payload: <strong>{data.theoretical_mb} MB</strong>
                          </span>
                          <span className="text-text-secondary block">
                            Parameters: <strong>{data.parameters.toLocaleString()}</strong>
                          </span>
                        </div>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <Legend wrapperStyle={{ fontSize: '11px', fontFamily: 'monospace', paddingTop: '10px' }} />
              <Bar
                dataKey="serialized_mb"
                name="Actual Serialized File Size on Disk (MB)"
                fill="#a855f7"
                radius={[4, 4, 0, 0]}
              />
              <Bar
                dataKey="theoretical_mb"
                name="Theoretical Raw Weight Payload (MB)"
                fill="#0284c7"
                radius={[4, 4, 0, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Full 6-Way Storage Footprint Table */}
        <div className="overflow-x-auto rounded-lg border border-surface-border font-mono text-xs">
          <table className="w-full text-left">
            <thead className="bg-surface-secondary/70 text-text-muted uppercase text-[10px] border-b border-surface-border">
              <tr>
                <th className="py-2.5 px-3">Architecture</th>
                <th className="py-2.5 px-3">Precision</th>
                <th className="py-2.5 px-3">Runtime Backend</th>
                <th className="py-2.5 px-3 text-right">Parameter Count</th>
                <th className="py-2.5 px-3 text-right">Exact Serialized Bytes</th>
                <th className="py-2.5 px-3 text-right">Serialized Size</th>
                <th className="py-2.5 px-3 text-right">Compression Ratio</th>
                <th className="py-2.5 px-3 text-right">Size Reduction</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-surface-border text-[11px]">
              {precisionVariants.map((v) => (
                <tr key={`${v.model_id}-${v.precision_key}`} className="hover:bg-surface-hover/30">
                  <td className="py-2.5 px-3 font-semibold text-text-primary">
                    {v.display_name}
                  </td>
                  <td className="py-2.5 px-3">
                    <Badge
                      variant={v.precision_key === 'int8' ? 'accent' : v.precision_key === 'fp16' ? 'purple' : 'default'}
                      className="text-[9px]"
                    >
                      {v.precision}
                    </Badge>
                  </td>
                  <td className="py-2.5 px-3 text-text-muted text-[10px]">
                    {v.runtime_backend}
                  </td>
                  <td className="py-2.5 px-3 text-right text-text-secondary">
                    {v.parameters.toLocaleString()}
                  </td>
                  <td className="py-2.5 px-3 text-right text-text-primary font-bold">
                    {v.serialized_bytes.toLocaleString()} B
                  </td>
                  <td className="py-2.5 px-3 text-right text-purple-400 font-bold">
                    {v.serialized_mb.toFixed(2)} MB
                  </td>
                  <td className="py-2.5 px-3 text-right font-bold text-accent">
                    {v.compression_ratio === 1 ? '1.00× (Baseline)' : formatRatio(v.compression_ratio)}
                  </td>
                  <td className="py-2.5 px-3 text-right font-bold text-emerald-400">
                    {v.size_reduction_pct === 0 ? '0.0%' : `-${v.size_reduction_pct.toFixed(2)}%`}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Detailed Canonical Byte Verification Note */}
        <div className="p-3.5 rounded-lg bg-surface-secondary/40 border border-surface-border text-xs text-text-secondary space-y-1.5 font-sans">
          <div className="flex items-center gap-2 text-text-primary font-semibold">
            <Info className="w-4 h-4 text-accent shrink-0" />
            <span>Actual Serialized Artifact Sizes vs. Theoretical Parametric Payload:</span>
          </div>
          <p className="text-[11px] leading-relaxed text-text-secondary">
            Reported file sizes reflect canonical serialized artifacts on disk measured via <code>stat().st_size</code>. In TorchScript INT8 archives, static scale factors, zero-point quantization vectors, graph schemas, and PyTorch container metadata add ~0.12–0.41 MB beyond theoretical raw integer payloads. Consequently, ResNet-18 INT8 measures <strong>11,296,669 bytes (11.30 MB, 74.77% reduction, 3.96×)</strong> rather than theoretical 4.00×, and MobileNetV3-Small INT8 measures <strong>1,942,499 bytes (1.94 MB, 68.92% reduction, 3.22×)</strong>.
          </p>
        </div>
      </CardContent>
    </Card>
  );
}
