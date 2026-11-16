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
import { Info, ArrowDown } from 'lucide-react';
import type { PrecisionVariant } from '@/data/types';
import { formatPercent } from '@/data/formatters';

interface CleanCorruptedPerformanceSectionProps {
  precisionVariants: PrecisionVariant[];
}

export function CleanCorruptedPerformanceSection({
  precisionVariants,
}: CleanCorruptedPerformanceSectionProps) {
  const [archFilter, setArchFilter] = useState<'all' | 'resnet18_cifar' | 'mobilenet_v3_small_cifar'>('all');

  const filteredVariants = precisionVariants.filter((v) => {
    if (archFilter === 'all') return true;
    return v.model_id === archFilter;
  });

  const chartData = filteredVariants.map((v) => ({
    name: `${v.display_name === 'ResNet-18' ? 'ResNet' : 'MobileNet'} ${v.precision}`,
    clean_accuracy: Number(v.clean_accuracy.toFixed(2)),
    corrupted_accuracy: Number(v.mean_corrupted_accuracy.toFixed(2)),
    accuracy_drop_pp: Number(v.mean_accuracy_drop_pp.toFixed(2)),
    precision: v.precision,
    display_name: v.display_name,
  }));

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
                Clean vs. Synthetic Corrupted Performance
              </CardTitle>
            </div>
            <CardDescription className="text-xs mt-1">
              Contrasting nominal test accuracy against mean predictive reliability across 30 custom synthetic corruption conditions.
            </CardDescription>
          </div>

          {/* Architecture Filter Buttons */}
          <div className="flex items-center gap-1.5 self-start sm:self-auto bg-surface-secondary/70 p-1 rounded-lg border border-surface-border">
            <button
              onClick={() => setArchFilter('all')}
              className={`px-2.5 py-1 text-xs font-mono rounded transition-colors ${
                archFilter === 'all'
                  ? 'bg-accent/20 text-accent font-bold border border-accent/40'
                  : 'text-text-muted hover:text-text-primary'
              }`}
            >
              All Models (6)
            </button>
            <button
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
              onClick={() => setArchFilter('mobilenet_v3_small_cifar')}
              className={`px-2.5 py-1 text-xs font-mono rounded transition-colors ${
                archFilter === 'mobilenet_v3_small_cifar'
                  ? 'bg-accent/20 text-accent font-bold border border-accent/40'
                  : 'text-text-muted hover:text-text-primary'
              }`}
            >
              MobileNetV3-Small
            </button>
          </div>
        </div>
      </CardHeader>

      <CardContent className="pt-4 space-y-5">
        {/* Visual Chart */}
        <div className="h-72 w-full bg-surface-secondary/20 p-3 rounded-xl border border-surface-border">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={chartData}
              margin={{ top: 15, right: 20, left: 0, bottom: 20 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#262626" />
              <XAxis
                dataKey="name"
                stroke="#737373"
                tick={{ fill: '#a3a3a3', fontSize: 11, fontFamily: 'monospace' }}
              />
              <YAxis
                stroke="#737373"
                domain={[0, 100]}
                unit="%"
                tick={{ fill: '#a3a3a3', fontSize: 11, fontFamily: 'monospace' }}
              />
              <RechartsTooltip
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    const data = payload[0].payload;
                    return (
                      <div className="bg-surface border border-surface-border p-2.5 rounded shadow-xl text-xs font-mono space-y-1">
                        <span className="font-bold text-text-primary block">{data.name}</span>
                        <span className="text-emerald-400 block">
                          Clean Test Accuracy: <strong>{data.clean_accuracy}%</strong>
                        </span>
                        <span className="text-accent block">
                          Mean Corrupted Accuracy: <strong>{data.corrupted_accuracy}%</strong>
                        </span>
                        <span className="text-rose-400 block">
                          Mean Accuracy Drop: <strong>-{data.accuracy_drop_pp} pp</strong>
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
                dataKey="clean_accuracy"
                name="Clean Test Accuracy (%)"
                fill="#10b981"
                radius={[4, 4, 0, 0]}
                barSize={24}
              />
              <Bar
                dataKey="corrupted_accuracy"
                name="Mean Corrupted Accuracy (%)"
                fill="#0284c7"
                radius={[4, 4, 0, 0]}
                barSize={24}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Accuracy Drop Cards Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {filteredVariants.map((v) => (
            <div
              key={v.config_id}
              className="p-3 rounded-lg border border-surface-border bg-surface-secondary/40 space-y-2"
            >
              <div className="flex items-center justify-between">
                <span className="font-semibold text-xs text-text-primary">
                  {v.display_name}
                </span>
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
              </div>

              <div className="grid grid-cols-2 gap-2 text-center font-mono text-[11px]">
                <div className="p-1.5 rounded bg-surface/70 border border-surface-border">
                  <span className="text-[10px] uppercase text-text-muted block">Clean</span>
                  <span className="font-bold text-emerald-400">
                    {formatPercent(v.clean_accuracy)}
                  </span>
                </div>
                <div className="p-1.5 rounded bg-surface/70 border border-surface-border">
                  <span className="text-[10px] uppercase text-text-muted block">Corrupted</span>
                  <span className="font-bold text-accent">
                    {formatPercent(v.mean_corrupted_accuracy)}
                  </span>
                </div>
              </div>

              <div className="flex items-center justify-between text-xs font-mono pt-1 border-t border-surface-border/60">
                <span className="text-[11px] text-text-muted flex items-center gap-1">
                  <ArrowDown className="w-3 h-3 text-rose-400" />
                  Mean Degradation:
                </span>
                <span className="font-bold text-rose-400">
                  -{v.mean_accuracy_drop_pp.toFixed(2)} pp
                </span>
              </div>
            </div>
          ))}
        </div>

        {/* Protocol & Dataset Context Note */}
        <div className="p-3.5 rounded-lg bg-surface-secondary/40 border border-surface-border text-xs text-text-secondary space-y-1.5">
          <div className="flex items-center gap-2 text-text-primary font-semibold">
            <Info className="w-4 h-4 text-accent shrink-0" />
            <span>Robustness Evaluation Protocol & Scope:</span>
          </div>
          <p className="text-[11px] leading-relaxed text-text-secondary">
            <strong>Clean test accuracy</strong> is evaluated on the standard held-out CIFAR-10 test split (10,000 images at canonical 32×32 resolution). <strong>Mean corrupted accuracy</strong> is computed across 300,000 evaluations per model configuration (10,000 test images × 6 custom synthetic transformations × 5 severity levels = 30 conditions). This suite comprises <em>Gaussian noise, Gaussian blur, horizontal motion blur, brightness darkening, contrast attenuation, and JPEG compression</em>. It does <strong>not</strong> use the official 15-corruption CIFAR-10-C benchmark.
          </p>
        </div>
      </CardContent>
    </Card>
  );
}
