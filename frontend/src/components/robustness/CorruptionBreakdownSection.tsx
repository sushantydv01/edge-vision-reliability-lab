import { useState, useMemo } from 'react';
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
import { Sparkles } from 'lucide-react';
import type { RobustnessSuite } from '@/data/types';
import { formatPercent } from '@/data/formatters';
import type { RobustnessFilterState } from './RobustnessControls';

interface CorruptionBreakdownSectionProps {
  filters: RobustnessFilterState;
  suite: RobustnessSuite;
}

export function CorruptionBreakdownSection({
  filters,
  suite,
}: CorruptionBreakdownSectionProps) {
  const [selectedCategory, setSelectedCategory] = useState<string>('all');

  const cleanAcc = suite.clean_baselines[`${filters.model_id}_${filters.precision}`] ?? 94.78;

  // Compute stats for each of the 6 corruptions under the current active model and precision
  const corruptionStats = useMemo(() => {
    const keys = Object.keys(suite.corruption_metadata);

    return keys.map((key) => {
      const meta = suite.corruption_metadata[key];
      // Filter conditions for this corruption (matching severity filter if set)
      const matching = suite.detailed_conditions.filter((c) => {
        if (c.model_id !== filters.model_id) return false;
        if (c.precision !== filters.precision) return false;
        if (c.corruption !== key) return false;
        if (filters.severity !== 'all' && c.severity !== Number(filters.severity)) return false;
        return true;
      });

      const avgAcc =
        matching.length > 0
          ? matching.reduce((sum, c) => sum + c.accuracy, 0) / matching.length
          : 0;

      const avgDropPP =
        matching.length > 0
          ? matching.reduce((sum, c) => sum + c.accuracy_drop_pp, 0) / matching.length
          : 0;

      const avgRelDrop =
        matching.length > 0
          ? matching.reduce((sum, c) => sum + c.relative_accuracy_drop, 0) / matching.length
          : 0;

      // Check comparison against the other model (to identify crossovers like JPEG compression)
      const otherModelId =
        filters.model_id === 'resnet18_cifar' ? 'mobilenet_v3_small_cifar' : 'resnet18_cifar';
      const otherMatching = suite.detailed_conditions.filter((c) => {
        if (c.model_id !== otherModelId) return false;
        if (c.precision !== filters.precision) return false;
        if (c.corruption !== key) return false;
        if (filters.severity !== 'all' && c.severity !== Number(filters.severity)) return false;
        return true;
      });
      const otherAvgAcc =
        otherMatching.length > 0
          ? otherMatching.reduce((sum, c) => sum + c.accuracy, 0) / otherMatching.length
          : 0;

      const isCrossover =
        filters.model_id === 'mobilenet_v3_small_cifar'
          ? avgAcc > otherAvgAcc
          : otherAvgAcc > avgAcc;

      return {
        key,
        name: meta.name,
        category: meta.category,
        description: meta.description,
        param_name: meta.param_name,
        corrupted_acc: Number(avgAcc.toFixed(2)),
        clean_acc: Number(cleanAcc.toFixed(2)),
        drop_pp: Number(avgDropPP.toFixed(2)),
        rel_drop_pct: Number(avgRelDrop.toFixed(2)),
        isCrossover,
      };
    });
  }, [filters, suite, cleanAcc]);

  // Filter by category
  const filteredCorruptions = corruptionStats.filter((c) => {
    if (selectedCategory === 'all') return true;
    return c.category === selectedCategory;
  });

  // Find the corruption with largest drop
  const largestDropCorr = [...corruptionStats].sort((a, b) => b.drop_pp - a.drop_pp)[0];

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
                Corruption Domain Breakdown & Sensitivity Matrix
              </CardTitle>
            </div>
            <CardDescription className="text-xs mt-1">
              Contrasting model degradation across noise, blur, photometric, and compression families to isolate inductive bias vulnerabilities.
            </CardDescription>
          </div>

          {/* Category Filter Pills */}
          <div className="flex flex-wrap items-center gap-1.5 self-start sm:self-auto bg-surface-secondary/70 p-1 rounded-lg border border-surface-border">
            {['all', 'Noise', 'Blur', 'Photometric', 'Digital'].map((cat) => (
              <button
                key={cat}
                onClick={() => setSelectedCategory(cat)}
                className={`px-2.5 py-1 text-xs font-mono rounded transition-colors ${
                  selectedCategory === cat
                    ? 'bg-accent/20 text-accent font-bold border border-accent/40'
                    : 'text-text-muted hover:text-text-primary'
                }`}
              >
                {cat === 'all' ? 'All (6)' : cat}
              </button>
            ))}
          </div>
        </div>
      </CardHeader>

      <CardContent className="pt-4 space-y-5">
        {/* Visual Bar Chart */}
        <div className="h-72 w-full bg-surface-secondary/20 p-3 rounded-xl border border-surface-border">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={filteredCorruptions}
              margin={{ top: 15, right: 20, left: 0, bottom: 20 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#262626" />
              <XAxis
                dataKey="name"
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
                        <span className="font-bold text-text-primary block">{data.name}</span>
                        <span className="text-[10px] text-text-muted block">Category: {data.category}</span>
                        <div className="pt-1 space-y-0.5 text-[11px]">
                          <span className="text-emerald-400 block">
                            Clean Accuracy: <strong>{data.clean_acc}%</strong>
                          </span>
                          <span className="text-accent block">
                            Corrupted Accuracy: <strong>{data.corrupted_acc}%</strong>
                          </span>
                          <span className="text-rose-400 block">
                            Absolute Drop: <strong>-{data.drop_pp} pp</strong>
                          </span>
                          <span className="text-amber-400 block">
                            Relative Drop: <strong>{data.rel_drop_pct}%</strong>
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
                barSize={20}
              />
              <Bar
                dataKey="corrupted_acc"
                name="Corrupted Accuracy (%)"
                fill="#0284c7"
                radius={[4, 4, 0, 0]}
                barSize={20}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* 6 Corruption Cards Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
          {filteredCorruptions.map((item) => {
            const isWorst = item.key === largestDropCorr?.key;

            return (
              <div
                key={item.key}
                className={`p-3.5 rounded-xl border bg-surface-secondary/40 space-y-2.5 transition-colors ${
                  isWorst ? 'border-rose-500/40 bg-rose-500/[0.03]' : 'border-surface-border'
                }`}
              >
                <div className="flex items-center justify-between pb-1 border-b border-surface-border">
                  <div>
                    <span className="font-bold text-xs text-text-primary block">{item.name}</span>
                    <span className="text-[10px] font-mono text-text-muted">{item.category} Family</span>
                  </div>
                  {isWorst ? (
                    <Badge variant="outline" className="text-rose-400 border-rose-500/40 text-[9px]">
                      Worst Drop
                    </Badge>
                  ) : item.key === 'jpeg_compression' ? (
                    <Badge variant="purple" className="text-[9px]">
                      Crossover Site
                    </Badge>
                  ) : (
                    <Badge variant="outline" className="text-[9px] text-text-muted">
                      {item.category}
                    </Badge>
                  )}
                </div>

                <div className="grid grid-cols-2 gap-2 text-center font-mono text-xs">
                  <div className="p-2 rounded bg-surface/80 border border-surface-border">
                    <span className="text-[10px] uppercase text-text-muted block">Corrupted Acc</span>
                    <strong className="text-accent text-sm">{formatPercent(item.corrupted_acc)}</strong>
                  </div>
                  <div className="p-2 rounded bg-surface/80 border border-surface-border">
                    <span className="text-[10px] uppercase text-text-muted block">Absolute Drop</span>
                    <strong className="text-rose-400 text-sm">-{item.drop_pp.toFixed(2)} pp</strong>
                  </div>
                </div>

                <div className="flex items-center justify-between text-[11px] font-mono text-text-secondary pt-1 border-t border-surface-border/50">
                  <span>Relative Degradation:</span>
                  <strong className="text-amber-400">{item.rel_drop_pct.toFixed(2)}%</strong>
                </div>

                <p className="text-[10px] font-sans text-text-muted leading-tight">
                  {item.description}
                </p>
              </div>
            );
          })}
        </div>

        {/* Crossover & Empirical Insight Banner */}
        <div className="p-3.5 rounded-lg bg-surface-secondary/40 border border-surface-border text-xs text-text-secondary space-y-2">
          <div className="flex items-center gap-2 text-text-primary font-semibold">
            <Sparkles className="w-4 h-4 text-accent shrink-0" />
            <span>Notable Architectural & Invariance Findings:</span>
          </div>
          <ul className="space-y-1.5 text-[11px] list-disc list-inside text-text-secondary leading-relaxed">
            <li>
              <strong>JPEG Compression Crossover:</strong> In this evaluation, MobileNetV3-Small measured higher mean accuracy under JPEG compression than ResNet-18 (69.66% vs 67.83% in FP32, a +1.83 pp difference). The benchmark records this empirical crossover across all three evaluated precision configurations, but does not independently isolate the architectural components responsible.
            </li>
            <li>
              <strong>Noise Sensitivity:</strong> Gaussian noise was accompanied by the largest observed accuracy drop across both models (-49.32 pp on ResNet-18 and -43.97 pp on MobileNetV3-Small), establishing that additive pixel noise produced the greatest performance degradation among tested synthetic perturbations.
            </li>
          </ul>
        </div>
      </CardContent>
    </Card>
  );
}
