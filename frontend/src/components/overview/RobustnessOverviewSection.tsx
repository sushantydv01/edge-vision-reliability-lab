import { useState } from 'react';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from 'recharts';
import { TrendingDown, Sparkles } from 'lucide-react';
import { researchData, formatPercent, formatDrop } from '@/data';

export function RobustnessOverviewSection() {
  const { severityCurves, perCorruptionData, baselineComparison } = researchData;
  const [activeChart, setActiveChart] = useState<'severity' | 'corruption'>('severity');

  return (
    <section className="space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-1 border-b border-surface-border">
        <div>
          <h2 className="text-lg font-bold text-text-primary tracking-tight">
            Environmental Robustness Analysis
          </h2>
          <p className="text-xs text-text-secondary">
            Empirical accuracy across 6 custom synthetic transformations and 5 discrete severity levels.
          </p>
        </div>
        <div className="flex items-center gap-1.5 p-1 bg-surface-secondary/70 rounded-lg border border-surface-border">
          <button
            onClick={() => setActiveChart('severity')}
            className={`px-2.5 py-1 text-xs font-medium rounded-md transition-all ${
              activeChart === 'severity'
                ? 'bg-accent text-white shadow-xs'
                : 'text-text-secondary hover:text-text-primary'
            }`}
          >
            Severity Degradation (1–5)
          </button>
          <button
            onClick={() => setActiveChart('corruption')}
            className={`px-2.5 py-1 text-xs font-medium rounded-md transition-all ${
              activeChart === 'corruption'
                ? 'bg-accent text-white shadow-xs'
                : 'text-text-secondary hover:text-text-primary'
            }`}
          >
            Per-Corruption Profile
          </button>
        </div>
      </div>

      {/* Summary Scorecards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <div className="p-3 bg-surface border border-surface-border rounded-lg space-y-0.5">
          <span className="text-[10px] text-text-muted uppercase font-mono">ResNet-18 Clean Ref.</span>
          <div className="text-sm sm:text-base font-bold font-mono text-emerald-400">
            {formatPercent(baselineComparison.resnet18.clean_accuracy)}
          </div>
          <div className="text-[10px] text-text-muted">Uncorrupted test set</div>
        </div>

        <div className="p-3 bg-surface border border-surface-border rounded-lg space-y-0.5">
          <span className="text-[10px] text-text-muted uppercase font-mono">MobileNet Clean Ref.</span>
          <div className="text-sm sm:text-base font-bold font-mono text-emerald-400">
            {formatPercent(baselineComparison.mobilenet_v3_small.clean_accuracy)}
          </div>
          <div className="text-[10px] text-text-muted">Uncorrupted test set</div>
        </div>

        <div className="p-3 bg-surface border border-surface-border rounded-lg space-y-0.5">
          <span className="text-[10px] text-text-muted uppercase font-mono">ResNet-18 Corrupted</span>
          <div className="text-sm sm:text-base font-bold font-mono text-accent">
            {formatPercent(baselineComparison.resnet18.mean_corrupted_accuracy)}
          </div>
          <div className="text-[10px] text-rose-400 flex items-center gap-1">
            <TrendingDown className="w-3 h-3" />
            <span>{formatDrop(baselineComparison.resnet18.mean_accuracy_drop_pp)} drop</span>
          </div>
        </div>

        <div className="p-3 bg-surface border border-surface-border rounded-lg space-y-0.5">
          <span className="text-[10px] text-text-muted uppercase font-mono">MobileNet Corrupted</span>
          <div className="text-sm sm:text-base font-bold font-mono text-purple-400">
            {formatPercent(baselineComparison.mobilenet_v3_small.mean_corrupted_accuracy)}
          </div>
          <div className="text-[10px] text-rose-400 flex items-center gap-1">
            <TrendingDown className="w-3 h-3" />
            <span>{formatDrop(baselineComparison.mobilenet_v3_small.mean_accuracy_drop_pp)} drop</span>
          </div>
        </div>
      </div>

      {/* Main Interactive Charts Container */}
      <Card>
        <CardHeader>
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <div>
              <CardTitle className="text-sm font-semibold">
                {activeChart === 'severity'
                  ? 'Degradation Trajectories Across Severity Levels (Clean to Severity 5)'
                  : 'Empirical Classification Accuracy Across the 6 Custom Transformations'}
              </CardTitle>
              <CardDescription className="text-xs">
                {activeChart === 'severity'
                  ? 'Evaluating accuracy decay as perturbation magnitude intensifies from 1 to 5'
                  : 'Comparing ResNet-18 and MobileNetV3-Small resilience under specific environmental stressors'}
              </CardDescription>
            </div>
            {activeChart === 'corruption' && (
              <Badge variant="accent" className="self-start sm:self-auto text-[10px]">
                <Sparkles className="w-3 h-3 mr-1 text-amber-300" />
                JPEG Crossover Detected
              </Badge>
            )}
          </div>
        </CardHeader>

        <CardContent>
          <div className="h-72 w-full pt-2">
            {activeChart === 'severity' ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={severityCurves} margin={{ top: 10, right: 20, left: -10, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" opacity={0.7} />
                  <XAxis
                    dataKey="label"
                    stroke="#64748b"
                    fontSize={11}
                    tickLine={false}
                  />
                  <YAxis
                    stroke="#64748b"
                    fontSize={11}
                    tickFormatter={(val) => `${val}%`}
                    domain={[30, 100]}
                    tickLine={false}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#0f172a',
                      borderColor: '#1e293b',
                      borderRadius: '0.5rem',
                      color: '#f8fafc',
                      fontSize: '11px',
                    }}
                    formatter={(val: number) => [`${val.toFixed(2)}%`, '']}
                  />
                  <Legend
                    wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }}
                    iconType="circle"
                  />
                  <Line
                    type="monotone"
                    dataKey="resnet_fp32"
                    name="ResNet-18 (FP32)"
                    stroke="#38bdf8"
                    strokeWidth={2.5}
                    dot={{ r: 3, fill: '#38bdf8' }}
                  />
                  <Line
                    type="monotone"
                    dataKey="resnet_int8"
                    name="ResNet-18 (INT8)"
                    stroke="#0284c7"
                    strokeWidth={1.5}
                    strokeDasharray="4 4"
                    dot={{ r: 2.5, fill: '#0284c7' }}
                  />
                  <Line
                    type="monotone"
                    dataKey="mobilenet_fp32"
                    name="MobileNetV3-Small (FP32)"
                    stroke="#c084fc"
                    strokeWidth={2.5}
                    dot={{ r: 3, fill: '#c084fc' }}
                  />
                  <Line
                    type="monotone"
                    dataKey="mobilenet_int8"
                    name="MobileNetV3-Small (INT8)"
                    stroke="#9333ea"
                    strokeWidth={1.5}
                    strokeDasharray="4 4"
                    dot={{ r: 2.5, fill: '#9333ea' }}
                  />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={perCorruptionData} margin={{ top: 10, right: 20, left: -10, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" opacity={0.7} />
                  <XAxis
                    dataKey="name"
                    stroke="#64748b"
                    fontSize={10}
                    tickLine={false}
                    interval={0}
                    tickFormatter={(val) => val.split(' ')[0]}
                  />
                  <YAxis
                    stroke="#64748b"
                    fontSize={11}
                    tickFormatter={(val) => `${val}%`}
                    domain={[30, 100]}
                    tickLine={false}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#0f172a',
                      borderColor: '#1e293b',
                      borderRadius: '0.5rem',
                      color: '#f8fafc',
                      fontSize: '11px',
                    }}
                    formatter={(val: number, name: string) => [`${val.toFixed(2)}%`, name]}
                    labelFormatter={(label) => `Transformation: ${label}`}
                  />
                  <Legend
                    wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }}
                    iconType="rect"
                  />
                  <Bar
                    dataKey="resnet_fp32"
                    name="ResNet-18 (FP32)"
                    fill="#38bdf8"
                    radius={[4, 4, 0, 0]}
                  />
                  <Bar
                    dataKey="mobilenet_fp32"
                    name="MobileNetV3-Small (FP32)"
                    fill="#c084fc"
                    radius={[4, 4, 0, 0]}
                  />
                </BarChart>
              </ResponsiveContainer>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Per-Transformation Analysis Table */}
      <Card>
        <CardHeader className="py-3">
          <CardTitle className="text-xs font-semibold text-text-primary">
            Detailed Transformation Profiles & Inductive Robustness Delta
          </CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-surface-secondary/70 text-text-muted font-mono uppercase text-[10px] border-b border-surface-border">
                <tr>
                  <th className="py-2 px-3">Corruption Transformation</th>
                  <th className="py-2 px-3">Category</th>
                  <th className="py-2 px-3 text-right">ResNet-18 Acc.</th>
                  <th className="py-2 px-3 text-right">ResNet-18 Drop</th>
                  <th className="py-2 px-3 text-right">MobileNet Acc.</th>
                  <th className="py-2 px-3 text-right">MobileNet Drop</th>
                  <th className="py-2 px-3 text-right">Advantage</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-surface-border font-mono text-[11px]">
                {perCorruptionData.map((c) => (
                  <tr key={c.key} className="hover:bg-surface-hover/30 transition-colors">
                    <td className="py-2 px-3 font-semibold text-text-primary font-sans">
                      {c.name}
                    </td>
                    <td className="py-2 px-3">
                      <Badge variant="outline" className="text-[9px] py-0 px-1">
                        {c.category}
                      </Badge>
                    </td>
                    <td className="py-2 px-3 text-right text-text-primary">
                      {formatPercent(c.resnet_fp32)}
                    </td>
                    <td className="py-2 px-3 text-right text-rose-400">
                      {formatDrop(c.resnet_drop_pp)}
                    </td>
                    <td className="py-2 px-3 text-right text-text-primary">
                      {formatPercent(c.mobilenet_fp32)}
                    </td>
                    <td className="py-2 px-3 text-right text-rose-400">
                      {formatDrop(c.mobilenet_drop_pp)}
                    </td>
                    <td className="py-2 px-3 text-right">
                      {c.is_crossover ? (
                        <span className="text-purple-400 font-bold">
                          MobileNet +{Math.abs(c.resnet_advantage_pp).toFixed(2)} pp
                        </span>
                      ) : (
                        <span className="text-accent font-semibold">
                          ResNet +{c.resnet_advantage_pp.toFixed(2)} pp
                        </span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </section>
  );
}
