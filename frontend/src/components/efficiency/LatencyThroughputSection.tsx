import { useState, useMemo } from 'react';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import {
  LineChart,
  Line,
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
import type { BenchmarkRecord } from '@/data/types';
import { formatMs } from '@/data/formatters';
import type { EfficiencyFilterState } from './EfficiencyControls';

interface LatencyThroughputSectionProps {
  filters: EfficiencyFilterState;
  benchmarkRecords: BenchmarkRecord[];
}

type SubViewMode = 'batch_scaling' | 'timing_decomposition' | 'device_comparison';

export function LatencyThroughputSection({
  filters,
  benchmarkRecords,
}: LatencyThroughputSectionProps) {
  const [subView, setSubView] = useState<SubViewMode>('batch_scaling');
  const [metricType, setMetricType] = useState<'latency' | 'throughput'>('throughput');

  // Filter records based on filters
  const activeRecords = useMemo(() => {
    return benchmarkRecords.filter((r) => {
      if (filters.model !== 'all' && r.model !== filters.model) return false;
      if (filters.precision !== 'all' && r.precision !== filters.precision) return false;
      if (filters.device !== 'all' && r.device !== filters.device) return false;
      if (filters.batch_size !== 'all' && r.batch_size !== filters.batch_size) return false;
      if (!filters.include_diagnostic && r.is_diagnostic) return false;
      return true;
    });
  }, [filters, benchmarkRecords]);

  // Batch Scaling Dataset (B=1, 8, 32, 64)
  // Group configurations by (model, precision, device) and map each batch size
  const batchScalingData = useMemo(() => {
    const batchSizes = [1, 8, 32, 64];

    return batchSizes.map((b) => {
      const row: Record<string, any> = { batch_size: `B=${b}`, b };

      // Helper to extract metric for a given curve
      const addCurve = (key: string, model: string, prec: string, dev: string) => {
        const match = benchmarkRecords.find(
          (r) =>
            r.model === model &&
            r.precision === prec &&
            r.device === dev &&
            r.batch_size === b &&
            (!filters.include_diagnostic ? !r.is_diagnostic : true)
        );
        if (match) {
          if (metricType === 'latency') {
            row[key] =
              filters.timing_mode === 'forward' ? match.median_forward_ms : match.median_total_ms;
            row[`${key}_per_img`] = Number((row[key] / b).toFixed(3));
          } else {
            row[key] =
              filters.timing_mode === 'forward'
                ? match.forward_only_throughput_fps
                : match.throughput_fps;
          }
        }
      };

      // ResNet Curves
      addCurve('resnet_fp32_cpu', 'resnet18_cifar', 'fp32', 'cpu');
      addCurve('resnet_fp32_mps', 'resnet18_cifar', 'fp32', 'mps');
      addCurve('resnet_fp16_mps', 'resnet18_cifar', 'fp16', 'mps');
      addCurve('resnet_int8_cpu', 'resnet18_cifar', 'int8', 'cpu');

      // MobileNet Curves
      addCurve('mobilenet_fp32_cpu', 'mobilenet_v3_small_cifar', 'fp32', 'cpu');
      addCurve('mobilenet_fp32_mps', 'mobilenet_v3_small_cifar', 'fp32', 'mps');
      addCurve('mobilenet_fp16_mps', 'mobilenet_v3_small_cifar', 'fp16', 'mps');
      addCurve('mobilenet_int8_cpu', 'mobilenet_v3_small_cifar', 'int8', 'cpu');

      return row;
    });
  }, [benchmarkRecords, metricType, filters.timing_mode, filters.include_diagnostic]);

  // Decomposition Dataset: Total vs Forward for active records
  const decompositionData = useMemo(() => {
    // Pick unique deployment configurations at active batch size (or B=64 if all)
    const targetBatch = filters.batch_size === 'all' ? 64 : filters.batch_size;
    const recordsAtBatch = benchmarkRecords.filter(
      (r) =>
        r.batch_size === targetBatch &&
        (filters.model !== 'all' ? r.model === filters.model : true) &&
        (filters.precision !== 'all' ? r.precision === filters.precision : true) &&
        (filters.device !== 'all' ? r.device === filters.device : true) &&
        (!filters.include_diagnostic ? !r.is_diagnostic : true)
    );

    return recordsAtBatch.map((r) => {
      const archName = r.model === 'resnet18_cifar' ? 'ResNet' : 'MobileNet';
      const label = `${archName} ${r.precision.toUpperCase()} (${r.device.toUpperCase()})`;
      const overhead = Number(Math.max(0, r.median_total_ms - r.median_forward_ms).toFixed(3));

      return {
        label,
        forward_ms: Number(r.median_forward_ms.toFixed(3)),
        overhead_ms: overhead,
        total_ms: Number(r.median_total_ms.toFixed(3)),
        throughput_fps: r.throughput_fps,
        batch_size: r.batch_size,
      };
    });
  }, [benchmarkRecords, filters]);

  // CPU vs MPS Comparative Records
  const cpuVsMpsPairs = useMemo(() => {
    // Find configurations that have both CPU and MPS records (e.g. FP32 batch 1..64)
    const pairs: Array<{
      model: string;
      precision: string;
      batch_size: number;
      cpu_lat: number;
      mps_lat: number;
      cpu_fps: number;
      mps_fps: number;
      speedup: number;
    }> = [];

    const cpuRecords = benchmarkRecords.filter(
      (r) => r.device === 'cpu' && (!filters.include_diagnostic ? !r.is_diagnostic : true)
    );
    for (const c of cpuRecords) {
      const mps = benchmarkRecords.find(
        (m) =>
          m.device === 'mps' &&
          m.model === c.model &&
          m.precision === c.precision &&
          m.batch_size === c.batch_size
      );
      if (mps) {
        const cpuVal = filters.timing_mode === 'forward' ? c.median_forward_ms : c.median_total_ms;
        const mpsVal = filters.timing_mode === 'forward' ? mps.median_forward_ms : mps.median_total_ms;
        const speedup = cpuVal / mpsVal;

        pairs.push({
          model: c.model === 'resnet18_cifar' ? 'ResNet-18' : 'MobileNetV3-Small',
          precision: c.precision.toUpperCase(),
          batch_size: c.batch_size,
          cpu_lat: cpuVal,
          mps_lat: mpsVal,
          cpu_fps: filters.timing_mode === 'forward' ? c.forward_only_throughput_fps : c.throughput_fps,
          mps_fps: filters.timing_mode === 'forward' ? mps.forward_only_throughput_fps : mps.throughput_fps,
          speedup: Number(speedup.toFixed(2)),
        });
      }
    }
    return pairs;
  }, [benchmarkRecords, filters.timing_mode, filters.include_diagnostic]);

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
                Inference Latency & Throughput Scaling Analysis
              </CardTitle>
            </div>
            <CardDescription className="text-xs mt-1">
              Multi-dimensional scaling across batch regimes ($B=1, 8, 32, 64$), runtime backends, and Apple Silicon execution pipelines.
            </CardDescription>
          </div>

          {/* Sub-view Navigation Tabs */}
          <div className="flex flex-wrap items-center gap-1.5 self-start sm:self-auto bg-surface-secondary/70 p-1 rounded-lg border border-surface-border">
            <button
              type="button"
              onClick={() => setSubView('batch_scaling')}
              className={`px-2.5 py-1 text-xs font-mono rounded transition-colors ${
                subView === 'batch_scaling'
                  ? 'bg-accent/20 text-accent font-bold border border-accent/40'
                  : 'text-text-muted hover:text-text-primary'
              }`}
            >
              Batch Scaling Curves
            </button>
            <button
              type="button"
              onClick={() => setSubView('timing_decomposition')}
              className={`px-2.5 py-1 text-xs font-mono rounded transition-colors ${
                subView === 'timing_decomposition'
                  ? 'bg-accent/20 text-accent font-bold border border-accent/40'
                  : 'text-text-muted hover:text-text-primary'
              }`}
            >
              Pipeline Decomposition
            </button>
            <button
              type="button"
              onClick={() => setSubView('device_comparison')}
              className={`px-2.5 py-1 text-xs font-mono rounded transition-colors ${
                subView === 'device_comparison'
                  ? 'bg-accent/20 text-accent font-bold border border-accent/40'
                  : 'text-text-muted hover:text-text-primary'
              }`}
            >
              CPU vs. MPS Acceleration
            </button>
          </div>
        </div>
      </CardHeader>

      <CardContent className="pt-4 space-y-5">
        {/* Sub-View 1: Batch Scaling Curves */}
        {subView === 'batch_scaling' && (
          <div className="space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <div className="flex items-center gap-2">
                <span className="text-xs font-mono text-text-muted">Metric Focus:</span>
                <div className="flex items-center rounded-lg bg-surface-secondary/70 p-1 border border-surface-border">
                  <button
                    type="button"
                    onClick={() => setMetricType('throughput')}
                    className={`px-3 py-1 text-xs font-mono rounded transition-colors ${
                      metricType === 'throughput'
                        ? 'bg-emerald-500/20 text-emerald-400 font-bold border border-emerald-500/40'
                        : 'text-text-muted hover:text-text-primary'
                    }`}
                  >
                    Throughput (img/s)
                  </button>
                  <button
                    type="button"
                    onClick={() => setMetricType('latency')}
                    className={`px-3 py-1 text-xs font-mono rounded transition-colors ${
                      metricType === 'latency'
                        ? 'bg-accent/20 text-accent font-bold border border-accent/40'
                        : 'text-text-muted hover:text-text-primary'
                    }`}
                  >
                    Latency (ms / batch)
                  </button>
                </div>
              </div>

              <div className="text-[11px] font-mono text-text-muted">
                {filters.timing_mode === 'forward' ? 'Evaluating Forward-Only Graph Execution' : 'Evaluating Total-Engine Latency Pipeline'}
              </div>
            </div>

            {/* Recharts Line Chart */}
            <div className="h-80 w-full bg-surface-secondary/20 p-3 rounded-xl border border-surface-border">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={batchScalingData} margin={{ top: 15, right: 25, left: 10, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#262626" />
                  <XAxis dataKey="batch_size" stroke="#737373" tick={{ fill: '#a3a3a3', fontSize: 11, fontFamily: 'monospace' }} />
                  <YAxis
                    stroke="#737373"
                    unit={metricType === 'throughput' ? ' fps' : ' ms'}
                    tick={{ fill: '#a3a3a3', fontSize: 11, fontFamily: 'monospace' }}
                  />
                  <RechartsTooltip
                    content={({ active, payload, label }) => {
                      if (active && payload && payload.length) {
                        return (
                          <div className="bg-surface border border-surface-border p-3 rounded shadow-xl text-xs font-mono space-y-1.5">
                            <span className="font-bold text-text-primary block">{label} Workload</span>
                            <div className="divide-y divide-surface-border/50 space-y-1 pt-1">
                              {payload.map((p: any) => (
                                <div key={p.dataKey} className="flex items-center justify-between gap-4 pt-1 text-[11px]">
                                  <span style={{ color: p.color }}>{p.name}:</span>
                                  <strong className="text-text-primary">
                                    {metricType === 'throughput'
                                      ? `${p.value.toLocaleString('en-US', { maximumFractionDigits: 1 })} img/s`
                                      : `${formatMs(p.value)}`}
                                  </strong>
                                </div>
                              ))}
                            </div>
                          </div>
                        );
                      }
                      return null;
                    }}
                  />
                  <Legend wrapperStyle={{ fontSize: '11px', fontFamily: 'monospace', paddingTop: '10px' }} />

                  {/* ResNet Lines */}
                  {(filters.model === 'all' || filters.model === 'resnet18_cifar') && (
                    <>
                      <Line
                        type="monotone"
                        dataKey="resnet_fp32_cpu"
                        name="ResNet FP32 (CPU)"
                        stroke="#f59e0b"
                        strokeWidth={2}
                        dot={{ r: 3 }}
                      />
                      <Line
                        type="monotone"
                        dataKey="resnet_fp32_mps"
                        name="ResNet FP32 (MPS)"
                        stroke="#3b82f6"
                        strokeWidth={2}
                        dot={{ r: 3 }}
                      />
                      <Line
                        type="monotone"
                        dataKey="resnet_fp16_mps"
                        name="ResNet FP16 (MPS)"
                        stroke="#8b5cf6"
                        strokeWidth={2}
                        strokeDasharray="4 2"
                        dot={{ r: 3 }}
                      />
                      <Line
                        type="monotone"
                        dataKey="resnet_int8_cpu"
                        name="ResNet INT8 (CPU)"
                        stroke="#10b981"
                        strokeWidth={2}
                        dot={{ r: 3 }}
                      />
                    </>
                  )}

                  {/* MobileNet Lines */}
                  {(filters.model === 'all' || filters.model === 'mobilenet_v3_small_cifar') && (
                    <>
                      <Line
                        type="monotone"
                        dataKey="mobilenet_fp32_mps"
                        name="MobileNet FP32 (MPS)"
                        stroke="#06b6d4"
                        strokeWidth={2.5}
                        dot={{ r: 4 }}
                      />
                      <Line
                        type="monotone"
                        dataKey="mobilenet_fp16_mps"
                        name="MobileNet FP16 (MPS)"
                        stroke="#a855f7"
                        strokeWidth={2.5}
                        strokeDasharray="4 2"
                        dot={{ r: 4 }}
                      />
                      <Line
                        type="monotone"
                        dataKey="mobilenet_int8_cpu"
                        name="MobileNet INT8 (CPU)"
                        stroke="#22c55e"
                        strokeWidth={2.5}
                        dot={{ r: 4 }}
                      />
                    </>
                  )}
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {/* Sub-View 2: Pipeline Timing Decomposition */}
        {subView === 'timing_decomposition' && (
          <div className="space-y-4">
            <div className="text-xs text-text-secondary leading-relaxed font-sans">
              Comparing isolated forward propagation execution (<code>median_forward_ms</code>) against complete host-to-device tensor allocation, batch preparation, and synchronization overhead (<code>median_total_ms</code>).
            </div>

            <div className="h-80 w-full bg-surface-secondary/20 p-3 rounded-xl border border-surface-border">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={decompositionData}
                  margin={{ top: 15, right: 25, left: 10, bottom: 35 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#262626" />
                  <XAxis
                    dataKey="label"
                    stroke="#737373"
                    tick={{ fill: '#a3a3a3', fontSize: 10, fontFamily: 'monospace' }}
                    angle={-15}
                    textAnchor="end"
                  />
                  <YAxis
                    stroke="#737373"
                    unit=" ms"
                    tick={{ fill: '#a3a3a3', fontSize: 11, fontFamily: 'monospace' }}
                  />
                  <RechartsTooltip
                    content={({ active, payload }) => {
                      if (active && payload && payload.length) {
                        const data = payload[0].payload;
                        return (
                          <div className="bg-surface border border-surface-border p-3 rounded shadow-xl text-xs font-mono space-y-1">
                            <span className="font-bold text-text-primary block">{data.label} (B={data.batch_size})</span>
                            <div className="pt-1 space-y-0.5 text-[11px]">
                              <span className="text-accent block">
                                Forward Execution: <strong>{data.forward_ms} ms</strong>
                              </span>
                              <span className="text-amber-400 block">
                                Pipeline Overhead: <strong>{data.overhead_ms} ms</strong>
                              </span>
                              <span className="text-emerald-400 block pt-1 border-t border-surface-border">
                                Total Latency: <strong>{data.total_ms} ms</strong>
                              </span>
                              <span className="text-text-muted block">
                                Throughput: <strong>{data.throughput_fps} img/s</strong>
                              </span>
                            </div>
                          </div>
                        );
                      }
                      return null;
                    }}
                  />
                  <Legend wrapperStyle={{ fontSize: '11px', fontFamily: 'monospace', paddingTop: '15px' }} />
                  <Bar
                    dataKey="forward_ms"
                    name="Forward Graph Execution (ms)"
                    fill="#0284c7"
                    stackId="a"
                    radius={[0, 0, 0, 0]}
                  />
                  <Bar
                    dataKey="overhead_ms"
                    name="Host-Device Allocation & Sync Overhead (ms)"
                    fill="#f59e0b"
                    stackId="a"
                    radius={[4, 4, 0, 0]}
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {/* Sub-View 3: CPU vs MPS Device Comparison */}
        {subView === 'device_comparison' && (
          <div className="space-y-4">
            <div className="overflow-x-auto rounded-lg border border-surface-border font-mono text-xs">
              <table className="w-full text-left">
                <thead className="bg-surface-secondary/70 text-text-muted uppercase text-[10px] border-b border-surface-border">
                  <tr>
                    <th className="py-2.5 px-3">Architecture</th>
                    <th className="py-2.5 px-3">Precision</th>
                    <th className="py-2.5 px-3">Batch Size</th>
                    <th className="py-2.5 px-3 text-right">CPU Latency</th>
                    <th className="py-2.5 px-3 text-right">MPS Latency</th>
                    <th className="py-2.5 px-3 text-right">CPU Throughput</th>
                    <th className="py-2.5 px-3 text-right">MPS Throughput</th>
                    <th className="py-2.5 px-3 text-right">MPS Speedup</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-surface-border text-[11px]">
                  {cpuVsMpsPairs.map((p, idx) => (
                    <tr key={`${p.model}-${p.precision}-${p.batch_size}-${idx}`} className="hover:bg-surface-hover/30">
                      <td className="py-2 px-3 font-semibold text-text-primary">{p.model}</td>
                      <td className="py-2 px-3 text-text-muted">{p.precision}</td>
                      <td className="py-2 px-3">B={p.batch_size}</td>
                      <td className="py-2 px-3 text-right text-text-secondary">{formatMs(p.cpu_lat)}</td>
                      <td className="py-2 px-3 text-right text-accent font-bold">{formatMs(p.mps_lat)}</td>
                      <td className="py-2 px-3 text-right text-text-muted">{p.cpu_fps.toFixed(1)} img/s</td>
                      <td className="py-2 px-3 text-right text-emerald-400 font-bold">{p.mps_fps.toFixed(1)} img/s</td>
                      <td className="py-2 px-3 text-right font-bold text-accent">
                        {p.speedup >= 1 ? `${p.speedup.toFixed(1)}× faster` : `${p.speedup.toFixed(2)}×`}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Detailed Benchmark Records Table */}
        <div className="space-y-2 pt-2">
          <div className="flex items-center justify-between text-xs font-mono text-text-secondary">
            <span className="font-bold text-text-primary">Active Empirical Records ({activeRecords.length}):</span>
            <span className="text-[11px] text-text-muted">Measurements conducted on Apple M3 (macOS 27.0, PyTorch 2.15.0)</span>
          </div>

          <div className="overflow-x-auto rounded-lg border border-surface-border font-mono text-xs max-h-64 overflow-y-auto">
            <table className="w-full text-left">
              <thead className="bg-surface-secondary/70 text-text-muted uppercase text-[10px] sticky top-0 border-b border-surface-border">
                <tr>
                  <th className="py-2 px-3">Model</th>
                  <th className="py-2 px-3">Precision</th>
                  <th className="py-2 px-3">Device</th>
                  <th className="py-2 px-3">Backend</th>
                  <th className="py-2 px-3 text-right">Batch</th>
                  <th className="py-2 px-3 text-right">Total Latency</th>
                  <th className="py-2 px-3 text-right">Forward Latency</th>
                  <th className="py-2 px-3 text-right">Per-Image</th>
                  <th className="py-2 px-3 text-right">Throughput</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-surface-border text-[11px]">
                {activeRecords.map((r, idx) => (
                  <tr key={`${r.model}-${r.precision}-${r.device}-${r.batch_size}-${idx}`} className="hover:bg-surface-hover/30">
                    <td className="py-2 px-3 font-semibold text-text-primary">
                      {r.model === 'resnet18_cifar' ? 'ResNet-18' : 'MobileNetV3-Small'}
                    </td>
                    <td className="py-2 px-3">
                      <Badge
                        variant={r.precision === 'int8' ? 'accent' : r.precision === 'fp16' ? 'purple' : 'default'}
                        className="text-[9px]"
                      >
                        {r.precision.toUpperCase()}
                      </Badge>
                    </td>
                    <td className="py-2 px-3 text-text-secondary uppercase">{r.device}</td>
                    <td className="py-2 px-3 text-text-muted text-[10px]">{r.backend}</td>
                    <td className="py-2 px-3 text-right">B={r.batch_size}</td>
                    <td className="py-2 px-3 text-right text-accent font-bold">{formatMs(r.median_total_ms)}</td>
                    <td className="py-2 px-3 text-right text-text-primary">{formatMs(r.median_forward_ms)}</td>
                    <td className="py-2 px-3 text-right text-text-muted">{formatMs(r.per_image_ms)}</td>
                    <td className="py-2 px-3 text-right text-emerald-400 font-bold">{r.throughput_fps.toFixed(1)} img/s</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Analytical Note */}
        <div className="p-3.5 rounded-lg bg-surface-secondary/40 border border-surface-border text-xs text-text-secondary space-y-1.5 font-sans">
          <div className="flex items-center gap-2 text-text-primary font-semibold">
            <Info className="w-4 h-4 text-accent shrink-0" />
            <span>Batch Scaling Dynamics & Pipeline Overhead:</span>
          </div>
          <p className="text-[11px] leading-relaxed text-text-secondary">
            Across batch sizes $B=1$ through $B=64$, throughput scales non-linearly. Batch 1 minimizes trigger-to-prediction latency for real-time edge processing (e.g., 0.58 ms for MobileNetV3-Small INT8), while Batch 64 saturates hardware compute units and amortizes pipeline dispatch overhead, delivering up to 7,754.8 img/s on CPU and 5,986.3 img/s on Metal MPS.
          </p>
        </div>
      </CardContent>
    </Card>
  );
}
