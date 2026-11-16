import { useState, useMemo } from 'react';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Select } from '@/components/ui/Select';
import { EmptyState } from '@/components/ui/EmptyState';
import {
  AlertTriangle,
  Info,
  SlidersHorizontal,
} from 'lucide-react';
import type { BenchmarkRecord } from '@/data/types';
import { formatMs, formatFps } from '@/data/formatters';

interface EfficiencyBenchmarkSectionProps {
  benchmarkRecords: BenchmarkRecord[];
}

export function EfficiencyBenchmarkSection({
  benchmarkRecords,
}: EfficiencyBenchmarkSectionProps) {
  const [modelFilter, setModelFilter] = useState<string>('all');
  const [precisionFilter, setPrecisionFilter] = useState<string>('all');
  const [deviceFilter, setDeviceFilter] = useState<string>('all');
  const [batchFilter, setBatchFilter] = useState<string>('all');
  const [metricView, setMetricView] = useState<'total' | 'forward'>('total');

  // Filter records
  const filteredRecords = useMemo(() => {
    return benchmarkRecords.filter((r) => {
      if (modelFilter !== 'all' && r.model !== modelFilter) return false;
      if (precisionFilter !== 'all' && r.precision !== precisionFilter) return false;
      if (deviceFilter !== 'all' && r.device !== deviceFilter) return false;
      if (batchFilter !== 'all' && r.batch_size !== Number(batchFilter)) return false;
      return true;
    });
  }, [benchmarkRecords, modelFilter, precisionFilter, deviceFilter, batchFilter]);

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
                Measured Hardware Inference Efficiency Benchmarks
              </CardTitle>
            </div>
            <CardDescription className="text-xs mt-1">
              Rigorous empirical latency and throughput measurements across devices (Apple M3 CPU vs. MPS), runtimes, batch sizes, and precisions.
            </CardDescription>
          </div>

          {/* Forward vs Total toggle */}
          <div className="flex items-center gap-1.5 self-start sm:self-auto bg-surface-secondary/70 p-1 rounded-lg border border-surface-border">
            <button
              onClick={() => setMetricView('total')}
              className={`px-2.5 py-1 text-xs font-mono rounded transition-colors ${
                metricView === 'total'
                  ? 'bg-accent/20 text-accent font-bold border border-accent/40'
                  : 'text-text-muted hover:text-text-primary'
              }`}
            >
              Total Engine Latency
            </button>
            <button
              onClick={() => setMetricView('forward')}
              className={`px-2.5 py-1 text-xs font-mono rounded transition-colors ${
                metricView === 'forward'
                  ? 'bg-accent/20 text-accent font-bold border border-accent/40'
                  : 'text-text-muted hover:text-text-primary'
              }`}
            >
              Forward-Only Pass
            </button>
          </div>
        </div>
      </CardHeader>

      <CardContent className="pt-4 space-y-4">
        {/* Multi-Dimensional Filter Controls */}
        <div className="p-3.5 rounded-xl bg-surface-secondary/40 border border-surface-border space-y-3">
          <div className="flex items-center gap-2 text-xs font-mono text-text-muted uppercase">
            <SlidersHorizontal className="w-3.5 h-3.5 text-accent" />
            <span>Filter Benchmark Measurements</span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
            <Select
              label="Model Architecture"
              value={modelFilter}
              onChange={(e) => setModelFilter(e.target.value)}
              options={[
                { value: 'all', label: 'All Architectures (2)' },
                { value: 'resnet18_cifar', label: 'ResNet-18' },
                { value: 'mobilenet_v3_small_cifar', label: 'MobileNetV3-Small' },
              ]}
            />
            <Select
              label="Precision Regime"
              value={precisionFilter}
              onChange={(e) => setPrecisionFilter(e.target.value)}
              options={[
                { value: 'all', label: 'All Precisions (FP32, FP16, INT8)' },
                { value: 'fp32', label: 'FP32' },
                { value: 'fp16', label: 'FP16' },
                { value: 'int8', label: 'INT8' },
              ]}
            />
            <Select
              label="Execution Device"
              value={deviceFilter}
              onChange={(e) => setDeviceFilter(e.target.value)}
              options={[
                { value: 'all', label: 'All Devices (CPU & MPS)' },
                { value: 'cpu', label: 'Host CPU (Apple M3)' },
                { value: 'mps', label: 'Apple Silicon MPS (Metal)' },
              ]}
            />
            <Select
              label="Batch Size"
              value={batchFilter}
              onChange={(e) => setBatchFilter(e.target.value)}
              options={[
                { value: 'all', label: 'All Batch Sizes (1, 8, 32, 64)' },
                { value: '1', label: 'Batch Size 1 (Single-image)' },
                { value: '8', label: 'Batch Size 8' },
                { value: '32', label: 'Batch Size 32' },
                { value: '64', label: 'Batch Size 64 (Peak Throughput)' },
              ]}
            />
          </div>
        </div>

        {/* Results Table or Empty State */}
        {filteredRecords.length === 0 ? (
          <div className="py-12">
            <EmptyState
              icon={<AlertTriangle className="w-8 h-8 text-amber-400" />}
              title="No Empirical Benchmark Matches Filter Criteria"
              description="The selected combination of architecture, precision, device, and batch size was not evaluated. In particular, INT8 quantization is executed strictly on CPU via QNNPACK and is not evaluated on MPS. Measurements are never fabricated or silently substituted."
              action={
                <button
                  onClick={() => {
                    setModelFilter('all');
                    setPrecisionFilter('all');
                    setDeviceFilter('all');
                    setBatchFilter('all');
                  }}
                  className="px-3 py-1.5 rounded-md text-xs font-mono bg-accent/20 border border-accent/40 text-accent hover:bg-accent/30 transition-colors"
                >
                  Reset Benchmark Filters
                </button>
              }
            />
          </div>
        ) : (
          <div className="overflow-x-auto rounded-lg border border-surface-border">
            <table className="w-full text-left text-xs">
              <thead className="bg-surface-secondary/70 text-text-muted font-mono uppercase text-[10px] border-b border-surface-border">
                <tr>
                  <th className="py-2.5 px-3">Model</th>
                  <th className="py-2.5 px-3">Precision</th>
                  <th className="py-2.5 px-3">Device</th>
                  <th className="py-2.5 px-3">Backend</th>
                  <th className="py-2.5 px-3 text-right">Batch</th>
                  <th className="py-2.5 px-3 text-right">
                    {metricView === 'total' ? 'Median Total (ms)' : 'Median Forward (ms)'}
                  </th>
                  <th className="py-2.5 px-3 text-right">Per-Image (ms)</th>
                  <th className="py-2.5 px-3 text-right">
                    {metricView === 'total' ? 'Throughput (img/s)' : 'Forward Throughput (img/s)'}
                  </th>
                  <th className="py-2.5 px-3">Benchmark Protocol & Execution Notes</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-surface-border font-mono text-[11px]">
                {filteredRecords.map((r, idx) => {
                  const latency = metricView === 'total' ? r.median_total_ms : r.median_forward_ms;
                  const throughput = metricView === 'total' ? r.throughput_fps : r.forward_only_throughput_fps;

                  return (
                    <tr
                      key={`${r.model}_${r.precision}_${r.device}_${r.batch_size}_${idx}`}
                      className={`hover:bg-surface-hover/30 ${
                        r.is_diagnostic ? 'bg-amber-500/[0.03]' : ''
                      }`}
                    >
                      <td className="py-2.5 px-3 font-semibold text-text-primary">
                        {r.model === 'resnet18_cifar' ? 'ResNet-18' : 'MobileNetV3-Small'}
                      </td>
                      <td className="py-2.5 px-3">
                        <Badge
                          variant={
                            r.precision === 'fp32'
                              ? 'default'
                              : r.precision === 'fp16'
                              ? 'purple'
                              : 'accent'
                          }
                          className="font-mono text-[10px] uppercase"
                        >
                          {r.precision}
                        </Badge>
                      </td>
                      <td className="py-2.5 px-3 font-semibold uppercase">
                        <span
                          className={
                            r.device === 'mps' ? 'text-purple-400' : 'text-text-primary'
                          }
                        >
                          {r.device}
                        </span>
                      </td>
                      <td className="py-2.5 px-3 text-text-secondary text-[11px]">
                        {r.backend}
                      </td>
                      <td className="py-2.5 px-3 text-right font-bold text-text-primary">
                        {r.batch_size}
                      </td>
                      <td className="py-2.5 px-3 text-right font-bold text-accent">
                        {formatMs(latency)}
                      </td>
                      <td className="py-2.5 px-3 text-right text-text-secondary">
                        {formatMs(r.per_image_ms)}
                      </td>
                      <td className="py-2.5 px-3 text-right font-bold text-emerald-400">
                        {formatFps(throughput)}
                      </td>
                      <td className="py-2.5 px-3 font-sans text-xs">
                        {r.is_diagnostic ? (
                          <span className="inline-flex items-center gap-1 text-amber-400 text-[11px]">
                            <AlertTriangle className="w-3 h-3 shrink-0" />
                            {r.note}
                          </span>
                        ) : (
                          <span className="text-text-muted text-[11px]">{r.note}</span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}

        {/* Technical Cavity and Benchmark Protocol Note */}
        <div className="p-3.5 rounded-lg bg-surface-secondary/40 border border-surface-border text-xs text-text-secondary space-y-2">
          <div className="flex items-center gap-2 text-text-primary font-semibold">
            <Info className="w-4 h-4 text-accent shrink-0" />
            <span>Benchmark Protocol Disclaimers & Comparability Guidelines:</span>
          </div>
          <ul className="space-y-1.5 text-[11px] list-disc list-inside text-text-secondary leading-relaxed">
            <li>
              <strong>Total Engine vs. Forward-Only Latency:</strong> Total latency accounts for full batch tensor preparation, host-to-device memory allocation, forward propagation, and synchronization. Forward-only latency measures isolated graph execution.
            </li>
            <li>
              <strong>CPU FP16 Diagnostic Measurements:</strong> Rows marked with an alert flag represent diagnostic measurements on CPU. They are not treated as representative of the intended accelerated FP16 deployment path. On this hardware platform, half-precision execution is intended for Apple Silicon MPS (Metal Performance Shaders).
            </li>
            <li>
              <strong>Cross-Device Non-Equivalence:</strong> Latency metrics on CPU (Apple M3 4-thread default) and MPS (Metal unified memory) reflect distinct silicon architectures and dispatch queues. They are benchmarked under identical warm-up and repetition protocols, but represent heterogeneous hardware execution paths.
            </li>
          </ul>
        </div>
      </CardContent>
    </Card>
  );
}
