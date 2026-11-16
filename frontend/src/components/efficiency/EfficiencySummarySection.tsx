import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import {
  Zap,
  Activity,
  HardDrive,
  Layers,
  ShieldCheck,
  TrendingDown,
  Clock,
} from 'lucide-react';
import type { BenchmarkRecord, PrecisionVariant } from '@/data/types';
import { formatMs, formatPercent } from '@/data/formatters';
import type { EfficiencyFilterState } from './EfficiencyControls';

interface EfficiencySummarySectionProps {
  filters: EfficiencyFilterState;
  benchmarkRecords: BenchmarkRecord[];
  precisionVariants: PrecisionVariant[];
}

export function EfficiencySummarySection({
  filters,
  benchmarkRecords,
  precisionVariants,
}: EfficiencySummarySectionProps) {
  // Filter records based on active filters
  const activeRecords = benchmarkRecords.filter((r) => {
    if (filters.model !== 'all' && r.model !== filters.model) return false;
    if (filters.precision !== 'all' && r.precision !== filters.precision) return false;
    if (filters.device !== 'all' && r.device !== filters.device) return false;
    if (filters.batch_size !== 'all' && r.batch_size !== filters.batch_size) return false;
    if (!filters.include_diagnostic && r.is_diagnostic) return false;
    return true;
  });

  // Pick representative or aggregated metrics
  const recordCount = activeRecords.length;

  // Selected or representative precision variant
  const activeVariant =
    filters.model !== 'all' && filters.precision !== 'all'
      ? precisionVariants.find(
          (v) => v.model_id === filters.model && v.precision_key === filters.precision
        )
      : null;

  // Compute summary stats across active records
  const minLatency =
    recordCount > 0
      ? Math.min(
          ...activeRecords.map((r) =>
            filters.timing_mode === 'forward' ? r.median_forward_ms : r.median_total_ms
          )
        )
      : 0;

  const maxThroughput =
    recordCount > 0
      ? Math.max(
          ...activeRecords.map((r) =>
            filters.timing_mode === 'forward'
              ? r.forward_only_throughput_fps
              : r.throughput_fps
          )
        )
      : 0;

  // Best latency record
  const bestLatencyRecord = activeRecords.find((r) =>
    (filters.timing_mode === 'forward' ? r.median_forward_ms : r.median_total_ms) === minLatency
  );

  // Best throughput record
  const bestThroughputRecord = activeRecords.find((r) =>
    (filters.timing_mode === 'forward'
      ? r.forward_only_throughput_fps
      : r.throughput_fps) === maxThroughput
  );

  // Active precision variant or pool extremes
  const minSizeVariant = [...precisionVariants]
    .filter((v) => (filters.model !== 'all' ? v.model_id === filters.model : true))
    .filter((v) => (filters.precision !== 'all' ? v.precision_key === filters.precision : true))
    .sort((a, b) => a.serialized_bytes - b.serialized_bytes)[0];

  const maxCleanAccVariant = [...precisionVariants]
    .filter((v) => (filters.model !== 'all' ? v.model_id === filters.model : true))
    .filter((v) => (filters.precision !== 'all' ? v.precision_key === filters.precision : true))
    .sort((a, b) => b.clean_accuracy - a.clean_accuracy)[0];

  const maxCorruptedAccVariant = [...precisionVariants]
    .filter((v) => (filters.model !== 'all' ? v.model_id === filters.model : true))
    .filter((v) => (filters.precision !== 'all' ? v.precision_key === filters.precision : true))
    .sort((a, b) => b.mean_corrupted_accuracy - a.mean_corrupted_accuracy)[0];

  return (
    <Card className="border-surface-border bg-surface/90">
      <CardHeader className="pb-3 border-b border-surface-border">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <div className="flex items-center gap-2">
              <Badge variant="accent" className="font-mono text-[10px]">
                Section B
              </Badge>
              <CardTitle className="text-base font-bold text-text-primary">
                Deployment Efficiency & Performance Telemetry Overview
              </CardTitle>
            </div>
            <CardDescription className="text-xs mt-1">
              Key operational benchmarks contrasting latency, throughput, serialized model footprint, and accuracy retention across the active evaluation pool.
            </CardDescription>
          </div>

          <div className="flex items-center gap-2 font-mono text-xs">
            <Badge variant="outline" className="text-text-muted">
              {recordCount} Active Measurement{recordCount === 1 ? '' : 's'}
            </Badge>
            <Badge variant="purple" className="text-purple-400">
              Apple Silicon M3 Unified Memory
            </Badge>
          </div>
        </div>
      </CardHeader>

      <CardContent className="pt-4 space-y-4">
        {/* 6 Metric Telemetry Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-3 font-mono">
          {/* 1. Median Latency */}
          <div className="p-3.5 rounded-xl bg-surface-secondary/50 border border-surface-border space-y-1">
            <div className="flex items-center justify-between">
              <span className="text-[10px] uppercase text-text-muted">
                {filters.timing_mode === 'forward' ? 'Forward Latency' : 'Total Latency'}
              </span>
              <Clock className="w-3.5 h-3.5 text-accent" />
            </div>
            <div className="text-xl font-bold text-accent">
              {minLatency > 0 ? formatMs(minLatency) : 'N/A'}
            </div>
            <div className="text-[10px] font-sans text-text-muted flex items-center justify-between pt-0.5">
              <span>{bestLatencyRecord ? `B=${bestLatencyRecord.batch_size} (${bestLatencyRecord.device.toUpperCase()})` : 'Active slice'}</span>
              <Badge variant="outline" className="text-[9px] py-0 px-1 border-accent/40 text-accent">
                Measured
              </Badge>
            </div>
          </div>

          {/* 2. Peak Throughput */}
          <div className="p-3.5 rounded-xl bg-surface-secondary/50 border border-surface-border space-y-1">
            <div className="flex items-center justify-between">
              <span className="text-[10px] uppercase text-text-muted">Max Throughput</span>
              <Activity className="w-3.5 h-3.5 text-emerald-400" />
            </div>
            <div className="text-xl font-bold text-emerald-400">
              {maxThroughput > 0 ? maxThroughput.toLocaleString('en-US', { maximumFractionDigits: 1 }) : 'N/A'}{' '}
              <span className="text-xs font-normal text-text-muted">img/s</span>
            </div>
            <div className="text-[10px] font-sans text-text-muted flex items-center justify-between pt-0.5">
              <span>{bestThroughputRecord ? `B=${bestThroughputRecord.batch_size} (${bestThroughputRecord.device.toUpperCase()})` : 'Active slice'}</span>
              <Badge variant="outline" className="text-[9px] py-0 px-1 border-emerald-500/40 text-emerald-400">
                Derived
              </Badge>
            </div>
          </div>

          {/* 3. Serialized Storage Size */}
          <div className="p-3.5 rounded-xl bg-surface-secondary/50 border border-surface-border space-y-1">
            <div className="flex items-center justify-between">
              <span className="text-[10px] uppercase text-text-muted">Serialized Size</span>
              <HardDrive className="w-3.5 h-3.5 text-purple-400" />
            </div>
            <div className="text-xl font-bold text-purple-400">
              {activeVariant ? `${activeVariant.serialized_mb.toFixed(2)} MB` : minSizeVariant ? `${minSizeVariant.serialized_mb.toFixed(2)} MB` : 'N/A'}
            </div>
            <div className="text-[10px] font-sans text-text-muted flex items-center justify-between pt-0.5">
              <span>{activeVariant ? `${activeVariant.display_name}` : minSizeVariant ? `Min: ${minSizeVariant.display_name}` : 'Exact bytes'}</span>
              <Badge variant="outline" className="text-[9px] py-0 px-1 border-purple-500/40 text-purple-400">
                Measured
              </Badge>
            </div>
          </div>

          {/* 4. Compression Ratio */}
          <div className="p-3.5 rounded-xl bg-surface-secondary/50 border border-surface-border space-y-1">
            <div className="flex items-center justify-between">
              <span className="text-[10px] uppercase text-text-muted">Compression</span>
              <Layers className="w-3.5 h-3.5 text-amber-400" />
            </div>
            <div className="text-xl font-bold text-amber-400">
              {activeVariant ? `${activeVariant.compression_ratio.toFixed(2)}×` : minSizeVariant ? `${minSizeVariant.compression_ratio.toFixed(2)}×` : '1.00×'}
            </div>
            <div className="text-[10px] font-sans text-text-muted flex items-center justify-between pt-0.5">
              <span>{activeVariant ? `${activeVariant.size_reduction_pct.toFixed(1)}% drop` : minSizeVariant ? `${minSizeVariant.size_reduction_pct.toFixed(1)}% max drop` : 'vs FP32 base'}</span>
              <Badge variant="outline" className="text-[9px] py-0 px-1 border-amber-500/40 text-amber-400">
                Derived
              </Badge>
            </div>
          </div>

          {/* 5. Clean Accuracy */}
          <div className="p-3.5 rounded-xl bg-surface-secondary/50 border border-surface-border space-y-1">
            <div className="flex items-center justify-between">
              <span className="text-[10px] uppercase text-text-muted">Clean Accuracy</span>
              <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
            </div>
            <div className="text-xl font-bold text-emerald-400">
              {activeVariant ? formatPercent(activeVariant.clean_accuracy) : maxCleanAccVariant ? formatPercent(maxCleanAccVariant.clean_accuracy) : 'N/A'}
            </div>
            <div className="text-[10px] font-sans text-text-muted flex items-center justify-between pt-0.5">
              <span>Held-out CIFAR-10</span>
              <Badge variant="outline" className="text-[9px] py-0 px-1 border-emerald-500/40 text-emerald-400">
                Measured
              </Badge>
            </div>
          </div>

          {/* 6. Corrupted Accuracy */}
          <div className="p-3.5 rounded-xl bg-surface-secondary/50 border border-surface-border space-y-1">
            <div className="flex items-center justify-between">
              <span className="text-[10px] uppercase text-text-muted">Corrupted Acc</span>
              <TrendingDown className="w-3.5 h-3.5 text-rose-400" />
            </div>
            <div className="text-xl font-bold text-rose-400">
              {activeVariant ? formatPercent(activeVariant.mean_corrupted_accuracy) : maxCorruptedAccVariant ? formatPercent(maxCorruptedAccVariant.mean_corrupted_accuracy) : 'N/A'}
            </div>
            <div className="text-[10px] font-sans text-text-muted flex items-center justify-between pt-0.5">
              <span>Mean 30 Conditions</span>
              <Badge variant="outline" className="text-[9px] py-0 px-1 border-rose-500/40 text-rose-400">
                Measured
              </Badge>
            </div>
          </div>
        </div>

        {/* Analytical Highlight Box */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 font-mono text-xs">
          {/* Lowest Latency Highlight */}
          <div className="p-3.5 rounded-xl bg-accent/5 border border-accent/20 space-y-1.5">
            <div className="flex items-center justify-between pb-1 border-b border-accent/20">
              <span className="font-bold text-text-primary flex items-center gap-1.5">
                <Zap className="w-3.5 h-3.5 text-accent" />
                Minimum Measured Latency
              </span>
              <Badge variant="outline" className="text-accent border-accent/30 text-[9px]">
                {bestLatencyRecord ? `B=${bestLatencyRecord.batch_size}` : 'Fastest'}
              </Badge>
            </div>
            {bestLatencyRecord ? (
              <div className="space-y-1 pt-0.5">
                <div className="flex items-baseline justify-between">
                  <span className="text-text-primary font-bold">
                    {bestLatencyRecord.model === 'resnet18_cifar' ? 'ResNet-18' : 'MobileNetV3-Small'} [{bestLatencyRecord.precision.toUpperCase()}]
                  </span>
                  <span className="text-accent font-bold">
                    {formatMs(filters.timing_mode === 'forward' ? bestLatencyRecord.median_forward_ms : bestLatencyRecord.median_total_ms)}
                  </span>
                </div>
                <p className="text-[11px] font-sans text-text-secondary leading-snug">
                  Evaluated on {bestLatencyRecord.device.toUpperCase()} ({bestLatencyRecord.backend}) at batch size {bestLatencyRecord.batch_size}.
                </p>
              </div>
            ) : (
              <p className="text-[11px] font-sans text-text-muted">No matching record in current view.</p>
            )}
          </div>

          {/* Highest Throughput Highlight */}
          <div className="p-3.5 rounded-xl bg-emerald-500/5 border border-emerald-500/20 space-y-1.5">
            <div className="flex items-center justify-between pb-1 border-b border-emerald-500/20">
              <span className="font-bold text-text-primary flex items-center gap-1.5">
                <Activity className="w-3.5 h-3.5 text-emerald-400" />
                Maximum Measured Throughput
              </span>
              <Badge variant="outline" className="text-emerald-400 border-emerald-500/30 text-[9px]">
                {bestThroughputRecord ? `B=${bestThroughputRecord.batch_size}` : 'Peak FPS'}
              </Badge>
            </div>
            {bestThroughputRecord ? (
              <div className="space-y-1 pt-0.5">
                <div className="flex items-baseline justify-between">
                  <span className="text-text-primary font-bold">
                    {bestThroughputRecord.model === 'resnet18_cifar' ? 'ResNet-18' : 'MobileNetV3-Small'} [{bestThroughputRecord.precision.toUpperCase()}]
                  </span>
                  <span className="text-emerald-400 font-bold">
                    {(filters.timing_mode === 'forward' ? bestThroughputRecord.forward_only_throughput_fps : bestThroughputRecord.throughput_fps).toLocaleString('en-US', { maximumFractionDigits: 1 })} img/s
                  </span>
                </div>
                <p className="text-[11px] font-sans text-text-secondary leading-snug">
                  Achieved on {bestThroughputRecord.device.toUpperCase()} ({bestThroughputRecord.backend}) at batch size {bestThroughputRecord.batch_size}.
                </p>
              </div>
            ) : (
              <p className="text-[11px] font-sans text-text-muted">No matching record in current view.</p>
            )}
          </div>

          {/* Storage Efficiency Highlight */}
          <div className="p-3.5 rounded-xl bg-purple-500/5 border border-purple-500/20 space-y-1.5">
            <div className="flex items-center justify-between pb-1 border-b border-purple-500/20">
              <span className="font-bold text-text-primary flex items-center gap-1.5">
                <HardDrive className="w-3.5 h-3.5 text-purple-400" />
                Smallest Serialized Footprint
              </span>
              <Badge variant="outline" className="text-purple-400 border-purple-500/30 text-[9px]">
                Storage Min
              </Badge>
            </div>
            {minSizeVariant ? (
              <div className="space-y-1 pt-0.5">
                <div className="flex items-baseline justify-between">
                  <span className="text-text-primary font-bold">
                    {minSizeVariant.display_name} [{minSizeVariant.precision}]
                  </span>
                  <span className="text-purple-400 font-bold">
                    {minSizeVariant.serialized_mb.toFixed(2)} MB
                  </span>
                </div>
                <p className="text-[11px] font-sans text-text-secondary leading-snug">
                  Yields {minSizeVariant.compression_ratio.toFixed(2)}× reduction ({minSizeVariant.size_reduction_pct.toFixed(1)}% drop) relative to {minSizeVariant.display_name} FP32.
                </p>
              </div>
            ) : (
              <p className="text-[11px] font-sans text-text-muted">No variant in current view.</p>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
