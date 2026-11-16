import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { Select } from '@/components/ui/Select';
import { RotateCcw, Info } from 'lucide-react';
import type { BenchmarkRecord } from '@/data/types';

export interface EfficiencyFilterState {
  model: 'all' | 'resnet18_cifar' | 'mobilenet_v3_small_cifar';
  precision: 'all' | 'fp32' | 'fp16' | 'int8';
  device: 'all' | 'cpu' | 'mps';
  batch_size: 'all' | 1 | 8 | 32 | 64;
  timing_mode: 'total' | 'forward';
  include_diagnostic: boolean;
}

interface EfficiencyControlsProps {
  filters: EfficiencyFilterState;
  onChange: (filters: EfficiencyFilterState) => void;
  onReset: () => void;
  benchmarkRecords: BenchmarkRecord[];
}

const ARCHITECTURE_OPTIONS = [
  { value: 'all', label: 'All Architectures (ResNet-18 & MobileNetV3)' },
  { value: 'resnet18_cifar', label: 'ResNet-18 (Dense Residual Baseline)' },
  { value: 'mobilenet_v3_small_cifar', label: 'MobileNetV3-Small (Inverted Residual)' },
];

const PRECISION_OPTIONS = [
  { value: 'all', label: 'All Precisions (FP32, FP16, INT8)' },
  { value: 'fp32', label: 'FP32 (Single Precision Standard)' },
  { value: 'fp16', label: 'FP16 (Half Precision Float)' },
  { value: 'int8', label: 'INT8 (Post-Training Static Quantization)' },
];

const DEVICE_OPTIONS = [
  { value: 'all', label: 'All Devices (CPU & Apple MPS)' },
  { value: 'cpu', label: 'Apple M3 CPU (4 Threads Default)' },
  { value: 'mps', label: 'Apple M3 Metal Performance Shaders (MPS)' },
];

const BATCH_SIZE_OPTIONS = [
  { value: 'all', label: 'All Batch Sizes (1, 8, 32, 64)' },
  { value: '1', label: 'Batch Size 1 (Latency-Critical Edge Trigger)' },
  { value: '8', label: 'Batch Size 8 (Small Micro-Batch)' },
  { value: '32', label: 'Batch Size 32 (Standard Workload)' },
  { value: '64', label: 'Batch Size 64 (Throughput Saturation)' },
];

export function EfficiencyControls({
  filters,
  onChange,
  onReset,
  benchmarkRecords,
}: EfficiencyControlsProps) {
  // Count matching records based on active filters
  const matchingCount = benchmarkRecords.filter((r) => {
    if (filters.model !== 'all' && r.model !== filters.model) return false;
    if (filters.precision !== 'all' && r.precision !== filters.precision) return false;
    if (filters.device !== 'all' && r.device !== filters.device) return false;
    if (filters.batch_size !== 'all' && r.batch_size !== filters.batch_size) return false;
    if (!filters.include_diagnostic && r.is_diagnostic) return false;
    return true;
  }).length;

  const handleModelChange = (val: string) => {
    onChange({ ...filters, model: val as EfficiencyFilterState['model'] });
  };

  const handlePrecisionChange = (val: string) => {
    const newPrec = val as EfficiencyFilterState['precision'];
    // If switching to INT8 and device was MPS, reset device to CPU because MPS does not support INT8
    const newDevice = newPrec === 'int8' && filters.device === 'mps' ? 'cpu' : filters.device;
    onChange({ ...filters, precision: newPrec, device: newDevice });
  };

  const handleDeviceChange = (val: string) => {
    const newDevice = val as EfficiencyFilterState['device'];
    // If switching to MPS and precision was INT8, reset precision to all or fp32 because MPS lacks INT8
    const newPrec = newDevice === 'mps' && filters.precision === 'int8' ? 'all' : filters.precision;
    onChange({ ...filters, device: newDevice, precision: newPrec });
  };

  const handleBatchChange = (val: string) => {
    const newBatch = val === 'all' ? 'all' : (Number(val) as 1 | 8 | 32 | 64);
    onChange({ ...filters, batch_size: newBatch });
  };

  return (
    <Card className="border-surface-border bg-surface/90">
      <CardHeader className="pb-3 border-b border-surface-border">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <div className="flex items-center gap-2">
              <Badge variant="accent" className="font-mono text-[10px]">
                Section A
              </Badge>
              <CardTitle className="text-base font-bold text-text-primary">
                Deployment Configuration & Benchmark Filter Controls
              </CardTitle>
            </div>
            <CardDescription className="text-xs mt-1">
              Select model architecture, execution runtime, compute backend, and batch regime across 40 empirical benchmark measurements on Apple Silicon M3.
            </CardDescription>
          </div>

          <div className="flex items-center gap-2 self-start sm:self-auto">
            <Badge variant="outline" className="font-mono text-xs">
              {matchingCount} Record{matchingCount === 1 ? '' : 's'} Selected
            </Badge>
            <Button
              variant="outline"
              size="sm"
              onClick={onReset}
              className="text-xs font-mono h-8 border-surface-border hover:bg-surface-hover"
            >
              <RotateCcw className="w-3.5 h-3.5 mr-1 text-text-muted" />
              Reset Filters
            </Button>
          </div>
        </div>
      </CardHeader>

      <CardContent className="pt-4 space-y-4">
        {/* Primary Filter Grid: Model, Precision, Device, Batch Size */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          {/* 1. Architecture Selector */}
          <div className="space-y-1">
            <label className="text-[11px] font-mono text-text-muted flex items-center justify-between">
              <span>Architecture</span>
              <span className="text-[10px] text-accent">2 Models</span>
            </label>
            <Select
              options={ARCHITECTURE_OPTIONS}
              value={filters.model}
              onChange={(e) => handleModelChange(e.target.value)}
            />
          </div>

          {/* 2. Precision Selector */}
          <div className="space-y-1">
            <label className="text-[11px] font-mono text-text-muted flex items-center justify-between">
              <span>Precision Mode</span>
              <span className="text-[10px] text-purple-400">3 Formats</span>
            </label>
            <Select
              options={PRECISION_OPTIONS}
              value={filters.precision}
              onChange={(e) => handlePrecisionChange(e.target.value)}
            />
          </div>

          {/* 3. Compute Device / Backend */}
          <div className="space-y-1">
            <label className="text-[11px] font-mono text-text-muted flex items-center justify-between">
              <span>Execution Device</span>
              <span className="text-[10px] text-emerald-400">CPU / MPS</span>
            </label>
            <Select
              options={DEVICE_OPTIONS}
              value={filters.device}
              onChange={(e) => handleDeviceChange(e.target.value)}
            />
          </div>

          {/* 4. Batch Size Selector */}
          <div className="space-y-1">
            <label className="text-[11px] font-mono text-text-muted flex items-center justify-between">
              <span>Batch Size</span>
              <span className="text-[10px] text-accent">B=1, 8, 32, 64</span>
            </label>
            <Select
              options={BATCH_SIZE_OPTIONS}
              value={String(filters.batch_size)}
              onChange={(e) => handleBatchChange(e.target.value)}
            />
          </div>
        </div>

        {/* Secondary Controls Bar: Timing Mode & Diagnostic Toggle */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pt-3 border-t border-surface-border">
          {/* Timing Mode Toggle (Total Engine vs Forward Only) */}
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono text-text-muted">Timing Metric:</span>
            <div className="flex items-center rounded-lg bg-surface-secondary/70 p-1 border border-surface-border">
              <button
                type="button"
                onClick={() => onChange({ ...filters, timing_mode: 'total' })}
                className={`px-3 py-1 text-xs font-mono rounded transition-colors ${
                  filters.timing_mode === 'total'
                    ? 'bg-accent/20 text-accent font-bold border border-accent/40'
                    : 'text-text-muted hover:text-text-primary'
                }`}
              >
                Total-Engine Latency (Pipeline)
              </button>
              <button
                type="button"
                onClick={() => onChange({ ...filters, timing_mode: 'forward' })}
                className={`px-3 py-1 text-xs font-mono rounded transition-colors ${
                  filters.timing_mode === 'forward'
                    ? 'bg-accent/20 text-accent font-bold border border-accent/40'
                    : 'text-text-muted hover:text-text-primary'
                }`}
              >
                Forward-Only Latency (Isolated Graph)
              </button>
            </div>
          </div>

          {/* Diagnostic CPU FP16 Toggle */}
          <div className="flex items-center gap-2 self-start sm:self-auto font-mono text-xs">
            <label className="flex items-center gap-2 cursor-pointer text-text-secondary hover:text-text-primary">
              <input
                type="checkbox"
                checked={filters.include_diagnostic}
                onChange={(e) =>
                  onChange({ ...filters, include_diagnostic: e.target.checked })
                }
                className="rounded border-surface-border bg-surface-secondary text-accent focus:ring-accent h-3.5 w-3.5"
              />
              <span>Include Diagnostic CPU FP16 Emulation Runs</span>
            </label>
          </div>
        </div>

        {/* Informational Guidance Alert */}
        <div className="p-3 rounded-lg bg-surface-secondary/40 border border-surface-border text-xs text-text-secondary flex items-start gap-2.5">
          <Info className="w-4 h-4 text-accent shrink-0 mt-0.5" />
          <div className="space-y-1">
            <p className="text-[11px] leading-relaxed">
              <strong>Runtime Support Guardrail:</strong> INT8 static quantization is evaluated exclusively on CPU using PyTorch's QNNPACK engine (Metal MPS does not support INT8 static quantization). Conversely, FP16 half-precision execution achieves native hardware acceleration on Apple Silicon MPS, but falls back to software emulation on CPU.
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
