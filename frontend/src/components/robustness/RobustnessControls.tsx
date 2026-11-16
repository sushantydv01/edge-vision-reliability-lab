import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { Select } from '@/components/ui/Select';
import { Sliders, RotateCcw, ShieldCheck, Activity } from 'lucide-react';
import type { RobustnessSuite, PrecisionVariant } from '@/data/types';
import { formatPercent } from '@/data/formatters';

export interface RobustnessFilterState {
  model_id: string;
  precision: 'fp32' | 'fp16' | 'int8';
  category: string;
  corruption: string;
  severity: string; // 'all' or '1'..'5'
}

interface RobustnessControlsProps {
  filters: RobustnessFilterState;
  onChange: (filters: RobustnessFilterState) => void;
  onReset: () => void;
  suite: RobustnessSuite;
  precisionVariants: PrecisionVariant[];
}

const ARCHITECTURE_OPTIONS = [
  { value: 'resnet18_cifar', label: 'ResNet-18 (Dense Residual)' },
  { value: 'mobilenet_v3_small_cifar', label: 'MobileNetV3-Small (Inverted Residual)' },
];

const PRECISION_OPTIONS = [
  { value: 'fp32', label: 'FP32 (PyTorch Eager)' },
  { value: 'fp16', label: 'FP16 (Half-Precision)' },
  { value: 'int8', label: 'INT8 (PTQ TorchScript)' },
];

const CATEGORY_OPTIONS = [
  { value: 'all', label: 'All Categories (4 Families)' },
  { value: 'Noise', label: 'Noise (Sensor Grain)' },
  { value: 'Blur', label: 'Blur (Defocus & Motion)' },
  { value: 'Photometric', label: 'Photometric (Illumination & Contrast)' },
  { value: 'Digital', label: 'Digital (JPEG Compression)' },
];

const CORRUPTION_OPTIONS = [
  { value: 'all', label: 'All Corruptions (6 Transformations)' },
  { value: 'gaussian_noise', label: 'Gaussian Noise [Noise]' },
  { value: 'gaussian_blur', label: 'Gaussian Blur [Blur]' },
  { value: 'motion_blur', label: 'Horizontal Motion Blur [Blur]' },
  { value: 'brightness', label: 'Brightness (Darkening) [Photometric]' },
  { value: 'contrast', label: 'Contrast Attenuation [Photometric]' },
  { value: 'jpeg_compression', label: 'JPEG Compression [Digital]' },
];

const SEVERITY_OPTIONS = [
  { value: 'all', label: 'All Severities (Mean 1–5)' },
  { value: '1', label: 'Severity 1 (Mildest Perturbation)' },
  { value: '2', label: 'Severity 2 (Moderate Perturbation)' },
  { value: '3', label: 'Severity 3 (Medium Perturbation)' },
  { value: '4', label: 'Severity 4 (Heavy Perturbation)' },
  { value: '5', label: 'Severity 5 (Severe Degradation)' },
];

export function RobustnessControls({
  filters,
  onChange,
  onReset,
  suite,
  precisionVariants,
}: RobustnessControlsProps) {
  const activeVariant = precisionVariants.find(
    (v) => v.model_id === filters.model_id && v.precision_key === filters.precision
  );

  const cleanAcc = suite.clean_baselines[`${filters.model_id}_${filters.precision}`] ?? 94.78;

  // Dynamically filter corruption options if category is selected
  const availableCorruptions = CORRUPTION_OPTIONS.filter((opt) => {
    if (opt.value === 'all') return true;
    if (filters.category === 'all') return true;
    const meta = suite.corruption_metadata[opt.value];
    return meta?.category === filters.category;
  });

  const handleCategoryChange = (cat: string) => {
    // If selected corruption doesn't match new category, reset corruption to 'all'
    if (cat !== 'all' && filters.corruption !== 'all') {
      const meta = suite.corruption_metadata[filters.corruption];
      if (meta?.category !== cat) {
        onChange({ ...filters, category: cat, corruption: 'all' });
        return;
      }
    }
    onChange({ ...filters, category: cat });
  };

  const handleCorruptionChange = (corr: string) => {
    if (corr !== 'all') {
      const meta = suite.corruption_metadata[corr];
      if (meta && filters.category !== 'all' && meta.category !== filters.category) {
        onChange({ ...filters, corruption: corr, category: meta.category });
        return;
      }
    }
    onChange({ ...filters, corruption: corr });
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
                Multi-Dimensional Robustness Exploration Controls
              </CardTitle>
            </div>
            <CardDescription className="text-xs mt-1">
              Select model architecture, precision quantization regime, corruption domain, and severity level to isolate reliability dynamics.
            </CardDescription>
          </div>

          <Button
            variant="ghost"
            size="sm"
            onClick={onReset}
            className="text-xs h-8 px-2.5 text-text-muted hover:text-text-primary self-start sm:self-auto"
            title="Reset to canonical baseline (ResNet-18 FP32, All Corruptions, All Severities)"
          >
            <RotateCcw className="w-3.5 h-3.5 mr-1.5" />
            Reset Baseline
          </Button>
        </div>
      </CardHeader>

      <CardContent className="pt-4 space-y-4">
        {/* Main Selection Controls Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
          <Select
            label="Model Architecture"
            value={filters.model_id}
            onChange={(e) => onChange({ ...filters, model_id: e.target.value })}
            options={ARCHITECTURE_OPTIONS}
          />
          <Select
            label="Precision Regime"
            value={filters.precision}
            onChange={(e) =>
              onChange({
                ...filters,
                precision: e.target.value as 'fp32' | 'fp16' | 'int8',
              })
            }
            options={PRECISION_OPTIONS}
          />
          <Select
            label="Corruption Category"
            value={filters.category}
            onChange={(e) => handleCategoryChange(e.target.value)}
            options={CATEGORY_OPTIONS}
          />
          <Select
            label="Specific Corruption"
            value={filters.corruption}
            onChange={(e) => handleCorruptionChange(e.target.value)}
            options={availableCorruptions}
          />
          <Select
            label="Severity Level"
            value={filters.severity}
            onChange={(e) => onChange({ ...filters, severity: e.target.value })}
            options={SEVERITY_OPTIONS}
          />
        </div>

        {/* Active Filter Configuration & Status Bar */}
        <div className="flex flex-wrap items-center justify-between gap-3 p-3 rounded-lg bg-surface-secondary/40 border border-surface-border text-xs font-mono">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-[11px] text-text-muted uppercase flex items-center gap-1">
              <Sliders className="w-3.5 h-3.5 text-accent" />
              Active Target:
            </span>
            <Badge variant="accent" className="text-[10px] uppercase">
              {filters.model_id === 'resnet18_cifar' ? 'ResNet-18' : 'MobileNetV3-Small'}
            </Badge>
            <Badge
              variant={
                filters.precision === 'fp32'
                  ? 'default'
                  : filters.precision === 'fp16'
                  ? 'purple'
                  : 'accent'
              }
              className="text-[10px] uppercase"
            >
              {filters.precision}
            </Badge>
            <Badge variant="outline" className="text-[10px]">
              {filters.category === 'all' ? 'All Categories' : filters.category}
            </Badge>
            <Badge variant="outline" className="text-[10px]">
              {filters.corruption === 'all'
                ? 'All 6 Corruptions'
                : suite.corruption_metadata[filters.corruption]?.name || filters.corruption}
            </Badge>
            <Badge variant="outline" className="text-[10px]">
              {filters.severity === 'all' ? 'Severities 1–5' : `Severity ${filters.severity}`}
            </Badge>
          </div>

          <div className="flex items-center gap-3 text-[11px]">
            <span className="text-text-muted flex items-center gap-1">
              <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
              Clean Baseline:
              <strong className="text-emerald-400 ml-0.5">{formatPercent(cleanAcc)}</strong>
            </span>
            <span>•</span>
            <span className="text-text-muted flex items-center gap-1">
              <Activity className="w-3.5 h-3.5 text-accent" />
              Backend:
              <strong className="text-text-primary ml-0.5">
                {activeVariant?.runtime_backend || 'PyTorch Eager'}
              </strong>
            </span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
