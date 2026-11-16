import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import {
  TrendingUp,
  HardDrive,
  Layers,
  Sparkles,
  Scale,
  Award,
} from 'lucide-react';
import type { PrecisionVariant } from '@/data/types';
import { formatPercent, formatRatio } from '@/data/formatters';

interface ComparisonInsightsProps {
  variantA: PrecisionVariant;
  variantB: PrecisionVariant;
  precisionVariants: PrecisionVariant[];
}

export function ComparisonInsights({
  variantA,
  variantB,
  precisionVariants,
}: ComparisonInsightsProps) {
  // Delta calculations between selected A and B (strictly A minus B)
  const cleanDelta = variantA.clean_accuracy - variantB.clean_accuracy;
  const corruptedDelta = variantA.mean_corrupted_accuracy - variantB.mean_corrupted_accuracy;
  const sizeDiffMB = (variantA.serialized_bytes - variantB.serialized_bytes) / (1000 * 1000);
  const sizeRatio =
    variantA.serialized_bytes >= variantB.serialized_bytes
      ? variantA.serialized_bytes / variantB.serialized_bytes
      : variantB.serialized_bytes / variantA.serialized_bytes;

  const largerConfig = variantA.serialized_bytes >= variantB.serialized_bytes ? variantA : variantB;
  const smallerConfig = variantA.serialized_bytes < variantB.serialized_bytes ? variantA : variantB;

  const largerParamConfig = variantA.parameters >= variantB.parameters ? variantA : variantB;
  const smallerParamConfig = variantA.parameters < variantB.parameters ? variantA : variantB;
  const paramRatio =
    variantA.parameters >= variantB.parameters
      ? variantA.parameters / variantB.parameters
      : variantB.parameters / variantA.parameters;

  // Project extremes across all 6 configurations
  const highestClean = [...precisionVariants].sort((a, b) => b.clean_accuracy - a.clean_accuracy)[0];
  const highestCorrupted = [...precisionVariants].sort((a, b) => b.mean_corrupted_accuracy - a.mean_corrupted_accuracy)[0];
  const smallestFootprint = [...precisionVariants].sort((a, b) => a.serialized_bytes - b.serialized_bytes)[0];
  const highestCompression = [...precisionVariants].sort((a, b) => b.compression_ratio - a.compression_ratio)[0];

  return (
    <Card className="border-surface-border bg-surface/90">
      <CardHeader className="pb-3 border-b border-surface-border">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-accent" />
            <CardTitle className="text-sm font-bold text-text-primary">
              Analytical Comparison Insights & Empirical Observations
            </CardTitle>
          </div>
          <Badge variant="accent" className="text-[10px]">
            Dynamic Evaluation
          </Badge>
        </div>
      </CardHeader>

      <CardContent className="pt-4 space-y-4">
        {/* Dynamic Comparison Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
          {/* 1. Accuracy Differential */}
          <div className="p-3 rounded-lg bg-surface-secondary/50 border border-surface-border space-y-1.5">
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-mono uppercase text-text-muted">Clean Accuracy Delta (A - B)</span>
              <TrendingUp className="w-3.5 h-3.5 text-accent" />
            </div>
            <div className="flex items-baseline gap-2 font-mono">
              <span
                className={`text-lg font-bold ${
                  cleanDelta > 0
                    ? 'text-accent'
                    : cleanDelta < 0
                    ? 'text-purple-400'
                    : 'text-text-primary'
                }`}
              >
                {cleanDelta > 0
                  ? `+${cleanDelta.toFixed(2)} pp`
                  : cleanDelta < 0
                  ? `${cleanDelta.toFixed(2)} pp`
                  : '0.00 pp'}
              </span>
              <span className="text-[11px] text-text-muted">
                {cleanDelta > 0
                  ? 'Config A higher'
                  : cleanDelta < 0
                  ? 'Config B higher'
                  : 'Parity'}
              </span>
            </div>
            <p className="text-[11px] text-text-secondary leading-snug">
              {cleanDelta !== 0
                ? `${cleanDelta > 0 ? 'Config A' : 'Config B'} (${cleanDelta > 0 ? variantA.display_name : variantB.display_name} [${cleanDelta > 0 ? variantA.precision : variantB.precision}]) achieved higher accuracy on the 10,000 CIFAR-10 test set (Config A: ${formatPercent(variantA.clean_accuracy)} vs Config B: ${formatPercent(variantB.clean_accuracy)}).`
                : 'Both selected configurations achieved identical clean test accuracy on the held-out test split.'}
            </p>
          </div>

          {/* 2. Corrupted Accuracy Differential */}
          <div className="p-3 rounded-lg bg-surface-secondary/50 border border-surface-border space-y-1.5">
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-mono uppercase text-text-muted">Corrupted Acc. Delta (A - B)</span>
              <Scale className="w-3.5 h-3.5 text-accent" />
            </div>
            <div className="flex items-baseline gap-2 font-mono">
              <span
                className={`text-lg font-bold ${
                  corruptedDelta > 0
                    ? 'text-accent'
                    : corruptedDelta < 0
                    ? 'text-purple-400'
                    : 'text-text-primary'
                }`}
              >
                {corruptedDelta > 0
                  ? `+${corruptedDelta.toFixed(2)} pp`
                  : corruptedDelta < 0
                  ? `${corruptedDelta.toFixed(2)} pp`
                  : '0.00 pp'}
              </span>
              <span className="text-[11px] text-text-muted">
                {corruptedDelta > 0
                  ? 'Config A higher'
                  : corruptedDelta < 0
                  ? 'Config B higher'
                  : 'Parity'}
              </span>
            </div>
            <p className="text-[11px] text-text-secondary leading-snug">
              {corruptedDelta !== 0
                ? `Averaged across all 30 custom synthetic conditions, ${corruptedDelta > 0 ? 'Config A' : 'Config B'} (${corruptedDelta > 0 ? variantA.display_name : variantB.display_name} [${corruptedDelta > 0 ? variantA.precision : variantB.precision}]) measured ${Math.abs(corruptedDelta).toFixed(2)} pp higher mean accuracy (Config A: ${formatPercent(variantA.mean_corrupted_accuracy)} vs Config B: ${formatPercent(variantB.mean_corrupted_accuracy)}).`
                : 'Both configurations averaged identical accuracy across the 30 synthetic corruption conditions.'}
            </p>
          </div>

          {/* 3. Disk Footprint Ratio */}
          <div className="p-3 rounded-lg bg-surface-secondary/50 border border-surface-border space-y-1.5">
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-mono uppercase text-text-muted">Storage Footprint Ratio</span>
              <HardDrive className="w-3.5 h-3.5 text-accent" />
            </div>
            <div className="flex items-baseline gap-2 font-mono">
              <span className="text-lg font-bold text-text-primary">
                {formatRatio(sizeRatio)}
              </span>
              <span className="text-[11px] text-text-muted">
                {sizeDiffMB !== 0 ? `${smallerConfig.display_name} is more compact` : 'Identical footprint'}
              </span>
            </div>
            <p className="text-[11px] text-text-secondary leading-snug">
              {sizeDiffMB !== 0
                ? `${largerConfig.display_name} [${largerConfig.precision}] occupies ${largerConfig.serialized_mb.toFixed(2)} MB vs ${smallerConfig.serialized_mb.toFixed(2)} MB on disk (Config A: ${variantA.serialized_mb.toFixed(2)} MB vs Config B: ${variantB.serialized_mb.toFixed(2)} MB).`
                : `Both selected models serialize to exactly ${variantA.serialized_mb.toFixed(2)} MB.`}
            </p>
          </div>

          {/* 4. Architectural Parameter Delta */}
          <div className="p-3 rounded-lg bg-surface-secondary/50 border border-surface-border space-y-1.5">
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-mono uppercase text-text-muted">Parameter Scale</span>
              <Layers className="w-3.5 h-3.5 text-accent" />
            </div>
            <div className="flex items-baseline gap-2 font-mono">
              <span className="text-lg font-bold text-text-primary">
                {variantA.parameters === variantB.parameters
                  ? 'Same Model'
                  : formatRatio(paramRatio)}
              </span>
              <span className="text-[11px] text-text-muted">
                {variantA.parameters !== variantB.parameters
                  ? `${variantA.parameters > variantB.parameters ? 'Config A' : 'Config B'} larger`
                  : 'Identical architecture'}
              </span>
            </div>
            <p className="text-[11px] text-text-secondary leading-snug">
              {variantA.parameters !== variantB.parameters
                ? `${largerParamConfig.display_name} contains ${(largerParamConfig.parameters / 1e6).toFixed(2)}M parameters whereas ${smallerParamConfig.display_name} contains ${(smallerParamConfig.parameters / 1e6).toFixed(2)}M parameters (${formatRatio(paramRatio)} parameter scale difference).`
                : `Both configurations share the identical underlying architecture graph (${variantA.display_name}) with identical parameter counts.`}
            </p>
          </div>
        </div>

        {/* Global Empirical Extremes Banner */}
        <div className="p-3 rounded-lg bg-surface-secondary/30 border border-surface-border text-xs space-y-2">
          <div className="flex items-center gap-1.5 text-text-primary font-semibold text-xs">
            <Award className="w-4 h-4 text-accent" />
            <span>Project Baseline Extremes (Across All 6 Evaluated Configurations):</span>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2 font-mono text-[11px]">
            <div className="bg-surface/80 p-2 rounded border border-surface-border">
              <span className="text-[10px] text-text-muted uppercase block">Top Clean Accuracy</span>
              <strong className="text-emerald-400 block">{highestClean.display_name} [{highestClean.precision}]: {formatPercent(highestClean.clean_accuracy)}</strong>
              <span className="text-[10px] text-text-muted">Held-out CIFAR-10 test set</span>
            </div>

            <div className="bg-surface/80 p-2 rounded border border-surface-border">
              <span className="text-[10px] text-text-muted uppercase block">Top Corrupted Accuracy</span>
              <strong className="text-emerald-400 block">{highestCorrupted.display_name} [{highestCorrupted.precision}]: {formatPercent(highestCorrupted.mean_corrupted_accuracy)}</strong>
              <span className="text-[10px] text-text-muted">Mean across 30 conditions</span>
            </div>

            <div className="bg-surface/80 p-2 rounded border border-surface-border">
              <span className="text-[10px] text-text-muted uppercase block">Most Compact Footprint</span>
              <strong className="text-accent block">{smallestFootprint.display_name} [{smallestFootprint.precision}]: {smallestFootprint.serialized_mb.toFixed(2)} MB</strong>
              <span className="text-[10px] text-text-muted">Serialized TorchScript archive</span>
            </div>

            <div className="bg-surface/80 p-2 rounded border border-surface-border">
              <span className="text-[10px] text-text-muted uppercase block">Max Compression Ratio</span>
              <strong className="text-accent block">{highestCompression.display_name} [{highestCompression.precision}]: {formatRatio(highestCompression.compression_ratio)}</strong>
              <span className="text-[10px] text-text-muted">Relative to FP32 serialized baseline</span>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
