import { useState } from 'react';
import { Badge } from '@/components/ui/Badge';
import { Gauge } from 'lucide-react';
import { researchData } from '@/data';
import {
  EfficiencyControls,
  type EfficiencyFilterState,
} from '@/components/efficiency/EfficiencyControls';
import { EfficiencySummarySection } from '@/components/efficiency/EfficiencySummarySection';
import { LatencyThroughputSection } from '@/components/efficiency/LatencyThroughputSection';
import { ModelSizeCompressionSection } from '@/components/efficiency/ModelSizeCompressionSection';
import { AccuracyEfficiencySection } from '@/components/efficiency/AccuracyEfficiencySection';
import { EfficiencyParetoSection } from '@/components/efficiency/EfficiencyParetoSection';
import { BenchmarkMethodologySection } from '@/components/efficiency/BenchmarkMethodologySection';

export function EfficiencyExplorerPage() {
  const [filters, setFilters] = useState<EfficiencyFilterState>({
    model: 'all',
    precision: 'all',
    device: 'all',
    batch_size: 'all',
    timing_mode: 'total',
    include_diagnostic: false,
  });

  const handleReset = () => {
    setFilters({
      model: 'all',
      precision: 'all',
      device: 'all',
      batch_size: 'all',
      timing_mode: 'total',
      include_diagnostic: false,
    });
  };

  return (
    <div className="space-y-8 pb-12">
      {/* 1. Header Banner */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-3 border-b border-surface-border">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-xl sm:text-2xl font-bold text-text-primary tracking-tight">
              Interactive Efficiency Explorer
            </h1>
            <Badge variant="accent">Module 05</Badge>
          </div>
          <p className="text-xs sm:text-sm text-text-secondary mt-1 max-w-4xl leading-relaxed">
            Multi-dimensional deployment efficiency workbench evaluating serialized on-disk footprint, latency scaling ($B=1, 8, 32, 64$), peak throughput (img/s), and mathematical Pareto optimality across 6 model configurations on Apple Silicon M3.
          </p>
        </div>

        <div className="flex items-center gap-2 font-mono text-xs">
          <Badge variant="outline" className="text-text-muted">
            40 Benchmark Records
          </Badge>
          <Badge variant="purple" className="text-purple-400">
            Apple M3 (Unified Memory)
          </Badge>
        </div>
      </div>

      {/* Analytical Focus Guidance Banner */}
      <div className="p-4 rounded-xl bg-gradient-to-r from-surface-secondary/70 via-surface-secondary/40 to-surface-secondary/70 border border-surface-border text-xs text-text-secondary space-y-1.5">
        <div className="flex items-center gap-2 text-text-primary font-bold">
          <Gauge className="w-4 h-4 text-accent shrink-0" />
          <span>Core Deployment Efficiency & Pareto Inquiries:</span>
        </div>
        <p className="text-[11px] leading-relaxed text-text-secondary pl-6">
          <em>Which configuration achieves the smallest storage footprint? Which yields the lowest single-image latency for real-time edge triggers? How do CPU and Metal MPS acceleration profiles compare under batching regimes? Which architectures define the multi-objective Pareto frontier?</em> Empirical results show that <strong>MobileNetV3-Small INT8</strong> provides the smallest artifact (1.94 MB) and highest batch throughput (7,754.8 img/s total, 11,531.5 img/s forward-only), while <strong>ResNet-18 INT8</strong> defines the high-accuracy Pareto boundary (11.30 MB, 65.76% mean corrupted accuracy).
        </p>
      </div>

      {/* Quick Jump Sub-Navigation Bar */}
      <div className="flex flex-wrap items-center gap-1.5 p-1.5 rounded-lg bg-surface-secondary/40 border border-surface-border font-mono text-xs text-text-muted">
        <span className="text-[10px] uppercase font-bold text-text-primary px-2">Jump To:</span>
        <a
          href="#controls"
          className="px-2 py-1 rounded hover:bg-surface-hover hover:text-text-primary transition-colors"
        >
          Controls (Sec A)
        </a>
        <a
          href="#summary"
          className="px-2 py-1 rounded hover:bg-surface-hover hover:text-text-primary transition-colors"
        >
          Telemetry Summary (Sec B)
        </a>
        <a
          href="#scaling"
          className="px-2 py-1 rounded hover:bg-surface-hover hover:text-text-primary transition-colors"
        >
          Batch & Latency Scaling (Sec C)
        </a>
        <a
          href="#storage"
          className="px-2 py-1 rounded hover:bg-surface-hover hover:text-text-primary transition-colors"
        >
          Storage & Compression (Sec D)
        </a>
        <a
          href="#tradeoffs"
          className="px-2 py-1 rounded hover:bg-surface-hover hover:text-text-primary transition-colors"
        >
          Accuracy Trade-offs (Sec E)
        </a>
        <a
          href="#pareto"
          className="px-2 py-1 rounded hover:bg-surface-hover hover:text-text-primary transition-colors"
        >
          Pareto Frontier (Sec F)
        </a>
        <a
          href="#methodology"
          className="px-2 py-1 rounded hover:bg-surface-hover hover:text-text-primary transition-colors"
        >
          Methodology (Sec G)
        </a>
      </div>

      {/* Section A: Deployment Controls */}
      <section id="controls" className="space-y-4 pt-2">
        <EfficiencyControls
          filters={filters}
          onChange={setFilters}
          onReset={handleReset}
          benchmarkRecords={researchData.benchmarkRecords}
        />
      </section>

      {/* Section B: Overview Telemetry */}
      <section id="summary" className="space-y-4 pt-2">
        <EfficiencySummarySection
          filters={filters}
          benchmarkRecords={researchData.benchmarkRecords}
          precisionVariants={researchData.precisionVariants}
        />
      </section>

      {/* Section C: Latency & Throughput Scaling */}
      <section id="scaling" className="space-y-4 pt-2">
        <LatencyThroughputSection
          filters={filters}
          benchmarkRecords={researchData.benchmarkRecords}
        />
      </section>

      {/* Section D: Model Size & Compression */}
      <section id="storage" className="space-y-4 pt-2">
        <ModelSizeCompressionSection
          precisionVariants={researchData.precisionVariants}
        />
      </section>

      {/* Section E: Accuracy-Efficiency Trade-offs */}
      <section id="tradeoffs" className="space-y-4 pt-2">
        <AccuracyEfficiencySection
          precisionVariants={researchData.precisionVariants}
          benchmarkRecords={researchData.benchmarkRecords}
        />
      </section>

      {/* Section F: Multi-Objective Pareto Frontier */}
      <section id="pareto" className="space-y-4 pt-2">
        <EfficiencyParetoSection
          paretoFrontiers={researchData.paretoFrontiers}
        />
      </section>

      {/* Section G: Methodology & Limitations */}
      <section id="methodology" className="space-y-4 pt-2">
        <BenchmarkMethodologySection />
      </section>
    </div>
  );
}
