import { useState } from 'react';
import { Badge } from '@/components/ui/Badge';
import { Shield } from 'lucide-react';
import { researchData } from '@/data';
import {
  RobustnessControls,
  type RobustnessFilterState,
} from '@/components/robustness/RobustnessControls';
import { RobustnessSummarySection } from '@/components/robustness/RobustnessSummarySection';
import { SeverityDegradationSection } from '@/components/robustness/SeverityDegradationSection';
import { CorruptionBreakdownSection } from '@/components/robustness/CorruptionBreakdownSection';
import { PrecisionRobustnessSection } from '@/components/robustness/PrecisionRobustnessSection';
import { RobustnessMethodologySection } from '@/components/robustness/RobustnessMethodologySection';

export function RobustnessLabPage() {
  const [filters, setFilters] = useState<RobustnessFilterState>({
    model_id: 'resnet18_cifar',
    precision: 'fp32',
    category: 'all',
    corruption: 'all',
    severity: 'all',
  });

  const handleReset = () => {
    setFilters({
      model_id: 'resnet18_cifar',
      precision: 'fp32',
      category: 'all',
      corruption: 'all',
      severity: 'all',
    });
  };

  return (
    <div className="space-y-8 pb-12">
      {/* 1. Header Banner */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-3 border-b border-surface-border">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-xl sm:text-2xl font-bold text-text-primary tracking-tight">
              Robustness Laboratory
            </h1>
            <Badge variant="accent">Module 04</Badge>
          </div>
          <p className="text-xs sm:text-sm text-text-secondary mt-1 max-w-4xl leading-relaxed">
            Systematic empirical evaluation of model resilience and failure boundaries across 6 custom synthetic corruption transformations and 5 progressive severity levels (30 conditions, 300,000 forward passes per model configuration).
          </p>
        </div>

        <div className="flex items-center gap-2 font-mono text-xs">
          <Badge variant="outline" className="text-text-muted">
            30 Conditions / Config
          </Badge>
          <Badge variant="purple" className="text-purple-400">
            Seed 42 (Deterministic)
          </Badge>
        </div>
      </div>

      {/* Analytical Focus Guidance Banner */}
      <div className="p-4 rounded-xl bg-gradient-to-r from-surface-secondary/70 via-surface-secondary/40 to-surface-secondary/70 border border-surface-border text-xs text-text-secondary space-y-1.5">
        <div className="flex items-center gap-2 text-text-primary font-bold">
          <Shield className="w-4 h-4 text-accent shrink-0" />
          <span>Core Robustness Inquiry & Stress-Testing Framework:</span>
        </div>
        <p className="text-[11px] leading-relaxed text-text-secondary pl-6">
          <em>How much visual distortion can each architecture tolerate before accuracy collapses? Which degradation modes inflict the highest damage? Does post-training quantization to INT8 or FP16 compromise corruption resilience? Which architecture maintains superior stability under specific environmental artifacts?</em> Empirical evaluations reveal that <strong>quantized models preserve or marginally improve robustness</strong> over FP32 baselines, and compact architectures exhibit distinct inductive strengths—such as MobileNetV3-Small outperforming ResNet-18 under JPEG compression (69.66% vs 67.83%).
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
          Reliability Summary (Sec B)
        </a>
        <a
          href="#severity"
          className="px-2 py-1 rounded hover:bg-surface-hover hover:text-text-primary transition-colors"
        >
          Severity Trajectory (Sec C)
        </a>
        <a
          href="#corruptions"
          className="px-2 py-1 rounded hover:bg-surface-hover hover:text-text-primary transition-colors"
        >
          Corruption Breakdown (Sec D)
        </a>
        <a
          href="#precision"
          className="px-2 py-1 rounded hover:bg-surface-hover hover:text-text-primary transition-colors"
        >
          Precision Stability (Sec E)
        </a>
        <a
          href="#methodology"
          className="px-2 py-1 rounded hover:bg-surface-hover hover:text-text-primary transition-colors"
        >
          Methodology (Sec F)
        </a>
      </div>

      {/* Section A: Exploration Controls */}
      <section id="controls" className="space-y-4 pt-2">
        <RobustnessControls
          filters={filters}
          onChange={setFilters}
          onReset={handleReset}
          suite={researchData.robustnessSuite}
          precisionVariants={researchData.precisionVariants}
        />
      </section>

      {/* Section B: Robustness Summary */}
      <section id="summary" className="space-y-4 pt-2">
        <RobustnessSummarySection
          filters={filters}
          suite={researchData.robustnessSuite}
        />
      </section>

      {/* Section C: Severity Degradation Trajectory */}
      <section id="severity" className="space-y-4 pt-2">
        <SeverityDegradationSection
          filters={filters}
          suite={researchData.robustnessSuite}
        />
      </section>

      {/* Section D: Corruption Breakdown */}
      <section id="corruptions" className="space-y-4 pt-2">
        <CorruptionBreakdownSection
          filters={filters}
          suite={researchData.robustnessSuite}
        />
      </section>

      {/* Section E: Precision Robustness Comparison */}
      <section id="precision" className="space-y-4 pt-2">
        <PrecisionRobustnessSection
          filters={filters}
          suite={researchData.robustnessSuite}
          precisionVariants={researchData.precisionVariants}
        />
      </section>

      {/* Section F: Experimental Methodology & Limitations */}
      <section id="methodology" className="space-y-4 pt-2">
        <RobustnessMethodologySection />
      </section>
    </div>
  );
}
