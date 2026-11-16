import { useState } from 'react';
import { Badge } from '@/components/ui/Badge';
import { Card } from '@/components/ui/Card';
import {
  GitCompare,
  Layers,
  Cpu,
  BarChart2,
  HardDrive,
  Target,
  Sliders,
  ShieldAlert,
} from 'lucide-react';
import { researchData } from '@/data';
import {
  ComparisonControls,
  type SelectedConfig,
} from '@/components/comparison/ComparisonControls';
import { ComparisonInsights } from '@/components/comparison/ComparisonInsights';
import { ArchitectureComparisonSection } from '@/components/comparison/ArchitectureComparisonSection';
import { PrecisionComparisonSection } from '@/components/comparison/PrecisionComparisonSection';
import { CleanCorruptedPerformanceSection } from '@/components/comparison/CleanCorruptedPerformanceSection';
import { EfficiencyBenchmarkSection } from '@/components/comparison/EfficiencyBenchmarkSection';
import { ParetoFrontierSection } from '@/components/comparison/ParetoFrontierSection';

export function ModelComparisonPage() {
  // Initialize comparison with canonical defaults: ResNet-18 FP32 vs MobileNetV3-Small FP32
  const [configA, setConfigA] = useState<SelectedConfig>({
    model_id: 'resnet18_cifar',
    precision: 'fp32',
  });

  const [configB, setConfigB] = useState<SelectedConfig>({
    model_id: 'mobilenet_v3_small_cifar',
    precision: 'fp32',
  });

  const handleSwap = () => {
    const temp = { ...configA };
    setConfigA({ ...configB });
    setConfigB(temp);
  };

  const handleReset = () => {
    setConfigA({ model_id: 'resnet18_cifar', precision: 'fp32' });
    setConfigB({ model_id: 'mobilenet_v3_small_cifar', precision: 'fp32' });
  };

  const variantA =
    researchData.precisionVariants.find(
      (v) => v.model_id === configA.model_id && v.precision_key === configA.precision
    ) || researchData.precisionVariants[0];

  const variantB =
    researchData.precisionVariants.find(
      (v) => v.model_id === configB.model_id && v.precision_key === configB.precision
    ) || researchData.precisionVariants[3];

  return (
    <div className="space-y-8 pb-12">
      {/* 1. Header Banner */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-3 border-b border-surface-border">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-xl sm:text-2xl font-bold text-text-primary tracking-tight">
              Model Comparison Dashboard
            </h1>
            <Badge variant="accent">Module 03</Badge>
          </div>
          <p className="text-xs sm:text-sm text-text-secondary mt-1 max-w-4xl leading-relaxed">
            Multivariate comparative evaluation examining architectural capacity, numerical precision quantization (FP32, FP16, INT8), predictive reliability under image degradation, and hardware execution efficiency on Apple Silicon.
          </p>
        </div>

        <div className="flex items-center gap-2 font-mono text-xs">
          <Badge variant="outline" className="text-text-muted">
            6 Configurations
          </Badge>
          <Badge variant="purple" className="text-purple-400">
            Apple M3 Unified
          </Badge>
        </div>
      </div>

      {/* Analytical Focus Guidance Banner */}
      <div className="p-4 rounded-xl bg-gradient-to-r from-surface-secondary/70 via-surface-secondary/40 to-surface-secondary/70 border border-surface-border text-xs text-text-secondary space-y-1.5">
        <div className="flex items-center gap-2 text-text-primary font-bold">
          <GitCompare className="w-4 h-4 text-accent shrink-0" />
          <span>Core Research Inquiry & Decision Matrix:</span>
        </div>
        <p className="text-[11px] leading-relaxed text-text-secondary pl-6">
          <em>Which architecture is more accurate? Which is smaller? What changes when FP32 is converted to FP16 or INT8? Does compression affect clean accuracy or corrupted-image reliability? Which configurations provide the best measured inference performance?</em> Empirical measurements demonstrate that <strong>there is no universally superior configuration</strong>; optimal selection depends strictly on deployment constraints across memory, accuracy tolerance, and latency budgets.
        </p>
      </div>

      {/* Quick Jump Sub-Navigation Bar */}
      <div className="flex flex-wrap items-center gap-1.5 p-1.5 rounded-lg bg-surface-secondary/40 border border-surface-border font-mono text-xs text-text-muted">
        <span className="text-[10px] uppercase font-bold text-text-primary px-2">Jump To:</span>
        <a
          href="#controls"
          className="px-2.5 py-1 rounded hover:bg-surface hover:text-text-primary transition-colors flex items-center gap-1"
        >
          <Sliders className="w-3 h-3 text-accent" />
          Controls & Insights
        </a>
        <a
          href="#architecture"
          className="px-2.5 py-1 rounded hover:bg-surface hover:text-text-primary transition-colors flex items-center gap-1"
        >
          <Layers className="w-3 h-3 text-accent" />
          Architecture (Sec B)
        </a>
        <a
          href="#precision"
          className="px-2.5 py-1 rounded hover:bg-surface hover:text-text-primary transition-colors flex items-center gap-1"
        >
          <Cpu className="w-3 h-3 text-accent" />
          Precision Matrix (Sec C)
        </a>
        <a
          href="#clean-corrupted"
          className="px-2.5 py-1 rounded hover:bg-surface hover:text-text-primary transition-colors flex items-center gap-1"
        >
          <BarChart2 className="w-3 h-3 text-accent" />
          Clean vs Corrupted (Sec D)
        </a>
        <a
          href="#efficiency"
          className="px-2.5 py-1 rounded hover:bg-surface hover:text-text-primary transition-colors flex items-center gap-1"
        >
          <HardDrive className="w-3 h-3 text-accent" />
          Efficiency Benchmarks (Sec E)
        </a>
        <a
          href="#pareto"
          className="px-2.5 py-1 rounded hover:bg-surface hover:text-text-primary transition-colors flex items-center gap-1"
        >
          <Target className="w-3 h-3 text-accent" />
          Pareto Frontier (Sec F)
        </a>
      </div>

      {/* Section A: Comparison Controls */}
      <section id="controls" className="space-y-4">
        <ComparisonControls
          configA={configA}
          configB={configB}
          onChangeA={setConfigA}
          onChangeB={setConfigB}
          onSwap={handleSwap}
          onReset={handleReset}
          precisionVariants={researchData.precisionVariants}
        />

        {/* Section Insights: Dynamic Analytical Callouts */}
        <ComparisonInsights
          variantA={variantA}
          variantB={variantB}
          precisionVariants={researchData.precisionVariants}
        />
      </section>

      {/* Section B: Architecture Comparison */}
      <section id="architecture" className="space-y-4 pt-2">
        <ArchitectureComparisonSection
          baselineComparison={researchData.baselineComparison}
        />
      </section>

      {/* Section C: Precision Comparison */}
      <section id="precision" className="space-y-4 pt-2">
        <PrecisionComparisonSection
          precisionVariants={researchData.precisionVariants}
        />
      </section>

      {/* Section D: Clean vs Corrupted Performance */}
      <section id="clean-corrupted" className="space-y-4 pt-2">
        <CleanCorruptedPerformanceSection
          precisionVariants={researchData.precisionVariants}
        />
      </section>

      {/* Section E: Efficiency Comparison Benchmarks */}
      <section id="efficiency" className="space-y-4 pt-2">
        <EfficiencyBenchmarkSection
          benchmarkRecords={researchData.benchmarkRecords}
        />
      </section>

      {/* Section F: Multi-Objective Pareto Frontier */}
      <section id="pareto" className="space-y-4 pt-2">
        <ParetoFrontierSection
          paretoFrontiers={researchData.paretoFrontiers}
        />
      </section>

      {/* Methodology Limitations Banner */}
      <Card className="p-4 bg-surface-secondary/40 border-surface-border text-xs text-text-secondary space-y-2">
        <div className="flex items-center gap-2 text-text-primary font-bold">
          <ShieldAlert className="w-4 h-4 text-accent shrink-0" />
          <span>Scientific Methodology & Empirical Evaluation Limitations:</span>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-[11px] text-text-secondary leading-relaxed pt-1">
          <div>
            <strong className="text-text-primary block">Dataset & Spatial Resolution:</strong>
            Evaluated exclusively on the CIFAR-10 test split (10,000 images at canonical 32×32 pixel spatial resolution). Results demonstrate algorithmic sensitivity under downscaled conditions and do not assert generalization to multi-megapixel high-resolution camera feeds or real-world industrial deployments.
          </div>
          <div>
            <strong className="text-text-primary block">Custom Synthetic Suite (Not CIFAR-10-C):</strong>
            Degradation stability is measured across 6 custom transformations × 5 severity levels (30 conditions), including 1D horizontal motion blur and photometric darkening factors [0.30, 0.90]. It is not the official 15-corruption CIFAR-10-C benchmark.
          </div>
          <div>
            <strong className="text-text-primary block">Experimental Controls & Seeding:</strong>
            All models were trained under a single fixed experimental seed (42). Measurements reflect deterministic evaluation passes on held-out splits without claiming multi-seed statistical significance.
          </div>
          <div>
            <strong className="text-text-primary block">Hardware & Runtime Comparability:</strong>
            Benchmark metrics reflect Apple M3 hardware (macOS, PyTorch 2.15.0). INT8 executes on CPU via QNNPACK; FP16 executes on Apple Silicon MPS. Measurements are not cross-compatible without taking silicon dispatch architecture into account.
          </div>
        </div>
      </Card>
    </div>
  );
}
