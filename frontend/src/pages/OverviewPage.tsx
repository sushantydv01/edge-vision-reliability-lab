import { Link } from 'react-router-dom';
import { Card } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import {
  ArrowRight,
  Zap,
  Layers,
  ShieldCheck,
  Gauge,
} from 'lucide-react';

import { ResearchHeader } from '@/components/overview/ResearchHeader';
import { BaselineComparisonSection } from '@/components/overview/BaselineComparisonSection';
import { PrecisionMatrixSection } from '@/components/overview/PrecisionMatrixSection';
import { RobustnessOverviewSection } from '@/components/overview/RobustnessOverviewSection';
import { EfficiencyHighlightsSection } from '@/components/overview/EfficiencyHighlightsSection';
import { KeyFindingsSection } from '@/components/overview/KeyFindingsSection';
import { MethodologyLimitationsSection } from '@/components/overview/MethodologyLimitationsSection';

export function OverviewPage() {
  return (
    <div className="space-y-8 pb-10">
      {/* 1. Research Header & Meta */}
      <ResearchHeader />

      {/* 2. Baseline Model Comparison */}
      <BaselineComparisonSection />

      {/* 3. Compression & Precision Matrix */}
      <PrecisionMatrixSection />

      {/* 4. Robustness Overview & Charts */}
      <RobustnessOverviewSection />

      {/* 5. Inference Efficiency Highlights */}
      <EfficiencyHighlightsSection />

      {/* 6. Key Empirical Research Findings */}
      <KeyFindingsSection />

      {/* 7. Methodology Limitations */}
      <MethodologyLimitationsSection />

      {/* 8. Laboratory Navigation Shortcuts */}
      <section className="space-y-4 pt-2">
        <div className="flex items-center justify-between pb-1 border-b border-surface-border">
          <div>
            <h2 className="text-base font-bold text-text-primary tracking-tight">
              Interactive Research Modules
            </h2>
            <p className="text-xs text-text-secondary">
              Deep dive into dedicated experimental workstations across the lab.
            </p>
          </div>
          <Badge variant="outline" className="text-[10px] font-mono">
            4 Interactive Suites
          </Badge>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <Link to="/inference" className="group">
            <Card className="p-4 h-full bg-surface border-surface-border hover:border-accent/50 transition-all hover:shadow-md flex flex-col justify-between">
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <div className="p-2 rounded-lg bg-accent/10 border border-accent/20 text-accent group-hover:bg-accent group-hover:text-white transition-colors">
                    <Zap className="w-4 h-4" />
                  </div>
                  <Badge variant="accent" className="text-[9px]">Module 02</Badge>
                </div>
                <h3 className="text-sm font-semibold text-text-primary group-hover:text-accent transition-colors">
                  Live Inference
                </h3>
                <p className="text-xs text-text-secondary leading-relaxed">
                  Interactive prediction workspace connecting to the local REST API and model registry.
                </p>
              </div>
              <div className="pt-3 flex items-center text-xs text-accent font-medium">
                <span>Open Workspace</span>
                <ArrowRight className="w-3.5 h-3.5 ml-1 group-hover:translate-x-1 transition-transform" />
              </div>
            </Card>
          </Link>

          <Link to="/comparison" className="group">
            <Card className="p-4 h-full bg-surface border-surface-border hover:border-accent/50 transition-all hover:shadow-md flex flex-col justify-between">
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <div className="p-2 rounded-lg bg-accent/10 border border-accent/20 text-accent group-hover:bg-accent group-hover:text-white transition-colors">
                    <Layers className="w-4 h-4" />
                  </div>
                  <Badge variant="default" className="text-[9px]">Module 03</Badge>
                </div>
                <h3 className="text-sm font-semibold text-text-primary group-hover:text-accent transition-colors">
                  Model Comparison
                </h3>
                <p className="text-xs text-text-secondary leading-relaxed">
                  Comprehensive 6-way cross-precision evaluation matrix and parameter analysis.
                </p>
              </div>
              <div className="pt-3 flex items-center text-xs text-accent font-medium">
                <span>View Comparison</span>
                <ArrowRight className="w-3.5 h-3.5 ml-1 group-hover:translate-x-1 transition-transform" />
              </div>
            </Card>
          </Link>

          <Link to="/robustness" className="group">
            <Card className="p-4 h-full bg-surface border-surface-border hover:border-accent/50 transition-all hover:shadow-md flex flex-col justify-between">
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <div className="p-2 rounded-lg bg-accent/10 border border-accent/20 text-accent group-hover:bg-accent group-hover:text-white transition-colors">
                    <ShieldCheck className="w-4 h-4" />
                  </div>
                  <Badge variant="outline" className="text-[9px]">Module 04</Badge>
                </div>
                <h3 className="text-sm font-semibold text-text-primary group-hover:text-accent transition-colors">
                  Robustness Lab
                </h3>
                <p className="text-xs text-text-secondary leading-relaxed">
                  Degradation curve inspector across 6 custom transformations and 5 severities.
                </p>
              </div>
              <div className="pt-3 flex items-center text-xs text-accent font-medium">
                <span>Explore Lab</span>
                <ArrowRight className="w-3.5 h-3.5 ml-1 group-hover:translate-x-1 transition-transform" />
              </div>
            </Card>
          </Link>

          <Link to="/efficiency" className="group">
            <Card className="p-4 h-full bg-surface border-surface-border hover:border-accent/50 transition-all hover:shadow-md flex flex-col justify-between">
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <div className="p-2 rounded-lg bg-accent/10 border border-accent/20 text-accent group-hover:bg-accent group-hover:text-white transition-colors">
                    <Gauge className="w-4 h-4" />
                  </div>
                  <Badge variant="outline" className="text-[9px]">Module 05</Badge>
                </div>
                <h3 className="text-sm font-semibold text-text-primary group-hover:text-accent transition-colors">
                  Efficiency Explorer
                </h3>
                <p className="text-xs text-text-secondary leading-relaxed">
                  Multi-objective Pareto optimality analysis and empirical CPU/MPS scaling curves.
                </p>
              </div>
              <div className="pt-3 flex items-center text-xs text-accent font-medium">
                <span>Explore Pareto</span>
                <ArrowRight className="w-3.5 h-3.5 ml-1 group-hover:translate-x-1 transition-transform" />
              </div>
            </Card>
          </Link>
        </div>
      </section>
    </div>
  );
}
