import { Badge } from '@/components/ui/Badge';
import { HelpCircle, Database, Cpu, Layers, Activity, AlertTriangle } from 'lucide-react';
import { researchData, formatNumber } from '@/data';

export function ResearchHeader() {
  const { metadata } = researchData;

  return (
    <div className="space-y-4">
      {/* Hero Banner / Research Question */}
      <div className="bg-surface border border-surface-border rounded-xl p-6 sm:p-8 relative overflow-hidden">
        <div className="absolute top-0 right-0 -mr-16 -mt-16 w-80 h-80 bg-accent/5 rounded-full blur-3xl pointer-events-none" />

        <div className="space-y-4 max-w-4xl">
          {/* Tag Badges */}
          <div className="flex flex-wrap items-center gap-2">
            <Badge variant="accent">Empirical Research Laboratory</Badge>
            <Badge variant="outline" className="font-mono">Milestone 5 Dashboard</Badge>
            <Badge variant="purple" className="font-mono">
              Hardware: {metadata.hardware.split('(')[0].trim()}
            </Badge>
          </div>

          {/* Title */}
          <div>
            <h1 className="text-2xl sm:text-3xl font-bold text-text-primary tracking-tight leading-tight">
              {metadata.title}
            </h1>
            <p className="text-xs sm:text-sm text-text-secondary mt-1">
              Cross-Precision Efficiency, Environmental Robustness & Inference Analysis
            </p>
          </div>

          {/* Central Research Question Callout */}
          <div className="p-4 bg-surface-secondary/80 border border-accent/20 rounded-lg shadow-xs">
            <div className="flex items-start gap-3">
              <HelpCircle className="w-5 h-5 text-accent shrink-0 mt-0.5" />
              <div className="space-y-1">
                <span className="text-[10px] font-mono uppercase tracking-wider text-accent font-semibold">
                  Central Research Question
                </span>
                <p className="text-xs sm:text-sm text-text-primary font-medium leading-relaxed italic">
                  "{metadata.objective}"
                </p>
              </div>
            </div>
          </div>

          {/* Experimental Scope Metric Pills */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-2">
            <div className="p-3 bg-surface-secondary/40 border border-surface-border rounded-lg space-y-1">
              <div className="flex items-center gap-1.5 text-[11px] text-text-muted">
                <Database className="w-3.5 h-3.5 text-accent" />
                <span>Dataset</span>
              </div>
              <div className="text-sm font-bold font-mono text-text-primary">
                {metadata.dataset}
              </div>
              <div className="text-[10px] text-text-muted">10,000 test set</div>
            </div>

            <div className="p-3 bg-surface-secondary/40 border border-surface-border rounded-lg space-y-1">
              <div className="flex items-center gap-1.5 text-[11px] text-text-muted">
                <Layers className="w-3.5 h-3.5 text-accent" />
                <span>Configurations</span>
              </div>
              <div className="text-sm font-bold font-mono text-text-primary">
                {metadata.architectures_count} Arch × {metadata.precisions_count} Prec
              </div>
              <div className="text-[10px] text-text-muted">6 standardized variants</div>
            </div>

            <div className="p-3 bg-surface-secondary/40 border border-surface-border rounded-lg space-y-1">
              <div className="flex items-center gap-1.5 text-[11px] text-text-muted">
                <Activity className="w-3.5 h-3.5 text-accent" />
                <span>Perturbations</span>
              </div>
              <div className="text-sm font-bold font-mono text-text-primary">
                {metadata.corruptions_count} × {metadata.severity_levels_count} Severities
              </div>
              <div className="text-[10px] text-text-muted">30 corrupted conditions</div>
            </div>

            <div
              className="p-3 bg-surface-secondary/40 border border-surface-border rounded-lg space-y-1"
              title="1,860,000 model-image prediction evaluations (6 model configurations × 31 conditions × 10,000 test images), not independent experiment runs."
            >
              <div className="flex items-center gap-1.5 text-[11px] text-text-muted">
                <Cpu className="w-3.5 h-3.5 text-accent" />
                <span>Prediction Evaluations</span>
              </div>
              <div className="text-sm font-bold font-mono text-text-primary">
                {formatNumber(metadata.total_project_predictions)}
              </div>
              <div className="text-[10px] text-text-muted">6 configs × 31 conds × 10k</div>
            </div>
          </div>

          {/* Methodology Context Notice */}
          <div className="flex items-start gap-2.5 p-3 rounded-lg bg-surface-secondary/30 border border-surface-border text-xs text-text-secondary leading-relaxed">
            <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
            <div>
              <strong className="text-text-primary font-medium">Custom Synthetic Suite Notice:</strong>{' '}
              The robustness evaluation employs a custom synthetic corruption suite (6 transformations × 5 severities)
              tailored to measure mathematical perturbation stability. It is not the official CIFAR-10-C benchmark,
              and results are reported as empirical top-1 classification accuracies rather than mean Corruption Error (mCE).
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
