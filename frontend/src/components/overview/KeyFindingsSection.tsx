import { Card } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { CheckCircle2 } from 'lucide-react';
import { researchData } from '@/data';

export function KeyFindingsSection() {
  const { keyFindings } = researchData;

  return (
    <section className="space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-1 border-b border-surface-border">
        <div>
          <h2 className="text-lg font-bold text-text-primary tracking-tight">
            Key Empirical Research Findings
          </h2>
          <p className="text-xs text-text-secondary">
            Synthesized conclusions grounded in 1,860,000 model-image prediction evaluations and empirical hardware profiling.
          </p>
        </div>
        <Badge variant="accent" className="font-mono text-[11px]">
          5 Verified Conclusions
        </Badge>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {keyFindings.map((finding) => (
          <Card key={finding.id} className="p-4 space-y-2 bg-surface/90 border-surface-border hover:border-surface-border-subtle transition-colors">
            <div className="flex items-center justify-between gap-2">
              <Badge variant="outline" className="text-[10px] font-mono">
                Finding {finding.id}
              </Badge>
              <Badge variant="default" className="text-[10px]">
                {finding.badge}
              </Badge>
            </div>

            <h3 className="text-sm font-bold text-text-primary tracking-tight leading-snug">
              {finding.title}
            </h3>

            <p className="text-xs text-text-secondary leading-relaxed">
              {finding.summary}
            </p>

            <div className="pt-2 border-t border-surface-border/60 flex items-start gap-1.5 text-[11px] text-text-muted font-mono">
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0 mt-0.5" />
              <span>
                <strong>Evidence:</strong> {finding.evidence}
              </span>
            </div>
          </Card>
        ))}
      </div>
    </section>
  );
}
