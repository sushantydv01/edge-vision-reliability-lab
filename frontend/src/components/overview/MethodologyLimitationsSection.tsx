import { Badge } from '@/components/ui/Badge';
import { AlertTriangle } from 'lucide-react';
import { researchData } from '@/data';

export function MethodologyLimitationsSection() {
  const { methodologyLimitations } = researchData;

  return (
    <section className="space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-1 border-b border-surface-border">
        <div>
          <h2 className="text-lg font-bold text-text-primary tracking-tight">
            Documented Methodology & Experimental Boundaries
          </h2>
          <p className="text-xs text-text-secondary">
            Formal experimental constraints and threats to validity established for scientific integrity.
          </p>
        </div>
        <Badge variant="outline" className="font-mono text-[11px] text-amber-400 border-amber-500/30">
          Scientific Boundaries
        </Badge>
      </div>

      <div className="p-4 rounded-xl bg-surface-secondary/40 border border-surface-border space-y-4">
        <div className="flex items-center gap-2 text-text-primary font-semibold text-xs sm:text-sm">
          <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0" />
          <span>Explicit Experimental Boundaries (Non-Generalization Notice)</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {methodologyLimitations.map((lim, idx) => (
            <div
              key={idx}
              className="p-3 rounded-lg bg-surface/60 border border-surface-border/80 space-y-1 text-xs"
            >
              <h4 className="font-semibold text-text-primary text-[11px] flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-amber-400 shrink-0" />
                {lim.title}
              </h4>
              <p className="text-[11px] text-text-secondary leading-relaxed pl-3">
                {lim.detail}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
