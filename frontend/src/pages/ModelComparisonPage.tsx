import { Card } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';

export function ModelComparisonPage() {
  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-2 border-b border-surface-border">
        <div>
          <div className="flex items-center gap-2">
            <h2 className="text-xl font-bold text-text-primary tracking-tight">
              Model Comparison Dashboard
            </h2>
            <Badge variant="accent">Module 03</Badge>
          </div>
          <p className="text-xs text-text-secondary mt-1">
            Comparative analysis of architecture topologies and precision quantization regimes.
          </p>
        </div>
      </div>
      <Card className="p-6 bg-surface-secondary/40 border-surface-border text-center">
        <p className="text-sm text-text-secondary">Model Comparison Dashboard Scaffold</p>
      </Card>
    </div>
  );
}
