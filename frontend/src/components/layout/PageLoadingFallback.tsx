import { Loader2, Microscope } from 'lucide-react';

export function PageLoadingFallback() {
  return (
    <div
      role="status"
      aria-live="polite"
      className="min-h-[50vh] flex flex-col items-center justify-center space-y-4 p-8 text-center"
    >
      <div className="relative">
        <div className="p-3 bg-surface-secondary rounded-xl border border-surface-border">
          <Microscope className="w-8 h-8 text-accent animate-pulse" />
        </div>
        <div className="absolute -bottom-1 -right-1 p-1 bg-surface rounded-full border border-surface-border">
          <Loader2 className="w-4 h-4 text-accent animate-spin" />
        </div>
      </div>
      <div className="space-y-1">
        <p className="text-sm font-semibold text-text-primary tracking-tight">
          Loading Analysis Workstation...
        </p>
        <p className="text-xs text-text-muted font-mono">
          Hydrating research telemetry and benchmark models
        </p>
      </div>
    </div>
  );
}
