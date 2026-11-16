import { cn } from '@/lib/utils';
import { Loader2 } from 'lucide-react';

export interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg';
  className?: string;
  label?: string;
}

export function LoadingSpinner({ size = 'md', className, label }: LoadingSpinnerProps) {
  const sizes = {
    sm: 'w-4 h-4',
    md: 'w-6 h-6',
    lg: 'w-8 h-8',
  };

  return (
    <div role="status" aria-live="polite" className="flex flex-col items-center justify-center p-4 gap-2">
      <Loader2 className={cn('animate-spin text-accent', sizes[size], className)} />
      {label && <p className="text-xs text-text-muted font-medium">{label}</p>}
      <span className="sr-only">{label || 'Loading...'}</span>
    </div>
  );
}

export function Skeleton({ className }: { className?: string }) {
  return (
    <div
      className={cn(
        'animate-pulse rounded bg-surface-secondary/70 border border-surface-border/50',
        className
      )}
    />
  );
}
