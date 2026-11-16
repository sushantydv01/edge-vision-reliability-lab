import React from 'react';
import { cn } from '@/lib/utils';

export interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: 'default' | 'secondary' | 'outline' | 'success' | 'warning' | 'accent' | 'purple';
}

export function Badge({ className, variant = 'default', ...props }: BadgeProps) {
  const variants = {
    default:
      'bg-surface-secondary text-text-primary border-surface-border',
    secondary:
      'bg-surface-hover text-text-secondary border-surface-border-subtle',
    outline:
      'bg-transparent text-text-secondary border-surface-border-subtle',
    accent:
      'bg-accent-subtle text-accent border-accent-border',
    success:
      'bg-emerald-500/10 text-emerald-400 border-emerald-500/20',
    warning:
      'bg-amber-500/10 text-amber-400 border-amber-500/20',
    purple:
      'bg-purple-500/10 text-purple-400 border-purple-500/20',
  };

  return (
    <span
      className={cn(
        'inline-flex items-center px-2 py-0.5 rounded text-[11px] font-medium font-mono border tracking-wide uppercase',
        variants[variant],
        className
      )}
      {...props}
    />
  );
}
