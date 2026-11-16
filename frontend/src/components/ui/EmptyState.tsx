import React from 'react';
import { cn } from '@/lib/utils';
import { Inbox } from 'lucide-react';

export interface EmptyStateProps {
  icon?: React.ReactNode;
  title: string;
  description: string;
  action?: React.ReactNode;
  className?: string;
}

export function EmptyState({
  icon = <Inbox className="w-8 h-8 text-text-muted" />,
  title,
  description,
  action,
  className,
}: EmptyStateProps) {
  return (
    <div
      className={cn(
        'flex flex-col items-center justify-center p-8 text-center bg-surface/50 border border-dashed border-surface-border rounded-lg space-y-3',
        className
      )}
    >
      <div className="p-3 bg-surface-secondary/80 rounded-full border border-surface-border">
        {icon}
      </div>
      <div className="space-y-1 max-w-sm">
        <h4 className="text-sm font-semibold text-text-primary tracking-tight">{title}</h4>
        <p className="text-xs text-text-secondary leading-relaxed">{description}</p>
      </div>
      {action && <div className="pt-2">{action}</div>}
    </div>
  );
}
