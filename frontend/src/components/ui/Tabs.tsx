import React from 'react';
import { cn } from '@/lib/utils';

export interface TabItem {
  id: string;
  label: string;
  icon?: React.ReactNode;
  badge?: string | number;
}

export interface TabsProps {
  items: TabItem[];
  activeId: string;
  onChange: (id: string) => void;
  className?: string;
  variant?: 'pills' | 'underline';
}

export function Tabs({ items, activeId, onChange, className, variant = 'pills' }: TabsProps) {
  if (variant === 'underline') {
    return (
      <div className={cn('flex space-x-4 border-b border-surface-border', className)}>
        {items.map((tab) => {
          const isActive = tab.id === activeId;
          return (
            <button
              key={tab.id}
              onClick={() => onChange(tab.id)}
              className={cn(
                'flex items-center gap-2 pb-2.5 text-xs font-medium transition-colors relative focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-accent',
                isActive
                  ? 'text-accent border-b-2 border-accent font-semibold'
                  : 'text-text-secondary hover:text-text-primary'
              )}
            >
              {tab.icon}
              <span>{tab.label}</span>
              {tab.badge !== undefined && (
                <span className="text-[10px] px-1.5 py-0.2 bg-surface-secondary rounded font-mono text-text-muted">
                  {tab.badge}
                </span>
              )}
            </button>
          );
        })}
      </div>
    );
  }

  return (
    <div
      className={cn(
        'inline-flex p-1 bg-surface-secondary border border-surface-border rounded-lg gap-1',
        className
      )}
      role="tablist"
    >
      {items.map((tab) => {
        const isActive = tab.id === activeId;
        return (
          <button
            key={tab.id}
            role="tab"
            aria-selected={isActive}
            onClick={() => onChange(tab.id)}
            className={cn(
              'flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent select-none',
              isActive
                ? 'bg-surface text-text-primary shadow-sm border border-surface-border-subtle font-semibold'
                : 'text-text-secondary hover:text-text-primary hover:bg-surface-hover/50'
            )}
          >
            {tab.icon}
            <span>{tab.label}</span>
            {tab.badge !== undefined && (
              <span
                className={cn(
                  'text-[10px] px-1.5 py-0.2 rounded font-mono ml-0.5',
                  isActive
                    ? 'bg-accent-subtle text-accent'
                    : 'bg-surface text-text-muted'
                )}
              >
                {tab.badge}
              </span>
            )}
          </button>
        );
      })}
    </div>
  );
}
