import { NavLink } from 'react-router-dom';
import { cn } from '@/lib/utils';
import {
  LayoutDashboard,
  Zap,
  GitCompare,
  ShieldAlert,
  Gauge,
  Microscope,
  Terminal,
} from 'lucide-react';

interface SidebarProps {
  onNavClick?: () => void;
  className?: string;
}

export const NAV_ITEMS = [
  {
    name: 'Overview',
    path: '/',
    icon: LayoutDashboard,
    badge: 'Research',
  },
  {
    name: 'Live Inference',
    path: '/inference',
    icon: Zap,
    badge: 'Engine',
  },
  {
    name: 'Model Comparison',
    path: '/comparison',
    icon: GitCompare,
    badge: 'Eval',
  },
  {
    name: 'Robustness Lab',
    path: '/robustness',
    icon: ShieldAlert,
    badge: 'Robustness',
  },
  {
    name: 'Efficiency Explorer',
    path: '/efficiency',
    icon: Gauge,
    badge: 'Pareto',
  },
];

export function Sidebar({ onNavClick, className }: SidebarProps) {
  return (
    <aside
      className={cn(
        'w-64 bg-surface flex flex-col border-r border-surface-border select-none h-full',
        className
      )}
    >
      {/* Brand & Identity */}
      <div className="p-5 border-b border-surface-border flex items-start gap-3">
        <div className="p-2 bg-accent-subtle rounded-lg border border-accent-border shrink-0 mt-0.5">
          <Microscope className="w-5 h-5 text-accent" />
        </div>
        <div className="space-y-0.5">
          <h1 className="text-sm font-bold text-text-primary tracking-tight leading-tight">
            Edge Vision Lab
          </h1>
          <p className="text-[11px] text-text-secondary leading-snug">
            Model Compression, Robustness & Inference Analysis
          </p>
        </div>
      </div>

      {/* Primary Navigation */}
      <nav className="flex-1 p-3 space-y-1 overflow-y-auto" aria-label="Main Navigation">
        <div className="px-3 py-1.5 text-[10px] font-mono uppercase tracking-wider text-text-muted font-semibold">
          Analysis Modules
        </div>
        {NAV_ITEMS.map((item) => {
          const Icon = item.icon;
          return (
            <NavLink
              key={item.path}
              to={item.path}
              end={item.path === '/'}
              onClick={onNavClick}
              className={({ isActive }) =>
                cn(
                  'flex items-center justify-between px-3 py-2 rounded-md text-xs font-medium transition-all group outline-none',
                  'focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-1 focus-visible:ring-offset-surface',
                  isActive
                    ? 'bg-accent/10 text-accent font-semibold border border-accent-border shadow-xs'
                    : 'text-text-secondary hover:text-text-primary hover:bg-surface-hover/70'
                )
              }
            >
              {({ isActive }) => (
                <>
                  <div className="flex items-center gap-2.5">
                    <Icon
                      className={cn(
                        'w-4 h-4 transition-colors',
                        isActive
                          ? 'text-accent'
                          : 'text-text-muted group-hover:text-text-primary'
                      )}
                    />
                    <span>{item.name}</span>
                  </div>
                  <span
                    className={cn(
                      'text-[9px] font-mono px-1.5 py-0.2 rounded uppercase tracking-wider',
                      isActive
                        ? 'bg-accent-subtle text-accent border border-accent-border'
                        : 'bg-surface-secondary text-text-muted'
                    )}
                  >
                    {item.badge}
                  </span>
                </>
              )}
            </NavLink>
          );
        })}
      </nav>

      {/* System Hardware Profile & Footer */}
      <div className="p-4 border-t border-surface-border bg-surface-secondary/40 space-y-2.5">
        <div className="flex items-center justify-between text-[11px]">
          <span className="text-text-muted flex items-center gap-1.5">
            <Terminal className="w-3.5 h-3.5" />
            Profile:
          </span>
          <span className="font-mono text-text-primary font-medium">Apple M3 (8 GB)</span>
        </div>
        <div className="pt-2 border-t border-surface-border/60 flex items-center justify-between text-[10px] text-text-muted font-mono">
          <span>CIFAR-10 Suite</span>
          <span className="text-accent">v0.1.0</span>
        </div>
      </div>
    </aside>
  );
}
