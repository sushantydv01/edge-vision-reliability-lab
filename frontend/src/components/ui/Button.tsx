import React, { forwardRef } from 'react';
import { cn } from '@/lib/utils';
import { Loader2 } from 'lucide-react';

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'default' | 'secondary' | 'outline' | 'ghost' | 'destructive' | 'accent';
  size?: 'sm' | 'md' | 'lg' | 'icon';
  loading?: boolean;
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  (
    {
      className,
      variant = 'default',
      size = 'md',
      loading = false,
      disabled,
      children,
      ...props
    },
    ref
  ) => {
    const baseStyles =
      'inline-flex items-center justify-center font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-background disabled:pointer-events-none disabled:opacity-50 select-none';

    const variants = {
      default:
        'bg-surface-secondary text-text-primary hover:bg-surface-hover border border-surface-border shadow-sm',
      secondary:
        'bg-surface-hover text-text-primary hover:bg-surface-border border border-surface-border-subtle',
      accent:
        'bg-accent text-white hover:bg-accent-hover shadow-sm font-semibold',
      outline:
        'bg-transparent text-text-primary hover:bg-surface border border-surface-border-subtle',
      ghost:
        'bg-transparent text-text-secondary hover:text-text-primary hover:bg-surface-hover',
      destructive:
        'bg-red-500/10 text-red-400 hover:bg-red-500/20 border border-red-500/30',
    };

    const sizes = {
      sm: 'h-8 px-3 text-xs rounded-md gap-1.5',
      md: 'h-9 px-4 text-sm rounded-md gap-2',
      lg: 'h-11 px-6 text-base rounded-lg gap-2.5',
      icon: 'h-9 w-9 rounded-md',
    };

    return (
      <button
        ref={ref}
        disabled={disabled || loading}
        className={cn(baseStyles, variants[variant], sizes[size], className)}
        {...props}
      >
        {loading && <Loader2 className="w-4 h-4 animate-spin shrink-0" />}
        {children}
      </button>
    );
  }
);

Button.displayName = 'Button';
