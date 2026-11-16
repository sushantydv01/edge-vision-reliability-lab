import React, { forwardRef } from 'react';
import { cn } from '@/lib/utils';

export interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  helperText?: string;
  error?: string;
}

export const Input = forwardRef<HTMLInputElement, InputProps>(
  ({ className, label, helperText, error, id, ...props }, ref) => {
    const inputId = id || (label ? label.toLowerCase().replace(/\s+/g, '-') : undefined);

    return (
      <div className="flex flex-col space-y-1.5 w-full">
        {label && (
          <label htmlFor={inputId} className="text-xs font-medium text-text-secondary">
            {label}
          </label>
        )}
        <input
          id={inputId}
          ref={ref}
          className={cn(
            'w-full bg-surface-secondary border rounded-md px-3 py-2 text-xs text-text-primary transition-colors',
            'border-surface-border placeholder:text-text-muted focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent',
            'disabled:opacity-50 disabled:cursor-not-allowed',
            error && 'border-red-500 focus:border-red-500 focus:ring-red-500',
            className
          )}
          {...props}
        />
        {helperText && !error && (
          <span className="text-[11px] text-text-muted">{helperText}</span>
        )}
        {error && <span className="text-[11px] text-red-400 font-medium">{error}</span>}
      </div>
    );
  }
);

Input.displayName = 'Input';
