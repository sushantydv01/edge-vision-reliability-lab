import { cn } from '@/lib/utils';
import { AlertTriangle, RotateCcw } from 'lucide-react';
import { Button } from './Button';

export interface ErrorStateProps {
  title?: string;
  message: string;
  onRetry?: () => void;
  className?: string;
  errorType?: string;
}

export function ErrorState({
  title = 'Execution Error',
  message,
  onRetry,
  className,
  errorType,
}: ErrorStateProps) {
  return (
    <div
      role="alert"
      className={cn(
        'flex flex-col p-4 bg-red-950/20 border border-red-500/30 rounded-lg text-text-primary space-y-3',
        className
      )}
    >
      <div className="flex items-start gap-3">
        <div className="p-1.5 bg-red-500/10 rounded-md border border-red-500/20 shrink-0 mt-0.5">
          <AlertTriangle className="w-4 h-4 text-red-400" />
        </div>
        <div className="flex-1 space-y-1">
          <div className="flex items-center gap-2">
            <h4 className="text-xs font-semibold text-red-300 tracking-tight">{title}</h4>
            {errorType && (
              <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-red-500/10 text-red-400 border border-red-500/20">
                {errorType}
              </span>
            )}
          </div>
          <p className="text-xs text-text-secondary leading-relaxed font-mono">{message}</p>
        </div>
      </div>
      {onRetry && (
        <div className="flex justify-end pt-1">
          <Button
            size="sm"
            variant="outline"
            onClick={onRetry}
            className="border-red-500/30 text-red-300 hover:bg-red-500/10"
          >
            <RotateCcw className="w-3.5 h-3.5 mr-1.5" />
            Retry Request
          </Button>
        </div>
      )}
    </div>
  );
}
