import { Component, ErrorInfo, ReactNode } from 'react';
import { AlertTriangle, RotateCcw } from 'lucide-react';
import { Button } from '@/components/ui/Button';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null,
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('Uncaught workstation rendering error:', error, errorInfo);
  }

  private handleReload = () => {
    window.location.reload();
  };

  private handleReset = () => {
    this.setState({ hasError: false, error: null });
  };

  public render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <div
          role="alert"
          aria-live="assertive"
          className="min-h-[50vh] flex flex-col items-center justify-center p-8 text-center"
        >
          <div className="max-w-md w-full p-6 rounded-xl bg-surface border border-red-500/30 space-y-4 shadow-xl">
            <div className="p-3 bg-red-500/10 rounded-xl border border-red-500/20 w-fit mx-auto text-red-400">
              <AlertTriangle className="w-8 h-8" />
            </div>

            <div className="space-y-1.5">
              <h3 className="text-base font-bold text-text-primary tracking-tight">
                Workstation Display Interrupted
              </h3>
              <p className="text-xs text-text-secondary leading-relaxed">
                An unexpected exception occurred while rendering this research component.
              </p>
            </div>

            {this.state.error && (
              <div className="p-3 rounded-lg bg-surface-secondary/70 border border-surface-border text-left font-mono text-[11px] text-red-300 overflow-x-auto max-h-32">
                {this.state.error.message}
              </div>
            )}

            <div className="flex items-center justify-center gap-3 pt-2">
              <Button
                variant="outline"
                size="sm"
                onClick={this.handleReset}
                className="text-xs font-mono"
              >
                Reset Component
              </Button>
              <Button
                variant="accent"
                size="sm"
                onClick={this.handleReload}
                className="text-xs font-mono"
              >
                <RotateCcw className="w-3.5 h-3.5 mr-1.5" />
                Reload Application
              </Button>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
