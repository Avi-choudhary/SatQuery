import { Component, type ErrorInfo, type ReactNode } from 'react';
import { AlertTriangle, RefreshCw } from 'lucide-react';

interface Props {
  children: ReactNode;
  fallbackTitle?: string;
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
    console.error('Uncaught error caught by ErrorBoundary:', error, errorInfo);
  }

  private handleReset = () => {
    this.setState({ hasError: false, error: null });
  };

  public render() {
    if (this.state.hasError) {
      return (
        <div className="flex h-full w-full min-h-[280px] flex-col items-center justify-center p-6 text-center bg-surface-2/40 border border-line rounded-2xl">
          <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-danger/15 text-danger mb-4">
            <AlertTriangle size={24} />
          </div>
          <h2 className="text-sm font-semibold text-ink mb-1">
            {this.props.fallbackTitle || 'Component Encountered an Issue'}
          </h2>
          <p className="text-xs text-ink-muted max-w-md mb-4 leading-relaxed font-mono">
            {this.state.error?.message || 'An unexpected rendering error occurred.'}
          </p>
          <div className="flex gap-2">
            <button
              type="button"
              onClick={this.handleReset}
              className="inline-flex items-center gap-1.5 rounded-lg bg-surface-3 border border-line px-3 py-1.5 text-xs text-ink hover:bg-surface-4 transition-colors cursor-pointer"
            >
              <RefreshCw size={13} />
              <span>Retry Component</span>
            </button>
            <button
              type="button"
              onClick={() => window.location.reload()}
              className="inline-flex items-center gap-1.5 rounded-lg bg-accent px-3 py-1.5 text-xs font-semibold text-space-black hover:bg-accent/90 transition-all cursor-pointer"
            >
              <span>Refresh Page</span>
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
