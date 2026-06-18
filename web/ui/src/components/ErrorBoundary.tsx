import { Component, type ReactNode } from 'react';

interface Props { children: ReactNode; label?: string; }
interface State { error: string | null; }

// Class component required — hooks cannot catch render errors.
export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(err: unknown): State {
    return { error: err instanceof Error ? `${err.message}\n\n${err.stack ?? ''}` : String(err) };
  }

  componentDidCatch(err: unknown, info: { componentStack: string }) {
    console.error('[ErrorBoundary]', err, info.componentStack);
  }

  render() {
    if (this.state.error) {
      return (
        <div className="flex flex-col items-center justify-center h-full gap-3 p-6 text-center">
          <div className="text-rose-500 text-xs font-bold uppercase tracking-widest">
            {this.props.label ?? 'Component error'}
          </div>
          <pre className="text-[10px] text-slate-400 bg-slate-950 p-3 rounded border border-slate-800 max-w-full overflow-auto whitespace-pre-wrap text-left">
            {this.state.error}
          </pre>
          <button
            onClick={() => this.setState({ error: null })}
            className="text-[10px] text-slate-500 hover:text-slate-300 underline"
          >
            retry
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}
