import React from 'react';
import { AlertTriangle, CheckCircle2, ChevronRight, Cpu } from 'lucide-react';
import { EmptyState } from '../ui/Feedback';
import { useApp } from '../../context/AppState';
import { STAGE_ACCENTS, STAGE_LABELS } from '../../lib/trace';
import { describeCompute } from '../../lib/api';
import type { TraceStep } from '../../lib/types';
import { cn } from '../../lib/utils';

const STATUS_ICON: Record<TraceStep['status'], React.ReactNode> = {
  success: <CheckCircle2 size={12} className="text-ok" aria-hidden />,
  warning: <AlertTriangle size={12} className="text-amber" aria-hidden />,
  error: <AlertTriangle size={12} className="text-danger" aria-hidden />,
};

const StepRow: React.FC<{ step: TraceStep; index: number; last: boolean }> = ({
  step,
  index,
  last,
}) => {
  const { selectedStep, setSelectedStep } = useApp();
  const expanded = selectedStep?.id === step.id;

  return (
    <li className="relative pl-6">
      {/* Rail connecting the steps into one visible pipeline. */}
      {!last && <span className="absolute left-[7px] top-5 h-full w-px bg-line" aria-hidden />}
      <span
        className={cn(
          'absolute left-[3px] top-[7px] h-2 w-2 rounded-full ring-4 ring-surface',
          step.status === 'error'
            ? 'bg-danger'
            : step.status === 'warning'
              ? 'bg-amber'
              : 'bg-accent'
        )}
        aria-hidden
      />

      <button
        type="button"
        onClick={() => setSelectedStep(expanded ? null : step)}
        aria-expanded={expanded}
        className="w-full rounded-lg px-2 py-1.5 text-left transition-colors hover:bg-surface-2 cursor-pointer"
      >
        <div className="flex items-center gap-2">
          <span className={cn('label-caps', STAGE_ACCENTS[step.stage])}>
            {STAGE_LABELS[step.stage]}
          </span>
          <span className="font-mono text-[10px] text-ink-faint">#{index + 1}</span>
          {STATUS_ICON[step.status]}
          {step.confidence !== undefined && (
            <span className="ml-auto font-mono text-[10px] text-ink-muted">
              {(step.confidence * 100).toFixed(1)}%
            </span>
          )}
          <ChevronRight
            size={13}
            className={cn(
              'text-ink-faint transition-transform',
              step.confidence === undefined && 'ml-auto',
              expanded && 'rotate-90'
            )}
            aria-hidden
          />
        </div>

        <p className="mt-1 text-[12.5px] leading-snug text-ink">{step.action}</p>

        {step.tool && (
          <p className="mt-0.5 flex items-center gap-1 font-mono text-[10px] text-ink-faint">
            <Cpu size={9} aria-hidden />
            {step.tool}
          </p>
        )}
      </button>

      {expanded && (
        <pre className="mx-2 mb-2 mt-1 overflow-x-auto whitespace-pre-wrap break-words rounded-lg border border-line bg-ground px-2.5 py-2 font-mono text-[10.5px] leading-relaxed text-ink-muted">
          {step.raw}
        </pre>
      )}
    </li>
  );
};

/**
 * The backend's execution log, rendered as a pipeline. Everything here comes
 * from `execution_trace.steps`; nothing is synthesised.
 */
export const TracePanel: React.FC = () => {
  const { activeTrace, backendStatus, backendPhase } = useApp();

  if (activeTrace.length === 0) {
    return (
      <EmptyState
        icon={<Cpu size={18} />}
        title="No trace yet"
        description="Ask a question and the agentic controller's execution log will appear here, step by step."
      />
    );
  }

  return (
    <div className="scrollbar-slim flex min-h-0 flex-1 flex-col overflow-y-auto">
      <ol className="flex-1 space-y-0.5 p-3">
        {activeTrace.map((step, index) => (
          <StepRow
            key={step.id}
            step={step}
            index={index}
            last={index === activeTrace.length - 1}
          />
        ))}
      </ol>

      <footer className="sticky bottom-0 flex items-center justify-between gap-3 border-t border-line bg-surface/95 px-3 py-2 backdrop-blur">
        <span className="label-caps text-ink-faint">
          {activeTrace.length} step{activeTrace.length === 1 ? '' : 's'}
        </span>
        <span className="truncate font-mono text-[10px] text-ink-muted">
          {backendPhase === 'online' ? describeCompute(backendStatus) : 'compute unknown'}
        </span>
      </footer>
    </div>
  );
};

export default TracePanel;
