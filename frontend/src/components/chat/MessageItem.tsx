import React from 'react';
import { motion } from 'framer-motion';
import { AlertTriangle, Radar, RotateCw, Terminal } from 'lucide-react';
import { Badge } from '../ui/Badge';
import { ThinkingDots } from '../ui/Feedback';
import { EvidenceCard } from './EvidenceCard';
import { useApp } from '../../context/AppState';
import { formatClock, formatDuration } from '../../lib/trace';
import type { ChatMessage } from '../../lib/types';
import { cn } from '../../lib/utils';

/** Distinct tools the backend actually reported for this turn. */
function toolsUsed(message: ChatMessage): string[] {
  const names = (message.trace ?? [])
    .map((step) => step.tool)
    .filter((name): name is string => Boolean(name));
  return Array.from(new Set(names));
}

const UserMessage: React.FC<{ message: ChatMessage }> = ({ message }) => (
  <div className="flex justify-end">
    <div className="max-w-[85%] rounded-2xl rounded-br-md border border-accent/25 bg-accent/10 px-3.5 py-2.5">
      <p className="whitespace-pre-wrap text-[14px] leading-relaxed text-ink">{message.content}</p>
      {message.datasetName && (
        <p className="mt-1.5 truncate font-mono text-[10px] text-ink-faint" title={message.datasetName}>
          on {message.datasetName}
        </p>
      )}
    </div>
  </div>
);

const AssistantMessage: React.FC<{ message: ChatMessage }> = ({ message }) => {
  const { retryMessage, openDock, isBusy } = useApp();
  const tools = toolsUsed(message);
  const isPending = message.status === 'pending';
  const isError = message.status === 'error';

  return (
    <div className="flex gap-3">
      <div
        className={cn(
          'mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-lg border',
          isError
            ? 'border-danger/30 bg-danger/10 text-danger'
            : 'border-accent/25 bg-accent/10 text-accent'
        )}
        aria-hidden
      >
        {isError ? <AlertTriangle size={14} /> : <Radar size={14} className={cn(isPending && 'animate-pulse')} />}
      </div>

      <div className="min-w-0 flex-1">
        {isPending ? (
          <div className="flex items-center gap-2.5 py-1">
            <ThinkingDots />
            <span className="text-[13px] text-ink-faint">
              Routing through the agentic controller…
            </span>
          </div>
        ) : (
          <>
            <div
              className={cn(
                'whitespace-pre-wrap text-[14px] leading-[1.65]',
                isError ? 'text-danger' : 'text-ink'
              )}
            >
              {message.content}
            </div>

            {isError && message.errorDetail && (
              <p className="mt-2 rounded-lg border border-danger/25 bg-danger/10 px-3 py-2 font-mono text-[11px] leading-relaxed text-ink-muted">
                {message.errorDetail}
              </p>
            )}

            <EvidenceCard message={message} />

            {/* Footer: only facts the backend or the client actually measured. */}
            <div className="mt-2.5 flex flex-wrap items-center gap-x-3 gap-y-1.5 font-mono text-[10px] text-ink-faint">
              <span>{formatClock(message.timestamp)}</span>
              {message.elapsedMs !== undefined && (
                <>
                  <span aria-hidden>·</span>
                  <span>{formatDuration(message.elapsedMs)}</span>
                </>
              )}
              {tools.length > 0 && (
                <>
                  <span aria-hidden>·</span>
                  <span className="truncate text-ink-muted">{tools.join(' → ')}</span>
                </>
              )}
              {message.trace && message.trace.length > 0 && (
                <button
                  type="button"
                  onClick={() => openDock('trace')}
                  className="inline-flex items-center gap-1 rounded px-1 py-0.5 text-ink-faint transition-colors hover:text-accent cursor-pointer"
                >
                  <Terminal size={10} aria-hidden />
                  {message.trace.length} steps
                </button>
              )}
              {isError && message.sourceQuery && (
                <button
                  type="button"
                  onClick={() => retryMessage(message.id)}
                  disabled={isBusy}
                  className="inline-flex items-center gap-1 rounded px-1 py-0.5 text-accent transition-colors hover:text-accent/80 disabled:opacity-40 cursor-pointer"
                >
                  <RotateCw size={10} aria-hidden />
                  Retry
                </button>
              )}
              {message.status === 'cancelled' && <Badge variant="quiet">cancelled</Badge>}
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export const MessageItem: React.FC<{ message: ChatMessage }> = ({ message }) => (
  <motion.div
    initial={{ opacity: 0, y: 6 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ duration: 0.18 }}
  >
    {message.role === 'user' ? (
      <UserMessage message={message} />
    ) : (
      <AssistantMessage message={message} />
    )}
  </motion.div>
);

export default MessageItem;
