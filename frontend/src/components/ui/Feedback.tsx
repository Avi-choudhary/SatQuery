import React from 'react';
import { AlertTriangle, Loader2 } from 'lucide-react';
import { cn } from '../../lib/utils';

interface EmptyStateProps {
  icon?: React.ReactNode;
  title: string;
  description?: React.ReactNode;
  action?: React.ReactNode;
  className?: string;
}

export const EmptyState: React.FC<EmptyStateProps> = ({
  icon,
  title,
  description,
  action,
  className,
}) => (
  <div
    className={cn(
      'flex flex-col items-center justify-center gap-3 px-6 py-12 text-center',
      className
    )}
  >
    {icon && (
      <div className="flex h-11 w-11 items-center justify-center rounded-xl border border-line bg-surface-2 text-ink-faint">
        {icon}
      </div>
    )}
    <div className="space-y-1.5">
      <p className="text-sm font-medium text-ink">{title}</p>
      {description && (
        <p className="mx-auto max-w-sm text-[13px] leading-relaxed text-ink-faint">{description}</p>
      )}
    </div>
    {action}
  </div>
);

interface InlineAlertProps {
  tone?: 'error' | 'warning' | 'info';
  title: string;
  children?: React.ReactNode;
  action?: React.ReactNode;
  className?: string;
}

const TONES = {
  error: 'border-danger/30 bg-danger/10 text-danger',
  warning: 'border-amber/30 bg-amber/10 text-amber',
  info: 'border-accent/30 bg-accent/10 text-accent',
} as const;

export const InlineAlert: React.FC<InlineAlertProps> = ({
  tone = 'error',
  title,
  children,
  action,
  className,
}) => (
  <div className={cn('rounded-lg border px-3 py-2.5', TONES[tone], className)} role="alert">
    <div className="flex items-start gap-2.5">
      <AlertTriangle size={14} className="mt-0.5 shrink-0" aria-hidden />
      <div className="min-w-0 flex-1 space-y-1">
        <p className="text-[13px] font-medium leading-snug">{title}</p>
        {children && <div className="text-xs leading-relaxed text-ink-muted">{children}</div>}
        {action && <div className="pt-1">{action}</div>}
      </div>
    </div>
  </div>
);

export const Spinner: React.FC<{ size?: number; className?: string }> = ({
  size = 14,
  className,
}) => <Loader2 size={size} className={cn('animate-spin', className)} aria-hidden />;

/** Three-dot "assistant is working" indicator. */
export const ThinkingDots: React.FC<{ className?: string }> = ({ className }) => (
  <span className={cn('inline-flex items-center gap-1', className)} aria-hidden>
    {[0, 1, 2].map((i) => (
      <span
        key={i}
        className="thinking-dot h-1.5 w-1.5 rounded-full bg-accent"
        style={{ animationDelay: `${i * 0.16}s` }}
      />
    ))}
  </span>
);
