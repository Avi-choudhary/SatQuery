import * as React from 'react';
import { cn } from '../../lib/utils';

type BadgeVariant = 'default' | 'success' | 'warning' | 'error' | 'info' | 'violet' | 'quiet';

interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: BadgeVariant;
  /** Renders a small leading status dot in the badge colour. */
  dot?: boolean;
}

const VARIANTS: Record<BadgeVariant, string> = {
  default: 'bg-surface-3 text-ink-muted border-line-strong',
  quiet: 'bg-transparent text-ink-faint border-line',
  success: 'bg-ok/10 text-ok border-ok/25',
  warning: 'bg-amber/10 text-amber border-amber/25',
  error: 'bg-danger/10 text-danger border-danger/25',
  info: 'bg-accent/10 text-accent border-accent/25',
  violet: 'bg-violet/10 text-violet border-violet/25',
};

export const Badge = React.forwardRef<HTMLSpanElement, BadgeProps>(
  ({ className, variant = 'default', dot, children, ...props }, ref) => (
    <span
      ref={ref}
      className={cn(
        'inline-flex items-center gap-1.5 rounded-md border px-2 py-[3px]',
        'font-mono text-[10px] leading-none tracking-wide whitespace-nowrap',
        VARIANTS[variant],
        className
      )}
      {...props}
    >
      {dot && <span className="h-1.5 w-1.5 rounded-full bg-current" aria-hidden />}
      {children}
    </span>
  )
);

Badge.displayName = 'Badge';
