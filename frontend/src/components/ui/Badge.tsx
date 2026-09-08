import * as React from 'react';
import { cn } from '../../lib/utils';

interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: 'default' | 'success' | 'warning' | 'error' | 'info';
}

export const Badge = React.forwardRef<HTMLSpanElement, BadgeProps>(
  ({ className, variant = 'default', ...props }, ref) => {
    const variants = {
      default: 'bg-white/10 text-white border border-white/20',
      success: 'bg-accent-teal/20 text-accent-teal border border-accent-teal/30',
      warning: 'bg-accent-warm/20 text-accent-warm border border-accent-warm/30',
      error: 'bg-red-500/20 text-red-500 border border-red-500/30',
      info: 'bg-accent-cyan/20 text-accent-cyan border border-accent-cyan/30',
    };

    return (
      <span
        ref={ref}
        className={cn('inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium', variants[variant], className)}
        {...props}
      />
    );
  }
);

Badge.displayName = 'Badge';
