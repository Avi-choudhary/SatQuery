import * as React from 'react';
import { cn } from '../../lib/utils';

interface PanelProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: 'default' | 'glass';
  title?: string;
  headerAction?: React.ReactNode;
}

export const Panel = React.forwardRef<HTMLDivElement, PanelProps>(
  ({ className, variant = 'default', title, headerAction, children, ...props }, ref) => {
    const variants = {
      default: 'bg-space-navy border border-white/10',
      glass: 'bg-space-navy/70 backdrop-blur-md border border-white/10 shadow-2xl',
    };

    return (
      <div
        ref={ref}
        className={cn('rounded-xl overflow-hidden', variants[variant], className)}
        {...props}
      >
        {title && (
          <div className="px-4 py-3 border-b border-white/10 bg-white/5 flex items-center justify-between">
            <h3 className="text-xs font-mono font-medium text-white/80 uppercase tracking-wider">{title}</h3>
            {headerAction}
          </div>
        )}
        {children}
      </div>
    );
  }
);

Panel.displayName = 'Panel';
