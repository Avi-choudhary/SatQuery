import React from 'react';
import { cn } from '../../lib/utils';

interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: 'default' | 'glass' | 'raised';
  interactive?: boolean;
}

const VARIANTS = {
  default: 'bg-surface border border-line',
  glass: 'bg-surface-2/60 backdrop-blur-lg border border-line',
  raised: 'bg-surface-2 border border-line-strong',
} as const;

export const Card = React.forwardRef<HTMLDivElement, CardProps>(
  ({ className, variant = 'default', interactive, ...props }, ref) => (
    <div
      ref={ref}
      className={cn(
        'rounded-panel p-5 transition-colors duration-150',
        VARIANTS[variant],
        interactive && 'cursor-pointer hover:border-accent/35 hover:bg-surface-3/60',
        className
      )}
      {...props}
    />
  )
);

Card.displayName = 'Card';
