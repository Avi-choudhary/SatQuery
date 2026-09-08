import React from 'react';
import { cn } from '../../lib/utils';

interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: 'default' | 'glass';
}

export const Card = React.forwardRef<HTMLDivElement, CardProps>(
  ({ className, variant = 'default', ...props }, ref) => {
    const variants = {
      default: 'bg-space-navy border border-white/10 shadow-none',
      glass: 'bg-white/5 backdrop-blur-md border border-white/10 shadow-none',
    };

    return (
      <div
        ref={ref}
        className={cn('rounded-xl p-6 transition-all', variants[variant], className)}
        {...props}
      />
    );
  }
);

Card.displayName = 'Card';
