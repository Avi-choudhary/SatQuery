import React from 'react';
import { Loader2 } from 'lucide-react';
import { cn } from '../../lib/utils';

type Variant = 'primary' | 'secondary' | 'ghost' | 'outline' | 'danger';
type Size = 'xs' | 'sm' | 'md' | 'lg';

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
  isLoading?: boolean;
}

const VARIANTS: Record<Variant, string> = {
  // `text-base` is Tailwind's font-size utility, so the ink colour for
  // on-accent text uses the explicit ground token instead.
  primary:
    'bg-accent text-space-black font-semibold hover:bg-accent/90 active:bg-accent/80 shadow-[0_1px_0_0_rgba(255,255,255,0.14)_inset]',
  secondary: 'bg-surface-3 text-ink hover:bg-surface-4 border border-line-strong',
  ghost: 'bg-transparent text-ink-muted hover:text-ink hover:bg-surface-3',
  outline: 'bg-transparent text-ink border border-line-strong hover:bg-surface-3 hover:border-accent/40',
  danger: 'bg-danger/10 text-danger border border-danger/30 hover:bg-danger/20',
};

const SIZES: Record<Size, string> = {
  xs: 'h-7 px-2.5 text-xs gap-1.5 rounded-md',
  sm: 'h-8 px-3 text-[13px] gap-1.5 rounded-lg',
  md: 'h-10 px-4 text-sm gap-2 rounded-lg',
  lg: 'h-12 px-6 text-[15px] gap-2 rounded-xl',
};

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = 'primary', size = 'md', isLoading, children, disabled, ...props }, ref) => (
    <button
      ref={ref}
      disabled={isLoading || disabled}
      className={cn(
        'inline-flex shrink-0 items-center justify-center whitespace-nowrap font-medium',
        'transition-colors duration-150 cursor-pointer select-none',
        'disabled:cursor-not-allowed disabled:opacity-45',
        VARIANTS[variant],
        SIZES[size],
        className
      )}
      {...props}
    >
      {isLoading && <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden />}
      {children}
    </button>
  )
);

Button.displayName = 'Button';

interface IconButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  label: string;
  active?: boolean;
  size?: 'sm' | 'md';
}

/**
 * Square icon-only control. `label` is required — these are used all over the
 * map chrome, where an unlabelled icon is unusable with a screen reader.
 */
export const IconButton = React.forwardRef<HTMLButtonElement, IconButtonProps>(
  ({ className, label, active, size = 'md', children, ...props }, ref) => (
    <button
      ref={ref}
      type="button"
      title={label}
      aria-label={label}
      aria-pressed={active}
      className={cn(
        'inline-flex items-center justify-center rounded-lg border transition-colors cursor-pointer',
        'disabled:cursor-not-allowed disabled:opacity-40',
        size === 'sm' ? 'h-7 w-7' : 'h-8 w-8',
        active
          ? 'bg-accent/15 text-accent border-accent/40'
          : 'bg-surface-2/80 text-ink-muted border-line hover:text-ink hover:bg-surface-3 hover:border-line-strong',
        className
      )}
      {...props}
    >
      {children}
    </button>
  )
);

IconButton.displayName = 'IconButton';
