import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { cn } from '../../lib/utils';
import { PieChart } from '../ui/PieChart';

interface MarkdownContentProps {
  content: string;
  className?: string;
  isError?: boolean;
}

export const MarkdownContent: React.FC<MarkdownContentProps> = ({
  content,
  className,
  isError = false,
}) => {
  if (isError) {
    return (
      <div className={cn('whitespace-pre-wrap text-[14px] leading-[1.65] text-danger', className)}>
        {content}
      </div>
    );
  }

  return (
    <div
      className={cn(
        'markdown-root text-[14px] leading-[1.65] text-ink max-w-full overflow-hidden',
        className
      )}
    >
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          table: ({ node, ...props }) => (
            <div className="my-3 w-full overflow-x-auto rounded-xl border border-line bg-surface-2/70 shadow-sm scrollbar-slim">
              <table className="w-full text-left text-xs border-collapse font-sans" {...props} />
            </div>
          ),
          thead: ({ node, ...props }) => (
            <thead className="border-b border-line bg-surface-3/90 text-ink font-semibold" {...props} />
          ),
          th: ({ node, ...props }) => (
            <th
              className="px-3.5 py-2.5 text-[11px] font-semibold uppercase tracking-wider text-ink"
              {...props}
            />
          ),
          tbody: ({ node, ...props }) => (
            <tbody className="divide-y divide-line/60" {...props} />
          ),
          tr: ({ node, ...props }) => (
            <tr className="hover:bg-surface-3/40 transition-colors" {...props} />
          ),
          td: ({ node, ...props }) => (
            <td className="px-3.5 py-2 text-ink leading-relaxed" {...props} />
          ),
          p: ({ node, ...props }) => (
            <p className="mb-2 last:mb-0 leading-[1.65]" {...props} />
          ),
          strong: ({ node, ...props }) => (
            <strong className="font-semibold text-ink" {...props} />
          ),
          em: ({ node, ...props }) => (
            <em className="italic text-ink/90" {...props} />
          ),
          h1: ({ node, ...props }) => (
            <h1 className="mb-2 mt-4 text-base font-bold text-ink first:mt-0" {...props} />
          ),
          h2: ({ node, ...props }) => (
            <h2 className="mb-2 mt-3.5 text-[15px] font-bold text-ink first:mt-0" {...props} />
          ),
          h3: ({ node, ...props }) => (
            <h3 className="mb-1.5 mt-3 text-[14px] font-semibold text-ink first:mt-0" {...props} />
          ),
          h4: ({ node, ...props }) => (
            <h4 className="mb-1 mt-2.5 text-[13px] font-semibold text-ink first:mt-0" {...props} />
          ),
          ul: ({ node, ...props }) => (
            <ul className="my-2 ml-4 list-disc space-y-1 pl-1 text-ink" {...props} />
          ),
          ol: ({ node, ...props }) => (
            <ol className="my-2 ml-4 list-decimal space-y-1 pl-1 text-ink" {...props} />
          ),
          li: ({ node, ...props }) => (
            <li className="leading-relaxed pl-0.5" {...props} />
          ),
          code: ({ node, className: codeClassName, children, ...props }: any) => {
            const match = /language-(\w+)/.exec(codeClassName || '');
            const isBlock = Boolean(match) || (typeof children === 'string' && children.includes('\n'));
            
            if (match && match[1] === 'pie') {
              try {
                const data = JSON.parse(String(children));
                return <PieChart data={data} />;
              } catch (e) {
                return (
                  <div className="my-2.5 rounded-xl border border-danger/25 bg-danger/10 p-3 text-[12px] text-danger">
                    Failed to render pie chart: Invalid JSON data.
                  </div>
                );
              }
            }

            if (isBlock) {
              return (
                <div className="my-2.5 overflow-x-auto rounded-xl border border-line bg-surface-3 p-3 font-mono text-[12px] text-ink scrollbar-slim">
                  <pre className="m-0 bg-transparent p-0">
                    <code className={codeClassName} {...props}>
                      {children}
                    </code>
                  </pre>
                </div>
              );
            }
            return (
              <code
                className="rounded-md border border-line/50 bg-surface-3 px-1.5 py-0.5 font-mono text-[12px] text-accent"
                {...props}
              >
                {children}
              </code>
            );
          },
          blockquote: ({ node, ...props }) => (
            <blockquote
              className="my-2.5 rounded-r-lg border-l-2 border-accent/70 bg-accent/5 py-1 pl-3.5 pr-2 italic text-ink-muted"
              {...props}
            />
          ),
          a: ({ node, href, ...props }) => (
            <a
              href={href}
              target="_blank"
              rel="noopener noreferrer"
              className="text-accent underline underline-offset-2 transition-colors hover:text-accent/80"
              {...props}
            />
          ),
          hr: ({ node, ...props }) => (
            <hr className="my-3 border-line" {...props} />
          ),
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
};

export default MarkdownContent;
