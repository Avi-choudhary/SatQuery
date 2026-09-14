import React from 'react';
import { Layers3, Maximize2, Ruler, Shapes } from 'lucide-react';
import { useApp } from '../../context/AppState';
import type { ChatMessage } from '../../lib/types';
import { cn } from '../../lib/utils';

interface EvidenceCardProps {
  message: ChatMessage;
}

/**
 * The hinge between the conversation and the map: when an answer carries
 * geographic evidence, this sits inline in the thread and pushes those
 * features onto the scene dock rather than opening a separate map screen.
 */
export const EvidenceCard: React.FC<EvidenceCardProps> = ({ message }) => {
  const { focusEvidence, focusedMessageId } = useApp();
  const evidence = message.evidence;

  if (!evidence?.geoJson || evidence.featureCount === 0) return null;

  const isFocused = focusedMessageId === message.id;

  return (
    <button
      type="button"
      onClick={() => focusEvidence(message.id)}
      className={cn(
        'group mt-3 flex w-full items-center gap-3 rounded-xl border px-3 py-2.5 text-left transition-colors cursor-pointer',
        isFocused
          ? 'border-accent/45 bg-accent/10'
          : 'border-line bg-surface-2/70 hover:border-accent/35 hover:bg-surface-3'
      )}
      aria-label={`Show ${evidence.featureCount} detected regions on the map`}
    >
      <span
        className={cn(
          'flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border',
          isFocused ? 'border-accent/40 bg-accent/15 text-accent' : 'border-line bg-surface-3 text-accent'
        )}
        aria-hidden
      >
        <Layers3 size={16} />
      </span>

      <span className="min-w-0 flex-1">
        <span className="block text-[13px] font-medium text-ink">
          {evidence.featureCount === 1
            ? '1 region on the map'
            : `${evidence.featureCount} regions on the map`}
        </span>
        <span className="mt-0.5 flex flex-wrap items-center gap-x-3 gap-y-0.5 font-mono text-[10.5px] text-ink-faint">
          <span className="inline-flex items-center gap-1">
            <Shapes size={10} aria-hidden />
            GeoJSON · WGS84
          </span>
          {evidence.areaHa !== undefined && (
            <span className="inline-flex items-center gap-1 text-amber">
              <Ruler size={10} aria-hidden />
              {evidence.areaHa.toLocaleString()} ha
            </span>
          )}
        </span>
      </span>

      <span
        className={cn(
          'flex items-center gap-1.5 rounded-md px-2 py-1 font-mono text-[10px] uppercase tracking-wider transition-colors',
          isFocused
            ? 'text-accent'
            : 'text-ink-faint group-hover:text-accent'
        )}
      >
        <Maximize2 size={11} aria-hidden />
        {isFocused ? 'On map' : 'Show'}
      </span>
    </button>
  );
};

export default EvidenceCard;
