import React from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Bot,
  ArrowRight,
  X,
  CheckCircle2,
  Calendar,
} from 'lucide-react';
import { useApp } from '../../../context/AppState';
import {
  type FetchAOIResponse,
  aoiResponseToSceneDataset,
  aoiResponseToSceneOverlay,
} from '../../../lib/interactiveMapApi';

interface AOIChatbotHandoffProps {
  scene: FetchAOIResponse | null;
  onDismiss: () => void;
}

export const AOIChatbotHandoff: React.FC<AOIChatbotHandoffProps> = ({ scene, onDismiss }) => {
  const navigate = useNavigate();
  const { setScene, openDock } = useApp();

  if (!scene) return null;

  const handleHandoff = () => {
    const dataset = aoiResponseToSceneDataset(scene);
    const overlay = aoiResponseToSceneOverlay(scene);

    // Update global state with the authentic fetched scene
    setScene(dataset, overlay);

    // Switch to Workspace view for conversation with the AI model
    navigate('/console');
    openDock('map');
  };

  const formattedDate = (() => {
    if (!scene.acquisition_date) return 'Recent Pass';
    const d = new Date(scene.acquisition_date);
    return isNaN(d.getTime())
      ? 'Recent Pass'
      : d.toLocaleDateString(undefined, {
          year: 'numeric',
          month: 'short',
          day: 'numeric',
        });
  })();

  return (
    <div className="absolute bottom-6 left-1/2 -translate-x-1/2 z-40 w-[92%] max-w-lg animate-in slide-in-from-bottom-4 duration-300 pointer-events-auto">
      <div className="rounded-2xl border border-accent/40 bg-surface/95 p-3.5 shadow-2xl backdrop-blur-xl text-ink font-sans">
        <div className="flex items-start justify-between gap-3">
          {/* Thumbnail Preview */}
          <div className="relative h-16 w-16 shrink-0 overflow-hidden rounded-xl border border-line bg-surface-3">
            <img
              src={scene.t1_image_url}
              alt="Fetched Scene Preview"
              className="h-full w-full object-cover"
              onError={(e) => {
                // Fallback to satellite icon if preview not yet rendered
                (e.currentTarget as HTMLElement).style.display = 'none';
              }}
            />
            <div className="absolute bottom-1 right-1 rounded bg-space-black/70 px-1 py-0.2 text-[8px] font-mono text-accent">
              10m
            </div>
          </div>

          {/* Info Details */}
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-1.5 mb-1">
              <span className="flex items-center gap-1 text-[11px] font-semibold text-accent font-mono">
                <CheckCircle2 size={12} />
                SCENE STREAMED
              </span>
              <span className="text-line">•</span>
              <span className="text-[11px] text-ink-muted truncate font-mono">
                {scene.sensor ? scene.sensor.split(' ')[0] : 'Scene'}
              </span>
            </div>

            <div className="text-xs font-semibold text-ink truncate mb-1">
              {scene.name}
            </div>

            <div className="flex items-center gap-3 text-[10px] font-mono text-ink-faint">
              <span className="flex items-center gap-1">
                <Calendar size={11} />
                {formattedDate}
              </span>
              <span>{scene.area_sq_km} km²</span>
              {typeof scene.cloud_cover === 'number' && !isNaN(scene.cloud_cover) && (
                <span>Clouds: {scene.cloud_cover.toFixed(1)}%</span>
              )}
            </div>
          </div>

          {/* Close button */}
          <button
            type="button"
            onClick={onDismiss}
            className="rounded p-1 text-ink-faint hover:text-ink transition-colors cursor-pointer"
            title="Dismiss notification"
          >
            <X size={14} />
          </button>
        </div>

        {/* Action Button */}
        <div className="mt-3 pt-2.5 border-t border-line flex items-center justify-between gap-2">
          <span className="text-[11px] text-ink-muted hidden sm:inline">
            Load this real scene into the AI assistant:
          </span>
          <button
            type="button"
            onClick={handleHandoff}
            className="w-full sm:w-auto inline-flex items-center justify-center gap-2 rounded-xl bg-accent px-4 py-2 text-xs font-semibold text-space-black shadow-[0_0_15px_rgba(0,242,255,0.3)] hover:bg-accent/90 transition-all cursor-pointer"
          >
            <Bot size={14} />
            <span>Analyze with SatQuery AI</span>
            <ArrowRight size={13} />
          </button>
        </div>
      </div>
    </div>
  );
};
