import React from 'react';
import { Link } from 'react-router-dom';
import { MessagesSquare } from 'lucide-react';
import MapViewport from '../components/map/MapViewport';
import { Badge } from '../components/ui/Badge';
import { useApp } from '../context/AppState';

/**
 * Full-bleed map, for when the imagery itself is the task. The workspace keeps
 * its own compact map in the dock; this is the escape hatch, not the default.
 */
const MapPage: React.FC = () => {
  const { dataset, overlay, focusedGeoJson, focusedOverlay } = useApp();

  return (
    <div className="flex h-full min-h-0 flex-col">
      <header className="flex h-11 shrink-0 items-center justify-between gap-3 border-b border-line bg-surface/60 px-4">
        <div className="flex min-w-0 items-center gap-2.5">
          <h1 className="text-[13px] font-medium text-ink">Map explorer</h1>
          {dataset ? (
            <Badge variant="info" dot>
              {dataset.name}
            </Badge>
          ) : (
            <Badge variant="quiet">no scene loaded</Badge>
          )}
        </div>

        <Link
          to="/console"
          className="inline-flex items-center gap-1.5 rounded-lg border border-line px-2.5 py-1.5 text-xs text-ink-muted transition-colors hover:border-accent/40 hover:text-accent"
        >
          <MessagesSquare size={13} />
          Back to conversation
        </Link>
      </header>

      <div className="min-h-0 flex-1">
        <MapViewport
          density="full"
          geoJsonData={focusedGeoJson}
          resultOverlay={focusedOverlay}
          datasetName={dataset?.name}
          sensor={dataset?.sensor}
          overlay={overlay}
        />
      </div>
    </div>
  );
};

export default MapPage;
