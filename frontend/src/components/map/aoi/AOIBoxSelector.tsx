import React, { useState, useRef, useCallback, useEffect } from 'react';
import type * as maplibregl from 'maplibre-gl';
import { Crosshair, Crop, AlertTriangle, X, Maximize2 } from 'lucide-react';

interface AOIBoxSelectorProps {
  map: maplibregl.Map | null;
  isActive: boolean;
  onCancel: () => void;
  onSelectBbox: (bbox: [number, number, number, number], areaKm2: number) => void;
}

const MAX_AREA_KM2 = 2500;

function calculateAreaKm2(minLon: number, minLat: number, maxLon: number, maxLat: number): number {
  const midLat = (minLat + maxLat) / 2.0;
  const latDist = Math.abs(maxLat - minLat) * 111.32;
  const lonDist = Math.abs(maxLon - minLon) * 111.32 * Math.cos((midLat * Math.PI) / 180);
  return Math.round(latDist * lonDist * 10) / 10;
}

export const AOIBoxSelector: React.FC<AOIBoxSelectorProps> = ({
  map,
  isActive,
  onCancel,
  onSelectBbox,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [startPoint, setStartPoint] = useState<{ x: number; y: number } | null>(null);
  const [currentPoint, setCurrentPoint] = useState<{ x: number; y: number } | null>(null);

  // Disable default map drag pan when drawing tool is active
  useEffect(() => {
    if (!map) return;
    if (isActive) {
      map.dragPan.disable();
    } else {
      map.dragPan.enable();
      setIsDragging(false);
      setStartPoint(null);
      setCurrentPoint(null);
    }
    return () => {
      if (map) map.dragPan.enable();
    };
  }, [map, isActive]);

  const handlePointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    if (e.button !== 0 || !containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    setStartPoint({ x, y });
    setCurrentPoint({ x, y });
    setIsDragging(true);
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
  };

  const handlePointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!isDragging || !containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(rect.width, e.clientX - rect.left));
    const y = Math.max(0, Math.min(rect.height, e.clientY - rect.top));
    setCurrentPoint({ x, y });
  };

  const handlePointerUp = () => {
    if (!isDragging || !startPoint || !currentPoint || !map) {
      setIsDragging(false);
      return;
    }
    setIsDragging(false);

    const minX = Math.min(startPoint.x, currentPoint.x);
    const maxX = Math.max(startPoint.x, currentPoint.x);
    const minY = Math.min(startPoint.y, currentPoint.y);
    const maxY = Math.max(startPoint.y, currentPoint.y);

    // Filter out accidental tiny clicks
    if (maxX - minX < 15 || maxY - minY < 15) {
      setStartPoint(null);
      setCurrentPoint(null);
      return;
    }

    // Convert screen pixels to geographic WGS84 coordinates
    const nw = map.unproject([minX, minY]);
    const se = map.unproject([maxX, maxY]);

    const minLon = Math.min(nw.lng, se.lng);
    const maxLon = Math.max(nw.lng, se.lng);
    const minLat = Math.min(nw.lat, se.lat);
    const maxLat = Math.max(nw.lat, se.lat);

    const area = calculateAreaKm2(minLon, minLat, maxLon, maxLat);
    if (area > MAX_AREA_KM2) {
      return; // Box remains visible with error message
    }

    onSelectBbox(
      [
        parseFloat(minLon.toFixed(5)),
        parseFloat(minLat.toFixed(5)),
        parseFloat(maxLon.toFixed(5)),
        parseFloat(maxLat.toFixed(5)),
      ],
      area
    );
  };

  const handleUseViewport = useCallback(() => {
    if (!map) return;
    const bounds = map.getBounds();
    const minLon = parseFloat(bounds.getWest().toFixed(5));
    const minLat = parseFloat(bounds.getSouth().toFixed(5));
    const maxLon = parseFloat(bounds.getEast().toFixed(5));
    const maxLat = parseFloat(bounds.getNorth().toFixed(5));
    const area = calculateAreaKm2(minLon, minLat, maxLon, maxLat);

    onSelectBbox([minLon, minLat, maxLon, maxLat], area);
  }, [map, onSelectBbox]);

  if (!isActive) return null;

  // Box geometry in screen pixels
  let boxStyle: React.CSSProperties | null = null;
  let areaPreview = 0;
  let isOversized = false;

  if (startPoint && currentPoint && map) {
    const left = Math.min(startPoint.x, currentPoint.x);
    const top = Math.min(startPoint.y, currentPoint.y);
    const width = Math.abs(currentPoint.x - startPoint.x);
    const height = Math.abs(currentPoint.y - startPoint.y);

    boxStyle = { left, top, width, height };

    const nw = map.unproject([left, top]);
    const se = map.unproject([left + width, top + height]);
    areaPreview = calculateAreaKm2(nw.lng, se.lat, se.lng, nw.lat);
    isOversized = areaPreview > MAX_AREA_KM2;
  }

  return (
    <div
      ref={containerRef}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      className="absolute inset-0 z-30 cursor-crosshair select-none overflow-hidden touch-none"
    >
      {/* Subtle darkened backdrop while drawing */}
      <div className="absolute inset-0 bg-space-black/20 pointer-events-none" />

      {/* Top Banner Guide */}
      <div className="absolute top-3 left-1/2 -translate-x-1/2 z-40 pointer-events-auto flex items-center gap-2 rounded-xl border border-accent/40 bg-surface/95 px-3.5 py-1.5 shadow-2xl backdrop-blur-md animate-in fade-in zoom-in-95 duration-200">
        <div className="flex items-center gap-2 text-xs font-mono text-accent">
          <Crosshair size={14} className="animate-pulse" />
          <span className="font-semibold uppercase tracking-wider">AOI SELECTION MODE</span>
        </div>
        <div className="h-3 w-px bg-white/20" />
        <span className="text-[11px] text-ink-muted">Drag on map to enclose target area</span>

        <button
          type="button"
          onClick={handleUseViewport}
          className="ml-2 inline-flex items-center gap-1 rounded-lg border border-line bg-surface-2 px-2 py-0.5 text-[10px] font-mono text-ink hover:border-accent/50 hover:text-accent transition-colors cursor-pointer"
          title="Select the entire currently visible map bounds"
        >
          <Maximize2 size={11} />
          <span>Use Viewport</span>
        </button>

        <button
          type="button"
          onClick={onCancel}
          className="ml-1 rounded p-0.5 text-ink-faint hover:text-ink transition-colors cursor-pointer"
          title="Cancel AOI selection"
        >
          <X size={14} />
        </button>
      </div>

      {/* Interactive Selection Box */}
      {boxStyle && (
        <div
          style={boxStyle}
          className={`absolute pointer-events-none border-2 transition-colors duration-75 ${
            isOversized
              ? 'border-danger bg-danger/15 shadow-[0_0_15px_rgba(255,59,48,0.4)]'
              : 'border-accent bg-accent/15 shadow-[0_0_20px_rgba(0,242,255,0.35)]'
          }`}
        >
          {/* Corner tick marks */}
          <div className="absolute -top-1 -left-1 w-2.5 h-2.5 border-t-2 border-l-2 border-white" />
          <div className="absolute -top-1 -right-1 w-2.5 h-2.5 border-t-2 border-r-2 border-white" />
          <div className="absolute -bottom-1 -left-1 w-2.5 h-2.5 border-b-2 border-l-2 border-white" />
          <div className="absolute -bottom-1 -right-1 w-2.5 h-2.5 border-b-2 border-r-2 border-white" />

          {/* Area & Coordinate HUD Tag */}
          <div className="absolute bottom-2 left-2 pointer-events-none">
            <div
              className={`flex items-center gap-1.5 rounded px-2 py-0.5 text-[10px] font-mono backdrop-blur-md border ${
                isOversized
                  ? 'bg-danger/90 text-white border-danger'
                  : 'bg-surface/90 text-accent border-accent/40'
              }`}
            >
              {isOversized ? <AlertTriangle size={11} /> : <Crop size={11} />}
              <span>{areaPreview} km²</span>
              {isOversized && <span className="font-bold text-[9px]">(Max: {MAX_AREA_KM2} km²)</span>}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
