// TODO(backend): Connect to MapLibre GL JS GeoJSON Layer or Raster Tile Source.
// Expected Data Props / Backend Response:
//   - overlayGeoJSON: GeoJSON.FeatureCollection with properties {
//       detection_id: string,
//       category: 'urban_growth' | 'water_body' | 'deforestation' | 'infrastructure',
//       confidence: number (0.0 to 1.0),
//       area_sq_meters: number,
//       delta_percentage: number,
//       t1_spectral_signature: number[],
//       t2_spectral_signature: number[]
//     }
//   - rasterMaskUrl?: string (COG or XYZ tile template for high-resolution pixel-level heatmap)
//   - activeLayers: { showBoundingBoxes: boolean, showHeatmap: boolean, showChangeMask: boolean }

import { motion } from 'framer-motion';
import { mockMapOverlays, type MapOverlay } from '../../mocks/mapOverlays';

interface ChangeDetectionOverlayProps {
  showBoundingBoxes?: boolean;
  showHeatmap?: boolean;
  showChangeMask?: boolean;
  selectedOverlayId?: string | null;
  onSelectOverlay?: (overlay: MapOverlay) => void;
}

export const ChangeDetectionOverlay: React.FC<ChangeDetectionOverlayProps> = ({
  showBoundingBoxes = true,
  showHeatmap = true,
  showChangeMask = true,
  selectedOverlayId,
  onSelectOverlay,
}) => {
  return (
    <div className="absolute inset-0 pointer-events-none overflow-hidden">
      {/* 1. Heatmap overlay layer (simulated raster density) */}
      {showHeatmap && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 0.55 }}
          transition={{ duration: 0.6 }}
          className="absolute inset-0 pointer-events-none"
        >
          <div
            className="absolute rounded-full blur-2xl"
            style={{
              top: '24%',
              left: '32%',
              width: '260px',
              height: '190px',
              background: 'radial-gradient(circle, rgba(0, 242, 255, 0.45) 0%, rgba(0, 222, 194, 0.25) 50%, transparent 75%)',
            }}
          />
          <div
            className="absolute rounded-full blur-3xl"
            style={{
              top: '55%',
              left: '60%',
              width: '180px',
              height: '140px',
              background: 'radial-gradient(circle, rgba(255, 170, 0, 0.4) 0%, transparent 70%)',
            }}
          />
        </motion.div>
      )}

      {/* 2. Vector Bounding Boxes & Polygonal Change Regions */}
      {mockMapOverlays.map((overlay) => {
        if (overlay.type === 'heatmap' && !showHeatmap) return null;
        if (overlay.type === 'bounding-box' && !showBoundingBoxes) return null;
        if (overlay.type === 'change-mask' && !showChangeMask) return null;

        const isSelected = selectedOverlayId === overlay.id;

        return (
          <motion.div
            key={overlay.id}
            initial={{ scale: 0.95, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ duration: 0.4 }}
            onClick={(e) => {
              e.stopPropagation();
              onSelectOverlay?.(overlay);
            }}
            className="absolute pointer-events-auto cursor-pointer group"
            style={{
              top: overlay.type === 'change-mask' ? '22%' : overlay.type === 'bounding-box' ? '54%' : '38%',
              left: overlay.type === 'change-mask' ? '30%' : overlay.type === 'bounding-box' ? '58%' : '15%',
              width: `${overlay.coordinates.width}px`,
              height: `${overlay.coordinates.height}px`,
            }}
          >
            {/* Border & Area Fill */}
            <div
              className={`w-full h-full rounded-md border-2 transition-all ${
                isSelected
                  ? 'border-white shadow-[0_0_15px_rgba(0,242,255,0.8)]'
                  : overlay.type === 'change-mask'
                  ? 'border-accent-cyan bg-accent-cyan/15 group-hover:bg-accent-cyan/25'
                  : 'border-accent-warm bg-accent-warm/15 group-hover:bg-accent-warm/25'
              }`}
            >
              {/* Corner crosshairs */}
              <div className="absolute -top-1 -left-1 w-2.5 h-2.5 border-t-2 border-l-2 border-white" />
              <div className="absolute -top-1 -right-1 w-2.5 h-2.5 border-t-2 border-r-2 border-white" />
              <div className="absolute -bottom-1 -left-1 w-2.5 h-2.5 border-b-2 border-l-2 border-white" />
              <div className="absolute -bottom-1 -right-1 w-2.5 h-2.5 border-b-2 border-r-2 border-white" />

              {/* Tag / Pill */}
              <div className="absolute -top-6 left-0 flex items-center gap-1.5 px-2 py-0.5 rounded bg-space-black/90 border border-white/20 text-[10px] font-mono whitespace-nowrap shadow-lg">
                <div
                  className="w-1.5 h-1.5 rounded-full"
                  style={{ backgroundColor: overlay.style.strokeColor }}
                />
                <span className="text-white font-medium">{overlay.label}</span>
                <span className="text-accent-cyan">{(overlay.confidence * 100).toFixed(1)}%</span>
              </div>

              {/* Hover metric tooltip */}
              {overlay.metrics && (
                <div className="opacity-0 group-hover:opacity-100 transition-opacity absolute top-full left-0 mt-1 bg-space-navy/95 border border-white/20 p-2 rounded text-[10px] font-mono text-white/90 z-30 shadow-2xl pointer-events-none min-w-[170px]">
                  <div className="text-accent-cyan font-bold uppercase mb-0.5">{overlay.category}</div>
                  {overlay.metrics.areaHectares && (
                    <div>Area: {overlay.metrics.areaHectares} ha</div>
                  )}
                  {overlay.metrics.deltaPercentage && (
                    <div className="text-accent-warm">Δ Growth: +{overlay.metrics.deltaPercentage}%</div>
                  )}
                  {overlay.metrics.spectralShift && (
                    <div className="text-slate-400 text-[9px]">{overlay.metrics.spectralShift}</div>
                  )}
                </div>
              )}
            </div>
          </motion.div>
        );
      })}
    </div>
  );
};

export default ChangeDetectionOverlay;
