import React, { useState } from 'react';
import {
  Satellite,
  Radar,
  CloudSun,
  Loader2,
  X,
  Sparkles,
  CheckCircle2,
  AlertCircle,
} from 'lucide-react';
import {
  fetchAOISatelliteScene,
  type FetchAOIResponse,
} from '../../../lib/interactiveMapApi';

interface AOIFetchModalProps {
  bbox: [number, number, number, number] | null;
  areaKm2: number;
  isOpen: boolean;
  onClose: () => void;
  onSuccess: (response: FetchAOIResponse) => void;
}

type SensorChoice = 'sentinel-2' | 'sentinel-1';

const STREAM_STEPS = [
  'Connecting to AWS STAC Catalog...',
  'Locating closest orbital pass...',
  'Streaming 10m COG pixels via HTTP Range Requests...',
  'Generating Web Mercator warped overlay...',
];

export const AOIFetchModal: React.FC<AOIFetchModalProps> = ({
  bbox,
  areaKm2,
  isOpen,
  onClose,
  onSuccess,
}) => {
  const [sensor, setSensor] = useState<SensorChoice>('sentinel-2');
  const [maxCloudCover, setMaxCloudCover] = useState<number>(20);
  const [isStreaming, setIsStreaming] = useState(false);
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const stepIntervalRef = React.useRef<ReturnType<typeof setInterval> | null>(null);

  React.useEffect(() => {
    return () => {
      if (stepIntervalRef.current) {
        clearInterval(stepIntervalRef.current);
      }
    };
  }, []);

  if (!isOpen || !bbox) return null;

  const [minLon, minLat, maxLon, maxLat] = bbox;

  const handleStartFetch = async () => {
    setIsStreaming(true);
    setError(null);
    setCurrentStepIndex(0);

    if (stepIntervalRef.current) clearInterval(stepIntervalRef.current);

    // Step animation timer for smooth user feedback
    stepIntervalRef.current = setInterval(() => {
      setCurrentStepIndex((prev) => (prev < STREAM_STEPS.length - 1 ? prev + 1 : prev));
    }, 2800);

    try {
      const response = await fetchAOISatelliteScene({
        bbox,
        sensor,
        max_cloud_cover: sensor === 'sentinel-2' ? maxCloudCover : undefined,
      });
      if (stepIntervalRef.current) clearInterval(stepIntervalRef.current);
      setIsStreaming(false);
      onSuccess(response);
    } catch (err: any) {
      if (stepIntervalRef.current) clearInterval(stepIntervalRef.current);
      setIsStreaming(false);
      setError(err?.message || 'Failed to stream satellite scene from STAC archive.');
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-space-black/60 backdrop-blur-sm animate-in fade-in duration-150">
      <div className="relative w-full max-w-md rounded-2xl border border-line bg-surface/95 p-5 shadow-2xl backdrop-blur-xl text-ink font-sans">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-line pb-3 mb-4">
          <div className="flex items-center gap-2">
            <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-accent/15 text-accent">
              <Satellite size={16} />
            </div>
            <div>
              <h3 className="text-sm font-semibold tracking-tight">Stream Satellite Imagery</h3>
              <p className="text-[11px] text-ink-muted">AWS Open Data STAC • Zero API Keys</p>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            disabled={isStreaming}
            className="rounded p-1 text-ink-faint hover:text-ink transition-colors cursor-pointer disabled:opacity-40"
          >
            <X size={16} />
          </button>
        </div>

        {/* AOI Details */}
        <div className="mb-4 rounded-xl border border-line bg-surface-2 p-3 text-[11px] font-mono">
          <div className="flex items-center justify-between text-ink-muted mb-1">
            <span>AREA OF INTEREST</span>
            <span className="text-accent font-bold">{areaKm2} km²</span>
          </div>
          <div className="text-ink-faint text-[10px] truncate">
            BBox: [{minLon}, {minLat}, {maxLon}, {maxLat}]
          </div>
        </div>

        {/* Sensor Selection Toggle */}
        <div className="mb-4">
          <label className="block text-[11px] font-medium text-ink-muted mb-2 uppercase tracking-wider font-mono">
            SELECT SENSOR
          </label>
          <div className="grid grid-cols-2 gap-2">
            <button
              type="button"
              disabled={isStreaming}
              onClick={() => setSensor('sentinel-2')}
              className={`flex flex-col items-start gap-1 p-3 rounded-xl border text-left transition-all cursor-pointer ${
                sensor === 'sentinel-2'
                  ? 'border-accent bg-accent/10 shadow-[0_0_12px_rgba(0,242,255,0.15)]'
                  : 'border-line bg-surface-2 hover:border-line-strong'
              }`}
            >
              <div className="flex items-center gap-1.5 text-xs font-semibold text-ink">
                <Satellite size={14} className={sensor === 'sentinel-2' ? 'text-accent' : 'text-ink-muted'} />
                <span>Sentinel-2</span>
              </div>
              <span className="text-[10px] text-ink-muted leading-tight">
                Optical • 10m True Color & NIR
              </span>
            </button>

            <button
              type="button"
              disabled={isStreaming}
              onClick={() => setSensor('sentinel-1')}
              className={`flex flex-col items-start gap-1 p-3 rounded-xl border text-left transition-all cursor-pointer ${
                sensor === 'sentinel-1'
                  ? 'border-accent-warm bg-accent-warm/10 shadow-[0_0_12px_rgba(255,170,0,0.15)]'
                  : 'border-line bg-surface-2 hover:border-line-strong'
              }`}
            >
              <div className="flex items-center gap-1.5 text-xs font-semibold text-ink">
                <Radar size={14} className={sensor === 'sentinel-1' ? 'text-accent-warm' : 'text-ink-muted'} />
                <span>Sentinel-1</span>
              </div>
              <span className="text-[10px] text-ink-muted leading-tight">
                SAR Radar • Day/Night All-Weather
              </span>
            </button>
          </div>
        </div>

        {/* Optical Cloud Cover Slider */}
        {sensor === 'sentinel-2' && (
          <div className="mb-4 rounded-xl border border-line bg-surface-2 p-3">
            <div className="flex items-center justify-between text-xs mb-1.5">
              <span className="text-[11px] text-ink-muted flex items-center gap-1.5">
                <CloudSun size={13} className="text-accent" />
                Max Cloud Coverage
              </span>
              <span className="font-mono text-xs font-bold text-ink">{maxCloudCover}%</span>
            </div>
            <input
              type="range"
              min={1}
              max={60}
              value={maxCloudCover}
              disabled={isStreaming}
              onChange={(e) => setMaxCloudCover(Number(e.target.value))}
              className="w-full h-1.5 bg-surface-3 rounded-lg appearance-none cursor-pointer accent-accent"
            />
            <div className="flex justify-between text-[9px] font-mono text-ink-faint mt-1">
              <span>Clear Sky (1%)</span>
              <span>Cloud Tolerant (60%)</span>
            </div>
          </div>
        )}

        {/* Error Banner */}
        {error && (
          <div className="mb-4 flex items-start gap-2 rounded-xl border border-danger/40 bg-danger/10 p-3 text-xs text-danger">
            <AlertCircle size={14} className="shrink-0 mt-0.5" />
            <div className="flex-1 text-[11px] leading-relaxed">{error}</div>
          </div>
        )}

        {/* Streaming Progress Status */}
        {isStreaming && (
          <div className="mb-4 rounded-xl border border-accent/30 bg-accent/5 p-3 font-mono text-xs">
            <div className="flex items-center gap-2 text-accent font-semibold mb-2">
              <Loader2 size={13} className="animate-spin" />
              <span>Streaming COG Pixels...</span>
            </div>
            <div className="space-y-1.5 text-[10px]">
              {STREAM_STEPS.map((step, idx) => {
                const isPassed = idx < currentStepIndex;
                const isCurrent = idx === currentStepIndex;
                return (
                  <div
                    key={step}
                    className={`flex items-center gap-2 ${
                      isPassed
                        ? 'text-accent'
                        : isCurrent
                        ? 'text-white font-bold animate-pulse'
                        : 'text-ink-faint'
                    }`}
                  >
                    {isPassed ? (
                      <CheckCircle2 size={11} className="text-accent" />
                    ) : (
                      <div className="h-1.5 w-1.5 rounded-full bg-current" />
                    )}
                    <span>{step}</span>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Action Buttons */}
        <div className="flex items-center justify-end gap-2 pt-2 border-t border-line">
          <button
            type="button"
            disabled={isStreaming}
            onClick={onClose}
            className="rounded-lg border border-line px-3 py-1.5 text-xs text-ink-muted hover:text-ink transition-colors cursor-pointer disabled:opacity-40"
          >
            Cancel
          </button>
          <button
            type="button"
            disabled={isStreaming}
            onClick={handleStartFetch}
            className="inline-flex items-center gap-1.5 rounded-lg bg-accent px-4 py-1.5 text-xs font-semibold text-space-black shadow-lg hover:bg-accent/90 transition-all cursor-pointer disabled:opacity-50"
          >
            {isStreaming ? (
              <>
                <Loader2 size={13} className="animate-spin" />
                <span>Streaming...</span>
              </>
            ) : (
              <>
                <Sparkles size={13} />
                <span>Fetch Satellite Scene</span>
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};
