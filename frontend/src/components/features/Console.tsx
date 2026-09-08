import React from 'react';
import { motion } from 'framer-motion';
import MapViewport from '../map/MapViewport';
import QueryPanel from './QueryPanel';
import ExecutionTracePanel from './ExecutionTracePanel';
import { Badge } from '../ui/Badge';
import { useTrace } from '../../context/TraceContext';
import type { UploadedDataset } from './UploadDropzone';

// ---------------------------------------------------------------------------
// ErrorBoundary – prevents MapLibre / WebGL crashes from blanking the screen
// ---------------------------------------------------------------------------
interface EBProps { children: React.ReactNode; }
interface EBState { hasError: boolean; error: Error | null; }

class MapErrorBoundary extends React.Component<EBProps, EBState> {
  constructor(props: EBProps) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): EBState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    console.error('[SatQuery] MapErrorBoundary caught:', error, info);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="flex flex-col items-center justify-center h-full bg-space-navy/80 rounded-2xl border border-red-500/30 p-8 text-center">
          <div className="text-red-400 text-lg font-semibold mb-2">Map Rendering Error</div>
          <p className="text-slate-400 text-sm max-w-md mb-4">
            The map viewport encountered an error and was safely caught.
            This usually happens when GeoJSON coordinates are outside the valid geographic range.
          </p>
          <button
            onClick={() => this.setState({ hasError: false, error: null })}
            className="px-4 py-2 bg-accent-teal/20 hover:bg-accent-teal/40 text-accent-teal border border-accent-teal/30 rounded-lg text-sm font-medium transition-colors"
          >
            Reload Map
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

export const Console = () => {
  const {
    activeGeoJson,
    activeDatasetName,
    activeSensor,
    activeSession,
    setActiveDatasetName,
    setActiveSensor,
    recordSession,
  } = useTrace();

  const handleQuerySuccess = (query: string, data: any, dataset?: UploadedDataset | null) => {
    const dsName = dataset?.name || activeDatasetName;
    if (dataset?.name) {
      setActiveDatasetName(dataset.name);
    }
    if (dataset?.sensor) {
      setActiveSensor(dataset.sensor);
    }
    recordSession(query, data, dsName, dataset?.sensor);
  };

  const handleDatasetChange = (dataset: UploadedDataset) => {
    if (dataset?.name) {
      setActiveDatasetName(dataset.name);
    }
    if (dataset?.sensor) {
      setActiveSensor(dataset.sensor);
    }
  };

  return (
    <section id="console" className="py-20 bg-space-black relative scroll-mt-6">
      {/* Subtle ambient radar grid backdrop */}
      <div className="absolute inset-0 radar-grid opacity-30 pointer-events-none" />

      <div className="container mx-auto px-4 sm:px-6 relative z-10">
        {/* Section Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="mb-8 flex flex-col md:flex-row md:items-end justify-between gap-4 border-b border-white/10 pb-6"
        >
          <div>
            <div className="flex items-center gap-2 mb-2">
              <Badge variant="info" className="text-[10px] uppercase font-mono tracking-widest">
                Centerpiece UI Shell
              </Badge>
              <span className="text-xs font-mono text-slate-500">
                // Fast-API Backend & MapLibre GL Active
              </span>
            </div>
            <h2 className="text-3xl md:text-4xl font-bold text-white tracking-tight">
              Interactive Mission Console
            </h2>
            <p className="text-sm text-slate-400 max-w-2xl mt-1">
              Ask natural language queries on optical and SAR imagery pairs. The agentic controller routes tasks to specialist AI networks with visual evidence and auditable execution traces.
            </p>
          </div>

          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-space-navy border border-white/10 text-xs font-mono text-slate-300">
              <div className="w-2 h-2 rounded-full bg-accent-teal animate-ping" />
              <span>MapLibre GL: Online</span>
            </div>
          </div>
        </motion.div>

        {/* Main Workstation: Split Screen (Left Query / Right Map) */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 h-[750px] mb-6">
          {/* Left: Query & Upload Panel (4 cols) */}
          <div className="lg:col-span-5 xl:col-span-4 h-full flex flex-col min-h-0">
            <QueryPanel
              onQuerySuccess={handleQuerySuccess}
              onDatasetChange={handleDatasetChange}
              activeDatasetName={activeDatasetName}
              activeSession={activeSession}
            />
          </div>

          {/* Right: Interactive Map Viewport (8 cols) */}
          <div className="lg:col-span-7 xl:col-span-8 h-full min-h-0">
            <MapErrorBoundary>
              <MapViewport
                geoJsonData={activeGeoJson}
                datasetName={activeDatasetName}
                sensor={activeSensor}
              />
            </MapErrorBoundary>
          </div>
        </div>

        {/* Bottom Collapsible Auditable Trace Readout */}
        <div className="w-full">
          <ExecutionTracePanel />
        </div>
      </div>
    </section>
  );
};

export default Console;
