// Mock auditable execution trace log for SatQuery AI Agentic Controller

export interface TraceStep {
  id: string;
  timestamp: string;
  stage: 'PARSER' | 'PREPROCESS' | 'ROUTING' | 'INFERENCE' | 'POSTPROCESS' | 'AGGREGATION';
  action: string;
  modelOrTool: string;
  details: string;
  latencyMs: number;
  confidence?: number;
  status: 'success' | 'running' | 'warning' | 'error';
}

export const mockExecutionLogs: TraceStep[] = [
  {
    id: 'step-1',
    timestamp: '10:00:01.104',
    stage: 'PARSER',
    action: 'Query Intent Disambiguation',
    modelOrTool: 'Agentic Controller (Mistral-Large-GEO)',
    details: 'Extracted semantic intent: "TEMPORAL_CHANGE_DETECTION", entities: ["built-up area", "urban extent"], temporal constraint: "bi-temporal pair [T1, T2]".',
    latencyMs: 320,
    confidence: 0.985,
    status: 'success',
  },
  {
    id: 'step-2',
    timestamp: '10:00:01.428',
    stage: 'PREPROCESS',
    action: 'Raster Radiometric Normalization',
    modelOrTool: 'GDAL / Rasterio Core',
    details: 'Co-registered T1 (2023-01-15) and T2 (2026-02-10) GeoTIFFs to EPSG:4326. Reprojected to 10m GSD grid. Cloud shadow mask applied.',
    latencyMs: 910,
    status: 'success',
  },
  {
    id: 'step-3',
    timestamp: '10:00:02.340',
    stage: 'ROUTING',
    action: 'Specialist AI Tool Dispatch',
    modelOrTool: 'Multi-Agent Router v2.4',
    details: 'Task demands differential feature extraction. Routing pair to Siam-NestedUNet (Change Detection Engine) over VQA baseline.',
    latencyMs: 110,
    status: 'success',
  },
  {
    id: 'step-4',
    timestamp: '10:00:02.452',
    stage: 'INFERENCE',
    action: 'Bi-Temporal Feature Correlation',
    modelOrTool: 'Siam-NestedUNet + EVA-02 Encoder',
    details: 'Extracted deep multi-scale feature pyramids. Computed difference tensor across 4 spectral bands (NIR, Red, Green, SWIR).',
    latencyMs: 1980,
    confidence: 0.942,
    status: 'success',
  },
  {
    id: 'step-5',
    timestamp: '10:00:04.435',
    stage: 'POSTPROCESS',
    action: 'Polygon Vectorization & Metric Tally',
    modelOrTool: 'Raster-to-GeoJSON Vectorizer',
    details: 'Generated binary change raster mask. Clustered contiguous positive regions. Net built-up delta: +342.6 ha across 21 clusters.',
    latencyMs: 640,
    status: 'success',
  },
  {
    id: 'step-6',
    timestamp: '10:00:05.078',
    stage: 'AGGREGATION',
    action: 'Synthesis & Evidence Packaging',
    modelOrTool: 'Agentic Synthesis Engine',
    details: 'Bound visual change mask overlay to MapLibre viewport coordinates. Emitted auditable trace with verifiable telemetry.',
    latencyMs: 220,
    confidence: 0.967,
    status: 'success',
  },
];
