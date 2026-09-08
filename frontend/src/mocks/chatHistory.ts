// Mock chat history for SatQuery AI conversational query panel

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  datasetName?: string;
  sensorType?: 'Optical (Sentinel-2)' | 'SAR (Sentinel-1)' | 'Fused (Optical+SAR)';
  acquisitionDates?: string[];
  metadata?: {
    confidence?: number;
    toolsUsed?: string[];
    areaChangedHa?: number;
    detectionCount?: number;
    executionTimeMs?: number;
  };
}

export const mockChatHistory: ChatMessage[] = [
  {
    id: 'msg-1',
    role: 'user',
    content: 'How much did the built-up area increase between these two dates?',
    timestamp: '2026-09-04T10:00:00Z',
    datasetName: 'Bengaluru_Urban_Corridor_T1_T2.tif',
    sensorType: 'Optical (Sentinel-2)',
    acquisitionDates: ['2023-01-15', '2026-02-10'],
  },
  {
    id: 'msg-2',
    role: 'assistant',
    content: 'Bi-temporal change detection indicates a 14.8% net increase in built-up impervious surfaces (approx. +342.6 hectares) between Jan 2023 and Feb 2026. Primary urbanization concentrated along the Outer Ring Road and northern tech corridor. A high-resolution change mask has been generated and overlaid on your viewport.',
    timestamp: '2026-09-04T10:00:42Z',
    datasetName: 'Bengaluru_Urban_Corridor_T1_T2.tif',
    sensorType: 'Optical (Sentinel-2)',
    metadata: {
      confidence: 0.942,
      toolsUsed: ['Intent Parser', 'GDAL Pre-processor', 'Siam-NestedUNet Change Detector'],
      areaChangedHa: 342.6,
      executionTimeMs: 4180,
    },
  },
  {
    id: 'msg-3',
    role: 'user',
    content: 'Identify all industrial storage tanks and calculate their spatial clustering.',
    timestamp: '2026-09-04T10:04:15Z',
    datasetName: 'Mangalore_Port_Industrial_SAR.tif',
    sensorType: 'SAR (Sentinel-1)',
    acquisitionDates: ['2026-08-22'],
  },
  {
    id: 'msg-4',
    role: 'assistant',
    content: 'Visual grounding agent localized 18 cylindrical storage tanks in the southern terminal area using dual-polarization (VV/VH) backscatter signatures. Bounding boxes with confidence > 89% are rendered on the map with coordinate centroids.',
    timestamp: '2026-09-04T10:04:38Z',
    datasetName: 'Mangalore_Port_Industrial_SAR.tif',
    sensorType: 'SAR (Sentinel-1)',
    metadata: {
      confidence: 0.915,
      toolsUsed: ['SAR Radiance Calibrator', 'Visual Grounding Transformer', 'IoU Filter'],
      detectionCount: 18,
      executionTimeMs: 2310,
    },
  },
];
