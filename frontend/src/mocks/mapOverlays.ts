// Mock map overlay data representing bounding boxes, change masks, and heatmaps

export interface MapOverlay {
  id: string;
  type: 'bounding-box' | 'change-mask' | 'heatmap';
  label: string;
  category: 'Urban Growth' | 'Infrastructure' | 'Water Body' | 'Vegetation Loss';
  confidence: number;
  coordinates: {
    lat: number;
    lng: number;
    width: number;
    height: number;
  };
  metrics?: {
    areaHectares?: number;
    deltaPercentage?: number;
    spectralShift?: string;
  };
  style: {
    strokeColor: string;
    fillColor: string;
    fillOpacity: number;
  };
}

export const mockMapOverlays: MapOverlay[] = [
  {
    id: 'overlay-change-1',
    type: 'change-mask',
    label: 'North Sector Urban Expansion',
    category: 'Urban Growth',
    confidence: 0.942,
    coordinates: {
      lat: 13.0358,
      lng: 77.6322,
      width: 220,
      height: 140,
    },
    metrics: {
      areaHectares: 214.2,
      deltaPercentage: 18.6,
      spectralShift: 'NDVI -> NDBI shift (+0.42)',
    },
    style: {
      strokeColor: '#00f2ff',
      fillColor: 'rgba(0, 242, 255, 0.25)',
      fillOpacity: 0.35,
    },
  },
  {
    id: 'overlay-bbox-2',
    type: 'bounding-box',
    label: 'Commercial Logistic Hub',
    category: 'Infrastructure',
    confidence: 0.912,
    coordinates: {
      lat: 12.9812,
      lng: 77.6984,
      width: 140,
      height: 95,
    },
    metrics: {
      areaHectares: 78.4,
      deltaPercentage: 12.1,
    },
    style: {
      strokeColor: '#ffaa00',
      fillColor: 'rgba(255, 170, 0, 0.15)',
      fillOpacity: 0.2,
    },
  },
  {
    id: 'overlay-heatmap-3',
    type: 'heatmap',
    label: 'Impervious Surface Transition Density',
    category: 'Urban Growth',
    confidence: 0.884,
    coordinates: {
      lat: 13.0112,
      lng: 77.5812,
      width: 180,
      height: 180,
    },
    metrics: {
      areaHectares: 50.0,
      spectralShift: 'SAR Backscatter +4.2 dB',
    },
    style: {
      strokeColor: '#00dec2',
      fillColor: 'rgba(0, 222, 194, 0.3)',
      fillOpacity: 0.4,
    },
  },
];
