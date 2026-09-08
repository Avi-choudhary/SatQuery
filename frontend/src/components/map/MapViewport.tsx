import React, { useState, useRef, useEffect, useCallback } from 'react';
import * as maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import { Panel } from '../ui/Panel';
import {
  ZoomIn,
  ZoomOut,
  Compass,
  Crosshair,
  Maximize2,
  Minimize2,
  Locate,
  Globe,
  MapPin,
  Search,
  Sliders,
  Play,
  Pause,
  Layers,
  Sparkles,
  ArrowLeftRight,
  Focus,
  Eye,
  EyeOff,
  Palette,
  X,
  Building2,
  Flag,
  Loader2
} from 'lucide-react';
import { useTrace, type SentinelOverlay } from '../../context/TraceContext';
import {
  searchLocations,
  parseCoordinates,
  type GeocodingResult,
  type LocationCategory
} from '../../lib/geocoding';

// Production-grade global basemaps (Google Maps style, 0 API keys required)
const SATELLITE_STYLE: any = {
  version: 8,
  sources: {
    'esri-satellite': {
      type: 'raster',
      tiles: [
        'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
      ],
      tileSize: 256,
      attribution: '&copy; Esri, Maxar',
      maxzoom: 19
    }
  },
  layers: [
    {
      id: 'esri-satellite-layer',
      type: 'raster',
      source: 'esri-satellite',
      minzoom: 0,
      maxzoom: 19
    }
  ]
};

const HYBRID_STYLE: any = {
  version: 8,
  sources: {
    'esri-satellite': {
      type: 'raster',
      tiles: [
        'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
      ],
      tileSize: 256,
      attribution: '&copy; Esri, Maxar',
      maxzoom: 19
    },
    'esri-boundaries': {
      type: 'raster',
      tiles: [
        'https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}'
      ],
      tileSize: 256,
      attribution: '&copy; Esri',
      maxzoom: 19
    }
  },
  layers: [
    {
      id: 'esri-satellite-layer',
      type: 'raster',
      source: 'esri-satellite',
      minzoom: 0,
      maxzoom: 19
    },
    {
      id: 'esri-boundaries-layer',
      type: 'raster',
      source: 'esri-boundaries',
      minzoom: 0,
      maxzoom: 19
    }
  ]
};

const STREETS_STYLE: any = {
  version: 8,
  sources: {
    'osm-streets': {
      type: 'raster',
      tiles: [
        'https://tile.openstreetmap.org/{z}/{x}/{y}.png'
      ],
      tileSize: 256,
      attribution: '&copy; OpenStreetMap contributors',
      maxzoom: 19
    }
  },
  layers: [
    {
      id: 'osm-streets-layer',
      type: 'raster',
      source: 'osm-streets',
      minzoom: 0,
      maxzoom: 19
    }
  ]
};

const DARK_TACTICAL_STYLE: any = {
  version: 8,
  sources: {
    'esri-dark-base': {
      type: 'raster',
      tiles: [
        'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}'
      ],
      tileSize: 256,
      attribution: '&copy; Esri, HERE, Garmin',
      maxzoom: 16
    },
    'esri-dark-ref': {
      type: 'raster',
      tiles: [
        'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}'
      ],
      tileSize: 256,
      attribution: '&copy; Esri',
      maxzoom: 16
    }
  },
  layers: [
    {
      id: 'esri-dark-base-layer',
      type: 'raster',
      source: 'esri-dark-base',
      minzoom: 0,
      maxzoom: 16
    },
    {
      id: 'esri-dark-ref-layer',
      type: 'raster',
      source: 'esri-dark-ref',
      minzoom: 0,
      maxzoom: 16
    }
  ]
};

// Preset bookmarks for instant exploration
const PRESET_BOOKMARKS = [
  { name: 'India (Subcontinent)', coords: [78.9629, 20.5937] as [number, number], zoom: 4.8 },
  { name: 'Bengaluru Urban', coords: [77.5946, 12.9716] as [number, number], zoom: 13.5 },
  { name: 'Delhi NCR Region', coords: [77.2090, 28.6139] as [number, number], zoom: 12.5 },
  { name: 'Mumbai Coastline', coords: [72.8777, 19.0760] as [number, number], zoom: 12.5 },
  { name: 'Himalayas / Siachen', coords: [77.1000, 35.3000] as [number, number], zoom: 9.5 },
  { name: 'Dubai Palm Jumeirah', coords: [55.1388, 25.1124] as [number, number], zoom: 13.0 },
  { name: 'Suez Canal (Egypt)', coords: [32.3436, 30.7050] as [number, number], zoom: 11.5 },
  { name: 'Global Earth View', coords: [15.0, 25.0] as [number, number], zoom: 2.2 },
];

export type BaseLayerType = 'satellite' | 'hybrid' | 'streets' | 'sar';
export type BandFilterType = 'true-color' | 'false-color-nir' | 'sar-contrast' | 'edge-boost';
export type TemporalModeType = 'swipe' | 'fade' | 't1' | 't2';

export interface MapViewportProps {
  geoJsonData?: any;
  datasetName?: string;
  sensor?: string;
  overlay?: SentinelOverlay | null;
}

export const MapViewport: React.FC<MapViewportProps> = ({ geoJsonData, datasetName, sensor, overlay: propOverlay }) => {
  const { activeOverlay: contextOverlay } = useTrace();
  const activeOverlay = propOverlay !== undefined ? propOverlay : contextOverlay;

  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);

  // Basemap & Viewport State
  const [activeBaseLayer, setActiveBaseLayer] = useState<BaseLayerType>('satellite');
  const [zoomLevel, setZoomLevel] = useState<number>(4.8);
  const [bearing, setBearing] = useState<number>(0);
  const [pitch, setPitch] = useState<number>(0);
  const [centerCoords, setCenterCoords] = useState<{ lat: number; lng: number }>({
    lat: 20.5937,
    lng: 78.9629
  });
  const [cursorCoords, setCursorCoords] = useState<{ lat: number; lng: number } | null>(null);

  // Sentinel Overlay Controls
  const [overlayOpacity, setOverlayOpacity] = useState<number>(0.92);
  const [temporalMode, setTemporalMode] = useState<TemporalModeType>('fade');
  const [temporalSlider, setTemporalSlider] = useState<number>(50); // 0 (100% T1) to 100 (100% T2)
  const [bandPreset, setBandPreset] = useState<BandFilterType>('true-color');
  const [isBlinking, setIsBlinking] = useState<boolean>(false);
  const [showOverlayControls, setShowOverlayControls] = useState<boolean>(true);
  const [isOverlayVisible, setIsOverlayVisible] = useState<boolean>(true);
  const [showColorLegend, setShowColorLegend] = useState<boolean>(false);

  // HUD & UI Tools
  const [showReticle, setShowReticle] = useState<boolean>(false);
  const [showBoundingBoxes, setShowBoundingBoxes] = useState<boolean>(true);
  const [isFullscreen, setIsFullscreen] = useState<boolean>(false);
  const [detectionCount, setDetectionCount] = useState<number>(0);
  const [detectionLabel, setDetectionLabel] = useState<string>('Global Interactive Slippy Map');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [searchResults, setSearchResults] = useState<GeocodingResult[]>([]);
  const [isSearching, setIsSearching] = useState<boolean>(false);
  const [showSearchResults, setShowSearchResults] = useState<boolean>(false);
  const searchAbortControllerRef = useRef<AbortController | null>(null);
  const searchContainerRef = useRef<HTMLDivElement>(null);
  const [showBookmarks, setShowBookmarks] = useState<boolean>(false);

  // Auto-switch basemap if SAR sensor is provided
  useEffect(() => {
    const isSar = (sensor && sensor.toLowerCase().includes('sar')) ||
                  (activeOverlay && activeOverlay.sensor.toLowerCase().includes('sar'));
    if (isSar && activeBaseLayer !== 'sar') {
      handleBaseLayerChange('sar');
    }
  }, [sensor, activeOverlay, activeBaseLayer]);

  // Blink comparison timer (rapidly toggles T1 and T2 at 1.2 Hz)
  useEffect(() => {
    if (!isBlinking) return;
    const interval = setInterval(() => {
      setTemporalSlider((prev) => (prev > 50 ? 0 : 100));
    }, 750);
    return () => clearInterval(interval);
  }, [isBlinking]);

  // Helper to extract bounding box from arbitrary GeoJSON
  const getBounds = useCallback((geojson: any): [[number, number], [number, number]] | null => {
    let minLng = Infinity, minLat = Infinity, maxLng = -Infinity, maxLat = -Infinity;

    const processCoords = (coords: any) => {
      if (Array.isArray(coords) && typeof coords[0] === 'number' && typeof coords[1] === 'number') {
        const [lng, lat] = coords;
        if (lng < minLng) minLng = lng;
        if (lat < minLat) minLat = lat;
        if (lng > maxLng) maxLng = lng;
        if (lat > maxLat) maxLat = lat;
      } else if (Array.isArray(coords)) {
        coords.forEach(processCoords);
      }
    };

    if (geojson?.type === 'FeatureCollection' && Array.isArray(geojson.features)) {
      geojson.features.forEach((f: any) => processCoords(f?.geometry?.coordinates));
    } else if (geojson?.type === 'Feature') {
      processCoords(geojson?.geometry?.coordinates);
    }

    if (minLng !== Infinity && maxLng !== -Infinity && !isNaN(minLng) && !isNaN(maxLng)) {
      return [[minLng, minLat], [maxLng, maxLat]];
    }
    return null;
  }, []);

  // Update or mount Sentinel raster imagery layers onto MapLibre WebGL canvas
  const updateOverlayLayers = useCallback((
    map: maplibregl.Map,
    ov: SentinelOverlay | null,
    opacity: number,
    mode: TemporalModeType,
    sliderVal: number,
    visible: boolean = true
  ) => {
    if (!ov || !ov.bounds || ov.bounds.length < 4) {
      // Clean up all overlay layers and sources safely
      ['sentinel-raster-layer-t2', 'sentinel-raster-layer-t1', 'sentinel-bbox-glow', 'sentinel-bbox-line'].forEach((id) => {
        if (map.getLayer(id)) map.removeLayer(id);
      });
      ['sentinel-raster-t2', 'sentinel-raster-t1', 'sentinel-bbox-source'].forEach((id) => {
        if (map.getSource(id)) map.removeSource(id);
      });
      return;
    }

    const isVisible = visible && opacity > 0.01;
    const visibilityVal = isVisible ? 'visible' : 'none';

    const [minLng, minLat, maxLng, maxLat] = ov.bounds;
    const imageCoordinates: [[number, number], [number, number], [number, number], [number, number]] = [
      [minLng, maxLat], // Top-Left
      [maxLng, maxLat], // Top-Right
      [maxLng, minLat], // Bottom-Right
      [minLng, minLat], // Bottom-Left
    ];

    // Calculate opacities based on temporal mode
    let t1Opacity = isVisible ? opacity : 0;
    let t2Opacity = 0;

    if (ov.mode === 'bi-temporal' && ov.t2ImageUrl) {
      if (mode === 't1') {
        t1Opacity = isVisible ? opacity : 0;
        t2Opacity = 0;
      } else if (mode === 't2') {
        t1Opacity = 0;
        t2Opacity = isVisible ? opacity : 0;
      } else {
        // fade or swipe blend
        const ratio = sliderVal / 100;
        t1Opacity = isVisible ? opacity * (1 - ratio) : 0;
        t2Opacity = isVisible ? opacity * ratio : 0;
      }
    }

    // Determine insertion position (below AI detection polygons so boxes stay on top)
    const beforeLayerId = map.getLayer('satquery-fill') ? 'satquery-fill' : undefined;

    // 1. Add or Update T1 Raster Source & Layer
    if (ov.t1ImageUrl) {
      try {
        const existingT1 = map.getSource('sentinel-raster-t1') as maplibregl.ImageSource | undefined;
        if (existingT1 && typeof existingT1.updateImage === 'function') {
          existingT1.updateImage({
            url: ov.t1ImageUrl,
            coordinates: imageCoordinates
          });
          if (map.getLayer('sentinel-raster-layer-t1')) {
            map.setLayoutProperty('sentinel-raster-layer-t1', 'visibility', visibilityVal);
            map.setPaintProperty('sentinel-raster-layer-t1', 'raster-opacity', t1Opacity);
          }
        } else {
          if (map.getLayer('sentinel-raster-layer-t1')) map.removeLayer('sentinel-raster-layer-t1');
          if (map.getSource('sentinel-raster-t1')) map.removeSource('sentinel-raster-t1');

          map.addSource('sentinel-raster-t1', {
            type: 'image',
            url: ov.t1ImageUrl,
            coordinates: imageCoordinates
          });
          map.addLayer(
            {
              id: 'sentinel-raster-layer-t1',
              type: 'raster',
              source: 'sentinel-raster-t1',
              layout: {
                visibility: visibilityVal
              },
              paint: {
                'raster-opacity': t1Opacity,
                'raster-fade-duration': 100
              }
            },
            beforeLayerId
          );
        }
      } catch (err) {
        console.warn('Sentinel T1 overlay update notice:', err);
      }
    } else {
      if (map.getLayer('sentinel-raster-layer-t1')) map.removeLayer('sentinel-raster-layer-t1');
      if (map.getSource('sentinel-raster-t1')) map.removeSource('sentinel-raster-t1');
    }

    // 2. Add or Update T2 Raster Source & Layer if bi-temporal
    if (ov.mode === 'bi-temporal' && ov.t2ImageUrl) {
      try {
        const existingT2 = map.getSource('sentinel-raster-t2') as maplibregl.ImageSource | undefined;
        if (existingT2 && typeof existingT2.updateImage === 'function') {
          existingT2.updateImage({
            url: ov.t2ImageUrl,
            coordinates: imageCoordinates
          });
          if (map.getLayer('sentinel-raster-layer-t2')) {
            map.setLayoutProperty('sentinel-raster-layer-t2', 'visibility', visibilityVal);
            map.setPaintProperty('sentinel-raster-layer-t2', 'raster-opacity', t2Opacity);
          }
        } else {
          if (map.getLayer('sentinel-raster-layer-t2')) map.removeLayer('sentinel-raster-layer-t2');
          if (map.getSource('sentinel-raster-t2')) map.removeSource('sentinel-raster-t2');

          map.addSource('sentinel-raster-t2', {
            type: 'image',
            url: ov.t2ImageUrl,
            coordinates: imageCoordinates
          });
          map.addLayer(
            {
              id: 'sentinel-raster-layer-t2',
              type: 'raster',
              source: 'sentinel-raster-t2',
              layout: {
                visibility: visibilityVal
              },
              paint: {
                'raster-opacity': t2Opacity,
                'raster-fade-duration': 100
              }
            },
            beforeLayerId
          );
        }
      } catch (err) {
        console.warn('Sentinel T2 overlay update notice:', err);
      }
    } else {
      if (map.getLayer('sentinel-raster-layer-t2')) map.removeLayer('sentinel-raster-layer-t2');
      if (map.getSource('sentinel-raster-t2')) map.removeSource('sentinel-raster-t2');
    }

    // 3. Add or Update Georeferenced Bounding Box Perimeter
    const bboxGeoJson: any = {
      type: 'Feature',
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [minLng, maxLat],
          [maxLng, maxLat],
          [maxLng, minLat],
          [minLng, minLat],
          [minLng, maxLat]
        ]]
      },
      properties: { name: ov.name }
    };

    try {
      const existingBBox = map.getSource('sentinel-bbox-source') as maplibregl.GeoJSONSource | undefined;
      if (existingBBox && typeof existingBBox.setData === 'function') {
        existingBBox.setData(bboxGeoJson);
        if (map.getLayer('sentinel-bbox-glow')) {
          map.setLayoutProperty('sentinel-bbox-glow', 'visibility', visibilityVal);
          map.setPaintProperty('sentinel-bbox-glow', 'line-opacity', isVisible ? 0.6 * opacity : 0);
        }
        if (map.getLayer('sentinel-bbox-line')) {
          map.setLayoutProperty('sentinel-bbox-line', 'visibility', visibilityVal);
          map.setPaintProperty('sentinel-bbox-line', 'line-opacity', isVisible ? 0.95 * opacity : 0);
        }
      } else {
        if (map.getLayer('sentinel-bbox-line')) map.removeLayer('sentinel-bbox-line');
        if (map.getLayer('sentinel-bbox-glow')) map.removeLayer('sentinel-bbox-glow');
        if (map.getSource('sentinel-bbox-source')) map.removeSource('sentinel-bbox-source');

        map.addSource('sentinel-bbox-source', {
          type: 'geojson',
          data: bboxGeoJson
        });

        // Neon cyan boundary glow halo
        map.addLayer({
          id: 'sentinel-bbox-glow',
          type: 'line',
          source: 'sentinel-bbox-source',
          layout: {
            visibility: visibilityVal
          },
          paint: {
            'line-color': '#00f2ff',
            'line-width': 5,
            'line-blur': 3,
            'line-opacity': isVisible ? 0.6 * opacity : 0
          }
        });

        // Crisp dashed boundary line
        map.addLayer({
          id: 'sentinel-bbox-line',
          type: 'line',
          source: 'sentinel-bbox-source',
          layout: {
            visibility: visibilityVal
          },
          paint: {
            'line-color': '#00f2ff',
            'line-width': 2,
            'line-dasharray': [4, 2],
            'line-opacity': isVisible ? 0.95 * opacity : 0
          }
        });
      }
    } catch (err) {
      console.warn('Sentinel bbox add notice:', err);
    }
  }, []);

  // Ensure AI visual grounding detection source & layers exist
  const ensureDetectionLayers = useCallback((map: maplibregl.Map, data: any) => {
    const emptyCollection = { type: 'FeatureCollection', features: [] };
    const sourceData = data || emptyCollection;

    if (!map.getSource('satquery-detections')) {
      map.addSource('satquery-detections', {
        type: 'geojson',
        data: sourceData
      });

      // Semi-transparent fill supporting feature properties (dynamic or fallback to cyan)
      map.addLayer({
        id: 'satquery-fill',
        type: 'fill',
        source: 'satquery-detections',
        paint: {
          'fill-color': [
            'coalesce',
            ['get', 'fillColor'],
            ['get', 'color'],
            '#00f2ff'
          ],
          'fill-opacity': [
            'coalesce',
            ['get', 'fillOpacity'],
            0.28
          ]
        }
      });

      // Neon glow halo
      map.addLayer({
        id: 'satquery-glow',
        type: 'line',
        source: 'satquery-detections',
        paint: {
          'line-color': [
            'coalesce',
            ['get', 'strokeColor'],
            ['get', 'color'],
            '#00f2ff'
          ],
          'line-width': 7,
          'line-blur': 4,
          'line-opacity': 0.75
        }
      });

      // Crisp boundary outline
      map.addLayer({
        id: 'satquery-outline',
        type: 'line',
        source: 'satquery-detections',
        paint: {
          'line-color': [
            'coalesce',
            ['get', 'strokeColor'],
            ['get', 'color'],
            '#00f2ff'
          ],
          'line-width': 2.5,
          'line-opacity': 0.95
        }
      });

      // Interactive popup on clicking detected polygon
      map.on('click', 'satquery-fill', (e) => {
        if (!e.features || e.features.length === 0) return;
        const feat = e.features[0];
        const props = (feat.properties as any) || {};
        const label = props.label || 'Detected Region';
        const areaHa = props.area_ha ? `${parseFloat(props.area_ha).toFixed(2)} ha` : null;
        const areaM2 = props.area_m2 ? `${parseFloat(props.area_m2).toLocaleString()} m²` : null;
        const typeStr = props.type || 'Detection';
        const color = props.strokeColor || props.color || '#00f2ff';

        new maplibregl.Popup({ closeButton: true, closeOnClick: true, className: 'satquery-detection-popup' })
          .setLngLat(e.lngLat)
          .setHTML(`
            <div style="padding: 8px; font-family: monospace; font-size: 11px; color: #f1f5f9; background: #0b0f19; border-radius: 6px; border: 1px solid ${color}; box-shadow: 0 0 15px rgba(0,0,0,0.8);">
              <div style="font-weight: 700; color: ${color}; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.5px;">${label}</div>
              <div style="font-size: 10px; color: #94a3b8; margin-bottom: 4px;">Classification: <strong>${typeStr}</strong></div>
              ${areaHa ? `<div style="font-size: 10px; color: #cbd5e1;">Area: <strong>${areaHa}</strong> (${areaM2})</div>` : ''}
              <div style="font-size: 9px; color: #64748b; margin-top: 4px;">Center: ${e.lngLat.lat.toFixed(4)}°N, ${e.lngLat.lng.toFixed(4)}°E</div>
            </div>
          `)
          .addTo(map);
      });

      map.on('mouseenter', 'satquery-fill', () => {
        map.getCanvas().style.cursor = 'pointer';
      });
      map.on('mouseleave', 'satquery-fill', () => {
        map.getCanvas().style.cursor = '';
      });
    } else {
      const source = map.getSource('satquery-detections') as maplibregl.GeoJSONSource;
      source.setData(sourceData);
    }
  }, []);

  // 1. Initialize MapLibre GL instance
  useEffect(() => {
    if (!mapContainerRef.current || mapRef.current) return;

    const initialCenter: [number, number] = activeOverlay?.center
      ? activeOverlay.center
      : [78.9629, 20.5937];
    const initialZoom = activeOverlay ? 13 : 4.8;

    const map = new maplibregl.Map({
      container: mapContainerRef.current,
      style: SATELLITE_STYLE,
      center: initialCenter,
      zoom: initialZoom,
      pitch: 0,
      bearing: 0,
      attributionControl: false,
      dragRotate: true,
      touchPitch: true,
      pitchWithRotate: true,
      maxPitch: 70
    });

    mapRef.current = map;

    // Standard Scale Bar at bottom left
    const scale = new maplibregl.ScaleControl({
      maxWidth: 100,
      unit: 'metric'
    });
    map.addControl(scale, 'bottom-left');

    map.on('move', () => {
      const center = map.getCenter();
      setCenterCoords({ lat: parseFloat(center.lat.toFixed(4)), lng: parseFloat(center.lng.toFixed(4)) });
      setZoomLevel(parseFloat(map.getZoom().toFixed(1)));
      setBearing(parseFloat(map.getBearing().toFixed(0)));
      setPitch(parseFloat(map.getPitch().toFixed(0)));
    });

    map.on('mousemove', (e) => {
      setCursorCoords({
        lat: parseFloat(e.lngLat.lat.toFixed(5)),
        lng: parseFloat(e.lngLat.lng.toFixed(5))
      });
    });

    map.on('mouseout', () => {
      setCursorCoords(null);
    });

    map.on('load', () => {
      ensureDetectionLayers(map, null);
      if (activeOverlay) {
        updateOverlayLayers(map, activeOverlay, overlayOpacity, temporalMode, temporalSlider, isOverlayVisible);
      }
    });

    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, []);

  // 2. React to activeOverlay updates (uploaded Sentinel images or selected presets)
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !activeOverlay || !activeOverlay.bounds) return;

    const applyOverlay = () => {
      updateOverlayLayers(map, activeOverlay, overlayOpacity, temporalMode, temporalSlider, isOverlayVisible);

      // Smooth camera glide to georeferenced bounding box
      const [minLng, minLat, maxLng, maxLat] = activeOverlay.bounds;
      map.fitBounds(
        [
          [minLng, minLat],
          [maxLng, maxLat]
        ],
        {
          padding: 85,
          maxZoom: 16.5,
          duration: 2200
        }
      );

      setDetectionLabel(
        `${activeOverlay.name} [${activeOverlay.mode === 'bi-temporal' ? 'Bi-Temporal Pair' : 'Single Scene'}]`
      );
    };

    if (map.isStyleLoaded()) {
      applyOverlay();
    } else {
      map.once('load', applyOverlay);
    }
  }, [activeOverlay, updateOverlayLayers, isOverlayVisible]);

  // 3. React to opacity, visibility, or temporal slider adjustments
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !activeOverlay || !map.isStyleLoaded()) return;

    const isVisible = isOverlayVisible && overlayOpacity > 0.01;
    const visibilityVal = isVisible ? 'visible' : 'none';

    let t1Op = isVisible ? overlayOpacity : 0;
    let t2Op = 0;

    if (activeOverlay.mode === 'bi-temporal' && activeOverlay.t2ImageUrl) {
      if (temporalMode === 't1') {
        t1Op = isVisible ? overlayOpacity : 0;
        t2Op = 0;
      } else if (temporalMode === 't2') {
        t1Op = 0;
        t2Op = isVisible ? overlayOpacity : 0;
      } else {
        const ratio = temporalSlider / 100;
        t1Op = isVisible ? overlayOpacity * (1 - ratio) : 0;
        t2Op = isVisible ? overlayOpacity * ratio : 0;
      }
    }

    if (map.getLayer('sentinel-raster-layer-t1')) {
      map.setLayoutProperty('sentinel-raster-layer-t1', 'visibility', visibilityVal);
      map.setPaintProperty('sentinel-raster-layer-t1', 'raster-opacity', t1Op);
    }
    if (map.getLayer('sentinel-raster-layer-t2')) {
      map.setLayoutProperty('sentinel-raster-layer-t2', 'visibility', visibilityVal);
      map.setPaintProperty('sentinel-raster-layer-t2', 'raster-opacity', t2Op);
    }
    if (map.getLayer('sentinel-bbox-glow')) {
      map.setLayoutProperty('sentinel-bbox-glow', 'visibility', visibilityVal);
      map.setPaintProperty('sentinel-bbox-glow', 'line-opacity', isVisible ? 0.6 * overlayOpacity : 0);
    }
    if (map.getLayer('sentinel-bbox-line')) {
      map.setLayoutProperty('sentinel-bbox-line', 'visibility', visibilityVal);
      map.setPaintProperty('sentinel-bbox-line', 'line-opacity', isVisible ? 0.95 * overlayOpacity : 0);
    }
  }, [overlayOpacity, isOverlayVisible, temporalSlider, temporalMode, activeOverlay]);

  // 4. React to dynamic GeoJSON detections from AI inference queries
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    try {
      let targetGeoJson: any = null;
      let parsedInput = geoJsonData;
      if (typeof parsedInput === 'string') {
        try {
          const trimmed = parsedInput.trim();
          if (trimmed.startsWith('{') || trimmed.startsWith('[')) {
            parsedInput = JSON.parse(trimmed);
          }
        } catch (e) {
          // Not a JSON string
        }
      }

      if (parsedInput) {
        if (parsedInput.type === 'FeatureCollection') {
          targetGeoJson = JSON.parse(JSON.stringify(parsedInput));
        } else if (parsedInput.type === 'Feature') {
          targetGeoJson = { type: 'FeatureCollection', features: [JSON.parse(JSON.stringify(parsedInput))] };
        } else if (Array.isArray(parsedInput) && parsedInput.length > 0) {
          targetGeoJson = { type: 'FeatureCollection', features: JSON.parse(JSON.stringify(parsedInput)) };
        }
      }

      if (!targetGeoJson || !targetGeoJson.features || targetGeoJson.features.length === 0) {
        const source = map.getSource('satquery-detections') as maplibregl.GeoJSONSource;
        if (source) {
          source.setData({ type: 'FeatureCollection', features: [] });
        }
        setDetectionCount(0);
        return;
      }

      // Check if GeoJSON coordinates are normalized [0, 1] relative coordinates
      const rawBounds = getBounds(targetGeoJson);
      if (rawBounds) {
        const [[rMinLng, rMinLat], [rMaxLng, rMaxLat]] = rawBounds;
        const isRelative = rMinLng >= -0.05 && rMaxLng <= 1.05 && rMinLat >= -0.05 && rMaxLat <= 1.05;

        if (isRelative) {
          const sceneBounds = activeOverlay?.bounds || [77.618, 13.022, 77.652, 13.048];
          const [minLng, minLat, maxLng, maxLat] = sceneBounds;

          targetGeoJson.features.forEach((feat: any) => {
            if (feat.geometry?.type === 'Polygon' && Array.isArray(feat.geometry.coordinates)) {
              feat.geometry.coordinates = feat.geometry.coordinates.map((ring: any[]) =>
                ring.map((pt: any) => [
                  minLng + pt[0] * (maxLng - minLng),
                  maxLat - pt[1] * (maxLat - minLat)
                ])
              );
            }
          });
        }
      }

      const source = map.getSource('satquery-detections') as maplibregl.GeoJSONSource;
      if (source) {
        source.setData(targetGeoJson);
      } else {
        ensureDetectionLayers(map, targetGeoJson);
      }

      const count = targetGeoJson.features ? targetGeoJson.features.length : 0;
      setDetectionCount(count);

      const bounds = getBounds(targetGeoJson);
      if (bounds) {
        const [[minLng, minLat], [maxLng, maxLat]] = bounds;

        // Validate that bounds are within MapLibre's valid coordinate range.
        // Raw projected coordinates (e.g. UTM metres) will fail this check.
        const isValidLngLat =
          minLng >= -180 && maxLng <= 180 &&
          minLat >= -90 && maxLat <= 90 &&
          isFinite(minLng) && isFinite(maxLng) &&
          isFinite(minLat) && isFinite(maxLat);

        if (isValidLngLat) {
          const midLng = (minLng + maxLng) / 2;
          const midLat = (minLat + maxLat) / 2;
          setCenterCoords({ lat: parseFloat(midLat.toFixed(4)), lng: parseFloat(midLng.toFixed(4)) });

          try {
            map.fitBounds(bounds, {
              padding: 85,
              maxZoom: 16.5,
              duration: 1800
            });
          } catch (fitErr) {
            console.warn('[SatQuery] fitBounds failed, using fallback:', fitErr);
          }

          const featureLabel = targetGeoJson.features?.[0]?.properties?.label;
          setDetectionLabel(
            featureLabel || (
              datasetName
                ? `${datasetName} [Active AI Detection]`
                : `Detected Feature [${midLat.toFixed(4)}°N, ${midLng.toFixed(4)}°E]`
            )
          );
        } else {
          // Coordinates are out of range (projected CRS); use overlay bounds as fallback
          console.warn('[SatQuery] GeoJSON bounds out of WGS84 range, using overlay fallback:', { minLng, minLat, maxLng, maxLat });
          if (activeOverlay?.bounds) {
            const [oMinLng, oMinLat, oMaxLng, oMaxLat] = activeOverlay.bounds;
            const midLng = (oMinLng + oMaxLng) / 2;
            const midLat = (oMinLat + oMaxLat) / 2;
            setCenterCoords({ lat: parseFloat(midLat.toFixed(4)), lng: parseFloat(midLng.toFixed(4)) });
            try {
              map.fitBounds([[oMinLng, oMinLat], [oMaxLng, oMaxLat]], {
                padding: 85,
                maxZoom: 16.5,
                duration: 1800
              });
            } catch (fitErr) {
              console.warn('[SatQuery] Overlay fallback fitBounds also failed:', fitErr);
            }
          }
        }
      }
    } catch (err) {
      console.error('[SatQuery] GeoJSON detection layer error caught by safety guard:', err);
    }
  }, [geoJsonData, datasetName, getBounds, ensureDetectionLayers, activeOverlay]);

  // 5. Basemap switching (Satellite, Hybrid, Streets, SAR)
  const handleBaseLayerChange = (layer: BaseLayerType) => {
    setActiveBaseLayer(layer);
    const map = mapRef.current;
    if (!map) return;

    let targetStyle = SATELLITE_STYLE;
    if (layer === 'hybrid') targetStyle = HYBRID_STYLE;
    else if (layer === 'streets') targetStyle = STREETS_STYLE;
    else if (layer === 'sar') targetStyle = DARK_TACTICAL_STYLE;

    map.setStyle(targetStyle);

    // Re-attach all dynamic Sentinel raster and vector layers after basemap reload
    map.once('style.load', () => {
      ensureDetectionLayers(map, geoJsonData);
      if (activeOverlay) {
        updateOverlayLayers(map, activeOverlay, overlayOpacity, temporalMode, temporalSlider, isOverlayVisible);
      }
    });
  };

  // 6. Layer Visibility Toggle
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    ['satquery-fill', 'satquery-glow', 'satquery-outline'].forEach((layerId) => {
      if (map.getLayer(layerId)) {
        map.setLayoutProperty(layerId, 'visibility', showBoundingBoxes ? 'visible' : 'none');
      }
    });
  }, [showBoundingBoxes]);

  // 7. Navigation Actions
  const handleZoomIn = () => mapRef.current?.zoomIn({ duration: 300 });
  const handleZoomOut = () => mapRef.current?.zoomOut({ duration: 300 });

  const handleResetNorthPitch = () => {
    mapRef.current?.easeTo({
      bearing: 0,
      pitch: 0,
      duration: 600
    });
  };

  const handleFlyToScene = () => {
    if (!mapRef.current || !activeOverlay?.bounds) return;
    const [minLng, minLat, maxLng, maxLat] = activeOverlay.bounds;
    mapRef.current.fitBounds(
      [
        [minLng, minLat],
        [maxLng, maxLat]
      ],
      {
        padding: 85,
        maxZoom: 16.5,
        duration: 2000
      }
    );
  };

  const handleLocateMe = () => {
    if (!navigator.geolocation) {
      alert('Geolocation is not supported by your browser.');
      return;
    }
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        const { latitude, longitude } = pos.coords;
        mapRef.current?.flyTo({
          center: [longitude, latitude],
          zoom: 13,
          duration: 2000,
          essential: true
        });
        setDetectionLabel(`User Location [${latitude.toFixed(4)}°N, ${longitude.toFixed(4)}°E]`);
      },
      (err) => console.warn('Geolocation notice:', err.message)
    );
  };

  const handleFlyToBookmark = (coords: [number, number], zoom: number, name: string) => {
    setShowBookmarks(false);
    mapRef.current?.flyTo({
      center: coords,
      zoom: zoom,
      pitch: zoom > 10 ? 25 : 0,
      duration: 2200,
      essential: true
    });
    setDetectionLabel(`Location: ${name}`);
  };

  // 8. Geocoding Search: Debounced query to OpenStreetMap Nominatim with local fallback
  useEffect(() => {
    const trimmed = searchQuery.trim();
    if (!trimmed || trimmed.length < 2) {
      setSearchResults([]);
      setIsSearching(false);
      setShowSearchResults(false);
      return;
    }

    if (searchAbortControllerRef.current) {
      searchAbortControllerRef.current.abort();
    }
    const controller = new AbortController();
    searchAbortControllerRef.current = controller;

    setIsSearching(true);
    const timer = setTimeout(async () => {
      try {
        const results = await searchLocations(trimmed, controller.signal);
        setSearchResults(results);
        setShowSearchResults(results.length > 0);
      } catch (err: any) {
        if (err.name !== 'AbortError') {
          console.warn('Geocoding search notice:', err);
        }
      } finally {
        setIsSearching(false);
      }
    }, 280);

    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [searchQuery]);

  // Close search suggestions on click outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (searchContainerRef.current && !searchContainerRef.current.contains(e.target as Node)) {
        setShowSearchResults(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Handle selection of any geocoded location (City, State, Country, or POI)
  const selectSearchResult = (item: GeocodingResult) => {
    setShowSearchResults(false);
    setSearchQuery('');
    const map = mapRef.current;
    if (!map) return;

    // Type-sensitive navigation:
    // Country -> Broad framing (zoom ~4.5)
    // State   -> Regional framing (zoom ~7.5)
    // City    -> High-detail urban street grid (zoom ~12.5)
    // Landmark-> Extreme detail (zoom ~15.0)
    if (item.bounds) {
      const maxZoom =
        item.category === 'country'
          ? 5.5
          : item.category === 'state'
          ? 8.5
          : item.category === 'city'
          ? 14.0
          : 16.5;

      map.fitBounds(item.bounds, {
        padding: 65,
        maxZoom,
        duration: 2200
      });
    } else {
      map.flyTo({
        center: item.center,
        zoom: item.zoom,
        pitch: item.category === 'city' || item.category === 'poi' ? 25 : 0,
        duration: 2200,
        essential: true
      });
    }

    setDetectionLabel(`Location: ${item.name} [${item.badgeLabel}]`);
  };

  const handleSearchSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const query = searchQuery.trim();
    if (!query) return;

    // 1. Raw Coordinates check (e.g. "28.6139, 77.2090")
    const coordMatch = parseCoordinates(query);
    if (coordMatch) {
      selectSearchResult(coordMatch);
      return;
    }

    // 2. If suggestions already loaded in state, pick top item
    if (searchResults.length > 0) {
      selectSearchResult(searchResults[0]);
      return;
    }

    // 3. Otherwise execute immediate search
    setIsSearching(true);
    try {
      const results = await searchLocations(query);
      if (results.length > 0) {
        selectSearchResult(results[0]);
      } else {
        const found = PRESET_BOOKMARKS.find((b) => b.name.toLowerCase().includes(query.toLowerCase()));
        if (found) {
          handleFlyToBookmark(found.coords, found.zoom, found.name);
          setSearchQuery('');
        } else {
          setDetectionLabel(`No location found for "${query}"`);
        }
      }
    } finally {
      setIsSearching(false);
    }
  };

  const toggleFullscreen = () => {
    if (!mapContainerRef.current) return;
    if (!document.fullscreenElement) {
      mapContainerRef.current.requestFullscreen().then(() => setIsFullscreen(true)).catch(() => {});
    } else {
      document.exitFullscreen().then(() => setIsFullscreen(false)).catch(() => {});
    }
  };

  // Compute CSS filter style for Band Visualization Presets
  const getBandFilterStyle = (): React.CSSProperties => {
    switch (bandPreset) {
      case 'false-color-nir':
        return { filter: 'hue-rotate(130deg) saturate(240%) contrast(120%)' };
      case 'sar-contrast':
        return { filter: 'contrast(190%) brightness(115%) grayscale(70%)' };
      case 'edge-boost':
        return { filter: 'contrast(240%) saturate(160%)' };
      default:
        return {};
    }
  };

  return (
    <Panel variant="glass" className="h-full w-full relative flex flex-col overflow-hidden select-none border border-white/10">
      {/* Top Telemetry & Control Bar */}
      <div className="px-3 py-2 border-b border-white/10 bg-space-black/85 backdrop-blur-md flex flex-wrap items-center justify-between gap-2 z-20">
        {/* Left Status & Title */}
        <div className="flex items-center gap-2.5">
          <div className="flex items-center gap-1.5 text-xs font-mono text-accent-cyan">
            <Globe size={14} className="text-accent-cyan" />
            <span className="font-semibold uppercase tracking-wider hidden sm:inline">SATQUERY MAP</span>
          </div>

          <div className="h-3.5 w-px bg-white/20 hidden md:block" />

          <span className="text-[11px] font-mono text-slate-300 truncate max-w-[200px] lg:max-w-xs">
            {detectionLabel}
          </span>
        </div>

        {/* Center/Right: Quick Search & Bookmarks */}
        <div className="flex items-center gap-2">
          {/* Quick Search with Autocomplete Dropdown & Type-Aware Zoom */}
          <div ref={searchContainerRef} className="relative">
            <form onSubmit={handleSearchSubmit} className="relative flex items-center">
              <input
                type="text"
                value={searchQuery}
                onFocus={() => {
                  if (searchResults.length > 0) setShowSearchResults(true);
                }}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search city, state, country..."
                className="bg-space-navy/90 border border-white/15 rounded-md pl-7 pr-7 py-1 text-[11px] font-mono text-white placeholder-slate-400 focus:outline-none focus:border-accent-cyan w-36 sm:w-48 md:w-56 lg:w-64 transition-all shadow-inner"
              />
              {isSearching ? (
                <Loader2 size={12} className="absolute left-2 text-accent-cyan animate-spin pointer-events-none" />
              ) : (
                <Search size={12} className="absolute left-2 text-slate-400 pointer-events-none" />
              )}

              {searchQuery && (
                <button
                  type="button"
                  onClick={() => {
                    setSearchQuery('');
                    setSearchResults([]);
                    setShowSearchResults(false);
                  }}
                  className="absolute right-2 text-slate-400 hover:text-white cursor-pointer p-0.5"
                  title="Clear search"
                >
                  <X size={11} />
                </button>
              )}
            </form>

            {/* Suggestions Dropdown */}
            {showSearchResults && searchResults.length > 0 && (
              <div className="absolute left-0 top-full mt-1.5 w-72 sm:w-80 bg-space-black/95 border border-white/20 rounded-xl shadow-2xl py-1 z-50 backdrop-blur-xl max-h-72 overflow-y-auto">
                <div className="px-3 py-1 text-[9px] font-mono uppercase tracking-wider text-slate-400 border-b border-white/10 flex justify-between items-center">
                  <span>Geocoding Results ({searchResults.length})</span>
                  <span className="text-[8px] text-accent-cyan">Auto-Zoom Hierarchy</span>
                </div>

                {searchResults.map((item) => {
                  const getCategoryBadgeClass = (cat: LocationCategory) => {
                    switch (cat) {
                      case 'country':
                        return 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30';
                      case 'state':
                        return 'bg-sky-500/20 text-sky-400 border-sky-500/30';
                      case 'city':
                        return 'bg-purple-500/20 text-purple-400 border-purple-500/30';
                      default:
                        return 'bg-accent-cyan/20 text-accent-cyan border-accent-cyan/30';
                    }
                  };

                  const getCategoryIcon = (cat: LocationCategory) => {
                    switch (cat) {
                      case 'country':
                        return <Flag size={12} className="text-emerald-400 shrink-0 mt-0.5" />;
                      case 'state':
                        return <MapPin size={12} className="text-sky-400 shrink-0 mt-0.5" />;
                      case 'city':
                        return <Building2 size={12} className="text-purple-400 shrink-0 mt-0.5" />;
                      default:
                        return <Search size={12} className="text-accent-cyan shrink-0 mt-0.5" />;
                    }
                  };

                  return (
                    <button
                      key={item.id}
                      type="button"
                      onClick={() => selectSearchResult(item)}
                      className="w-full text-left px-3 py-2 text-xs text-slate-200 hover:bg-accent-cyan/15 hover:text-white flex items-start gap-2.5 transition-colors border-b border-white/5 last:border-none cursor-pointer"
                    >
                      {getCategoryIcon(item.category)}
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center justify-between gap-2">
                          <span className="font-semibold truncate text-white">{item.name}</span>
                          <span
                            className={`text-[9px] font-mono px-1.5 py-0.2 rounded border ${getCategoryBadgeClass(
                              item.category
                            )} shrink-0`}
                          >
                            {item.badgeLabel}
                          </span>
                        </div>
                        <p className="text-[10px] text-slate-400 truncate mt-0.5 font-sans">
                          {item.displayName}
                        </p>
                      </div>
                    </button>
                  );
                })}
              </div>
            )}
          </div>

          {/* Location Bookmark Dropdown */}
          <div className="relative">
            <button
              type="button"
              onClick={() => setShowBookmarks(!showBookmarks)}
              className="flex items-center gap-1 px-2 py-1 bg-white/5 hover:bg-white/10 border border-white/10 rounded-md text-[10px] font-mono text-slate-300 transition-colors cursor-pointer"
              title="Quick Jump Presets"
            >
              <MapPin size={11} className="text-accent-cyan" />
              <span className="hidden sm:inline">Jump</span>
            </button>

            {showBookmarks && (
              <div className="absolute right-0 top-full mt-1 w-52 bg-space-black/95 border border-white/15 rounded-lg shadow-2xl py-1 z-50 backdrop-blur-lg">
                <div className="px-3 py-1 text-[9px] font-mono uppercase text-slate-400 border-b border-white/10">
                  Select Location Bookmark
                </div>
                {PRESET_BOOKMARKS.map((b) => (
                  <button
                    key={b.name}
                    type="button"
                    onClick={() => handleFlyToBookmark(b.coords, b.zoom, b.name)}
                    className="w-full text-left px-3 py-1.5 text-xs text-slate-200 hover:bg-accent-cyan/15 hover:text-accent-cyan flex items-center justify-between transition-colors cursor-pointer"
                  >
                    <span>{b.name}</span>
                    <span className="text-[10px] font-mono text-slate-400">{b.zoom}x</span>
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Active Overlay Visibility Quick Toggle */}
          {activeOverlay && (
            <button
              type="button"
              onClick={() => setIsOverlayVisible(!isOverlayVisible)}
              className={`flex items-center gap-1.5 px-2 py-1 rounded-md text-[10px] font-mono transition-all cursor-pointer border ${
                isOverlayVisible && overlayOpacity > 0.01
                  ? 'bg-accent-cyan/15 text-accent-cyan border-accent-cyan/40 shadow-[0_0_8px_rgba(0,242,255,0.25)]'
                  : 'bg-white/5 text-slate-400 border-white/10 hover:text-white'
              }`}
              title={isOverlayVisible ? "Turn Sentinel overlay OFF" : "Turn Sentinel overlay ON"}
            >
              {isOverlayVisible && overlayOpacity > 0.01 ? (
                <Eye size={12} className="text-accent-cyan" />
              ) : (
                <EyeOff size={12} />
              )}
              <span className="hidden sm:inline">Overlay</span>
              <span className="font-bold">{isOverlayVisible && overlayOpacity > 0.01 ? 'ON' : 'OFF'}</span>
            </button>
          )}

          {/* Spectral Color Guide Modal Trigger */}
          {activeOverlay && (
            <button
              type="button"
              onClick={() => setShowColorLegend(!showColorLegend)}
              className={`flex items-center gap-1.5 px-2 py-1 rounded-md text-[10px] font-mono transition-all cursor-pointer border ${
                showColorLegend
                  ? 'bg-accent-warm/20 text-accent-warm border-accent-warm/40 shadow-[0_0_8px_rgba(255,170,0,0.3)]'
                  : 'bg-white/5 text-slate-300 border-white/10 hover:text-white'
              }`}
              title="Spectral Color Guide: Learn what Red, Blue, Green, and Black mean"
            >
              <Palette size={12} className={showColorLegend ? "text-accent-warm" : "text-accent-cyan"} />
              <span className="hidden md:inline">Color Guide</span>
            </button>
          )}

          {/* Google Maps Style Basemap Switcher */}
          <div className="flex items-center gap-0.5 bg-space-black/90 border border-white/15 p-0.5 rounded-lg text-[10px] font-mono">
            {(['satellite', 'hybrid', 'streets', 'sar'] as BaseLayerType[]).map((layer) => (
              <button
                key={layer}
                type="button"
                onClick={() => handleBaseLayerChange(layer)}
                className={`px-2 py-1 rounded transition-colors cursor-pointer capitalize ${
                  activeBaseLayer === layer
                    ? 'bg-accent-cyan text-space-black font-semibold shadow-sm'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                {layer === 'sar' ? 'SAR Dark' : layer}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Main Map Canvas Area with Filter Effects */}
      <div className="relative flex-1 bg-space-navy overflow-hidden">
        {/* Real MapLibre WebGL Canvas */}
        <div
          ref={mapContainerRef}
          style={getBandFilterStyle()}
          className="absolute inset-0 w-full h-full cursor-grab active:cursor-grabbing transition-all duration-300"
        />

        {/* Optional HUD Reticle */}
        {showReticle && (
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none opacity-40">
            <Crosshair size={90} className="text-accent-cyan stroke-[0.8]" />
          </div>
        )}

        {/* Top-Center Georeferenced Scene Telemetry Banner */}
        {activeOverlay && activeOverlay.bounds && (
          <div className="absolute top-3 left-1/2 -translate-x-1/2 z-10 pointer-events-auto">
            <div className="bg-space-black/90 backdrop-blur-md border border-accent-cyan/40 px-3.5 py-1.5 rounded-lg text-[10px] font-mono text-white shadow-[0_0_20px_rgba(0,242,255,0.25)] flex items-center gap-3">
              <div className="flex items-center gap-1.5 text-accent-cyan">
                <Focus size={13} className="animate-pulse" />
                <span className="font-bold uppercase tracking-wider">{activeOverlay.sensor.split(' ')[0]}</span>
              </div>
              <span className="text-white/30">|</span>
              <span className="text-slate-300 font-medium truncate max-w-[150px] sm:max-w-xs">{activeOverlay.name}</span>
              <span className="text-white/30 hidden sm:inline">|</span>
              <span className="text-slate-400 hidden sm:inline">{activeOverlay.resolution}</span>
              {activeOverlay.areaSqKm && (
                <>
                  <span className="text-white/30 hidden md:inline">|</span>
                  <span className="text-accent-cyan hidden md:inline">{activeOverlay.areaSqKm} km²</span>
                </>
              )}
              <button
                type="button"
                onClick={handleFlyToScene}
                className="ml-1 px-2 py-0.5 rounded bg-accent-cyan/20 hover:bg-accent-cyan hover:text-space-black text-accent-cyan transition-colors cursor-pointer text-[9px] uppercase font-bold"
                title="Fly directly to satellite image footprint"
              >
                Center Scene
              </button>
            </div>
          </div>
        )}

        {/* Live Telemetry HUD (Top Left) */}
        <div className="absolute top-3 left-3 flex flex-col gap-1.5 z-10 pointer-events-none">
          <div className="bg-space-black/85 backdrop-blur-md border border-white/10 px-3 py-2 rounded-lg text-[10px] font-mono text-slate-300 shadow-xl space-y-1 pointer-events-auto min-w-[170px]">
            {cursorCoords ? (
              <div className="flex justify-between gap-3 border-b border-white/10 pb-1">
                <span className="text-accent-cyan font-semibold">CURSOR</span>
                <span className="text-accent-cyan">{cursorCoords.lat}°N, {cursorCoords.lng}°E</span>
              </div>
            ) : (
              <div className="flex justify-between gap-3 border-b border-white/10 pb-1">
                <span className="text-white/40">CENTER</span>
                <span>{centerCoords.lat}°N, {centerCoords.lng}°E</span>
              </div>
            )}
            <div className="flex justify-between gap-3">
              <span className="text-white/40">ZOOM</span>
              <span>{zoomLevel}x</span>
            </div>
            <div className="flex justify-between gap-3">
              <span className="text-white/40">PITCH / TILT</span>
              <span>{pitch}°</span>
            </div>
            <div
              onClick={() => activeOverlay && setIsOverlayVisible(!isOverlayVisible)}
              className={`flex justify-between gap-3 ${activeOverlay ? 'cursor-pointer hover:text-accent-cyan transition-colors' : ''}`}
              title={activeOverlay ? 'Click to toggle overlay visibility' : undefined}
            >
              <span className="text-white/40 flex items-center gap-1">
                OVERLAY
                {activeOverlay && (
                  isOverlayVisible && overlayOpacity > 0.01 ? (
                    <Eye size={10} className="text-accent-cyan" />
                  ) : (
                    <EyeOff size={10} className="text-slate-500" />
                  )
                )}
              </span>
              <span className={activeOverlay && isOverlayVisible && overlayOpacity > 0.01 ? 'text-accent-cyan font-semibold' : 'text-slate-500 font-semibold'}>
                {activeOverlay
                  ? !isOverlayVisible
                    ? 'OFF (Hidden)'
                    : overlayOpacity <= 0.01
                    ? '0% (Hidden)'
                    : `${Math.round(overlayOpacity * 100)}%`
                  : 'None'}
              </span>
            </div>
          </div>
        </div>

        {/* Floating Google Maps-Style Control Stack (Top Right) */}
        <div className="absolute top-3 right-3 flex flex-col gap-1.5 z-10">
          <div className="bg-space-black/85 backdrop-blur-md border border-white/10 rounded-lg p-1 flex flex-col gap-1 text-slate-300 shadow-xl">
            {/* Zoom In */}
            <button
              type="button"
              onClick={handleZoomIn}
              className="p-1.5 hover:bg-white/10 hover:text-accent-cyan rounded transition-colors cursor-pointer"
              title="Zoom in (+)"
            >
              <ZoomIn size={16} />
            </button>

            {/* Zoom Out */}
            <button
              type="button"
              onClick={handleZoomOut}
              className="p-1.5 hover:bg-white/10 hover:text-accent-cyan rounded transition-colors cursor-pointer"
              title="Zoom out (-)"
            >
              <ZoomOut size={16} />
            </button>

            <div className="h-px bg-white/10 mx-1" />

            {/* Compass / Reset North & Tilt */}
            <button
              type="button"
              onClick={handleResetNorthPitch}
              className="p-1.5 hover:bg-white/10 hover:text-accent-cyan rounded transition-transform cursor-pointer relative flex items-center justify-center"
              title="Reset North & 2D Flat View"
            >
              <div style={{ transform: `rotate(${-bearing}deg)` }} className="transition-transform duration-150">
                <Compass size={16} className={bearing !== 0 ? 'text-accent-cyan' : 'text-slate-400'} />
              </div>
            </button>

            {/* Locate Me (GPS) */}
            <button
              type="button"
              onClick={handleLocateMe}
              className="p-1.5 hover:bg-white/10 hover:text-accent-cyan rounded transition-colors cursor-pointer"
              title="Fly to My Current Location"
            >
              <Locate size={16} />
            </button>

            {/* Reticle Toggle */}
            <button
              type="button"
              onClick={() => setShowReticle(!showReticle)}
              className={`p-1.5 hover:bg-white/10 rounded transition-colors cursor-pointer ${
                showReticle ? 'text-accent-cyan bg-accent-cyan/15' : 'text-slate-400'
              }`}
              title="Toggle HUD Crosshair Reticle"
            >
              <Crosshair size={16} />
            </button>

            {/* Overlay Controls Toggle */}
            {activeOverlay && (
              <button
                type="button"
                onClick={() => setShowOverlayControls(!showOverlayControls)}
                className={`p-1.5 hover:bg-white/10 rounded transition-colors cursor-pointer ${
                  showOverlayControls ? 'text-accent-cyan bg-accent-cyan/15' : 'text-slate-400'
                }`}
                title="Toggle Sentinel Overlay Controls"
              >
                <Sliders size={16} />
              </button>
            )}

            <div className="h-px bg-white/10 mx-1" />

            {/* Fullscreen Toggle */}
            <button
              type="button"
              onClick={toggleFullscreen}
              className="p-1.5 hover:bg-white/10 hover:text-accent-cyan rounded transition-colors cursor-pointer"
              title={isFullscreen ? 'Exit Fullscreen' : 'Fullscreen Map'}
            >
              {isFullscreen ? <Minimize2 size={16} /> : <Maximize2 size={16} />}
            </button>
          </div>
        </div>

        {/* WOW FACTOR: Multi-Temporal Before/After Swipe & Blend Control Dock (Bottom Center) */}
        {activeOverlay && showOverlayControls && (
          <div className="absolute bottom-4 left-1/2 -translate-x-1/2 z-10 w-11/12 max-w-xl">
            <div className="bg-space-black/95 backdrop-blur-xl border border-white/20 p-2.5 rounded-xl shadow-2xl space-y-2">
              {/* Top Row: Temporal Mode Switcher & Opacity */}
              <div className="flex flex-wrap items-center justify-between gap-2 text-[10px] font-mono">
                {/* Mode Selector */}
                <div className="flex items-center gap-1.5">
                  <span className="text-slate-400 uppercase tracking-wider flex items-center gap-1">
                    <Layers size={11} className="text-accent-cyan" />
                    Mode:
                  </span>

                  {activeOverlay.mode === 'bi-temporal' && activeOverlay.t2ImageUrl ? (
                    <div className="flex items-center gap-0.5 bg-white/5 p-0.5 rounded-lg border border-white/10">
                      <button
                        type="button"
                        onClick={() => {
                          setTemporalMode('fade');
                          setIsBlinking(false);
                        }}
                        className={`px-2 py-0.5 rounded transition-colors cursor-pointer ${
                          temporalMode === 'fade' && !isBlinking
                            ? 'bg-accent-cyan text-space-black font-semibold'
                            : 'text-slate-400 hover:text-white'
                        }`}
                      >
                        Swipe/Blend
                      </button>
                      <button
                        type="button"
                        onClick={() => {
                          setTemporalMode('t1');
                          setIsBlinking(false);
                        }}
                        className={`px-2 py-0.5 rounded transition-colors cursor-pointer ${
                          temporalMode === 't1' && !isBlinking
                            ? 'bg-accent-cyan text-space-black font-semibold'
                            : 'text-slate-400 hover:text-white'
                        }`}
                      >
                        T1 (Pre)
                      </button>
                      <button
                        type="button"
                        onClick={() => {
                          setTemporalMode('t2');
                          setIsBlinking(false);
                        }}
                        className={`px-2 py-0.5 rounded transition-colors cursor-pointer ${
                          temporalMode === 't2' && !isBlinking
                            ? 'bg-accent-cyan text-space-black font-semibold'
                            : 'text-slate-400 hover:text-white'
                        }`}
                      >
                        T2 (Post)
                      </button>
                      <button
                        type="button"
                        onClick={() => setIsBlinking(!isBlinking)}
                        className={`px-2 py-0.5 rounded transition-colors cursor-pointer flex items-center gap-1 ${
                          isBlinking
                            ? 'bg-accent-warm text-space-black font-bold animate-pulse'
                            : 'text-slate-400 hover:text-white'
                        }`}
                        title="Blink comparison toggles T1 and T2 at 1.2Hz"
                      >
                        {isBlinking ? <Pause size={10} /> : <Play size={10} />}
                        Blink
                      </button>
                    </div>
                  ) : (
                    <span className="text-accent-cyan font-semibold">Single Georeferenced Scene</span>
                  )}
                </div>

                {/* Opacity Slider & Quick Toggle */}
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() => setIsOverlayVisible(!isOverlayVisible)}
                    className={`px-1.5 py-0.5 rounded text-[9px] font-mono flex items-center gap-1 cursor-pointer transition-colors ${
                      isOverlayVisible && overlayOpacity > 0.01
                        ? 'bg-accent-cyan/20 text-accent-cyan border border-accent-cyan/40'
                        : 'bg-white/10 text-slate-400 border border-white/10 hover:text-white'
                    }`}
                    title={isOverlayVisible ? 'Turn overlay OFF' : 'Turn overlay ON'}
                  >
                    {isOverlayVisible && overlayOpacity > 0.01 ? <Eye size={10} /> : <EyeOff size={10} />}
                    <span>{isOverlayVisible && overlayOpacity > 0.01 ? 'ON' : 'OFF'}</span>
                  </button>

                  <span className="text-slate-400">Opacity:</span>
                  <input
                    type="range"
                    min="0"
                    max="100"
                    value={isOverlayVisible ? Math.round(overlayOpacity * 100) : 0}
                    onChange={(e) => {
                      const val = parseFloat(e.target.value) / 100;
                      setOverlayOpacity(val);
                      if (!isOverlayVisible && val > 0) setIsOverlayVisible(true);
                    }}
                    className="w-20 accent-accent-cyan cursor-pointer"
                  />
                  <span className="text-accent-cyan w-7 text-right">
                    {!isOverlayVisible ? '0%' : `${Math.round(overlayOpacity * 100)}%`}
                  </span>
                </div>
              </div>

              {/* Bottom Row: Bi-Temporal Split Swipe Slider (if bi-temporal) */}
              {activeOverlay.mode === 'bi-temporal' && activeOverlay.t2ImageUrl && (
                <div className="pt-1 border-t border-white/10 flex items-center gap-3">
                  <span className="text-[10px] font-mono text-slate-400 shrink-0 flex items-center gap-1">
                    <ArrowLeftRight size={11} className="text-accent-cyan" />
                    T1 (Before)
                  </span>

                  <div className="relative flex-1 flex items-center">
                    <input
                      type="range"
                      min="0"
                      max="100"
                      value={temporalSlider}
                      onChange={(e) => {
                        setTemporalSlider(parseFloat(e.target.value));
                        if (temporalMode !== 'fade') setTemporalMode('fade');
                      }}
                      className="w-full accent-accent-cyan cursor-pointer h-2 bg-space-navy rounded-lg"
                    />
                  </div>

                  <span className="text-[10px] font-mono text-slate-400 shrink-0">
                    T2 (After)
                  </span>
                </div>
              )}

              {/* Band Visualization Presets Pill Box */}
              <div className="pt-1 border-t border-white/10 flex items-center justify-between text-[9px] font-mono">
                <span className="text-slate-400 flex items-center gap-1">
                  <Sparkles size={11} className="text-accent-warm" />
                  Band Mode:
                </span>
                <div className="flex items-center gap-1">
                  {(
                    [
                      ['true-color', 'RGB True Color'],
                      ['false-color-nir', 'NIR Infrared (Canopy)'],
                      ['sar-contrast', 'SAR Radar Facets'],
                      ['edge-boost', 'Edge Boost']
                    ] as [BandFilterType, string][]
                  ).map(([presetKey, label]) => (
                    <button
                      key={presetKey}
                      type="button"
                      onClick={() => setBandPreset(presetKey)}
                      className={`px-1.5 py-0.5 rounded transition-colors cursor-pointer ${
                        bandPreset === presetKey
                          ? 'bg-white/20 text-white font-semibold border border-white/30'
                          : 'text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      {label}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Corner Coordinates HUD Chips (Only when overlay is active and visible) */}
        {activeOverlay?.bounds && isOverlayVisible && overlayOpacity > 0.01 && (
          <>
            {/* Top-Left: NW */}
            <div className="absolute top-16 left-3 z-10 pointer-events-none hidden sm:block">
              <div className="bg-space-black/75 border border-accent-cyan/30 px-1.5 py-0.5 rounded text-[9px] font-mono text-slate-300 backdrop-blur">
                NW: {activeOverlay.bounds[3].toFixed(4)}°N, {activeOverlay.bounds[0].toFixed(4)}°E
              </div>
            </div>
            {/* Bottom-Left: SW */}
            <div className="absolute bottom-16 left-3 z-10 pointer-events-none hidden sm:block">
              <div className="bg-space-black/75 border border-accent-cyan/30 px-1.5 py-0.5 rounded text-[9px] font-mono text-slate-300 backdrop-blur">
                SW: {activeOverlay.bounds[1].toFixed(4)}°N, {activeOverlay.bounds[0].toFixed(4)}°E
              </div>
            </div>
          </>
        )}

        {/* Remote Sensing Spectral Color Legend Modal */}
        {showColorLegend && (
          <div className="absolute top-14 right-3 md:right-16 z-30 w-84 max-w-[calc(100vw-2rem)] bg-space-black/95 backdrop-blur-xl border border-accent-cyan/30 rounded-xl shadow-2xl p-3.5 text-white font-mono animate-in fade-in zoom-in-95 duration-200">
            <div className="flex items-center justify-between border-b border-white/10 pb-2 mb-2.5">
              <div className="flex items-center gap-2 text-xs font-bold text-accent-cyan">
                <Palette size={14} />
                <span>SENTINEL SPECTRAL COLOR GUIDE</span>
              </div>
              <button
                type="button"
                onClick={() => setShowColorLegend(false)}
                className="text-slate-400 hover:text-white p-0.5 rounded cursor-pointer"
              >
                <X size={14} />
              </button>
            </div>

            <p className="text-[10px] text-slate-300 leading-relaxed mb-3">
              Satellite multi-spectral sensors capture wavelengths beyond human vision (e.g. Near-Infrared B8, Shortwave Infrared). Here is what the overlay colors indicate:
            </p>

            <div className="space-y-2 text-[10px]">
              <div className="flex items-start gap-2.5 bg-red-500/10 border border-red-500/30 rounded-lg p-2">
                <span className="w-3 h-3 rounded-full bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.8)] shrink-0 mt-0.5" />
                <div>
                  <div className="font-bold text-red-400">RED / CRIMSON / PINK</div>
                  <div className="text-slate-300">
                    <span className="font-semibold text-white">Dense Healthy Vegetation & Canopy:</span> Living plant chlorophyll strongly reflects Near-Infrared (NIR / Band 8). In standard Color-Infrared (CIR) composites, farmlands, forest reserves (Delhi Ridge), and parks glow bright red.
                  </div>
                </div>
              </div>

              <div className="flex items-start gap-2.5 bg-cyan-500/10 border border-cyan-500/30 rounded-lg p-2">
                <span className="w-3 h-3 rounded-full bg-cyan-400 shadow-[0_0_8px_rgba(34,211,238,0.8)] shrink-0 mt-0.5" />
                <div>
                  <div className="font-bold text-cyan-300">CYAN / SLATE BLUE / GRAY</div>
                  <div className="text-slate-300">
                    <span className="font-semibold text-white">Urban Infrastructure & Concrete:</span> Impervious surfaces, buildings, asphalt roadways, airport runways, and industrial complexes reflect visible light evenly without vegetative NIR reflectance.
                  </div>
                </div>
              </div>

              <div className="flex items-start gap-2.5 bg-blue-950/40 border border-blue-500/30 rounded-lg p-2">
                <span className="w-3 h-3 rounded-full bg-blue-900 border border-blue-400 shadow-[0_0_8px_rgba(59,130,246,0.6)] shrink-0 mt-0.5" />
                <div>
                  <div className="font-bold text-blue-300">DEEP NAVY / BLACK</div>
                  <div className="text-slate-300">
                    <span className="font-semibold text-white">Open Water Bodies:</span> Clear water absorbs Near-Infrared radiation completely, resulting in deep navy or black signatures for the Yamuna River, reservoirs, and lakes.
                  </div>
                </div>
              </div>

              <div className="flex items-start gap-2.5 bg-amber-500/10 border border-amber-500/30 rounded-lg p-2">
                <span className="w-3 h-3 rounded-full bg-amber-400 shadow-[0_0_8px_rgba(251,191,36,0.8)] shrink-0 mt-0.5" />
                <div>
                  <div className="font-bold text-amber-300">OCHRE / TAN / BROWN</div>
                  <div className="text-slate-300">
                    <span className="font-semibold text-white">Bare Soil & Sand:</span> Fallow agricultural ground, desert sands, construction excavations, and river sediment deposits.
                  </div>
                </div>
              </div>
            </div>

            <div className="mt-3 pt-2 border-t border-white/10 flex justify-between items-center text-[9px] text-slate-400">
              <span>Sentinel-2 MSI (B4, B3, B2, B8)</span>
              <button
                type="button"
                onClick={() => setShowColorLegend(false)}
                className="text-accent-cyan hover:underline cursor-pointer"
              >
                Close Guide
              </button>
            </div>
          </div>
        )}

        {/* Global Explorer Telemetry Badge (Bottom Right) */}
        <div className="absolute bottom-4 right-4 z-10 hidden sm:block">
          <div className="bg-space-black/85 backdrop-blur-md border border-white/10 px-3 py-1.5 rounded-lg text-[10px] font-mono text-white/70 shadow-xl flex items-center gap-2.5">
            <div
              className={`w-2 h-2 rounded-full ${
                activeOverlay ? 'bg-accent-cyan animate-pulse shadow-[0_0_8px_rgba(0,242,255,0.8)]' : 'bg-accent-teal'
              }`}
            />
            <span>
              {activeOverlay ? `Sentinel Overlay • ${activeOverlay.sensor.split(' ')[0]}` : 'Global Slippy Map'}
            </span>

            {detectionCount > 0 && (
              <>
                <span className="text-white/30">|</span>
                <button
                  type="button"
                  onClick={() => setShowBoundingBoxes(!showBoundingBoxes)}
                  className={`px-1.5 py-0.5 rounded cursor-pointer transition-colors ${
                    showBoundingBoxes ? 'bg-accent-cyan/20 text-accent-cyan' : 'text-slate-500 line-through'
                  }`}
                  title="Toggle AI visual detection boundaries"
                >
                  AI Targets: {detectionCount}
                </button>
              </>
            )}

            <span className="text-white/30">|</span>
            <span className="text-slate-400">MapLibre WebGL</span>
          </div>
        </div>
      </div>
    </Panel>
  );
};

export default MapViewport;
