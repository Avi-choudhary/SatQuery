import React, { useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { UploadCloud, CheckCircle2, FileCode, Sparkles } from 'lucide-react';
import { Badge } from '../ui/Badge';
import { useTrace, type SentinelOverlay } from '../../context/TraceContext';
import { uploadImagery, type ImageryUploadResponse } from '../../lib/api';

export interface UploadedDataset {
  name: string;
  size: string;
  sensor: 'Optical (Sentinel-2)' | 'SAR (Sentinel-1)' | 'Fused (Optical+SAR)';
  mode: 'single' | 'bi-temporal';
  projection: string;
  resolution: string;
  file?: File;
  files?: File[];
  t1Filename?: string;
  t2Filename?: string | null;
}

interface UploadDropzoneProps {
  onDatasetSelect?: (dataset: UploadedDataset) => void;
}

export const PRESET_OVERLAYS: Record<string, SentinelOverlay> = {
  bengaluru_urban: {
    name: 'Bengaluru_Urban_Corridor_T1_T2.tif',
    sensor: 'Optical (Sentinel-2)',
    mode: 'bi-temporal',
    bounds: [77.618, 13.022, 77.652, 13.048],
    center: [77.635, 13.035],
    crs: 'EPSG:4326 (WGS84)',
    resolution: '10.0m GSD',
    areaSqKm: 10.5,
    t1ImageUrl: 'http://localhost:8000/static/Bengaluru_T1_Pre.png',
    t2ImageUrl: 'http://localhost:8000/static/Bengaluru_T2_Post.png',
    t1Filename: 'Bengaluru_T1_Pre.png',
    t2Filename: 'Bengaluru_T2_Post.png'
  },
  delhi_temporal: {
    name: 'Delhi_MultiYear_2018_2026_Pair.tif',
    sensor: 'Optical (Sentinel-2 Multi-Temporal)',
    mode: 'bi-temporal',
    bounds: [77.265, 28.580, 77.365, 28.690],
    center: [77.315, 28.635],
    crs: 'EPSG:32643 (UTM 43N)',
    resolution: '10.0m GSD',
    areaSqKm: 124.5,
    t1ImageUrl: 'http://localhost:8000/static/prev_east_delhi_2018_S2.png',
    t2ImageUrl: 'http://localhost:8000/static/prev_delhi_20260112_S2.png',
    t1Filename: 'east_delhi_2018_S2.tif',
    t2Filename: 'delhi_20260112_S2.tif'
  },
  delhi_s2: {
    name: 'delhi_20260112_S2.tif',
    sensor: 'Optical (Sentinel-2)',
    mode: 'single',
    bounds: [77.044156, 28.545082, 77.356719, 28.854927],
    center: [77.200438, 28.700004],
    crs: 'EPSG:32643 (UTM 43N)',
    resolution: '10.0m GSD',
    areaSqKm: 1050.4,
    t1ImageUrl: 'http://localhost:8000/static/delhi_preview.png',
    t1Filename: 'delhi_preview.png'
  },
  mangalore_sar: {
    name: 'Mangalore_Harbor_SAR_VV.tif',
    sensor: 'SAR (Sentinel-1)',
    mode: 'single',
    bounds: [74.780, 12.850, 74.880, 12.950],
    center: [74.830, 12.900],
    crs: 'EPSG:4326 (WGS84)',
    resolution: '10.0m GSD',
    areaSqKm: 118.2,
    t1ImageUrl: 'http://localhost:8000/static/Mangalore_SAR_VV.png',
    t1Filename: 'Mangalore_SAR_VV.png'
  }
};

export const UploadDropzone: React.FC<UploadDropzoneProps> = ({ onDatasetSelect }) => {
  const {
    setActiveOverlay,
    setActiveDatasetName,
    setActiveSensor,
    activeUploadedDataset,
    setActiveUploadedDataset
  } = useTrace();
  const [isDragging, setIsDragging] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<number | null>(null);
  const [statusText, setStatusText] = useState<string>('Ingesting GeoTIFF metadata...');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const activeDataset = activeUploadedDataset;

  const applyOverlayAndDataset = (dataset: UploadedDataset, overlay: SentinelOverlay) => {
    setActiveUploadedDataset(dataset);
    setActiveDatasetName(dataset.name);
    setActiveSensor(dataset.sensor);
    setActiveOverlay(overlay);
    onDatasetSelect?.(dataset);
  };

  const loadPreset = (presetKey: 'bengaluru_urban' | 'delhi_s2' | 'mangalore_sar' | 'delhi_temporal') => {
    const overlay = PRESET_OVERLAYS[presetKey];
    if (!overlay) return;

    const dataset: UploadedDataset = {
      name: overlay.name,
      size: presetKey === 'delhi_s2' ? '117.4 MB' : (presetKey === 'delhi_temporal' ? '129.9 MB' : (presetKey === 'bengaluru_urban' ? '148.2 MB' : '48.6 MB')),
      sensor: overlay.sensor as any,
      mode: overlay.mode,
      projection: overlay.crs,
      resolution: overlay.resolution,
    };

    applyOverlayAndDataset(dataset, overlay);
  };

  const handleUploadFiles = async (files: FileList | File[]) => {
    if (!files || files.length === 0) return;

    const fileList = Array.from(files);
    const primaryFile = fileList[0];
    const isBiTemporal = fileList.length > 1;
    const isTiff = primaryFile.name.toLowerCase().endsWith('.tif') || primaryFile.name.toLowerCase().endsWith('.tiff');

    setUploadProgress(20);
    setStatusText(isTiff ? 'Reading GeoTIFF metadata & extracting raster bands...' : 'Processing satellite imagery...');

    // Immediately register the dataset in state so queries sent right after drop bind to this scene
    const initialDataset: UploadedDataset = {
      name: primaryFile.name,
      size: `${(primaryFile.size / (1024 * 1024)).toFixed(1)} MB`,
      sensor: (isTiff ? 'Optical (Sentinel-2)' : 'Optical (Satellite)') as any,
      mode: isBiTemporal ? 'bi-temporal' : 'single',
      projection: 'EPSG:4326 (WGS84)',
      resolution: '10.0m GSD',
      file: primaryFile,
      files: fileList.length > 1 ? fileList : [primaryFile],
      t1Filename: primaryFile.name,
      t2Filename: fileList[1]?.name
    };
    setActiveUploadedDataset(initialDataset);
    setActiveDatasetName(primaryFile.name);
    setActiveSensor(initialDataset.sensor);
    onDatasetSelect?.(initialDataset);

    // If standard web image (PNG/JPG), create instant local blob preview
    if (!isTiff) {
      const instantUrlT1 = URL.createObjectURL(primaryFile);
      const instantUrlT2 = fileList[1] ? URL.createObjectURL(fileList[1]) : undefined;
      const defaultBounds: [number, number, number, number] = [77.618, 13.022, 77.652, 13.048];

      const instantOverlay: SentinelOverlay = {
        name: primaryFile.name,
        sensor: 'Optical (Satellite)',
        mode: isBiTemporal ? 'bi-temporal' : 'single',
        bounds: defaultBounds,
        center: [(defaultBounds[0] + defaultBounds[2]) / 2, (defaultBounds[1] + defaultBounds[3]) / 2],
        crs: 'EPSG:4326 (WGS84)',
        resolution: '10.0m GSD',
        areaSqKm: 10.5,
        t1ImageUrl: instantUrlT1,
        t2ImageUrl: instantUrlT2,
        t1Filename: primaryFile.name,
        t2Filename: fileList[1]?.name
      };

      applyOverlayAndDataset(initialDataset, instantOverlay);
    }

    // Process with backend GIS pipeline to extract georeferenced bounds & fresh RGB preview
    try {
      setUploadProgress(50);
      const resp: ImageryUploadResponse = await uploadImagery(fileList);
      setUploadProgress(100);
      setTimeout(() => setUploadProgress(null), 400);

      const refinedOverlay: SentinelOverlay = {
        datasetId: resp.dataset_id,
        name: resp.name,
        sensor: resp.sensor,
        mode: resp.mode,
        bounds: resp.wgs84_bounds,
        center: resp.center,
        crs: resp.crs,
        resolution: resp.resolution,
        areaSqKm: resp.area_sq_km,
        t1ImageUrl: resp.t1_image_url,
        t2ImageUrl: resp.t2_image_url,
        t1Filename: resp.t1_filename,
        t2Filename: resp.t2_filename
      };

      const refinedDataset: UploadedDataset = {
        name: resp.name,
        size: `${(primaryFile.size / (1024 * 1024)).toFixed(1)} MB`,
        sensor: resp.sensor as any,
        mode: resp.mode,
        projection: resp.crs,
        resolution: resp.resolution,
        file: primaryFile,
        files: fileList.length > 1 ? fileList : [primaryFile],
        t1Filename: resp.t1_filename,
        t2Filename: resp.t2_filename
      };

      applyOverlayAndDataset(refinedDataset, refinedOverlay);
    } catch (err) {
      console.warn('Backend imagery upload notification:', err);
      setUploadProgress(100);
      setTimeout(() => setUploadProgress(null), 400);
    }
  };


  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleUploadFiles(e.dataTransfer.files);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      handleUploadFiles(e.target.files);
    }
  };

  return (
    <div className="space-y-3">
      {/* Dropzone Box */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        className={`relative border-2 border-dashed rounded-xl p-4 text-center cursor-pointer transition-all ${
          isDragging
            ? 'border-accent-cyan bg-accent-cyan/10 shadow-[0_0_20px_rgba(0,242,255,0.2)]'
            : 'border-white/15 bg-white/[0.02] hover:border-accent-cyan/40 hover:bg-white/[0.04]'
        }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".tif,.tiff,.geotiff,.png,.jpg,.jpeg"
          multiple
          className="hidden"
          onChange={handleFileChange}
        />

        <div className="flex flex-col items-center justify-center gap-2">
          <div className="w-10 h-10 rounded-full bg-accent-cyan/10 border border-accent-cyan/30 flex items-center justify-center text-accent-cyan">
            <UploadCloud size={20} />
          </div>

          <div>
            <p className="text-xs font-medium text-white">
              Drag & Drop <span className="text-accent-cyan">GeoTIFF</span> or Sentinel pairs
            </p>
            <p className="text-[10px] text-slate-400 mt-0.5">
              Overlays directly at exact geographic coordinates • Single or Bi-Temporal pairs
            </p>
          </div>
        </div>

        {/* Upload Progress Bar */}
        <AnimatePresence>
          {uploadProgress !== null && (
            <motion.div
              initial={{ opacity: 0, y: 5 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="absolute inset-x-3 bottom-2 bg-space-black/95 p-2 rounded-lg border border-accent-cyan/40 shadow-xl"
            >
              <div className="flex items-center justify-between text-[10px] font-mono text-accent-cyan mb-1">
                <span>{statusText}</span>
                <span>{uploadProgress}%</span>
              </div>
              <div className="w-full h-1.5 bg-white/10 rounded-full overflow-hidden">
                <div
                  className="h-full bg-accent-cyan transition-all duration-200 shadow-[0_0_8px_#00f2ff]"
                  style={{ width: `${uploadProgress}%` }}
                />
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Active Ingested Dataset Card */}
      {activeDataset && (
        <div className="bg-white/5 border border-white/10 rounded-lg p-3 text-xs">
          <div className="flex items-start justify-between gap-2 mb-2">
            <div className="flex items-center gap-2 min-w-0">
              <FileCode size={16} className="text-accent-cyan shrink-0" />
              <span className="font-mono text-white text-[11px] truncate" title={activeDataset.name}>
                {activeDataset.name}
              </span>
            </div>
            <Badge variant="success" className="text-[9px] uppercase px-1.5 py-0.5 shrink-0">
              <CheckCircle2 size={10} className="mr-1 inline" /> Overlay Active
            </Badge>
          </div>

          <div className="grid grid-cols-3 gap-1.5 text-[10px] font-mono text-slate-400 bg-space-black/40 p-2 rounded">
            <div>
              <span className="text-white/40 block text-[9px]">SENSOR</span>
              <span className="text-accent-cyan">{activeDataset.sensor.split(' ')[0]}</span>
            </div>
            <div>
              <span className="text-white/40 block text-[9px]">CRS</span>
              <span className="text-white">{activeDataset.projection.split(' ')[0]}</span>
            </div>
            <div>
              <span className="text-white/40 block text-[9px]">MODE</span>
              <span className="text-white capitalize">{activeDataset.mode}</span>
            </div>
          </div>
        </div>
      )}

      {/* Instant High-Interest Sentinel Presets */}
      <div className="flex flex-wrap items-center gap-1.5 pt-1">
        <Sparkles size={12} className="text-accent-warm shrink-0" />
        <span className="text-[10px] font-mono text-slate-400 uppercase tracking-wider">Presets:</span>

        <button
          type="button"
          onClick={() => loadPreset('bengaluru_urban')}
          className="text-[10px] font-mono px-2 py-0.5 rounded bg-white/5 hover:bg-accent-cyan/20 hover:text-accent-cyan text-slate-300 transition-colors cursor-pointer"
          title="Bengaluru Urban Corridor (Bi-Temporal T1 & T2 with Swipe Slider)"
        >
          Bengaluru (T1/T2 Pair)
        </button>

        <button
          type="button"
          onClick={() => loadPreset('delhi_temporal')}
          className="text-[10px] font-mono px-2 py-0.5 rounded bg-white/5 hover:bg-accent-cyan/20 hover:text-accent-cyan text-slate-300 transition-colors cursor-pointer"
          title="Delhi Multi-Year (Sentinel-2 2018 vs 2026 Bi-Temporal Pair with ChangeFormer)"
        >
          Delhi (2018/2026 Pair)
        </button>

        <button
          type="button"
          onClick={() => loadPreset('delhi_s2')}
          className="text-[10px] font-mono px-2 py-0.5 rounded bg-white/5 hover:bg-accent-cyan/20 hover:text-accent-cyan text-slate-300 transition-colors cursor-pointer"
          title="Delhi NCR (Real 117MB Sentinel-2 GeoTIFF)"
        >
          Delhi (Real 117MB COG)
        </button>

        <button
          type="button"
          onClick={() => loadPreset('mangalore_sar')}
          className="text-[10px] font-mono px-2 py-0.5 rounded bg-white/5 hover:bg-accent-cyan/20 hover:text-accent-cyan text-slate-300 transition-colors cursor-pointer"
          title="Mangalore Harbor (Sentinel-1 SAR Radar)"
        >
          SAR Radar (Port)
        </button>
      </div>
    </div>
  );
};

export default UploadDropzone;
