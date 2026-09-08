/**
 * SatQuery AI Frontend API Client
 * Connects to the local FastAPI backend (which routes remote GPU inference as needed).
 */

export const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1';

export interface SatQueryResponse {
  text_answer: string;
  response?: string;
  confidence?: number;
  visual_evidence: any[];
  execution_trace?: {
    steps?: string[];
    logs?: string[];
    summary?: string;
    [key: string]: any;
  };
  trace_log?: string[];
}

export interface SystemStatusResponse {
  service: string;
  status: string;
  execution_mode: string;
  model_engine: {
    remote_service_url?: string;
    remote_host_status?: string;
    remote_vram?: string;
    local_cuda_available?: boolean;
    local_device?: string;
    local_vram_gb?: number;
  };
  network: {
    lan_ip: string;
    port: number;
    api_endpoint: string;
  };
  specialists: Record<string, string>;
}

export async function fetchSystemStatus(): Promise<SystemStatusResponse> {
  const res = await fetch(`${API_BASE_URL}/status`, {
    headers: { 'Accept': 'application/json' },
  });
  if (!res.ok) {
    throw new Error(`Status check failed: HTTP ${res.status}`);
  }
  return res.json();
}

export async function submitSatQuery(
  query: string,
  file?: File | File[] | null,
  datasetName?: string
): Promise<SatQueryResponse> {
  const formData = new FormData();
  formData.append('query', query);
  if (Array.isArray(file)) {
    file.forEach((f) => formData.append('files', f));
  } else if (file) {
    formData.append('files', file);
  }
  if (datasetName) {
    formData.append('dataset_name', datasetName);
  }

  const res = await fetch(`${API_BASE_URL}/satquery`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const errorText = await res.text();
    throw new Error(`SatQuery API error (${res.status}): ${errorText}`);
  }

  return res.json();
}

export interface ImageryUploadResponse {
  dataset_id: string;
  name: string;
  sensor: string;
  mode: 'single' | 'bi-temporal';
  wgs84_bounds: [number, number, number, number];
  center: [number, number];
  crs: string;
  resolution: string;
  area_sq_km: number;
  t1_image_url: string;
  t2_image_url?: string | null;
  t1_filename?: string;
  t2_filename?: string | null;
}

export async function uploadImagery(files: File[], sensorType?: string): Promise<ImageryUploadResponse> {
  const formData = new FormData();
  files.forEach((f) => formData.append('files', f));
  if (sensorType) formData.append('sensor_type', sensorType);

  const res = await fetch(`${API_BASE_URL}/imagery/upload`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    throw new Error(`Upload failed: ${res.statusText}`);
  }
  return res.json();
}

export async function fetchImageryPresets(): Promise<ImageryUploadResponse[]> {
  const res = await fetch(`${API_BASE_URL}/imagery/presets`, {
    headers: { Accept: 'application/json' },
  });
  if (!res.ok) {
    throw new Error(`Failed to fetch presets: ${res.statusText}`);
  }
  return res.json();
}

