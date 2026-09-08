# SatQuery AI 🛰️

> **Multimodal Agentic Geospatial Intelligence & Remote Sensing Assistant**  
> *Smart India Hackathon 2026 | ISRO Problem Statement 26167*

---

## 🌟 Overview

**SatQuery AI** is an end-to-end multimodal geospatial analysis platform designed to process, analyze, and query multi-sensor Earth observation data through natural language. Built for high-resolution satellite remote sensing, SatQuery combines cutting-edge Vision-Language Models (VLMs), bi-temporal transformer-based change detection, and automated GIS pipelines for Sentinel-1 (SAR) and Sentinel-2 (Optical) imagery.

### Key Capabilities

1. **Multimodal Satellite VQA & Reasoning**: Powered by a custom fine-tuned **Qwen3-VL-2B** remote-sensing specialist model capable of complex geospatial queries, object identification, and multi-spectral interpretation.
2. **Bi-Temporal Change Detection**: Powered by **ChangeFormerV6**, detecting ground changes across dual-epoch satellite passes with pixel-level precision.
3. **Automated GIS Extraction & Coregistration**: Automated pipeline for Sentinel-1 GRD and Sentinel-2 MSI data—performing radiometric calibration, reprojection to EPSG:4326 (WGS84), coregistration, and dynamic tiling.
4. **Visual Grounding & Vectorization**: Converts model detections into standards-compliant WGS84 GeoJSON polygons with bounding boxes, area calculations, and confidence scores.
5. **Real-time Map Console**: Interactive web application featuring MapLibre GL geospatial map layers, 3D globe visualization (Three.js), side-by-side swipe comparison, and telemetry tracking.

---

## 🏗️ Architecture & Monorepo Structure

```
SatQuery/
├── backend/
│   └── SatQuery-master/
│       └── backend/              # Central FastAPI Agentic Controller & API routes
│           ├── api/              # REST endpoints (/api/v1/satquery, /status)
│           ├── core/             # Application config and lifecycle
│           ├── schemas/          # Pydantic request/response schemas
│           ├── services/         # Agentic routing, VLM caller, ChangeFormer hook
│           ├── utils/            # GIS pre-processing and GeoJSON utilities
│           └── temp_uploads/     # Temporary file ingestion (git-ignored)
│
├── frontend/                     # React 19 + TypeScript + Vite Web Console
│   ├── src/
│   │   ├── components/           # MapLibre map, inspector, chat console, 3D globe
│   │   ├── pages/                # Analysis console, dashboard, settings
│   │   └── context/              # State management & WebSocket/HTTP services
│   └── package.json
│
├── gis_extraction_logic/         # Remote Sensing Data Engine
│   ├── scripts/                  # Sentinel-1 & Sentinel-2 export/pipeline scripts
│   ├── src/                      # SAR calibration, coregistration, optical processing
│   └── data/                     # Data tiers: raw/, interim/, processed/ (git-ignored)
│
├── Model Training/               # VLM Fine-Tuning & GPU Model Server
│   ├── 01_get_annotations.py     # Annotation harvesting from BigEarthNet
│   ├── 05_train_qwen3_vl.py      # Qwen3-VL LoRA fine-tuning script
│   ├── run_gpu_host.py           # Dedicated remote/local GPU inference service
│   ├── satquery_model.py         # PyTorch inference pipeline & memory optimization
│   └── output/                   # Model checkpoints & LoRA weights (git-ignored)
│
├── Bi-Temporal ChangeFormer/     # Change Detection Module
│   └── SatqueryAI/backend/       # ChangeFormerV6 inference service & wrappers
│
├── tests/                        # Full-stack end-to-end integration tests
│   └── test_full_stack_verification.py
│
├── requirements.txt              # Unified Python dependencies (CUDA 12.8 compatible)
├── run_local.bat                 # 1-Click full-stack launcher (Desktop PC mode)
├── run_laptop.bat                # 1-Click frontend + backend launcher (Remote GPU mode)
└── start_gpu_host.bat            # 1-Click GPU inference host launcher
```

---

## 🚀 Quick Start

### 1. Prerequisites

- **Operating System**: Windows 10/11 or Linux
- **Python**: Python 3.10 to 3.13 (Python 3.13 recommended with CUDA 12.8 PyTorch)
- **Node.js**: Node.js 18+ (LTS) & npm
- **GPU (Optional but recommended)**: NVIDIA GPU with CUDA support for accelerated local VLM inference.

### 2. Environment Setup

Copy `.env.example` to `.env`:

```bash
# In repository root:
cp .env.example .env

# In backend/SatQuery-master/backend:
cp backend/SatQuery-master/backend/.env.example backend/SatQuery-master/backend/.env
```

Configuration parameters:
```env
# Set empty to run directly on the local GPU, or provide remote URL:
MODEL_SERVICE_URL=
HOST=0.0.0.0
PORT=8000
```

### 3. Python Dependencies

Create and activate a virtual environment, then install the unified requirements:

```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies (CUDA-enabled torch if using NVIDIA GPU):
pip install -r requirements.txt
```

### 4. Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

---

## ⚡ Running the Platform

We provide convenient batch scripts for multi-modal orchestration:

| Launcher Script | Description |
| :--- | :--- |
| `run_local.bat` | Starts the **FastAPI backend** (:8000), launches the **Vite frontend** (:5173), and auto-opens `http://localhost:5173/console` on a single PC. |
| `run_laptop.bat` | Starts the backend and frontend configured to route GPU inference calls to a remote GPU host. |
| `start_gpu_host.bat` | Starts the dedicated **Qwen3-VL GPU inference host** on a machine equipped with high-VRAM NVIDIA GPUs. |

Alternatively, start services individually:

```bash
# Terminal 1: Backend
cd backend/SatQuery-master/backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Frontend
cd frontend
npm run dev
```

---

## 📦 Model Weights & Checkpoints

Due to GitHub's file size policies (100 MB hard limit), model checkpoints and heavy datasets are not tracked in Git. 

To run offline inference, place trained model weights in their designated directories:

1. **Qwen3-VL Merged / LoRA Weights**:
   - Location: `Model Training/output/qwen3_vl_satquery_merged/`
   - Files: `model-*.safetensors`, `config.json`, `tokenizer.json`
2. **ChangeFormer V6 Checkpoint**:
   - Location: `Bi-Temporal ChangeFormer/SatqueryAI/backend/external/ChangeFormer/checkpoints/`
   - File: `changeformer_checkpoint.pth` (or `CD_ChangeFormerV6.pth`)

---

## 🧪 Testing & Verification

Run the end-to-end integration test suite to verify FastAPI endpoints, GIS metadata extraction, CRS reprojection, and agentic tool routing:

```bash
python tests/test_full_stack_verification.py
```

---

## 📄 License & Attribution

- Developed for **Smart India Hackathon (SIH 2026)** — ISRO Problem Statement 26167.
- External model architectures credit to their respective authors: [Qwen-VL Team](https://github.com/QwenLM/Qwen-VL) and [ChangeFormer](https://github.com/wgcban/ChangeFormer).
