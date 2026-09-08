# SatQuery AI Frontend

This project is the frontend for SatQuery AI, an agentic vision-language assistant for satellite and SAR remote sensing image analysis.

## Core Features

- **Premium Space-Tech Aesthetic**: Dark-mode-first, minimalistic design inspired by premium space-tech products (Linear, SpaceX).
- **3D Hero Section**: Interactive 3D Earth/satellite orbit visualization using React Three Fiber and Drei.
- **Interactive AI Console**:
  - **Map Viewport**: Placeholder for MapLibre GL JS with mock satellite imagery and overlays.
  cap **Query Panel**: Natural language interface with file upload (GeoTIFF) and chat history.
  - **Execution Trace Panel**: Glassmorphic technical readout of the agentic pipeline execution.
- **Agentic Pipeline Visualization**: Scroll-driven animation showing the workflow from upload to results.
- **Capability Showcase**: Feature grid for VQA, Visual Grounding, and Change Detection.

## Tech Stack

- **Framework**: React (Vite) + TypeScript
- **Styling**: Tailwind CSS
- **Animation**: Framer Motion
- **3D Graphics**: React Three Fiber + Drei
- **Mapping**: MapLibre GL JS (placeholder)

## Development Notes

- All backend-dependent components are marked with `// TODO(backend):`.
- Mock data is stored in `src/mocks/` to facilitate easy integration with a FastAPI backend later.
- UI primitives are located in `src/components/ui/`.
