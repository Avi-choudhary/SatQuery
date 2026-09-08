# Claude Code Prompt — SatQuery AI Frontend (Design Phase)

Copy everything below into Claude Code in your project's terminal.

---

## Prompt

I'm building the frontend for **SatQuery AI**, an agentic vision-language assistant for satellite and SAR remote sensing image analysis, built for Smart India Hackathon 2026 (Problem Statement 26167, sponsored by ISRO, Space Technology theme).

### What the product does
Users upload satellite imagery (optical or SAR, single images or bi-temporal pairs, as GeoTIFFs) and ask natural-language questions like "How much did the built-up area increase between these dates?" An Agentic Controller reads the query, routes it to the right specialist AI tool (VQA, visual grounding, or change detection), processes the imagery, and returns a conversational answer plus visual evidence — bounding boxes, change masks, or heatmaps — overlaid directly on an interactive map, alongside an auditable execution log showing which models were used and why.

### What I need right now
Only the **frontend design and static UI shell** — no backend logic, no real model calls, no real GIS processing. Everything backend-dependent should be built as clean, clearly-labeled placeholder components with realistic mock data, ready for me to wire up to a FastAPI backend later.

### Design direction
Build something that looks like a premium, funded space-tech startup product — not a hackathon prototype. Think: the polish of Linear, Vercel, or a SpaceX/Planet Labs product page, crossed with a geospatial data tool.

- **Aesthetic**: dark-mode-first, minimalistic, generous whitespace, premium — not cluttered with dashboard chrome. Avoid generic "AI SaaS" gradients and stock Bootstrap-y layouts.
- **Color palette**: deep space navy/near-black background (#05070C-ish range), one crisp accent color evoking radar/satellite telemetry (cyan, electric teal, or a muted signal-green), and a warm accent used sparingly for highlights/alerts. High contrast, no more than 2 accent colors total.
- **Typography**: a clean geometric sans for headings (e.g. something in the Inter/Geist/Söhne family) paired with a monospace font for data, coordinates, model names, and the execution log — this reinforces the "technical instrument" feel.
- **3D elements**: a subtle rotating 3D Earth/satellite orbit visualization in the hero section (React Three Fiber / Three.js), performant and not distracting — low-poly or wireframe/glow style rather than photorealistic, so it matches the minimalist theme. Consider a scroll-linked camera movement that ties the 3D scene to the page's narrative as the user scrolls.
- **Motion**: use Framer Motion for scroll-triggered reveals, staggered fade/slide-ins on feature cards, smooth page transitions, and micro-interactions on buttons/inputs. Motion should feel precise and engineered, not bouncy or playful.
- **Glassmorphism used sparingly**: translucent panels with subtle blur/border only for overlay elements like the execution log or map legend — not applied everywhere.

### Tech stack
- React (Vite), TypeScript
- Tailwind CSS for styling
- Framer Motion for animation
- React Three Fiber + drei for the 3D hero element
- MapLibre GL JS wrapped in a placeholder component (no live tiles/data wiring yet — mock a map background/static state)
- Component structure should be clean and modular — one component per section, shared UI primitives (Button, Card, Badge, Panel) in a `components/ui/` folder

### Page structure

1. **Hero / Landing**
   - Full-viewport hero with the 3D satellite/Earth element as background or side visual
   - Headline framing SatQuery AI as an agentic assistant for satellite imagery Q&A
   - Subheadline explaining the core capability in one sentence
   - Primary CTA ("Try a Query" / "Launch Console") and secondary CTA ("How It Works")
   - Scroll-cue animation

2. **How It Works / Architecture**
   - Animated, scroll-driven visual walkthrough of the pipeline: Upload → Intent Parsing → GIS Pre-processing → Specialist AI Tool (VQA / Grounding / Change Detection) → Aggregator & Trace Logger → Results
   - Each stage should animate into view as the user scrolls, not just appear as a static diagram

3. **Interactive Console (placeholder)**
   - This is the core product screen — build it as the centerpiece, clearly marked as UI-only for now
   - Layout: left/right split — chat-style query panel on one side, map viewport on the other
   - **Map placeholder**: a component named clearly (e.g. `MapViewport.tsx`) with a mock static satellite-image background, placeholder bounding-box/heatmap overlays using dummy coordinates, and comments marking exactly where MapLibre GL initialization, GeoJSON overlay rendering, and live tile/layer props will be wired in
   - **Query panel**: text input for natural-language queries, file-upload dropzone styled for GeoTIFF/image uploads (mock upload state only), and a scrollable conversation thread showing example Q&A with mock responses
   - **Execution log panel**: collapsible glass panel showing a mock "Auditable Trace" — which tool was selected, confidence score, processing steps — styled like a technical readout

4. **Capabilities / Feature grid**
   - Cards for the three tool types (Single-Image VQA, Visual Grounding, Bi-Temporal Change Detection) plus the cross-modal SAR+optical fusion capability
   - Short description and a small icon or micro-animation per card

5. **Tech / Model stack showcase**
   - Understated section listing the underlying models/frameworks (e.g. vision-language model, GDAL/Rasterio, change-detection network) as a credibility signal, styled like a minimal logo/spec strip rather than a marketing section

6. **Footer**
   - Team/hackathon attribution (SIH 2026, PS 26167, ISRO), links, minimal

### Placeholder conventions (important)
For every component that will need real backend/data wiring later:
- Name the component clearly and descriptively (`MapViewport`, `QueryInput`, `ExecutionTracePanel`, `UploadDropzone`, `ChangeDetectionOverlay`, etc.)
- Add a `// TODO(backend):` comment at the top of the file describing exactly what real data/prop shape it will eventually receive
- Use realistic mock data (typed with proper TypeScript interfaces) rather than lorem ipsum, so the eventual data-wiring is a drop-in replacement, not a rewrite
- Keep mock data in a separate `mocks/` folder, imported by components — not hardcoded inline

### Deliverable for this session
Scaffold the project, build out all sections above with real (not placeholder) visual design and animation, and get it running locally with `npm run dev`. Prioritize the Hero and Interactive Console sections first since those carry the most design weight — the rest can be lighter passes we iterate on after.
