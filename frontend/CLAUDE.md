# SatQuery AI — frontend

Chat-first workspace for the SatQuery agentic vision-language assistant. The
conversation is the product surface; the map and the execution trace are
companions to it, not the main event.

## Running it

```bash
# 1. backend (separate shell)
cd ../backend/SatQuery-master/backend && python main.py

# 2. frontend
npm install && npm run dev
```

The dev server binds all interfaces and proxies `/api` and `/static` to
`http://127.0.0.1:8000`, so a second machine on the LAN can drive the UI
without any rebuild. Override the backend with `SATQUERY_BACKEND`, or point the
browser straight at another host with `VITE_API_BASE_URL`. See `.env.example`.

## Layout

```
┌────┬──────────────────────────┬─────────────────┐
│    │  conversation            │  scene dock     │
│rail│  (max-w 46rem, centred)  │  Map│Trace│Scene│
│    │  composer                │  resizable      │
└────┴──────────────────────────┴─────────────────┘
```

- **Rail** (`components/layout/AppShell.tsx`) — icon-only nav plus a live
  backend indicator.
- **Conversation** (`components/chat/`) — thread, composer, and the evidence
  card that links an answer to its polygons on the map.
- **Scene dock** (`components/scene/SceneDock.tsx`) — drag-resizable,
  collapsible to a strip. The map stays mounted across tab switches so the
  MapLibre context and camera survive.
- **History** (`components/history/HistoryPanel.tsx`) — a conversation rail
  that opens *beside* the workspace rather than replacing it, so the list and
  the thread it points at are on screen together. Grouped Today / Yesterday /
  weekday dates / Older, with collapsible groups, a "Show N more" cut-off past
  20, per-row rename and delete, and a filter toggle for search. Picking a row
  restores that thread into the workspace. `/history` redirects to `/console`;
  the rail button toggles the panel.

The thread is width-capped even when the dock is closed, so the chat never
sprawls edge to edge on a wide display.

## Backend contract

`lib/api.ts` is the only place that talks to the server.

| Endpoint | Used by |
| --- | --- |
| `GET /api/v1/status` | backend health poll, Settings, compute label |
| `POST /api/v1/satquery` | every question (multipart: `query`, `files[]`, `dataset_name`) |
| `POST /api/v1/imagery/upload` | scene ingestion, returns WGS84 bounds + preview URLs |
| `GET /api/v1/imagery/presets` | demo scenes |
| `/static/**` | raster previews and generated masks |

Notes:

- Requests default to **same-origin relative paths**. Absolute
  `http://localhost:8000` URLs only resolve on the backend host and defeat the
  proxy; `resolveAssetUrl()` rewrites any that slip through.
- Imagery bytes are uploaded **once**. Later turns reference the scene by
  `dataset_name`, which the backend re-resolves from `temp_uploads/`.
- Queries carry an `AbortController`, so the composer's stop button really
  cancels, and inference gets a 10-minute ceiling rather than the default one.

## Trace honesty

`lib/trace.ts` parses `execution_trace.steps` by content. Per-step latency and
audit hashes are **not** rendered, because the backend does not report them —
an earlier version fabricated both and labelled them "verified". Wall-clock
timing shown on a message is measured by the client and is real.

## Landing hero

`components/features/EarthScene.tsx` renders a photoreal globe: four concentric
shells (surface, night-side city lights, cloud deck, fresnel atmosphere) plus a
real spacecraft on an inclined orbit. Assets are bundled, never fetched, so the
hero renders identically offline:

- `public/textures/` — the three.js repo's NASA-derived planet maps
- `public/models/radarsat-1.glb` — RADARSAT-1 from NASA's 3D Resources
  collection ("free and without copyright"). A SAR imaging satellite, which is
  what this project's Sentinel-1 work is actually about.

Two things about that model are load-bearing:

1. **Its materials are replaced on load.** The export's MeshPhysicalMaterials
   render as a black cut-out however the scene is lit — they carry glTF
   extension state needing a backdrop this scene lacks. `SatelliteModel` keeps
   each part's authored base colour and rebuilds a plain standard material.
2. **It has its own local key/fill lights.** The scene's single distant "sun"
   leaves it silhouetted against the limb.

Geometry is re-centred and scaled to `MODEL_TARGET_SIZE` automatically, so
dropping a different GLB in `public/models/` needs no other change. Orbit
geometry lives in `ORBIT_RADIUS` / `ORBIT_TILT`, shared by the spacecraft and
the traced path so the two cannot drift apart.

`StudioEnvironment` assigns a PMREM-filtered `RoomEnvironment` (bundled inside
three) to `scene.environment`, so metallic parts have something to reflect
without pulling an HDRI off a CDN. Only Standard/Physical materials read it, so
the Earth's Phong shells are unaffected.

## Georeferenced overlays

Uploaded rasters are placed on the map by **warping them to EPSG:3857 first**
(`gis_pipeline.build_web_overlay`), then handing MapLibre the corner
coordinates of the warped extent. The previous approach — render the raster in
its native CRS and stretch that PNG across its WGS84 bounding box — misregisters
three separate ways:

* a UTM grid is not a Mercator grid, so the image shears (hundreds of metres on
  a city-sized scene, worse further from the central meridian);
* `transform_bounds` returns the *envelope* of a reprojected footprint, which
  for any rotated scene is strictly larger than the image;
* MapLibre interpolates an image source linearly in Mercator, so even a 4326
  raster placed by lat/lon corners is compressed towards the poles.

After warping, the PNG's pixel grid *is* the map's pixel grid. Nodata becomes
transparent via the warped source mask, so the skirt never paints over the
basemap.

A file with no CRS returns `georeferenced: false` and no bounds. The UI says so
and skips the overlay; it does **not** invent a location (the old code dropped
every non-georeferenced upload over Bengaluru).

Analysis rasters — currently the change mask — come back through
`visual_evidence` as `{type: "ImageOverlay", url, wgs84_bounds, opacity}`,
already warped by the backend, and are drawn by `MapViewport`'s
`resultOverlay` prop.

## Change detection

The optical path uses **IR-MAD** (Canty & Nielsen 2008) in
`Bi-Temporal ChangeFormer/SatqueryAI/backend/services/irmad.py`, falling back to
the previous CVA if the covariance is degenerate (single band, too few valid
pixels). SAR still uses log-ratio, which is correct for multiplicative speckle.

Why IR-MAD: it is built on canonical correlation, which is invariant to affine
radiometric transforms of either image. Sun angle, atmosphere and calibration
drift between dates therefore produce no signal, where CVA reports change across
the whole scene. It also gives a chi-squared statistic with a calibrated
no-change probability instead of an arbitrary magnitude.

One non-obvious implementation detail, learned the hard way: the MAD variates
must be projected from **mean-centred** data. The canonical vectors come from
covariances, so projecting raw radiances leaves a constant offset in every
variate; on a pair with any radiometric shift that offset dwarfs the
standardisation, the statistic saturates, and the reweighting collapses to
all-zero weights on the next pass.

The final statistic is rescaled so its median matches the chi-squared median.
The reweighting deliberately fits the covariance to the most-unchanged pixels,
which inflates everything else by a constant factor; without the rescale a
"p < 1e-4" threshold flags an order of magnitude more pixels than it claims.
This assumes most of the scene is unchanged — `calibration_scale` is reported so
a caller can see how far off that was.

## Build notes

`vite.config.ts` sets `optimizeDeps.exclude: ['maplibre-gl']`. Without it Vite's
dependency pre-bundling rewrites maplibre's entry but does not emit its worker
chunk, so the worker 404s, the map style never finishes loading, and the canvas
renders black **with no error logged**. If the basemap ever goes blank, check
that exclusion first.

Routes are code-split in `App.tsx`; the landing page carries three.js and must
stay lazy so the workspace does not download a WebGL scene graph it never uses.

**Never hide the map with `display: none`.** A MapLibre map in a zero-size
container never completes a first render, so its `load` event never fires, and
every `addSource`/`addLayer` afterwards throws "Style is not done loading" —
silently, because the call sites catch it. The scene dock therefore keeps the
map laid out across tab switches and drops its opacity instead.

Related: sources and layers are created once inside the `load` handler and only
*updated* afterwards (`updateImage`, `setData`, `setPaintProperty`). Those
setters do not require a loaded style; adding a source lazily does.

## Design system

Tokens live in `index.css` under `@theme`:

- ground/surface scale `ground → surface → surface-2 → surface-3 → surface-4`
- hairlines `line`, `line-strong`
- text `ink`, `ink-muted`, `ink-faint`
- accents `accent`, `teal`, `amber`, `violet`, `danger`, `ok`

The legacy `space-black` / `accent-cyan` aliases are kept so the marketing
sections keep working. Avoid `text-base` for colour — that is Tailwind's
font-size utility; the ground token is `ground` for exactly this reason.

Primitives are in `components/ui/`: `Button`/`IconButton`, `Badge`, `Card`,
`Segmented`, and `Feedback` (`EmptyState`, `InlineAlert`, `Spinner`,
`ThinkingDots`). `IconButton` requires a `label` — the map chrome is dense with
icon-only controls.

## Known gaps

- Sign-in is UI only; there is no auth backend. Do not treat it as a boundary.
- Answers arrive in one response. The backend has no streaming endpoint yet —
  when it gains one, the pending-message plumbing in `context/AppState.tsx` is
  where tokens should land.
