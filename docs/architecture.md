# Architecture

Robotic Renderers compares renderer output, not scene implementations. One canonical
MuJoCo `MjSpec` owns the Franka pose, props, material intent, camera intrinsics and
trajectory, checker scale, sky, and key/fill/rim lighting.

## Data flow

```text
robotic_renderers.scene
  ├─ MuJoCo model/data ──> OpenGL
  │                    └─> MuJoCo Warp
  └─ neutral scene bundle
       ├─> Blender bundle ──> isolated Blender process
       ├─> OpenUSD ─────────> remote ovrtx worker
       └─> Mitsuba bundle ──> remote Mitsuba worker

rendered RGB/AOV frames
  ├─> still image
  ├─> local live display
  ├─> atomic video encoder
  ├─> synchronized tiled comparison
  └─> optional Cosmos 3 conditioning
```

Backend translators may convert units and renderer-specific light or material
parameters, but they must not introduce different geometry, motion, framing, or scene
content. Exposure/color-response compensation is allowed only when it is explicit and
documented in [backend parity](backend-parity.md).

## Package boundaries

```text
src/robotic_renderers/
  cli.py                 command-line parsing and user-facing orchestration
  defaults.py            shared dimensions, timing, hosts, and writable paths
  scene.py               canonical MjSpec and analytical trajectories
  scene_bundle.py        renderer-neutral geometry/material/frame export
  geometry.py            primitive mesh and normal generation
  media.py               image, display, and video functions
  remote.py              SSH, transfer, and remote job lifecycle
  processes.py           local subprocess lifecycle
  registry.py            backend metadata and functional dispatch
  backends/              adapters, scene translations, and isolated worker entry points
  benchmarks/            specialized throughput harnesses
  workflows/             tiled video and Cosmos orchestration
runtime/
  */                     frozen renderer-specific remote environments
```

There are no top-level Python entry points. Import `robotic_renderers` or use the
installed `robotic-renderers` and `robotic-renderers-throughput` commands.

## Backend contract

Adapters expose module-level functions where the operation applies:

- `backend_status`
- `render_frame`
- `render_still`
- `render_live`
- `record_video`
- `benchmark`

Callers use keyword arguments. Unsupported operations should be described by backend
metadata and fail with a clear error, rather than accepting and ignoring arguments.
Backend modules own renderer-specific setup; orchestration belongs in `cli.py` and the
workflow modules.

## Isolation

Blender executes with its bundled, version-matched Python API. ovrtx, CUDA Mitsuba, and
Cosmos execute in dedicated remote uv environments. The application environment must
not import a mismatched `bpy`, ovrtx native extension, or CUDA-only dependency.

Temporary frame sequences and remote job directories are job-scoped. Successful output
is moved into place only after encoding or download validation.
