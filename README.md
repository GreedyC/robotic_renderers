# Robotic Renderers

[![CI](https://github.com/tkelestemur/robotic_renderers/actions/workflows/ci.yml/badge.svg)](https://github.com/tkelestemur/robotic_renderers/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/tkelestemur/robotic_renderers)](https://github.com/tkelestemur/robotic_renderers/releases/latest)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/downloads/)
[![Apache-2.0](https://img.shields.io/github/license/tkelestemur/robotic_renderers)](LICENSE)

Robotic Renderers makes renderer comparisons reproducible: one deterministic Franka FR3
scene, five rendering pipelines, and a shared contract for geometry, motion, materials,
lighting, cameras, and timing.

[![Autoplaying tiled comparison of all five rendering backends](videos/all_tiled_preview.gif)](videos/all_tiled.mp4)

The preview loops automatically. Select it to open the full 30-second, 3840x1440
comparison video.

## Why this project exists

- **Fair comparisons:** every backend receives the same scene content and analytical
  camera trajectory.
- **Useful outputs:** render stills, live views, MP4 recordings, synchronized tiled
  comparisons, benchmarks, and ovrtx AOVs through one CLI.
- **Reproducible environments:** Python dependencies, remote renderer runtimes, and the
  MuJoCo Menagerie source revision are pinned and documented.

Cosmos 3 is available as an optional generative post-process. It is not treated as a
renderer or a source of simulator ground truth.

The canonical environment is assembled once with MuJoCo `MjSpec`. Every backend
receives the same floor-mounted robot, three primitive props, two mesh props, material
intent, 0.33 m floor tiles, gradient sky, and directional-key/point-fill/spot-rim light
rig. The default 1280x720 video follows one analytical 30-second camera trajectory with
two orbits, close inspection passes, establishing views, and changing camera height.

[Release v0.1.0](https://github.com/tkelestemur/robotic_renderers/releases/tag/v0.1.0)
· [Documentation](docs/)
· [Architecture](docs/architecture.md)
· [Adding a backend](docs/adding-a-backend.md)
· [Renderer parity](docs/backend-parity.md)
· [Roadmap and feedback](https://github.com/tkelestemur/robotic_renderers/discussions/6)

## Backend support

| Backend | Execution | Main outputs |
| --- | --- | --- |
| MuJoCo OpenGL | Local macOS or Linux | Live, still, video, benchmark |
| MuJoCo Warp | Local Apple-silicon CPU or remote Linux CUDA | Live, still, video, benchmark |
| Blender Cycles | External Blender on macOS or Linux | Live, still, video, benchmark |
| NVIDIA ovrtx | Remote Linux RTX | Real-time/path-traced RGB, AOVs, video, benchmark |
| Mitsuba 3 | Remote Linux CUDA/OptiX | Live, still, video, benchmark |

## Try the local reference in 60 seconds

With Python 3.12 and [uv](https://docs.astral.sh/uv/) installed, render directly from the
versioned `v0.1.0` source tag without cloning the repository:

```bash
uvx --from git+https://github.com/tkelestemur/robotic_renderers.git@v0.1.0 \
  robotic-renderers still --backend opengl --output opengl.png
```

## Full installation

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- Optional: Blender for the Blender backend
- Optional: an SSH-accessible Linux CUDA/RTX host for remote Warp, ovrtx, Mitsuba, and
  Cosmos workflows

```bash
git clone https://github.com/tkelestemur/robotic_renderers.git
cd robotic_renderers
uv sync --locked
uv run robotic-renderers status --backend opengl
```

Repository `uv sync` installs the `dev` and `renderers` dependency groups by default.
Package consumers can keep a smaller environment and select `robotic-renderers[warp]`,
`robotic-renderers[ovrtx]`, or `robotic-renderers[all]` as needed.

Blender runs as an external application with its bundled, version-matched Python API.
Do not install an unrelated `bpy` wheel into the uv environment. Remote renderers use
separate uv environments and keep downloaded runtimes and caches outside Git.
Use `uv run robotic-renderers status` after configuring an SSH host to probe all five
backends.

The dependency refresh uses MuJoCo/MuJoCo Warp 3.13.0, Warp 1.17.0, NumPy 2.5.3,
pygame-ce 2.5.8, OpenUSD 26.8, ovrtx 0.5.0.377615/ovstage 0.2.0.377349, and
Mitsuba 3.9.1/Dr.Jit 1.5.0. The main environment and all three remote locks include
the latest compatible transitive dependencies. The throughput harness downloads
Blender 5.2.1 LTS with a verified SHA-256; local Blender remains selectable through
`BLENDER_BIN`. Throughput workers explicitly enable Blender's Python environment
support for the project source path and fail immediately if they exit before readiness.

ovrtx 0.5 output lookup uses full RenderVar paths for RGB, annotations, and throughput
readback. Its replacement bounce settings use zero extra specular/transmission
bounces, following upstream's conversion of the previous limits. The canonical scene
and camera trajectory are preserved.

Cosmos uses vLLM-Omni 0.28.0's release commit with matching vLLM 0.28.0. Although
vLLM 0.29.0 is available, Omni requires a matching release line; see the
[upstream installation guide](https://github.com/vllm-project/vllm-omni/blob/v0.28.0/docs/getting_started/installation/gpu/cuda.inc.md).
Setup resolves the pair together and rejects an existing runtime with stale versions.
To upgrade an existing Cosmos service:

```bash
uv run robotic-renderers cosmos stop --host render-host
uv run robotic-renderers cosmos setup --host render-host --force
uv run robotic-renderers cosmos start --host render-host
```

The FR3 meshes and MJCF are vendored from MuJoCo Menagerie commit
`71f066ad0be9cd271f7ed58c030243ef157af9f4`. See
[THIRD_PARTY.md](https://github.com/tkelestemur/robotic_renderers/blob/main/THIRD_PARTY.md)
for provenance and dependency licenses.

## Quick start

Inspect or export the canonical scene:

```bash
uv run robotic-renderers scene
uv run robotic-renderers scene --export build/portable_scene
```

Render local stills:

```bash
uv run robotic-renderers still --backend opengl \
  --output outputs/opengl.png
uv run robotic-renderers still --backend warp --supersample 2 \
  --output outputs/warp.png
uv run robotic-renderers still --backend blender --samples 256 \
  --output outputs/blender.png
```

Open a live view:

```bash
uv run robotic-renderers live --backend opengl
uv run robotic-renderers live --backend warp
uv run robotic-renderers live --backend blender
```

Record the complete camera sequence:

```bash
uv run robotic-renderers record --backend blender \
  --duration 30 --fps 24 --width 1280 --height 720 --samples 256 \
  --output outputs/blender.mp4
```

Blender renders isolated lossless frames in the platform cache before local MP4
encoding. Its stills and videos are atomically published only after validation. Other
finite recorders follow the same output rule.

The same functional adapters are available to Python callers:

```python
from robotic_renderers import load_backend

opengl = load_backend("opengl")
result = opengl.render_still(output_path="outputs/opengl.png")
```

## Remote RTX backends

Remote commands use the neutral `render-host` placeholder by default. Configure that
alias in SSH, set `ROBOTIC_RENDERERS_HOST`, or pass your own host explicitly:

On Ubuntu 24.04, install the OpenGL loader required by the official ovrtx plugins:

```bash
sudo apt-get install --no-install-recommends libopengl0
```

```bash
uv run robotic-renderers setup --backend ovrtx --host render-host
uv run robotic-renderers setup --backend mitsuba --host render-host

uv run robotic-renderers still --backend ovrtx --host render-host \
  --render-mode pathtraced --samples 256 --output outputs/ovrtx.png
uv run robotic-renderers still --backend mitsuba --host render-host \
  --samples 256 --output outputs/mitsuba.png
```

Set one default host for every CLI command:

```bash
export ROBOTIC_RENDERERS_HOST=render-host
```

Direct Python calls can instead override an adapter with
`ROBOT_RENDERING_WARP_HOST`, `ROBOT_RENDERING_OVRTX_HOST`,
`ROBOT_RENDERING_MITSUBA_HOST`, or `ROBOT_RENDERING_COSMOS_HOST`.

See [remote rendering](https://github.com/tkelestemur/robotic_renderers/blob/main/docs/remote-rendering.md)
for prerequisites, runtime locations, and security guidance.

### ovrtx modes and AOVs

`pathtraced` is the still, video, and benchmark default. `realtime` is the live default:

```bash
uv run robotic-renderers live --backend ovrtx --host render-host \
  --render-mode realtime
uv run robotic-renderers still --backend ovrtx --host render-host \
  --render-mode pathtraced --samples 512 \
  --output outputs/ovrtx_pathtraced.png
```

Normalized auxiliary outputs are `depth`, `normals`, `albedo`, `semantic`, and
`instance`:

```bash
uv run robotic-renderers still --backend ovrtx --host render-host \
  --render-mode pathtraced --samples 256 \
  --aov depth --aov normals --aov albedo --aov semantic --aov instance \
  --aov-output-dir outputs/ovrtx_aovs --output outputs/ovrtx.png
```

When AOVs accompany path-traced RGB, RGB remains the converged path-tracing product and
the annotations come from a second synchronized real-time product. Raw arrays and label
metadata are retained alongside stable previews.

## Tiled comparison video

Render all five backends concurrently and build a labeled 3x2 comparison:

```bash
uv run robotic-renderers record --backend all --host render-host \
  --duration 30 --fps 24 --width 1280 --height 720 \
  --samples 64 --supersample 1 --keep-components \
  --output outputs/all_tiled.mp4
```

`--width` and `--height` describe each tile, so this example produces 3840x1440 output.
Labels and frame timing remain synchronized. The workflow fails if a requested backend
fails, and it does not publish a partial comparison.

## Benchmarking

Measure renderer work separately from video encoding:

```bash
uv run robotic-renderers benchmark --backend opengl --frames 30
uv run robotic-renderers benchmark --backend mitsuba --host render-host \
  --frames 30 --samples 32
uv run robotic-renderers benchmark --backend all --host render-host \
  --frames 3 --samples 32 --output outputs/benchmark.json
```

The multi-GPU scaling harness remains available:

```bash
uv run robotic-renderers-throughput --host render-host
```

Results depend on renderer settings and do not imply equal perceptual quality. Read the
[benchmark methodology](https://github.com/tkelestemur/robotic_renderers/blob/main/docs/benchmarking.md)
and [ovrtx license notice](https://github.com/tkelestemur/robotic_renderers/blob/main/THIRD_PARTY.md#nvidia-ovrtx-and-ovstage)
before publishing comparisons.

## Cosmos 3 enhancement

Install and start the loopback-only remote service:

```bash
uv run robotic-renderers cosmos setup --host render-host
uv run robotic-renderers cosmos start --host render-host
uv run robotic-renderers cosmos status --host render-host
```

Generate a depth-guided video:

```bash
uv run robotic-renderers cosmos enhance --backend blender --mode depth \
  --host render-host --steps 50 --output outputs/cosmos3_depth.mp4
```

Image-to-video, video-to-video, segmentation, and combined depth/segmentation controls
are also supported. See
[Cosmos 3 enhancement](https://github.com/tkelestemur/robotic_renderers/blob/main/docs/cosmos3.md)
for reproducibility, runtime, and model-license guidance.

## Example media

Curated 30-second examples live in `videos/`. They use the same 1280x720, 24 fps camera
sequence; `all_tiled.mp4` is the labeled 3840x1440 composition. New MP4 files are stored
with Git LFS. Existing examples are intentionally exempt until a dedicated history
migration.

## Development

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest -q
```

Portable CI excludes tests marked `integration`, `gpu`, or `remote`. Renderer changes
also require a relevant still or short-video smoke test.

For a quick recording check, use three frames at reduced resolution. Check each
backend separately, then the complete tiled workflow on a configured GPU host:

```bash
uv run robotic-renderers record --backend opengl --duration 0.5 --fps 6 \
  --width 320 --height 180 --output outputs/smoke_opengl.mp4
uv run robotic-renderers record --backend all --host render-host \
  --duration 0.5 --fps 6 --width 320 --height 180 --samples 4 --supersample 1 \
  --keep-components --output outputs/smoke_all.mp4
```

Further documentation:

- [Architecture](https://github.com/tkelestemur/robotic_renderers/blob/main/docs/architecture.md)
- [Backends](https://github.com/tkelestemur/robotic_renderers/blob/main/docs/backends.md)
- [Renderer parity](https://github.com/tkelestemur/robotic_renderers/blob/main/docs/backend-parity.md)
- [Remote rendering](https://github.com/tkelestemur/robotic_renderers/blob/main/docs/remote-rendering.md)
- [Benchmarking](https://github.com/tkelestemur/robotic_renderers/blob/main/docs/benchmarking.md)
- [Development](https://github.com/tkelestemur/robotic_renderers/blob/main/docs/development.md)
- [Release checklist](https://github.com/tkelestemur/robotic_renderers/blob/main/docs/releasing.md)
- [Contributing](https://github.com/tkelestemur/robotic_renderers/blob/main/CONTRIBUTING.md)
- [Security policy](https://github.com/tkelestemur/robotic_renderers/blob/main/SECURITY.md)

Robotic Renderers is licensed under
[Apache-2.0](https://github.com/tkelestemur/robotic_renderers/blob/main/LICENSE).

If Robotic Renderers helps your work, consider starring the repository and sharing the
comparison with other robotics and graphics developers.
