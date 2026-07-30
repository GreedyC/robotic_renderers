# Robotic Renderers

Robotic Renderers compares five rendering pipelines on one deterministic Franka FR3
scene:

- MuJoCo OpenGL
- MuJoCo Warp
- Blender Cycles
- NVIDIA ovrtx, in real-time or converged path-traced mode
- Mitsuba 3 CUDA/OptiX

Cosmos 3 is available as an optional generative post-process. It is not treated as a
renderer or a source of simulator ground truth.

The canonical environment is assembled once with MuJoCo `MjSpec`. Every backend
receives the same floor-mounted robot, three primitive props, two mesh props, material
intent, 0.33 m floor tiles, gradient sky, and directional-key/point-fill/spot-rim light
rig. The default 1280x720 video follows one analytical 30-second camera trajectory with
two orbits, close inspection passes, establishing views, and changing camera height.

See the [architecture](https://github.com/tkelestemur/robotic_renderers/blob/main/docs/architecture.md)
and [backend parity](https://github.com/tkelestemur/robotic_renderers/blob/main/docs/backend-parity.md)
guides for the comparison contract.

## Requirements and installation

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- Blender for the Blender backend
- an SSH-accessible Linux CUDA/RTX host for ovrtx and CUDA Mitsuba

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
