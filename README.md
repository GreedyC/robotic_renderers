# FR3 renderer evaluation

This app renders one deterministic Franka FR3 scene through four pipelines:

- MuJoCo's native OpenGL renderer
- MuJoCo Warp's ray renderer on the local Apple-silicon CPU
- Blender Cycles on Apple Silicon/Metal
- NVIDIA ovrtx path tracing on the `l4` RTX GPU over SSH

The scene is built once with MuJoCo `MjSpec`. It contains the Menagerie FR3 fixed to
the floor, a box, sphere, and cylinder primitive, two true mesh objects, one orbiting
evaluation camera, a sky, PBR-like materials, and three lights. Every backend receives
the same analytical robot and camera trajectories, avoiding drift in renderer
comparisons. The camera follows a close 2.35 m orbit around the robot origin with an
eight-second revolution period.

## Install

The checked-in lockfile uses Python 3.12 and pins the renderer APIs:

```bash
uv sync
uv run python main.py status
```

This installs `mujoco==3.10.0`, `mujoco-warp==3.10.0.1`, `warp-lang==1.15.0`,
`usd-core==26.5`, the image/video stack, and development tools into `.venv`.

Blender remains a desktop application rather than a Python library:

- Blender 5.1.2 is invoked as a separate process. Its bundled, exactly
  version-matched `bpy` runs `render_blender.py`; no second pip environment is needed.

ovrtx has no macOS or Metal build. `setup --backend ovrtx` installs NVIDIA's official
Linux x86-64 runtime in a separate uv-managed environment on `l4`; the macOS app still
generates the OpenUSD scene, displays live frames, and encodes videos locally.

Cosmos 3 is an optional generative post-process rather than a fifth renderer. The app
runs on macOS and sends a temporary conditioning bundle to the multi-GPU `g4` host.
Its pinned Python 3.12 environment lives at `~/code/robot_rendering_cosmos3/.venv` and
is created entirely with uv; the 64B model cache remains outside this repository.

The FR3 source and meshes are vendored under `third_party/` from MuJoCo Menagerie
commit `71f066ad0be9cd271f7ed58c030243ef157af9f4`. See `THIRD_PARTY.md` and the copied
Apache-2.0 licenses.

## Environment parity

All backends use the same 0.33 m checker tiles, material colors, close orbit camera,
gradient sky, and three-light rig: a shadow-casting directional key, cool point fill,
and warm spot rim. Blender maps those MjSpec settings to equivalent native light types
instead of substituting a physical sky and area-light studio. Small exposure
compensations account for each renderer's different color-response convention; the
reference-frame mean RGB levels are approximately 0.127 for OpenGL, 0.158 for Blender,
0.132 for Warp, and 0.148 for ovrtx on a normalized 0–1 scale.

## Commands

Inspect or export the canonical scene:

```bash
uv run python main.py scene
uv run python main.py scene --export build/portable_scene
```

Render still images:

```bash
uv run python main.py still --backend opengl --output outputs/opengl.png
uv run python main.py still --backend warp --supersample 2 --output outputs/warp.png
uv run python main.py still --backend blender --samples 256 --output outputs/blender.png
uv run python main.py still --backend ovrtx --host l4 --samples 256 --output outputs/ovrtx.png
```

Open live views:

```bash
uv run python main.py live --backend opengl
uv run python main.py live --backend warp
uv run python main.py live --backend blender
uv run python main.py live --backend ovrtx --host l4
```

OpenGL and Warp use the same local Pygame display path. Blender opens its interactive
rendered viewport. ovrtx streams frames from the configured RTX host into the same
local display path.

Record videos:

```bash
uv run python main.py record --backend opengl --duration 5 --fps 30 --output outputs/opengl.mp4
uv run python main.py record --backend warp --duration 5 --fps 30 --output outputs/warp.mp4
uv run python main.py record --backend blender --samples 256 --duration 5 --fps 30 --output outputs/blender.mp4
uv run python main.py record --backend ovrtx --host l4 --samples 256 --duration 5 --fps 30 --output outputs/ovrtx.mp4
```

Blender first renders lossless PNG frames and then the uv environment encodes the
MP4, so an interrupted path-tracing job does not corrupt one monolithic movie.

Record all four backends concurrently and create a labeled 2x2 comparison:

```bash
uv run python main.py record --backend all --duration 5 --fps 30 \
  --width 1280 --height 720 --samples 64 --supersample 1 \
  --keep-components --output videos/all_tiled.mp4
```

For `--backend all`, `--width` and `--height` are the resolution of each tile; the
example above produces a 2560x1440 output. OpenGL, Warp, Blender, and ovrtx launch as
isolated concurrent processes with identical timing and camera arguments. The app
rejects the whole comparison if any backend fails and labels every tile.
`--keep-components` copies the four synchronized source MP4s into
`videos/all_tiled_components/`.

Benchmark rendering separately from video encoding:

```bash
uv run python main.py benchmark --backend opengl --frames 30
uv run python main.py benchmark --backend warp --frames 30
uv run python main.py benchmark --backend all --frames 3 --samples 32 \
  --output outputs/benchmark.json
```

An `all` benchmark records per-backend errors without preventing the other measurements.
Blender path tracing is intentionally much heavier, so keep its frame and sample counts
modest for iteration and use `--samples 256` or higher for final work.

## Cosmos 3 photoreal enhancement

Install and start the persistent, pinned vLLM-Omni service on `g4`:

```bash
uv run python main.py cosmos setup --host g4
uv run python main.py cosmos start --host g4
uv run python main.py cosmos status --host g4
```

Generate a depth-guided 1280x720 video from a newly rendered Blender keyframe:

```bash
uv run python main.py cosmos enhance --backend blender --mode depth \
  --steps 50 --output videos/cosmos3_super_depth.mp4
```

Use a single existing rendered image, a rendered video, segmentation, or both structural
controls:

```bash
uv run python main.py cosmos enhance --mode i2v --input outputs/blender.png
uv run python main.py cosmos enhance --mode v2v --input videos/blender.mp4
uv run python main.py cosmos enhance --backend opengl --mode seg
uv run python main.py cosmos enhance --backend opengl --mode depth-seg
```

The quality preset uses `nvidia/Cosmos3-Super` (64B) in BF16, 50 diffusion steps, the
model's full 4096-token prompt budget, and NVIDIA's native 720p shape. The server uses
four-way HSDP and Ulysses sequence parallelism on GPUs 0-3, all on `g4`'s first NUMA
node. CFG-branch parallelism stays disabled because depth transfer has three guidance
branches, which cannot be evenly split across two groups; the branches instead run in
lockstep at their full recommended guidance. This is faster on `g4` than eight-way
sharding because its eight cards have no NVLink and the larger group crosses CPU sockets.
Model size, precision, steps, and output quality are unchanged. I2V/V2V defaults to 189
frames; transfer control defaults to the documented 121-frame chunk. The structured
prompt describes the exact FR3, five props, common lighting, and orbit camera, while a
detailed negative prompt penalizes robot deformation and temporal drift.

The server raises vLLM-Omni's default ten-minute synchronous-video timeout to two hours.
Cosmos3-Super at this quality can legitimately exceed ten minutes on PCIe-only GPUs;
without this setting the API discards a healthy generation partway through diffusion.

Depth mode is the default because it retains the robot silhouette, floor contact, prop
layout, and orbit motion throughout the generated clip. The app locks one RGB source
keyframe and renders an aligned inverse-metric depth trajectory from MjSpec. Sending one
keyframe is also much faster for Blender and avoids duplicating a full RGB video inside
vLLM-Omni's multiprocessing transport. `seg` adds stable masks for the robot, floor, and
each prop; `depth-seg` supplies both modalities. Pass `--keep-bundle` to preserve these
inputs beside the output for inspection.

The public Cosmos3-Super checkpoint does not require a token. NVIDIA's optional
`Cosmos-1.0-Guardrail` checkpoint is separately gated, so this local, fixed robot-scene
service starts with guardrails disabled. Do not expose its loopback-only API as a public
service. Stop it when it is no longer needed:

```bash
uv run python main.py cosmos stop --host g4
```

Official references:

- [NVIDIA Cosmos 3 overview](https://research.nvidia.com/labs/cosmos-lab/cosmos3/)
- [NVIDIA Cosmos repository](https://github.com/NVIDIA/cosmos)
- [Cosmos3-Super model card](https://huggingface.co/nvidia/Cosmos3-Super)
- [vLLM-Omni Cosmos3-Super recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/cosmos3/Cosmos3-Super.md)
- [Diffusers Cosmos 3 transfer controls](https://huggingface.co/docs/diffusers/main/en/api/pipelines/cosmos3)

## Maximum-throughput FPS benchmark

`benchmark_fps.py` measures and scales the same 1280×720 FR3 scene on the `g4` host:

```bash
uv run python benchmark_fps.py --host g4
```

The default sweep uses every detected GPU and applies a backend-specific strategy:

- MuJoCo OpenGL runs independent headless EGL contexts in multiprocessing workers and
  distributes them across GPUs.
- MuJoCo Warp creates a true `nworld` batch on each GPU and scales batch size separately
  from GPU count.
- Blender keeps persistent Cycles sessions alive, uses the GPU-native OptiX denoiser,
  constrains CPU threads, binds workers to each GPU's NUMA node, and distributes complete
  frames across independent processes. The automatic thread budget decreases as worker
  count grows.
- ovrtx references the canonical OpenUSD scene multiple times in a full-resolution tiled
  RenderProduct and runs one GPU-pinned tiled process per GPU. In ovrtx 0.3, one tiled
  process accepted the multi-GPU configuration but only exercised one GPU in testing.

Initialization, shader compilation, warmup, video encoding, and SSH transfer are excluded.
The primary `throughput_fps` metric measures completed images through each backend's normal
result API and includes GPU-to-host RGB transfer for Warp and ovrtx. The report also shows
renderer-call FPS, which can exclude asynchronous result mapping. Blender and ovrtx use the
same configurable sample count, 32 by default. Repeat only the measured peak configurations:

```bash
uv run python benchmark_fps.py --host g4 --frames 24 --samples 32 \
  --opengl-workers 128 --warp-batches 128 \
  --blender-workers 64 --blender-denoiser OPTIX --ovrtx-batches 4
```

Results are written to `outputs/fps_benchmark_g4.json` and a readable Markdown report at
`outputs/fps_benchmark_g4.md`. The setup is reproducible: the script syncs the locked uv
environment, installs the matching Blender 5.1.2 Linux binary in the user's cache, and
installs the pinned ovrtx runtime without changing system packages.

The checked-in g4 result at 1280×720 ranked completed-image throughput as Warp
(966.18 FPS), ovrtx (243.62 FPS), OpenGL (139.17 FPS), and Blender (43.01 FPS). Blender
improved 14.83× over its original 2.90-FPS longer-run baseline without reducing its
32 samples or 12-bounce limit. At this frame size, splitting one frame across all eight
GPUs reached only 2.81 FPS; 64 persistent frame workers, eight per GPU, were much faster.
These compare the configured pipelines rather than equal perceptual quality: OpenGL and Warp use
their native quality settings, while Blender and ovrtx use 32 path-tracing samples.

## MuJoCo Warp on macOS

The Apple-silicon wheel exposes Warp's CPU device, and that is the default for stills,
live viewing, video, and benchmarks. No CUDA system is required. The implementation
enables textures, hard shadows, ambient/direct lighting, skybox, specular highlights,
emission, back-face culling, and optional supersampling before Lanczos downsampling.
The renderer is still single-hit/direct-lighting rather than a path tracer; it has no
global illumination, soft shadow sampling, or multibounce reflection.

The old SSH CUDA worker remains optional for comparisons:

```bash
uv run python main.py setup --backend warp --host l4
uv run python main.py record --backend warp --remote-warp --host l4
```

Setup rsyncs this project (excluding local caches and outputs), recreates the locked uv
environment at `~/.cache/robot_rendering` on `l4`, and leaves outputs local. If a new VM
does not have uv yet:

```bash
ssh l4 'curl -LsSf https://astral.sh/uv/install.sh | sh'
```

## NVIDIA ovrtx on the L4

NVIDIA ovrtx 0.3 is pre-release software and officially supports Windows x86-64,
Linux x86-64, and Linux aarch64 with an RTX-capable NVIDIA GPU. It cannot execute on
macOS. Install its official, SHA-256-verified Linux release on the configured host:

```bash
uv run python main.py setup --backend ovrtx --host l4
```

The runtime and editable Python package live in the uv-managed environment at
`~/code/robot_rendering_ovrtx/.venv` on `l4`. The large NVIDIA runtime is not copied
into this repository. Setup downloads release `v0.3.0`, verifies digest
`5569e44b…f1d5f9`, and installs `ovrtx==0.3.0.312915` with NumPy 2.2.6.

`render_ovrtx.py` converts the exact compiled MuJoCo mesh buffers, PBR parameters,
visible primitive geometry, checker floor, per-frame robot transforms, and shared
orbit camera into a validated binary OpenUSD crate with time-sampled transforms.
Stills and videos use converged `PathTracing` with adaptive sampling, OptiX denoising,
and up to 12 bounces. The worker warms up material/texture streaming before returning
the first frame. Live mode uses
`Real-Time Path-Tracing`. The remote worker maps `LdrColor` to CPU memory and streams
RGB frames over SSH into the same local Pygame and FFmpeg mechanisms used elsewhere.
The first render compiles and caches shaders and is therefore much slower.

Review the ovrtx license before publishing benchmark or performance comparisons. Its
terms restrict third-party disclosure of benchmark, competitive-analysis, regression,
and performance results except as described by NVIDIA or with NVIDIA's permission.

Official references:

- [ovrtx repository and platform requirements](https://github.com/NVIDIA-Omniverse/ovrtx)
- [ovrtx v0.3.0 release](https://github.com/NVIDIA-Omniverse/ovrtx/releases/tag/v0.3.0)
- [ovrtx documentation](https://nvidia-omniverse.github.io/ovrtx/)

## Blender setup and quality

The app auto-detects:

```text
/Applications/Blender.app/Contents/MacOS/Blender
```

Cycles uses Metal, adaptive sampling with a `0.001` threshold, denoising, AgX, up to 12
bounces, three large area lights, a physical sky, PBR materials, and micro-bevels on
primitive props. `--samples 256` is the default still/video preset; live mode uses the
rendered Eevee viewport for responsiveness.

Blender has no built-in MJCF importer. The uv process compiles the MjSpec, exports the
exact mesh buffers and per-frame geom transforms, and Blender reconstructs that data.
This keeps mesh recentering and robot kinematics aligned with MuJoCo.

## Source layout

| File | Purpose |
| --- | --- |
| `scene.py` | Canonical MjSpec, deterministic motion, portable MJCF export |
| `render_mujoco_opengl.py` | Native OpenGL still/live/video/benchmark functions |
| `render_mujoco_warp.py` | Local macOS CPU renderer and optional SSH worker/stream |
| `render_blender.py` | MuJoCo-to-Blender adapter and Cycles/Eevee worker |
| `render_ovrtx.py` | OpenUSD export plus remote RTX frame streaming over SSH |
| `render_tiled.py` | Concurrent four-backend recording and labeled 2x2 composition |
| `render_cosmos3.py` | 64B generative enhancement plus RGB/depth/segmentation conditioning |
| `benchmark_fps.py` | Remote multi-GPU scaling benchmark and Markdown/JSON reporting |
| `render_common.py` | Shared PNG, MP4, resize, hash, and live-display functions |
| `main.py` | Unified command-line app |

The application code defines functions only; it does not define application classes.
The renderer libraries and SDKs naturally use their own classes internally.

## Example videos

The `videos/` directory contains two-second, 1280×720, 24 fps H.264/yuv420p examples.
They use the same 48-pose analytical robot and orbit-camera trajectories. The Blender
sample uses 16 Cycles samples and the ovrtx sample uses 32 samples to keep the examples
compact; use the default 256 for final comparisons. `ovrtx.mp4` is rendered on `l4`.
`all_tiled.mp4` is the labeled 2×2 four-backend composition.
`cosmos3_super_depth.mp4` is the depth-guided Cosmos3-Super enhancement generated on
`g4` from the same 1280x720 scene.

## Verification

```bash
uv run ruff check .
uv run pytest -q
```

The tests compile/export the MjSpec, validate deterministic robot/camera trajectories,
render an OpenGL smoke frame, verify video writing and four-stream tiled composition,
validate the ovrtx OpenUSD bundle, validate Cosmos request quality/parallelism and its
depth/segmentation bundles, check the required backend API, and enforce the
no-application-classes constraint.
