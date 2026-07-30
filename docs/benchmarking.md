# Benchmarking

Benchmarks separate rendering from video encoding and SSH artifact transfer:

```bash
uv run robotic-renderers benchmark --backend opengl --frames 30
uv run robotic-renderers benchmark --backend warp --frames 30
uv run robotic-renderers benchmark --backend mitsuba --host render-host \
  --frames 30 --samples 32
uv run robotic-renderers benchmark --backend all --host render-host \
  --frames 3 --samples 32 --output outputs/benchmark.json
```

The multi-GPU scaling harness runs the same 1280x720 scene:

```bash
uv run robotic-renderers-throughput --host render-host
```

## Measurement contract

Initialization, dependency setup, shader compilation, warmup, video encoding, and SSH
transfer are excluded. `throughput_fps` measures completed images through each
backend's normal result API. For Warp, ovrtx, and Mitsuba it includes GPU-to-host RGB
transfer. Renderer-call timing may also be reported when asynchronous mapping makes the
distinction useful.

The unified `--warmup` count is executed before measured frames. Blender builds one
Cycles scene in its isolated process, renders the warmup frames to populate persistent
data and shader state, and reports only the following frames in `render_seconds` and
`render_fps`. `process_seconds` remains available to show end-to-end worker cost.

Video results distinguish `video_fps`, the encoded timeline rate, from `render_fps`,
the measured production throughput. They must not be combined into an ambiguous `fps`
field.

The harness scales backends according to their execution models:

- OpenGL uses independent headless EGL contexts.
- Warp batches multiple worlds per GPU.
- Blender distributes frames across persistent isolated workers.
- ovrtx uses GPU-pinned tiled render products.
- Mitsuba distributes persistent CUDA path-tracing workers.

These are pipeline-throughput measurements, not equal-perceptual-quality comparisons.
Always record resolution, samples, bounce depth, denoiser, device model, GPU count,
driver, renderer versions, worker count, batch size, warmup, and measured frame count.

## Publication

Do not commit generated benchmark reports. NVIDIA ovrtx is governed by terms that can
restrict third-party disclosure of benchmark and competitive-performance data. Review
[THIRD_PARTY.md](../THIRD_PARTY.md) and obtain any necessary permission before
publishing results.
