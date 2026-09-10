# Renderer backends

## MuJoCo OpenGL

OpenGL is the smallest local reference path and requires no separate setup:

```bash
uv run robotic-renderers still --backend opengl --output outputs/opengl.png
uv run robotic-renderers live --backend opengl
uv run robotic-renderers record --backend opengl --output outputs/opengl.mp4
```

It uses MuJoCo's native renderer, so it is useful for fast iteration and camera/scene
validation.

## MuJoCo Warp

Warp runs on the Apple-silicon CPU device by default and can use a remote CUDA host:

```bash
uv run robotic-renderers still --backend warp --supersample 2 \
  --output outputs/warp.png
uv run robotic-renderers setup --backend warp --remote-warp --host render-host
uv run robotic-renderers record --backend warp --remote-warp --host render-host \
  --supersample 2 --output outputs/warp.mp4
```

Supersampling improves final image edges but increases rendering cost quadratically.

## Blender

Blender is an external application. The adapter searches `BLENDER_BIN`, `PATH`, and
standard application locations. It always runs the renderer through Blender's bundled
Python; do not install an unrelated `bpy` wheel.

```bash
uv run robotic-renderers still --backend blender --samples 256 \
  --output outputs/blender.png
uv run robotic-renderers live --backend blender
uv run robotic-renderers record --backend blender --samples 256 \
  --output outputs/blender.mp4
```

Final output uses Cycles. Videos render lossless frames in isolated jobs and use the
application environment for final encoding. Blender jobs live under the platform cache,
not inside the installed package or source tree. Stills and videos are staged beside
their requested destination and atomically published after validation.

`live` remains attached to the Blender process: close Blender to finish an unbounded
session. When `--duration` is provided, the adapter stops the process group at the
deadline, reaps it, and removes its temporary scene.

## NVIDIA ovrtx

ovrtx requires Linux, an RTX-capable NVIDIA GPU, and a working SSH host. Setup installs
the official pinned v0.5 wheels from a frozen uv lock in a dedicated remote environment:

```bash
uv run robotic-renderers setup --backend ovrtx --host render-host
uv run robotic-renderers still --backend ovrtx --host render-host \
  --render-mode pathtraced --samples 256 --output outputs/ovrtx.png
uv run robotic-renderers live --backend ovrtx --host render-host --render-mode realtime
```

Request auxiliary output by repeating `--aov`:

```bash
uv run robotic-renderers still --backend ovrtx --host render-host \
  --render-mode pathtraced --samples 256 \
  --aov depth --aov normals --aov albedo --aov semantic --aov instance \
  --aov-output-dir outputs/ovrtx_aovs --output outputs/ovrtx.png
```

See [backend parity](backend-parity.md) for the path-traced RGB/AOV synchronization
contract and [third-party notices](../THIRD_PARTY.md) before publishing performance
comparisons.

## Mitsuba 3

Mitsuba's CUDA/OptiX variant runs in a dedicated remote uv environment:

```bash
uv run robotic-renderers setup --backend mitsuba --host render-host
uv run robotic-renderers still --backend mitsuba --host render-host --samples 256 \
  --output outputs/mitsuba.png
uv run robotic-renderers live --backend mitsuba --host render-host --samples 64
```

Final output uses the path integrator with configurable samples, multiple importance
sampling, 12-bounce transport, and guided OptiX denoising.

## Common video and tiled comparison

Each finite recording defaults to the complete 30-second camera sequence:

```bash
uv run robotic-renderers record --backend blender --duration 30 --fps 24 \
  --width 1280 --height 720 --samples 256 --output outputs/blender.mp4
uv run robotic-renderers record --backend all --duration 30 --fps 24 \
  --width 1280 --height 720 --samples 64 --supersample 1 \
  --keep-components --output outputs/all_tiled.mp4
```

For `--backend all`, dimensions apply to each tile. Five 1280x720 inputs produce a
3840x1440 labeled 3x2 composition.
