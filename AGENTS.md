# AGENTS.md

## Project scope

This repository compares rendering pipelines on one deterministic Franka FR3 robot
scene. Keep renderer inputs equivalent so that output differences reflect rendering
quality, not different geometry, animation, cameras, materials, or lighting.

The canonical environment is assembled with MuJoCo `MjSpec` in `scene.py`. The regular
backends are MuJoCo OpenGL, MuJoCo Warp, Blender Cycles, remote NVIDIA ovrtx, and
remote Mitsuba 3. Cosmos 3 is an optional generative post-process, not another renderer.

## Development environment

- Use Python 3.12 and the checked-in uv lockfile: `uv sync`.
- Run Python tools through `uv run` so they use the managed environment.
- Blender is an external application and runs `render_blender.py` with its bundled
  version-matched Python API. Do not install a mismatched `bpy` wheel into the uv
  environment.
- Linux CUDA/RTX rendering, remote benchmarks, and Cosmos 3 use a configurable SSH host.
  Documentation and tests must use the neutral `render-host` placeholder and must not
  expose maintainer infrastructure. Keep model caches, downloaded runtimes, and generated
  build products outside Git.

## Architecture and coding rules

- Prefer small module-level functions. Do not introduce application classes or other
  object-oriented abstractions.
- Treat `scene.py` as the source of truth for robot pose, props, materials, lighting,
  floor scale, camera intrinsics, and orbit trajectory.
- Keep the default output at 1280x720 unless a caller explicitly overrides it.
- Preserve the close, slow camera orbit around the robot origin for recorded videos.
- Backend adapters should expose the same functional surface where applicable:
  `backend_status`, `render_frame`, `render_still`, `render_live`, `record_video`, and
  `benchmark`.
- Add user-facing orchestration to `main.py`; keep backend-specific implementation in
  the corresponding `render_*.py` module.
- Keep Blender subprocesses and renderer workers isolated. Do not move Blender rendering
  into the uv interpreter.
- Keep labels and frame timing synchronized in tiled comparison videos.
- Treat ovrtx `pathtraced` and `realtime` as explicit modes of one backend. When
  path-traced RGB is requested with auxiliary outputs, keep RGB on the converged
  path-tracing product and use a synchronized real-time annotation product for AOVs.

## Renderer parity

- A backend translation must preserve canonical geometry, transforms, materials, camera
  motion, checker-tile scale, sky, and the key/fill/rim light rig.
- Small exposure or color-response compensation is acceptable only to match perceived
  light levels. Document intentional backend-specific deviations in `README.md`.
- Do not improve one renderer by silently changing its scene content or framing.
- Prefer high-quality settings for final examples, but keep sample counts and other
  quality controls configurable for iteration and benchmarks.

## Validation

Before committing a change, run:

```bash
uv run ruff check .
uv run pytest -q
```

For renderer changes, also run the smallest relevant still or short video smoke test.
For orchestration changes, verify both the individual backend path and `--backend all`.
Do not claim a remote backend works unless its remote command and returned artifact were
actually checked.

## Assets and generated files

- Preserve the MuJoCo Menagerie provenance, pinned commit, and Apache-2.0 license files
  under `third_party/`. Avoid gratuitous edits to vendored FR3 assets.
- Keep curated example videos in `videos/`. Do not replace them unless a task requires a
  newly verified example.
- Never commit `.venv`, `build`, `outputs`, caches, rendered frame sequences, downloaded
  Blender/ovrtx distributions, model weights, access tokens, or credentials.
- The Cosmos service must remain loopback-only on its remote host. Do not expose its API
  publicly or add secrets to command-line examples.

## Documentation

Update `README.md` whenever commands, defaults, backend availability, remote host setup,
or performance behavior changes. Keep examples executable from the repository root.
