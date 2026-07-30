# Contributing

Thank you for improving Robotic Renderers. Changes should preserve a fair comparison:
geometry, animation, cameras, materials, lighting, and timing originate in the canonical
MuJoCo scene and remain equivalent across backends.

## Development setup

Install [uv](https://docs.astral.sh/uv/) and Python 3.12, then run:

```bash
git clone https://github.com/tkelestemur/robotic_renderers.git
cd robotic_renderers
uv sync --locked
uv run robotic-renderers status --backend opengl
```

Blender remains an external application. Do not install a separately versioned `bpy`
wheel into this environment. NVIDIA ovrtx, CUDA Mitsuba, and Cosmos run in isolated
environments on a configured Linux RTX host.

## Making a change

1. Open an issue for substantial behavior or architecture changes.
2. Keep `src/robotic_renderers/scene.py` as the source of truth for renderer-visible
   scene data.
3. Prefer small module-level functions and avoid application classes.
4. Keep backend adapters aligned with the common functional interface.
5. Add tests for behavior changes and update the relevant documentation.
6. Avoid committing generated output, runtime downloads, caches, credentials, or model
   weights.

Install Git LFS before adding or replacing curated MP4 examples:

```bash
git lfs install
```

The video-manifest tests verify the real media bytes, so contributors must fetch LFS
objects before running the complete repository suite.

Run the portable validation suite before submitting a pull request:

```bash
uv run ruff format .
uv run ruff check .
uv run pytest -q
```

Renderer changes also require the smallest relevant still or short-video smoke test.
Remote claims must name the hardware/runtime used and be backed by a checked returned
artifact. Tests that require Blender, a graphics context, CUDA, or SSH should be marked
as integration tests so that portable CI remains deterministic.

## Pull requests

Describe:

- the user-visible change;
- whether scene parity changed;
- the commands you ran;
- the renderer, operating system, and hardware used for integration tests;
- any intentional exposure or color-response compensation.

Do not include third-party benchmark numbers unless their licenses permit publication.
By submitting a contribution, you agree that it is licensed under Apache-2.0.
