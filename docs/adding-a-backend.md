# Adding a renderer backend

Start from the existing adapters, not a second implementation of the scene. A backend
may translate renderer units and material conventions, but **must not change geometry,
framing, camera animation, props or lighting layout** to improve its comparison image.
Read [architecture](architecture.md) and [backend parity](backend-parity.md) first.

## Integration checklist

| File or area | What to change |
| --- | --- |
| `src/robotic_renderers/backends/` | Add a module of functions; no application classes. Use `mujoco_opengl.py` for a local adapter or `blender.py` for an isolated worker. |
| `src/robotic_renderers/registry.py` | Add the name, import path, title, remote mode, sample and AOV capabilities to `_BACKENDS`. |
| `src/robotic_renderers/cli.py` | Route status, still, live, record and benchmark keyword arguments. The names come from the registry, but command dispatch has backend-specific branches. |
| `src/robotic_renderers/workflows/tiled.py` | Review all-backend routing and synchronized labels/times. Do not silently skip the new backend. |
| `pyproject.toml`, `uv.lock`, `runtime/` | Declare supported dependencies; keep external or CUDA-only runtimes isolated. Update the appropriate locked environment. |
| `tests/` | Extend discovery/API expectations, CLI routing tests and adapter tests; add a marked integration smoke test. |
| `README.md`, `docs/backend-parity.md`, `THIRD_PARTY.md` | Describe availability, commands, intentional response compensation and dependency/asset licensing. |

Inspect the signatures and result dictionaries in the adapter closest to yours. Local
OpenGL status takes no arguments; remote adapters accept a host. There is not one
universal signature for every backend, so the CLI must pass only supported arguments.
Metadata describes capabilities; reject unsupported operations explicitly rather than
accepting and ignoring options.

## Functional surface

The following is a **non-executable outline**, not a ready-to-import adapter. Replace
each placeholder with the implementation and result contract of the nearest existing
adapter. Test discovery currently expects all six names to be callable.

```python
BACKEND_NAME = "new_backend"


def backend_status(): ...  # availability, version, device, live/record support and diagnostic notes


def render_frame(*, width=1280, height=720, time_s=1.0): ...  # one canonical-time RGB frame


def render_still(
    output_path, *, width=1280, height=720, time_s=1.0
): ...  # artifact path, elapsed time, dimensions and digest


def render_live(
    *, width=1280, height=720, fps=30, duration=None
): ...  # display statistics; release the renderer when the viewer exits


def record_video(
    output_path, *, duration=30.0, fps=30, width=1280, height=720
): ...  # frame count, encoding/render timings and validated output


def benchmark(
    *, frame_count=30, width=640, height=360, fps=30, warmup=3
): ...  # separate initialization, warmup and measured rendering
```

## Canonical inputs and lifecycle

Use `scene.create_model_and_data`, `scene.set_scene_time` and `scene.trajectory_times`
for direct MuJoCo adapters. Translators should consume
`scene_bundle.prepare_scene_bundle`: geometry, meshes, materials, camera/frame
transforms and lights originate in the canonical scene. Inspect `tests/test_scene.py`
and `tests/test_scene_bundle.py` before mapping the bundle to your renderer.

Preserve defaults and frame timing. Document exposure or color-response compensation
in the README and parity guide; never hide different scene content behind it. Preserve
metric AOV values and stable semantics if the backend supplies auxiliary outputs.

Keep workers separate: Blender uses its own bundled Python; remote native/CUDA modules
belong in their isolated uv environments, not the portable application interpreter.
Use `processes.py` and `remote.py` lifecycle conventions and neutral `render-host`
examples. Put temporary bundles/frames in job-scoped directories, as in Blender's
`_new_job`, and close processes and renderer contexts on error as well as success.
Use `media.atomic_output_path`, `write_png` and `write_video` for local outputs;
publish downloaded artifacts only after validation. Never commit generated output.

## Validation from the repository root

Fetch LFS media before the complete portable suite; the manifest tests validate actual
video bytes, not LFS pointers. With Git LFS installed:

```bash
git lfs pull
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run pytest -q -m "not integration and not gpu and not remote"
```

At minimum, extend `tests/test_backends.py` and `tests/test_cli.py` to verify discovery,
all six functions, argument forwarding, unsupported options, unavailable dependencies
and all-backend routing. Test translation/parity and failed-output cleanup without
requiring a renderer, using the existing tests as examples.

Add the smallest still or short-video integration smoke test, marked `integration`,
`gpu` or `remote` as appropriate. Check dimensions, nontrivial frame content, timing
and the returned artifact. Report the actual operating system, renderer/runtime and
hardware; a mocked routing test is not evidence that a remote renderer works.

Finish by reviewing `LICENSE`, `THIRD_PARTY.md` and existing provenance for any imported
assets, runtimes or published benchmarks. Keep credentials, model weights and caches
outside Git. Link the adapter's setup and limitations from the README.
