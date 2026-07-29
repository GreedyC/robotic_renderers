# Development

## Environment

Use Python 3.12 and the checked-in lockfile:

```bash
uv sync --locked
```

The repository defaults to the `dev` and `renderers` dependency groups, so this installs
the full development surface. Published-package users can select the `warp`, `ovrtx`,
or `all` extra instead. Run project tools with `uv run`. Blender remains external and
remote runtimes remain in their own uv environments.

## Validation

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest -q
```

Tests use these markers:

- `integration`: external applications or a real graphics context;
- `gpu`: CUDA/RTX hardware;
- `remote`: SSH or a remote service.

Hosted CI runs the portable subset:

```bash
uv run pytest -q -m "not integration and not gpu and not remote"
```

Renderer changes additionally need the smallest relevant still or video smoke test.
Orchestration changes must exercise one backend and `--backend all`.

## Functional style

Prefer small module-level functions. Keep renderer-neutral work outside backend modules,
and avoid introducing application classes. A helper belongs in a shared module when it
has the same semantics in multiple renderers; superficially similar renderer API calls
can remain local when their failure or lifetime behavior differs.

Preserve unrelated changes in a dirty working tree. Generated files belong under
ignored `build/` or `outputs/`, not beside source.
