## Summary

Describe the user-visible change and why it is needed.

## Renderer parity

- [ ] Scene geometry, transforms, materials, camera motion, floor scale, sky, and light
      layout remain equivalent.
- [ ] Any intentional exposure or color-response compensation is documented.
- [ ] This change does not affect renderer parity.

## Validation

List exact commands and, for integration tests, the OS, renderer, GPU, and remote
runtime used.

- [ ] `uv run ruff format --check .`
- [ ] `uv run ruff check .`
- [ ] `uv run pytest -q`
- [ ] Relevant still or short-video smoke test
- [ ] Individual backend and `--backend all` paths, when orchestration changed

## Release considerations

- [ ] Documentation and changelog are updated where needed.
- [ ] No generated output, credentials, caches, model weights, or private host details
      are included.
- [ ] Third-party and benchmark publication terms were reviewed where applicable.
