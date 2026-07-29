# Release checklist

1. Confirm `CHANGELOG.md`, version metadata, README commands, backend status, and
   `THIRD_PARTY.md` are current.
2. Review the complete locked dependency inventory and every redistributed asset
   license.
3. Verify that no credentials, private host details, caches, model weights, frame
   sequences, or generated benchmark reports are tracked.
4. Run:

   ```bash
   uv sync --locked
   uv run ruff format --check .
   uv run ruff check .
   uv run pytest -q
   uv build
   ```

5. Install the wheel into a fresh Python 3.12 environment. Check
   `robotic-renderers --help`, scene creation, a local still, package version metadata,
   and bundled FR3 asset/license access.
6. Run opt-in Blender and remote smoke tests on their documented platforms. Inspect
   every returned artifact.
7. Ensure new MP4 files use Git LFS. Do not rewrite existing video history as part of
   an unrelated release.
8. Create a signed release tag and GitHub release. Put large example media in release
   assets where practical.
9. Publish to the package index only from the tagged, clean commit using trusted
   publishing.

Do not publish ovrtx performance comparisons until NVIDIA's applicable terms have been
reviewed and any required permission obtained.
