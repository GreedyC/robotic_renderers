# ovrtx runtime

This environment pins the official NVIDIA ovrtx and ovstage Python
distributions used by the remote Linux/RTX backend.

Create it on the rendering host with:

```bash
uv sync --directory runtime/ovrtx --frozen --no-dev
```

The lock targets Python 3.12 on Linux x86-64. The pinned ovstage wheel uses
the `manylinux_2_35` platform tag, so the host needs glibc 2.35 or newer. The
renderer also requires a compatible NVIDIA driver and RTX GPU; those system
requirements are not represented by the Python lockfile.
