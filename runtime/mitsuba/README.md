# Mitsuba runtime

This is the frozen Python environment used by the remote Mitsuba 3 backend.

Create it on the rendering host with:

```bash
uv sync --directory runtime/mitsuba --frozen --no-dev
```

The lock targets Python 3.12 on Linux x86-64. The pinned Mitsuba wheel uses
the `manylinux_2_28` platform tag, so the host needs glibc 2.28 or newer.
Mitsuba selects its CUDA/OptiX variant at runtime. The Python lock does not
install or validate the host NVIDIA driver.
