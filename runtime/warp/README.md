# MuJoCo Warp runtime

This frozen Python 3.12 environment contains only the dependencies used by the
remote Linux/CUDA Warp worker. The application source and Franka assets are
uploaded separately and exposed to the worker with `PYTHONPATH`.

Create or verify the environment on a Linux x86-64 rendering host with:

```bash
uv sync --directory runtime/warp --frozen --no-dev
```

The runtime pins MuJoCo, MuJoCo Warp, Warp, NumPy, Pillow, and the ImageIO
packages needed for still and video output. It also includes OpenUSD because
the shared benchmark preparation path authors an ovrtx scene before starting
the isolated renderer workers. The NVIDIA driver and CUDA runtime are host
requirements and are not installed by this lockfile.
