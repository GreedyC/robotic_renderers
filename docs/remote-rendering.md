# Remote rendering

ovrtx and CUDA Mitsuba always run remotely. MuJoCo Warp can run locally or through the
same SSH workflow. Commands use the neutral `render-host` placeholder by default.

## Host selection

Configure `render-host` in SSH or pass `--host` explicitly:

```bash
uv run robotic-renderers setup --backend ovrtx --host render-host
```

Set the default host used by every CLI command:

```bash
export ROBOTIC_RENDERERS_HOST=render-host
```

An explicit `--host` takes precedence. Direct Python calls can instead use the
adapter-specific `ROBOT_RENDERING_WARP_HOST`, `ROBOT_RENDERING_OVRTX_HOST`,
`ROBOT_RENDERING_MITSUBA_HOST`, or `ROBOT_RENDERING_COSMOS_HOST`.

The host must be reachable with non-interactive SSH. Configure aliases, users,
ProxyJump, and identity files in `~/.ssh/config`; do not put credentials in commands or
this repository.

## Prerequisites

The remote Linux machine needs:

- an SSH server and enough storage in the user's home directory;
- Python 3.12 managed by uv;
- an NVIDIA driver compatible with the selected CUDA/RTX runtime;
- `rsync`, `curl`, and standard archive tools needed by backend setup.

The official ovrtx wheel also loads the system OpenGL dispatcher. On Ubuntu 24.04:

```bash
sudo apt-get install --no-install-recommends libopengl0
```

The first ovrtx render with a fresh shader cache can spend several minutes compiling
shaders before producing a frame; the 0.5 upgrade smoke test took about 15 minutes on
an L4. The next short recording completed in about 23 seconds. Preserve the runtime
and its `ovrtx/bin/cache` directory between jobs so later runs can reuse these shaders.
Different rendering configurations, including throughput workers, can trigger additional
shader compilation.

Install uv on a new host using the current instructions from
[Astral](https://docs.astral.sh/uv/getting-started/installation/). Review installation
scripts before piping downloaded content into a shell.

## Runtime isolation

Remote setup creates backend-specific roots:

| Backend | Default remote root |
| --- | --- |
| Warp | `~/.cache/robot_rendering` |
| ovrtx | `~/code/robot_rendering_ovrtx` |
| Mitsuba | `~/code/robot_rendering_mitsuba` |
| Cosmos | `~/code/robot_rendering_cosmos3` |

Runtime distributions, shader caches, model weights, generated bundles, and worker logs
stay outside Git. Job directories are unique and should be removed after completion.
The transfer layer uses an explicit source allowlist; do not broaden it to upload an
arbitrary dirty working tree.

## Operational safety

- Keep host values and output paths trusted; they are execution boundaries.
- Keep the Cosmos HTTP API bound to `127.0.0.1` and access it only through SSH.
- Treat remote PID files as untrusted stale state and verify process identity before
  signaling a worker.
- Upload into a job-specific directory and download into a temporary local path.
- Validate returned dimensions/frame count before atomically replacing final output.
- Use `--force` setup only when you intend to recreate a backend environment.

Do not claim a remote backend works based only on setup output. Render a small artifact,
download it, and inspect the returned metadata.
