# Third-party software and assets

Robotic Renderers is licensed under Apache-2.0. That license does not replace the
licenses of the dependencies, applications, models, or assets described below. Review
the upstream terms before redistributing a built environment or publishing generated
material.

## Vendored Franka FR3 assets

`third_party/mujoco_menagerie/franka_fr3` is copied from
[`google-deepmind/mujoco_menagerie`](https://github.com/google-deepmind/mujoco_menagerie)
at commit `71f066ad0be9cd271f7ed58c030243ef157af9f4`.

The model is Apache-2.0 licensed. The Menagerie repository license and the model-specific
license are retained at:

- `third_party/mujoco_menagerie/LICENSE`
- `third_party/mujoco_menagerie/franka_fr3/LICENSE`

The wheel includes those model assets and license notices as package data.

## Python runtime dependencies

The uv lockfile is the authoritative inventory of exact versions and transitive
dependencies. The direct dependencies are:

| Component | Purpose | Upstream license |
| --- | --- | --- |
| [MuJoCo](https://github.com/google-deepmind/mujoco) | Canonical scene and OpenGL renderer | Apache-2.0 |
| [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) | Warp renderer | Apache-2.0 |
| [NVIDIA Warp](https://github.com/NVIDIA/warp) | MuJoCo Warp compute runtime | Apache-2.0 |
| [OpenUSD](https://github.com/PixarAnimationStudios/OpenUSD) (`usd-core`) | OpenUSD scene export | Modified Apache-2.0 |
| [NumPy](https://github.com/numpy/numpy) | Array and geometry processing | BSD-3-Clause |
| [ImageIO](https://github.com/imageio/imageio) | Image/video I/O | BSD-2-Clause |
| [imageio-ffmpeg](https://github.com/imageio/imageio-ffmpeg) | FFmpeg process wrapper and binaries | BSD-2-Clause; bundled FFmpeg has its own terms |
| [Pillow](https://github.com/python-pillow/Pillow) | Image encoding and labels | MIT-CMU/HPND |
| [pygame-ce](https://github.com/pygame-community/pygame-ce) | Local live-view window | LGPL-2.1-or-later |

Binary wheels can contain separately licensed components. Their installed metadata and
notices remain authoritative.

## External renderer applications and remote runtimes

These components are installed or downloaded separately and are not redistributed in
this repository or its Python wheel.

### Blender

[Blender](https://www.blender.org/about/license/) is an external application licensed
under GPL-3.0-or-later. Robotic Renderers launches Blender as an isolated process and
does not vendor Blender or a `bpy` wheel.

### NVIDIA ovrtx and ovstage

The remote setup installs the official Linux x86-64 Python distributions
`ovrtx==0.4.0.346409` and `ovstage==0.1.0.346039` published by
[`NVIDIA-Omniverse/ovrtx`](https://github.com/NVIDIA-Omniverse/ovrtx). Their complete
transitive dependency set and artifact hashes are pinned in `runtime/ovrtx/uv.lock`;
setup installs exclusively from that lock with `uv sync --frozen`.

ovrtx and ovstage are governed by NVIDIA's software license agreement and applicable
product-specific terms, not this project's Apache-2.0 license. Those terms include
restrictions concerning third-party disclosure of benchmark, competitive-analysis,
regression, and performance results. Obtain any required permission before publishing
such results.

### Mitsuba 3

The remote Mitsuba environment installs
[`mitsuba==3.9.0`](https://github.com/mitsuba-renderer/mitsuba3) and its Dr.Jit runtime
from Python wheels. Mitsuba's upstream license is BSD-3-Clause with an additional
enhancement grant; consult the license included with the installed distribution.

### Cosmos 3

Cosmos is optional and runs in a separate remote environment:

- [`vllm==0.26.0`](https://github.com/vllm-project/vllm), Apache-2.0
- [`vllm-omni`](https://github.com/vllm-project/vllm-omni), Apache-2.0, pinned to
  commit `39663d52872019fb0eb74bed8f1e811190486b0f`
- [`nvidia/Cosmos3-Super`](https://huggingface.co/nvidia/Cosmos3-Super), OpenMDW-1.1

The model weights are downloaded to the remote model cache and are not part of this
repository. Model use and redistribution remain subject to the model card and
OpenMDW-1.1 terms.
