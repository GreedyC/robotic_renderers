"""NVIDIA ovrtx backend rendered on an RTX Linux host and streamed to macOS."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import re
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from pathlib import Path

import numpy as np

from robotic_renderers.defaults import DEFAULT_RENDER_HOST
from robotic_renderers.geometry import (
    geom_mesh as _shared_geom_mesh,
)
from robotic_renderers.geometry import (
    surface_normals as _surface_normals,
)
from robotic_renderers.geometry import (
    write_gradient_sky,
)
from robotic_renderers.remote import (
    REMOTE_PACKAGE_EXCLUDES,
    resolve_runtime_project,
    rsync_command,
    rsync_remote,
    run_ssh,
    ssh_command,
    worker_cleanup_script,
    write_pid_identity,
)
from robotic_renderers.remote import (
    read_exact as _read_exact,
)

BACKEND_NAME = "ovrtx"
OVRTX_VERSION = "0.4.0.346409"
OVSTAGE_VERSION = "0.1.0.346039"
DEFAULT_HOST = os.environ.get("ROBOT_RENDERING_OVRTX_HOST", DEFAULT_RENDER_HOST)
PACKAGE_SOURCE = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_SOURCE.parents[1]
REMOTE_ROOT = "code/robot_rendering_ovrtx"
STREAM_MAGIC = b"ROBOT_RENDERING_OVRTX_RGB_STREAM_V1\n"
STREAM_MAGIC_AOV = b"ROBOT_RENDERING_OVRTX_AOV_STREAM_V2\n"
STREAM_ERROR = b"ROBOT_RENDERING_OVRTX_ERROR_V1\n"
MAX_HANDSHAKE_BYTES = 4096
MAX_ERROR_BYTES = 16 * 1024 * 1024
MAX_AOV_PAYLOAD_BYTES = 512 * 1024 * 1024
# MuJoCo and USD/ovrtx express light intensity in different renderer-specific
# units. These are the only intentional light-value compensations in the USD
# translation and preserve the previously calibrated perceived brightness.
OVRTX_LIGHT_INTENSITY_MULTIPLIERS = {
    "directional": 210.0,
    "point": 525.0 / 9.0,
    "spot": 294.0 / 5.0,
}
OVRTX_DOME_INTENSITY = 59.0

RENDER_MODE_TOKENS = {
    "pathtraced": "PathTracing",
    # ovrtx 0.4's generated USD schema documents the spaced spelling, but the
    # live renderer rejects it and explicitly selects this runtime token.
    "realtime": "RealTimePathTracing",
}
RENDER_MODE_ALIASES = {
    "pathtraced": "pathtraced",
    "path_traced": "pathtraced",
    "path-traced": "pathtraced",
    "pathtracing": "pathtraced",
    "path_tracing": "pathtraced",
    "path-tracing": "pathtraced",
    "PathTracing": "pathtraced",
    "realtime": "realtime",
    "real_time": "realtime",
    "real-time": "realtime",
    "realtimepathtracing": "realtime",
    "real-timepath-tracing": "realtime",
    "RealTimePathTracing": "realtime",
    "Real-Time Path-Tracing": "realtime",
}
AOV_RENDER_VARS = {
    "depth": ("DistanceToImagePlaneSD",),
    "normals": ("NormalSD",),
    "albedo": ("DiffuseAlbedoSD",),
    "semantic": ("SemanticSegmentation", "SemanticIdMap"),
    # ovrtx 0.4 has no native instance render variable. Unique USD ``label``
    # semantics are combined with class semantics in SemanticSegmentation, then
    # remapped below into stable per-geometry instance identifiers.
    "instance": ("SemanticSegmentation", "SemanticIdMap"),
}
DEPTH_PREVIEW_RANGE_METERS = (1.5, 30.0)
OPENUSD_INSTALL_HINT = (
    "Local OpenUSD scene export requires usd-core. "
    "Install it with `pip install 'robotic-renderers[ovrtx]'`."
)


def _job_root() -> Path:
    from robotic_renderers.defaults import cache_directory

    root = cache_directory() / "jobs" / BACKEND_NAME
    root.mkdir(parents=True, exist_ok=True)
    return root


def _aov_payload_limit(width: int, height: int) -> int:
    pixel_count = max(1, int(width)) * max(1, int(height))
    resolution_budget = 16 * 1024 * 1024 + 64 * pixel_count
    return min(MAX_AOV_PAYLOAD_BYTES, resolution_budget)


def _run(command: list[str], **kwargs):
    return subprocess.run(command, check=True, **kwargs)


def _ssh_probe(host: str, command: str, timeout: float = 12.0) -> subprocess.CompletedProcess:
    return run_ssh(
        host,
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _normalize_render_mode(render_mode: str) -> tuple[str, str]:
    value = str(render_mode).strip()
    normalized = RENDER_MODE_ALIASES.get(value)
    if normalized is None:
        normalized = RENDER_MODE_ALIASES.get(value.lower().replace(" ", ""))
    if normalized is None:
        choices = ", ".join(RENDER_MODE_TOKENS)
        raise ValueError(f"Unsupported ovrtx render mode {render_mode!r}; choose {choices}")
    return normalized, RENDER_MODE_TOKENS[normalized]


def _normalize_aovs(aovs=()) -> tuple[str, ...]:
    if aovs is None:
        return ()
    if isinstance(aovs, str):
        values = [value for value in re.split(r"[\s,]+", aovs) if value]
    else:
        values = [str(value) for value in aovs]
    normalized = []
    for value in values:
        name = value.strip().lower()
        if name not in AOV_RENDER_VARS:
            choices = ", ".join(AOV_RENDER_VARS)
            raise ValueError(f"Unsupported ovrtx AOV {value!r}; choose from {choices}")
        if name not in normalized:
            normalized.append(name)
    return tuple(normalized)


def _requested_render_vars(aovs: tuple[str, ...]) -> tuple[str, ...]:
    # The RTPT camera sensor initializes its non-color output graph from the
    # color output. Keep LdrColor on the annotation product even though only
    # the primary product is returned as canonical RGB.
    render_vars = ["LdrColor"] if aovs else []
    for aov in aovs:
        for render_var in AOV_RENDER_VARS[aov]:
            if render_var not in render_vars:
                render_vars.append(render_var)
    return tuple(render_vars)


def _load_sdf():
    try:
        from pxr import Sdf
    except (ImportError, OSError) as exc:
        raise RuntimeError(OPENUSD_INSTALL_HINT) from exc
    return Sdf


def _openusd_error() -> str | None:
    try:
        _load_sdf()
    except RuntimeError as exc:
        return str(exc)
    return None


def backend_status(host: str = DEFAULT_HOST) -> dict[str, object]:
    error = None
    gpu = None
    installed = False
    openusd_error = _openusd_error()
    opengl_probe = shlex.quote('import ctypes; ctypes.CDLL("libOpenGL.so.0")')
    version_probe = shlex.quote(
        "from importlib.metadata import version; import ovrtx, ovstage; "
        f"assert version('ovrtx') == '{OVRTX_VERSION}'; "
        f"assert version('ovstage') == '{OVSTAGE_VERSION}'"
    )
    try:
        probe = _ssh_probe(
            host,
            f'root="$HOME/{REMOTE_ROOT}"; '
            "gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1); "
            'test -n "$gpu" || exit 3; '
            'test -x "$root/.venv/bin/python" || exit 4; '
            'test -f "$root/runtime/uv.lock" || exit 5; '
            'test -f "$root/source/robotic_renderers/backends/ovrtx.py" || exit 7; '
            f'cd "$root"; "$root/.venv/bin/python" -c {opengl_probe} || exit 8; '
            'PYTHONPATH="$root/source" "$root/.venv/bin/python" -c '
            f"{version_probe} || exit 6; "
            'printf "%s" "$gpu"',
        )
        installed = probe.returncode == 0
        gpu = probe.stdout.strip() or None
        if not installed:
            messages = {
                3: "No NVIDIA GPU was reported by nvidia-smi",
                4: "The uv-managed ovrtx environment is not installed",
                5: "The frozen ovrtx runtime project is not installed",
                6: "The ovrtx/ovstage Python packages could not be imported",
                7: "The packaged ovrtx worker source is not installed",
                8: (
                    "libOpenGL.so.0 is unavailable on the remote host; on Ubuntu run "
                    "`sudo apt-get install --no-install-recommends libopengl0`"
                ),
            }
            error = messages.get(probe.returncode, probe.stderr.strip() or "Remote probe failed")
    except (OSError, subprocess.SubprocessError) as exc:
        error = str(exc)
    if openusd_error:
        error = f"{error}; {openusd_error}" if error else openusd_error
    return {
        "backend": BACKEND_NAME,
        "available": installed and openusd_error is None,
        "version": OVRTX_VERSION if installed else None,
        "local_openusd": openusd_error is None,
        "device": gpu,
        "execution": f"SSH ({host})",
        "host": host,
        "live": True,
        "record": True,
        "render_modes": dict(RENDER_MODE_TOKENS),
        "aovs": {
            "depth": "DistanceToImagePlaneSD (metric float32)",
            "normals": "NormalSD (world-space float32)",
            "albedo": "DiffuseAlbedoSD (uint8)",
            "semantic": "SemanticSegmentation + SemanticIdMap",
            "instance": (
                "Derived from SemanticSegmentation using unique authored USD label semantics; "
                "ovrtx 0.4 has no native instance render variable"
            ),
        },
        "notes": (
            "ovrtx is Linux/Windows RTX-only. The renderer runs on the configured RTX host "
            "while OpenUSD scene generation, live display, and MP4 encoding remain on macOS. "
            "Non-color AOVs are produced by an RTPT annotation RenderProduct because ovrtx "
            "0.4 PathTracing exposes only LdrColor and HdrColor."
        ),
        "error": error,
    }


def _remote_install_script(upload_id: str, force: bool = False) -> str:
    reset = 'rm -rf "$root/.venv"' if force else ":"
    return f"""set -euo pipefail
umask 077
root="$HOME/{REMOTE_ROOT}"
incoming="$root/.uploads/{upload_id}"
test -f "$incoming/runtime/pyproject.toml"
test -f "$incoming/runtime/uv.lock"
test -f "$incoming/source/robotic_renderers/backends/ovrtx.py"
install -d -m 700 "$root"
exec 9>"$root/setup.lock"
chmod 600 "$root/setup.lock"
flock -w 120 9
{reset}
uv_bin=$(command -v uv)
test -n "$uv_bin"
UV_PROJECT_ENVIRONMENT="$root/.venv" \
  "$uv_bin" sync --directory "$incoming/runtime" --frozen --no-dev
PYTHONPATH="$incoming/source" "$root/.venv/bin/python" - <<'PY'
import ctypes
import ovrtx
import ovstage
from importlib.metadata import version

try:
    ctypes.CDLL("libOpenGL.so.0")
except OSError as exc:
    raise SystemExit(
        "ovrtx requires libOpenGL.so.0; on Ubuntu run: "
        "sudo apt-get install --no-install-recommends libopengl0"
    ) from exc
assert ovrtx.__version__ == "0.4.0"
assert version("ovrtx") == "{OVRTX_VERSION}"
assert version("ovstage") == "{OVSTAGE_VERSION}"
PY

old="$root/.previous-{upload_id}"
rm -rf "$old"
install -d -m 700 "$old"
if [ -e "$root/source" ]; then mv "$root/source" "$old/source"; fi
if [ -e "$root/runtime" ]; then mv "$root/runtime" "$old/runtime"; fi
mv "$incoming/source" "$root/source"
mv "$incoming/runtime" "$root/runtime"
rmdir "$incoming"
rm -rf "$old"

echo "official locked ovrtx runtime import succeeded"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | sed -n '1p'
"""


def setup_remote(host: str = DEFAULT_HOST, force: bool = False) -> dict[str, object]:
    """Install and synchronize the locked official ovrtx runtime on ``host``."""
    runtime_project = resolve_runtime_project(PACKAGE_SOURCE, PROJECT_ROOT, BACKEND_NAME)
    for required in (
        PACKAGE_SOURCE / "__init__.py",
        runtime_project / "pyproject.toml",
        runtime_project / "uv.lock",
    ):
        if not required.is_file():
            raise FileNotFoundError(f"Required ovrtx runtime source is missing: {required}")

    uv_probe = _ssh_probe(host, "command -v uv")
    if uv_probe.returncode != 0:
        raise RuntimeError(
            f"uv is not installed on {host}. Install it with: "
            f"ssh {host} 'curl -LsSf https://astral.sh/uv/install.sh | sh'"
        )
    upload_id = uuid.uuid4().hex
    remote_upload = f"~/{REMOTE_ROOT}/.uploads/{upload_id}"
    run_ssh(
        host,
        f'umask 077; install -d -m 700 "$HOME/{REMOTE_ROOT}/.uploads/{upload_id}/source/'
        'robotic_renderers" '
        f'"$HOME/{REMOTE_ROOT}/.uploads/{upload_id}/runtime"',
    )
    try:
        _run(
            rsync_command(
                f"{PACKAGE_SOURCE}/",
                rsync_remote(host, f"{remote_upload}/source/robotic_renderers/"),
                delete=True,
                excludes=REMOTE_PACKAGE_EXCLUDES,
            )
        )
        _run(
            rsync_command(
                f"{runtime_project}/",
                rsync_remote(host, f"{remote_upload}/runtime/"),
                delete=True,
                excludes=(".venv", "__pycache__"),
            )
        )
        completed = subprocess.run(
            ssh_command(host, "bash -s"),
            input=_remote_install_script(upload_id, force),
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            details = (completed.stdout + "\n" + completed.stderr)[-12000:]
            raise RuntimeError(f"ovrtx setup failed on {host}:\n{details}")
    finally:
        run_ssh(
            host,
            f'rm -rf "$HOME/{REMOTE_ROOT}/.uploads/{upload_id}"',
            check=False,
            timeout=30.0,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    return {
        "backend": BACKEND_NAME,
        "host": host,
        "synced": True,
        "cached": False,
        "version": OVRTX_VERSION,
        "details": completed.stdout.strip().splitlines()[-2:],
    }


def _usd_identifier(value: str) -> str:
    identifier = re.sub(r"[^A-Za-z0-9_]", "_", value)
    if not identifier or identifier[0].isdigit():
        identifier = f"_{identifier}"
    return identifier


def _format_number(value: float) -> str:
    if abs(float(value)) < 1e-12:
        return "0"
    return f"{float(value):.9g}"


def _format_vec3(values: np.ndarray) -> str:
    return ", ".join(
        f"({_format_number(x)}, {_format_number(y)}, {_format_number(z)})"
        for x, y, z in np.asarray(values).reshape(-1, 3)
    )


def _format_indices(values: np.ndarray) -> str:
    return ", ".join(str(int(value)) for value in np.asarray(values).reshape(-1))


def _plane_mesh(size: float = 100.0, tile_size: float = 1.0 / 3.0):
    repeats = size / tile_size
    vertices = np.asarray(
        [
            (-size, -size, 0.0),
            (size, -size, 0.0),
            (size, size, 0.0),
            (-size, size, 0.0),
        ],
        dtype=np.float32,
    )
    faces = np.asarray([(0, 1, 2), (0, 2, 3)], dtype=np.int32)
    uvs = np.asarray(
        [
            (0.0, 0.0),
            (repeats, 0.0),
            (repeats, repeats),
            (0.0, 0.0),
            (repeats, repeats),
            (0.0, repeats),
        ],
        dtype=np.float32,
    )
    return vertices, faces, uvs


def _geom_mesh(description: dict[str, object], geometry, meshes):
    return _shared_geom_mesh(description, geometry, meshes, renderer_name="ovrtx")


def _write_material(stream, identifier: str, description: dict[str, object]) -> None:
    rgba = description["rgba"]
    diffuse = ", ".join(_format_number(component) for component in rgba[:3])
    roughness = _format_number(max(0.035, description["roughness"]))
    specular = _format_number(min(1.0, 0.25 + 0.5 * description["specular"]))
    texture = description.get("diffuse_texture")
    diffuse_input = (
        f"                asset inputs:diffuse_texture = @./{texture}@\n"
        "                bool inputs:project_uvw = 0\n"
        if texture
        else f"                color3f inputs:diffuse_color_constant = ({diffuse})\n"
    )
    stream.write(
        f'''        def Material "{identifier}" {{
            token outputs:mdl:surface.connect = </World/Materials/{identifier}/Shader.outputs:out>
            def Shader "Shader" {{
                uniform token info:implementationSource = "sourceAsset"
                uniform asset info:mdl:sourceAsset = @OmniPBR.mdl@
                uniform token info:mdl:sourceAsset:subIdentifier = "OmniPBR"
{diffuse_input.rstrip()}
                float inputs:metallic_constant = {_format_number(description["metallic"])}
                float inputs:reflection_roughness_constant = {roughness}
                float inputs:specular_level = {specular}
                token outputs:out
            }}
        }}
'''
    )


def _write_mesh(
    stream,
    name: str,
    vertices,
    faces,
    material: str,
    indent: str,
    smooth: bool = True,
    uvs=None,
) -> None:
    if smooth:
        normals = _surface_normals(vertices, faces)
        normal_interpolation = "vertex"
    else:
        triangles = np.asarray(vertices, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
        face_normals = np.cross(
            triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
        )
        lengths = np.linalg.norm(face_normals, axis=1, keepdims=True)
        face_normals /= np.maximum(lengths, 1e-12)
        normals = np.repeat(face_normals.astype(np.float32), 3, axis=0)
        normal_interpolation = "faceVarying"
    formatted_normals = _format_vec3(normals)
    texture_coordinates = ""
    if uvs is not None:
        formatted_uvs = ", ".join(
            f"({_format_number(uv[0])}, {_format_number(uv[1])})" for uv in uvs
        )
        texture_coordinates = (
            f"{indent}    texCoord2f[] primvars:st = [{formatted_uvs}] "
            '(interpolation = "faceVarying")\n'
        )
    stream.write(
        f'''{indent}def Mesh "{name}" (
{indent}    prepend apiSchemas = ["MaterialBindingAPI"]
{indent}) {{
{indent}    uniform token subdivisionScheme = "none"
{indent}    uniform token orientation = "rightHanded"
{indent}    int[] faceVertexCounts = [{", ".join("3" for _ in range(len(faces)))}]
{indent}    int[] faceVertexIndices = [{_format_indices(faces)}]
{indent}    point3f[] points = [{_format_vec3(vertices)}]
{indent}    normal3f[] normals = [{formatted_normals}] (interpolation = "{normal_interpolation}")
{texture_coordinates.rstrip()}
{indent}    rel material:binding = </World/Materials/{material}>
{indent}}}
'''
    )


def _matrix(position, rotation) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3).T
    matrix[3, :3] = np.asarray(position, dtype=np.float64)
    return matrix


def _format_matrix(matrix: np.ndarray) -> str:
    rows = [
        f"({', '.join(_format_number(value) for value in row)})"
        for row in np.asarray(matrix).reshape(4, 4)
    ]
    return f"({', '.join(rows)})"


def _rotation_toward(direction) -> np.ndarray:
    forward = np.asarray(direction, dtype=np.float64)
    forward /= np.linalg.norm(forward)
    local_z = -forward
    reference_up = np.array([0.0, 0.0, 1.0])
    local_y = reference_up - np.dot(reference_up, local_z) * local_z
    if np.linalg.norm(local_y) < 1e-8:
        local_y = np.array([0.0, 1.0, 0.0])
    local_y /= np.linalg.norm(local_y)
    local_x = np.cross(local_y, local_z)
    return np.column_stack((local_x, local_y, local_z))


def _light_identifier(name: str) -> str:
    identifier = "".join(part.capitalize() for part in str(name).split("_"))
    return _usd_identifier(identifier)


def _lights_usda(lights: list[dict[str, object]]) -> str:
    """Translate canonical scene-bundle lights into USD Lux primitives."""
    blocks = []
    for light in lights:
        light_type = str(light["type"])
        try:
            intensity_multiplier = OVRTX_LIGHT_INTENSITY_MULTIPLIERS[light_type]
        except KeyError as exc:
            raise ValueError(f"Unsupported ovrtx light type {light_type!r}") from exc

        identifier = _light_identifier(str(light["name"]))
        color = ", ".join(_format_number(value) for value in light["diffuse"])
        intensity = _format_number(float(light["intensity"]) * intensity_multiplier)
        shadow = str(bool(light["cast_shadow"])).lower()
        position = light["position"]
        direction = light["direction"]

        if light_type == "directional":
            transform = _format_matrix(_matrix(position, _rotation_toward(direction)))
            blocks.append(
                f'''    def DistantLight "{identifier}" (
        prepend apiSchemas = ["ShadowAPI"]
    ) {{
        color3f inputs:color = ({color})
        float inputs:intensity = {intensity}
        bool inputs:shadow:enable = {shadow}
        matrix4d xformOp:transform = {transform}
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }}'''
            )
        elif light_type == "point":
            translation = ", ".join(_format_number(value) for value in position)
            radius = _format_number(light["bulb_radius"])
            blocks.append(
                f'''    def SphereLight "{identifier}" (
        prepend apiSchemas = ["ShadowAPI"]
    ) {{
        color3f inputs:color = ({color})
        float inputs:intensity = {intensity}
        float inputs:radius = {radius}
        bool inputs:shadow:enable = {shadow}
        double3 xformOp:translate = ({translation})
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }}'''
            )
        else:
            transform = _format_matrix(_matrix(position, _rotation_toward(direction)))
            radius = _format_number(light["bulb_radius"])
            # MuJoCo cutoff is the full cone angle; USD ShapingAPI expects the
            # half-angle. Shaping focus is the direct counterpart to exponent.
            cone_angle = _format_number(float(light["cutoff"]) * 0.5)
            focus = _format_number(light["exponent"])
            blocks.append(
                f'''    def DiskLight "{identifier}" (
        prepend apiSchemas = ["ShapingAPI", "ShadowAPI"]
    ) {{
        color3f inputs:color = ({color})
        float inputs:intensity = {intensity}
        float inputs:radius = {radius}
        bool inputs:shadow:enable = {shadow}
        float inputs:shaping:cone:angle = {cone_angle}
        float inputs:shaping:focus = {focus}
        matrix4d xformOp:transform = {transform}
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }}'''
            )
    return "\n".join(blocks)


def _write_gradient_sky(destination: Path) -> Path:
    return write_gradient_sky(destination, width=512, height=256)


def _write_floor_checker(destination: Path) -> Path:
    from PIL import Image

    dark = np.asarray([0.022, 0.027, 0.033])
    light = np.asarray([0.094, 0.113, 0.133])
    linear = np.empty((256, 256, 3), dtype=np.float64)
    linear[:128, :128] = dark
    linear[128:, 128:] = dark
    linear[:128, 128:] = light
    linear[128:, :128] = light
    srgb = np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * np.power(linear, 1.0 / 2.4) - 0.055,
    )
    output = destination / "floor_checker.png"
    Image.fromarray(np.rint(np.clip(srgb, 0.0, 1.0) * 255.0).astype(np.uint8)).save(output)
    return output


def _time_sampled_transforms(frames, transform_at_frame) -> str:
    lines = []
    for index, frame in enumerate(frames, start=1):
        transform = transform_at_frame(frame)
        matrix = _matrix(transform["position"], transform["rotation"])
        lines.append(f"                {index}: {_format_matrix(matrix)},")
    return "{\n" + "\n".join(lines) + "\n            }"


def _geom_semantics(index: int, description: dict[str, object]) -> tuple[str, str]:
    name = str(description["name"])
    if name == "floor":
        semantic_class = "floor"
    elif name == "red_box":
        semantic_class = "box"
    elif name == "blue_sphere":
        semantic_class = "sphere"
    elif name == "gold_cylinder":
        semantic_class = "cylinder"
    elif name in {"green_torus_mesh", "gold_gem_mesh"}:
        semantic_class = "mesh_object"
    else:
        semantic_class = "robot"
    instance = f"{_usd_identifier(name).lower()}_{index:03d}"
    return semantic_class, instance


def _render_var_definitions(render_vars: tuple[str, ...], indent: str) -> str:
    return "\n".join(
        f'''{indent}def RenderVar "{name}" {{
{indent}    string sourceName = "{name}"
{indent}}}'''
        for name in render_vars
    )


def _annotation_product_usda(
    render_vars: tuple[str, ...],
    width: int,
    height: int,
) -> str:
    if not render_vars:
        return ""
    ordered_vars = ", ".join(f"<{name}>" for name in render_vars)
    definitions = _render_var_definitions(render_vars, "        ")
    return f"""
    def RenderProduct "Annotations" {{
        int2 resolution = ({width}, {height})
        rel camera = </World/Camera>
        rel orderedVars = [{ordered_vars}]
        token omni:rtx:background:source:type = "domeLight"
        token omni:rtx:rendermode = "RealTimePathTracing"
        token[] omni:rtx:waitForEvents = ["AllLoadingFinished", "OnlyOnFirstRequest"]
        int omni:rtx:rtpt:maxBounces = 8
        int omni:rtx:rtpt:maxSpecularAndTransmissionBounces = 8
{definitions}
    }}
"""


def prepare_ovrtx_bundle(
    destination: str | Path,
    times: list[float],
    width: int,
    height: int,
    fps: int,
    samples: int = 256,
    render_mode: str = "PathTracing",
    aovs=(),
) -> Path:
    from robotic_renderers.scene_bundle import prepare_scene_bundle

    destination = Path(destination).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    render_mode_name, render_mode_token = _normalize_render_mode(render_mode)
    normalized_aovs = _normalize_aovs(aovs)
    annotation_render_vars = _requested_render_vars(normalized_aovs)
    needs_semantics = bool({"semantic", "instance"} & set(normalized_aovs))
    manifest_path = prepare_scene_bundle(destination, times, width, height, fps)
    manifest = json.loads(manifest_path.read_text())
    geometry = np.load(destination / manifest["geometry"])
    materials: dict[str, dict[str, object]] = {}
    material_ids: dict[str, str] = {}
    for geom in manifest["geoms"]:
        description = geom["material"]
        name = description["name"]
        identifier = _usd_identifier(name)
        material_ids[name] = identifier
        materials.setdefault(identifier, description)
    floor_checker_path = _write_floor_checker(destination)
    floor_checker = {
        "rgba": [1.0, 1.0, 1.0, 1.0],
        "metallic": 0.0,
        "roughness": 0.5,
        "specular": 0.25,
        "diffuse_texture": floor_checker_path.name,
    }
    materials["floor_checker"] = floor_checker
    sky_path = _write_gradient_sky(destination)

    scene_path = destination / "scene.usda"
    geom_paths = []
    with scene_path.open("w") as stream:
        stream.write(
            """#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
"""
        )
        stream.write(
            f"    startTimeCode = 1\n"
            f"    endTimeCode = {max(1, len(manifest['frames']))}\n"
            f"    timeCodesPerSecond = {fps}\n"
        )
        stream.write(
            """)

def Xform "World" {
    def Scope "Materials" {
"""
        )
        for identifier, description in materials.items():
            _write_material(stream, identifier, description)
        stream.write('    }\n    def Xform "Geometry" {\n')
        for index, geom in enumerate(manifest["geoms"]):
            identifier = f"Geom_{index:03d}_{_usd_identifier(geom['name'])}"
            geom_paths.append(f"/World/Geometry/{identifier}")
            transform_samples = _time_sampled_transforms(
                manifest["frames"], lambda frame, i=index: frame["transforms"][i]
            )
            semantic_metadata = ""
            xform_header = f'def Xform "{identifier}" {{'
            if needs_semantics:
                semantic_class, instance = _geom_semantics(index, geom)
                xform_header = (
                    f'def Xform "{identifier}" (\n'
                    "            prepend apiSchemas = "
                    '["SemanticsAPI:class", "SemanticsAPI:label"]\n'
                    "        ) {"
                )
                semantic_metadata = (
                    f"\n            string semantic:class:params:semanticData = "
                    f'"{semantic_class}"'
                    '\n            string semantic:class:params:semanticType = "class"'
                    f'\n            string semantic:label:params:semanticData = "{instance}"'
                    '\n            string semantic:label:params:semanticType = "label"'
                )
            stream.write(
                f"""        {xform_header}{semantic_metadata}
            matrix4d xformOp:transform.timeSamples = {transform_samples}
            uniform token[] xformOpOrder = ["xformOp:transform"]
"""
            )
            if geom["type"] == "plane":
                vertices, faces, uvs = _plane_mesh()
                _write_mesh(
                    stream,
                    "Tiles_0",
                    vertices,
                    faces,
                    "floor_checker",
                    "            ",
                    uvs=uvs,
                )
            else:
                vertices, faces = _geom_mesh(geom, geometry, manifest["meshes"])
                _write_mesh(
                    stream,
                    "Surface",
                    vertices,
                    faces,
                    material_ids[geom["material"]["name"]],
                    "            ",
                    smooth=(
                        geom["type"] != "box"
                        and not (
                            geom["type"] == "mesh"
                            and manifest["meshes"][str(geom["mesh_id"])]["name"] == "mesh_gem"
                        )
                    ),
                )
            stream.write("        }\n")
        fovy = float(manifest["camera"]["fovy"])
        vertical_aperture = 24.0
        focal_length = 0.5 * vertical_aperture / math.tan(math.radians(fovy) / 2.0)
        horizontal_aperture = vertical_aperture * width / height
        camera_samples = _time_sampled_transforms(manifest["frames"], lambda frame: frame["camera"])
        lights_usda = _lights_usda(manifest["lights"])
        annotation_product = _annotation_product_usda(
            annotation_render_vars,
            width,
            height,
        )
        stream.write(
            f'''    }}
    def Camera "Camera" (
        prepend apiSchemas = ["OmniRtxCameraAutoExposureAPI_1", "OmniRtxCameraExposureAPI_1"]
    ) {{
        matrix4d xformOp:transform.timeSamples = {camera_samples}
        uniform token[] xformOpOrder = ["xformOp:transform"]
        float2 clippingRange = (0.01, 50)
        float exposure = 0
        float exposure:fStop = 5
        float exposure:iso = 100
        float exposure:responsivity = 1.1026709
        float exposure:time = 0.02
        float focalLength = {_format_number(focal_length)}
        float horizontalAperture = {_format_number(horizontal_aperture)}
        float verticalAperture = {_format_number(vertical_aperture)}
        float focusDistance = 2.35
        float fStop = 0
        bool omni:rtx:autoExposure:enabled = false
    }}
{lights_usda}
    def DomeLight "StudioDome" (
        prepend apiSchemas = ["ShapingAPI"]
    ) {{
        float inputs:intensity = {_format_number(OVRTX_DOME_INTENSITY)}
        asset inputs:texture:file = @./{sky_path.name}@
        token inputs:texture:format = "latlong"
        double3 xformOp:rotateXYZ = (270, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]
    }}
}}

def Scope "Render" {{
    def RenderProduct "Camera" {{
        int2 resolution = ({width}, {height})
        rel camera = </World/Camera>
        rel orderedVars = [<LdrColor>]
        token omni:rtx:background:source:type = "domeLight"
        token omni:rtx:rendermode = "{render_mode_token}"
        token[] omni:rtx:waitForEvents = ["AllLoadingFinished", "OnlyOnFirstRequest"]
        int omni:rtx:pt:samplesPerPixel = {max(1, int(samples))}
        int omni:rtx:pt:samplesPerIteration = 4
        bool omni:rtx:pt:adaptiveSampling:enabled = true
        bool omni:rtx:pt:denoising:enabled = true
        bool omni:rtx:pt:denoising:optix:temporal = false
        int omni:rtx:pt:limits:maxBounces = 12
        int omni:rtx:pt:limits:maxGlossyBounces = 8
        int omni:rtx:rtpt:maxBounces = 8
        int omni:rtx:rtpt:maxSpecularAndTransmissionBounces = 8
        def RenderVar "LdrColor" {{
            string sourceName = "LdrColor"
        }}
    }}
{annotation_product}
}}
'''
        )

    Sdf = _load_sdf()
    layer = Sdf.Layer.FindOrOpen(str(scene_path))
    if layer is None:
        raise RuntimeError(f"Pixar USD rejected the generated scene: {scene_path}")
    crate_path = scene_path.with_suffix(".usdc")
    if not layer.Export(str(crate_path)):
        raise RuntimeError(f"Could not export binary OpenUSD scene: {crate_path}")
    scene_path.unlink()
    scene_path = crate_path

    geom_transforms = np.asarray(
        [
            [
                _matrix(transform["position"], transform["rotation"])
                for transform in frame["transforms"]
            ]
            for frame in manifest["frames"]
        ],
        dtype=np.float64,
    )
    camera_transforms = np.asarray(
        [
            [_matrix(frame["camera"]["position"], frame["camera"]["rotation"])]
            for frame in manifest["frames"]
        ],
        dtype=np.float64,
    )
    animation_path = destination / "animation.npz"
    np.savez_compressed(
        animation_path,
        geom_transforms=geom_transforms,
        camera_transforms=camera_transforms,
    )
    bundle = {
        "scene": scene_path.name,
        "animation": animation_path.name,
        "geom_paths": geom_paths,
        "camera_path": "/World/Camera",
        "render_product": "/Render/Camera",
        "width": width,
        "height": height,
        "fps": fps,
        "frames": len(times),
        "samples": samples,
        "render_mode": render_mode_token,
        "render_mode_name": render_mode_name,
        "aovs": list(normalized_aovs),
        "annotation_render_vars": list(annotation_render_vars),
        "annotation_product": "/Render/Annotations" if annotation_render_vars else None,
        "animated_in_usd": True,
    }
    bundle_path = destination / "ovrtx_bundle.json"
    bundle_path.write_text(json.dumps(bundle, separators=(",", ":")))
    return bundle_path


def _upload_job(host: str, local_job: Path, job_id: str) -> str:
    remote_job = f".jobs/{job_id}"
    run_ssh(
        host,
        f'umask 077; install -d -m 700 "$HOME/{REMOTE_ROOT}/{remote_job}"',
    )
    try:
        _run(
            rsync_command(
                f"{local_job}/",
                rsync_remote(host, f"~/{REMOTE_ROOT}/{remote_job}/"),
            )
        )
    except BaseException:
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            _cleanup_remote_job(host, job_id)
        raise
    return remote_job


def _cleanup_remote_job(host: str, job_id: str) -> None:
    run_ssh(
        host,
        f'rm -rf "$HOME/{REMOTE_ROOT}/.jobs/{job_id}"',
        check=False,
        timeout=30.0,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _remote_worker_command(remote_job: str, loop: bool) -> str:
    arguments = [
        ".venv/bin/python",
        "-m",
        "robotic_renderers.backends.ovrtx",
        "__worker",
        "--bundle",
        f"{remote_job}/ovrtx_bundle.json",
    ]
    if loop:
        arguments.append("--loop")
    return (
        f'cd "$HOME/{REMOTE_ROOT}" && '
        'export PYTHONPATH="$PWD/source" && '
        f"exec setsid --wait {shlex.join(arguments)}"
    )


def _terminate_remote_worker(host: str, remote_job: str) -> None:
    command = worker_cleanup_script(
        REMOTE_ROOT,
        remote_job,
        "robotic_renderers.backends.ovrtx",
        "worker.pid",
    )
    with contextlib.suppress(OSError, subprocess.SubprocessError):
        subprocess.run(
            ssh_command(host, command),
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=8,
        )


def _stderr_after_exit(process: subprocess.Popen[bytes]) -> str:
    """Read stderr only when EOF is guaranteed by an exited child process."""
    if process.poll() is None:
        return ""
    assert process.stderr is not None
    return process.stderr.read().decode(errors="replace")


def _terminate_and_reap(process: subprocess.Popen[bytes], timeout: float = 5.0) -> None:
    """Terminate a child process and always wait after a SIGKILL escalation."""
    if process.poll() is None:
        with contextlib.suppress(ProcessLookupError):
            process.terminate()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            process.kill()
        process.wait()


def _remote_stream_frames(
    host: str,
    remote_job: str,
    width: int,
    height: int,
    loop: bool,
    aovs=(),
):
    normalized_aovs = _normalize_aovs(aovs)
    process = subprocess.Popen(
        ssh_command(host, _remote_worker_command(remote_job, loop)),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    assert process.stdout is not None
    assert process.stderr is not None
    completed_normally = False
    try:
        handshake = process.stdout.readline(MAX_HANDSHAKE_BYTES)
        if handshake == STREAM_ERROR:
            message_length = struct.unpack(
                "!I",
                _read_exact(process.stdout, 4, max_bytes=4),
            )[0]
            message = _read_exact(
                process.stdout,
                message_length,
                max_bytes=MAX_ERROR_BYTES,
            ).decode(errors="replace")
            raise RuntimeError(f"ovrtx worker failed before rendering:\n{message}")
        expected_magic = STREAM_MAGIC_AOV if normalized_aovs else STREAM_MAGIC
        if handshake != expected_magic:
            stderr = _stderr_after_exit(process)
            raise RuntimeError(
                "ovrtx worker ended before its stream handshake:\n"
                + handshake.decode(errors="replace")
                + stderr[-12000:]
            )
        expected = width * height * 3
        while True:
            length_bytes = process.stdout.read(4)
            if not length_bytes:
                break
            if len(length_bytes) != 4:
                length_bytes += _read_exact(
                    process.stdout,
                    4 - len(length_bytes),
                    max_bytes=4,
                )
            frame_length = struct.unpack("!I", length_bytes)[0]
            if frame_length == 0:
                message_length = struct.unpack(
                    "!I",
                    _read_exact(process.stdout, 4, max_bytes=4),
                )[0]
                message = _read_exact(
                    process.stdout,
                    message_length,
                    max_bytes=MAX_ERROR_BYTES,
                ).decode(errors="replace")
                raise RuntimeError(f"ovrtx worker failed while rendering:\n{message}")
            if not normalized_aovs and frame_length != expected:
                raise RuntimeError(f"ovrtx sent {frame_length} bytes; expected {expected}")
            payload_limit = _aov_payload_limit(width, height) if normalized_aovs else expected
            payload = _read_exact(
                process.stdout,
                frame_length,
                max_bytes=payload_limit,
            )
            if normalized_aovs:
                yield _decode_aov_frame(payload)
            else:
                frame = np.frombuffer(payload, dtype=np.uint8)
                yield frame.reshape(height, width, 3).copy()
        return_code = process.wait()
        if return_code != 0:
            stderr = process.stderr.read().decode(errors="replace")
            raise RuntimeError(f"ovrtx worker exited with {return_code}:\n{stderr[-12000:]}")
        completed_normally = True
    finally:
        if not completed_normally:
            _terminate_remote_worker(host, remote_job)
        _terminate_and_reap(process)


def _frame_source(
    times: list[float],
    width: int,
    height: int,
    fps: int,
    samples: int,
    host: str,
    render_mode: str,
    loop: bool = False,
    aovs=(),
):
    render_mode_name, _ = _normalize_render_mode(render_mode)
    normalized_aovs = _normalize_aovs(aovs)
    setup_remote(host)
    job_id = uuid.uuid4().hex
    with tempfile.TemporaryDirectory(prefix="ovrtx-", dir=_job_root()) as temporary:
        job = Path(temporary)
        prepare_ovrtx_bundle(
            job,
            times,
            width,
            height,
            fps,
            samples,
            render_mode_name,
            normalized_aovs,
        )
        remote_job = _upload_job(host, job, job_id)
        try:
            for frame in _remote_stream_frames(
                host,
                remote_job,
                width,
                height,
                loop,
                normalized_aovs,
            ):
                if isinstance(frame, dict):
                    yield frame
                else:
                    yield {"rgb": frame, "_labels": {}}
        finally:
            _cleanup_remote_job(host, job_id)


def _aov_directory(
    aov_output_dir: str | Path | None,
    purpose: str,
    primary_output: str | Path | None = None,
) -> Path:
    if aov_output_dir is not None:
        return Path(aov_output_dir).expanduser().resolve()
    if primary_output is not None:
        primary = Path(primary_output).expanduser().resolve()
        return primary.parent / f"{primary.stem}_aovs"
    from robotic_renderers.defaults import output_directory

    return output_directory() / f"ovrtx_{purpose}_aovs"


def _aov_preview(name: str, array: np.ndarray) -> np.ndarray:
    values = np.asarray(array)
    if name == "depth":
        depth = values.astype(np.float32)
        valid = np.isfinite(depth) & (depth > 0.0)
        preview = np.zeros(depth.shape, dtype=np.float32)
        near, far = DEPTH_PREVIEW_RANGE_METERS
        clamped = np.clip(depth[valid], near, far)
        preview[valid] = 1.0 - (clamped - near) / (far - near)
        image = np.repeat(preview[..., None], 3, axis=-1)
        return np.rint(image * 255.0).astype(np.uint8)
    if name == "normals":
        image = np.clip(values[..., :3] * 0.5 + 0.5, 0.0, 1.0)
        return np.rint(image * 255.0).astype(np.uint8)
    if name == "albedo":
        return np.asarray(values[..., :3], dtype=np.uint8)
    identifiers = np.asarray(values, dtype=np.uint32)
    red = (identifiers * np.uint32(73) + np.uint32(41)) & np.uint32(255)
    green = (identifiers * np.uint32(151) + np.uint32(17)) & np.uint32(255)
    blue = (identifiers * np.uint32(199) + np.uint32(89)) & np.uint32(255)
    image = np.stack((red, green, blue), axis=-1).astype(np.uint8)
    image[identifiers == 0] = 0
    return image


def _aov_source(name: str) -> str:
    if name == "instance":
        return "SemanticSegmentation + SemanticIdMap (derived label channel)"
    return " + ".join(AOV_RENDER_VARS[name])


def _new_aov_staging_directory(output_dir: Path) -> Path:
    output_dir = output_dir.expanduser().resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    return Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.part-",
            dir=output_dir.parent,
        )
    )


def _publish_staged_aovs(staging: Path, output_dir: Path, artifact_prefix: str) -> None:
    """Publish a fully staged AOV set and roll back any interrupted swap."""
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = sorted(staging.iterdir(), key=lambda path: path.name)
    if not artifacts:
        raise RuntimeError("Cannot publish an empty ovrtx AOV set")
    previous_artifacts = sorted(
        (path for path in output_dir.iterdir() if path.name.startswith(f"{artifact_prefix}_")),
        key=lambda path: path.name,
    )

    backup = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.previous-",
            dir=output_dir.parent,
        )
    )
    backed_up: list[Path] = []
    published: list[Path] = []
    try:
        for target in previous_artifacts:
            os.replace(target, backup / target.name)
            backed_up.append(target)
        for artifact in artifacts:
            target = output_dir / artifact.name
            os.replace(artifact, target)
            published.append(target)
    except BaseException:
        for target in reversed(published):
            if target.exists() or target.is_symlink():
                os.replace(target, staging / target.name)
        for target in reversed(backed_up):
            previous = backup / target.name
            if previous.exists() or previous.is_symlink():
                os.replace(previous, target)
        raise
    finally:
        shutil.rmtree(backup, ignore_errors=True)


def _write_aov_frame(
    frame: dict[str, object],
    output_dir: Path,
    prefix: str,
    aovs: tuple[str, ...],
) -> dict[str, object]:
    from robotic_renderers.media import write_png

    output_dir = output_dir.expanduser().resolve()
    staging = _new_aov_staging_directory(output_dir)
    labels = frame.get("_labels", {})
    labels = labels if isinstance(labels, dict) else {}
    outputs = {}
    try:
        for name in aovs:
            array = np.asarray(frame[name])
            data_name = f"{prefix}_{name}.npy"
            preview_name = f"{prefix}_{name}.png"
            np.save(staging / data_name, array, allow_pickle=False)
            write_png(
                _aov_preview(name, array),
                staging / preview_name,
            )
            description: dict[str, object] = {
                "data": str(output_dir / data_name),
                "preview": str(output_dir / preview_name),
                "render_var": _aov_source(name),
                "dtype": str(array.dtype),
                "shape": list(array.shape),
            }
            if name == "depth":
                description["preview_range_m"] = list(DEPTH_PREVIEW_RANGE_METERS)
            if name in labels:
                labels_name = f"{prefix}_{name}_labels.json"
                (staging / labels_name).write_text(
                    json.dumps(labels[name], indent=2, sort_keys=True) + "\n"
                )
                description["labels"] = str(output_dir / labels_name)
            outputs[name] = description
        _publish_staged_aovs(staging, output_dir, prefix)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return outputs


def _single_frame_data(
    width: int,
    height: int,
    time_s: float,
    samples: int,
    host: str,
    render_mode: str,
    aovs: tuple[str, ...],
) -> dict[str, object]:
    source = _frame_source(
        [time_s],
        width,
        height,
        1,
        samples,
        host,
        render_mode,
        aovs=aovs,
    )
    try:
        return next(source)
    finally:
        source.close()


def render_frame(
    width: int = 1280,
    height: int = 720,
    time_s: float = 1.0,
    samples: int = 256,
    host: str = DEFAULT_HOST,
    render_mode: str = "pathtraced",
    aovs=(),
    aov_output_dir: str | Path | None = None,
) -> np.ndarray:
    normalized_mode, _ = _normalize_render_mode(render_mode)
    normalized_aovs = _normalize_aovs(aovs)
    frame = _single_frame_data(
        width,
        height,
        time_s,
        samples,
        host,
        normalized_mode,
        normalized_aovs,
    )
    if normalized_aovs:
        _write_aov_frame(
            frame,
            _aov_directory(aov_output_dir, "frame"),
            "frame",
            normalized_aovs,
        )
    return np.asarray(frame["rgb"], dtype=np.uint8)


def render_still(
    output_path: str | Path,
    width: int = 1280,
    height: int = 720,
    time_s: float = 1.0,
    samples: int = 256,
    host: str = DEFAULT_HOST,
    render_mode: str = "pathtraced",
    aovs=(),
    aov_output_dir: str | Path | None = None,
) -> dict[str, object]:
    from robotic_renderers.media import image_digest, write_png

    normalized_mode, render_mode_token = _normalize_render_mode(render_mode)
    normalized_aovs = _normalize_aovs(aovs)
    started = time.perf_counter()
    frame = _single_frame_data(
        width,
        height,
        time_s,
        samples,
        host,
        normalized_mode,
        normalized_aovs,
    )
    image = np.asarray(frame["rgb"], dtype=np.uint8)
    output = write_png(image, output_path)
    aov_outputs = {}
    if normalized_aovs:
        aov_outputs = _write_aov_frame(
            frame,
            _aov_directory(aov_output_dir, "still", output),
            Path(output).stem,
            normalized_aovs,
        )
    return {
        "backend": BACKEND_NAME,
        "output": str(output),
        "seconds": time.perf_counter() - started,
        "sha256": image_digest(image),
        "width": width,
        "height": height,
        "samples": samples,
        "render_mode": normalized_mode,
        "render_mode_token": render_mode_token,
        "aovs": aov_outputs,
        "device": f"NVIDIA RTX on {host}",
    }


def _live_rgb_frames(
    frames,
    aovs: tuple[str, ...],
    output_dir: Path | None,
    outputs: dict[str, object],
):
    try:
        for frame in frames:
            if aovs and output_dir is not None:
                outputs.clear()
                outputs.update(_write_aov_frame(frame, output_dir, "latest", aovs))
            yield np.asarray(frame["rgb"], dtype=np.uint8)
    finally:
        close = getattr(frames, "close", None)
        if close is not None:
            close()


def render_live(
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    duration: float | None = None,
    samples: int = 64,
    host: str = DEFAULT_HOST,
    render_mode: str = "realtime",
    aovs=(),
    aov_output_dir: str | Path | None = None,
) -> dict[str, object]:
    from robotic_renderers.media import display_frames
    from robotic_renderers.scene import CAMERA_ORBIT_PERIOD, trajectory_times

    normalized_mode, render_mode_token = _normalize_render_mode(render_mode)
    normalized_aovs = _normalize_aovs(aovs)
    source_duration = duration if duration is not None else CAMERA_ORBIT_PERIOD
    times = list(trajectory_times(source_duration, fps))
    aov_outputs: dict[str, object] = {}
    output_dir = _aov_directory(aov_output_dir, "live") if normalized_aovs else None
    frames = _frame_source(
        times,
        width,
        height,
        fps,
        samples,
        host,
        normalized_mode,
        loop=duration is None,
        aovs=normalized_aovs,
    )
    result = display_frames(
        _live_rgb_frames(frames, normalized_aovs, output_dir, aov_outputs),
        width,
        height,
        f"NVIDIA ovrtx {normalized_mode} ({host})",
        target_fps=fps,
    )
    return {
        "backend": BACKEND_NAME,
        "host": host,
        "render_mode": normalized_mode,
        "render_mode_token": render_mode_token,
        "aovs": aov_outputs,
        **result,
    }


def _video_writer(output_path: Path, fps: int):
    import imageio.v2 as imageio

    output_path.parent.mkdir(parents=True, exist_ok=True)
    return imageio.get_writer(
        output_path,
        format="FFMPEG",
        mode="I",
        fps=fps,
        codec="libx264",
        quality=7,
        pixelformat="yuv420p",
        macro_block_size=1,
        ffmpeg_log_level="warning",
        output_params=["-movflags", "+faststart"],
    )


@contextlib.contextmanager
def _atomic_video_writer(output_path: Path, fps: int):
    from robotic_renderers.media import atomic_output_path

    with atomic_output_path(output_path) as (temporary, output):
        writer = _video_writer(temporary, fps)
        try:
            yield writer, output
        finally:
            writer.close()


def _write_video_and_aovs(
    frames,
    output_path: str | Path,
    fps: int,
    aovs: tuple[str, ...],
    aov_output_dir: Path | None,
    timings: dict[str, float] | None = None,
    expected_frame_count: int | None = None,
) -> tuple[Path, int, float, dict[str, object]]:
    from robotic_renderers.media import atomic_output_path, ensure_rgb8

    output = Path(output_path).expanduser().resolve()
    roundtrip_started = time.perf_counter()
    frame_count = 0
    remote_production_seconds = 0.0
    remote_cleanup_seconds = 0.0
    local_encode_seconds = 0.0
    publication_seconds = 0.0
    aov_writers: dict[str, object] = {}
    aov_frame_dirs: dict[str, Path] = {}
    aov_metadata: dict[str, dict[str, object]] = {}
    aov_outputs: dict[str, object] = {}
    labels_seen: dict[str, object] = {}
    aov_staging = None
    source = iter(frames)
    video_writers: list[object] = []
    try:
        with atomic_output_path(output) as (temporary_output, output):
            try:
                rgb_writer = _video_writer(temporary_output, fps)
                video_writers.append(rgb_writer)
                if aovs:
                    assert aov_output_dir is not None
                    aov_output_dir = aov_output_dir.expanduser().resolve()
                    aov_staging = _new_aov_staging_directory(aov_output_dir)
                    for name in aovs:
                        video_name = f"{output.stem}_{name}.mp4"
                        frames_name = f"{output.stem}_{name}_frames"
                        frames_dir = aov_staging / frames_name
                        frames_dir.mkdir(parents=True)
                        writer = _video_writer(aov_staging / video_name, fps)
                        video_writers.append(writer)
                        aov_writers[name] = writer
                        aov_frame_dirs[name] = frames_dir
                        aov_outputs[name] = {
                            "video": str(aov_output_dir / video_name),
                            "frames_dir": str(aov_output_dir / frames_name),
                            "render_var": _aov_source(name),
                        }
                        if name == "depth":
                            aov_outputs[name]["preview_range_m"] = list(DEPTH_PREVIEW_RANGE_METERS)

                while expected_frame_count is None or frame_count < expected_frame_count:
                    production_started = time.perf_counter()
                    try:
                        frame = next(source)
                    except StopIteration:
                        remote_production_seconds += time.perf_counter() - production_started
                        break
                    except BaseException:
                        remote_production_seconds += time.perf_counter() - production_started
                        raise
                    remote_production_seconds += time.perf_counter() - production_started

                    encode_started = time.perf_counter()
                    rgb_writer.append_data(ensure_rgb8(frame["rgb"]))
                    for name, writer in aov_writers.items():
                        array = np.asarray(frame[name])
                        writer.append_data(_aov_preview(name, array))
                        np.save(
                            aov_frame_dirs[name] / f"frame_{frame_count:06d}.npy",
                            array,
                            allow_pickle=False,
                        )
                        metadata = aov_metadata.setdefault(
                            name,
                            {"dtype": str(array.dtype), "shape": list(array.shape)},
                        )
                        if metadata["dtype"] != str(array.dtype) or metadata["shape"] != list(
                            array.shape
                        ):
                            raise RuntimeError(
                                f"ovrtx {name} AOV changed dtype or shape during video"
                            )
                    labels = frame.get("_labels", {})
                    if isinstance(labels, dict):
                        labels_seen.update(labels)
                    frame_count += 1
                    local_encode_seconds += time.perf_counter() - encode_started
            finally:
                encode_started = time.perf_counter()
                close_error = None
                for writer in reversed(video_writers):
                    try:
                        writer.close()
                    except BaseException as exc:
                        if close_error is None:
                            close_error = exc
                local_encode_seconds += time.perf_counter() - encode_started

                production_started = time.perf_counter()
                close = getattr(source, "close", None)
                if close is not None:
                    close()
                remote_cleanup_seconds += time.perf_counter() - production_started
                if close_error is not None:
                    raise close_error

            if frame_count == 0:
                raise RuntimeError("ovrtx video source produced no frames")
            if expected_frame_count is not None and frame_count != expected_frame_count:
                raise RuntimeError(
                    f"ovrtx video source produced {frame_count} frames; "
                    f"expected {expected_frame_count}"
                )

            if aov_staging is not None:
                encode_started = time.perf_counter()
                for name, labels in labels_seen.items():
                    if name not in aov_outputs:
                        continue
                    labels_name = f"{output.stem}_{name}_labels.json"
                    (aov_staging / labels_name).write_text(
                        json.dumps(labels, indent=2, sort_keys=True) + "\n"
                    )
                    aov_outputs[name]["labels"] = str(aov_output_dir / labels_name)
                for name, description in aov_outputs.items():
                    manifest_name = f"{output.stem}_{name}_manifest.json"
                    manifest = {
                        "backend": BACKEND_NAME,
                        "aov": name,
                        "render_var": description["render_var"],
                        "fps": fps,
                        "frames": frame_count,
                        "dtype": aov_metadata[name]["dtype"],
                        "shape": aov_metadata[name]["shape"],
                        "frame_pattern": (f"{Path(description['frames_dir']).name}/frame_%06d.npy"),
                        "preview_video": Path(description["video"]).name,
                    }
                    if "preview_range_m" in description:
                        manifest["preview_range_m"] = description["preview_range_m"]
                    if "labels" in description:
                        manifest["labels"] = Path(description["labels"]).name
                    (aov_staging / manifest_name).write_text(
                        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
                    )
                    description["manifest"] = str(aov_output_dir / manifest_name)
                local_encode_seconds += time.perf_counter() - encode_started

                publication_started = time.perf_counter()
                _publish_staged_aovs(aov_staging, aov_output_dir, output.stem)
                publication_seconds = time.perf_counter() - publication_started
    finally:
        if aov_staging is not None:
            shutil.rmtree(aov_staging, ignore_errors=True)

    roundtrip_seconds = time.perf_counter() - roundtrip_started
    if timings is not None:
        timings.update(
            {
                "remote_production_seconds": remote_production_seconds,
                "remote_cleanup_seconds": remote_cleanup_seconds,
                "render_seconds": remote_production_seconds,
                "local_encode_seconds": local_encode_seconds,
                "encode_seconds": local_encode_seconds,
                "aov_publication_seconds": publication_seconds,
                "publication_seconds": publication_seconds,
                "roundtrip_seconds": roundtrip_seconds,
            }
        )
    return output, frame_count, local_encode_seconds, aov_outputs


def record_video(
    output_path: str | Path,
    duration: float = 30.0,
    fps: int = 30,
    width: int = 1280,
    height: int = 720,
    samples: int = 256,
    host: str = DEFAULT_HOST,
    render_mode: str = "pathtraced",
    aovs=(),
    aov_output_dir: str | Path | None = None,
) -> dict[str, object]:
    from robotic_renderers.scene import trajectory_times

    normalized_mode, render_mode_token = _normalize_render_mode(render_mode)
    normalized_aovs = _normalize_aovs(aovs)
    started = time.perf_counter()
    times = list(trajectory_times(duration, fps))
    output_dir = _aov_directory(aov_output_dir, "video", output_path) if normalized_aovs else None
    timing: dict[str, float] = {}
    output, frames, encode_seconds, aov_outputs = _write_video_and_aovs(
        _frame_source(
            times,
            width,
            height,
            fps,
            samples,
            host,
            normalized_mode,
            aovs=normalized_aovs,
        ),
        output_path,
        fps,
        normalized_aovs,
        output_dir,
        timings=timing,
        expected_frame_count=len(times),
    )
    seconds = time.perf_counter() - started
    render_seconds = timing["render_seconds"]
    return {
        "backend": BACKEND_NAME,
        "output": str(output),
        "frames": frames,
        "seconds": seconds,
        "roundtrip_seconds": seconds,
        "roundtrip_fps": frames / max(seconds, 1e-9),
        "render_seconds": render_seconds,
        "remote_production_seconds": timing["remote_production_seconds"],
        "remote_cleanup_seconds": timing["remote_cleanup_seconds"],
        "encode_seconds": encode_seconds,
        "local_encode_seconds": timing["local_encode_seconds"],
        "aov_publication_seconds": timing["aov_publication_seconds"],
        "publication_seconds": timing["publication_seconds"],
        "video_fps": fps,
        "render_fps": frames / max(render_seconds, 1e-9),
        "width": width,
        "height": height,
        "samples": samples,
        "render_mode": normalized_mode,
        "render_mode_token": render_mode_token,
        "aovs": aov_outputs,
        "device": f"NVIDIA RTX on {host}",
    }


def benchmark(
    frame_count: int = 10,
    width: int = 640,
    height: int = 360,
    fps: int = 30,
    warmup: int = 3,
    samples: int = 32,
    host: str = DEFAULT_HOST,
    render_mode: str = "pathtraced",
    aovs=(),
    aov_output_dir: str | Path | None = None,
) -> dict[str, object]:
    from robotic_renderers.media import image_digest

    if frame_count <= 0:
        raise ValueError(f"frame_count must be positive, got {frame_count}")
    if warmup < 0:
        raise ValueError(f"warmup must be non-negative, got {warmup}")
    normalized_mode, render_mode_token = _normalize_render_mode(render_mode)
    normalized_aovs = _normalize_aovs(aovs)
    # At least one unmeasured frame is needed because creating the remote
    # renderer and waiting for its stream handshake occur on the first pull.
    effective_warmup = max(1, warmup)
    total = effective_warmup + frame_count
    times = [index / fps for index in range(total)]
    roundtrip_started = time.perf_counter()
    source = iter(
        _frame_source(
            times,
            width,
            height,
            fps,
            samples,
            host,
            normalized_mode,
            aovs=normalized_aovs,
        )
    )
    last = None
    try:
        for _ in range(effective_warmup):
            next(source)
        render_started = time.perf_counter()
        setup_warmup_seconds = render_started - roundtrip_started
        for _ in range(frame_count):
            last = next(source)
        render_seconds = time.perf_counter() - render_started
    except StopIteration as exc:
        raise RuntimeError("ovrtx benchmark produced fewer frames than requested") from exc
    finally:
        close = getattr(source, "close", None)
        if close is not None:
            close()
    if last is None:
        raise RuntimeError("ovrtx benchmark produced no measured frames")
    aov_outputs = {}
    if normalized_aovs:
        aov_outputs = _write_aov_frame(
            last,
            _aov_directory(aov_output_dir, "benchmark"),
            "last_frame",
            normalized_aovs,
        )
    roundtrip_seconds = time.perf_counter() - roundtrip_started
    return {
        "backend": BACKEND_NAME,
        "device": f"NVIDIA RTX on {host}",
        "width": width,
        "height": height,
        "samples": samples,
        "frames": frame_count,
        "warmup": warmup,
        "effective_warmup": effective_warmup,
        "setup_warmup_seconds": setup_warmup_seconds,
        "render_mode": normalized_mode,
        "render_mode_token": render_mode_token,
        "aovs": aov_outputs,
        "render_seconds": render_seconds,
        "render_fps": frame_count / max(render_seconds, 1e-9),
        "roundtrip_seconds": roundtrip_seconds,
        "roundtrip_fps": frame_count / max(roundtrip_seconds, 1e-9),
        "last_frame_sha256": image_digest(last["rgb"]),
    }


def _map_render_var(frame, name: str) -> np.ndarray:
    import ovrtx

    mapping = frame.render_vars[name].map(device=ovrtx.Device.CPU)
    try:
        view = np.from_dlpack(mapping)
        result = np.ascontiguousarray(view).copy()
        del view
        return result
    finally:
        mapping.unmap()


def _decode_semantic_id_map(tensor: np.ndarray) -> dict[int, str]:
    data = np.ascontiguousarray(tensor).view(np.uint8).reshape(-1)
    if data.size < 4:
        return {}
    entry_dtype = np.dtype([("id", "<u4", (4,)), ("label_length", "<u4"), ("label_offset", "<u4")])
    num_entries = int.from_bytes(data[-4:].tobytes(), byteorder="little")
    entries_size = num_entries * entry_dtype.itemsize
    if entries_size > data.size - 4:
        raise RuntimeError("ovrtx SemanticIdMap has an invalid entry table")
    entries = data[:entries_size].view(entry_dtype).reshape(num_entries)
    labels = {}
    for entry in entries:
        label_offset = int(entry["label_offset"])
        label_length = int(entry["label_length"])
        label_end = label_offset + label_length
        if label_end > data.size:
            raise RuntimeError("ovrtx SemanticIdMap has an invalid label offset")
        label = data[label_offset:label_end].tobytes().decode("utf-8")
        labels[int(entry["id"][0])] = label.rstrip("\x00").rstrip()
    return labels


def _semantic_fields(label: str) -> dict[str, str]:
    fields = {}
    for component in label.split(";"):
        key, separator, value = component.partition(":")
        if separator and value.strip():
            fields[key.strip()] = value.strip()
    return fields


def _remap_segmentation(
    raw_segmentation: np.ndarray,
    semantic_id_map: np.ndarray,
    semantic_type: str,
) -> tuple[np.ndarray, dict[str, str]]:
    labels_by_raw_id = _decode_semantic_id_map(semantic_id_map)
    values_by_raw_id = {
        raw_id: _semantic_fields(label).get(semantic_type)
        for raw_id, label in labels_by_raw_id.items()
    }
    values = sorted({value for value in values_by_raw_id.values() if value})
    id_by_value = {value: index for index, value in enumerate(values, start=1)}
    output = np.zeros(np.squeeze(raw_segmentation).shape, dtype=np.uint32)
    for raw_id, value in values_by_raw_id.items():
        if value:
            output[np.squeeze(raw_segmentation) == raw_id] = id_by_value[value]
    labels = {"0": "unlabeled"}
    labels.update({str(identifier): value for value, identifier in id_by_value.items()})
    return output, labels


def _encode_aov_frame(frame: dict[str, object]) -> bytes:
    metadata = json.dumps(frame.get("_labels", {}), separators=(",", ":")).encode()
    arrays = {
        name: value
        for name, value in frame.items()
        if name != "_labels" and isinstance(value, np.ndarray)
    }
    arrays["_metadata"] = np.frombuffer(metadata, dtype=np.uint8)
    output = io.BytesIO()
    np.savez(output, **arrays)
    return output.getvalue()


def _decode_aov_frame(payload: bytes) -> dict[str, object]:
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        frame = {
            name: np.ascontiguousarray(archive[name]).copy()
            for name in archive.files
            if name != "_metadata"
        }
        metadata = bytes(np.asarray(archive["_metadata"], dtype=np.uint8))
    frame["_labels"] = json.loads(metadata.decode()) if metadata else {}
    return frame


def _worker_frame(
    renderer,
    bundle,
    animation,
    index: int,
    stage=None,
    ordinal: int | None = None,
    update_stage: bool = True,
) -> dict[str, object]:
    import ovrtx

    if bundle.get("animated_in_usd"):
        time_code = (index + 1) / bundle["fps"]
        if stage is None:
            # Compatibility path retained for isolated tests and ovrtx 0.3
            # callers. The 0.4 worker uses ovstage below.
            renderer.update_from_usd_time(time_code)
        elif update_stage:
            import ovstage

            assert ordinal is not None
            ovstage.population.update_from_usd_time(
                stage,
                ordinal=ordinal,
                time_code=time_code,
            )
            stage.advance_write_floor(ordinal, ovstage.Scope.ALL).wait()
    else:
        if index == 0:
            print("[robot-rendering] applying geometry transforms", flush=True)
        renderer.write_attribute(
            prim_paths=bundle["geom_paths"],
            attribute_name="omni:xform",
            tensor=animation["geom_transforms"][index],
            semantic=ovrtx.Semantic.XFORM_MAT4x4,
        )
        if index == 0:
            print("[robot-rendering] applying camera transform", flush=True)
        renderer.write_attribute(
            prim_paths=[bundle["camera_path"]],
            attribute_name="omni:xform",
            tensor=animation["camera_transforms"][index],
            semantic=ovrtx.Semantic.XFORM_MAT4x4,
        )
    if index == 0:
        print("[robot-rendering] stepping RTX renderer", flush=True)
    step_arguments = {
        "render_products": {bundle["render_product"]},
        "delta_time": 1.0 / bundle["fps"],
    }
    if stage is not None:
        step_arguments["ordinal"] = ordinal
    products = renderer.step(**step_arguments)
    product = products[bundle["render_product"]]
    if not product.frames:
        raise RuntimeError("ovrtx returned no camera frame")
    if index == 0:
        print("[robot-rendering] mapping first LdrColor frame", flush=True)
    ldr_color = _map_render_var(product.frames[0], "LdrColor")
    expected_shape = (bundle["height"], bundle["width"], 4)
    if ldr_color.shape != expected_shape:
        raise RuntimeError(f"ovrtx LdrColor has shape {ldr_color.shape}; expected {expected_shape}")
    outputs: dict[str, object] = {
        "rgb": np.ascontiguousarray(ldr_color[:, :, :3]),
        "_labels": {},
    }
    aovs = tuple(bundle.get("aovs", ()))
    if not aovs:
        return outputs

    annotation_arguments = {
        "render_products": {bundle["annotation_product"]},
        "delta_time": 1.0 / bundle["fps"],
    }
    if stage is not None:
        annotation_arguments["ordinal"] = ordinal
    annotation_products = renderer.step(**annotation_arguments)
    annotation_product = annotation_products[bundle["annotation_product"]]
    if not annotation_product.frames:
        raise RuntimeError("ovrtx returned no annotation frame")
    annotation_frame = annotation_product.frames[0]
    mapped = {
        name: _map_render_var(annotation_frame, name) for name in bundle["annotation_render_vars"]
    }
    if "depth" in aovs:
        outputs["depth"] = np.ascontiguousarray(mapped["DistanceToImagePlaneSD"][..., 0])
    if "normals" in aovs:
        outputs["normals"] = np.ascontiguousarray(mapped["NormalSD"][..., :3])
    if "albedo" in aovs:
        outputs["albedo"] = np.ascontiguousarray(mapped["DiffuseAlbedoSD"][..., :3])
    if {"semantic", "instance"} & set(aovs):
        raw_segmentation = mapped["SemanticSegmentation"]
        semantic_id_map = mapped["SemanticIdMap"]
        labels = outputs["_labels"]
        assert isinstance(labels, dict)
        if "semantic" in aovs:
            outputs["semantic"], labels["semantic"] = _remap_segmentation(
                raw_segmentation,
                semantic_id_map,
                "class",
            )
        if "instance" in aovs:
            outputs["instance"], labels["instance"] = _remap_segmentation(
                raw_segmentation,
                semantic_id_map,
                "label",
            )
    return outputs


def _worker_main(arguments: list[str]) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args(arguments)

    binary_fd = os.dup(sys.stdout.fileno())
    bundle_path = Path(args.bundle).resolve()
    pid_path = bundle_path.parent / "worker.pid"
    write_pid_identity(pid_path)
    log_path = bundle_path.parent / "ovrtx_worker.log"
    log_stream = log_path.open("a", buffering=1)
    os.dup2(log_stream.fileno(), sys.stdout.fileno())
    os.dup2(log_stream.fileno(), sys.stderr.fileno())
    binary = os.fdopen(binary_fd, "wb", buffering=0)

    renderer = None
    stage = None
    stream_started = False
    try:
        import ovrtx
        import ovstage

        bundle = json.loads(bundle_path.read_text())
        animation = np.load(bundle_path.parent / bundle["animation"])
        config = ovrtx.RendererConfig(
            log_level="warn",
            log_file_path=str(log_path),
            active_cuda_gpus="0",
            keep_system_alive=True,
        )
        print("[robot-rendering] creating ovrtx renderer", flush=True)
        renderer = ovrtx.Renderer(config)
        stage = ovstage.Stage(f"robot-rendering-{os.getpid()}")
        renderer.attach_ovstage(stage)
        print("[robot-rendering] populating ovstage from binary OpenUSD scene", flush=True)
        ordinal = 1
        ovstage.population.open_usd(
            stage,
            str(bundle_path.parent / bundle["scene"]),
            ordinal=ordinal,
            time_code=1.0 / bundle["fps"],
            domains=ovstage.PopulationDomain.RENDERING,
        )
        stage.advance_write_floor(ordinal, ovstage.Scope.ALL).wait()
        warmup_frames = 4 if bundle["render_mode"] == "PathTracing" else 2
        print(f"[robot-rendering] warming up {warmup_frames} frames", flush=True)
        first = None
        for _ in range(warmup_frames):
            first = _worker_frame(
                renderer,
                bundle,
                animation,
                0,
                stage=stage,
                ordinal=ordinal,
                update_stage=False,
            )
        assert first is not None
        print("[robot-rendering] first frame complete", flush=True)
        stream_aovs = bool(bundle.get("aovs"))
        binary.write(STREAM_MAGIC_AOV if stream_aovs else STREAM_MAGIC)
        stream_started = True
        index = 0
        while True:
            if index == 0 and first is not None:
                frame = first
                first = None
            else:
                ordinal += 1
                frame = _worker_frame(
                    renderer,
                    bundle,
                    animation,
                    index,
                    stage=stage,
                    ordinal=ordinal,
                )
            payload = _encode_aov_frame(frame) if stream_aovs else frame["rgb"].tobytes()
            binary.write(struct.pack("!I", len(payload)))
            binary.write(payload)
            index += 1
            if index >= bundle["frames"]:
                if not args.loop:
                    break
                index = 0
    except BrokenPipeError:
        return 0
    except BaseException as exc:
        traceback.print_exc(file=log_stream)
        message = f"{type(exc).__name__}: {exc}".encode(errors="replace")
        try:
            if stream_started:
                binary.write(struct.pack("!I", 0))
            else:
                binary.write(STREAM_ERROR)
            binary.write(struct.pack("!I", len(message)))
            binary.write(message)
        except BrokenPipeError:
            pass
        return 1
    finally:
        binary.close()
        if renderer is not None:
            if stage is not None:
                with contextlib.suppress(Exception):
                    renderer.detach_ovstage()
            renderer.destroy()
        if stage is not None:
            stage.destroy()
        log_stream.close()
        pid_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "__worker":
        raise SystemExit(_worker_main(sys.argv[2:]))
    raise SystemExit("Use `robotic-renderers`; __worker is reserved for the SSH runner.")
