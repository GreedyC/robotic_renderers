"""NVIDIA ovrtx backend rendered on an RTX Linux host and streamed to macOS."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import re
import shlex
import struct
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from pathlib import Path

import numpy as np

BACKEND_NAME = "ovrtx"
OVRTX_VERSION = "0.3.0.312915"
OVRTX_RELEASE = "v0.3.0"
OVRTX_ARCHIVE = "ovrtx@0.3.0.312915.cec773e1.manylinux_2_35_x86_64.zip"
OVRTX_SHA256 = "5569e44b18d2d39f23f374c9352dac9c87b8115892209c243c98732085f1d5f9"
OVRTX_URL = f"https://github.com/NVIDIA-Omniverse/ovrtx/releases/download/{OVRTX_RELEASE}/{OVRTX_ARCHIVE}"
DEFAULT_HOST = os.environ.get("ROBOT_RENDERING_OVRTX_HOST", "l4")
ROOT = Path(__file__).resolve().parent
REMOTE_ROOT = "code/robot_rendering_ovrtx"
STREAM_MAGIC = b"ROBOT_RENDERING_OVRTX_RGB_STREAM_V1\n"
STREAM_ERROR = b"ROBOT_RENDERING_OVRTX_ERROR_V1\n"
_SYNCED_HOSTS: set[str] = set()


def _run(command: list[str], **kwargs):
    return subprocess.run(command, check=True, **kwargs)


def _ssh_probe(host: str, command: str, timeout: float = 12.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host, command],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def backend_status(host: str = DEFAULT_HOST) -> dict[str, object]:
    error = None
    gpu = None
    installed = False
    try:
        probe = _ssh_probe(
            host,
            f'root="$HOME/{REMOTE_ROOT}"; '
            'gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1); '
            'test -n "$gpu" || exit 3; '
            'test -x "$root/.venv/bin/python" || exit 4; '
            'test -f "$root/vendor/runtime/bin/libovrtx-dynamic.so" || exit 5; '
            'cd "$root"; "$root/.venv/bin/python" -c "import ovrtx" || exit 6; '
            'printf "%s" "$gpu"',
        )
        installed = probe.returncode == 0
        gpu = probe.stdout.strip() or None
        if not installed:
            messages = {
                3: "No NVIDIA GPU was reported by nvidia-smi",
                4: "The uv-managed ovrtx environment is not installed",
                5: "The ovrtx native runtime is not installed",
                6: "The ovrtx Python package could not be imported",
            }
            error = messages.get(probe.returncode, probe.stderr.strip() or "Remote probe failed")
    except (OSError, subprocess.SubprocessError) as exc:
        error = str(exc)
    return {
        "backend": BACKEND_NAME,
        "available": installed,
        "version": OVRTX_VERSION if installed else None,
        "device": gpu,
        "execution": f"SSH ({host})",
        "host": host,
        "live": True,
        "record": True,
        "notes": (
            "ovrtx is Linux/Windows RTX-only. The renderer runs on the configured RTX host "
            "while "
            "OpenUSD scene generation, live display, and MP4 encoding remain on macOS."
        ),
        "error": error,
    }


def _remote_install_script(force: bool = False) -> str:
    reset = 'rm -rf "$runtime" "$root/.venv"; rm -f "$archive"' if force else ":"
    return f"""set -euo pipefail
root="$HOME/{REMOTE_ROOT}"
vendor="$root/vendor"
runtime="$vendor/runtime"
archive="$vendor/{OVRTX_ARCHIVE}"
mkdir -p "$vendor"
{reset}
if [ ! -f "$runtime/bin/libovrtx-dynamic.so" ]; then
  if [ ! -s "$archive" ]; then
    curl -L --fail --retry 5 --retry-delay 5 -o "$archive.part" {shlex.quote(OVRTX_URL)}
    mv "$archive.part" "$archive"
  fi
  printf '%s  %s\n' {shlex.quote(OVRTX_SHA256)} "$archive" | sha256sum -c -
  rm -rf "$vendor/runtime.tmp"
  mkdir -p "$vendor/runtime.tmp"
  if command -v unzip >/dev/null 2>&1; then
    unzip -q "$archive" -d "$vendor/runtime.tmp"
  else
    python3 - "$archive" "$vendor/runtime.tmp" <<'PY'
import os
import shutil
import stat
import sys
import zipfile
from pathlib import Path

archive_path = Path(sys.argv[1])
destination = Path(sys.argv[2]).resolve()
with zipfile.ZipFile(archive_path) as archive_file:
    for entry in archive_file.infolist():
        output = (destination / entry.filename).resolve()
        if destination not in output.parents and output != destination:
            raise RuntimeError(f"Unsafe ZIP entry: {{entry.filename}}")
        mode = entry.external_attr >> 16
        if entry.is_dir():
            output.mkdir(parents=True, exist_ok=True)
        elif stat.S_ISLNK(mode):
            output.parent.mkdir(parents=True, exist_ok=True)
            output.unlink(missing_ok=True)
            os.symlink(archive_file.read(entry).decode(), output)
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            with archive_file.open(entry) as source, output.open("wb") as target:
                shutil.copyfileobj(source, target)
            if mode:
                output.chmod(stat.S_IMODE(mode))
PY
  fi
  rm -rf "$runtime"
  mv "$vendor/runtime.tmp" "$runtime"
  rm -f "$archive"
fi
cp "$root/ovrtx_runtime_pyproject.toml" "$runtime/python/pyproject.toml"
if [ ! -x "$root/.venv/bin/python" ]; then
  "$HOME/.local/bin/uv" venv --python 3.12 "$root/.venv"
fi
"$HOME/.local/bin/uv" pip install --python "$root/.venv/bin/python" \
  'numpy==2.2.6' -e "$runtime/python"
cd "$root"
"$root/.venv/bin/python" -c 'import ovrtx; print("ovrtx import succeeded")'
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | sed -n '1p'
"""


def setup_remote(host: str = DEFAULT_HOST, force: bool = False) -> dict[str, object]:
    if host in _SYNCED_HOSTS and not force:
        return {"backend": BACKEND_NAME, "host": host, "synced": True, "cached": True}
    uv_probe = _ssh_probe(host, 'test -x "$HOME/.local/bin/uv"')
    if uv_probe.returncode != 0:
        raise RuntimeError(
            f"uv is not installed on {host}. Install it with: "
            f"ssh {host} 'curl -LsSf https://astral.sh/uv/install.sh | sh'"
        )
    _run(["ssh", host, f'mkdir -p "$HOME/{REMOTE_ROOT}"'])
    _run(
        [
            "rsync",
            "-az",
            str(ROOT / "render_ovrtx.py"),
            str(ROOT / "ovrtx_runtime_pyproject.toml"),
            f"{host}:~/{REMOTE_ROOT}/",
        ]
    )
    if not force:
        ready = _ssh_probe(
            host,
            f'root="$HOME/{REMOTE_ROOT}"; '
            'test -x "$root/.venv/bin/python" && '
            'test -f "$root/vendor/runtime/bin/libovrtx-dynamic.so" && '
            'cd "$root" && "$root/.venv/bin/python" -c "import ovrtx"',
            timeout=30.0,
        )
        if ready.returncode == 0:
            _SYNCED_HOSTS.add(host)
            return {
                "backend": BACKEND_NAME,
                "host": host,
                "synced": True,
                "cached": True,
                "version": OVRTX_VERSION,
            }
    completed = subprocess.run(
        ["ssh", host, "bash -s"],
        input=_remote_install_script(force),
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        details = (completed.stdout + "\n" + completed.stderr)[-12000:]
        raise RuntimeError(f"ovrtx setup failed on {host}:\n{details}")
    _SYNCED_HOSTS.add(host)
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


def _surface_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(vertices, dtype=np.float64)
    triangles = np.asarray(vertices, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    face_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.maximum(lengths, 1e-12)
    return normals.astype(np.float32)


def _box_mesh(size: list[float]) -> tuple[np.ndarray, np.ndarray]:
    x, y, z = map(float, size)
    vertices = np.array(
        [
            [-x, -y, -z],
            [x, -y, -z],
            [x, y, -z],
            [-x, y, -z],
            [-x, -y, z],
            [x, -y, z],
            [x, y, z],
            [-x, y, z],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=np.int32,
    )
    return vertices, faces


def _sphere_mesh(
    radii: tuple[float, float, float], segments: int = 96, rings: int = 64
) -> tuple[np.ndarray, np.ndarray]:
    vertices = []
    for ring in range(rings + 1):
        latitude = -math.pi / 2.0 + math.pi * ring / rings
        radial = math.cos(latitude)
        for segment in range(segments):
            longitude = 2.0 * math.pi * segment / segments
            vertices.append(
                (
                    radii[0] * radial * math.cos(longitude),
                    radii[1] * radial * math.sin(longitude),
                    radii[2] * math.sin(latitude),
                )
            )
    faces = []
    for ring in range(rings):
        for segment in range(segments):
            next_segment = (segment + 1) % segments
            lower = ring * segments + segment
            lower_next = ring * segments + next_segment
            upper = (ring + 1) * segments + segment
            upper_next = (ring + 1) * segments + next_segment
            if ring > 0:
                faces.append((lower, lower_next, upper_next))
            if ring < rings - 1:
                faces.append((lower, upper_next, upper))
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def _cylinder_mesh(radius: float, half_height: float, segments: int = 96):
    vertices = []
    for z in (-half_height, half_height):
        for segment in range(segments):
            angle = 2.0 * math.pi * segment / segments
            vertices.append((radius * math.cos(angle), radius * math.sin(angle), z))
    bottom_center = len(vertices)
    vertices.append((0.0, 0.0, -half_height))
    top_center = len(vertices)
    vertices.append((0.0, 0.0, half_height))
    faces = []
    for segment in range(segments):
        next_segment = (segment + 1) % segments
        faces.extend(
            [
                (segment, next_segment, segments + next_segment),
                (segment, segments + next_segment, segments + segment),
                (bottom_center, next_segment, segment),
                (top_center, segments + segment, segments + next_segment),
            ]
        )
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32)


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
    geom_type = description["type"]
    size = description["size"]
    if geom_type == "box":
        return _box_mesh(size)
    if geom_type == "sphere":
        return _sphere_mesh((size[0], size[0], size[0]))
    if geom_type == "ellipsoid":
        return _sphere_mesh((size[0], size[1], size[2]))
    if geom_type == "capsule":
        return _sphere_mesh((size[0], size[0], size[0] + size[1]))
    if geom_type == "cylinder":
        return _cylinder_mesh(size[0], size[1])
    if geom_type == "mesh":
        mesh = meshes[str(description["mesh_id"])]
        return (
            np.asarray(geometry[mesh["vertices"]], dtype=np.float32),
            np.asarray(geometry[mesh["faces"]], dtype=np.int32),
        )
    raise ValueError(f"Unsupported ovrtx geom type: {geom_type}")


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


def _write_gradient_sky(destination: Path) -> Path:
    from PIL import Image

    width, height = 512, 256
    latitude = np.pi / 2.0 - np.pi * (np.arange(height) + 0.5) / height
    factor = np.maximum(0.0, np.sin(latitude))[:, np.newaxis]
    horizon = np.array([0.003, 0.004, 0.009])
    zenith = np.array([0.48, 0.60, 0.83])
    linear = horizon + factor * (zenith - horizon)
    linear = np.broadcast_to(linear[:, np.newaxis, :], (height, width, 3))
    srgb = np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * np.power(linear, 1.0 / 2.4) - 0.055,
    )
    output = destination / "studio_gradient.png"
    Image.fromarray(np.rint(np.clip(srgb, 0.0, 1.0) * 255.0).astype(np.uint8)).save(output)
    return output


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


def prepare_ovrtx_bundle(
    destination: str | Path,
    times: list[float],
    width: int,
    height: int,
    fps: int,
    samples: int = 256,
    render_mode: str = "PathTracing",
) -> Path:
    from render_blender import prepare_blender_bundle

    destination = Path(destination).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = prepare_blender_bundle(destination, times, width, height, fps)
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
            '''#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
'''
        )
        stream.write(
            f"    startTimeCode = 1\n"
            f"    endTimeCode = {max(1, len(manifest['frames']))}\n"
            f"    timeCodesPerSecond = {fps}\n"
        )
        stream.write(
            ''')

def Xform "World" {
    def Scope "Materials" {
'''
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
            stream.write(
                f'''        def Xform "{identifier}" {{
            matrix4d xformOp:transform.timeSamples = {transform_samples}
            uniform token[] xformOpOrder = ["xformOp:transform"]
'''
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
                            and manifest["meshes"][str(geom["mesh_id"])]["name"]
                            == "mesh_gem"
                        )
                    ),
                )
            stream.write("        }\n")
        fovy = float(manifest["camera"]["fovy"])
        vertical_aperture = 24.0
        focal_length = 0.5 * vertical_aperture / math.tan(math.radians(fovy) / 2.0)
        horizontal_aperture = vertical_aperture * width / height
        key_transform = _format_matrix(
            _matrix([-1.5, -1.0, 3.0], _rotation_toward([0.45, 0.2, -1.0]))
        )
        rim_transform = _format_matrix(
            _matrix([-0.65, 0.9, 1.8], _rotation_toward([0.5, -0.6, -1.0]))
        )
        camera_samples = _time_sampled_transforms(
            manifest["frames"], lambda frame: frame["camera"]
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
    def DistantLight "SunKey" {{
        color3f inputs:color = (0.95, 0.91, 0.84)
        float inputs:intensity = 588
        float inputs:angle = 3
        matrix4d xformOp:transform = {key_transform}
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }}
    def SphereLight "SoftFill" {{
        color3f inputs:color = (0.34, 0.47, 0.7)
        float inputs:intensity = 525
        float inputs:radius = 0.25
        double3 xformOp:translate = (0.2, -1.5, 1.45)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }}
    def DiskLight "WarmRim" (
        prepend apiSchemas = ["ShapingAPI"]
    ) {{
        color3f inputs:color = (0.8, 0.46, 0.25)
        float inputs:intensity = 294
        float inputs:radius = 0.08
        float inputs:shaping:cone:angle = 27.5
        float inputs:shaping:cone:softness = 0.35
        float inputs:shaping:focus = 2
        matrix4d xformOp:transform = {rim_transform}
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }}
    def DomeLight "StudioDome" (
        prepend apiSchemas = ["ShapingAPI"]
    ) {{
        float inputs:intensity = 59
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
        token omni:rtx:rendermode = "{render_mode}"
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
}}
'''
        )

    from pxr import Sdf

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
        "render_mode": render_mode,
        "animated_in_usd": True,
    }
    bundle_path = destination / "ovrtx_bundle.json"
    bundle_path.write_text(json.dumps(bundle, separators=(",", ":")))
    return bundle_path


def _upload_job(host: str, local_job: Path, job_id: str) -> str:
    remote_job = f".jobs/{job_id}"
    _run(["ssh", host, f'mkdir -p "$HOME/{REMOTE_ROOT}/{remote_job}"'])
    _run(["rsync", "-az", f"{local_job}/", f"{host}:~/{REMOTE_ROOT}/{remote_job}/"])
    return remote_job


def _cleanup_remote_job(host: str, job_id: str) -> None:
    subprocess.run(
        ["ssh", host, f'rm -rf "$HOME/{REMOTE_ROOT}/.jobs/{job_id}"'],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _remote_worker_command(remote_job: str, loop: bool) -> str:
    arguments = [
        ".venv/bin/python",
        "render_ovrtx.py",
        "__worker",
        "--bundle",
        f"{remote_job}/ovrtx_bundle.json",
    ]
    if loop:
        arguments.append("--loop")
    return f'cd "$HOME/{REMOTE_ROOT}" && exec setsid --wait {shlex.join(arguments)}'


def _terminate_remote_worker(host: str, remote_job: str) -> None:
    pid_file = f'$HOME/{REMOTE_ROOT}/{remote_job}/worker.pid'
    command = (
        f'pid_file="{pid_file}"; '
        'test -r "$pid_file" || exit 0; '
        'pid=$(cat "$pid_file"); '
        'case "$pid" in (*[!0-9]*|"") exit 0;; esac; '
        'test -r "/proc/$pid/cmdline" || exit 0; '
        'command=$(tr "\\0" " " < "/proc/$pid/cmdline"); '
        f'case "$command" in (*render_ovrtx.py*{shlex.quote(remote_job)}*) ;; (*) exit 0;; esac; '
        'kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true; '
        'for delay in 1 2 3 4 5; do kill -0 "$pid" 2>/dev/null || exit 0; sleep 0.2; done; '
        'kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true'
    )
    with contextlib.suppress(OSError, subprocess.SubprocessError):
        subprocess.run(
            ["ssh", "-o", "BatchMode=yes", host, command],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )


def _read_exact(stream, byte_count: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < byte_count:
        chunk = stream.read(byte_count - len(chunks))
        if not chunk:
            raise EOFError("Remote ovrtx RGB stream closed unexpectedly")
        chunks.extend(chunk)
    return bytes(chunks)


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


def _remote_stream_frames(host: str, remote_job: str, width: int, height: int, loop: bool):
    process = subprocess.Popen(
        ["ssh", host, _remote_worker_command(remote_job, loop)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    assert process.stdout is not None
    assert process.stderr is not None
    completed_normally = False
    try:
        handshake = process.stdout.readline()
        if handshake == STREAM_ERROR:
            message_length = struct.unpack("!I", _read_exact(process.stdout, 4))[0]
            message = _read_exact(process.stdout, message_length).decode(errors="replace")
            raise RuntimeError(f"ovrtx worker failed before rendering:\n{message}")
        if handshake != STREAM_MAGIC:
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
                length_bytes += _read_exact(process.stdout, 4 - len(length_bytes))
            frame_length = struct.unpack("!I", length_bytes)[0]
            if frame_length == 0:
                message_length = struct.unpack("!I", _read_exact(process.stdout, 4))[0]
                message = _read_exact(process.stdout, message_length).decode(errors="replace")
                raise RuntimeError(f"ovrtx worker failed while rendering:\n{message}")
            if frame_length != expected:
                raise RuntimeError(f"ovrtx sent {frame_length} bytes; expected {expected}")
            frame = np.frombuffer(_read_exact(process.stdout, frame_length), dtype=np.uint8)
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
):
    setup_remote(host)
    job_id = uuid.uuid4().hex
    (ROOT / "build").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ovrtx-", dir=ROOT / "build") as temporary:
        job = Path(temporary)
        prepare_ovrtx_bundle(job, times, width, height, fps, samples, render_mode)
        remote_job = _upload_job(host, job, job_id)
        try:
            yield from _remote_stream_frames(host, remote_job, width, height, loop)
        finally:
            _cleanup_remote_job(host, job_id)


def render_frame(
    width: int = 1280,
    height: int = 720,
    time_s: float = 1.0,
    samples: int = 256,
    host: str = DEFAULT_HOST,
) -> np.ndarray:
    source = _frame_source([time_s], width, height, 1, samples, host, "PathTracing")
    try:
        return next(source)
    finally:
        source.close()


def render_still(
    output_path: str | Path,
    width: int = 1280,
    height: int = 720,
    time_s: float = 1.0,
    samples: int = 256,
    host: str = DEFAULT_HOST,
) -> dict[str, object]:
    from render_common import image_digest, write_png

    started = time.perf_counter()
    image = render_frame(width, height, time_s, samples, host)
    output = write_png(image, output_path)
    return {
        "backend": BACKEND_NAME,
        "output": str(output),
        "seconds": time.perf_counter() - started,
        "sha256": image_digest(image),
        "width": width,
        "height": height,
        "samples": samples,
        "device": f"NVIDIA RTX on {host}",
    }


def render_live(
    width: int = 960,
    height: int = 540,
    fps: int = 30,
    duration: float | None = None,
    samples: int = 64,
    host: str = DEFAULT_HOST,
) -> dict[str, object]:
    from render_common import display_frames
    from scene import CAMERA_ORBIT_PERIOD, trajectory_times

    source_duration = duration if duration is not None else CAMERA_ORBIT_PERIOD
    times = list(trajectory_times(source_duration, fps))
    result = display_frames(
        _frame_source(
            times,
            width,
            height,
            fps,
            samples,
            host,
            "Real-Time Path-Tracing",
            loop=duration is None,
        ),
        width,
        height,
        f"NVIDIA ovrtx ({host})",
        target_fps=fps,
    )
    return {"backend": BACKEND_NAME, "host": host, **result}


def record_video(
    output_path: str | Path,
    duration: float = 5.0,
    fps: int = 30,
    width: int = 1280,
    height: int = 720,
    samples: int = 256,
    host: str = DEFAULT_HOST,
) -> dict[str, object]:
    from render_common import write_video
    from scene import trajectory_times

    started = time.perf_counter()
    times = list(trajectory_times(duration, fps))
    output, frames, encode_seconds = write_video(
        _frame_source(times, width, height, fps, samples, host, "PathTracing"),
        output_path,
        fps,
    )
    seconds = time.perf_counter() - started
    return {
        "backend": BACKEND_NAME,
        "output": str(output),
        "frames": frames,
        "seconds": seconds,
        "encode_seconds": encode_seconds,
        "fps": frames / max(seconds, 1e-9),
        "width": width,
        "height": height,
        "samples": samples,
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
) -> dict[str, object]:
    from render_common import image_digest

    total = warmup + frame_count
    times = [index / fps for index in range(total)]
    started = time.perf_counter()
    frames = _frame_source(times, width, height, fps, samples, host, "PathTracing")
    last = None
    for index, frame in enumerate(frames):
        if index >= warmup:
            last = frame
    seconds = time.perf_counter() - started
    return {
        "backend": BACKEND_NAME,
        "device": f"NVIDIA RTX on {host}",
        "width": width,
        "height": height,
        "samples": samples,
        "frames": frame_count,
        "roundtrip_seconds": seconds,
        "roundtrip_fps": frame_count / max(seconds, 1e-9),
        "last_frame_sha256": image_digest(last),
    }


def _worker_frame(renderer, bundle, animation, index: int) -> np.ndarray:
    import ovrtx

    if bundle.get("animated_in_usd"):
        # ovrtx does not advance authored USD time samples from step() alone.
        # Its explicit time API accepts seconds (not USD timecodes), while this
        # stage stores frames at timecodes 1..N with timeCodesPerSecond=fps.
        renderer.update_from_usd_time((index + 1) / bundle["fps"])
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
    products = renderer.step(
        render_products={bundle["render_product"]},
        delta_time=1.0 / bundle["fps"],
    )
    product = products[bundle["render_product"]]
    if not product.frames:
        raise RuntimeError("ovrtx returned no camera frame")
    if index == 0:
        print("[robot-rendering] mapping first LdrColor frame", flush=True)
    mapping = product.frames[0].render_vars["LdrColor"].map(device=ovrtx.Device.CPU)
    try:
        view = np.from_dlpack(mapping)
        expected_shape = (bundle["height"], bundle["width"], 4)
        if view.shape != expected_shape:
            raise RuntimeError(
                f"ovrtx LdrColor has shape {view.shape}; expected {expected_shape}"
            )
        image = np.ascontiguousarray(view[:, :, :3]).copy()
        del view
        return image
    finally:
        mapping.unmap()


def _worker_main(arguments: list[str]) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args(arguments)

    binary_fd = os.dup(sys.stdout.fileno())
    bundle_path = Path(args.bundle).resolve()
    pid_path = bundle_path.parent / "worker.pid"
    pid_path.write_text(str(os.getpid()))
    log_path = bundle_path.parent / "ovrtx_worker.log"
    log_stream = log_path.open("a", buffering=1)
    os.dup2(log_stream.fileno(), sys.stdout.fileno())
    os.dup2(log_stream.fileno(), sys.stderr.fileno())
    binary = os.fdopen(binary_fd, "wb", buffering=0)

    renderer = None
    stream_started = False
    try:
        import ovrtx

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
        print("[robot-rendering] opening binary OpenUSD scene", flush=True)
        renderer.open_usd(str(bundle_path.parent / bundle["scene"]))
        warmup_frames = 4 if bundle["render_mode"] == "PathTracing" else 2
        print(f"[robot-rendering] warming up {warmup_frames} frames", flush=True)
        first = None
        for _ in range(warmup_frames):
            renderer.reset(time=0.0)
            first = _worker_frame(renderer, bundle, animation, 0)
        assert first is not None
        print("[robot-rendering] first frame complete", flush=True)
        binary.write(STREAM_MAGIC)
        stream_started = True
        index = 0
        while True:
            frame = first if index == 0 else _worker_frame(renderer, bundle, animation, index)
            payload = frame.tobytes()
            binary.write(struct.pack("!I", len(payload)))
            binary.write(payload)
            index += 1
            if index >= bundle["frames"]:
                if not args.loop:
                    break
                renderer.reset(time=0.0)
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
            del renderer
        log_stream.close()
        pid_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "__worker":
        raise SystemExit(_worker_main(sys.argv[2:]))
    raise SystemExit("Use main.py; __worker is reserved for the SSH runner.")
