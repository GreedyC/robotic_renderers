"""MuJoCo Warp ray-renderer backend for macOS CPU, with optional SSH CUDA."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

from robotic_renderers.defaults import (
    DEFAULT_HEIGHT,
    DEFAULT_RECORDING_DURATION,
    DEFAULT_RENDER_HOST,
    DEFAULT_VIDEO_FPS,
    DEFAULT_WIDTH,
)
from robotic_renderers.media import (
    atomic_output_path,
    display_frames,
    image_digest,
    resize_rgb,
    write_png,
    write_video,
)
from robotic_renderers.processes import terminate_and_reap
from robotic_renderers.remote import (
    read_exact,
    resolve_runtime_project,
    validate_ssh_host,
)
from robotic_renderers.scene import create_model_and_data, set_scene_time, trajectory_times

BACKEND_NAME = "warp"
WARP_EXPOSURE_SCALE = 0.42
DEFAULT_HOST = os.environ.get("ROBOT_RENDERING_WARP_HOST", DEFAULT_RENDER_HOST)
REMOTE_ROOT = ".cache/robot_rendering"
STREAM_MAGIC = b"ROBOT_RENDERING_RGB_STREAM_V1\n"
RESULT_MAGIC = "ROBOT_RENDERING_RESULT="
SSH_OPTIONS = (
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=8",
    "-o",
    "ServerAliveInterval=15",
    "-o",
    "ServerAliveCountMax=3",
)
REMOTE_BUNDLE_FILES = (
    ".python-version",
    "LICENSE",
    "THIRD_PARTY.md",
)
REMOTE_RUNTIME_DIRECTORY = Path("runtime/warp")
REMOTE_BUNDLE_DIRECTORIES = (REMOTE_RUNTIME_DIRECTORY, Path("src"), Path("third_party"))
REMOTE_BUNDLE_EXCLUDED_PARTS = {
    ".cache",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "checkpoints",
    "dist",
    "models",
    "outputs",
    "videos",
    "weights",
}
REMOTE_BUNDLE_SENSITIVE_NAMES = {
    ".env",
    "credentials.json",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "id_rsa",
}
REMOTE_BUNDLE_SENSITIVE_SUFFIXES = (".key", ".p12", ".pem", ".pfx", ".pyc", ".pyo")
_SYNCED_HOSTS: dict[str, str] = {}
PACKAGE_SOURCE = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_SOURCE.parents[1]


def _find_repository_root() -> Path | None:
    candidates = (Path.cwd(), PROJECT_ROOT)
    for candidate in candidates:
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "src" / "robotic_renderers" / "__init__.py"
        ).is_file():
            return candidate.resolve()
    return None


def _repository_root() -> Path:
    root = _find_repository_root()
    if root is not None:
        return root
    raise RuntimeError(
        "Remote Warp setup found neither a robotic-renderers source checkout "
        "nor packaged worker/runtime data. Reinstall the package."
    )


def _run(command: list[str], **kwargs):
    return subprocess.run(command, check=True, **kwargs)


def _ssh_command(host: str, *arguments: str) -> list[str]:
    return ["ssh", *SSH_OPTIONS, validate_ssh_host(host), *arguments]


def _rsync_ssh_transport() -> str:
    return shlex.join(["ssh", *SSH_OPTIONS])


def _bundle_path_is_safe(relative_path: Path) -> bool:
    if relative_path.is_absolute() or ".." in relative_path.parts:
        return False
    if any(part in REMOTE_BUNDLE_EXCLUDED_PARTS for part in relative_path.parts):
        return False
    if any(part == ".env" or part.startswith(".env.") for part in relative_path.parts):
        return False
    name = relative_path.name
    if name in REMOTE_BUNDLE_SENSITIVE_NAMES:
        return False
    return not name.lower().endswith(REMOTE_BUNDLE_SENSITIVE_SUFFIXES)


def _source_bundle_sources(root: Path) -> dict[Path, Path]:
    root = root.resolve()
    required = (
        REMOTE_RUNTIME_DIRECTORY / "pyproject.toml",
        REMOTE_RUNTIME_DIRECTORY / "uv.lock",
        Path("src/robotic_renderers/backends/mujoco_warp.py"),
        Path("src/robotic_renderers/scene.py"),
    )
    missing = [str(path) for path in required if not (root / path).is_file()]
    if missing:
        raise FileNotFoundError(f"Remote Warp bundle is missing required files: {missing}")

    sources: dict[Path, Path] = {}
    for name in REMOTE_BUNDLE_FILES:
        relative_path = Path(name)
        if (root / relative_path).is_file() and _bundle_path_is_safe(relative_path):
            sources[relative_path] = root / relative_path
    for relative_directory in REMOTE_BUNDLE_DIRECTORIES:
        directory = root / relative_directory
        if not directory.is_dir():
            continue
        for source in directory.rglob("*"):
            if source.is_file() or source.is_symlink():
                relative_path = source.relative_to(root)
                if _bundle_path_is_safe(relative_path):
                    sources[relative_path] = source
    return sources


def _installed_bundle_sources(
    package_source: Path = PACKAGE_SOURCE,
    runtime_project: Path | None = None,
) -> dict[Path, Path]:
    """Map installed package files back into the remote source layout."""
    package_source = package_source.resolve()
    if runtime_project is None:
        runtime_project = resolve_runtime_project(
            package_source,
            package_source.parents[1],
            BACKEND_NAME,
        )
    runtime_project = runtime_project.resolve()
    required = (
        package_source / "__init__.py",
        package_source / "backends" / "mujoco_warp.py",
        package_source / "scene.py",
        package_source / "data" / "franka_fr3" / "fr3.xml",
        runtime_project / "pyproject.toml",
        runtime_project / "uv.lock",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Installed remote Warp bundle is missing required files: {missing}"
        )

    sources: dict[Path, Path] = {}
    for source in package_source.rglob("*"):
        if not (source.is_file() or source.is_symlink()):
            continue
        package_relative = source.relative_to(package_source)
        if package_relative.parts[:2] == ("data", "runtime"):
            continue
        relative_path = Path("src/robotic_renderers") / package_relative
        if _bundle_path_is_safe(relative_path):
            sources[relative_path] = source
    for source in runtime_project.rglob("*"):
        if source.is_file() or source.is_symlink():
            relative_path = REMOTE_RUNTIME_DIRECTORY / source.relative_to(runtime_project)
            if _bundle_path_is_safe(relative_path):
                sources[relative_path] = source
    return sources


def _remote_bundle_sources(root: Path | None = None) -> dict[Path, Path]:
    if root is not None:
        return _source_bundle_sources(root)
    checkout = _find_repository_root()
    if checkout is not None:
        return _source_bundle_sources(checkout)
    return _installed_bundle_sources()


def _remote_bundle_entries(root: Path | None = None) -> tuple[Path, ...]:
    """Return the explicit, secret-resistant file manifest sent to the Warp host."""
    return tuple(sorted(_remote_bundle_sources(root), key=lambda path: path.as_posix()))


def _remote_bundle_fingerprint(root: Path | None = None) -> str:
    digest = hashlib.sha256()
    sources = _remote_bundle_sources(root)
    for relative_path in sorted(sources, key=lambda path: path.as_posix()):
        source = sources[relative_path]
        digest.update(relative_path.as_posix().encode())
        if source.is_symlink():
            digest.update(os.readlink(source).encode())
            continue
        with source.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _stage_remote_bundle(
    destination: Path,
    root: Path | None = None,
) -> tuple[Path, ...]:
    """Copy only the manifest into an isolated rsync source directory."""
    destination.mkdir(parents=True, exist_ok=True)
    sources = _remote_bundle_sources(root)
    entries = tuple(sorted(sources, key=lambda path: path.as_posix()))
    for relative_path in entries:
        source = sources[relative_path]
        target = destination / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_symlink():
            link_target = Path(os.readlink(source))
            if link_target.is_absolute() or ".." in link_target.parts:
                raise ValueError(f"Unsafe symlink in remote Warp bundle: {relative_path}")
            target.symlink_to(link_target)
        else:
            shutil.copy2(source, target)
    return entries


def _remote_shell_command(arguments: list[str]) -> str:
    remote_root = f"$HOME/{REMOTE_ROOT}"
    worker = shlex.join(
        [
            f"{remote_root}/.venv/bin/python",
            "-m",
            "robotic_renderers.backends.mujoco_warp",
            "__worker",
            *arguments,
        ]
    )
    for token in (f"{remote_root}/.venv/bin/python",):
        worker = worker.replace(shlex.quote(token), f'"{token}"')
    return (
        f'root="{remote_root}"; project="$root/source"; cd "$root" && '
        f'PYTHONPATH="$project/src" {worker}'
    )


def _remote_sync_script(incoming: str) -> str:
    return f"""set -euo pipefail
umask 077
root="$HOME/{REMOTE_ROOT}"
incoming="$root/{incoming}"
runtime="$incoming/{REMOTE_RUNTIME_DIRECTORY.as_posix()}"
test -f "$runtime/pyproject.toml"
test -f "$runtime/uv.lock"
exec 9>"$root/sync.lock"
chmod 600 "$root/sync.lock"
flock -w 120 9
UV_PROJECT_ENVIRONMENT="$root/.venv" "$HOME/.local/bin/uv" sync \
  --directory "$runtime" --python 3.12 --frozen --no-dev
PYTHONPATH="$incoming/src" "$root/.venv/bin/python" -c \
  'import imageio, mujoco, mujoco_warp, numpy, PIL, warp; from pxr import Sdf'
old="$root/source.old-$$"
if [ -e "$root/source" ]; then mv "$root/source" "$old"; fi
mv "$incoming" "$root/source"
rm -rf "$old"
"""


def backend_status(host: str = DEFAULT_HOST) -> dict[str, object]:
    local_device = f"CPU ({os.uname().machine})"
    return {
        "backend": BACKEND_NAME,
        "available": True,
        "version": getattr(mjw, "__version__", "3.11.0"),
        "device": local_device,
        "execution": "local",
        "live": True,
        "record": True,
        "notes": (
            "The Warp ray renderer runs locally on macOS through Warp's CPU device. "
            f"Remote CUDA on {host} remains optional via --remote-warp."
        ),
    }


def setup_remote(host: str = DEFAULT_HOST, force: bool = False) -> dict[str, object]:
    host = validate_ssh_host(host)
    fingerprint = _remote_bundle_fingerprint()
    if _SYNCED_HOSTS.get(host) == fingerprint and not force:
        return {"host": host, "synced": True, "cached": True}

    uv_probe = subprocess.run(
        _ssh_command(host, 'test -x "$HOME/.local/bin/uv"'),
        capture_output=True,
        text=True,
        timeout=15.0,
    )
    if uv_probe.returncode != 0:
        raise RuntimeError(
            f"uv is not installed on {host}. Install it with: "
            f"ssh {host} 'curl -LsSf https://astral.sh/uv/install.sh | sh'"
        )

    upload_id = uuid.uuid4().hex
    incoming = f"source.upload-{upload_id}"
    remote_incoming = f"{host}:~/{REMOTE_ROOT}/{incoming}/"
    _run(
        _ssh_command(
            host,
            f'umask 077; install -d -m 700 "$HOME/{REMOTE_ROOT}" '
            f'"$HOME/{REMOTE_ROOT}/.jobs" "$HOME/{REMOTE_ROOT}/{incoming}"; '
            f'chmod 700 "$HOME/{REMOTE_ROOT}" "$HOME/{REMOTE_ROOT}/.jobs" '
            f'"$HOME/{REMOTE_ROOT}/{incoming}"',
        )
    )
    try:
        with tempfile.TemporaryDirectory(prefix="robot-rendering-warp-bundle-") as temporary:
            bundle = Path(temporary) / "source"
            _stage_remote_bundle(bundle)
            _run(
                [
                    "rsync",
                    "-az",
                    "-e",
                    _rsync_ssh_transport(),
                    f"{bundle}/",
                    remote_incoming,
                ]
            )
        _run(
            _ssh_command(host, "bash -s"),
            input=_remote_sync_script(incoming),
            text=True,
        )
    finally:
        subprocess.run(
            _ssh_command(host, f'rm -rf "$HOME/{REMOTE_ROOT}/{incoming}"'),
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    _SYNCED_HOSTS[host] = fingerprint
    return {"host": host, "synced": True, "cached": False}


def create_renderer(
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    supersample: int = 2,
    device: str | None = None,
) -> dict[str, object]:
    if supersample < 1:
        raise ValueError("supersample must be at least 1")
    wp.init()
    if device is None:
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
    wp.set_device(device)

    render_width = width * supersample
    render_height = height * supersample
    model, host_data = create_model_and_data(width=render_width, height=render_height)
    warp_model = mjw.put_model(model)
    floor_material_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "floor_material")
    if floor_material_id >= 0:
        texture_repeats = warp_model.mat_texrepeat.numpy()
        # Warp's procedural checker sampler produces two checks per texture period,
        # while MuJoCo OpenGL treats a period as one floor tile.
        texture_repeats[:, floor_material_id] *= 0.5
        warp_model.mat_texrepeat.assign(texture_repeats)
    warp_data = mjw.put_data(model, host_data, nworld=1)
    render_context = mjw.create_render_context(
        model,
        nworld=1,
        cam_res=(render_width, render_height),
        render_rgb=True,
        render_depth=False,
        render_seg=False,
        use_textures=True,
        use_shadows=True,
        use_ambient_lighting=True,
        enabled_geom_groups=[0, 1, 2],
        render_skybox=True,
        enable_backface_culling=True,
        enable_specular=True,
        enable_emission=True,
        enable_per_light_ambient=True,
    )
    rgb = wp.empty((1, render_height, render_width), dtype=wp.vec3)
    return {
        "model": model,
        "host_data": host_data,
        "warp_model": warp_model,
        "warp_data": warp_data,
        "render_context": render_context,
        "rgb": rgb,
        "device": device,
        "width": width,
        "height": height,
        "render_width": render_width,
        "render_height": render_height,
        "supersample": supersample,
    }


def _copy_kinematics_to_warp(context: dict[str, object]) -> None:
    model = context["model"]
    host_data = context["host_data"]
    warp_data = context["warp_data"]
    warp_data.geom_xpos.assign(np.asarray(host_data.geom_xpos, dtype=np.float32)[None])
    warp_data.geom_xmat.assign(
        np.asarray(host_data.geom_xmat, dtype=np.float32).reshape(1, model.ngeom, 3, 3)
    )
    warp_data.cam_xpos.assign(np.asarray(host_data.cam_xpos, dtype=np.float32)[None])
    warp_data.cam_xmat.assign(
        np.asarray(host_data.cam_xmat, dtype=np.float32).reshape(1, model.ncam, 3, 3)
    )


def render_context_frame(context: dict[str, object], time_s: float = 0.0) -> np.ndarray:
    set_scene_time(context["model"], context["host_data"], time_s)
    _copy_kinematics_to_warp(context)
    mjw.refit_bvh(context["warp_model"], context["warp_data"], context["render_context"])
    mjw.render(context["warp_model"], context["warp_data"], context["render_context"])
    mjw.get_rgb(context["render_context"], 0, context["rgb"])
    float_image = np.clip(
        context["rgb"].numpy()[0] * WARP_EXPOSURE_SCALE,
        0.0,
        1.0,
    )
    image = np.rint(float_image * 255.0).astype(np.uint8)
    return resize_rgb(image, context["width"], context["height"])


def render_frame_local(
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    time_s: float = 1.0,
    supersample: int = 2,
    device: str | None = None,
) -> np.ndarray:
    context = create_renderer(width, height, supersample=supersample, device=device)
    return render_context_frame(context, time_s=time_s)


def render_frame(
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    time_s: float = 1.0,
    supersample: int = 2,
    host: str = DEFAULT_HOST,
    local: bool = True,
) -> np.ndarray:
    if local:
        return render_frame_local(width, height, time_s, supersample)
    from PIL import Image

    with tempfile.TemporaryDirectory(prefix="mujoco-warp-frame-") as temporary:
        output = Path(temporary) / "frame.png"
        render_still(output, width, height, time_s, supersample, host, local=False)
        return np.asarray(Image.open(output).convert("RGB"))


def render_still_local(
    output_path: str | Path,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    time_s: float = 1.0,
    supersample: int = 2,
    device: str | None = None,
) -> dict[str, object]:
    started = time.perf_counter()
    image = render_frame_local(width, height, time_s, supersample, device)
    output = write_png(image, output_path)
    return {
        "backend": BACKEND_NAME,
        "output": str(output),
        "seconds": time.perf_counter() - started,
        "sha256": image_digest(image),
        "width": width,
        "height": height,
        "device": device or ("cuda:0" if wp.is_cuda_available() else "cpu"),
        "supersample": supersample,
    }


def _finite_frames(
    context: dict[str, object],
    duration: float,
    fps: int,
    timing: dict[str, float] | None = None,
):
    for time_s in trajectory_times(duration, fps):
        started = time.perf_counter()
        frame = render_context_frame(context, time_s)
        if timing is not None:
            timing["render_seconds"] = timing.get("render_seconds", 0.0) + (
                time.perf_counter() - started
            )
        yield frame


def record_video_local(
    output_path: str | Path,
    duration: float = DEFAULT_RECORDING_DURATION,
    fps: int = DEFAULT_VIDEO_FPS,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    supersample: int = 2,
    device: str | None = None,
) -> dict[str, object]:
    context = create_renderer(width, height, supersample=supersample, device=device)
    timing: dict[str, float] = {}
    output, frame_count, total_seconds = write_video(
        _finite_frames(context, duration=duration, fps=fps, timing=timing),
        output_path,
        fps=fps,
    )
    render_seconds = timing.get("render_seconds", 0.0)
    encode_seconds = max(0.0, total_seconds - render_seconds)
    return {
        "backend": BACKEND_NAME,
        "output": str(output),
        "frames": frame_count,
        "seconds": total_seconds,
        "render_seconds": render_seconds,
        "encode_seconds": encode_seconds,
        "video_fps": fps,
        "render_fps": frame_count / max(render_seconds, 1e-9),
        "width": width,
        "height": height,
        "device": context["device"],
        "supersample": supersample,
    }


def benchmark_local(
    frame_count: int = 30,
    width: int = 640,
    height: int = 360,
    fps: int = 30,
    warmup: int = 3,
    supersample: int = 1,
    device: str | None = None,
) -> dict[str, object]:
    init_started = time.perf_counter()
    context = create_renderer(width, height, supersample=supersample, device=device)
    init_seconds = time.perf_counter() - init_started
    for index in range(warmup):
        render_context_frame(context, index / fps)
    started = time.perf_counter()
    last = None
    for index in range(frame_count):
        last = render_context_frame(context, index / fps)
    render_seconds = time.perf_counter() - started
    return {
        "backend": BACKEND_NAME,
        "device": context["device"],
        "width": width,
        "height": height,
        "supersample": supersample,
        "frames": frame_count,
        "init_seconds": init_seconds,
        "render_seconds": render_seconds,
        "render_fps": frame_count / max(render_seconds, 1e-9),
        "last_frame_sha256": image_digest(last),
    }


def _parse_remote_result(stdout: str) -> dict[str, object]:
    for line in reversed(stdout.splitlines()):
        if line.startswith(RESULT_MAGIC):
            return json.loads(line[len(RESULT_MAGIC) :])
    raise RuntimeError(f"Remote worker returned no result marker. Output:\n{stdout[-4000:]}")


def _run_remote_worker(host: str, arguments: list[str]) -> dict[str, object]:
    completed = _run(
        _ssh_command(host, _remote_shell_command(arguments)),
        capture_output=True,
        text=True,
    )
    return _parse_remote_result(completed.stdout)


def _copy_remote_output(host: str, remote_path: str, output_path: str | Path) -> Path:
    with atomic_output_path(output_path) as (temporary, output):
        _run(
            [
                "scp",
                *SSH_OPTIONS,
                f"{validate_ssh_host(host)}:~/{REMOTE_ROOT}/{remote_path}",
                str(temporary),
            ]
        )
    return output


def _cleanup_remote_job(host: str, job_id: str) -> None:
    subprocess.run(
        _ssh_command(host, f'rm -rf "$HOME/{REMOTE_ROOT}/.jobs/{job_id}"'),
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def render_still(
    output_path: str | Path,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    time_s: float = 1.0,
    supersample: int = 2,
    host: str = DEFAULT_HOST,
    local: bool = True,
) -> dict[str, object]:
    if local:
        return render_still_local(output_path, width, height, time_s, supersample)
    setup_remote(host)
    job_id = uuid.uuid4().hex
    remote_output = f".jobs/{job_id}/frame.png"
    started = time.perf_counter()
    try:
        result = _run_remote_worker(
            host,
            [
                "still",
                "--output",
                remote_output,
                "--width",
                str(width),
                "--height",
                str(height),
                "--time",
                str(time_s),
                "--supersample",
                str(supersample),
            ],
        )
        output = _copy_remote_output(host, remote_output, output_path)
    finally:
        _cleanup_remote_job(host, job_id)
    return {
        **result,
        "output": str(output),
        "host": host,
        "roundtrip_seconds": time.perf_counter() - started,
    }


def record_video(
    output_path: str | Path,
    duration: float = DEFAULT_RECORDING_DURATION,
    fps: int = DEFAULT_VIDEO_FPS,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    supersample: int = 2,
    host: str = DEFAULT_HOST,
    local: bool = True,
) -> dict[str, object]:
    if local:
        return record_video_local(
            output_path, duration, fps, width, height, supersample=supersample
        )
    setup_remote(host)
    job_id = uuid.uuid4().hex
    remote_output = f".jobs/{job_id}/video.mp4"
    started = time.perf_counter()
    try:
        result = _run_remote_worker(
            host,
            [
                "record",
                "--output",
                remote_output,
                "--duration",
                str(duration),
                "--fps",
                str(fps),
                "--width",
                str(width),
                "--height",
                str(height),
                "--supersample",
                str(supersample),
            ],
        )
        output = _copy_remote_output(host, remote_output, output_path)
    finally:
        _cleanup_remote_job(host, job_id)
    return {
        **result,
        "output": str(output),
        "host": host,
        "roundtrip_seconds": time.perf_counter() - started,
    }


def benchmark(
    frame_count: int = 30,
    width: int = 640,
    height: int = 360,
    fps: int = 30,
    warmup: int = 3,
    supersample: int = 1,
    host: str = DEFAULT_HOST,
    local: bool = True,
) -> dict[str, object]:
    if local:
        return benchmark_local(frame_count, width, height, fps, warmup, supersample=supersample)
    setup_remote(host)
    result = _run_remote_worker(
        host,
        [
            "benchmark",
            "--frames",
            str(frame_count),
            "--width",
            str(width),
            "--height",
            str(height),
            "--fps",
            str(fps),
            "--warmup",
            str(warmup),
            "--supersample",
            str(supersample),
        ],
    )
    return {**result, "host": host}


def _remote_stream_frames(
    host: str,
    width: int,
    height: int,
    fps: int,
    duration: float | None,
    supersample: int,
):
    arguments = [
        "stream",
        "--width",
        str(width),
        "--height",
        str(height),
        "--fps",
        str(fps),
        "--supersample",
        str(supersample),
    ]
    if duration is not None:
        arguments.extend(["--duration", str(duration)])
    process = subprocess.Popen(
        _ssh_command(host, _remote_shell_command(arguments)),
        stdout=subprocess.PIPE,
        stderr=None,
        bufsize=0,
        start_new_session=True,
    )
    assert process.stdout is not None
    startup_lines: list[bytes] = []
    frame_count = 0
    expected_frames = None if duration is None else sum(1 for _ in trajectory_times(duration, fps))
    try:
        while True:
            line = process.stdout.readline()
            if not line:
                return_code = process.poll()
                if return_code not in (None, 0):
                    raise RuntimeError(
                        f"Remote Warp stream worker exited with {return_code} before its handshake"
                    )
                raise RuntimeError(
                    "Remote stream ended before its handshake:\n"
                    + b"".join(startup_lines[-40:]).decode(errors="replace")
                )
            if line == STREAM_MAGIC:
                break
            startup_lines.append(line)
        expected = width * height * 3
        while True:
            length_bytes = process.stdout.read(4)
            if not length_bytes:
                break
            if len(length_bytes) != 4:
                length_bytes += read_exact(process.stdout, 4 - len(length_bytes), max_bytes=4)
            frame_length = struct.unpack("!I", length_bytes)[0]
            if frame_length != expected:
                raise RuntimeError(f"Remote stream sent {frame_length} bytes; expected {expected}")
            frame = np.frombuffer(
                read_exact(process.stdout, frame_length, max_bytes=expected),
                dtype=np.uint8,
            )
            frame_count += 1
            if expected_frames is not None and frame_count > expected_frames:
                raise RuntimeError(
                    f"Remote Warp stream produced more than {expected_frames} frames"
                )
            yield frame.reshape(height, width, 3).copy()
        try:
            return_code = process.wait(timeout=5.0)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                "Remote Warp stream closed stdout but did not exit within 5 seconds"
            ) from exc
        if return_code != 0:
            raise RuntimeError(f"Remote Warp stream worker exited with {return_code}")
        if expected_frames is not None and frame_count != expected_frames:
            raise RuntimeError(
                f"Remote Warp stream produced {frame_count} frames; expected {expected_frames}"
            )
    finally:
        terminate_and_reap({"warp": process}, grace_seconds=3.0)


def render_live(
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    fps: int = DEFAULT_VIDEO_FPS,
    duration: float | None = None,
    supersample: int = 1,
    host: str = DEFAULT_HOST,
    local: bool = True,
) -> dict[str, object]:
    if local:
        context = create_renderer(width, height, supersample=supersample, device="cpu")

        def frames():
            frame_index = 0
            while duration is None or frame_index / fps < duration:
                yield render_context_frame(context, frame_index / fps)
                frame_index += 1

        result = display_frames(frames(), width, height, "MuJoCo Warp (local CPU)", target_fps=fps)
        return {"backend": BACKEND_NAME, "device": "cpu", **result}

    setup_remote(host)
    result = display_frames(
        _remote_stream_frames(host, width, height, fps, duration, supersample),
        width,
        height,
        f"MuJoCo Warp (CUDA on {host})",
        target_fps=fps,
    )
    return {"backend": BACKEND_NAME, "host": host, **result}


def _worker_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("mode", choices=["still", "record", "benchmark", "stream"])
    parser.add_argument("--output")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--time", type=float, default=1.0)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--fps", type=int, default=DEFAULT_VIDEO_FPS)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--supersample", type=int, default=2)
    return parser


def _worker_stream(args: argparse.Namespace) -> None:
    context = create_renderer(args.width, args.height, args.supersample, device="cuda:0")
    render_context_frame(context, 0.0)  # JIT and allocate before the binary handshake.
    # Warp writes startup/JIT logs through the text wrapper; drain them before raw bytes.
    sys.stdout.flush()
    sys.stdout.buffer.write(STREAM_MAGIC)
    sys.stdout.buffer.flush()

    def stream_times():
        if args.duration is not None:
            yield from trajectory_times(args.duration, args.fps)
            return
        frame_index = 0
        while True:
            yield frame_index / args.fps
            frame_index += 1

    try:
        for time_s in stream_times():
            frame = render_context_frame(context, time_s)
            payload = frame.tobytes()
            sys.stdout.buffer.write(struct.pack("!I", len(payload)))
            sys.stdout.buffer.write(payload)
            sys.stdout.buffer.flush()
    except BrokenPipeError:
        return


def _worker_main(arguments: list[str]) -> int:
    args = _worker_parser().parse_args(arguments)
    if args.mode == "stream":
        _worker_stream(args)
        return 0
    if args.mode == "still":
        if not args.output:
            raise SystemExit("--output is required")
        result = render_still_local(
            args.output,
            args.width,
            args.height,
            args.time,
            args.supersample,
            device="cuda:0",
        )
    elif args.mode == "record":
        if not args.output or args.duration is None:
            raise SystemExit("--output and --duration are required")
        result = record_video_local(
            args.output,
            args.duration,
            args.fps,
            args.width,
            args.height,
            args.supersample,
            device="cuda:0",
        )
    else:
        result = benchmark_local(
            args.frames,
            args.width,
            args.height,
            args.fps,
            args.warmup,
            args.supersample,
            device="cuda:0",
        )
    print(RESULT_MAGIC + json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "__worker":
        raise SystemExit(_worker_main(sys.argv[2:]))
    raise SystemExit("Use `robotic-renderers`; __worker is reserved for the SSH runner.")
