"""Mitsuba 3 CUDA backend rendered on a remote Linux RTX host.

The local uv process translates the canonical MuJoCo ``MjSpec`` scene into a
small renderer-neutral bundle. Long-lived remote workers load that bundle once,
update mesh and camera transforms in place, and render with Mitsuba's
``cuda_ad_rgb`` variant. Finite frame sequences are sharded over the host GPUs.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
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

BACKEND_NAME = "mitsuba"
MITSUBA_VERSION = "3.9.0"
DEFAULT_HOST = os.environ.get("ROBOT_RENDERING_MITSUBA_HOST", DEFAULT_RENDER_HOST)
PACKAGE_SOURCE = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_SOURCE.parents[1]
REMOTE_ROOT = "code/robot_rendering_mitsuba"
STREAM_MAGIC = b"ROBOT_RENDERING_MITSUBA_RGB_STREAM_V1\n"
STREAM_ERROR = b"ROBOT_RENDERING_MITSUBA_ERROR_V1\n"
MAX_HANDSHAKE_BYTES = 4096
MAX_ERROR_BYTES = 16 * 1024 * 1024
MITSUBA_LIGHT_INTENSITY_SCALE = {
    "directional": 1.0,
    "point": 1.0,
    "spot": 2.0,
}


def _job_root() -> Path:
    from robotic_renderers.defaults import cache_directory

    root = cache_directory() / "jobs" / BACKEND_NAME
    root.mkdir(parents=True, exist_ok=True)
    return root


def _run(command: list[str], **kwargs):
    return subprocess.run(command, check=True, **kwargs)


def _ssh_probe(host: str, command: str, timeout: float = 20.0) -> subprocess.CompletedProcess:
    return run_ssh(
        host,
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _gpu_inventory(host: str) -> list[str]:
    probe = _ssh_probe(
        host,
        "nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null",
    )
    if probe.returncode != 0:
        return []
    return [line.strip() for line in probe.stdout.splitlines() if line.strip()]


def backend_status(host: str = DEFAULT_HOST) -> dict[str, object]:
    """Return remote CUDA availability without mutating the host."""
    error = None
    version = None
    devices: list[str] = []
    try:
        devices = _gpu_inventory(host)
        if not devices:
            error = "No NVIDIA GPU was reported by nvidia-smi"
        else:
            probe_code = (
                'import mitsuba as mi; mi.set_variant("cuda_ad_rgb"); print(mi.__version__)'
            )
            probe = _ssh_probe(
                host,
                f'root="$HOME/{REMOTE_ROOT}"; '
                'test -x "$root/.venv/bin/python" || exit 4; '
                'test -f "$root/runtime/uv.lock" || exit 5; '
                'test -f "$root/source/robotic_renderers/backends/mitsuba.py" || exit 6; '
                'cd "$root"; PYTHONPATH="$root/source" '
                "CUDA_VISIBLE_DEVICES=0 MI_DEFAULT_VARIANT=cuda_ad_rgb "
                f'"$root/.venv/bin/python" -c {shlex.quote(probe_code)}',
                timeout=45.0,
            )
            if probe.returncode == 0:
                version = probe.stdout.strip().splitlines()[-1]
            else:
                messages = {
                    4: "The uv-managed Mitsuba environment is not installed",
                    5: "The frozen Mitsuba runtime project is not installed",
                    6: "The packaged Mitsuba worker source is not installed",
                }
                error = messages.get(
                    probe.returncode,
                    probe.stderr.strip() or probe.stdout.strip() or "Remote probe failed",
                )
    except (OSError, subprocess.SubprocessError) as exc:
        error = str(exc)
    available = bool(devices) and version == MITSUBA_VERSION and error is None
    return {
        "backend": BACKEND_NAME,
        "available": available,
        "version": version,
        "device": devices[0] if devices else None,
        "devices": devices,
        "device_count": len(devices),
        "execution": f"SSH ({host})",
        "host": host,
        "live": True,
        "record": True,
        "notes": (
            "Mitsuba uses cuda_ad_rgb on Linux. Finite sequences use persistent "
            "one-process-per-GPU workers; display and MP4 encoding remain local."
        ),
        "error": error,
    }


def _remote_install_script(upload_id: str, force: bool) -> str:
    reset = 'rm -rf "$root/.venv"' if force else ":"
    return f"""set -euo pipefail
umask 077
root="$HOME/{REMOTE_ROOT}"
incoming="$root/.uploads/{upload_id}"
test -f "$incoming/runtime/pyproject.toml"
test -f "$incoming/runtime/uv.lock"
test -f "$incoming/source/robotic_renderers/backends/mitsuba.py"
install -d -m 700 "$root"
exec 9>"$root/setup.lock"
chmod 600 "$root/setup.lock"
flock -w 120 9
{reset}
uv_bin=$(command -v uv)
test -n "$uv_bin"
UV_PROJECT_ENVIRONMENT="$root/.venv" \
  "$uv_bin" sync --directory "$incoming/runtime" --frozen --no-dev
PYTHONPATH="$incoming/source" \
CUDA_VISIBLE_DEVICES=0 MI_DEFAULT_VARIANT=cuda_ad_rgb \
  "$root/.venv/bin/python" -c \
  'import mitsuba as mi; mi.set_variant("cuda_ad_rgb"); print(mi.__version__, mi.variant())'

old="$root/.previous-{upload_id}"
rm -rf "$old"
install -d -m 700 "$old"
if [ -e "$root/source" ]; then mv "$root/source" "$old/source"; fi
if [ -e "$root/runtime" ]; then mv "$root/runtime" "$old/runtime"; fi
mv "$incoming/source" "$root/source"
mv "$incoming/runtime" "$root/runtime"
rmdir "$incoming"
rm -rf "$old"

rm -f "$root/render_mitsuba.py" "$root/mitsuba_runtime_pyproject.toml" \
  "$root/pyproject.toml" "$root/uv.lock"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | sed -n '1p'
"""


def setup_remote(host: str = DEFAULT_HOST, force: bool = False) -> dict[str, object]:
    """Install and synchronize the uv-managed Mitsuba worker on ``host``."""
    runtime_project = resolve_runtime_project(PACKAGE_SOURCE, PROJECT_ROOT, BACKEND_NAME)
    for required in (
        PACKAGE_SOURCE / "__init__.py",
        runtime_project / "pyproject.toml",
        runtime_project / "uv.lock",
    ):
        if not required.is_file():
            raise FileNotFoundError(f"Required Mitsuba runtime source is missing: {required}")

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
            raise RuntimeError(f"Mitsuba setup failed on {host}:\n{details}")
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
        "version": MITSUBA_VERSION,
        "details": completed.stdout.strip().splitlines()[-2:],
    }


def _geom_mesh(description: dict[str, object], source_geometry, meshes):
    return _shared_geom_mesh(
        description,
        source_geometry,
        meshes,
        renderer_name="Mitsuba",
    )


def _write_binary_ply(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray | None = None,
) -> None:
    vertices = np.asarray(vertices, dtype="<f4")
    if normals is not None:
        vertex_data = np.ascontiguousarray(
            np.concatenate((vertices, np.asarray(normals, dtype="<f4")), axis=1)
        )
    else:
        vertex_data = np.ascontiguousarray(vertices)
    faces = np.asarray(faces, dtype=np.int32)
    face_records = np.empty(
        len(faces),
        dtype=np.dtype([("count", "u1"), ("indices", "<i4", (3,))], align=False),
    )
    face_records["count"] = 3
    face_records["indices"] = faces
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        + (
            "property float nx\nproperty float ny\nproperty float nz\n"
            if normals is not None
            else ""
        )
        + f"element face {len(faces)}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    ).encode()
    with path.open("wb") as stream:
        stream.write(header)
        stream.write(vertex_data.tobytes())
        stream.write(face_records.tobytes())


def _write_gradient_sky(destination: Path) -> Path:
    return write_gradient_sky(destination, width=1024, height=512)


def prepare_mitsuba_bundle(
    destination: str | Path,
    times: list[float],
    width: int,
    height: int,
    fps: int,
    samples: int = 256,
) -> Path:
    """Translate canonical compiled geometry and animation into a portable bundle."""
    from robotic_renderers.scene import create_model_and_data
    from robotic_renderers.scene_bundle import prepare_scene_bundle

    destination = Path(destination).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = prepare_scene_bundle(destination, times, width, height, fps)
    manifest = json.loads(manifest_path.read_text())
    source_geometry = np.load(destination / manifest["geometry"])
    model, _ = create_model_and_data(width=width, height=height)
    mesh_directory = destination / "mitsuba_meshes"
    mesh_directory.mkdir(exist_ok=True)
    runtime_geometry: dict[str, np.ndarray] = {}
    shapes = []
    for index, description in enumerate(manifest["geoms"]):
        if description["type"] == "plane":
            shapes.append({"kind": "plane", "parameter": f"geom_{index:03d}"})
            continue
        vertices, faces = _geom_mesh(description, source_geometry, manifest["meshes"])
        smooth = not (
            description["type"] in {"box", "cylinder"}
            or (
                description["type"] == "mesh"
                and manifest["meshes"][str(description["mesh_id"])]["name"] == "mesh_gem"
            )
        )
        normals = _surface_normals(vertices, faces)
        if description["type"] == "mesh" and smooth:
            mesh_id = int(description["mesh_id"])
            normal_address = int(model.mesh_normaladr[mesh_id])
            face_address = int(model.mesh_faceadr[mesh_id])
            face_count = int(model.mesh_facenum[mesh_id])
            normal_indices = np.asarray(
                model.mesh_facenormal[face_address : face_address + face_count],
                dtype=np.int64,
            )
            corner_normals = np.asarray(
                model.mesh_normal[normal_address + normal_indices],
                dtype=np.float32,
            ).reshape(-1, 3)
            vertices = np.asarray(vertices[faces], dtype=np.float32).reshape(-1, 3)
            faces = np.arange(len(vertices), dtype=np.int32).reshape(-1, 3)
            normals = corner_normals
        mesh_path = mesh_directory / f"geom_{index:03d}.ply"
        _write_binary_ply(mesh_path, vertices, faces, normals if smooth else None)
        runtime_geometry[f"vertices_{index}"] = vertices
        runtime_geometry[f"normals_{index}"] = normals
        shapes.append(
            {
                "kind": "mesh",
                "parameter": f"geom_{index:03d}",
                "filename": str(mesh_path.relative_to(destination)),
                "vertices": f"vertices_{index}",
                "normals": f"normals_{index}",
                "smooth": smooth,
            }
        )
    runtime_geometry_path = destination / "mitsuba_geometry.npz"
    np.savez_compressed(runtime_geometry_path, **runtime_geometry)
    sky_path = _write_gradient_sky(destination)
    bundle = {
        "manifest": manifest_path.name,
        "geometry": runtime_geometry_path.name,
        "sky": sky_path.name,
        "shapes": shapes,
        "width": width,
        "height": height,
        "fps": fps,
        "frames": len(times),
        "samples": max(1, int(samples)),
        "variant": "cuda_ad_rgb",
        "max_depth": 12,
        "rr_depth": 5,
        "denoise": True,
        "tone_exposure": 0.36,
    }
    bundle_path = destination / "mitsuba_bundle.json"
    bundle_path.write_text(json.dumps(bundle, separators=(",", ":")))
    return bundle_path


def _upload_job(host: str, local_job: Path, job_id: str) -> str:
    remote_job = f".jobs/{job_id}"
    try:
        run_ssh(
            host,
            f'umask 077; install -d -m 700 "$HOME/{REMOTE_ROOT}/{remote_job}"',
        )
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


def _terminate_remote_job_workers(host: str, remote_job: str) -> None:
    command = worker_cleanup_script(
        REMOTE_ROOT,
        remote_job,
        "robotic_renderers.backends.mitsuba",
        "worker*.pid",
    )
    with contextlib.suppress(OSError, subprocess.SubprocessError):
        subprocess.run(
            ssh_command(host, command),
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=8,
        )


def _terminate_and_reap(process: subprocess.Popen[bytes], timeout: float = 5.0) -> None:
    if process.poll() is None:
        with contextlib.suppress(ProcessLookupError):
            process.terminate()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            process.kill()
        process.wait()


def _terminate_processes_and_reap(
    host: str,
    remote_job: str,
    processes: list[subprocess.Popen],
) -> None:
    """Stop remote workers, then terminate, escalate, and reap every local SSH child."""
    _terminate_remote_job_workers(host, remote_job)
    for process in processes:
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            _terminate_and_reap(process, timeout=3.0)


def _spawn_worker_processes(
    host: str,
    remote_job: str,
    commands: list[str],
) -> list[subprocess.Popen[str]]:
    """Spawn all SSH workers without leaking an earlier child if a later spawn fails."""
    processes: list[subprocess.Popen[str]] = []
    try:
        for command in commands:
            processes.append(
                subprocess.Popen(
                    ssh_command(host, command),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            )
    except BaseException:
        _terminate_processes_and_reap(host, remote_job, processes)
        raise
    return processes


def _stderr_after_exit(process: subprocess.Popen[bytes]) -> str:
    """Read stderr only when an exited child guarantees EOF."""
    if process.poll() is None:
        return ""
    assert process.stderr is not None
    return process.stderr.read().decode(errors="replace")


def _remote_stream_command(remote_job: str, loop: bool) -> str:
    arguments = [
        ".venv/bin/python",
        "-m",
        "robotic_renderers.backends.mitsuba",
        "__worker",
        "--mode",
        "stream",
        "--bundle",
        f"{remote_job}/mitsuba_bundle.json",
        "--pid-file",
        f"{remote_job}/worker_stream.pid",
    ]
    if loop:
        arguments.append("--loop")
    return (
        f'cd "$HOME/{REMOTE_ROOT}" && '
        'export PYTHONPATH="$PWD/source" && '
        "exec setsid --wait env CUDA_VISIBLE_DEVICES=0 MI_DEFAULT_VARIANT=cuda_ad_rgb "
        f"{shlex.join(arguments)}"
    )


def _remote_stream_frames(host: str, remote_job: str, width: int, height: int, loop: bool):
    process = subprocess.Popen(
        ssh_command(host, _remote_stream_command(remote_job, loop)),
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
            raise RuntimeError(f"Mitsuba worker failed before rendering:\n{message}")
        if handshake != STREAM_MAGIC:
            return_code = process.poll()
            stderr = _stderr_after_exit(process)
            state = f"exit {return_code}" if return_code is not None else "still running"
            raise RuntimeError(
                f"Mitsuba worker produced an invalid handshake ({state}):\n"
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
                raise RuntimeError(f"Mitsuba worker failed while rendering:\n{message}")
            if frame_length != expected:
                raise RuntimeError(f"Mitsuba sent {frame_length} bytes; expected {expected}")
            frame = np.frombuffer(
                _read_exact(
                    process.stdout,
                    frame_length,
                    max_bytes=expected,
                ),
                dtype=np.uint8,
            )
            yield frame.reshape(height, width, 3).copy()
        return_code = process.wait()
        if return_code != 0:
            stderr = process.stderr.read().decode(errors="replace")
            raise RuntimeError(f"Mitsuba worker exited with {return_code}:\n{stderr[-12000:]}")
        completed_normally = True
    finally:
        if not completed_normally:
            _terminate_remote_job_workers(host, remote_job)
            _terminate_and_reap(process)
        elif process.poll() is None:
            process.wait()


def _frame_source(
    times: list[float],
    width: int,
    height: int,
    fps: int,
    samples: int,
    host: str,
    loop: bool = False,
):
    setup_remote(host)
    job_id = uuid.uuid4().hex
    with tempfile.TemporaryDirectory(prefix="mitsuba-", dir=_job_root()) as temporary:
        job = Path(temporary)
        prepare_mitsuba_bundle(job, times, width, height, fps, samples)
        remote_job = _upload_job(host, job, job_id)
        try:
            yield from _remote_stream_frames(host, remote_job, width, height, loop)
        finally:
            _cleanup_remote_job(host, job_id)


def _worker_count(host: str, frame_count: int, workers: int | None) -> int:
    available = max(1, len(_gpu_inventory(host)))
    requested = available if workers is None else max(1, int(workers))
    return min(available, requested, max(1, frame_count))


def _batch_worker_command(
    remote_job: str,
    gpu: int,
    indices: list[int],
    output_directory: str,
) -> str:
    arguments = [
        ".venv/bin/python",
        "-m",
        "robotic_renderers.backends.mitsuba",
        "__worker",
        "--mode",
        "batch",
        "--bundle",
        f"{remote_job}/mitsuba_bundle.json",
        "--indices",
        ",".join(map(str, indices)),
        "--output",
        output_directory,
        "--pid-file",
        f"{remote_job}/worker_gpu_{gpu}.pid",
    ]
    return (
        f'cd "$HOME/{REMOTE_ROOT}" && '
        'export PYTHONPATH="$PWD/source" && '
        f"exec setsid --wait env CUDA_VISIBLE_DEVICES={gpu} "
        f"MI_DEFAULT_VARIANT=cuda_ad_rgb {shlex.join(arguments)}"
    )


def _collect_processes_fail_fast(
    host: str,
    remote_job: str,
    processes: list[subprocess.Popen[str]],
    label: str,
) -> tuple[list[tuple[str, str]], float]:
    started = time.perf_counter()
    pending = set(range(len(processes)))
    failure_index = None
    try:
        while pending:
            for index in tuple(pending):
                return_code = processes[index].poll()
                if return_code is None:
                    continue
                pending.remove(index)
                if return_code != 0:
                    failure_index = index
                    break
            if failure_index is not None:
                break
            if pending:
                time.sleep(0.05)
    except BaseException:
        _terminate_processes_and_reap(host, remote_job, processes)
        raise
    if failure_index is not None:
        _terminate_processes_and_reap(host, remote_job, processes)
    try:
        outputs = [process.communicate() for process in processes]
    except BaseException:
        _terminate_processes_and_reap(host, remote_job, processes)
        raise
    if failure_index is not None:
        details = []
        for index, (process, (stdout, stderr)) in enumerate(zip(processes, outputs, strict=True)):
            if process.returncode:
                details.append(
                    f"{label} {index} exited with {process.returncode}:\n"
                    f"{(stdout + stderr)[-12000:]}"
                )
        raise RuntimeError(f"Mitsuba {label} failed:\n" + "\n".join(details))
    return outputs, time.perf_counter() - started


def _run_remote_batch(
    host: str,
    remote_job: str,
    frame_count: int,
    workers: int | None,
) -> tuple[float, int]:
    worker_count = _worker_count(host, frame_count, workers)
    assignments = [list(range(worker, frame_count, worker_count)) for worker in range(worker_count)]
    output_directory = f"{remote_job}/frames"
    run_ssh(
        host,
        f'umask 077; install -d -m 700 "$HOME/{REMOTE_ROOT}/{output_directory}"',
    )
    commands = [
        _batch_worker_command(remote_job, gpu, indices, output_directory)
        for gpu, indices in enumerate(assignments)
    ]
    processes = _spawn_worker_processes(host, remote_job, commands)
    _, seconds = _collect_processes_fail_fast(host, remote_job, processes, "batch worker")
    return seconds, worker_count


def render_frame(
    width: int = 1280,
    height: int = 720,
    time_s: float = 1.0,
    samples: int = 256,
    host: str = DEFAULT_HOST,
) -> np.ndarray:
    source = _frame_source([time_s], width, height, 1, samples, host)
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
    from robotic_renderers.media import image_digest, write_png

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
        "device": f"Mitsuba CUDA on {host}",
    }


def render_live(
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    duration: float | None = None,
    samples: int = 64,
    host: str = DEFAULT_HOST,
) -> dict[str, object]:
    from robotic_renderers.media import display_frames
    from robotic_renderers.scene import CAMERA_ORBIT_PERIOD, trajectory_times

    source_duration = duration if duration is not None else CAMERA_ORBIT_PERIOD
    times = list(trajectory_times(source_duration, fps))
    result = display_frames(
        _frame_source(times, width, height, fps, samples, host, loop=duration is None),
        width,
        height,
        f"Mitsuba CUDA ({host})",
        target_fps=fps,
    )
    return {"backend": BACKEND_NAME, "host": host, **result}


def record_video(
    output_path: str | Path,
    duration: float = 30.0,
    fps: int = 30,
    width: int = 1280,
    height: int = 720,
    samples: int = 256,
    host: str = DEFAULT_HOST,
    workers: int | None = None,
) -> dict[str, object]:
    from PIL import Image

    from robotic_renderers.media import write_video
    from robotic_renderers.scene import trajectory_times

    setup_remote(host)
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    times = list(trajectory_times(duration, fps))
    job_id = uuid.uuid4().hex
    with tempfile.TemporaryDirectory(
        prefix="mitsuba-video-",
        dir=_job_root(),
    ) as temporary:
        job = Path(temporary)
        bundle = prepare_mitsuba_bundle(job, times, width, height, fps, samples)
        remote_job = _upload_job(host, job, job_id)
        frame_directory = job / "rendered_frames"
        frame_directory.mkdir()
        started = time.perf_counter()
        try:
            render_seconds, worker_count = _run_remote_batch(host, remote_job, len(times), workers)
            _run(
                rsync_command(
                    rsync_remote(
                        host,
                        f"~/{REMOTE_ROOT}/{remote_job}/frames/",
                    ),
                    f"{frame_directory}/",
                )
            )
            frame_paths = sorted(frame_directory.glob("frame_*.png"))
            if len(frame_paths) != len(times):
                raise RuntimeError(
                    f"Mitsuba produced {len(frame_paths)} frames; expected {len(times)}"
                )

            def frames():
                for frame_path in frame_paths:
                    with Image.open(frame_path) as image:
                        yield np.asarray(image.convert("RGB"))

            _, frame_count, encode_seconds = write_video(frames(), output, fps)
            seconds = time.perf_counter() - started
            return {
                "backend": BACKEND_NAME,
                "output": str(output),
                "frames": frame_count,
                "seconds": seconds,
                "render_seconds": render_seconds,
                "encode_seconds": encode_seconds,
                "video_fps": fps,
                "render_fps": frame_count / max(render_seconds, 1e-9),
                "width": width,
                "height": height,
                "samples": samples,
                "workers": worker_count,
                "device": f"Mitsuba CUDA on {host}",
                "bundle": bundle.name,
            }
        finally:
            _cleanup_remote_job(host, job_id)


def _benchmark_worker_command(
    remote_job: str,
    gpu: int,
    count: int,
    warmup: int,
    start_index: int,
    stride: int,
) -> str:
    arguments = [
        ".venv/bin/python",
        "-m",
        "robotic_renderers.backends.mitsuba",
        "__worker",
        "--mode",
        "benchmark",
        "--bundle",
        f"{remote_job}/mitsuba_bundle.json",
        "--count",
        str(count),
        "--warmup",
        str(warmup),
        "--start-index",
        str(start_index),
        "--stride",
        str(stride),
        "--pid-file",
        f"{remote_job}/worker_benchmark_gpu_{gpu}.pid",
    ]
    return (
        f'cd "$HOME/{REMOTE_ROOT}" && '
        'export PYTHONPATH="$PWD/source" && '
        f"exec setsid --wait env CUDA_VISIBLE_DEVICES={gpu} "
        f"MI_DEFAULT_VARIANT=cuda_ad_rgb {shlex.join(arguments)}"
    )


def benchmark(
    frame_count: int = 10,
    width: int = 640,
    height: int = 360,
    fps: int = 30,
    warmup: int = 3,
    samples: int = 32,
    host: str = DEFAULT_HOST,
    workers: int | None = None,
) -> dict[str, object]:
    """Measure persistent CUDA throughput after per-worker JIT warmup."""
    from robotic_renderers.scene import trajectory_times

    if frame_count < 1:
        raise ValueError("frame_count must be at least 1")
    setup_remote(host)
    worker_count = _worker_count(host, frame_count, workers)
    # Every worker needs enough authored transforms for its warmup and measured frames.
    per_worker_counts = [
        len(range(worker, frame_count, worker_count)) for worker in range(worker_count)
    ]
    authored_count = max(1, warmup + max(per_worker_counts)) * worker_count
    times = list(trajectory_times(authored_count / fps, fps))
    job_id = uuid.uuid4().hex
    with tempfile.TemporaryDirectory(
        prefix="mitsuba-benchmark-",
        dir=_job_root(),
    ) as temporary:
        job = Path(temporary)
        prepare_mitsuba_bundle(job, times, width, height, fps, samples)
        remote_job = _upload_job(host, job, job_id)
        commands = [
            _benchmark_worker_command(
                remote_job,
                gpu,
                per_worker_counts[gpu],
                warmup,
                gpu,
                worker_count,
            )
            for gpu in range(worker_count)
        ]
        try:
            processes = _spawn_worker_processes(host, remote_job, commands)
            outputs, wall_seconds = _collect_processes_fail_fast(
                host, remote_job, processes, "benchmark worker"
            )
            results = [json.loads(stdout.strip().splitlines()[-1]) for stdout, _ in outputs]
            render_seconds = max(result["measured_seconds"] for result in results)
            return {
                "backend": BACKEND_NAME,
                "device": f"Mitsuba CUDA on {host}",
                "host": host,
                "width": width,
                "height": height,
                "samples": samples,
                "frames": frame_count,
                "warmup": warmup,
                "workers": worker_count,
                "render_seconds": render_seconds,
                "render_fps": frame_count / max(render_seconds, 1e-9),
                "roundtrip_seconds": wall_seconds,
                "roundtrip_fps": frame_count / max(wall_seconds, 1e-9),
                "worker_results": results,
            }
        finally:
            _terminate_remote_job_workers(host, remote_job)
            _cleanup_remote_job(host, job_id)


def _worker_transform(mi, position, rotation):
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = np.asarray(rotation, dtype=np.float32)
    matrix[:3, 3] = np.asarray(position, dtype=np.float32)
    return mi.ScalarTransform4f(matrix)


def _worker_look_at(mi, camera):
    position = np.asarray(camera["position"], dtype=np.float32)
    rotation = np.asarray(camera["rotation"], dtype=np.float32)
    return mi.ScalarTransform4f.look_at(
        origin=position,
        target=position - rotation[:, 2],
        up=rotation[:, 1],
    )


def _worker_spot_beam_width(cutoff: float, exponent: float) -> float:
    """Approximate MuJoCo's cosine-power falloff with Mitsuba's linear ramp."""
    cutoff = max(0.01, min(89.9, float(cutoff)))
    exponent = max(0.0, float(exponent))
    if exponent == 0.0:
        return cutoff
    half_power_angle = math.degrees(math.acos(0.5 ** (1.0 / exponent)))
    return min(cutoff, max(0.0, 2.0 * half_power_angle - cutoff))


def _worker_light_up(direction: np.ndarray) -> list[float]:
    normalized = direction / np.linalg.norm(direction)
    if abs(float(normalized[2])) > 0.98:
        return [0.0, 1.0, 0.0]
    return [0.0, 0.0, 1.0]


def _worker_light_description(mi, light: dict[str, object]) -> tuple[dict[str, object], bool]:
    """Translate one canonical light and retain its shadow intent for parity reporting."""
    light_type = str(light["type"])
    try:
        unit_scale = MITSUBA_LIGHT_INTENSITY_SCALE[light_type]
    except KeyError as exc:
        raise RuntimeError(f"Unsupported Mitsuba light type {light_type!r}") from exc
    color = np.asarray(light["diffuse"], dtype=np.float64)
    intensity = color * float(light["intensity"]) * unit_scale
    spectrum = {"type": "rgb", "value": intensity.tolist()}
    cast_shadow = bool(light["cast_shadow"])
    if light_type == "directional":
        emitter = {
            "type": "directional",
            "direction": list(light["direction"]),
            "irradiance": spectrum,
        }
    elif light_type == "point":
        emitter = {
            "type": "point",
            "position": list(light["position"]),
            "intensity": spectrum,
        }
    else:
        position = np.asarray(light["position"], dtype=np.float64)
        direction = np.asarray(light["direction"], dtype=np.float64)
        if not np.isfinite(direction).all() or np.linalg.norm(direction) < 1e-12:
            raise ValueError(f"Mitsuba spot light {light['name']!r} has an invalid direction")
        cutoff = max(0.01, min(89.9, float(light["cutoff"])))
        emitter = {
            "type": "spot",
            "to_world": mi.ScalarTransform4f.look_at(
                origin=position.tolist(),
                target=(position + direction).tolist(),
                up=_worker_light_up(direction),
            ),
            "intensity": spectrum,
            "cutoff_angle": cutoff,
            "beam_width": _worker_spot_beam_width(cutoff, float(light["exponent"])),
        }
    return emitter, cast_shadow


def _worker_material(mi, description: dict[str, object], floor: bool = False):
    rgba = description["rgba"]
    base_color: object
    if floor:
        base_color = {
            "type": "checkerboard",
            "color0": {"type": "rgb", "value": [0.094, 0.113, 0.133]},
            "color1": {"type": "rgb", "value": [0.022, 0.027, 0.033]},
            # Mitsuba's checker switches color twice per UV period. A scale of
            # 300 on the 200 m floor therefore yields the canonical 1/3 m tiles.
            "to_uv": mi.ScalarTransform3f.scale([300.0, 300.0]),
        }
    else:
        base_color = {"type": "rgb", "value": rgba[:3]}
    return {
        "type": "principled",
        "base_color": base_color,
        "metallic": float(description["metallic"]),
        "roughness": max(0.035, float(description["roughness"])),
        "specular": min(1.0, 0.25 + 0.5 * float(description["specular"])),
    }


def _worker_build_scene(mi, bundle_path: Path):
    bundle = json.loads(bundle_path.read_text())
    manifest_path = bundle_path.parent / bundle["manifest"]
    manifest = json.loads(manifest_path.read_text())
    first_frame = manifest["frames"][0]
    sensor = {
        "type": "perspective",
        "fov_axis": "y",
        "fov": float(manifest["camera"]["fovy"]),
        "near_clip": 0.01,
        "far_clip": 50.0,
        "to_world": _worker_look_at(mi, first_frame["camera"]),
        "sampler": {"type": "independent", "sample_count": bundle["samples"]},
        "film": {
            "type": "hdrfilm",
            "width": bundle["width"],
            "height": bundle["height"],
            "pixel_format": "rgb",
            "component_format": "float32",
            "sample_border": True,
            # OptiX expects a box-filtered input. Wider reconstruction filters
            # correlate neighboring noise and significantly weaken denoising.
            "rfilter": {"type": "box"},
        },
    }
    scene_description: dict[str, object] = {
        "type": "scene",
        "integrator": {
            "type": "aov",
            "aovs": "albedo:albedo,normal:sh_normal",
            "integrator": {
                "type": "path",
                "max_depth": bundle["max_depth"],
                "rr_depth": bundle["rr_depth"],
            },
        },
        "sensor": sensor,
        "studio_sky": {
            "type": "envmap",
            "filename": str(bundle_path.parent / bundle["sky"]),
            "scale": 0.35,
            "to_world": mi.ScalarTransform4f.rotate([1.0, 0.0, 0.0], -90.0),
        },
    }
    light_cast_shadow = {}
    for light in manifest["lights"]:
        emitter, cast_shadow = _worker_light_description(mi, light)
        name = str(light["name"])
        scene_description[name] = emitter
        light_cast_shadow[name] = cast_shadow
    for description, shape, transform in zip(
        manifest["geoms"],
        bundle["shapes"],
        first_frame["transforms"],
        strict=True,
    ):
        to_world = _worker_transform(mi, transform["position"], transform["rotation"])
        if shape["kind"] == "plane":
            scene_description[shape["parameter"]] = {
                "type": "rectangle",
                "to_world": to_world @ mi.ScalarTransform4f.scale([100.0, 100.0, 1.0]),
                "bsdf": _worker_material(mi, description["material"], floor=True),
            }
        else:
            scene_description[shape["parameter"]] = {
                "type": "ply",
                "filename": str(bundle_path.parent / shape["filename"]),
                "face_normals": not shape["smooth"],
                "to_world": to_world,
                "bsdf": _worker_material(mi, description["material"]),
            }
    # The default optimizer merges geometrically identical shapes into instances.
    # Keep explicit meshes so each articulated link remains independently movable.
    scene = mi.load_dict(scene_description, optimize=False)
    parameters = mi.traverse(scene)
    geometry = np.load(bundle_path.parent / bundle["geometry"])
    denoiser = (
        mi.OptixDenoiser(
            [bundle["width"], bundle["height"]],
            True,
            True,
            False,
        )
        if bundle["denoise"]
        else None
    )
    return {
        "bundle": bundle,
        "manifest": manifest,
        "scene": scene,
        "parameters": parameters,
        "geometry": geometry,
        "denoiser": denoiser,
        # Mitsuba's path tracer always evaluates emitter visibility and has no
        # per-emitter shadow switch. Retain the canonical flags so this explicit
        # backend limitation is inspectable instead of silently losing metadata.
        "light_cast_shadow": light_cast_shadow,
    }


def _worker_apply_frame(mi, runtime, index: int) -> None:
    manifest = runtime["manifest"]
    frame = manifest["frames"][index]
    parameters = runtime["parameters"]
    geometry = runtime["geometry"]
    for shape, transform in zip(runtime["bundle"]["shapes"], frame["transforms"], strict=True):
        if shape["kind"] == "plane":
            continue
        rotation = np.asarray(transform["rotation"], dtype=np.float32)
        position = np.asarray(transform["position"], dtype=np.float32)
        vertices = np.asarray(geometry[shape["vertices"]], dtype=np.float32)
        world_vertices = vertices @ rotation.T + position
        parameters[f"{shape['parameter']}.vertex_positions"] = world_vertices.reshape(-1)
        normal_key = f"{shape['parameter']}.vertex_normals"
        if shape["smooth"] and normal_key in parameters:
            normals = np.asarray(geometry[shape["normals"]], dtype=np.float32)
            parameters[normal_key] = (normals @ rotation.T).reshape(-1)
    parameters["sensor.to_world"] = mi.Transform4f(_worker_look_at(mi, frame["camera"]))
    parameters.update()


def _worker_tonemap(image, exposure: float) -> np.ndarray:
    linear = np.asarray(image, dtype=np.float32)
    if linear.shape[-1] > 3:
        linear = linear[:, :, :3]
    # Point lights on glossy metallic surfaces can produce rare, extremely
    # energetic single-pixel paths. Remove only isolated outliers before the
    # filmic curve; coherent highlights and edges retain their local support.
    height, width = linear.shape[:2]
    padded = np.pad(linear, ((1, 1), (1, 1), (0, 0)), mode="edge")
    neighbors = np.stack(
        [
            padded[y : y + height, x : x + width]
            for y in range(3)
            for x in range(3)
            if (y, x) != (1, 1)
        ],
        axis=0,
    )
    local_median = np.median(neighbors, axis=0)
    luminance = linear @ np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
    median_luminance = local_median @ np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
    fireflies = (luminance > 5.0 * median_luminance + 1.0) & (np.max(linear, axis=2) > 8.0)
    linear = np.where(fireflies[:, :, np.newaxis], local_median, linear)
    linear = np.maximum(0.0, linear * float(exposure))
    mapped = (linear * (2.51 * linear + 0.03)) / (linear * (2.43 * linear + 0.59) + 0.14)
    mapped = np.clip(mapped, 0.0, 1.0)
    srgb = np.where(
        mapped <= 0.0031308,
        12.92 * mapped,
        1.055 * np.power(mapped, 1.0 / 2.4) - 0.055,
    )
    return np.ascontiguousarray(np.rint(srgb * 255.0).astype(np.uint8))


def _worker_render_frame(mi, runtime, index: int) -> np.ndarray:
    bundle = runtime["bundle"]
    _worker_apply_frame(mi, runtime, index)
    image = mi.render(
        runtime["scene"],
        spp=bundle["samples"],
        seed=index,
    )
    if runtime["denoiser"] is not None:
        image = runtime["denoiser"](
            image[:, :, :3],
            image[:, :, 3:6],
            image[:, :, 6:9],
        )
    else:
        image = image[:, :, :3]
    return _worker_tonemap(image, bundle["tone_exposure"])


def _worker_import_mitsuba():
    import mitsuba as mi

    mi.set_variant("cuda_ad_rgb")
    return mi


def _worker_stream(args: argparse.Namespace) -> int:
    binary_fd = os.dup(sys.stdout.fileno())
    bundle_path = Path(args.bundle).resolve()
    log_path = bundle_path.parent / "mitsuba_stream.log"
    log_stream = log_path.open("a", buffering=1)
    os.dup2(log_stream.fileno(), sys.stdout.fileno())
    os.dup2(log_stream.fileno(), sys.stderr.fileno())
    binary = os.fdopen(binary_fd, "wb", buffering=0)
    stream_started = False
    try:
        mi = _worker_import_mitsuba()
        runtime = _worker_build_scene(mi, bundle_path)
        binary.write(STREAM_MAGIC)
        stream_started = True
        index = 0
        while True:
            frame = _worker_render_frame(mi, runtime, index)
            payload = frame.tobytes()
            binary.write(struct.pack("!I", len(payload)))
            binary.write(payload)
            index += 1
            if index >= runtime["bundle"]["frames"]:
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
        log_stream.close()
    return 0


def _worker_batch(args: argparse.Namespace) -> int:
    from PIL import Image

    mi = _worker_import_mitsuba()
    bundle_path = Path(args.bundle).resolve()
    runtime = _worker_build_scene(mi, bundle_path)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    indices = [int(value) for value in args.indices.split(",") if value]
    started = time.perf_counter()
    for index in indices:
        frame = _worker_render_frame(mi, runtime, index)
        Image.fromarray(frame).save(output / f"frame_{index + 1:06d}.png", optimize=True)
    print(
        json.dumps(
            {
                "frames": len(indices),
                "seconds": time.perf_counter() - started,
                "indices": indices,
            },
            separators=(",", ":"),
        ),
        flush=True,
    )
    return 0


def _worker_benchmark(args: argparse.Namespace) -> int:
    mi = _worker_import_mitsuba()
    runtime = _worker_build_scene(mi, Path(args.bundle).resolve())
    frame_total = runtime["bundle"]["frames"]
    for offset in range(args.warmup):
        index = (args.start_index + offset * args.stride) % frame_total
        _worker_render_frame(mi, runtime, index)
    started = time.perf_counter()
    checksum = 0
    for offset in range(args.count):
        index = (args.start_index + (args.warmup + offset) * args.stride) % frame_total
        frame = _worker_render_frame(mi, runtime, index)
        checksum = (checksum + int(frame[0, 0, 0])) % 65536
    measured_seconds = time.perf_counter() - started
    print(
        json.dumps(
            {
                "frames": args.count,
                "measured_seconds": measured_seconds,
                "render_fps": args.count / max(measured_seconds, 1e-9),
                "checksum": checksum,
            },
            separators=(",", ":"),
        ),
        flush=True,
    )
    return 0


def _worker_arguments(arguments: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mode", choices=["stream", "batch", "benchmark"], required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--indices", default="")
    parser.add_argument("--output")
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--pid-file")
    return parser.parse_args(arguments)


def _worker_main(arguments: list[str]) -> int:
    args = _worker_arguments(arguments)
    pid_path = Path(args.pid_file).resolve() if args.pid_file else None
    if pid_path is not None:
        write_pid_identity(pid_path)
    try:
        if args.mode == "stream":
            return _worker_stream(args)
        if args.mode == "batch":
            if not args.output:
                raise SystemExit("--output is required in batch mode")
            return _worker_batch(args)
        return _worker_benchmark(args)
    finally:
        if pid_path is not None:
            pid_path.unlink(missing_ok=True)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "__worker":
        raise SystemExit(_worker_main(sys.argv[2:]))
    raise SystemExit("Use `robotic-renderers`; __worker is reserved for the SSH runner.")
