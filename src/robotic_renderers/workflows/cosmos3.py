"""Cosmos 3 generative post-processing for the canonical FR3 scene.

The renderer application remains macOS-native.  Only the 64B Cosmos3-Super
inference service runs on the configured multi-GPU Linux host.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from robotic_renderers.backends.mujoco_opengl import (
    close_renderer,
    create_renderer,
    enable_quality_flags,
)
from robotic_renderers.defaults import DEFAULT_RENDER_HOST
from robotic_renderers.media import atomic_output_path, image_digest, write_png, write_video
from robotic_renderers.registry import load_backend
from robotic_renderers.remote import validate_ssh_host
from robotic_renderers.scene import CAMERA_NAME, set_scene_time, trajectory_times

BACKEND_NAME = "cosmos3"
DEFAULT_HOST = os.environ.get("ROBOT_RENDERING_COSMOS_HOST", DEFAULT_RENDER_HOST)
DEFAULT_MODEL = "nvidia/Cosmos3-Super"
DEFAULT_MODEL_REVISION = os.environ.get(
    "ROBOT_RENDERING_COSMOS_MODEL_REVISION",
    "e0262be9d8f7586bc24c069a2aed2b665bdff266",
)
VLLM_VERSION = "0.26.0"
VLLM_OMNI_COMMIT = "39663d52872019fb0eb74bed8f1e811190486b0f"
REMOTE_ROOT = "code/robot_rendering_cosmos3"
DEFAULT_PORT = 8091


def _parse_gpu_ids(value: str) -> tuple[int, ...]:
    try:
        gpu_ids = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError("Cosmos GPU IDs must be comma-separated non-negative integers") from exc
    if len(gpu_ids) != 4:
        raise ValueError("This Cosmos3-Super preset requires exactly four selected GPUs")
    if len(set(gpu_ids)) != len(gpu_ids) or any(gpu_id < 0 for gpu_id in gpu_ids):
        raise ValueError("Cosmos GPU IDs must be unique non-negative integers")
    return gpu_ids


SERVER_GPU_IDS = _parse_gpu_ids(os.environ.get("ROBOTIC_RENDERERS_COSMOS_GPU_IDS", "0,1,2,3"))
SERVER_GPU_LIST = ",".join(map(str, SERVER_GPU_IDS))
TRANSFER_MODES = ("depth", "seg", "depth-seg")
GENERATION_MODES = ("i2v", "v2v", *TRANSFER_MODES)
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

DEFAULT_PROMPT = json.dumps(
    {
        "subject": (
            "A real Franka Research 3 seven-axis collaborative robot, bolted directly to "
            "a dark tiled studio floor. Preserve its exact white and black link geometry, "
            "joint count, proportions, cables, red Franka logos, and parallel gripper."
        ),
        "scene": (
            "In front of the robot are the same five small objects: a red cube, glossy blue "
            "sphere, gold cylinder, green torus, and faceted gold gem, at their fixed positions."
        ),
        "lighting": (
            "Physically plausible neutral studio lighting with a soft daylight key, cool fill, "
            "subtle warm rim light, realistic contact shadows, reflections, and balanced exposure."
        ),
        "camera": (
            "A full-frame cinema camera smoothly completes multiple orbits around the stationary "
            "floor-mounted robot while always looking at its center. It uses deliberate physical "
            "dolly-in and dolly-out moves plus low and high viewpoints, with a fixed focal length, "
            "no cuts, and no jitter."
        ),
        "quality": (
            "Photoreal live-action footage, physically based materials, fine surface texture, "
            "natural global illumination, sharp mechanical edges, high temporal consistency, "
            "1280x720, 24 fps."
        ),
    },
    separators=(",", ":"),
)

DEFAULT_NEGATIVE_PROMPT = (
    "CGI, 3D render, game engine, illustration, cartoon, plastic toy, low quality, low resolution, "
    "blurry, overexposed, underexposed, crushed blacks, distorted robot, changed robot design, "
    "extra joints, missing joints, duplicated links, bent rigid links, malformed gripper, floating "
    "objects, moved objects, wrong object colors, geometry drift, camera jitter, sudden camera "
    "cut, "
    "flicker, temporal inconsistency, motion smearing, text, caption, watermark, logo corruption"
)


def _run(command: list[str], **kwargs):
    return subprocess.run(command, check=True, **kwargs)


def _ssh_command(host: str, *arguments: str) -> list[str]:
    return ["ssh", *SSH_OPTIONS, validate_ssh_host(host), *arguments]


def _rsync_ssh_transport() -> str:
    return shlex.join(["ssh", *SSH_OPTIONS])


def _ssh_probe(host: str, command: str, timeout: float = 15.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        _ssh_command(host, command),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _remote_setup_script(force: bool = False) -> str:
    reset = 'rm -rf "$root/.venv"' if force else ":"
    package = f"vllm-omni @ git+https://github.com/vllm-project/vllm-omni.git@{VLLM_OMNI_COMMIT}"
    return f"""set -euo pipefail
umask 077
root="$HOME/{REMOTE_ROOT}"
install -d -m 700 "$root" "$root/.server" "$root/.jobs"
chmod 700 "$root" "$root/.server" "$root/.jobs"
cd "$root"
{reset}
uv_bin=$(command -v uv || true)
test -n "$uv_bin" || {{ echo 'uv is not installed' >&2; exit 2; }}
if [ ! -x .venv/bin/vllm ]; then
  rm -rf .venv
  "$uv_bin" venv --python 3.12 --managed-python --seed .venv
  "$uv_bin" pip install --python .venv/bin/python --torch-backend=cu130 \
    {shlex.quote(f"vllm=={VLLM_VERSION}")}
  "$uv_bin" pip install --python .venv/bin/python --torch-backend=cu130 \
    {shlex.quote(package)}
fi
.venv/bin/python - <<'PY'
import torch
import vllm
import vllm_omni
assert torch.cuda.is_available()
assert torch.cuda.device_count() > {max(SERVER_GPU_IDS)}, torch.cuda.device_count()
print(f"vLLM {{vllm.__version__}}, torch {{torch.__version__}}, GPUs {{torch.cuda.device_count()}}")
PY
"""


def setup_remote(host: str = DEFAULT_HOST, force: bool = False) -> dict[str, object]:
    completed = subprocess.run(
        _ssh_command(host, "bash -s"),
        input=_remote_setup_script(force),
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        details = (completed.stdout + "\n" + completed.stderr)[-16000:]
        raise RuntimeError(f"Cosmos 3 runtime setup failed on {host}:\n{details}")
    return {
        "backend": BACKEND_NAME,
        "host": host,
        "installed": True,
        "model": DEFAULT_MODEL,
        "model_revision": DEFAULT_MODEL_REVISION,
        "vllm": VLLM_VERSION,
        "vllm_omni_commit": VLLM_OMNI_COMMIT,
        "details": completed.stdout.strip().splitlines()[-1],
    }


def _server_probe(host: str, port: int = DEFAULT_PORT) -> subprocess.CompletedProcess:
    return _ssh_probe(
        host,
        f'root="$HOME/{REMOTE_ROOT}"; '
        'test -s "$root/.server/auth.header" || exit 4; '
        f'curl -fsS --max-time 3 --header "@$root/.server/auth.header" '
        f"http://127.0.0.1:{int(port)}/v1/models",
        timeout=8.0,
    )


def _server_configuration_probe(
    host: str,
    port: int,
    model: str,
    model_revision: str,
) -> subprocess.CompletedProcess:
    command = (
        f'set -e; root="$HOME/{REMOTE_ROOT}"; state="$root/.server"; '
        + _server_identity_shell()
        + '\ncosmos_process_matches "$state/pid" "$state/starttime"\n'
        + f'test "$(cat "$state/port")" = {shlex.quote(str(int(port)))}\n'
        + f'test "$(cat "$state/model")" = {shlex.quote(model)}\n'
        + f'test "$(cat "$state/revision")" = {shlex.quote(model_revision)}\n'
        + f'test "$(cat "$state/gpus")" = {shlex.quote(SERVER_GPU_LIST)}'
    )
    return _ssh_probe(host, command)


def backend_status(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> dict[str, object]:
    error = None
    installed = False
    gpu_count = 0
    gpu_name = None
    server_ready = False
    try:
        probe = _ssh_probe(
            host,
            f'root="$HOME/{REMOTE_ROOT}"; '
            'test -x "$root/.venv/bin/vllm" || exit 4; '
            'count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d " "); '
            "name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1); "
            'printf "%s\\n%s" "$count" "$name"',
        )
        installed = probe.returncode == 0
        if installed:
            lines = probe.stdout.splitlines()
            gpu_count = int(lines[0])
            gpu_name = lines[1] if len(lines) > 1 else None
            server_ready = (
                _server_probe(host, port).returncode == 0
                and _server_configuration_probe(
                    host, port, DEFAULT_MODEL, DEFAULT_MODEL_REVISION
                ).returncode
                == 0
            )
        else:
            error = "The uv-managed Cosmos 3 runtime is not installed"
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        error = str(exc)
    selected_gpus_available = installed and max(SERVER_GPU_IDS) < gpu_count
    if installed and not selected_gpus_available:
        selection_error = (
            f"Configured Cosmos GPUs {SERVER_GPU_LIST} are not available; "
            f"the host reports {gpu_count} GPUs"
        )
        error = f"{error}; {selection_error}" if error else selection_error
    return {
        "backend": BACKEND_NAME,
        "available": selected_gpus_available,
        "runtime_installed": installed,
        "server_ready": server_ready,
        "model": DEFAULT_MODEL,
        "model_revision": DEFAULT_MODEL_REVISION,
        "precision": "BF16",
        "device": f"{gpu_count}x {gpu_name}" if gpu_name else None,
        "gpu_ids": list(SERVER_GPU_IDS),
        "execution": f"SSH ({host})",
        "host": host,
        "port": int(port),
        "modes": list(GENERATION_MODES),
        "controls": ["depth", "segmentation"],
        "notes": (
            f"Cosmos3-Super 64B uses configured GPUs {SERVER_GPU_LIST} with HSDP and "
            "Ulysses. Select GPUs on one NUMA node when possible; topology is host-specific "
            "and is not inferred automatically. CFG branches run in lockstep because transfer "
            "control has three branches. NVIDIA's gated optional guardrail is disabled for "
            "this local robot scene."
        ),
        "error": error,
    }


def _server_identity_shell() -> str:
    """Return shell helpers that bind a PID to its Linux process start time."""
    return r"""
cosmos_proc_start() {
  candidate=$1
  test -r "/proc/$candidate/stat" || return 1
  stat_line=$(cat "/proc/$candidate/stat") || return 1
  stat_tail=${stat_line##*) }
  set -- $stat_tail
  test -n "${20:-}" || return 1
  printf '%s\n' "${20}"
}

cosmos_vllm_pid() {
  candidate=$1
  case "$candidate" in
    ""|*[!0-9]*) return 1 ;;
  esac
  test "$candidate" -gt 1 || return 1
  test -r "/proc/$candidate/cmdline" || return 1
  cmdline=$(tr '\000' ' ' < "/proc/$candidate/cmdline") || return 1
  case "$cmdline" in
    *"/vllm serve "*|*" vllm serve "*) return 0 ;;
    *) return 1 ;;
  esac
}

cosmos_pid_matches() {
  candidate=$1
  expected_start=$2
  cosmos_vllm_pid "$candidate" || return 1
  actual_start=$(cosmos_proc_start "$candidate") || return 1
  test "$actual_start" = "$expected_start"
}

cosmos_process_matches() {
  pid_file=$1
  start_file=$2
  test -s "$pid_file" && test -s "$start_file" || return 1
  read -r candidate < "$pid_file" || return 1
  read -r expected_start < "$start_file" || return 1
  cosmos_pid_matches "$candidate" "$expected_start"
}

cosmos_stop_pid() {
  candidate=$1
  expected_start=$2
  cosmos_pid_matches "$candidate" "$expected_start" || return 0
  kill -TERM -- "-$candidate" 2>/dev/null || kill -TERM "$candidate" 2>/dev/null || true
  for _ in $(seq 1 30); do
    cosmos_pid_matches "$candidate" "$expected_start" || return 0
    sleep 1
  done
  cosmos_pid_matches "$candidate" "$expected_start" || return 0
  kill -KILL -- "-$candidate" 2>/dev/null || kill -KILL "$candidate" 2>/dev/null || true
}
"""


def _remote_server_command(
    port: int,
    model: str,
    model_revision: str = DEFAULT_MODEL_REVISION,
) -> str:
    arguments = [
        ".venv/bin/vllm",
        "serve",
        model,
        "--omni",
        "--host",
        "127.0.0.1",
        "--port",
        str(int(port)),
        "--revision",
        model_revision,
        "--ulysses-degree",
        str(len(SERVER_GPU_IDS)),
        "--use-hsdp",
        "--hsdp-shard-size",
        str(len(SERVER_GPU_IDS)),
        "--init-timeout",
        "1800",
        "--no-guardrails",
    ]
    return f"""set -euo pipefail
umask 077
root="$HOME/{REMOTE_ROOT}"
state="$root/.server"
install -d -m 700 "$root" "$state" "$root/.jobs"
chmod 700 "$root" "$state" "$root/.jobs"
cd "$root"
test -x .venv/bin/vllm
{_server_identity_shell()}
exec 9>"$state/start.lock"
chmod 600 "$state/start.lock"
flock -w 60 9

if cosmos_process_matches "$state/pid" "$state/starttime"; then
  running_port=$(cat "$state/port" 2>/dev/null || true)
  running_model=$(cat "$state/model" 2>/dev/null || true)
  running_revision=$(cat "$state/revision" 2>/dev/null || true)
  running_gpus=$(cat "$state/gpus" 2>/dev/null || true)
  if [ "$running_port" = {shlex.quote(str(int(port)))} ] &&
     [ "$running_model" = {shlex.quote(model)} ] &&
     [ "$running_revision" = {shlex.quote(model_revision)} ] &&
     [ "$running_gpus" = {shlex.quote(SERVER_GPU_LIST)} ] &&
     [ -s "$state/api-key" ] &&
     [ -s "$state/auth.header" ]; then
    exit 0
  fi
  echo "A managed Cosmos server is already running with different settings" >&2
  exit 5
fi

rm -f "$state/pid" "$state/starttime" "$state/model" "$state/revision" "$state/port" \
  "$state/gpus"

# A legacy PID file has no process start time, so it cannot safely authorize a
# signal. Remove it only when it is stale; require manual migration otherwise.
if [ -s server.pid ]; then
  read -r legacy_pid < server.pid || legacy_pid=
  if cosmos_vllm_pid "$legacy_pid"; then
    echo "Refusing to replace a live server recorded without PID identity; stop it manually" >&2
    exit 7
  fi
  rm -f server.pid server.model
fi

if [ ! -s "$state/api-key" ]; then
  .venv/bin/python -c 'import secrets; print(secrets.token_urlsafe(48))' > "$state/api-key.tmp"
  chmod 600 "$state/api-key.tmp"
  mv "$state/api-key.tmp" "$state/api-key"
fi
api_key=$(cat "$state/api-key")
printf 'Authorization: Bearer %s\\n' "$api_key" > "$state/auth.header.tmp"
chmod 600 "$state/auth.header.tmp"
mv "$state/auth.header.tmp" "$state/auth.header"

export CUDA_VISIBLE_DEVICES={SERVER_GPU_LIST}
export HF_XET_HIGH_PERFORMANCE=1
export PYTHONUNBUFFERED=1
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=7200
export VLLM_OMNI_INPUT_WAIT_TIMEOUT_S=7200
export VLLM_API_KEY="$api_key"
touch server.log
chmod 600 server.log
nohup setsid {shlex.join(arguments)} > server.log 2>&1 < /dev/null &
server_pid=$!
server_start=$(cosmos_proc_start "$server_pid")
printf '%s\\n' "$server_pid" > "$state/pid.tmp"
printf '%s\\n' "$server_start" > "$state/starttime.tmp"
printf '%s\\n' {shlex.quote(model)} > "$state/model.tmp"
printf '%s\\n' {shlex.quote(model_revision)} > "$state/revision.tmp"
printf '%s\\n' {shlex.quote(str(int(port)))} > "$state/port.tmp"
printf '%s\\n' {shlex.quote(SERVER_GPU_LIST)} > "$state/gpus.tmp"
chmod 600 "$state/pid.tmp" "$state/starttime.tmp" "$state/model.tmp" \
  "$state/revision.tmp" "$state/port.tmp" "$state/gpus.tmp"
mv "$state/pid.tmp" "$state/pid"
mv "$state/starttime.tmp" "$state/starttime"
mv "$state/model.tmp" "$state/model"
mv "$state/revision.tmp" "$state/revision"
mv "$state/port.tmp" "$state/port"
mv "$state/gpus.tmp" "$state/gpus"
"""


def start_server(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    model: str = DEFAULT_MODEL,
    wait: bool = True,
    timeout: float = 2400.0,
    model_revision: str = DEFAULT_MODEL_REVISION,
) -> dict[str, object]:
    current = _server_probe(host, port)
    configured = _server_configuration_probe(host, port, model, model_revision)
    if current.returncode == 0 and configured.returncode == 0:
        return {
            "backend": BACKEND_NAME,
            "host": host,
            "port": int(port),
            "model": model,
            "model_revision": model_revision,
            "server_ready": True,
            "cached": True,
        }
    installed = backend_status(host, port)
    if not installed["runtime_installed"]:
        setup_remote(host)
    completed = subprocess.run(
        _ssh_command(host, "bash -s"),
        input=_remote_server_command(port, model, model_revision),
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError((completed.stderr or completed.stdout).strip())
    if not wait:
        return {
            "backend": BACKEND_NAME,
            "host": host,
            "port": int(port),
            "model": model,
            "model_revision": model_revision,
            "server_ready": False,
            "log": f"~/{REMOTE_ROOT}/server.log",
        }

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        probe = _server_probe(host, port)
        if probe.returncode == 0:
            return {
                "backend": BACKEND_NAME,
                "host": host,
                "port": int(port),
                "model": model,
                "model_revision": model_revision,
                "server_ready": True,
                "cached": False,
            }
        alive = _ssh_probe(
            host,
            f'root="$HOME/{REMOTE_ROOT}"; state="$root/.server"; '
            + _server_identity_shell()
            + '\ncosmos_process_matches "$state/pid" "$state/starttime"',
        )
        if alive.returncode != 0:
            log = _ssh_probe(host, f'tail -120 "$HOME/{REMOTE_ROOT}/server.log"')
            raise RuntimeError(f"Cosmos 3 server exited during startup:\n{log.stdout}{log.stderr}")
        time.sleep(10.0)
    raise TimeoutError(
        f"Cosmos 3 did not become ready on {host}:{port} within {timeout:.0f} seconds; "
        f"inspect ~/{REMOTE_ROOT}/server.log"
    )


def _remote_stop_command() -> str:
    return f"""set -euo pipefail
umask 077
root="$HOME/{REMOTE_ROOT}"
state="$root/.server"
install -d -m 700 "$root" "$state"
chmod 700 "$root" "$state"
{_server_identity_shell()}
exec 9>"$state/start.lock"
chmod 600 "$state/start.lock"
flock -w 60 9

if cosmos_process_matches "$state/pid" "$state/starttime"; then
  read -r server_pid < "$state/pid"
  read -r server_start < "$state/starttime"
  cosmos_stop_pid "$server_pid" "$server_start"
fi
rm -f "$state/pid" "$state/starttime" "$state/model" "$state/revision" "$state/port" \
  "$state/gpus"

# Never authorize a signal from a legacy PID file that lacks a start time.
if [ -s "$root/server.pid" ]; then
  read -r legacy_pid < "$root/server.pid" || legacy_pid=
  if cosmos_vllm_pid "$legacy_pid"; then
    echo "Refusing to signal a live server recorded without PID identity" >&2
  else
    rm -f "$root/server.pid" "$root/server.model"
  fi
fi
"""


def stop_server(host: str = DEFAULT_HOST) -> dict[str, object]:
    _run(_ssh_command(host, "bash -s"), input=_remote_stop_command(), text=True)
    return {"backend": BACKEND_NAME, "host": host, "server_ready": False}


def _update_scene(context: dict[str, object], time_s: float) -> None:
    set_scene_time(context["model"], context["data"], time_s)
    context["renderer"].update_scene(
        context["data"], camera=CAMERA_NAME, scene_option=context["option"]
    )
    enable_quality_flags(context["renderer"])


def _depth_control_frame(context: dict[str, object], time_s: float) -> np.ndarray:
    _update_scene(context, time_s)
    renderer = context["renderer"]
    renderer.enable_depth_rendering()
    try:
        depth = renderer.render().copy()
    finally:
        renderer.disable_depth_rendering()
    # Inverse metric depth matches the common monocular-depth convention: nearby
    # surfaces are bright and the far background is black.  Fixed bounds prevent
    # per-frame contrast pumping during the orbit.
    clipped = np.clip(np.nan_to_num(depth, nan=50.0, posinf=50.0), 1.5, 30.0)
    inverse = 1.0 / clipped
    normalized = (inverse - 1.0 / 30.0) / (1.0 / 1.5 - 1.0 / 30.0)
    gray = np.rint(np.clip(normalized, 0.0, 1.0) ** 0.7 * 255.0).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


def _segmentation_palette(model: mujoco.MjModel) -> np.ndarray:
    palette = np.zeros((model.ngeom, 3), dtype=np.uint8)
    prop_colors = {
        "floor": (96, 96, 96),
        "red_box": (220, 45, 35),
        "blue_sphere": (35, 80, 220),
        "gold_cylinder": (225, 165, 35),
        "green_torus_mesh": (35, 175, 85),
        "gold_gem_mesh": (235, 190, 55),
    }
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        palette[geom_id] = prop_colors.get(name, (75, 145, 205))
    return palette


def _segmentation_control_frame(
    context: dict[str, object], palette: np.ndarray, time_s: float
) -> np.ndarray:
    _update_scene(context, time_s)
    renderer = context["renderer"]
    renderer.enable_segmentation_rendering()
    try:
        segmentation = renderer.render().copy()
    finally:
        renderer.disable_segmentation_rendering()
    output = np.zeros((*segmentation.shape[:2], 3), dtype=np.uint8)
    geom_ids = segmentation[:, :, 0]
    geom_pixels = (segmentation[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & (geom_ids >= 0)
    output[geom_pixels] = palette[geom_ids[geom_pixels]]
    return output


def prepare_conditioning_bundle(
    destination: str | Path,
    width: int = 1280,
    height: int = 720,
    frames: int = 121,
    fps: int = 24,
    include_depth: bool = True,
    include_segmentation: bool = True,
) -> dict[str, object]:
    destination = Path(destination).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    times = list(trajectory_times(frames / fps, fps))
    outputs: dict[str, str] = {}

    if include_depth:
        context = create_renderer(width, height)
        try:
            path, count, _ = write_video(
                (_depth_control_frame(context, t) for t in times),
                destination / "depth.mp4",
                fps,
            )
        finally:
            close_renderer(context)
        if count != frames:
            raise RuntimeError(f"Depth control has {count} frames, expected {frames}")
        outputs["depth"] = str(path)

    if include_segmentation:
        context = create_renderer(width, height)
        palette = _segmentation_palette(context["model"])
        try:
            path, count, _ = write_video(
                (_segmentation_control_frame(context, palette, t) for t in times),
                destination / "segmentation.mp4",
                fps,
            )
        finally:
            close_renderer(context)
        if count != frames:
            raise RuntimeError(f"Segmentation control has {count} frames, expected {frames}")
        outputs["segmentation"] = str(path)

    manifest = {
        "scene": "fr3_renderer_evaluation",
        "width": int(width),
        "height": int(height),
        "frames": int(frames),
        "fps": int(fps),
        "controls": {name: Path(path).name for name, path in outputs.items()},
        "depth_encoding": "inverse metric depth, fixed 1.5m..30m, near=white",
        "segmentation_encoding": "black background, gray floor, blue robot, prop colors",
    }
    manifest_path = destination / "conditioning.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    manifest["controls"] = outputs
    manifest["manifest"] = str(manifest_path)
    return manifest


def _render_source_still(
    backend: str,
    output: Path,
    width: int,
    height: int,
    samples: int,
    host: str,
) -> dict[str, object]:
    module = load_backend(backend)
    if backend == "opengl":
        return module.render_still(
            output_path=output,
            width=width,
            height=height,
            time_s=0.0,
        )
    if backend == "warp":
        return module.render_still(
            output_path=output,
            width=width,
            height=height,
            time_s=0.0,
            supersample=2,
            host=host,
            local=True,
        )
    if backend == "blender":
        return module.render_still(
            output_path=output,
            width=width,
            height=height,
            time_s=0.0,
            samples=samples,
        )
    return module.render_still(
        output_path=output,
        width=width,
        height=height,
        time_s=0.0,
        samples=samples,
        host=host,
    )


def _render_source_video(
    backend: str,
    output: Path,
    width: int,
    height: int,
    frames: int,
    fps: int,
    samples: int,
    host: str,
) -> dict[str, object]:
    module = load_backend(backend)
    duration = frames / fps
    if backend == "opengl":
        return module.record_video(
            output_path=output,
            duration=duration,
            fps=fps,
            width=width,
            height=height,
        )
    if backend == "warp":
        return module.record_video(
            output_path=output,
            duration=duration,
            fps=fps,
            width=width,
            height=height,
            supersample=2,
            host=host,
            local=True,
        )
    if backend == "blender":
        return module.record_video(
            output_path=output,
            duration=duration,
            fps=fps,
            width=width,
            height=height,
            samples=samples,
            keep_frames=False,
        )
    return module.record_video(
        output_path=output,
        duration=duration,
        fps=fps,
        width=width,
        height=height,
        samples=samples,
        host=host,
    )


def _extract_video_frame(input_path: Path, output_path: Path, index: int = 0) -> Path:
    reader = imageio.get_reader(input_path)
    try:
        frame = reader.get_data(index)
    finally:
        reader.close()
    return write_png(frame, output_path)


def _one_frame_video(frame: np.ndarray, destination: Path, fps: int) -> Path:
    path, count, _ = write_video([frame], destination / "input.mp4", fps)
    if count != 1:
        raise RuntimeError(f"Transfer keyframe video has {count} frames, expected 1")
    return path


def _copy_input(input_path: Path, destination: Path, mode: str, fps: int) -> Path:
    input_path = input_path.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    destination.mkdir(parents=True, exist_ok=True)
    suffix = input_path.suffix.lower()
    image_suffixes = {".png", ".jpg", ".jpeg", ".webp"}
    video_suffixes = {".mp4", ".mov", ".m4v", ".webm"}
    if mode == "i2v" and suffix in video_suffixes:
        return _extract_video_frame(input_path, destination / "input.png")
    if mode == "i2v" and suffix not in image_suffixes:
        raise ValueError("I2V input must be an image or a video with a decodable first frame")
    if mode == "i2v":
        return write_png(imageio.imread(input_path), destination / "input.png")
    if mode in TRANSFER_MODES:
        if suffix in video_suffixes:
            keyframe_path = _extract_video_frame(input_path, destination / "source_keyframe.png")
            keyframe = imageio.imread(keyframe_path)
        elif suffix in image_suffixes:
            keyframe = imageio.imread(input_path)
            write_png(keyframe, destination / "source_keyframe.png")
        else:
            raise ValueError("Transfer input must be a rendered image or video")
        return _one_frame_video(keyframe, destination, fps)
    if suffix not in video_suffixes:
        raise ValueError(f"{mode} input must be an MP4/MOV/WebM video")
    target = destination / f"input{suffix}"
    shutil.copy2(input_path, target)
    return target


def _private_manifest_value(value: object, root: Path) -> object:
    """Remove machine-specific absolute paths from retained job metadata."""
    if isinstance(value, dict):
        return {str(key): _private_manifest_value(item, root) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_private_manifest_value(item, root) for item in value]
    if not isinstance(value, str):
        return value
    candidate = Path(value)
    if not candidate.is_absolute():
        return value
    try:
        return candidate.relative_to(root).as_posix()
    except ValueError:
        return candidate.name


def _prepare_job(
    backend: str,
    mode: str,
    input_path: str | Path | None,
    width: int,
    height: int,
    frames: int,
    fps: int,
    samples: int,
    renderer_host: str,
) -> tuple[Path, Path, dict[str, object]]:
    job = Path(tempfile.mkdtemp(prefix="robot_rendering_cosmos3_"))
    try:
        if input_path is not None:
            source = _copy_input(Path(input_path), job, mode, fps)
            source_result = {"input": str(Path(input_path).expanduser().resolve())}
        elif mode == "i2v":
            source = job / "input.png"
            source_result = _render_source_still(
                backend, source, width, height, samples, renderer_host
            )
        elif mode in TRANSFER_MODES:
            keyframe_path = job / "source_keyframe.png"
            source_result = _render_source_still(
                backend, keyframe_path, width, height, samples, renderer_host
            )
            source = _one_frame_video(imageio.imread(keyframe_path), job, fps)
        else:
            source = job / "input.mp4"
            source_result = _render_source_video(
                backend, source, width, height, frames, fps, samples, renderer_host
            )

        conditioning: dict[str, object] = {}
        if mode in TRANSFER_MODES:
            conditioning = prepare_conditioning_bundle(
                job,
                width,
                height,
                frames,
                fps,
                include_depth="depth" in mode,
                include_segmentation="seg" in mode,
            )
        manifest = {
            "backend": backend,
            "mode": mode,
            "model": DEFAULT_MODEL,
            "model_revision": DEFAULT_MODEL_REVISION,
            "source": str(source),
            "source_result": source_result,
            "conditioning": conditioning,
        }
        retained_manifest = _private_manifest_value(manifest, job)
        (job / "job.json").write_text(json.dumps(retained_manifest, indent=2, default=str) + "\n")
        return job, source, manifest
    except Exception:
        shutil.rmtree(job, ignore_errors=True)
        raise


def _remote_home(host: str) -> str:
    probe = _ssh_probe(host, 'printf "%s" "$HOME"')
    if probe.returncode != 0 or not probe.stdout.startswith("/"):
        raise RuntimeError(f"Could not determine HOME on {host}: {probe.stderr.strip()}")
    return probe.stdout.strip()


def _transfer_extra(mode: str, remote_job: str, frames: int, fps: int) -> dict[str, object]:
    extra: dict[str, object] = {
        "use_resolution_template": False,
        "use_duration_template": False,
        "guardrails": False,
        "max_frames": int(frames),
        "num_frames": int(frames),
        "fps": int(fps),
        "num_video_frames_per_chunk": int(frames),
        "num_conditional_frames": 1,
        "num_first_chunk_conditional_frames": 1,
        "resolution": "720",
        "share_vision_temporal_positions": True,
        "guidance_scale": 3.0,
        "flow_shift": 10.0,
    }
    if "depth" in mode:
        extra["depth"] = {"control_path": f"{remote_job}/depth.mp4"}
    if "seg" in mode:
        extra["seg"] = {"control_path": f"{remote_job}/segmentation.mp4"}
    extra["control_guidance"] = 2.0 if mode == "seg" else 1.5
    return extra


def _write_request_text_files(job: Path, prompt: str, negative_prompt: str) -> None:
    """Write sensitive request text to private files instead of process arguments."""
    for name, value in (("prompt.txt", prompt), ("negative_prompt.txt", negative_prompt)):
        path = job / name
        path.write_text(value, encoding="utf-8")
        path.chmod(0o600)


def _request_command(
    remote_job: str,
    input_name: str,
    mode: str,
    port: int,
    model: str,
    width: int,
    height: int,
    frames: int,
    fps: int,
    steps: int,
    guidance: float,
    flow_shift: float,
    seed: int,
) -> str:
    if mode in TRANSFER_MODES:
        extra = _transfer_extra(mode, remote_job, frames, fps)
    elif mode == "v2v":
        extra = {
            "use_resolution_template": False,
            "use_duration_template": False,
            "guardrails": False,
            "condition_frame_indexes_vision": [0, 1],
            "condition_video_keep": "first",
        }
    else:
        extra = {
            "use_resolution_template": False,
            "use_duration_template": False,
            "guardrails": False,
        }
    input_type = "image/png" if mode == "i2v" else "video/mp4"
    prompt_path = f"{remote_job}/prompt.txt"
    negative_prompt_path = f"{remote_job}/negative_prompt.txt"
    arguments = [
        "curl",
        "-sS",
        "--fail-with-body",
        "-X",
        "POST",
        f"http://127.0.0.1:{int(port)}/v1/videos/sync",
        "-H",
        "Accept: video/mp4",
        "-H",
        f"@{REMOTE_ROOT}/.server/auth.header",
        "--form-string",
        f"model={model}",
        "--form",
        f"prompt=<{prompt_path}",
        "--form",
        f"negative_prompt=<{negative_prompt_path}",
        "--form-string",
        f"size={int(width)}x{int(height)}",
        "--form-string",
        f"num_frames={int(frames)}",
        "--form-string",
        f"fps={int(fps)}",
        "--form-string",
        f"num_inference_steps={int(steps)}",
        "--form-string",
        f"guidance_scale={float(guidance)}",
        "--form-string",
        "max_sequence_length=4096",
        "--form-string",
        f"flow_shift={float(flow_shift)}",
        "--form-string",
        f"seed={int(seed)}",
        "--form-string",
        f"extra_params={json.dumps(extra, separators=(',', ':'))}",
        "-F",
        f"input_reference=@{remote_job}/{input_name};type={input_type}",
        "-o",
        f"{remote_job}/output.part.mp4",
    ]
    return (
        "set -euo pipefail\n"
        "umask 077\n"
        f'root="$HOME/{REMOTE_ROOT}"\n'
        f"chmod 700 {shlex.quote(remote_job)}\n"
        f"test -s {shlex.quote(prompt_path)}\n"
        f"test -s {shlex.quote(negative_prompt_path)}\n"
        'test -s "$root/.server/auth.header"\n'
        f'cd "$HOME"\n'
        f"rm -f {shlex.quote(remote_job + '/output.part.mp4')}\n"
        f"{shlex.join(arguments)}\n"
        f"test -s {shlex.quote(remote_job + '/output.part.mp4')}\n"
        f"mv {shlex.quote(remote_job + '/output.part.mp4')} "
        f"{shlex.quote(remote_job + '/output.mp4')}\n"
    )


def _video_result(path: Path, preview_path: Path | None) -> dict[str, object]:
    reader = imageio.get_reader(path)
    try:
        metadata = reader.get_meta_data()
        reported_frames = metadata.get("nframes", 0)
        frame_count = int(reported_frames) if np.isfinite(reported_frames) else 0
        if frame_count <= 0 or frame_count > 1_000_000:
            frame_count = int(round(float(metadata["duration"]) * float(metadata["fps"])))
        middle = reader.get_data(max(0, frame_count // 2))
    finally:
        reader.close()
    preview = write_png(middle, preview_path) if preview_path else None
    return {
        "output": str(path),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "frames": frame_count,
        "video_fps": metadata.get("fps"),
        "duration": metadata.get("duration"),
        "preview": str(preview) if preview else None,
        "preview_sha256": image_digest(middle),
    }


def enhance_render(
    output_path: str | Path,
    backend: str = "blender",
    mode: str = "i2v",
    input_path: str | Path | None = None,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    width: int = 1280,
    height: int = 720,
    frames: int | None = None,
    fps: int = 24,
    steps: int = 50,
    guidance: float | None = None,
    flow_shift: float | None = None,
    seed: int = 17,
    samples: int = 256,
    prompt: str = DEFAULT_PROMPT,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    preview_path: str | Path | None = None,
    start_if_needed: bool = True,
    keep_bundle: bool = False,
) -> dict[str, object]:
    if mode not in GENERATION_MODES:
        raise ValueError(f"Unsupported Cosmos 3 mode {mode!r}; choose from {GENERATION_MODES}")
    if (width, height) != (1280, 720):
        raise ValueError("The high-quality Cosmos 3 preset is fixed at 1280x720")
    frames = int(frames or (189 if mode in ("i2v", "v2v") else 121))
    guidance = float(guidance if guidance is not None else (3.0 if mode in TRANSFER_MODES else 6.0))
    flow_shift = float(flow_shift if flow_shift is not None else 10.0)
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    preview = Path(preview_path).expanduser().resolve() if preview_path else None
    if preview:
        preview.parent.mkdir(parents=True, exist_ok=True)

    if start_if_needed:
        start_server(host, port, DEFAULT_MODEL, wait=True)
    elif _server_probe(host, port).returncode != 0:
        raise RuntimeError(f"Cosmos 3 server is not ready on {host}:{port}")

    job, source, manifest = _prepare_job(
        backend, mode, input_path, width, height, frames, fps, samples, host
    )
    remote_job = None
    started = time.perf_counter()
    try:
        job_id = uuid.uuid4().hex
        remote_job_rel = f"{REMOTE_ROOT}/.jobs/{job_id}"
        remote_job = f"{_remote_home(host)}/{remote_job_rel}"
        _write_request_text_files(job, prompt, negative_prompt)
        _run(
            _ssh_command(
                host,
                f"umask 077; install -d -m 700 {shlex.quote(remote_job)}",
            )
        )
        _run(
            [
                "rsync",
                "-az",
                "-e",
                _rsync_ssh_transport(),
                f"{job}/",
                f"{host}:{remote_job}/",
            ]
        )
        command = _request_command(
            remote_job,
            source.name,
            mode,
            port,
            DEFAULT_MODEL,
            width,
            height,
            frames,
            fps,
            steps,
            guidance,
            flow_shift,
            seed,
        )
        completed = subprocess.run(
            _ssh_command(host, "bash -s"),
            input=command,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            error_body = _ssh_probe(
                host,
                f"test -f {shlex.quote(remote_job + '/output.part.mp4')} && "
                f"tail -c 12000 {shlex.quote(remote_job + '/output.part.mp4')} || true",
            ).stdout
            raise RuntimeError(
                "Cosmos 3 generation failed:\n"
                + (completed.stderr or completed.stdout)
                + (f"\nServer response:\n{error_body}" if error_body else "")
            )
        with atomic_output_path(output) as (temporary, output):
            _run(
                [
                    "rsync",
                    "-az",
                    "-e",
                    _rsync_ssh_transport(),
                    f"{validate_ssh_host(host)}:{remote_job}/output.mp4",
                    str(temporary),
                ]
            )
        result = _video_result(output, preview)
        return {
            "backend": BACKEND_NAME,
            "source_backend": backend,
            "mode": mode,
            "host": host,
            "model": DEFAULT_MODEL,
            "model_revision": DEFAULT_MODEL_REVISION,
            "model_parameters": "64B",
            "precision": "BF16",
            "width": width,
            "height": height,
            "requested_frames": frames,
            "requested_fps": fps,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "flow_shift": flow_shift,
            "seed": seed,
            "seconds": time.perf_counter() - started,
            "conditioning": list(manifest.get("conditioning", {}).get("controls", {})),
            **result,
        }
    finally:
        if remote_job is not None:
            _ssh_probe(host, f"rm -rf {shlex.quote(remote_job)}", timeout=30.0)
        if keep_bundle:
            bundle_destination = output.with_suffix("").with_name(output.stem + "_conditioning")
            if bundle_destination.exists():
                shutil.rmtree(bundle_destination)
            shutil.move(str(job), bundle_destination)
        else:
            shutil.rmtree(job, ignore_errors=True)
