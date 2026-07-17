"""MuJoCo Warp ray-renderer backend for macOS CPU, with optional SSH CUDA."""

from __future__ import annotations

import argparse
import json
import os
import shlex
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

from render_common import display_frames, image_digest, resize_rgb, write_png, write_video
from scene import create_model_and_data, set_scene_time, trajectory_times

BACKEND_NAME = "mujoco_warp"
WARP_EXPOSURE_SCALE = 0.42
DEFAULT_HOST = os.environ.get("ROBOT_RENDERING_WARP_HOST", "l4")
ROOT = Path(__file__).resolve().parent
REMOTE_ROOT = ".cache/robot_rendering"
STREAM_MAGIC = b"ROBOT_RENDERING_RGB_STREAM_V1\n"
RESULT_MAGIC = "ROBOT_RENDERING_RESULT="
_SYNCED_HOSTS: set[str] = set()


def _run(command: list[str], **kwargs):
    return subprocess.run(command, check=True, **kwargs)


def _remote_shell_command(arguments: list[str]) -> str:
    worker = shlex.join(
        [
            "$HOME/.local/bin/uv",
            "run",
            "--frozen",
            "--no-dev",
            "python",
            "render_mujoco_warp.py",
            "__worker",
            *arguments,
        ]
    )
    # shlex quotes $HOME above, so restore only the known executable token.
    worker = worker.replace("'$HOME/.local/bin/uv'", '"$HOME/.local/bin/uv"', 1)
    return f'cd "$HOME/{REMOTE_ROOT}" && {worker}'


def backend_status(host: str = DEFAULT_HOST) -> dict[str, object]:
    local_device = f"CPU ({os.uname().machine})"
    return {
        "backend": BACKEND_NAME,
        "available": True,
        "version": getattr(mjw, "__version__", "3.10.0.1"),
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
    if host in _SYNCED_HOSTS and not force:
        return {"host": host, "synced": True, "cached": True}

    uv_probe = subprocess.run(
        ["ssh", host, 'test -x "$HOME/.local/bin/uv"'],
        capture_output=True,
        text=True,
    )
    if uv_probe.returncode != 0:
        raise RuntimeError(
            f"uv is not installed on {host}. Install it with: "
            f"ssh {host} 'curl -LsSf https://astral.sh/uv/install.sh | sh'"
        )

    _run(["ssh", host, f'mkdir -p "$HOME/{REMOTE_ROOT}"'])
    rsync_command = [
        "rsync",
        "-az",
        "--delete",
        "--exclude=.venv",
        "--exclude=.git",
        "--exclude=.jobs",
        "--exclude=outputs",
        "--exclude=build",
        "--exclude=__pycache__",
        "--exclude=*.pyc",
        "--exclude=.pytest_cache",
        "--exclude=.ruff_cache",
        f"{ROOT}/",
        f"{host}:~/{REMOTE_ROOT}/",
    ]
    _run(rsync_command)
    sync_command = f'cd "$HOME/{REMOTE_ROOT}" && "$HOME/.local/bin/uv" sync --frozen --no-dev'
    _run(["ssh", host, sync_command])
    _SYNCED_HOSTS.add(host)
    return {"host": host, "synced": True, "cached": False}


def create_renderer(
    width: int = 1280,
    height: int = 720,
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
    width: int = 1280,
    height: int = 720,
    time_s: float = 1.0,
    supersample: int = 2,
    device: str | None = None,
) -> np.ndarray:
    context = create_renderer(width, height, supersample=supersample, device=device)
    return render_context_frame(context, time_s=time_s)


def render_frame(
    width: int = 1280,
    height: int = 720,
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
    width: int = 1280,
    height: int = 720,
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


def _finite_frames(context: dict[str, object], duration: float, fps: int):
    for time_s in trajectory_times(duration, fps):
        yield render_context_frame(context, time_s)


def record_video_local(
    output_path: str | Path,
    duration: float = 5.0,
    fps: int = 30,
    width: int = 1280,
    height: int = 720,
    supersample: int = 2,
    device: str | None = None,
) -> dict[str, object]:
    context = create_renderer(width, height, supersample=supersample, device=device)
    output, frame_count, seconds = write_video(
        _finite_frames(context, duration=duration, fps=fps),
        output_path,
        fps=fps,
    )
    return {
        "backend": BACKEND_NAME,
        "output": str(output),
        "frames": frame_count,
        "seconds": seconds,
        "fps": frame_count / max(seconds, 1e-9),
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
        ["ssh", host, _remote_shell_command(arguments)],
        capture_output=True,
        text=True,
    )
    return _parse_remote_result(completed.stdout)


def _copy_remote_output(host: str, remote_path: str, output_path: str | Path) -> Path:
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    _run(["scp", f"{host}:~/{REMOTE_ROOT}/{remote_path}", str(output)])
    return output


def _cleanup_remote_job(host: str, job_id: str) -> None:
    subprocess.run(
        ["ssh", host, f'rm -rf "$HOME/{REMOTE_ROOT}/.jobs/{job_id}"'],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def render_still(
    output_path: str | Path,
    width: int = 1280,
    height: int = 720,
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
    duration: float = 5.0,
    fps: int = 30,
    width: int = 1280,
    height: int = 720,
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


def _read_exact(stream, byte_count: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < byte_count:
        chunk = stream.read(byte_count - len(chunks))
        if not chunk:
            raise EOFError("Remote RGB stream closed unexpectedly")
        chunks.extend(chunk)
    return bytes(chunks)


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
        ["ssh", host, _remote_shell_command(arguments)],
        stdout=subprocess.PIPE,
        stderr=None,
        bufsize=0,
    )
    assert process.stdout is not None
    startup_lines: list[bytes] = []
    try:
        while True:
            line = process.stdout.readline()
            if not line:
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
                length_bytes += _read_exact(process.stdout, 4 - len(length_bytes))
            frame_length = struct.unpack("!I", length_bytes)[0]
            if frame_length != expected:
                raise RuntimeError(f"Remote stream sent {frame_length} bytes; expected {expected}")
            frame = np.frombuffer(_read_exact(process.stdout, frame_length), dtype=np.uint8)
            yield frame.reshape(height, width, 3).copy()
    finally:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()


def render_live(
    width: int = 960,
    height: int = 540,
    fps: int = 30,
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
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--time", type=float, default=1.0)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--fps", type=int, default=30)
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
    frame_index = 0
    try:
        while args.duration is None or frame_index / args.fps < args.duration:
            frame = render_context_frame(context, frame_index / args.fps)
            payload = frame.tobytes()
            sys.stdout.buffer.write(struct.pack("!I", len(payload)))
            sys.stdout.buffer.write(payload)
            sys.stdout.buffer.flush()
            frame_index += 1
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
    raise SystemExit("Use main.py; __worker is reserved for the SSH runner.")
