"""Maximize and compare steady-state renderer throughput on a remote GPU host.

The public entry point runs from macOS and coordinates workers on an SSH host. Internal
worker modes deliberately keep heavyweight renderer imports out of the parent process so
GPU selection environment variables are applied before EGL, CUDA, Warp, Blender, or ovrtx
are initialized.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import platform
import queue as queue_module
import shlex
import shutil
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REMOTE_ROOT = ".cache/robot_rendering"
OVRTX_REMOTE_ROOT = "code/robot_rendering_ovrtx"
RESULT_MAGIC = "ROBOT_RENDERING_FPS_RESULT="
BLENDER_VERSION = "5.1.2"
BLENDER_ARCHIVE = f"blender-{BLENDER_VERSION}-linux-x64.tar.xz"
BLENDER_URL = (
    "https://mirror.clarkson.edu/blender/release/Blender5.1/" + BLENDER_ARCHIVE
)
BLENDER_SHA256 = "aaccb355f50183979b698bcce7467103a76261b5fa59f4972295842662a285fb"
BLENDER_REMOTE = f".cache/blender/blender-{BLENDER_VERSION}-linux-x64/blender"
BACKENDS = ("opengl", "warp", "blender", "ovrtx")


def _run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(command, check=True, **kwargs)


def _parse_positive_ints(value: str) -> list[int]:
    values = []
    for token in value.split(","):
        number = int(token.strip())
        if number < 1:
            raise argparse.ArgumentTypeError("values must be positive integers")
        if number not in values:
            values.append(number)
    if not values:
        raise argparse.ArgumentTypeError("at least one value is required")
    return values


def _parse_gpus(value: str) -> list[int]:
    values = _parse_positive_ints(
        ",".join(str(int(token.strip()) + 1) for token in value.split(","))
    )
    return [value - 1 for value in values]


def _parse_auto_positive_int(value: str) -> int | None:
    if value.strip().lower() == "auto":
        return None
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer or 'auto'")
    return number


def _emit_result(result: dict[str, object]) -> None:
    print(RESULT_MAGIC + json.dumps(result, sort_keys=True), flush=True)


def _extract_result(output: str) -> dict[str, object]:
    for line in reversed(output.splitlines()):
        if line.startswith(RESULT_MAGIC):
            return json.loads(line[len(RESULT_MAGIC) :])
    raise RuntimeError(f"Remote worker returned no result marker:\n{output[-8000:]}")


def _remote_uv_command(arguments: list[str]) -> str:
    command = [
        "$HOME/.local/bin/uv",
        "run",
        "--frozen",
        "--no-dev",
        "python",
        "benchmark_fps.py",
        *arguments,
    ]
    rendered = shlex.join(command).replace("'$HOME/.local/bin/uv'", '"$HOME/.local/bin/uv"')
    return f'cd "$HOME/{REMOTE_ROOT}" && {rendered}'


def _remote_ovrtx_command(arguments: list[str]) -> str:
    command = [".venv/bin/python", "benchmark_fps.py", *arguments]
    return f'cd "$HOME/{OVRTX_REMOTE_ROOT}" && {shlex.join(command)}'


def _remote_result(host: str, command: str, timeout: float | None = None) -> dict[str, object]:
    completed = subprocess.run(
        ["ssh", host, command],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    combined = completed.stdout + "\n" + completed.stderr
    if completed.returncode != 0:
        raise RuntimeError(
            f"Remote command failed with exit {completed.returncode}:\n{combined[-12000:]}"
        )
    return _extract_result(combined)


def _setup_blender(host: str) -> dict[str, object]:
    script = f'''set -euo pipefail
root="$HOME/.cache/blender"
archive="$root/{BLENDER_ARCHIVE}"
binary="$root/blender-{BLENDER_VERSION}-linux-x64/blender"
mkdir -p "$root"
if [ ! -x "$binary" ]; then
  if [ ! -s "$archive" ]; then
    curl -L --fail --retry 5 --retry-delay 3 -o "$archive.part" {shlex.quote(BLENDER_URL)}
    mv "$archive.part" "$archive"
  fi
  printf '%s  %s\n' {BLENDER_SHA256} "$archive" | sha256sum -c -
  tar -xf "$archive" -C "$root"
  rm -f "$archive"
fi
"$binary" --version | head -1
'''
    completed = subprocess.run(
        ["ssh", host, "bash -s"], input=script, capture_output=True, text=True
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Blender setup failed on {host}:\n{completed.stderr[-8000:]}")
    return {"binary": f"~/{BLENDER_REMOTE}", "version": completed.stdout.strip()}


def setup_remote(host: str) -> dict[str, object]:
    import render_mujoco_warp
    import render_ovrtx

    warp = render_mujoco_warp.setup_remote(host, force=True)
    ovrtx = render_ovrtx.setup_remote(host)
    blender = _setup_blender(host)
    _run(
        [
            "rsync",
            "-az",
            str(ROOT / "benchmark_fps.py"),
            f"{host}:~/{OVRTX_REMOTE_ROOT}/benchmark_fps.py",
        ]
    )
    return {"warp_project": warp, "ovrtx": ovrtx, "blender": blender}


def _hardware_info() -> dict[str, object]:
    gpu_query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    gpus = []
    for line in gpu_query.stdout.splitlines():
        index, name, driver, memory, capability = [part.strip() for part in line.split(",")]
        gpus.append(
            {
                "index": int(index),
                "name": name,
                "driver": driver,
                "memory_mib": int(memory),
                "compute_capability": capability,
            }
        )
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "logical_cpus": os.cpu_count(),
        "gpus": gpus,
    }


def _opengl_process(
    worker_index: int,
    gpu: int,
    width: int,
    height: int,
    frames: int,
    warmup: int,
    fps: int,
    start_event,
    queue,
) -> None:
    try:
        os.environ["MUJOCO_GL"] = "egl"
        os.environ["PYOPENGL_PLATFORM"] = "egl"
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(gpu)
        import numpy as np

        import render_mujoco_opengl as backend

        context = backend.create_renderer(width, height)
        try:
            last = None
            for index in range(warmup):
                last = backend.render_context_frame(context, index / fps)
            queue.put({"ready": True, "worker": worker_index, "gpu": gpu})
            if not start_event.wait(timeout=600):
                raise TimeoutError("OpenGL start barrier timed out")
            started = time.perf_counter()
            for index in range(frames):
                last = backend.render_context_frame(context, index / fps)
            seconds = time.perf_counter() - started
            queue.put(
                {
                    "worker": worker_index,
                    "gpu": gpu,
                    "seconds": seconds,
                    "checksum": int(np.asarray(last, dtype=np.uint64).sum() % (2**32)),
                }
            )
        finally:
            backend.close_renderer(context)
    except BaseException as exc:
        queue.put(
            {
                "worker": worker_index,
                "gpu": gpu,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )


def _collect_ready(processes, queue, expected: int, label: str) -> None:
    ready = 0
    deadline = time.monotonic() + 900
    while ready < expected:
        timeout = max(0.1, min(10.0, deadline - time.monotonic()))
        if deadline <= time.monotonic():
            raise TimeoutError(f"{label} workers did not reach the warmup barrier")
        try:
            message = queue.get(timeout=timeout)
        except queue_module.Empty:
            failed = [
                process.pid
                for process in processes
                if process.exitcode not in (None, 0)
            ]
            if failed:
                raise RuntimeError(
                    f"{label} workers exited during initialization: {failed}"
                ) from None
            continue
        if message.get("error"):
            raise RuntimeError(message.get("traceback") or message["error"])
        if message.get("ready"):
            ready += 1
    failed = [process.pid for process in processes if process.exitcode not in (None, 0)]
    if failed:
        raise RuntimeError(f"{label} workers exited during initialization: {failed}")


def _finish_process_trial(processes, queue, expected: int, label: str) -> list[dict[str, object]]:
    results = []
    deadline = time.monotonic() + 1800
    while len(results) < expected:
        if deadline <= time.monotonic():
            raise TimeoutError(f"{label} workers did not complete")
        try:
            message = queue.get(
                timeout=min(10.0, max(0.1, deadline - time.monotonic()))
            )
        except queue_module.Empty:
            failed = [
                process.pid
                for process in processes
                if process.exitcode not in (None, 0)
            ]
            if failed:
                raise RuntimeError(
                    f"{label} workers exited before returning results: {failed}"
                ) from None
            continue
        if message.get("error"):
            raise RuntimeError(message.get("traceback") or message["error"])
        if not message.get("ready"):
            results.append(message)
    for process in processes:
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        if process.exitcode != 0:
            raise RuntimeError(f"{label} worker {process.pid} exited with {process.exitcode}")
    return results


def _opengl_trial(
    workers: int,
    gpus: list[int],
    width: int,
    height: int,
    frames: int,
    warmup: int,
    fps: int,
) -> dict[str, object]:
    context = multiprocessing.get_context("spawn")
    start_event = context.Event()
    queue = context.Queue()
    processes = []
    for worker_index in range(workers):
        gpu = gpus[worker_index % len(gpus)]
        process = context.Process(
            target=_opengl_process,
            args=(
                worker_index,
                gpu,
                width,
                height,
                frames,
                warmup,
                fps,
                start_event,
                queue,
            ),
        )
        process.start()
        processes.append(process)
    try:
        _collect_ready(processes, queue, workers, "OpenGL")
        started = time.perf_counter()
        start_event.set()
        results = _finish_process_trial(processes, queue, workers, "OpenGL")
        wall_seconds = time.perf_counter() - started
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
    total_frames = workers * frames
    return {
        "workers": workers,
        "gpus_used": sorted({int(result["gpu"]) for result in results}),
        "frames_per_worker": frames,
        "total_frames": total_frames,
        "wall_seconds": wall_seconds,
        "render_fps": total_frames / max(wall_seconds, 1e-9),
        "rgb_fps": total_frames / max(wall_seconds, 1e-9),
        "throughput_fps": total_frames / max(wall_seconds, 1e-9),
        "worker_seconds": [result["seconds"] for result in results],
        "checksums": [result["checksum"] for result in results],
    }


def _create_warp_context(width: int, height: int, batch: int):
    import mujoco
    import mujoco_warp as mjw
    import numpy as np
    import warp as wp

    from scene import create_model_and_data

    wp.init()
    wp.set_device("cuda:0")
    model, host_data = create_model_and_data(width=width, height=height)
    warp_model = mjw.put_model(model)
    floor_material_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_MATERIAL, "floor_material"
    )
    if floor_material_id >= 0:
        repeats = warp_model.mat_texrepeat.numpy()
        repeats[:, floor_material_id] *= 0.5
        warp_model.mat_texrepeat.assign(repeats)
    warp_data = mjw.put_data(model, host_data, nworld=batch)
    render_context = mjw.create_render_context(
        model,
        nworld=batch,
        cam_res=(width, height),
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
    return {
        "model": model,
        "host_data": host_data,
        "warp_model": warp_model,
        "warp_data": warp_data,
        "render_context": render_context,
        "rgb": wp.empty((batch, height, width), dtype=wp.vec3),
        "batch": batch,
        "np": np,
        "mjw": mjw,
        "wp": wp,
    }


def _warp_render(context: dict[str, object], time_s: float) -> None:
    from scene import set_scene_time

    np = context["np"]
    mjw = context["mjw"]
    model = context["model"]
    host_data = context["host_data"]
    warp_data = context["warp_data"]
    batch = context["batch"]
    set_scene_time(model, host_data, time_s)
    warp_data.geom_xpos.assign(np.repeat(host_data.geom_xpos[None], batch, axis=0))
    warp_data.geom_xmat.assign(
        np.repeat(host_data.geom_xmat.reshape(1, model.ngeom, 3, 3), batch, axis=0)
    )
    warp_data.cam_xpos.assign(np.repeat(host_data.cam_xpos[None], batch, axis=0))
    warp_data.cam_xmat.assign(
        np.repeat(host_data.cam_xmat.reshape(1, model.ncam, 3, 3), batch, axis=0)
    )
    mjw.refit_bvh(context["warp_model"], warp_data, context["render_context"])
    mjw.render(context["warp_model"], warp_data, context["render_context"])
    mjw.get_rgb(context["render_context"], 0, context["rgb"])
    context["wp"].synchronize()


def _warp_readback(context: dict[str, object]) -> int:
    np = context["np"]
    pixels = context["rgb"].numpy()
    corrected = np.rint(np.clip(pixels * 0.42, 0.0, 1.0) * 255.0).astype(np.uint8)
    return int(corrected[:, 0, 0].astype(np.uint64).sum() % (2**32))


def _warp_process(
    gpu: int,
    width: int,
    height: int,
    batch: int,
    steps: int,
    warmup: int,
    fps: int,
    start_event,
    queue,
) -> None:
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        context = _create_warp_context(width, height, batch)
        for index in range(warmup):
            _warp_render(context, index / fps)
            _warp_readback(context)
        queue.put({"ready": True, "gpu": gpu})
        if not start_event.wait(timeout=600):
            raise TimeoutError("Warp start barrier timed out")
        render_seconds = 0.0
        rgb_seconds = 0.0
        checksum = 0
        for index in range(steps):
            started = time.perf_counter()
            _warp_render(context, index / fps)
            render_seconds += time.perf_counter() - started
            started = time.perf_counter()
            checksum ^= _warp_readback(context)
            rgb_seconds += time.perf_counter() - started
        queue.put(
            {
                "gpu": gpu,
                "render_seconds": render_seconds,
                "rgb_seconds": rgb_seconds,
                "checksum": checksum,
            }
        )
    except BaseException as exc:
        queue.put(
            {
                "gpu": gpu,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )


def _warp_trial(
    batch: int,
    gpus: list[int],
    width: int,
    height: int,
    steps: int,
    warmup: int,
    fps: int,
) -> dict[str, object]:
    context = multiprocessing.get_context("spawn")
    start_event = context.Event()
    queue = context.Queue()
    processes = []
    for gpu in gpus:
        process = context.Process(
            target=_warp_process,
            args=(gpu, width, height, batch, steps, warmup, fps, start_event, queue),
        )
        process.start()
        processes.append(process)
    try:
        _collect_ready(processes, queue, len(processes), "Warp")
        start_event.set()
        results = _finish_process_trial(processes, queue, len(processes), "Warp")
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
    frames = batch * steps * len(gpus)
    render_seconds = max(float(result["render_seconds"]) for result in results)
    rgb_seconds = max(float(result["rgb_seconds"]) for result in results)
    total_seconds = max(
        float(result["render_seconds"]) + float(result["rgb_seconds"])
        for result in results
    )
    return {
        "batch_per_gpu": batch,
        "gpu_processes": len(gpus),
        "total_environments": batch * len(gpus),
        "steps": steps,
        "total_frames": frames,
        "render_seconds": render_seconds,
        "rgb_seconds": rgb_seconds,
        "render_fps": frames / max(render_seconds, 1e-9),
        "rgb_fps": frames / max(total_seconds, 1e-9),
        "throughput_fps": frames / max(total_seconds, 1e-9),
        "checksums": [result["checksum"] for result in results],
        "gpus_used": gpus,
    }


def _blender_worker(arguments: list[str]) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--ready", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--samples", type=int, required=True)
    args = parser.parse_args(arguments)

    import bpy

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import render_blender

    manifest = render_blender._worker_build_scene(
        Path(args.manifest).resolve(), args.samples, live=False
    )
    frames = [frame["frame"] for frame in manifest["frames"]]
    for index in range(args.warmup):
        bpy.context.scene.frame_set(frames[index % len(frames)])
        bpy.ops.render.render(write_still=False)
    Path(args.ready).touch()
    deadline = time.monotonic() + 900
    while not Path(args.start).exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("Blender start barrier timed out")
        time.sleep(0.05)
    started = time.perf_counter()
    frame_seconds = []
    for index in range(args.frames):
        bpy.context.scene.frame_set(frames[index % len(frames)])
        frame_started = time.perf_counter()
        bpy.ops.render.render(write_still=False)
        frame_seconds.append(time.perf_counter() - frame_started)
    seconds = time.perf_counter() - started
    result = {
        "seconds": seconds,
        "frame_seconds": frame_seconds,
        "frames": args.frames,
        "device": bpy.context.scene.get("robot_rendering_cycles_device", "unknown"),
        "devices": list(
            bpy.context.scene.get("robot_rendering_cycles_devices", ["unknown"])
        ),
        "result_size": [
            bpy.context.scene.render.resolution_x,
            bpy.context.scene.render.resolution_y,
        ],
        "threads": bpy.context.scene.render.threads,
        "persistent_data": bpy.context.scene.render.use_persistent_data,
        "denoiser": (
            bpy.context.scene.cycles.denoiser
            if bpy.context.scene.cycles.use_denoising
            else "NONE"
        ),
        "max_bounces": bpy.context.scene.cycles.max_bounces,
    }
    Path(args.result).write_text(json.dumps(result))
    return 0


def _gpu_numa_nodes(gpus: list[int]) -> dict[int, int]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,pci.bus_id",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    nodes = {}
    for line in completed.stdout.splitlines():
        index_text, bus_id = (part.strip() for part in line.split(",", 1))
        index = int(index_text)
        if index not in gpus:
            continue
        domain, bus, device = bus_id.lower().split(":")
        sysfs_path = Path("/sys/bus/pci/devices") / f"{domain[-4:]}:{bus}:{device}"
        try:
            node = int((sysfs_path / "numa_node").read_text().strip())
        except (OSError, ValueError):
            continue
        if node >= 0:
            nodes[index] = node
    return nodes


def _blender_trial(
    blender: Path,
    manifest: Path,
    workdir: Path,
    workers: int,
    gpus: list[int],
    frames: int,
    warmup: int,
    samples: int,
    gpus_per_worker: int,
    cpu_threads: int,
    denoiser: str,
    persistent_data: bool,
    max_bounces: int,
    numa_bind: bool,
) -> dict[str, object]:
    if gpus_per_worker > len(gpus):
        raise ValueError(
            f"gpus_per_worker={gpus_per_worker} exceeds {len(gpus)} selected GPUs"
        )
    trial = workdir / (
        f"blender_w{workers}_g{gpus_per_worker}_t{cpu_threads}_"
        f"{denoiser.lower()}_p{int(persistent_data)}_b{max_bounces}"
    )
    shutil.rmtree(trial, ignore_errors=True)
    trial.mkdir(parents=True)
    start_path = trial / "start"
    processes = []
    streams = []
    worker_gpu_groups = []
    numa_bound_workers = []
    gpu_numa_nodes = _gpu_numa_nodes(gpus) if numa_bind else {}
    for worker in range(workers):
        result_path = trial / f"result_{worker}.json"
        ready_path = trial / f"ready_{worker}"
        log_path = trial / f"worker_{worker}.log"
        worker_gpus = [
            gpus[(worker * gpus_per_worker + offset) % len(gpus)]
            for offset in range(gpus_per_worker)
        ]
        worker_gpu_groups.append(worker_gpus)
        command = [
            str(blender),
            "--background",
            "--python",
            str(Path(__file__).resolve()),
            "--",
            "__blender_worker",
            "--manifest",
            str(manifest),
            "--result",
            str(result_path),
            "--ready",
            str(ready_path),
            "--start",
            str(start_path),
            "--frames",
            str(frames),
            "--warmup",
            str(warmup),
            "--samples",
            str(samples),
        ]
        worker_nodes = {gpu_numa_nodes.get(gpu) for gpu in worker_gpus}
        if None not in worker_nodes and len(worker_nodes) == 1 and shutil.which("numactl"):
            node = worker_nodes.pop()
            command = [
                "numactl",
                f"--cpunodebind={node}",
                f"--membind={node}",
                *command,
            ]
            numa_bound_workers.append(worker)
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, worker_gpus))
        environment["BLENDER_CYCLES_DEVICE"] = "OPTIX"
        environment["BLENDER_CPU_THREADS"] = str(cpu_threads)
        environment["BLENDER_DENOISER"] = denoiser
        environment["BLENDER_PERSISTENT_DATA"] = "1" if persistent_data else "0"
        environment["BLENDER_MAX_BOUNCES"] = str(max_bounces)
        environment["OMP_NUM_THREADS"] = str(cpu_threads)
        environment["OPENBLAS_NUM_THREADS"] = str(cpu_threads)
        stream = log_path.open("wb")
        streams.append(stream)
        processes.append(
            subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=environment)
        )
    try:
        deadline = time.monotonic() + 1200
        while True:
            ready_count = len(list(trial.glob("ready_*")))
            if ready_count == workers:
                break
            failures = [
                process.returncode
                for process in processes
                if process.poll() not in (None, 0)
            ]
            if failures:
                logs = "\n".join(
                    path.read_text(errors="replace")[-4000:]
                    for path in trial.glob("*.log")
                )
                raise RuntimeError(f"Blender worker failed during warmup:\n{logs}")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Only {ready_count}/{workers} Blender workers became ready")
            time.sleep(0.25)
        started = time.perf_counter()
        start_path.touch()
        return_codes = [process.wait(timeout=1800) for process in processes]
        wall_seconds = time.perf_counter() - started
        if any(return_codes):
            logs = "\n".join(
                path.read_text(errors="replace")[-4000:]
                for path in trial.glob("*.log")
            )
            raise RuntimeError(f"Blender workers failed with {return_codes}:\n{logs}")
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for stream in streams:
            stream.close()
    results = [
        json.loads((trial / f"result_{worker}.json").read_text())
        for worker in range(workers)
    ]
    total_frames = workers * frames
    return {
        "workers": workers,
        "gpus_per_worker": gpus_per_worker,
        "worker_gpu_groups": worker_gpu_groups,
        "gpus_used": sorted({gpu for group in worker_gpu_groups for gpu in group}),
        "cpu_threads_per_worker": cpu_threads,
        "numa_requested": numa_bind,
        "numa_bound": len(numa_bound_workers) == workers,
        "persistent_data": persistent_data,
        "denoiser": denoiser,
        "max_bounces": max_bounces,
        "frames_per_worker": frames,
        "total_frames": total_frames,
        "wall_seconds": wall_seconds,
        "render_fps": total_frames / max(wall_seconds, 1e-9),
        "rgb_fps": None,
        "throughput_fps": total_frames / max(wall_seconds, 1e-9),
        "worker_seconds": [result["seconds"] for result in results],
        "frame_seconds": [result["frame_seconds"] for result in results],
        "devices": sorted({result["device"] for result in results}),
        "device_names": [result["devices"] for result in results],
        "result_sizes": [result["result_size"] for result in results],
    }


def _write_ovrtx_multi_env_stage(bundle_path: Path, environments: int) -> tuple[Path, str]:
    bundle = json.loads(bundle_path.read_text())
    columns = math.ceil(math.sqrt(environments))
    rows = math.ceil(environments / columns)
    if columns * rows != environments:
        raise ValueError("ovrtx batch sizes must form a complete rectangular tile grid")
    references = []
    cameras = []
    for index in range(environments):
        name = f"Environment_{index:04d}"
        references.append(
            f'''    def Xform "{name}" (
        prepend references = @./{bundle["scene"]}@
    ) {{
        double3 xformOp:translate = ({index * 1000}, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }}'''
        )
        cameras.append(f"</World/{name}/Camera>")
    camera_targets = ",\n        ".join(cameras)
    output_width = int(bundle["width"]) * columns
    output_height = int(bundle["height"]) * rows
    stage = f'''#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
    startTimeCode = 1
    endTimeCode = {bundle["frames"]}
    timeCodesPerSecond = {bundle["fps"]}
)

def Xform "World" {{
{chr(10).join(references)}
}}

def Scope "Render" {{
    def RenderProduct "MultiEnvironment" {{
        int2 resolution = ({output_width}, {output_height})
        rel camera = [
        {camera_targets}
        ]
        rel orderedVars = [<LdrColor>]
        token omni:rtx:background:source:type = "domeLight"
        token omni:rtx:rendermode = "PathTracing"
        token[] omni:rtx:waitForEvents = ["AllLoadingFinished", "OnlyOnFirstRequest"]
        int omni:rtx:pt:samplesPerPixel = {bundle["samples"]}
        int omni:rtx:pt:samplesPerIteration = 4
        bool omni:rtx:pt:adaptiveSampling:enabled = true
        bool omni:rtx:pt:denoising:enabled = true
        bool omni:rtx:pt:denoising:optix:temporal = false
        int omni:rtx:pt:limits:maxBounces = 12
        int omni:rtx:pt:limits:maxGlossyBounces = 8
        uint[] deviceIds = [0]
        def RenderVar "LdrColor" {{
            string sourceName = "LdrColor"
        }}
    }}
}}
'''
    stage_path = bundle_path.parent / f"multi_environment_{environments}.usda"
    stage_path.write_text(stage)
    return stage_path, "/Render/MultiEnvironment"


def _map_ovrtx_product(product) -> tuple[list[list[int]], int]:
    import numpy as np
    import ovrtx

    shapes = []
    checksum = 0
    for frame in product.frames:
        mapping = frame.render_vars["LdrColor"].map(device=ovrtx.Device.CPU)
        try:
            view = np.from_dlpack(mapping)
            copied = np.ascontiguousarray(view[:, :, :3]).copy()
            shapes.append(list(copied.shape))
            checksum ^= int(copied[0, 0].astype(np.uint64).sum())
            del view
        finally:
            mapping.unmap()
    return shapes, checksum


def _ovrtx_process(
    bundle_path: Path,
    environments: int,
    stage_path: Path,
    render_product: str,
    gpu: int,
    steps: int,
    warmup: int,
    start_event,
    queue,
) -> None:
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        import ovrtx

        bundle = json.loads(bundle_path.read_text())
        log_path = bundle_path.parent / f"ovrtx_{environments}_gpu_{gpu}.log"
        renderer = ovrtx.Renderer(
            ovrtx.RendererConfig(
                log_level="warn",
                log_file_path=str(log_path),
                active_cuda_gpus="0",
                keep_system_alive=True,
            )
        )
        renderer.open_usd(str(stage_path))
        shapes = []
        checksum = 0
        for index in range(warmup):
            renderer.update_from_usd_time((index % bundle["frames"] + 1) / bundle["fps"])
            products = renderer.step(
                render_products={render_product}, delta_time=1 / bundle["fps"]
            )
            shapes, value = _map_ovrtx_product(products[render_product])
            checksum ^= value
        queue.put({"ready": True, "gpu": gpu})
        if not start_event.wait(timeout=1200):
            raise TimeoutError("ovrtx start barrier timed out")
        render_seconds = 0.0
        rgb_seconds = 0.0
        for index in range(steps):
            renderer.update_from_usd_time((index % bundle["frames"] + 1) / bundle["fps"])
            started = time.perf_counter()
            products = renderer.step(
                render_products={render_product}, delta_time=1 / bundle["fps"]
            )
            render_seconds += time.perf_counter() - started
            started = time.perf_counter()
            shapes, value = _map_ovrtx_product(products[render_product])
            rgb_seconds += time.perf_counter() - started
            checksum ^= value
        expected_pixels = environments * int(bundle["width"]) * int(bundle["height"])
        actual_pixels = sum(shape[0] * shape[1] for shape in shapes)
        if actual_pixels != expected_pixels:
            raise RuntimeError(
                f"ovrtx tiled output has {actual_pixels} pixels; expected {expected_pixels} "
                f"for {environments} full-resolution environments (shapes={shapes})"
            )
        del renderer
        queue.put(
            {
                "gpu": gpu,
                "render_seconds": render_seconds,
                "readback_seconds": rgb_seconds,
                "output_shapes": shapes,
                "checksum": checksum,
            }
        )
    except BaseException as exc:
        queue.put(
            {
                "gpu": gpu,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )


def _ovrtx_trial(
    bundle_path: Path,
    environments: int,
    gpus: list[int],
    steps: int,
    warmup: int,
) -> dict[str, object]:
    stage_path, render_product = _write_ovrtx_multi_env_stage(bundle_path, environments)
    context = multiprocessing.get_context("spawn")
    start_event = context.Event()
    queue = context.Queue()
    processes = []
    for gpu in gpus:
        process = context.Process(
            target=_ovrtx_process,
            args=(
                bundle_path,
                environments,
                stage_path,
                render_product,
                gpu,
                steps,
                warmup,
                start_event,
                queue,
            ),
        )
        process.start()
        processes.append(process)
    try:
        _collect_ready(processes, queue, len(processes), "ovrtx")
        start_event.set()
        results = _finish_process_trial(processes, queue, len(processes), "ovrtx")
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
    render_seconds = max(float(result["render_seconds"]) for result in results)
    total_seconds = max(
        float(result["render_seconds"]) + float(result["readback_seconds"])
        for result in results
    )
    readback_seconds = max(float(result["readback_seconds"]) for result in results)
    total_frames = environments * steps * len(gpus)
    return {
        "batch_per_gpu": environments,
        "gpu_processes": len(gpus),
        "total_environments": environments * len(gpus),
        "gpus_used": gpus,
        "steps": steps,
        "total_frames": total_frames,
        "render_seconds": render_seconds,
        "readback_seconds": readback_seconds,
        "render_fps": total_frames / max(render_seconds, 1e-9),
        "rgb_fps": total_frames / max(total_seconds, 1e-9),
        "throughput_fps": total_frames / max(total_seconds, 1e-9),
        "output_shapes": [result["output_shapes"] for result in results],
        "checksums": [result["checksum"] for result in results],
    }


def _prepare_bundles(
    destination: Path,
    frames: int,
    width: int,
    height: int,
    fps: int,
    samples: int,
) -> dict[str, object]:
    from render_blender import prepare_blender_bundle
    from render_ovrtx import prepare_ovrtx_bundle

    shutil.rmtree(destination, ignore_errors=True)
    blender_dir = destination / "blender"
    ovrtx_dir = destination / "ovrtx"
    times = [index / fps for index in range(max(1, frames))]
    blender_manifest = prepare_blender_bundle(blender_dir, times, width, height, fps)
    ovrtx_bundle = prepare_ovrtx_bundle(
        ovrtx_dir, times, width, height, fps, samples, "PathTracing"
    )
    return {
        "destination": str(destination),
        "blender_manifest": str(blender_manifest),
        "ovrtx_bundle": str(ovrtx_bundle),
    }


def _trial_arguments(args, gpus: list[int]) -> list[str]:
    return [
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--frames",
        str(args.frames),
        "--warmup",
        str(args.warmup),
        "--fps",
        str(args.fps),
        "--gpus",
        ",".join(map(str, gpus)),
    ]


def _best_trial(trials: list[dict[str, object]]) -> dict[str, object] | None:
    successful = [
        trial
        for trial in trials
        if "error" not in trial and trial.get("throughput_fps", trial.get("render_fps"))
    ]
    return (
        max(
            successful,
            key=lambda trial: float(trial.get("throughput_fps", trial["render_fps"])),
        )
        if successful
        else None
    )


def _run_trials(values, function) -> list[dict[str, object]]:
    trials = []
    for value in values:
        try:
            trials.append(function(value))
        except Exception as exc:
            trials.append({"configuration": value, "error": f"{type(exc).__name__}: {exc}"})
    return trials


def run_suite(args: argparse.Namespace) -> dict[str, object]:
    setup = setup_remote(args.host) if not args.no_setup else None
    hardware = _remote_result(args.host, _remote_uv_command(["__hardware"]))
    available_gpus = [int(gpu["index"]) for gpu in hardware["gpus"]]
    gpus = args.gpus if args.gpus is not None else available_gpus
    missing = sorted(set(gpus) - set(available_gpus))
    if missing:
        raise ValueError(f"Requested GPU indices are unavailable on {args.host}: {missing}")
    remote_workdir = f"~/{REMOTE_ROOT}/build/fps_benchmark"
    prepared = _remote_result(
        args.host,
        _remote_uv_command(
            [
                "__prepare",
                "--destination",
                "build/fps_benchmark",
                "--frames",
                str(args.frames),
                "--width",
                str(args.width),
                "--height",
                str(args.height),
                "--fps",
                str(args.fps),
                "--samples",
                str(args.samples),
            ]
        ),
        timeout=300,
    )
    common = _trial_arguments(args, gpus)
    results = {}
    if "opengl" in args.backends:
        trials = _run_trials(
            args.opengl_workers,
            lambda workers: _remote_result(
                args.host,
                _remote_uv_command(["__opengl", "--workers", str(workers), *common]),
                timeout=1800,
            ),
        )
        results["opengl"] = {
            "strategy": "independent EGL contexts distributed across GPUs",
            "trials": trials,
            "best": _best_trial(trials),
        }
    if "warp" in args.backends:
        trials = _run_trials(
            args.warp_batches,
            lambda batch: _remote_result(
                args.host,
                _remote_uv_command(["__warp", "--batch", str(batch), *common]),
                timeout=1800,
            ),
        )
        results["warp"] = {
            "strategy": "one batched nworld renderer per GPU",
            "trials": trials,
            "best": _best_trial(trials),
        }
    if "blender" in args.backends:
        def run_blender_trial(workers: int) -> dict[str, object]:
            cpu_threads = args.blender_cpu_threads
            if cpu_threads is None:
                logical_cpus = int(hardware["logical_cpus"])
                cpu_threads = max(1, min(8, logical_cpus // max(1, workers * 6)))
            return _remote_result(
                args.host,
                _remote_uv_command(
                    [
                        "__blender",
                        "--blender",
                        f"~/{BLENDER_REMOTE}",
                        "--manifest",
                        prepared["blender_manifest"],
                        "--workdir",
                        f"{remote_workdir}/blender_trials",
                        "--workers",
                        str(workers),
                        "--samples",
                        str(args.samples),
                        "--gpus-per-worker",
                        str(args.blender_gpus_per_worker),
                        "--cpu-threads",
                        str(cpu_threads),
                        "--denoiser",
                        args.blender_denoiser,
                        "--persistent-data",
                        str(int(args.blender_persistent_data)),
                        "--max-bounces",
                        str(args.blender_max_bounces),
                        "--numa-bind",
                        str(int(args.blender_numa_bind)),
                        *common,
                    ]
                ),
                timeout=3600,
            )

        trials = _run_trials(
            args.blender_workers,
            run_blender_trial,
        )
        results["blender"] = {
            "strategy": (
                "persistent headless Cycles/OptiX workers with explicit GPU groups, "
                "CPU thread limits, and NUMA affinity"
            ),
            "trials": trials,
            "best": _best_trial(trials),
        }
    if "ovrtx" in args.backends:
        trials = _run_trials(
            args.ovrtx_batches,
            lambda environments: _remote_result(
                args.host,
                _remote_ovrtx_command(
                    [
                        "__ovrtx",
                        "--bundle",
                        prepared["ovrtx_bundle"],
                        "--environments",
                        str(environments),
                        "--steps",
                        str(args.frames),
                        "--warmup",
                        str(args.warmup),
                        "--gpus",
                        ",".join(map(str, gpus)),
                    ]
                ),
                timeout=3600,
            ),
        )
        results["ovrtx"] = {
            "strategy": "one full-resolution tiled multi-camera process per GPU",
            "trials": trials,
            "best": _best_trial(trials),
        }
    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "host": args.host,
        "hardware": hardware,
        "configuration": {
            "scene": "fr3_renderer_evaluation",
            "width": args.width,
            "height": args.height,
            "fps_trajectory": args.fps,
            "timed_steps_or_frames": args.frames,
            "warmup": args.warmup,
            "path_tracing_samples": args.samples,
            "blender": {
                "gpus_per_worker": args.blender_gpus_per_worker,
                "cpu_threads_per_worker": args.blender_cpu_threads or "auto",
                "denoiser": args.blender_denoiser,
                "persistent_data": args.blender_persistent_data,
                "max_bounces": args.blender_max_bounces,
                "numa_bind": args.blender_numa_bind,
            },
            "gpus": gpus,
            "encoding_included": False,
        },
        "methodology": (
            "Steady-state rendering of the canonical moving FR3 scene. Initialization, shader "
            "compilation, warmup, video encoding, and SSH transfer are excluded. "
            "throughput_fps is the comparable completed-image metric; render_fps measures the "
            "renderer call and may exclude an asynchronous GPU-to-host mapping step."
        ),
        "setup": setup,
        "results": results,
    }


def _format_number(value: object) -> str:
    if value is None:
        return "—"
    return f"{float(value):,.2f}"


def _trial_label(backend: str, trial: dict[str, object]) -> str:
    if backend == "warp":
        return f"batch {trial['batch_per_gpu']} × {trial['gpu_processes']} GPUs"
    if backend == "ovrtx":
        return f"tile {trial['batch_per_gpu']} × {trial['gpu_processes']} GPU processes"
    if backend == "blender" and int(trial.get("gpus_per_worker", 1)) > 1:
        noun = "process" if int(trial["workers"]) == 1 else "processes"
        return f"{trial['workers']} {noun} × {trial['gpus_per_worker']} GPUs"
    noun = "process" if int(trial["workers"]) == 1 else "processes"
    return f"{trial['workers']} {noun}"


def format_markdown(result: dict[str, object]) -> str:
    config = result["configuration"]
    hardware = result["hardware"]
    lines = [
        "# Renderer throughput benchmark",
        "",
        f"Generated: {result['generated_at']}",
        "",
        "## Test configuration",
        "",
        f"- Host: `{result['host']}` (`{hardware['hostname']}`)",
        f"- CPU: {hardware['logical_cpus']} logical CPUs, {hardware['architecture']}",
        f"- GPUs: {len(hardware['gpus'])} × {hardware['gpus'][0]['name']}",
        f"- Resolution: {config['width']}×{config['height']} per environment",
        f"- Scene: `{config['scene']}` with the shared robot and orbit-camera trajectory",
        f"- Path tracers: {config['path_tracing_samples']} samples per pixel",
        f"- Warmup: {config['warmup']} frames; timed work: "
        f"{config['timed_steps_or_frames']} frames/steps"
        + (
            f" ({config['blender_timed_frames']} frames for Blender)"
            if config.get("blender_timed_frames")
            else ""
        ),
        "- Video encoding, initialization, shader compilation, and SSH transfer are excluded.",
        "",
        "`throughput_fps` is the primary comparison: it measures completed images available "
        "through each backend's normal result API. For Warp and ovrtx this includes GPU-to-host "
        "RGB transfer; OpenGL always returns host pixels; Blender completes an internal Render "
        "Result. `render_fps` is also shown, but can exclude asynchronous result mapping.",
        "",
        "## Best measured throughput",
        "",
        "| Rank | Backend | Best configuration | Throughput FPS | Render-call FPS |",
        "| ---: | --- | --- | ---: | ---: |",
    ]
    ranking = []
    for backend, details in result["results"].items():
        if details.get("best"):
            ranking.append((backend, details))
    ranking.sort(
        key=lambda item: float(
            item[1]["best"].get("throughput_fps", item[1]["best"]["render_fps"])
        ),
        reverse=True,
    )
    for rank, (backend, details) in enumerate(ranking, start=1):
        best = details["best"]
        lines.append(
            f"| {rank} | {backend} | {_trial_label(backend, best)} | "
            f"{_format_number(best.get('throughput_fps', best['render_fps']))} | "
            f"{_format_number(best['render_fps'])} |"
        )
    if result.get("findings"):
        lines.extend(["", "## Findings", ""])
        lines.extend(f"- {finding}" for finding in result["findings"])
    for backend, details in result["results"].items():
        lines.extend(
            [
                "",
                f"## {backend} scaling",
                "",
                f"Strategy: {details['strategy']}.",
                "",
                "| Configuration | Environments/workers | Throughput FPS | Render-call FPS "
                "| Host RGB FPS | Notes |",
                "| --- | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for trial in details["trials"]:
            if "error" in trial:
                lines.append(
                    f"| {trial.get('configuration', 'unknown')} | — | — | — | — | "
                    f"{trial['error']} |"
                )
                continue
            name = _trial_label(backend, trial)
            if backend in {"warp", "ovrtx"}:
                count = trial["total_environments"]
            else:
                count = trial["workers"]
            notes = f"GPUs {trial.get('gpus_used', [])}"
            if backend == "blender":
                notes += (
                    f"; {trial['cpu_threads_per_worker']} CPU threads/process; "
                    f"{trial['denoiser']} denoiser; "
                    f"{'persistent' if trial['persistent_data'] else 'non-persistent'} data"
                )
            lines.append(
                f"| {name} | {count} | "
                f"{_format_number(trial.get('throughput_fps', trial['render_fps']))} | "
                f"{_format_number(trial['render_fps'])} | "
                f"{_format_number(trial.get('rgb_fps'))} | {notes} |"
            )
        if backend == "blender" and details.get("optimization_comparison"):
            lines.extend(
                [
                    "",
                    "### Blender optimization A/B",
                    "",
                    "| Configuration | Throughput FPS | Relative to original |",
                    "| --- | ---: | ---: |",
                ]
            )
            baseline = float(details["optimization_comparison"][0]["throughput_fps"])
            for comparison in details["optimization_comparison"]:
                fps = float(comparison["throughput_fps"])
                lines.append(
                    f"| {comparison['label']} | {_format_number(fps)} | "
                    f"{fps / baseline:.2f}× |"
                )
        if backend == "blender" and details.get("device_layout_trials"):
            lines.extend(
                [
                    "",
                    "### Blender GPU layout A/B",
                    "",
                    "| Layout | Throughput FPS |",
                    "| --- | ---: |",
                ]
            )
            for layout in details["device_layout_trials"]:
                lines.append(
                    f"| {_trial_label('blender', layout)} | "
                    f"{_format_number(layout['throughput_fps'])} |"
                )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Use the peak of each scaling curve for aggregate dataset-generation throughput. "
            "Use the single-worker or single-environment row for latency-sensitive interactive "
            "rendering. These figures compare the configured pipelines, not equal perceptual "
            "quality: OpenGL and Warp use their native raster/ray settings, while Blender and "
            "ovrtx use the stated path-tracing sample count.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scale and compare renderer FPS on a remote multi-GPU host."
    )
    parser.add_argument("--host", default="g4")
    parser.add_argument("--backends", nargs="+", choices=BACKENDS, default=list(BACKENDS))
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30, help="Analytical trajectory sampling rate")
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--gpus", type=_parse_gpus)
    parser.add_argument(
        "--opengl-workers",
        type=_parse_positive_ints,
        default=[1, 2, 4, 8, 16, 32, 64, 128],
    )
    parser.add_argument(
        "--warp-batches", type=_parse_positive_ints, default=[1, 4, 16, 64, 128, 256]
    )
    parser.add_argument(
        "--blender-workers",
        type=_parse_positive_ints,
        default=[1, 2, 4, 8, 16, 32, 48, 64, 96],
    )
    parser.add_argument("--blender-gpus-per-worker", type=int, default=1)
    parser.add_argument(
        "--blender-cpu-threads",
        type=_parse_auto_positive_int,
        default=None,
        help="CPU threads per process; auto budgets about one-sixth of host CPUs across workers",
    )
    parser.add_argument(
        "--blender-denoiser",
        choices=("OPTIX", "OPENIMAGEDENOISE", "NONE"),
        default="OPTIX",
    )
    parser.add_argument(
        "--blender-persistent-data",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--blender-max-bounces", type=int, default=12)
    parser.add_argument(
        "--blender-numa-bind", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--ovrtx-batches", type=_parse_positive_ints, default=[1, 4, 16]
    )
    parser.add_argument("--no-setup", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("outputs/fps_benchmark_g4.json"))
    parser.add_argument("--report", type=Path, default=Path("outputs/fps_benchmark_g4.md"))
    return parser


def _internal_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--gpus", type=_parse_gpus, default=[0])
    return parser


def _run_internal(arguments: list[str]) -> int:
    mode = arguments[0]
    if mode == "__blender_worker":
        return _blender_worker(arguments[1:])
    if mode == "__hardware":
        _emit_result(_hardware_info())
        return 0
    if mode == "__prepare":
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--destination", type=Path, required=True)
        parser.add_argument("--frames", type=int, required=True)
        parser.add_argument("--width", type=int, required=True)
        parser.add_argument("--height", type=int, required=True)
        parser.add_argument("--fps", type=int, required=True)
        parser.add_argument("--samples", type=int, required=True)
        args = parser.parse_args(arguments[1:])
        _emit_result(
            _prepare_bundles(
                args.destination.resolve(),
                args.frames,
                args.width,
                args.height,
                args.fps,
                args.samples,
            )
        )
        return 0
    if mode == "__ovrtx":
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--bundle", type=Path, required=True)
        parser.add_argument("--environments", type=int, required=True)
        parser.add_argument("--steps", type=int, required=True)
        parser.add_argument("--warmup", type=int, required=True)
        parser.add_argument("--gpus", type=_parse_gpus, required=True)
        args = parser.parse_args(arguments[1:])
        _emit_result(
            _ovrtx_trial(
                args.bundle.expanduser().resolve(),
                args.environments,
                args.gpus,
                args.steps,
                args.warmup,
            )
        )
        return 0
    parser = _internal_parser()
    if mode == "__opengl":
        parser.add_argument("--workers", type=int, required=True)
        args = parser.parse_args(arguments[1:])
        _emit_result(
            _opengl_trial(
                args.workers,
                args.gpus,
                args.width,
                args.height,
                args.frames,
                args.warmup,
                args.fps,
            )
        )
        return 0
    if mode == "__warp":
        parser.add_argument("--batch", type=int, required=True)
        args = parser.parse_args(arguments[1:])
        _emit_result(
            _warp_trial(
                args.batch,
                args.gpus,
                args.width,
                args.height,
                args.frames,
                args.warmup,
                args.fps,
            )
        )
        return 0
    if mode == "__blender":
        parser.add_argument("--blender", type=Path, required=True)
        parser.add_argument("--manifest", type=Path, required=True)
        parser.add_argument("--workdir", type=Path, required=True)
        parser.add_argument("--workers", type=int, required=True)
        parser.add_argument("--samples", type=int, required=True)
        parser.add_argument("--gpus-per-worker", type=int, required=True)
        parser.add_argument("--cpu-threads", type=int, required=True)
        parser.add_argument("--denoiser", required=True)
        parser.add_argument("--persistent-data", type=int, choices=(0, 1), required=True)
        parser.add_argument("--max-bounces", type=int, required=True)
        parser.add_argument("--numa-bind", type=int, choices=(0, 1), required=True)
        args = parser.parse_args(arguments[1:])
        _emit_result(
            _blender_trial(
                args.blender.expanduser().resolve(),
                args.manifest.expanduser().resolve(),
                args.workdir.expanduser().resolve(),
                args.workers,
                args.gpus,
                args.frames,
                args.warmup,
                args.samples,
                args.gpus_per_worker,
                args.cpu_threads,
                args.denoiser,
                bool(args.persistent_data),
                args.max_bounces,
                bool(args.numa_bind),
            )
        )
        return 0
    raise ValueError(f"Unknown internal mode: {mode}")


def main(argv: list[str] | None = None) -> int:
    if argv is None and "--" in sys.argv:
        arguments = list(sys.argv[sys.argv.index("--") + 1 :])
    else:
        arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        if arguments and arguments[0].startswith("__"):
            return _run_internal(arguments)
        args = build_parser().parse_args(arguments)
        result = run_suite(args)
        output = args.output.expanduser().resolve()
        report = args.report.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        report.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        report.write_text(format_markdown(result))
        print(format_markdown(result))
        print(f"\nJSON: {output}\nReport: {report}")
        return 0
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        traceback.print_exc()
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
