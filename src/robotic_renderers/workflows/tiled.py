"""Concurrent multi-backend recording and labeled video composition."""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterable
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from robotic_renderers.defaults import (
    DEFAULT_HEIGHT,
    DEFAULT_PATH_TRACING_SAMPLES,
    DEFAULT_RECORDING_DURATION,
    DEFAULT_RENDER_HOST,
    DEFAULT_VIDEO_FPS,
    DEFAULT_WIDTH,
    cache_directory,
)
from robotic_renderers.media import resize_rgb, write_video
from robotic_renderers.processes import terminate_and_reap, wait_fail_fast
from robotic_renderers.registry import backend_names, backend_spec, backend_titles, load_backend

BACKEND_ORDER = backend_names()
BACKEND_TITLES = backend_titles()


def _title_font(tile_height: int):
    size = max(18, min(48, tile_height // 18))
    return ImageFont.load_default(size=size)


def _add_title(image: Image.Image, title: str, font) -> None:
    draw = ImageDraw.Draw(image)
    left = max(10, image.width // 80)
    top = max(10, image.height // 45)
    box = draw.textbbox((left, top), title, font=font, stroke_width=1)
    padding_x = max(10, image.width // 100)
    padding_y = max(6, image.height // 120)
    draw.rounded_rectangle(
        (
            box[0] - padding_x,
            box[1] - padding_y,
            box[2] + padding_x,
            box[3] + padding_y,
        ),
        radius=max(6, padding_y),
        fill=(8, 10, 14),
        outline=(225, 230, 238),
        width=max(1, image.height // 360),
    )
    draw.text(
        (left, top),
        title,
        font=font,
        fill=(248, 250, 252),
        stroke_width=1,
        stroke_fill=(0, 0, 0),
    )


def tiled_frames(
    video_paths: dict[str, str | Path],
    frame_count: int,
    tile_width: int,
    tile_height: int,
) -> Iterable[np.ndarray]:
    readers = {
        backend: imageio.get_reader(Path(video_paths[backend]).expanduser().resolve())
        for backend in BACKEND_ORDER
    }
    font = _title_font(tile_height)
    columns = math.ceil(math.sqrt(len(BACKEND_ORDER)))
    rows = math.ceil(len(BACKEND_ORDER) / columns)
    try:
        for frame_index in range(frame_count):
            canvas = Image.new("RGB", (tile_width * columns, tile_height * rows), (5, 6, 8))
            for index, backend in enumerate(BACKEND_ORDER):
                try:
                    frame = readers[backend].get_data(frame_index)
                except IndexError as exc:
                    raise RuntimeError(
                        f"{backend} ended before synchronized frame {frame_index + 1}"
                    ) from exc
                frame = resize_rgb(frame, tile_width, tile_height)
                tile = Image.fromarray(frame)
                _add_title(tile, BACKEND_TITLES[backend], font)
                x = (index % columns) * tile_width
                y = (index // columns) * tile_height
                canvas.paste(tile, (x, y))

            separator = ImageDraw.Draw(canvas)
            line_width = max(2, min(tile_width, tile_height) // 240)
            for column in range(1, columns):
                x = column * tile_width
                separator.rectangle(
                    (x - line_width, 0, x + line_width, tile_height * rows),
                    fill=(5, 6, 8),
                )
            for row in range(1, rows):
                y = row * tile_height
                separator.rectangle(
                    (0, y - line_width, tile_width * columns, y + line_width),
                    fill=(5, 6, 8),
                )
            yield np.asarray(canvas)
    finally:
        for reader in readers.values():
            reader.close()


def compose_tiled_video(
    video_paths: dict[str, str | Path],
    output_path: str | Path,
    duration: float,
    fps: int,
    tile_width: int,
    tile_height: int,
) -> dict[str, object]:
    missing = [backend for backend in BACKEND_ORDER if backend not in video_paths]
    if missing:
        raise ValueError(f"Missing component videos: {', '.join(missing)}")
    frame_count = max(1, int(round(duration * fps)))
    _validate_component_videos(video_paths, frame_count, fps, tile_width, tile_height)
    columns = math.ceil(math.sqrt(len(BACKEND_ORDER)))
    rows = math.ceil(len(BACKEND_ORDER) / columns)
    output, written, seconds = write_video(
        tiled_frames(video_paths, frame_count, tile_width, tile_height),
        output_path,
        fps=fps,
        quality=6,
    )
    return {
        "output": str(output),
        "frames": written,
        "video_fps": fps,
        "duration": written / float(fps),
        "tile_width": tile_width,
        "tile_height": tile_height,
        "columns": columns,
        "rows": rows,
        "width": tile_width * columns,
        "height": tile_height * rows,
        "compose_seconds": seconds,
    }


def _validate_component_videos(
    video_paths: dict[str, str | Path],
    frame_count: int,
    fps: int,
    width: int,
    height: int,
) -> None:
    """Reject components whose basic timing or dimensions do not match."""
    expected_size = (width, height)
    for backend in BACKEND_ORDER:
        path = Path(video_paths[backend]).expanduser().resolve()
        reader = imageio.get_reader(path)
        try:
            metadata = reader.get_meta_data()
            actual_fps = float(metadata.get("fps", 0.0))
            actual_size = tuple(metadata.get("size", ()))
            actual_frames = int(reader.count_frames())
        finally:
            reader.close()
        if not math.isclose(actual_fps, float(fps), rel_tol=0.0, abs_tol=0.01):
            raise ValueError(f"{backend} video FPS is {actual_fps}, expected {fps}: {path}")
        if actual_size != expected_size:
            raise ValueError(
                f"{backend} video size is {actual_size}, expected {expected_size}: {path}"
            )
        if actual_frames != frame_count:
            raise ValueError(
                f"{backend} has {actual_frames} frames, expected {frame_count}: {path}"
            )


def _preflight(host: str) -> dict[str, object]:
    statuses = {}
    for backend in BACKEND_ORDER:
        module = load_backend(backend)
        remote = backend_spec(backend)["remote"]
        statuses[backend] = (
            module.backend_status(host) if remote in (True, "optional") else module.backend_status()
        )
    unavailable = [name for name, status in statuses.items() if not status["available"]]
    if unavailable:
        raise RuntimeError(f"Unavailable backends: {', '.join(unavailable)}")
    return statuses


def _backend_command(
    backend: str,
    component_path: Path,
    duration: float,
    fps: int,
    width: int,
    height: int,
    supersample: int,
    samples: int,
    host: str,
    remote_warp: bool,
    render_mode: str,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "robotic_renderers",
        "record",
        "--backend",
        backend,
        "--duration",
        str(duration),
        "--fps",
        str(fps),
        "--width",
        str(width),
        "--height",
        str(height),
        "--output",
        str(component_path),
    ]
    if backend == "warp":
        command.extend(("--supersample", str(supersample)))
        if remote_warp:
            command.extend(("--remote-warp", "--host", host))
    elif backend == "blender":
        command.extend(("--samples", str(samples)))
    elif backend == "ovrtx":
        command.extend(("--samples", str(samples), "--host", host, "--render-mode", render_mode))
    elif backend == "mitsuba":
        command.extend(("--samples", str(samples), "--host", host))
    return command


def record_all_backends(
    output_path: str | Path,
    duration: float = DEFAULT_RECORDING_DURATION,
    fps: int = DEFAULT_VIDEO_FPS,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    supersample: int = 1,
    samples: int = DEFAULT_PATH_TRACING_SAMPLES,
    host: str = DEFAULT_RENDER_HOST,
    remote_warp: bool = False,
    keep_components: bool = False,
    render_mode: str = "pathtraced",
) -> dict[str, object]:
    statuses = _preflight(host)
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    build = cache_directory() / "tiled"
    build.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="tiled-render-", dir=build) as temporary:
        job = Path(temporary)
        component_paths = {backend: job / f"{backend}.mp4" for backend in BACKEND_ORDER}
        log_paths = {backend: job / f"{backend}.log" for backend in BACKEND_ORDER}
        processes = {}
        log_streams = {}
        try:
            for backend in BACKEND_ORDER:
                command = _backend_command(
                    backend,
                    component_paths[backend],
                    duration,
                    fps,
                    width,
                    height,
                    supersample,
                    samples,
                    host,
                    remote_warp,
                    render_mode,
                )
                log_streams[backend] = log_paths[backend].open("wb")
                processes[backend] = subprocess.Popen(
                    command,
                    stdout=log_streams[backend],
                    stderr=subprocess.STDOUT,
                    start_new_session=os.name == "posix",
                )

            return_codes = wait_fail_fast(processes)
        except BaseException:
            terminate_and_reap(processes)
            raise
        finally:
            for stream in log_streams.values():
                stream.close()

        failures = {}
        for backend, return_code in return_codes.items():
            if return_code != 0:
                log = log_paths[backend].read_text(errors="replace")
                failures[backend] = log[-3000:].strip() or f"exit code {return_code}"
        if failures:
            details = "\n\n".join(
                f"[{backend}]\n{message}" for backend, message in failures.items()
            )
            raise RuntimeError(f"Concurrent renderer failure:\n{details}")

        composition = compose_tiled_video(
            component_paths,
            output,
            duration,
            fps,
            width,
            height,
        )
        kept = None
        if keep_components:
            kept = output.with_name(f"{output.stem}_components")
            if kept.exists():
                shutil.rmtree(kept)
            kept.mkdir(parents=True)
            for backend, component in component_paths.items():
                shutil.copy2(component, kept / f"{backend}.mp4")

    return {
        "backend": "all",
        **composition,
        "parallel_seconds": time.perf_counter() - started,
        "components": {backend: BACKEND_TITLES[backend] for backend in BACKEND_ORDER},
        "component_directory": str(kept) if kept else None,
        "statuses": statuses,
    }
