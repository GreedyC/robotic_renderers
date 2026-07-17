"""Concurrent multi-backend recording and labeled video composition."""

from __future__ import annotations

import math
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

from render_common import resize_rgb, write_video

ROOT = Path(__file__).resolve().parent
BACKEND_ORDER = ("opengl", "warp", "blender", "ovrtx")
BACKEND_TITLES = {
    "opengl": "MuJoCo OpenGL",
    "warp": "MuJoCo Warp",
    "blender": "Blender Cycles",
    "ovrtx": "NVIDIA ovrtx",
}


def _title_font(tile_height: int):
    size = max(18, min(48, tile_height // 18))
    candidates = (
        Path("/System/Library/Fonts/SFNS.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
    )
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


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
            canvas = Image.new(
                "RGB", (tile_width * columns, tile_height * rows), (5, 6, 8)
            )
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
    columns = math.ceil(math.sqrt(len(BACKEND_ORDER)))
    rows = math.ceil(len(BACKEND_ORDER) / columns)
    output, written, seconds = write_video(
        tiled_frames(video_paths, frame_count, tile_width, tile_height),
        output_path,
        fps=fps,
    )
    return {
        "output": str(output),
        "frames": written,
        "fps": fps,
        "duration": written / float(fps),
        "tile_width": tile_width,
        "tile_height": tile_height,
        "columns": columns,
        "rows": rows,
        "width": tile_width * columns,
        "height": tile_height * rows,
        "compose_seconds": seconds,
    }


def _preflight(host: str) -> dict[str, object]:
    import render_blender
    import render_mujoco_opengl
    import render_mujoco_warp
    import render_ovrtx

    statuses = {
        "opengl": render_mujoco_opengl.backend_status(),
        "warp": render_mujoco_warp.backend_status(),
        "blender": render_blender.backend_status(),
        "ovrtx": render_ovrtx.backend_status(host),
    }
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
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "main.py"),
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
        command.extend(("--supersample", str(supersample), "--host", host))
        if remote_warp:
            command.append("--remote-warp")
    elif backend == "blender":
        command.extend(("--samples", str(samples)))
    elif backend == "ovrtx":
        command.extend(("--samples", str(samples), "--host", host))
    return command


def record_all_backends(
    output_path: str | Path,
    duration: float = 5.0,
    fps: int = 30,
    width: int = 1280,
    height: int = 720,
    supersample: int = 1,
    samples: int = 256,
    host: str = "l4",
    remote_warp: bool = False,
    keep_components: bool = False,
) -> dict[str, object]:
    statuses = _preflight(host)
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    (ROOT / "build").mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="tiled-render-", dir=ROOT / "build") as temporary:
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
                )
                log_streams[backend] = log_paths[backend].open("wb")
                processes[backend] = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    stdout=log_streams[backend],
                    stderr=subprocess.STDOUT,
                )

            return_codes = {backend: process.wait() for backend, process in processes.items()}
        except BaseException:
            for process in processes.values():
                if process.poll() is None:
                    process.terminate()
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
