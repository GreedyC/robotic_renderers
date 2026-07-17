"""Small renderer-independent image, live-view, and video helpers."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Iterable
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image


def ensure_rgb8(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating):
            array = np.clip(array, 0.0, 1.0) * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise ValueError(f"Expected HxWx3/4 image, got {array.shape}")
    return np.ascontiguousarray(array[:, :, :3])


def resize_rgb(image: np.ndarray, width: int, height: int) -> np.ndarray:
    image = ensure_rgb8(image)
    if image.shape[:2] == (height, width):
        return image
    resized = Image.fromarray(image).resize((width, height), Image.Resampling.LANCZOS)
    return np.asarray(resized)


def write_png(image: np.ndarray, output_path: str | Path) -> Path:
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(ensure_rgb8(image)).save(output_path, format="PNG", optimize=True)
    return output_path


def write_video(
    frames: Iterable[np.ndarray],
    output_path: str | Path,
    fps: int,
    codec: str = "libx264",
) -> tuple[Path, int, float]:
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    frame_count = 0
    writer = imageio.get_writer(
        output_path,
        format="FFMPEG",
        mode="I",
        fps=fps,
        codec=codec,
        quality=9,
        pixelformat="yuv420p",
        macro_block_size=1,
        ffmpeg_log_level="warning",
        output_params=["-movflags", "+faststart"],
    )
    try:
        for frame in frames:
            writer.append_data(ensure_rgb8(frame))
            frame_count += 1
    finally:
        writer.close()
    return output_path, frame_count, time.perf_counter() - started


def display_frames(
    frames: Iterable[np.ndarray],
    width: int,
    height: int,
    title: str,
    target_fps: int = 30,
) -> dict[str, float | int]:
    import pygame

    pygame.init()
    pygame.display.set_caption(f"{title} — Esc/Q to close")
    window = pygame.display.set_mode((width, height), pygame.RESIZABLE)
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)
    started = time.perf_counter()
    frame_count = 0
    running = True
    source = iter(frames)
    try:
        while running:
            for event in pygame.event.get():
                close_key = event.type == pygame.KEYDOWN and event.key in (
                    pygame.K_ESCAPE,
                    pygame.K_q,
                )
                if event.type == pygame.QUIT or close_key:
                    running = False
            if not running:
                break
            try:
                frame = ensure_rgb8(next(source))
            except StopIteration:
                break

            surface = pygame.surfarray.make_surface(np.swapaxes(frame, 0, 1))
            window_size = window.get_size()
            if surface.get_size() != window_size:
                surface = pygame.transform.smoothscale(surface, window_size)
            window.blit(surface, (0, 0))

            frame_count += 1
            elapsed = max(time.perf_counter() - started, 1e-9)
            label = font.render(
                f"{title}  |  {frame_count / elapsed:5.1f} display FPS",
                True,
                (245, 245, 245),
                (18, 18, 22),
            )
            window.blit(label, (12, 10))
            pygame.display.flip()
            if target_fps > 0:
                clock.tick(target_fps)
    finally:
        close = getattr(source, "close", None)
        if close is not None:
            close()
        pygame.quit()
    elapsed = time.perf_counter() - started
    return {
        "frames": frame_count,
        "seconds": elapsed,
        "display_fps": frame_count / max(elapsed, 1e-9),
    }


def image_digest(image: np.ndarray) -> str:
    return hashlib.sha256(ensure_rgb8(image).tobytes()).hexdigest()
