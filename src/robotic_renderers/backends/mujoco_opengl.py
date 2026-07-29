"""MuJoCo's native OpenGL renderer backend."""

from __future__ import annotations

import platform
import time
from pathlib import Path

import mujoco
import numpy as np

from robotic_renderers.defaults import (
    DEFAULT_HEIGHT,
    DEFAULT_RECORDING_DURATION,
    DEFAULT_VIDEO_FPS,
    DEFAULT_WIDTH,
)
from robotic_renderers.media import display_frames, image_digest, write_png, write_video
from robotic_renderers.scene import (
    CAMERA_NAME,
    create_model_and_data,
    set_scene_time,
    trajectory_times,
)

BACKEND_NAME = "opengl"
OPENGL_OUTPUT_GAMMA = 0.85


def backend_status() -> dict[str, object]:
    return {
        "backend": BACKEND_NAME,
        "available": True,
        "version": mujoco.__version__,
        "device": f"OpenGL ({platform.system()} {platform.machine()})",
        "live": True,
        "record": True,
        "notes": "MuJoCo Phong lighting, shadow maps, MSAA, skybox, and planar reflection.",
    }


def create_renderer(
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
) -> dict[str, object]:
    model, data = create_model_and_data(width=width, height=height)
    option = mujoco.MjvOption()
    option.geomgroup[:] = 0
    option.geomgroup[:3] = 1
    renderer = mujoco.Renderer(model, height=height, width=width)
    return {
        "model": model,
        "data": data,
        "option": option,
        "renderer": renderer,
        "width": width,
        "height": height,
    }


def close_renderer(context: dict[str, object]) -> None:
    renderer = context.get("renderer")
    if renderer is not None:
        renderer.close()


def enable_quality_flags(renderer: mujoco.Renderer) -> None:
    for flag in (
        mujoco.mjtRndFlag.mjRND_SHADOW,
        mujoco.mjtRndFlag.mjRND_REFLECTION,
        mujoco.mjtRndFlag.mjRND_SKYBOX,
        mujoco.mjtRndFlag.mjRND_HAZE,
        mujoco.mjtRndFlag.mjRND_CULL_FACE,
    ):
        renderer.scene.flags[int(flag)] = 1


def render_context_frame(context: dict[str, object], time_s: float = 0.0):
    model = context["model"]
    data = context["data"]
    renderer = context["renderer"]
    set_scene_time(model, data, time_s)
    renderer.update_scene(data, camera=CAMERA_NAME, scene_option=context["option"])
    enable_quality_flags(renderer)
    image = renderer.render().copy()
    # Lift OpenGL's shadow-heavy framebuffer response to the common reference
    # without changing highlights or the authored light intensities.
    corrected = np.power(image.astype(np.float32) / 255.0, OPENGL_OUTPUT_GAMMA)
    return np.rint(corrected * 255.0).astype(np.uint8)


def render_frame(
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    time_s: float = 1.0,
):
    context = create_renderer(width=width, height=height)
    try:
        return render_context_frame(context, time_s=time_s)
    finally:
        close_renderer(context)


def render_still(
    output_path: str | Path,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    time_s: float = 1.0,
) -> dict[str, object]:
    started = time.perf_counter()
    image = render_frame(width=width, height=height, time_s=time_s)
    output = write_png(image, output_path)
    return {
        "backend": BACKEND_NAME,
        "output": str(output),
        "seconds": time.perf_counter() - started,
        "sha256": image_digest(image),
        "width": width,
        "height": height,
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


def _live_frames(context: dict[str, object], fps: int, duration: float | None):
    frame_index = 0
    while duration is None or frame_index / fps < duration:
        yield render_context_frame(context, frame_index / fps)
        frame_index += 1


def render_live(
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    fps: int = DEFAULT_VIDEO_FPS,
    duration: float | None = None,
) -> dict[str, object]:
    context = create_renderer(width=width, height=height)
    try:
        result = display_frames(
            _live_frames(context, fps=fps, duration=duration),
            width=width,
            height=height,
            target_fps=fps,
            title="MuJoCo OpenGL",
        )
        return {"backend": BACKEND_NAME, **result}
    finally:
        close_renderer(context)


def record_video(
    output_path: str | Path,
    duration: float = DEFAULT_RECORDING_DURATION,
    fps: int = DEFAULT_VIDEO_FPS,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
) -> dict[str, object]:
    context = create_renderer(width=width, height=height)
    timing: dict[str, float] = {}
    try:
        output, frame_count, total_seconds = write_video(
            _finite_frames(context, duration=duration, fps=fps, timing=timing),
            output_path,
            fps=fps,
        )
    finally:
        close_renderer(context)
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
    }


def benchmark(
    frame_count: int = 30,
    width: int = 640,
    height: int = 360,
    fps: int = 30,
    warmup: int = 3,
) -> dict[str, object]:
    init_started = time.perf_counter()
    context = create_renderer(width=width, height=height)
    init_seconds = time.perf_counter() - init_started
    try:
        for index in range(warmup):
            render_context_frame(context, index / fps)
        started = time.perf_counter()
        last = None
        for index in range(frame_count):
            last = render_context_frame(context, index / fps)
        render_seconds = time.perf_counter() - started
    finally:
        close_renderer(context)
    return {
        "backend": BACKEND_NAME,
        "device": backend_status()["device"],
        "width": width,
        "height": height,
        "frames": frame_count,
        "init_seconds": init_seconds,
        "render_seconds": render_seconds,
        "render_fps": frame_count / max(render_seconds, 1e-9),
        "last_frame_sha256": image_digest(last),
    }
