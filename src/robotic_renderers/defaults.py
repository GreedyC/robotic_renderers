"""Shared user-facing defaults and writable directory locations."""

from __future__ import annotations

import os
import sys
from pathlib import Path

DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_VIDEO_FPS = 30
DEFAULT_RECORDING_DURATION = 30.0
DEFAULT_RENDER_HOST = os.environ.get("ROBOTIC_RENDERERS_HOST", "render-host")
DEFAULT_PATH_TRACING_SAMPLES = 256
DEFAULT_LIVE_SAMPLES = 64
DEFAULT_BENCHMARK_SAMPLES = 32


def output_directory() -> Path:
    """Return the user-writable output directory, creating it when needed."""
    configured = os.environ.get("ROBOTIC_RENDERERS_OUTPUT_DIR")
    directory = Path(configured).expanduser() if configured else Path.cwd() / "outputs"
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def cache_directory() -> Path:
    """Return the user-writable cache directory, creating it when needed."""
    configured = os.environ.get("ROBOTIC_RENDERERS_CACHE_DIR")
    if configured:
        directory = Path(configured).expanduser()
    elif sys.platform == "darwin":
        directory = Path.home() / "Library" / "Caches" / "robotic_renderers"
    else:
        directory = Path.home() / ".cache" / "robotic_renderers"
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    return directory
