"""Deterministic Franka FR3 rendering pipeline comparison."""

from importlib.metadata import PackageNotFoundError, version

from robotic_renderers.registry import backend_names, load_backend

try:
    __version__ = version("robotic-renderers")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = ["__version__", "backend_names", "load_backend"]
