"""Single source of truth for renderer names, modules, titles, and capabilities."""

from __future__ import annotations

import importlib

_BACKENDS = {
    "opengl": {
        "module": "robotic_renderers.backends.mujoco_opengl",
        "title": "MuJoCo OpenGL",
        "remote": False,
        "samples": False,
        "aovs": False,
    },
    "warp": {
        "module": "robotic_renderers.backends.mujoco_warp",
        "title": "MuJoCo Warp",
        "remote": "optional",
        "samples": False,
        "aovs": False,
    },
    "blender": {
        "module": "robotic_renderers.backends.blender",
        "title": "Blender Cycles",
        "remote": False,
        "samples": True,
        "aovs": False,
    },
    "ovrtx": {
        "module": "robotic_renderers.backends.ovrtx",
        "title": "NVIDIA ovrtx",
        "remote": True,
        "samples": True,
        "aovs": True,
    },
    "mitsuba": {
        "module": "robotic_renderers.backends.mitsuba",
        "title": "Mitsuba 3",
        "remote": True,
        "samples": True,
        "aovs": False,
    },
}


def backend_names() -> tuple[str, ...]:
    return tuple(_BACKENDS)


def backend_modules() -> dict[str, str]:
    return {name: str(spec["module"]) for name, spec in _BACKENDS.items()}


def backend_titles() -> dict[str, str]:
    return {name: str(spec["title"]) for name, spec in _BACKENDS.items()}


def backend_spec(name: str) -> dict[str, object]:
    try:
        return dict(_BACKENDS[name])
    except KeyError as exc:
        raise ValueError(f"Unknown renderer backend: {name}") from exc


def load_backend(name: str):
    try:
        return importlib.import_module(str(backend_spec(name)["module"]))
    except ModuleNotFoundError as exc:
        extra = "warp" if name == "warp" else "all"
        raise RuntimeError(
            f"{name} backend dependency {exc.name!r} is unavailable. "
            f"Install robotic-renderers[{extra}]."
        ) from exc
