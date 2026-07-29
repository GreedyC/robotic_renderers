"""Compatibility alias and legacy worker entry point for MuJoCo Warp."""

import runpy
import sys

if __name__ == "__main__":
    runpy.run_module("robotic_renderers.backends.mujoco_warp", run_name="__main__")
else:
    from robotic_renderers.backends import mujoco_warp as _implementation

    sys.modules[__name__] = _implementation
