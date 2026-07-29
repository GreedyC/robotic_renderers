"""Compatibility alias and Blender worker entry point."""

import runpy
import sys

if __name__ == "__main__":
    runpy.run_module("robotic_renderers.backends.blender", run_name="__main__")
else:
    from robotic_renderers.backends import blender as _implementation

    sys.modules[__name__] = _implementation
