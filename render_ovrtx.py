"""Compatibility alias and legacy worker entry point for NVIDIA ovrtx."""

import runpy
import sys

if __name__ == "__main__":
    runpy.run_module("robotic_renderers.backends.ovrtx", run_name="__main__")
else:
    from robotic_renderers.backends import ovrtx as _implementation

    sys.modules[__name__] = _implementation
