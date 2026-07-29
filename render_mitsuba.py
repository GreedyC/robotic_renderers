"""Compatibility alias and legacy worker entry point for Mitsuba."""

import runpy
import sys

if __name__ == "__main__":
    runpy.run_module("robotic_renderers.backends.mitsuba", run_name="__main__")
else:
    from robotic_renderers.backends import mitsuba as _implementation

    sys.modules[__name__] = _implementation
