"""Compatibility alias for the tiled comparison workflow."""

import sys

from robotic_renderers.workflows import tiled as _implementation

sys.modules[__name__] = _implementation
