"""Compatibility alias for :mod:`robotic_renderers.geometry`."""

import sys

from robotic_renderers import geometry as _implementation

sys.modules[__name__] = _implementation
