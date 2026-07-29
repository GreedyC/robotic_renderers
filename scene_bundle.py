"""Compatibility alias for :mod:`robotic_renderers.scene_bundle`."""

import sys

from robotic_renderers import scene_bundle as _implementation

sys.modules[__name__] = _implementation
