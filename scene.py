"""Compatibility alias for :mod:`robotic_renderers.scene`."""

import sys

from robotic_renderers import scene as _implementation

sys.modules[__name__] = _implementation
