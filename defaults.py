"""Compatibility alias for :mod:`robotic_renderers.defaults`."""

import sys

from robotic_renderers import defaults as _implementation

sys.modules[__name__] = _implementation
