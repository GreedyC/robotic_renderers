"""Compatibility alias for :mod:`robotic_renderers.remote`."""

import sys

from robotic_renderers import remote as _implementation

sys.modules[__name__] = _implementation
