"""Compatibility alias for :mod:`robotic_renderers.processes`."""

import sys

from robotic_renderers import processes as _implementation

sys.modules[__name__] = _implementation
