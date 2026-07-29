"""Compatibility alias for :mod:`robotic_renderers.registry`."""

import sys

from robotic_renderers import registry as _implementation

sys.modules[__name__] = _implementation
