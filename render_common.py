"""Compatibility alias for :mod:`robotic_renderers.media`."""

import sys

from robotic_renderers import media as _implementation

sys.modules[__name__] = _implementation
