"""Compatibility alias for the Cosmos 3 post-process workflow."""

import sys

from robotic_renderers.workflows import cosmos3 as _implementation

sys.modules[__name__] = _implementation
