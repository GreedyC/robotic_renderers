"""Compatibility alias for the MuJoCo OpenGL backend."""

import sys

from robotic_renderers.backends import mujoco_opengl as _implementation

sys.modules[__name__] = _implementation
