"""Compatibility entry point; prefer ``robotic-renderers`` or ``python -m robotic_renderers``."""

import sys

if __name__ == "__main__":
    from robotic_renderers.cli import main

    raise SystemExit(main())
else:
    from robotic_renderers import cli as _implementation

    sys.modules[__name__] = _implementation
