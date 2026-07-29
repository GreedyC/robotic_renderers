"""Compatibility alias and entry point for the throughput benchmark."""

import runpy
import sys

if __name__ == "__main__":
    runpy.run_module("benchmarks.throughput", run_name="__main__")
else:
    from benchmarks import throughput as _implementation

    sys.modules[__name__] = _implementation
