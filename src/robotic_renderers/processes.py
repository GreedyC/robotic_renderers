"""Small functional helpers for supervising renderer subprocesses."""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import time
from collections.abc import Mapping


def _signal_process(process: subprocess.Popen, signal_number: int) -> None:
    if process.poll() is not None:
        return
    with contextlib.suppress(OSError, ProcessLookupError):
        if os.name == "posix":
            os.killpg(process.pid, signal_number)
        else:
            process.send_signal(signal_number)


def terminate_and_reap(
    processes: Mapping[str, subprocess.Popen],
    grace_seconds: float = 5.0,
) -> dict[str, int]:
    """Terminate process groups, escalate after a deadline, and reap every child."""
    for process in processes.values():
        _signal_process(process, signal.SIGTERM)

    deadline = time.monotonic() + max(0.0, grace_seconds)
    while time.monotonic() < deadline:
        if all(process.poll() is not None for process in processes.values()):
            break
        time.sleep(0.05)

    for process in processes.values():
        _signal_process(process, signal.SIGKILL)

    return_codes = {}
    for name, process in processes.items():
        try:
            return_codes[name] = process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            _signal_process(process, signal.SIGKILL)
            return_codes[name] = process.wait()
    return return_codes


def wait_fail_fast(
    processes: Mapping[str, subprocess.Popen],
    poll_interval: float = 0.1,
    grace_seconds: float = 5.0,
) -> dict[str, int]:
    """Wait for all children, cancelling siblings as soon as one fails."""
    pending = set(processes)
    return_codes: dict[str, int] = {}
    while pending:
        for name in tuple(pending):
            return_code = processes[name].poll()
            if return_code is None:
                continue
            pending.remove(name)
            return_codes[name] = return_code
            if return_code != 0 and pending:
                terminated = terminate_and_reap(
                    {pending_name: processes[pending_name] for pending_name in pending},
                    grace_seconds=grace_seconds,
                )
                return_codes.update(terminated)
                return return_codes
        if pending:
            time.sleep(max(0.01, poll_interval))
    return return_codes
