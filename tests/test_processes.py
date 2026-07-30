from types import SimpleNamespace

from robotic_renderers import processes


def _fake_process(return_code):
    state = {"return_code": return_code, "signals": []}

    def poll():
        return state["return_code"]

    def wait(timeout=None):
        return state["return_code"]

    process = SimpleNamespace(pid=12345, poll=poll, wait=wait)
    return process, state


def test_wait_fail_fast_terminates_and_reaps_pending_siblings(monkeypatch):
    failed, failed_state = _fake_process(1)
    hanging, hanging_state = _fake_process(None)

    def signal_process(process, signal_number):
        state = failed_state if process is failed else hanging_state
        state["signals"].append(signal_number)
        if process is hanging:
            state["return_code"] = -signal_number

    monkeypatch.setattr(processes, "_signal_process", signal_process)

    return_codes = processes.wait_fail_fast(
        {"failed": failed, "hanging": hanging},
        poll_interval=0.01,
        grace_seconds=0.01,
    )

    assert return_codes["failed"] == 1
    assert return_codes["hanging"] < 0
    assert hanging_state["signals"]
