"""Shared SSH, rsync, and binary-stream helpers for remote renderers."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from collections.abc import Sequence
from pathlib import Path

_HOST_PATTERN = re.compile(r"^(?:[A-Za-z0-9][A-Za-z0-9_.-]*@)?[A-Za-z0-9][A-Za-z0-9_.-]*$")
DEFAULT_CONNECT_TIMEOUT = 10
DEFAULT_OPERATION_TIMEOUT = 300
DEFAULT_MAX_PAYLOAD_BYTES = 2 * 1024 * 1024 * 1024
REMOTE_PACKAGE_EXCLUDES = ("/data/", "__pycache__/", "*.py[co]")


def ssh_options(connect_timeout: int = DEFAULT_CONNECT_TIMEOUT) -> list[str]:
    """Return consistent noninteractive SSH options for ssh, scp, and rsync."""
    return [
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={max(1, int(connect_timeout))}",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=3",
    ]


def validate_ssh_host(host: str) -> str:
    """Validate a hostname or SSH-config alias before passing it to a subprocess."""
    if not _HOST_PATTERN.fullmatch(host):
        raise ValueError(
            "SSH host must be a hostname, user@hostname, or configured alias containing "
            f"only letters, digits, dots, underscores, and hyphens; got {host!r}"
        )
    return host


def ssh_command(
    host: str,
    remote_command: str,
    *,
    batch_mode: bool = True,
    connect_timeout: int = DEFAULT_CONNECT_TIMEOUT,
) -> list[str]:
    command = ["ssh"]
    if batch_mode:
        command.extend(ssh_options(connect_timeout))
    else:
        command.extend(("-o", f"ConnectTimeout={max(1, int(connect_timeout))}"))
    command.extend((validate_ssh_host(host), remote_command))
    return command


def run_ssh(
    host: str,
    remote_command: str,
    *,
    timeout: float | None = DEFAULT_OPERATION_TIMEOUT,
    check: bool = True,
    **kwargs,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        ssh_command(host, remote_command),
        check=check,
        timeout=timeout,
        **kwargs,
    )


def rsync_remote(host: str, remote_path: str) -> str:
    """Build an rsync remote target after validating the host argument."""
    return f"{validate_ssh_host(host)}:{remote_path}"


def rsync_ssh_transport(connect_timeout: int = DEFAULT_CONNECT_TIMEOUT) -> str:
    """Return the hardened remote shell string consumed by ``rsync -e``."""
    return shlex.join(["ssh", *ssh_options(connect_timeout)])


def rsync_command(
    source: str | Path,
    destination: str,
    *,
    delete: bool = False,
    excludes: Sequence[str] = (),
) -> list[str]:
    """Build a deterministic rsync command for an explicit source tree."""
    command = ["rsync", "-az", "-e", rsync_ssh_transport()]
    if delete:
        command.append("--delete")
    for pattern in excludes:
        command.append(f"--exclude={pattern}")
    command.extend((str(source), destination))
    return command


def resolve_runtime_project(
    package_source: str | Path,
    project_root: str | Path,
    backend: str,
) -> Path:
    """Find a frozen remote runtime in a checkout or an installed wheel."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", backend):
        raise ValueError(f"Unsafe backend runtime name: {backend!r}")
    candidates = (
        Path(project_root) / "runtime" / backend,
        Path(package_source) / "data" / "runtime" / backend,
    )
    for candidate in candidates:
        if (candidate / "pyproject.toml").is_file() and (candidate / "uv.lock").is_file():
            return candidate
    locations = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Frozen {backend} runtime is unavailable; searched: {locations}")


def read_exact(stream, byte_count: int, *, max_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES) -> bytes:
    """Read one bounded binary payload or raise on truncated input."""
    byte_count = int(byte_count)
    if byte_count < 0 or byte_count > max_bytes:
        raise RuntimeError(
            f"Remote payload length {byte_count} is outside the allowed range 0..{max_bytes}"
        )
    chunks = bytearray()
    while len(chunks) < byte_count:
        chunk = stream.read(byte_count - len(chunks))
        if not chunk:
            raise EOFError(
                f"Remote renderer stream ended after {len(chunks)} of {byte_count} bytes"
            )
        chunks.extend(chunk)
    return bytes(chunks)


def parse_linux_process_starttime(stat_line: str) -> str:
    """Extract Linux ``/proc/<pid>/stat`` field 22, including spaced comm values."""
    _, separator, fields = stat_line.rpartition(") ")
    values = fields.split()
    if not separator or len(values) < 20 or not values[19].isdigit():
        raise ValueError("Invalid Linux process stat record")
    return values[19]


def write_pid_identity(path: str | Path) -> Path:
    """Atomically write ``PID starttime`` so cleanup cannot signal a reused PID."""
    path = Path(path)
    pid = os.getpid()
    starttime = parse_linux_process_starttime(Path(f"/proc/{pid}/stat").read_text())
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{pid}.tmp")
    temporary.write_text(f"{pid} {starttime}\n")
    temporary.chmod(0o600)
    os.replace(temporary, path)
    return path


def worker_cleanup_script(
    remote_root: str,
    remote_job: str,
    module_name: str,
    pid_pattern: str,
) -> str:
    """Build Linux cleanup that revalidates PID start time before every signal."""
    safe_value = re.compile(r"^[A-Za-z0-9_. /-]+$")
    for label, value in (
        ("remote root", remote_root),
        ("remote job", remote_job),
        ("module", module_name),
    ):
        if not safe_value.fullmatch(value) or ".." in value.split("/"):
            raise ValueError(f"Unsafe {label}: {value!r}")
    if not re.fullmatch(r"[A-Za-z0-9_.*-]+", pid_pattern) or "/" in pid_pattern:
        raise ValueError(f"Unsafe PID pattern: {pid_pattern!r}")
    return f"""set -u
root="$HOME/{remote_root}/{remote_job}"
module_marker={shlex.quote(module_name)}
job_marker={shlex.quote(remote_job)}

proc_start() {{
  candidate=$1
  test -r "/proc/$candidate/stat" || return 1
  stat_line=$(cat "/proc/$candidate/stat") || return 1
  stat_tail=${{stat_line##*) }}
  set -- $stat_tail
  test -n "${{20:-}}" || return 1
  printf '%s\\n' "${{20}}"
}}

worker_matches() {{
  pid_file=$1
  test -s "$pid_file" || return 1
  read -r candidate expected_start extra < "$pid_file" || return 1
  test -z "${{extra:-}}" || return 1
  case "$candidate:$expected_start" in
    *[!0-9:]*|:*|*:) return 1 ;;
  esac
  test "$candidate" -gt 1 || return 1
  actual_start=$(proc_start "$candidate") || return 1
  test "$actual_start" = "$expected_start" || return 1
  command_line=$(tr '\\000' ' ' < "/proc/$candidate/cmdline") || return 1
  case "$command_line" in
    *"$module_marker"*"$job_marker"*) return 0 ;;
    *) return 1 ;;
  esac
}}

for pid_file in "$root"/{pid_pattern}; do
  worker_matches "$pid_file" || continue
  read -r candidate _ < "$pid_file"
  worker_matches "$pid_file" || continue
  if ! kill -TERM -- "-$candidate" 2>/dev/null; then
    worker_matches "$pid_file" || continue
    read -r candidate _ < "$pid_file"
    worker_matches "$pid_file" || continue
    kill -TERM "$candidate" 2>/dev/null || true
  fi
done

for _ in 1 2 3 4 5; do
  alive=0
  for pid_file in "$root"/{pid_pattern}; do
    worker_matches "$pid_file" && alive=1
  done
  test "$alive" -eq 0 && exit 0
  sleep 0.2
done

for pid_file in "$root"/{pid_pattern}; do
  worker_matches "$pid_file" || continue
  read -r candidate _ < "$pid_file"
  worker_matches "$pid_file" || continue
  if ! kill -KILL -- "-$candidate" 2>/dev/null; then
    worker_matches "$pid_file" || continue
    read -r candidate _ < "$pid_file"
    worker_matches "$pid_file" || continue
    kill -KILL "$candidate" 2>/dev/null || true
  fi
done
"""


def command_with_ssh_options(command: Sequence[str], host_index: int = 1) -> list[str]:
    """Harden a simple ``ssh host ...`` command while retaining its remote arguments."""
    command = list(command)
    if not command or command[0] != "ssh" or len(command) <= host_index:
        raise ValueError(f"Expected an ssh command, got {command!r}")
    host = validate_ssh_host(command[host_index])
    return ["ssh", *ssh_options(), host, *command[host_index + 1 :]]
