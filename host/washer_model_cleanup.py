"""Close only the local OpenPI listener captured for a washer episode."""

from dataclasses import asdict, dataclass
from ipaddress import ip_address
import json
import os
from pathlib import Path
import re
import select
import signal
import subprocess
import sys
from time import monotonic


_PROC = Path("/proc")
_SYSTEM_PYTHON = "/usr/bin/python3"
_SS = "/usr/bin/ss"
_IP = "/usr/sbin/ip"
_QUERY_S = 0.35
_CLOSE_S = 3.0
_PID_PATTERN = re.compile(r"\bpid=(\d+),fd=\d+\b")
_ERRORS = (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError)


@dataclass(frozen=True)
class PolicyProcessIdentity:
    pid: int
    start_ticks: int
    host: str
    port: int
    checkpoint: str
    asset_id: str
    config: str


def _policy_fields(policy):
    host, port = policy["host"], policy["port"]
    checkpoint, asset_id, config = policy["checkpoint"], policy["asset_id"], policy["config"]
    if not isinstance(host, str):
        raise ValueError("invalid washer model host")
    if host != "localhost":
        host = str(ip_address(host))
    if (type(port) is not int or not 1 <= port <= 65535 or port == 8084
            or not all(isinstance(value, str) and value for value in (checkpoint, asset_id, config))
            or not os.path.isabs(checkpoint)):
        raise ValueError("invalid washer model identity")
    return {"host": host, "port": port, "checkpoint": os.path.realpath(checkpoint),
            "asset_id": asset_id, "config": config}


def _listener_rows(port, timeout):
    result = subprocess.run(
        [_SS, "-H", "-ltnp", f"sport = :{port}"], capture_output=True, text=True,
        check=True, timeout=timeout,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def _local_address(host, timeout):
    result = subprocess.run(
        [_IP, "-j", "address", "show"], capture_output=True, text=True,
        check=True, timeout=timeout,
    )
    return any(info.get("local") == host for interface in json.loads(result.stdout)
               for info in interface.get("addr_info", []))


def _listener_pid(host, port, deadline):
    rows = _listener_rows(port, _query_time(deadline))
    if not rows:
        return None
    if len(rows) != 1:
        raise ValueError("washer model listener is ambiguous")
    fields = rows[0].split(maxsplit=5)
    if len(fields) != 6 or fields[0] != "LISTEN":
        raise ValueError("washer model listener has no visible owner")
    address, _, endpoint_port = fields[3].rpartition(":")
    address = address.removeprefix("[").removesuffix("]")
    if endpoint_port != str(port):
        raise ValueError("washer model listener port mismatch")
    if host == "localhost":
        matches_host = address in {"127.0.0.1", "::1", "0.0.0.0", "::"}
    else:
        matches_host = address == host
        if address in {"0.0.0.0", "::"}:
            matches_host = (ip_address(address).version == ip_address(host).version
                            and _local_address(host, _query_time(deadline)))
    owners = {int(pid) for pid in _PID_PATTERN.findall(fields[5])}
    if not matches_host or len(owners) != 1:
        raise ValueError("washer model listener owner mismatch")
    pid = owners.pop()
    if pid <= 1 or pid == os.getpid():
        raise ValueError("invalid washer model owner")
    return pid


def _query_time(deadline):
    remaining = deadline - monotonic()
    if remaining <= 0:
        raise TimeoutError("washer model verification timed out")
    return min(_QUERY_S, remaining)


def _start_ticks(pid):
    # The command name may contain spaces or parentheses. Field 22 follows
    # the last closing parenthesis and is the twentieth field in that suffix.
    text = (_PROC / str(pid) / "stat").read_text()
    fields = text.rpartition(") ")[2].split()
    if len(fields) <= 19:
        raise ValueError("malformed washer model process stat")
    return int(fields[19])


def _argument(argv, flag):
    values = []
    for index, value in enumerate(argv):
        if value == flag and index + 1 < len(argv):
            values.append(argv[index + 1])
        elif value.startswith(flag + "="):
            values.append(value[len(flag) + 1:])
    return values[0] if len(values) == 1 else None


def _matches_process(pid, expected):
    process = _PROC / str(pid)
    if process.stat().st_uid != os.geteuid():
        return False
    argv = [os.fsdecode(value) for value in (process / "cmdline").read_bytes().split(b"\0") if value]
    if not argv or not re.fullmatch(r"python(?:\d+(?:\.\d+)?)?", Path(argv[0]).name):
        return False
    script = next((index for index, value in enumerate(argv[1:], 1) if not value.startswith("-")), None)
    if script is None or Path(argv[script]).name != "serve_policy.py":
        return False
    args = argv[script + 1:]
    checkpoint = _argument(args, "--policy.dir")
    if not checkpoint or "policy:checkpoint" not in args:
        return False
    if not os.path.isabs(checkpoint):
        checkpoint = str((process / "cwd").resolve(strict=True) / checkpoint)
    return (
        os.path.realpath(checkpoint) == expected["checkpoint"]
        and _argument(args, "--port") == str(expected["port"])
        and _argument(args, "--policy.asset-id") == expected["asset_id"]
        and _argument(args, "--policy.config") == expected["config"]
    )


def capture_policy_process(policy):
    """Capture the exact local listener identity before the episode starts."""
    try:
        expected = _policy_fields(policy)
        deadline = monotonic() + 4 * _QUERY_S
        pid = _listener_pid(expected["host"], expected["port"], deadline)
        if pid is None:
            return None
        start_ticks = _start_ticks(pid)
        if (not _matches_process(pid, expected)
                or _listener_pid(expected["host"], expected["port"], deadline) != pid
                or _start_ticks(pid) != start_ticks):
            return None
        return PolicyProcessIdentity(pid=pid, start_ticks=start_ticks, **expected)
    except _ERRORS:
        return None


def _port_free(port, deadline):
    return not _listener_rows(port, _query_time(deadline))


def _close_captured(identity, expected):
    # This function runs in system Python, which provides Linux pidfd_open.
    deadline = monotonic() + _CLOSE_S - 0.2
    try:
        pidfd = os.pidfd_open(identity.pid)
    except ProcessLookupError:
        return _port_free(identity.port, deadline)
    try:
        if select.select([pidfd], [], [], 0)[0]:
            return _port_free(identity.port, deadline)
        try:
            current_start = _start_ticks(identity.pid)
        except FileNotFoundError:
            return bool(select.select([pidfd], [], [], 0)[0]) and _port_free(identity.port, deadline)
        if current_start != identity.start_ticks:
            return _port_free(identity.port, deadline)
        if not _matches_process(identity.pid, expected):
            return False
        if _listener_pid(identity.host, identity.port, deadline) != identity.pid:
            return bool(select.select([pidfd], [], [], 0)[0]) and _port_free(identity.port, deadline)
        try:
            signal.pidfd_send_signal(pidfd, signal.SIGTERM)
        except ProcessLookupError:
            return _port_free(identity.port, deadline)
        remaining = max(0.0, deadline - monotonic() - _QUERY_S)
        if not select.select([pidfd], [], [], remaining)[0]:
            return False
        return _port_free(identity.port, deadline)
    finally:
        os.close(pidfd)


def close_policy_process(identity, policy):
    """End a captured listener and confirm its process and listening port are gone."""
    if not isinstance(identity, PolicyProcessIdentity):
        return False
    try:
        expected = _policy_fields(policy)
        if any(getattr(identity, key) != value for key, value in expected.items()):
            return False
        result = subprocess.run(
            [_SYSTEM_PYTHON, "-B", str(Path(__file__).resolve()), "--close-captured"],
            input=json.dumps(asdict(identity)), text=True, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, check=False, timeout=_CLOSE_S,
        )
        return result.returncode == 0
    except _ERRORS:
        return False


def _main():
    if sys.argv[1:] != ["--close-captured"]:
        return 2
    try:
        payload = json.load(sys.stdin)
        identity = PolicyProcessIdentity(**payload)
        if (type(identity.pid) is not int or identity.pid <= 1 or identity.pid == os.getpid()
                or type(identity.start_ticks) is not int or identity.start_ticks <= 0):
            return 1
        return 0 if _close_captured(identity, _policy_fields(payload)) else 1
    except (*_ERRORS, AttributeError):
        return 1


if __name__ == "__main__":
    sys.exit(_main())
