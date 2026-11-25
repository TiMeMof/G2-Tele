"""Route-scoped model ownership and stopped evidence for 8084 HomeCare washer requests."""

import json
import os
from pathlib import Path
from time import monotonic, sleep, time_ns

from washer_tasks import CONFIG_PATH, SETTLE_BUDGET_S
from network_config import get_robot_host

# Share launch, listener ownership and cleanup with the existing 5090 client.
from agentos21.providers.agibot.washer_policy_process import (
    CLEANUP_BUDGET_S, WasherPolicyConfig, WasherPolicyError, WasherPolicyProcess,
)
from agentos21.providers.agibot.washer_task_client import _motion_sample


def managed_readiness(config):
    managed = config["managedPolicy"]
    if not managed["enabled"]:
        return "MANAGED_WASHER_DISABLED"
    latch = Path(managed["motionLatchPath"])
    if latch.exists():
        return "WASHER_MOTION_UNKNOWN"
    if not latch.parent.is_dir() or not os.access(latch.parent, os.W_OK):
        return "WASHER_MOTION_LATCH_UNAVAILABLE"
    return "READY"


def model_runner(config):
    values = config["managedPolicy"]
    return WasherPolicyProcess(WasherPolicyConfig(
        openpi_dir=values["openpiDir"], python_path=values["pythonPath"],
        task_config_path=str(CONFIG_PATH), startup_timeout_s=values["startupTimeoutS"],
    ))


def wait_for_stopped(*, cancelled=lambda: False):
    """Read .61 only. Stopped samples are not door, garment or tray evidence."""
    deadline = monotonic() + SETTLE_BUDGET_S
    previous = None
    quiet_since = None
    while not cancelled() and (remaining := deadline - monotonic()) > 0:
        try:
            stamp, velocity = _motion_sample(get_robot_host(), 8891, min(0.5, remaining))
        except (OSError, ValueError):
            previous = None
            quiet_since = None
        else:
            observed_at = monotonic()
            consecutive = previous is not None and 0 < stamp - previous <= 750_000_000
            if velocity > 0.02:
                quiet_since = None
            elif quiet_since is None or not consecutive:
                quiet_since = observed_at
            elif observed_at - quiet_since >= 0.35:
                return True
            previous = stamp
        sleep(min(0.15, max(0.0, deadline - monotonic())))
    return False


def _sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def claim_motion(config, task_id):
    """Persist before Popen. Restart or any ambiguous result leaves this latch present."""
    path = Path(config["managedPolicy"]["motionLatchPath"])
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as handle:
        json.dump({"taskId": task_id, "robotHost": get_robot_host(),
                   "startedAtNs": time_ns()}, handle, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    _sync_directory(path.parent)


def clear_motion(config):
    """Call only when no runner launched, or when completion and settling are verified."""
    path = Path(config["managedPolicy"]["motionLatchPath"])
    path.unlink()
    _sync_directory(path.parent)
