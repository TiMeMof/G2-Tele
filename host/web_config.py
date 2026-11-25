"""Persistent web preferences. This module never imports a robot client."""
import json
import os
import posixpath
import tempfile
from pathlib import Path
from network_config import get_robot_host
from robot_target import normalize_robot_host

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "web_settings.json"
# A1 provides preferences only. A2/A3/A4 must implement the runtime before enabling it.
CAPABILITIES = {"human_takeover": False, "data_collection": False, "timeline": False}


def defaults():
    return {"human_takeover_enabled": True,
            "data_dir": "/data/yfmo",
            "max_interventions": 3,
            "robot_host": get_robot_host()}


def normalize(body):
    required = {"human_takeover_enabled", "data_dir", "max_interventions"}
    if not isinstance(body, dict) or set(body) not in (required, required | {"robot_host"}):
        raise ValueError("配置必须包含人工接管开关、数据目录、接管次数和机器人 IP")
    enabled = body["human_takeover_enabled"]
    if type(enabled) is not bool:
        raise ValueError("人工接管开关必须是布尔值")
    limit = body["max_interventions"]
    if limit is not None and (type(limit) is not int or limit < 0):
        raise ValueError("接管次数必须为非负整数，或选择不限次数")
    raw = body["data_dir"]
    if not isinstance(raw, str) or not raw.strip() or "\x00" in raw:
        raise ValueError("请填写服务器上的绝对目录")
    # This is a path on robot 226, not on the web-server host. Never mkdir here.
    directory = raw.strip()
    if not directory.startswith("/") or "\\" in directory:
        raise ValueError("数据目录必须是机器人61上的绝对路径")
    return {"human_takeover_enabled": enabled, "data_dir": posixpath.normpath(directory),
            "max_interventions": limit,
            "robot_host": normalize_robot_host(body.get("robot_host", get_robot_host()))}


def load():
    if not CONFIG_PATH.exists():
        return defaults()
    return normalize(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))


def save(body):
    config = normalize(body)
    fd, temporary = tempfile.mkstemp(prefix=".web-settings-", suffix=".tmp", dir=BASE_DIR)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(config, output, ensure_ascii=False, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, CONFIG_PATH)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return config
