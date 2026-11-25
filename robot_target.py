"""Resolve the robot address saved by the evaluation page without contacting it."""
import ipaddress
import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
WEB_CONFIG_PATH = BASE_DIR / "intergrated_test" / "web_settings.json"
SETUP_PATH = BASE_DIR / "config" / "clothes_robot_setup.json"


def normalize_robot_host(value):
    if not isinstance(value, str):
        raise ValueError("请填写有效的机器人 IPv4 地址")
    try:
        address = ipaddress.IPv4Address(value.strip())
    except ipaddress.AddressValueError as exc:
        raise ValueError("请填写有效的机器人 IPv4 地址，例如 192.168.50.74") from exc
    if address.is_unspecified or address.is_multicast or address.is_loopback or int(address) == 0xffffffff:
        raise ValueError("请填写机器人的实际 IPv4 地址")
    return str(address)


def configured_robot_host():
    if WEB_CONFIG_PATH.exists():
        value = json.loads(WEB_CONFIG_PATH.read_text(encoding="utf-8")).get("robot_host")
        if value is not None:
            return normalize_robot_host(value)
    value = os.environ.get("VLA_ROBOT_HOST")
    return normalize_robot_host(value) if value else None


def get_robot_host(default=None):
    host = configured_robot_host()
    if host:
        return host
    if default is None:
        default = json.loads(SETUP_PATH.read_text(encoding="utf-8"))["robot"]["host"]
    return normalize_robot_host(default)
