"""Single public address configuration; credentials stay in receiver.token."""
import json
import os
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from robot_target import configured_robot_host, get_robot_host as resolve_robot_host
DEFAULTS = {"robot_host":"10.42.0.101", "policy_host":"192.168.50.113",
            "monitor_port":8086, "collection_port":8087}
def load():
    path = Path(os.environ.get("VLA_NETWORK_CONFIG", Path(__file__).with_name("network.json")))
    return {**DEFAULTS, **(json.loads(path.read_text()) if path.exists() else {})}
def get_policy_host(): return os.environ.get("VLA_POLICY_HOST", load()["policy_host"])
def get_robot_host(): return resolve_robot_host(load()["robot_host"])
def configure_proxy_bypass(host=None):
    host = host or get_policy_host()
    for key in ("NO_PROXY","no_proxy"):
        values = [p for p in os.environ.get(key,"").split(",") if p]
        for value in [host,get_robot_host(),"127.0.0.1","localhost"]:
            if value not in values: values.append(value)
        os.environ[key] = ",".join(values)
    return host
