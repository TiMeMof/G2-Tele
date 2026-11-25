"""Read-only fixed-destination proxy. Never calls robot control endpoints."""
import http.client
import json
import math
import time

from network_config import get_robot_host, load
MONITOR_PORT = load()["monitor_port"]


def fetch_status():
    started = time.monotonic()
    conn = http.client.HTTPConnection(get_robot_host(), MONITOR_PORT, timeout=.7)
    try:
        conn.request("GET", "/api/vr/status", headers={"Connection": "close"})
        response = conn.getresponse()
        raw = response.read(65537)
        if response.status != 200 or len(raw) > 65536:
            raise ValueError("invalid monitor response")
        data = json.loads(raw)
        if data.get("protocol") != "vr-monitor/1" or data.get("threshold") != 0.2 or data.get("comparison") != ">=":
            raise ValueError("monitor protocol mismatch")
        if type(data.get("fresh")) is not bool or data.get("takeover") is not None and type(data.get("takeover")) is not bool:
            raise ValueError("invalid monitor state")
        if data.get("stale_ms") != 250 or not isinstance(data.get("instance_id"), str):
            raise ValueError("invalid monitor metadata")
        rate = data.get("rate_hz")
        if type(rate) not in (int,float) or not math.isfinite(rate) or rate < 0:
            raise ValueError("invalid rate")
        if data["fresh"]:
            for side in ("left", "right"):
                hand = data["hands"][side]
                for key in ("arm", "gripper"):
                    if type(hand[key]) not in (int,float) or not math.isfinite(hand[key]) or not 0 <= hand[key] <= 1:
                        raise ValueError("invalid hand input")
                if type(hand.get("active")) is not bool or hand["active"] != (hand["arm"] >= .2):
                    raise ValueError("invalid hand state")
            expected = any(data["hands"][side]["active"] for side in ("left","right"))
            if data["takeover"] is not None and data["takeover"] != expected:
                raise ValueError("invalid takeover state")
        age = data.get("age_ms")
        if age is not None and (type(age) not in (int,float) or not math.isfinite(age) or age < 0):
            raise ValueError("invalid frame age")
        elapsed_ms = (time.monotonic()-started)*1000
        data["transport_ms"] = elapsed_ms
        if age is not None:
            data["age_ms"] = age + elapsed_ms  # conservative upper bound; no cross-host clock subtraction
        if data["fresh"] and (age is None or data["age_ms"] > 250):
            data.update(fresh=False, takeover=None, reason="transport_stale", hands={"left":None,"right":None})
        if not data["fresh"]:
            data["takeover"] = None
        return 200, data
    except (OSError, ValueError, TypeError, AttributeError, KeyError, http.client.HTTPException):
        return 503, {"success": False, "fresh": False, "takeover": None,
                     "message": "\u65e0\u6cd5\u8bfb\u53d6 机器人 \u7684 VR \u89c2\u5bdf\u670d\u52a1\uff0c\u8bf7\u68c0\u67e5 机器人 \u7684 8086 \u670d\u52a1\u662f\u5426\u542f\u52a8\u3002", "reason": "monitor_unavailable"}
    finally:
        conn.close()
