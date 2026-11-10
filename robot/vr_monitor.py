#!/usr/bin/env python3
"""In-memory, subscriber-only VR monitor. No robot control or episode recording."""
import argparse
from collections import deque
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import signal
import sys
import threading
import time
import uuid
from vr_core import decode_vr, decode_network, SCHEMA_FILE_SHA256

THRESHOLD = 0.2
STALE_MS = 250
PROTOCOL = "vr-monitor/1"
SCHEMAS = {
    "rh_msgs_v4_pb/robot_frame_data_pb2.py": SCHEMA_FILE_SHA256,
    "genie_msgs_pb/msg/VRNetwork_pb2.py": "350a8060da9964faf5b9ef95f3a156871857279d0df8a46bf4231dc1633b6ae7",
}
TOPICS = {"vr": ("/remote/vr_data", "rh_msgs_v4.RobotFrameData"),
          "network": ("/remote/vr_network", "genie_msgs.msg.pb.VRNetwork")}


class MonitorState:
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.lock = threading.Lock()
        self.instance_id = uuid.uuid4().hex
        self.running = True
        self.error = None
        self.vr_error = "waiting_for_vr"
        self.frames = 0
        self.last_rx = None
        self.source_identity = None
        self.source_payload = None
        self.transport = None
        self.hands = {}
        self.times = deque(maxlen=256)
        self.edges = deque(maxlen=32)
        self.edge_seq = 0
        self.network = None
        self.network_rx = None
        self.last_network_transport = None

    def fail(self, message):
        with self.lock:
            self.error = message
            self.running = False

    def feed(self, name, payload, publisher, sequence):
        now = self.clock()
        with self.lock:
            if name == "network":
                identity = (publisher, sequence)
                if identity == self.last_network_transport:
                    return
                self.last_network_transport = identity
                try:
                    self.network = decode_network(payload)
                    self.network_rx = now
                except (ValueError, UnicodeError):
                    self.network = None
                    self.network_rx = None
                return
            identity = (publisher, sequence)
            if identity == self.transport:
                return
            previous_transport = self.transport
            self.transport = identity
            try:
                data = decode_vr(payload)
            except ValueError:
                self.vr_error = "decode_error"
                return
            self.frames += 1
            source = (publisher, int(data["frame_index_u64"]), int(data["timestamp_raw_u64"]))
            if source == self.source_identity:
                if payload != self.source_payload:
                    self.vr_error = "conflicting_device_frame"
                return  # native forwarder repeat: do not refresh freshness
            previous_source = self.source_identity
            self.source_identity = source
            self.source_payload = bytes(payload)
            if source[2] == 0 or not data["both_hands_present"]:
                self.vr_error = "invalid_or_missing_hand"
                return
            if previous_transport and publisher == previous_transport[0] and sequence <= previous_transport[1]:
                self.vr_error = "out_of_order"
                return
            if previous_source and (publisher != previous_source[0] or source[1] <= previous_source[1] or source[2] < previous_source[2]):
                self.vr_error = "source_restarted"
                return
            previous_hands = self.hands if self._fresh(now) else {}
            self.hands = {d["side"]: {"arm": d["hand_trig"], "gripper": d["index_trig"],
                                      "active": d["hand_trig"] >= THRESHOLD}
                          for d in data["devices"] if d["side"] in ("left", "right")}
            self.last_rx = now
            self.times.append(now)
            self.vr_error = None
            for side, hand in self.hands.items():
                previous = previous_hands.get(side)
                if previous is None or previous["active"] != hand["active"]:
                    self.edge_seq += 1
                    self.edges.append({"seq": self.edge_seq, "at": now, "side": side,
                                       "active": hand["active"],
                                       "kind": "initial_state" if previous is None else "edge"})

    def _fresh(self, now):
        return self.running and self.error is None and self.vr_error is None and self.last_rx is not None and (now-self.last_rx)*1000 <= STALE_MS

    def snapshot(self):
        now = self.clock()
        with self.lock:
            age = None if self.last_rx is None else max(0, (now-self.last_rx)*1000)
            fresh = self._fresh(now)
            reason = self.error or self.vr_error or (None if fresh else "vr_stale")
            net_age = None if self.network_rx is None else max(0,(now-self.network_rx)*1000)
            connected = self.network.get("is_connected") if self.network is not None and net_age <= 2500 else None
            # A fresh explicit disconnection contradicts button ownership.
            known = fresh and connected is not False
            hands = self.hands if fresh else {"left": None, "right": None}
            takeover = any(h["active"] for h in self.hands.values()) if known else None
            valid_times = [x for x in self.times if now-x <= 2]
            hz = (len(valid_times)-1)/(valid_times[-1]-valid_times[0]) if len(valid_times)>1 and valid_times[-1]>valid_times[0] else 0.0
            return {"success": True, "protocol": PROTOCOL, "instance_id": self.instance_id,
                    "sample_id": str(self.source_identity[1]) if self.source_identity else None,
                    "threshold": THRESHOLD, "comparison": ">=", "stale_ms": STALE_MS,
                    "fresh": fresh, "reason": "vr_disconnected" if fresh and connected is False else reason,
                    "takeover": takeover, "hands": hands, "age_ms": age,
                    "received_frames": self.frames, "rate_hz": hz,
                    "connection": {"connected": connected, "age_ms": net_age},
                    "recent_edges": [{k:v for k,v in e.items() if k != "at"} | {"age_ms": (now-e["at"])*1000}
                                    for e in self.edges if now-e["at"] <= 30],
                    "control_ready": False, "semantics": "vr_button_threshold_only"}


def observe(state, stop):
    g = None
    subscribers = {}
    try:
        for relative, expected in SCHEMAS.items():
            if hashlib.sha256((Path("/home/agi/app/gdk/lib")/relative).read_bytes()).hexdigest() != expected:
                raise RuntimeError("schema_changed")
        sys.path.insert(0, "/home/agi/app/gdk/lib")
        import agibot_gdk as g
        qos = g.QoS()
        qos.history = g.QoSHistory.kKeepLast
        qos.depth = 100
        qos.reliability = g.QoSReliability.kBestEffort
        qos.durability = g.QoSDurability.kVolatile
        for name, (topic, typ) in TOPICS.items():
            subscribers[name] = g.Subscriber(topic, typ, qos)
        while not stop.is_set():
            for name, subscriber in subscribers.items():
                data, info = subscriber.subscribe(False)
                if data:
                    state.feed(name, bytes(data), int(info.publish_id), int(info.seq))
            stop.wait(.002)
    except Exception as exc:
        state.fail(type(exc).__name__ + ": " + str(exc))
    finally:
        subscribers.clear()
        if g is not None:
            g.release_dds()
        with state.lock:
            state.running = False


def handler_for(state):
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(2)

        def reply(self, status, body):
            data = json.dumps(body, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            if self.path == "/api/vr/status":
                self.reply(200, state.snapshot())
            else:
                self.reply(404, {"success": False})

        def do_POST(self):
            self.reply(405, {"success": False, "message": "read_only"})
        do_PUT = do_POST
        do_DELETE = do_POST
        do_PATCH = do_POST

        def log_message(self, *args):
            pass  # polling should not produce an unbounded log
    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8086)
    args = ap.parse_args()
    state, stop = MonitorState(), threading.Event()
    server = ThreadingHTTPServer((args.host, args.port), handler_for(state))
    worker = threading.Thread(target=observe, args=(state, stop), daemon=True)
    worker.start()
    def shutdown(signum, frame):
        stop.set()
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    print(json.dumps({"service": PROTOCOL, "listen": [args.host,args.port], "read_only": True}), flush=True)
    try:
        server.serve_forever(poll_interval=.2)
    finally:
        stop.set()
        server.server_close()
        worker.join(timeout=3)


if __name__ == "__main__":
    main()
