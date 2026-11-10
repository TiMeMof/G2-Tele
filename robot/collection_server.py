"""Authenticated robot-local collection, persistence, and review API."""
import argparse
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
import socket
import threading
import time
from urllib.parse import parse_qs, urlsplit
from urllib.request import build_opener, ProxyHandler
from native_collector import NativeCollector
from storage import SessionStore
from record_view import build_episode

BOOT = Path("/proc/sys/kernel/random/boot_id").read_text().strip() if Path("/proc/sys/kernel/random/boot_id").exists() else "test"

def stamp():
    return {"host": socket.gethostname(), "wall_ns": str(time.time_ns()),
            "monotonic_ns": str(time.monotonic_ns()), "boot_id": BOOT}

def read_json(url, timeout=2):
    with build_opener(ProxyHandler({})).open(url, timeout=timeout) as response:
        return json.load(response)

def readiness(collector):
    status = collector.readiness()
    try:
        robot = read_json("http://127.0.0.1:8890/system/status")
        data = robot.get("data", {})
        status["robot_ready"] = (robot.get("success") is True and
             data.get("current_mode") == "remote_env" and data.get("instance_running") is True)
    except Exception as exc:
        status["robot_ready"] = False; status["robot_error"] = str(exc)
    try: status["vr"] = read_json("http://127.0.0.1:8086/api/vr/status")
    except Exception as exc: status["vr"] = {"fresh": False, "reason": str(exc)}
    return status

def handler_for(store, token, ready=readiness):
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup(); self.connection.settimeout(20)

        def reply(self, code, value):
            data = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            try: self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError): pass

        def authorized(self):
            supplied = self.headers.get("Authorization", "")
            if not hmac.compare_digest(supplied, "Bearer " + token):
                self.reply(401, {"success": False, "message": "unauthorized"}); return False
            return True

        def body(self):
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 1_048_576: raise ValueError("请求大小无效")
            value = json.loads(self.rfile.read(size))
            if not isinstance(value, dict): raise ValueError("请求须为JSON对象")
            return value

        def do_GET(self):
            if not self.authorized(): return
            split = urlsplit(self.path); parts = split.path.strip("/").split("/")
            try:
                if split.path == "/api/health":
                    return self.reply(200, {"success": True, **ready(store.collector),
                                           "storage_root": str(store.root), "robot_control": False})
                if split.path == "/api/clock":
                    received = stamp()
                    return self.reply(200, {"success": True, "received": received, "sent": stamp()})
                if split.path == "/api/records":
                    filters = {k:v[-1] for k,v in parse_qs(split.query).items()}
                    return self.reply(200, store.list_sessions(filters))
                if len(parts) == 3 and parts[:2] == ["api", "records"]:
                    return self.reply(200, {"success": True, "episode": store.detail(parts[2])})
                if len(parts) == 5 and parts[:2] == ["api", "records"] and parts[3] == "chunks":
                    records = store.records(parts[2]); cid = parts[4]
                    matches = [r for r in records if r["kind"] == "chunk" and r["data"].get("chunk_id") == cid]
                    if not matches: raise FileNotFoundError("动作块不存在")
                    dispatches = [r for r in records if r["kind"] == "dispatch" and r["data"].get("chunk_id") == cid]
                    return self.reply(200, {"success": True, "record": matches[0], "dispatches": dispatches})
                self.reply(404, {"success": False, "message": "not_found"})
            except FileNotFoundError as exc: self.reply(404, {"success": False, "message": str(exc)})
            except (ValueError, KeyError, OSError) as exc: self.reply(400, {"success": False, "message": str(exc)})

        def do_POST(self):
            if not self.authorized(): return
            parts = urlsplit(self.path).path.strip("/").split("/")
            try:
                body = self.body()
                if body.get("data_dir") is not None:
                    requested=Path(body["data_dir"])
                    if not requested.is_absolute() or (requested/"sessions").resolve()!=store.root:
                        raise ValueError("采集根目录须与机器人配置一致："+str(store.root.parent))
                if parts == ["api", "probe"]:
                    return self.reply(200, {"success": True, "directory": str(store.root)})
                if parts == ["api", "sessions"]:
                    status = ready(store.collector)
                    vr = status.get("vr", {})
                    if (not status.get("ready") or not status.get("robot_ready") or
                            vr.get("fresh") is not True or vr.get("takeover") is not False):
                        raise ValueError("机器人、录制器或VR尚未就绪")
                    value = store.create(body["episode_id"], body["task_id"], body["category"], body.get("policy"))
                    if value["collection_state"] == "preparing": value = store.start(body["episode_id"])
                    return self.reply(200, {"success": True, "session": value})
                if parts == ["api", "records"]:
                    return self.reply(200, {"success": True, **store.put_record(body["record"])})
                if len(parts) == 4 and parts[:2] == ["api", "sessions"] and parts[3] == "stop":
                    return self.reply(200, {"success": True, "session": store.stop(parts[2],body.get("expected_sequence"))})
                if len(parts) == 4 and parts[:2] == ["api", "records"] and parts[3] == "annotations":
                    detail = store.annotate(parts[2], body["round"], body.get("reasons", []),
                                            body.get("note", ""), body["revision"])
                    return self.reply(200, {"success": True, "episode": detail})
                if len(parts) == 4 and parts[:2] == ["api", "records"] and parts[3] == "finalize":
                    return self.reply(200, {"success": True,
                         "episode": store.finalize(parts[2], body["revision"])})
                self.reply(404, {"success": False, "message": "not_found"})
            except FileNotFoundError as exc: self.reply(404, {"success": False, "message": str(exc)})
            except (ValueError, KeyError) as exc: self.reply(409, {"success": False, "message": str(exc)})
            except Exception as exc: self.reply(503, {"success": False, "message": str(exc)})

        def do_DELETE(self): self.reply(405, {"success": False})
        do_PUT = do_DELETE
        def log_message(self, *args): pass
    return Handler

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8087)
    parser.add_argument("--root", default="/data/yfmo/sessions")
    parser.add_argument("--token-file", default=str(Path(__file__).with_name("receiver.token")))
    args = parser.parse_args()
    token_path = Path(args.token_file)
    token = token_path.read_text().strip()
    if len(token) < 32: raise ValueError("接收凭据长度不足")
    if os.name != "nt" and token_path.stat().st_mode & 0o077:
        raise ValueError("接收凭据权限须为0600")
    store = SessionStore(args.root, NativeCollector(), build_episode)
    server = ThreadingHTTPServer((args.host, args.port), handler_for(store, token))
    def shutdown(*unused):
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, shutdown); signal.signal(signal.SIGINT, shutdown)
    print(json.dumps({"service": "takeover-collection/2", "port": args.port,
                      "root": args.root, "motion_control": False}), flush=True)
    try: server.serve_forever(poll_interval=.2)
    finally: server.server_close()

if __name__ == "__main__": main()
