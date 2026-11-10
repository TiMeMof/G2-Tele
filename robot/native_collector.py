"""Bridge to the robot's installed DDS recorder and camera recorders.

No request is published on import, readiness checks, or service startup.
Only explicit start()/stop() publish recording requests, never motion requests.
"""
import importlib
import json
import sys
import threading
import time
from pathlib import Path

CAMERAS = ("head_color", "hand_left_color", "hand_right_color",
           "head_stereo_left", "head_stereo_right", "head_left_fisheye",
           "head_right_fisheye", "head_back_fisheye")

class NativeCollector:
    def __init__(self, sdk_dir="/home/agi/app/gdk/lib", max_duration_ms=3_600_000,
                 connect=None, timeout=12, topics=None):
        self.sdk_dir = Path(sdk_dir)
        self.max_duration_ms = max_duration_ms
        self.timeout = timeout
        self.connect_factory = connect
        self.topics = topics
        self.connections = None
        self.lock = threading.RLock()

    def readiness(self):
        needed = [self.sdk_dir / "dlb_msgs_pb/dlb_record_pb2.py",
                  Path("/home/agi/app/conf/record/record_topic.json")]
        return {"ready": all(p.exists() for p in needed),
                "backend": "native_dds_and_cameras",
                "required_files": [str(p) for p in needed],
                "max_duration_ms": self.max_duration_ms}

    def _connect(self):
        if self.connections is not None: return
        if self.connect_factory:
            self.g, self.proto = self.connect_factory()
        else:
            sys.path.insert(0, str(self.sdk_dir))
            self.g = importlib.import_module("agibot_gdk")
            self.proto = importlib.import_module("dlb_msgs_pb.dlb_record_pb2")
        qos = self.g.QoS()
        qos.history = self.g.QoSHistory.kKeepLast
        qos.depth = 32
        qos.reliability = self.g.QoSReliability.kReliable
        qos.durability = self.g.QoSDurability.kVolatile
        self.connections = {}
        response_qos = self.g.QoS()
        response_qos.history = self.g.QoSHistory.kKeepLast
        response_qos.depth = 32
        response_qos.reliability = self.g.QoSReliability.kBestEffort
        response_qos.durability = self.g.QoSDurability.kVolatile
        for stream in ("dds", "video"):
            prefix = "/dlb_msgs/record_" + stream
            self.connections[stream] = (
                self.g.Publisher(prefix + "_request", "dlb_msgs.RecordRequest", qos),
                self.g.Subscriber(prefix + "_response", "dlb_msgs.RecordResponse", response_qos))
        # Allow discovery before the first explicit collection request.
        time.sleep(.4)

    def properties(self, sid, raw):
        topics = self.topics
        if topics is None:
            config = json.loads(Path("/home/agi/app/conf/record/record_topic.json").read_text())
            topics = [entry["name"] for entry in config]
        if not topics or not all(isinstance(t,str) and t.startswith("/") for t in topics):
            raise ValueError("原生录制话题配置无效")
        return {
            "dds": [{"key": topic, "storage_folder": str(raw / "record")} for topic in topics],
            "video": [{"key": c, "storage_folder": str(raw / "camera" / c)} for c in (*CAMERAS,"head_depth")],
        }

    def request(self, sid, properties, stop=False):
        message = self.proto.RecordRequest()
        message.uuid = sid
        message.command = self.proto.RecordRequest.STOP if stop else self.proto.RecordRequest.START
        message.timestamp_utc = time.time_ns()
        # The current robot exposes no valid PTP device and its raw headers use
        # UTC with a zero PTP field. Do not fabricate PTP by adding the TAI offset.
        message.timestamp_ptp = 0
        for values in properties:
            prop = message.record_props.add()
            prop.key = values["key"]
            prop.storage_folder = values["storage_folder"]
            prop.pre_win = 0
            prop.post_win = self.max_duration_ms
        return message

    def _publish(self, sid, properties, stop):
        for stream, props in properties.items():
            if not props: continue
            info = self.g.PubMsgInfo()
            info.source_ts_ns = time.time_ns()
            info.frame_id = int(time.monotonic_ns() & 0x7fffffff)
            result = self.connections[stream][0].publish(self.request(sid, props, stop), info)
            if result is False: raise RuntimeError("录制请求发送失败：" + stream)

    def _responses(self, sid):
        found = {}
        for stream, pair in self.connections.items():
            for _ in range(64):
                raw, _info = pair[1].subscribe(False)
                if not raw: break
                response = self.proto.RecordResponse()
                response.ParseFromString(bytes(raw))
                if response.req_uuid != sid: continue
                result = {"result": int(response.result), "error": response.err_msg,
                          "failures": [{"key": p.key, "error": p.err_msg} for p in response.fail_infos]}
                if result["result"] != self.proto.RecordResponse.SUCCESS:
                    raise RuntimeError("原生录制失败：" + stream + " " + str(result))
                found[stream] = result
        return found

    def start(self, sid, raw_directory):
        with self.lock:
            self._connect()
            raw = Path(raw_directory)
            props = self.properties(sid, raw)
            for items in props.values():
                for p in items: Path(p["storage_folder"]).mkdir(parents=True, exist_ok=True)
            self._publish(sid, props, False)
            deadline = time.monotonic() + self.timeout
            responses = {}
            try:
                while time.monotonic() < deadline:
                    responses.update(self._responses(sid))
                    dds_ok = any(p.stat().st_size > 8 for p in (raw / "record").glob("*.pbdat"))
                    video_ok = all(any(p.stat().st_size > 0 for p in (raw / "camera" / c).glob("*.h265")) for c in CAMERAS)
                    depth_ok = any(p.stat().st_size > 0 for p in (raw / "camera/head_depth").glob("*.zst"))
                    if dds_ok and video_ok and depth_ok:
                        return {"uuid": sid, "properties": props, "raw_directory": str(raw),
                                "start_confirmed_by": "native_files_created_with_payload",
                                "responses": responses, "max_duration_ms": self.max_duration_ms}
                    time.sleep(.1)
                raise TimeoutError("原始消息、彩色视频或深度文件未就绪，未开始模型执行")
            except Exception:
                # Best effort stops only this request's own recording keys.
                try: self._publish(sid, props, True)
                except Exception: pass
                raise

    def stop(self, sid, receipt):
        with self.lock:
            self._connect()
            props = receipt.get("properties")
            if not props: raise ValueError("缺少本任务的原生录制标识，不能猜测停止目标")
            self._responses(sid)  # discard any earlier START completion responses
            self._publish(sid, props, True)
            found = {}; deadline = time.monotonic() + self.timeout
            while time.monotonic() < deadline:
                found.update(self._responses(sid))
                if set(found) >= {k for k,v in props.items() if v}:
                    return {"uuid": sid, "stop_confirmed_by": "matching_native_responses",
                            "responses": found, "wall_ns": str(time.time_ns())}
                time.sleep(.05)
            raise TimeoutError("未收到所有录制器的结束确认；保留数据并标为待核查，不能归档")
