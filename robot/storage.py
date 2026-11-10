"""Robot-local classified sessions. Raw payloads are never copied to the host."""
import copy
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import threading
import time
import unicodedata

ID = re.compile(r"^[0-9a-f]{32}$")
REASONS = {
    "initial_out_of_distribution": "初始状态不适合模型",
    "repetitive_actions": "相似动作循环", "no_progress": "推进停滞",
    "collision": "碰撞", "preventive_stop": "预防性终止",
    "grasp_failed": "抓取失败", "object_slipped": "物体滑落",
    "wrong_pose": "目标位置或姿态错误", "wrong_stage": "动作阶段错误",
    "unreachable": "限位或不可达", "unstable_actions": "抖动或发散",
    "runtime_error": "推理、通信或执行异常", "other": "其他",
}

def encode(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      sort_keys=True, separators=(",", ":")).encode("utf-8")

def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encode(value)); stream.flush(); os.fsync(stream.fileno())
        os.replace(tmp, path)
        if hasattr(os, "O_DIRECTORY"):
            descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try: os.fsync(descriptor)
            finally: os.close(descriptor)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def component(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 160:
        raise ValueError("分类和任务名称须为1到160字")
    value = unicodedata.normalize("NFKC", value.strip())
    if value in {".", ".."} or any(c in value for c in "/\\\0"):
        raise ValueError("分类和任务名称不能包含路径分隔符")
    safe = re.sub(r'[<>:"|?*\x00-\x1f]', "_", value).strip(" .")[:64] or "task"
    return safe + "_" + hashlib.sha256(value.encode()).hexdigest()[:8]

class SessionStore:
    def __init__(self, root, collector, view_builder=None):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.index = self.root / "_index"; self.index.mkdir(exist_ok=True)
        self.lock = threading.RLock()
        self.collector = collector
        self.view_builder = view_builder

    def directory(self, sid):
        if not isinstance(sid, str) or not ID.fullmatch(sid):
            raise ValueError("无效任务ID")
        pointer = self.index / (sid + ".json")
        if pointer.is_symlink(): raise ValueError("无效索引")
        relative = json.loads(pointer.read_text(encoding="utf-8"))["relative_dir"]
        path = self.root / relative
        if self.root not in path.resolve().parents or path.resolve() != path.absolute():
            raise ValueError("任务目录越界或包含符号链接")
        return path

    def manifest(self, sid):
        return json.loads((self.directory(sid) / "session.json").read_text(encoding="utf-8"))

    def _save(self, directory, value):
        value["revision"] += 1
        value["updated_wall_ns"] = str(time.time_ns())
        atomic(directory / "session.json", value)
        return copy.deepcopy(value)

    def create(self, sid, task_id, category, policy=None):
        with self.lock:
            if not isinstance(sid, str) or not ID.fullmatch(sid): raise ValueError("无效任务ID")
            if (self.index / (sid + ".json")).exists():
                old = self.manifest(sid)
                if (old["task_id"], old["category"]) != (task_id, category):
                    raise ValueError("同一任务ID对应了不同任务")
                return old
            for index in self.index.glob("*.json"):
                prior = self.manifest(index.stem)
                if prior["collection_state"] in {"preparing","recording","stop_failed"}:
                    raise ValueError("上一条录制尚未结束，请先在采集数据页结束它")
            day = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d")
            path = self.root / component(category) / component(task_id) / day / sid
            if self.root not in path.resolve().parents: raise ValueError("目录越界")
            path.mkdir(parents=True, exist_ok=False)
            for name in ["raw", "model", "annotations"]: (path / name).mkdir()
            value = {"schema": "takeover-session/2", "episode_id": sid,
                     "task_id": task_id, "category": category, "date": day,
                     "policy": policy or {}, "revision": 0, "collection_state": "preparing",
                     "archive_state": "draft", "created_wall_ns": str(time.time_ns()),
                     "raw_directory": str(path / "raw"), "round_annotations": {},
                     "result": None, "collection_error": None}
            atomic(path / "session.json", value)
            atomic(self.index / (sid + ".json"), {"relative_dir": str(path.relative_to(self.root))})
            return value

    def start(self, sid):
        with self.lock:
            path = self.directory(sid); value = self.manifest(sid)
            if value["collection_state"] == "recording": return value
            if value["collection_state"] != "preparing": raise ValueError("该任务不能重新录制")
            if hasattr(self.collector, "properties"):
                value["collector"] = {"uuid":sid, "properties":self.collector.properties(sid,path / "raw")}
                self._save(path,value)
            try:
                value["collector"] = self.collector.start(sid, path / "raw")
                value["collection_state"] = "recording"
                value["recording_start_wall_ns"] = str(time.time_ns())
            except Exception as exc:
                value["collection_state"] = "stop_failed"
                value["collection_error"] = str(exc)
                self._save(path, value)
                raise
            return self._save(path, value)

    def stop(self, sid, expected_sequence=None):
        with self.lock:
            path = self.directory(sid); value = self.manifest(sid)
            if expected_sequence is not None:
                if type(expected_sequence) is not int or expected_sequence<1: raise ValueError("记录屏障无效")
                value["expected_last_sequence"] = expected_sequence
            if value["collection_state"] == "stopped":
                return self._save(path,value) if expected_sequence is not None else value
            if value["collection_state"] not in {"recording", "stop_failed", "preparing"}:
                raise ValueError("当前没有可结束的录制")
            try:
                value["stop_receipt"] = self.collector.stop(sid, value.get("collector", {}))
                value["collection_state"] = "stopped"
                value["recording_end_wall_ns"] = str(time.time_ns())
                value["collection_error"] = None
            except Exception as exc:
                value["collection_state"] = "stop_failed"
                value["collection_error"] = str(exc)
                self._save(path, value)
                raise
            return self._save(path, value)

    def put_record(self, record):
        sid, rid = record.get("episode_id"), record.get("record_id")
        if not isinstance(rid, str) or not ID.fullmatch(rid): raise ValueError("无效记录ID")
        if record.get("kind") not in {"metadata", "chunk", "dispatch", "events", "result", "session_end", "initialization", "error"}:
            raise ValueError("无效记录类型")
        raw = encode(record)
        if len(raw) > 1_000_000: raise ValueError("记录过大")
        digest = hashlib.sha256(raw).hexdigest()
        with self.lock:
            path = self.directory(sid)
            target = path / "model" / (rid + ".json")
            if target.exists():
                if target.is_symlink() or hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                    raise ValueError("记录ID冲突")
                return {"duplicate": True, "record_sha256": digest}
            value = self.manifest(sid)
            if value["archive_state"] == "complete": raise ValueError("已归档任务不能增加记录")
            atomic(target, record)
            if record["kind"] == "result":
                result = record.get("data", {}).get("result")
                if isinstance(result, dict):
                    value["result"] = result
                    self._save(path, value)
            return {"duplicate": False, "record_sha256": digest}

    def records(self, sid):
        result = []
        for p in self.directory(sid).joinpath("model").glob("*.json"):
            if p.is_symlink() or p.stat().st_size > 1_000_000: raise ValueError("记录文件无效")
            result.append(json.loads(p.read_text(encoding="utf-8")))
        if len(result) > 20000: raise ValueError("单任务记录数超过浏览上限")
        return result

    def detail(self, sid):
        with self.lock:
            value = self.manifest(sid); records = self.records(sid)
            if self.view_builder:
                _summary, detail = self.view_builder(sid, records, receipts=len(records))
            else:
                starts = [e for r in records if r["kind"] == "events"
                          for e in r["data"].get("events", []) if e["kind"] == "human_round_started"]
                detail = {"episode_id": sid, "rounds": [{"round": e["round"]} for e in starts]}
            annotations = value["round_annotations"]
            for item in detail["rounds"]:
                item["annotation"] = annotations.get(str(item["round"]))
            missing = [r["round"] for r in detail["rounds"]
                       if not self.annotation_valid(annotations.get(str(r["round"])))]
            detail.update(category=value["category"], session_directory=str(self.directory(sid)),
                          directory_226=str(self.directory(sid)), raw_directory=value["raw_directory"],
                          session_revision=value["revision"], collection_state=value["collection_state"],
                          collection_error=value["collection_error"], archive_state=value["archive_state"],
                          missing_annotations=missing, pending_count=0,
                          reason_options=REASONS, result=value["result"] or detail.get("result"))
            sequences = sorted(r.get("sequence",0) for r in records)
            expected = value.get("expected_last_sequence", max(sequences,default=0))
            records_complete = sequences == list(range(1,expected+1))
            if not records_complete:
                detail.setdefault("warnings",[]).append("模型记录序号存在缺口或尚未完成传输")
            detail["can_finalize"] = (value["collection_state"] == "stopped" and
                                      not missing and isinstance(value["result"], dict) and
                                      records_complete and detail.get("complete",True))
            return detail

    @staticmethod
    def annotation_valid(value):
        return bool(value and (value.get("reasons") or value.get("note", "").strip()) and
                    ("other" not in value.get("reasons", []) or value.get("note", "").strip()))

    def annotate(self, sid, round_id, reasons, note, expected_revision):
        if type(round_id) is not int or round_id < 1: raise ValueError("无效接管轮次")
        if not isinstance(reasons, list) or any(r not in REASONS for r in reasons):
            raise ValueError("接管原因无效")
        if not isinstance(note, str) or len(note) > 2000: raise ValueError("说明最长2000字")
        with self.lock:
            value = self.manifest(sid)
            if expected_revision != value["revision"]: raise ValueError("标注版本已变化，请刷新")
            if value["archive_state"] == "complete": raise ValueError("任务已经归档")
            if round_id not in {r["round"] for r in self.detail(sid)["rounds"]}:
                raise ValueError("该接管区间不存在")
            item = {"round": round_id, "reasons": list(dict.fromkeys(reasons)),
                    "note": note.strip(), "updated_wall_ns": str(time.time_ns())}
            value["round_annotations"][str(round_id)] = item
            atomic(self.directory(sid) / "annotations" / ("round_" + str(round_id) + ".json"), item)
            self._save(self.directory(sid), value)
            return self.detail(sid)

    def finalize(self, sid, expected_revision):
        with self.lock:
            value = self.manifest(sid)
            if value["archive_state"] == "complete": return self.detail(sid)
            if value["revision"] != expected_revision: raise ValueError("任务状态已变化，请刷新")
            detail = self.detail(sid)
            if not detail["can_finalize"]:
                raise ValueError("需结束录制、填写整条结果并补齐每次接管原因")
            if detail.get("warnings") and any("缺口" in w or "损坏" in w for w in detail["warnings"]):
                raise ValueError("记录存在缺口，不能标为完整")
            value["archive_state"] = "complete"
            value["finalized_wall_ns"] = str(time.time_ns())
            self._save(self.directory(sid), value)
            return self.detail(sid)

    def list_sessions(self, filters):
        rows, errors = [], []
        for p in self.index.glob("*.json"):
            try:
                d = self.detail(p.stem)
                if filters.get("task") and filters["task"].casefold() not in d["task_id"].casefold(): continue
                if filters.get("category") and filters["category"] != d["category"]: continue
                if filters.get("date") and filters["date"] != self.manifest(p.stem)["date"]: continue
                if filters.get("outcome") and filters["outcome"] != d.get("outcome"): continue
                if filters.get("human") == "yes" and not d["rounds"]: continue
                if filters.get("human") == "no" and d["rounds"]: continue
                rows.append({k:v for k,v in d.items() if k not in {"events","chunks","metadata","hands","segments","review_windows_merged"}})
            except (ValueError, OSError, KeyError) as exc:
                errors.append({"episode_id": p.stem, "message": str(exc)})
        rows.sort(key=lambda x: int(x.get("started_wall_ns") or 0), reverse=True)
        offset = max(0, int(filters.get("offset", 0))); limit = min(100, max(1, int(filters.get("limit", 50))))
        return {"success": True, "records": rows[offset:offset+limit], "total": len(rows), "errors": errors}
