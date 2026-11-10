#!/usr/bin/env python3
"""Integrated robot test web server (A1: UI and preferences)."""

import json
import os
import signal
import subprocess
import sys
import threading
import web_config
import vr_proxy
from takeover_runtime import Runtime, WasherAdapter, make_recorder
from episode_control import Conflict
from washer_tasks import HOMECARE_ROBOT_HOST, MOTION_UNKNOWN_EXIT, TASK_IDS, single_task_config

takeover = Runtime(vr_proxy.fetch_status, WasherAdapter, recorder_factory=make_recorder)
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from remote_records import RemoteRecordBrowser


BASE_DIR = Path(__file__).resolve().parent
records = RemoteRecordBrowser(BASE_DIR / "action_spool")
HOST = "0.0.0.0"
PORT = 8084

TASKS = {
    '左臂从洗衣机拿衣服': {'name': '左臂从洗衣机拿衣服', 'type': 'washer'},
    "双臂关洗衣机门": {"name": "双臂关洗衣机门", "type": "washer"},
    "左臂放浅蓝色衣服进洗衣机": {"name": "左臂放浅蓝色衣服进洗衣机", "type": "washer"},
    "右臂开洗衣机门": {"name": "右臂开洗衣机门", "type": "washer"},
    "left_arm_drawer_pick_omeprazole_bottle": {"name": "左臂抽屉拿奥美拉唑瓶", "type": "pick"},
    "left_arm_drawer_pick_amoxicillin_box": {"name": "左臂抽屉拿阿莫西林盒", "type": "pick"},
    "left_arm_drawer_pick_omeprazole_box": {"name": "左臂抽屉拿奥美拉唑盒", "type": "pick"},
    "left_arm_drawer_pick_kangfuxin_bottle": {"name": "左臂抽屉拿康复新液瓶", "type": "pick"},
    "left_arm_drawer_pick_kangfuxin_box": {"name": "左臂抽屉拿康复新液盒", "type": "pick"},
    "left_arm_pick_vitamin_c": {"name": "左臂拿维生素C", "type": "pick"},
    "left_arm_pick_omeprazole_box": {"name": "左臂拿奥美拉唑盒", "type": "pick"},
    "left_arm_pick_omeprazole_bottle": {"name": "左臂拿奥美拉唑瓶", "type": "pick"},
    "left_arm_pick_kangfuxin_bottle": {"name": "左臂拿康复新液瓶", "type": "pick"},
    "left_arm_pick_amoxicillin_box": {"name": "左臂拿阿莫西林盒", "type": "pick"},
    "left_arm_pick_kangfuxin_box": {"name": "左臂拿康复新液盒", "type": "pick"},
    "right_arm_pick_wahaha": {"name": "右臂拿娃哈哈", "type": "pick"},
    "right_arm_pick_yibao": {"name": "右臂拿怡宝", "type": "pick"},
    "right_arm_pick_tissue": {"name": "右臂拿纸巾", "type": "pick"},
    "right_arm_pick_glasses": {"name": "右臂拿眼镜", "type": "pick"},
    "left_arm_to_get_hot_water": {"name": "左臂接热水", "type": "water"},
    "left_arm_to_get_cold_water": {"name": "左臂接冷水", "type": "water"},
    "左臂饮水机接热水": {"name": "左臂饮水机接热水", "type": "water"},
    "左臂饮水机接凉水": {"name": "左臂饮水机接凉水", "type": "water"},
    "left_arm_place_omeprazole_box": {"name": "左臂放奥美拉唑盒", "type": "place"},
    "left_arm_place_kangfuxin_bottle": {"name": "左臂放康复新液瓶", "type": "place"},
    "right_arm_place_water": {"name": "右臂放水", "type": "place"},
    "right_arm_place_tissue": {"name": "右臂放纸巾", "type": "place"},
    "right_arm_place_glasses": {"name": "右臂放眼镜", "type": "place"},
    "right_arm_pull_drawer": {"name": "右臂拉抽屉", "type": "drawer"},
    "right_arm_push_drawer": {"name": "右臂关抽屉", "type": "drawer"},
}

run_lock = threading.Lock()
process_lock = threading.Lock()
current_process = None
current_task_id = None
current_operation = None
washer_motion_unknown = False


def tail_output(text: str, max_lines: int = 80) -> str:
    return "\n".join(text.splitlines()[-max_lines:])


def run_robot_task(task_id: str, *, single_task=False) -> tuple[int, dict]:
    global current_process, current_task_id, current_operation, washer_motion_unknown

    if not isinstance(task_id, str) or task_id not in TASKS:
        return HTTPStatus.BAD_REQUEST, {"success": False, "message": f"未知任务: {task_id}"}
    if not run_lock.acquire(blocking=False):
        return HTTPStatus.CONFLICT, {"success": False, "message": "已有测试正在运行，请等待完成"}

    try:
        config = web_config.load()
        if takeover.active():
            run_lock.release()
            return HTTPStatus.CONFLICT, {"success": False, "message": "人工接管任务尚未结束"}
        washer_config = single_task_config() if single_task else None
        if single_task and (task_id not in TASK_IDS or not washer_config
                            or config["human_takeover_enabled"] or washer_motion_unknown):
            run_lock.release()
            return HTTPStatus.CONFLICT, {
                "success": False, "taskId": task_id, "returnCode": 1,
                "message": "限时洗衣机模式不可用，请检查配置或确认上一次轨迹已停止",
            }
        if config["human_takeover_enabled"]:
            try:
                return HTTPStatus.OK, takeover.start(task_id, config)
            except (Conflict, ValueError, OSError) as exc:
                return HTTPStatus.CONFLICT, {"success": False, "message": str(exc)}
            finally:
                run_lock.release()
    except (OSError, ValueError) as exc:
        run_lock.release()
        return HTTPStatus.BAD_REQUEST, {"success": False, "message": f"无法读取配置: {exc}"}
    with process_lock:
        current_task_id = task_id
        current_operation = "washer-single-task" if single_task else "run"

    task = TASKS[task_id]
    # 默认使用异步版本（高频闭环，约 6~7 Hz）
    if task["type"] == "water":
        command = [
            sys.executable,
            str(BASE_DIR / "run_water.py"),
            "--task",
            task_id,
        ]
    elif task["type"] == "pick":
        command = [
            sys.executable,
            str(BASE_DIR / "run_pick.py"),
            "--task",
            task_id,
        ]
    elif task["type"] == "place":
        command = [
            sys.executable,
            str(BASE_DIR / "run_place.py"),
            "--task",
            task_id,
        ]
    elif task["type"] == "washer":
        command = [sys.executable, str(BASE_DIR / "run_washer.py"), "--task", task_id]
        if single_task:
            command += ["--run-seconds", str(washer_config["runSeconds"][task_id])]
    else:  # drawer type
        command = [
            sys.executable,
            str(BASE_DIR / "run_drawer.py"),
            "--task",
            task_id,
        ]

    # ========== 同步版本切换说明 ==========
    # 如需使用同步版本（完整执行动作块后再推理，约 1 Hz）：
    # 1. 注释掉上面的整个 if/elif/else 代码块（第60-80行）
    # 2. 取消下面这段注释：
    # if task["type"] == "pick":
    #     command = [
    #         sys.executable,
    #         str(BASE_DIR / "run_pick_synchronized.py"),
    #         "--task",
    #         task_id,
    #         "--record",  # 可选：如需保存图片和诊断数据，保留此行；否则删除
    #     ]
    # elif task["type"] == "place":
    #     command = [
    #         sys.executable,
    #         str(BASE_DIR / "run_place.py"),
    #         "--task",
    #         task_id,
    #     ]
    # else:  # drawer type
    #     command = [
    #         sys.executable,
    #         str(BASE_DIR / "run_drawer.py"),
    #         "--task",
    #         task_id,
    #     ]
    # ========================================

    try:
        print(f"[INFO] 开始测试: {task['name']} ({task_id})", flush=True)
        process = subprocess.Popen(
            command,
            cwd=BASE_DIR,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=(os.name != "nt"),
        )
        with process_lock:
            current_process = process
            current_task_id = task_id

        if single_task:
            try:
                stdout, stderr = process.communicate(timeout=washer_config["maxDurationS"])
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                return_code = MOTION_UNKNOWN_EXIT
            else:
                return_code = process.returncode
            if return_code == MOTION_UNKNOWN_EXIT or return_code < 0:
                with process_lock:
                    washer_motion_unknown = True
                return HTTPStatus.GATEWAY_TIMEOUT, {
                    "success": False, "taskId": task_id, "returnCode": return_code,
                    "motionState": "unknown", "message": "未确认轨迹结束，后续洗衣机限时任务已锁定",
                    "output": tail_output((stdout or "") + "\n" + (stderr or "")),
                }
        else:
            stdout, stderr = process.communicate()
            return_code = process.returncode

        output = tail_output((stdout or "") + "\n" + (stderr or ""))
        ok = return_code == 0
        return (
            HTTPStatus.OK if ok else HTTPStatus.INTERNAL_SERVER_ERROR,
            {
                "success": ok,
                "taskId": task_id,
                "taskName": task["name"],
                "returnCode": return_code,
                **({"motionState": "finished"} if single_task and ok else {}),
                "message": ("轨迹执行已结束，等待后态验证" if single_task and ok
                            else f"{task['name']} 测试{'成功' if ok else '失败'}"),
                "output": output,
            },
        )
    except Exception as exc:
        if single_task and current_process is not None:
            with process_lock:
                washer_motion_unknown = True
            if current_process.poll() is None:
                current_process.kill()
                current_process.communicate()
            return HTTPStatus.INTERNAL_SERVER_ERROR, {
                "success": False, "taskId": task_id, "returnCode": MOTION_UNKNOWN_EXIT,
                "motionState": "unknown", "message": f"未确认轨迹结束: {exc}",
            }
        return HTTPStatus.INTERNAL_SERVER_ERROR, {
            "success": False,
            "taskId": task_id,
            "taskName": task["name"],
            "message": f"{task['name']} 测试启动失败: {exc}",
            "output": "",
        }
    finally:
        with process_lock:
            current_process = None
            current_task_id = None
            current_operation = None
        run_lock.release()


def stop_current_task() -> tuple[int, dict]:
    if takeover.active():
        snapshot = takeover.status()['episode']
        try:
            return HTTPStatus.OK, takeover.command({'command': 'pause', 'episodeId': snapshot['id'],
                'revision': snapshot['revision'], 'requestId': 'interrupt-'+str(snapshot['revision'])})
        except (Conflict, ValueError) as exc:
            return HTTPStatus.CONFLICT, {"success": False, "message": str(exc)}
    with process_lock:
        process = current_process
        task_id = current_task_id
        single_task = current_operation == "washer-single-task"
    if process is None or process.poll() is not None:
        return HTTPStatus.CONFLICT, {"success": False, "message": "当前没有正在运行的测试"}

    task_name = TASKS.get(task_id, {}).get("name", task_id or "未知任务")
    try:
        if os.name == "nt":
            process.send_signal(signal.SIGINT)
        else:
            os.killpg(process.pid, signal.SIGINT)
        if single_task:
            return HTTPStatus.ACCEPTED, {
                "success": True, "message": f"已请求 {task_name} 停止继续下发，轨迹状态待确认",
            }
        return HTTPStatus.OK, {"success": True, "message": f"已向 {task_name} 发送 Ctrl+C"}
    except Exception as exc:
        return HTTPStatus.INTERNAL_SERVER_ERROR, {"success": False, "message": f"发送 Ctrl+C 失败: {exc}"}


def restore_robot_task(task_id: str) -> tuple[int, dict]:
    """执行机器人复位操作"""
    global current_task_id, current_operation
    if not isinstance(task_id, str) or task_id not in TASKS:
        return HTTPStatus.BAD_REQUEST, {"success": False, "message": f"未知任务: {task_id}"}
    if not run_lock.acquire(blocking=False):
        return HTTPStatus.CONFLICT, {"success": False, "message": "已有测试或复位正在运行，请等待完成"}

    if takeover.active():
        run_lock.release()
        return HTTPStatus.CONFLICT, {"success": False, "message": "请从人工接管面板复位当前任务"}
    task = TASKS[task_id]
    command = [sys.executable, str(BASE_DIR / "restore.py"), task_id]
    with process_lock:
        current_task_id = task_id
        current_operation = "restore"

    try:
        print(f"[INFO] 开始复位: {task['name']} ({task_id})", flush=True)
        completed = subprocess.run(
            command,
            cwd=BASE_DIR,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        output = tail_output((completed.stdout or "") + "\n" + (completed.stderr or ""))
        ok = completed.returncode == 0
        return (
            HTTPStatus.OK if ok else HTTPStatus.INTERNAL_SERVER_ERROR,
            {
                "success": ok,
                "taskId": task_id,
                "taskName": task["name"],
                "returnCode": completed.returncode,
                "message": f"{task['name']} 复位{'成功' if ok else '失败'}",
                "output": output,
            },
        )
    except Exception as exc:
        return HTTPStatus.INTERNAL_SERVER_ERROR, {
            "success": False,
            "taskId": task_id,
            "taskName": task["name"],
            "message": f"{task['name']} 复位启动失败: {exc}",
            "output": "",
        }
    finally:
        with process_lock:
            current_task_id = None
            current_operation = None
        run_lock.release()


def configuration_status():
    with process_lock:
        task_id, operation = current_task_id, current_operation
        motion_unknown = washer_motion_unknown
    config = web_config.load()
    capabilities = dict(web_config.CAPABILITIES)
    washer_config = single_task_config()
    if washer_config and not motion_unknown and not config["human_takeover_enabled"]:
        capabilities["washer_single_task_v1"] = {
            "robotHost": HOMECARE_ROBOT_HOST,
            "maxDurationS": washer_config["maxDurationS"],
        }
    runtime = takeover.status()
    episode = runtime['episode']
    active = takeover.active()
    return {"success": True, "config": config, "washerMotionUnknown": motion_unknown, "capabilities": {**capabilities,"human_takeover":runtime["readiness"]["ready"],"action_chunk_recording":runtime["readiness"]["persistent_recording_ready"],"timeline":True},
            "locked": run_lock.locked() or active,
            "taskId": episode['taskId'] if active else task_id,
            "operation": 'takeover' if active else operation,
            "takeover_readiness": runtime['readiness'], "episode": episode,"recording_error":runtime.get("recording_error")}


class IntegratedHandler(BaseHTTPRequestHandler):
    server_version = "AgibotIntegratedTest/1.0"

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        origin = self.headers.get("Origin")
        if origin and urlparse(origin).netloc == self.headers.get("Host"):
            self.send_header("Access-Control-Allow-Origin", origin)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        if path in {"/", "/intergrated_test.html"}:
            self.send_file(BASE_DIR / "intergrated_test.html", "text/html; charset=utf-8")
            return
        if path in {"/data_browser.js", "/data_browser.css"}:
            self.send_file(BASE_DIR / path[1:], "text/javascript; charset=utf-8" if path.endswith('.js') else "text/css; charset=utf-8")
            return
        if path == '/api/records' or path.startswith('/api/records/'):
            try:
                parts=path.strip('/').split('/')
                if path == '/api/records':
                    payload=records.listing(parse_qs(urlparse(self.path).query))
                elif len(parts)==3:
                    payload=records.detail(parts[2])
                elif len(parts)==5 and parts[3]=='chunks':
                    payload=records.chunk(parts[2],parts[4])
                else:
                    self.send_json(HTTPStatus.NOT_FOUND, {'success':False,'message':'Not found'});return
                self.send_json(HTTPStatus.OK,payload)
            except FileNotFoundError as exc:
                self.send_json(HTTPStatus.NOT_FOUND,{'success':False,'message':str(exc)})
            except (ValueError,TypeError,OSError) as exc:
                self.send_json(HTTPStatus.BAD_REQUEST,{'success':False,'message':str(exc)})
            return
        if path == "/api/vr/status":
            status, payload = vr_proxy.fetch_status()
            self.send_json(status, payload)
            return
        if path == "/api/episode":
            self.send_json(HTTPStatus.OK, takeover.status())
            return
        if path == "/api/config":
            try:
                self.send_json(HTTPStatus.OK, configuration_status())
            except (OSError, ValueError) as exc:
                self.send_json(HTTPStatus.BAD_REQUEST, {"success": False, "message": f"无法读取配置: {exc}"})
            return
        if path == "/api/tasks":
            self.send_json(HTTPStatus.OK, {"success": True, "tasks": TASKS})
            return
        self.send_json(HTTPStatus.NOT_FOUND, {"success": False, "message": "Not found"})

    def do_POST(self):
        origin = self.headers.get("Origin")
        if origin and urlparse(origin).netloc != self.headers.get("Host"):
            self.send_json(HTTPStatus.FORBIDDEN, {"success":False,"message":"来源不匹配"})
            return
        review_path = urlparse(self.path).path.strip('/').split('/')
        if len(review_path)==4 and review_path[:2]==['api','records']:
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=16384: raise ValueError('标注请求大小无效')
                body=json.loads(self.rfile.read(size))
                if not isinstance(body,dict): raise ValueError('无效标注请求')
                if review_path[3]=='stop' and takeover.active() and takeover.episode.id==review_path[2]:
                    raise ValueError('当前任务尚未结束，请先在运行页结束任务')
                self.send_json(HTTPStatus.OK,records.write(review_path[2],review_path[3],body))
            except (ValueError,OSError,KeyError) as exc:
                self.send_json(HTTPStatus.CONFLICT,{'success':False,'message':str(exc)})
            return
        path = urlparse(self.path).path
        if path == "/api/stop-task":
            status, payload = stop_current_task()
            self.send_json(status, payload)
            return
        if path not in {"/api/run-task", "/api/run-washer-task", "/api/restore-task", "/api/config", "/api/episode/command"}:
            self.send_json(HTTPStatus.NOT_FOUND, {"success": False, "message": "Not found"})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if not 0 <= content_length <= 65536:
                raise ValueError("请求过大")
            raw_body = self.rfile.read(content_length).decode("utf-8") if content_length else "{}"
            body = json.loads(raw_body or "{}")
            if not isinstance(body, dict):
                raise ValueError("JSON对象必需")
        except (ValueError, UnicodeDecodeError):
            self.send_json(HTTPStatus.BAD_REQUEST, {"success": False, "message": "JSON格式不正确"})
            return

        if path == "/api/episode/command":
            try:
                self.send_json(HTTPStatus.OK, takeover.command(body))
            except Conflict as exc:
                self.send_json(HTTPStatus.CONFLICT, {"success": False, "message": str(exc)})
            except (ValueError, TypeError, OSError) as exc:
                self.send_json(HTTPStatus.BAD_REQUEST, {"success": False, "message": str(exc)})
            return
        if path == "/api/config":
            if takeover.active():
                self.send_json(HTTPStatus.CONFLICT, {"success": False, "message": "任务未结束，配置已锁定"})
                return
            if not run_lock.acquire(blocking=False):
                self.send_json(HTTPStatus.CONFLICT, {"success": False, "message": "任务运行或复位中，配置已锁定"})
                return
            try:
                config = web_config.save(body)
                self.send_json(HTTPStatus.OK, {"success": True, "config": config, "message": "配置已保存；运行前会检查226记录目录可写性"})
            except (OSError, ValueError) as exc:
                self.send_json(HTTPStatus.BAD_REQUEST, {"success": False, "message": f"配置未保存: {exc}"})
            finally:
                run_lock.release()
            return
        if path == "/api/restore-task":
            status, payload = restore_robot_task(body.get("taskId"))
        else:
            status, payload = run_robot_task(
                body.get("taskId"), single_task=path == "/api/run-washer-task",
            )
        self.send_json(status, payload)

    def send_file(self, path: Path, content_type: str):
        if not path.exists():
            self.send_json(HTTPStatus.NOT_FOUND, {"success": False, "message": f"文件不存在: {path.name}"})
            return
        content = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def send_json(self, status: int, payload: dict):
        content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, fmt, *args):
        print(f"[HTTP] {self.address_string()} - {fmt % args}", flush=True)


def main():
    from chunk_log import shared_uploader
    shared_uploader()  # recover only file transfers; never resume a robot task
    print("=" * 60)
    print("Agibot 综合任务测试服务")
    print("=" * 60)
    print(f"测试页面: http://localhost:{PORT}/intergrated_test.html")
    print("按 Ctrl+C 停止服务")
    print("=" * 60)
    ThreadingHTTPServer((HOST, PORT), IntegratedHandler).serve_forever()


if __name__ == "__main__":
    main()
