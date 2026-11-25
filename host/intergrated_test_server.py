#!/usr/bin/env python3
"""Integrated robot test web server (A1: UI and preferences)."""

import json
import os
import re
import select
import signal
import socket
import subprocess
import sys
import threading
from contextlib import closing
from time import monotonic, sleep
import web_config
import chunk_log
from network_config import get_robot_host, configure_proxy_bypass
import vr_proxy
from takeover_runtime import Runtime, WasherAdapter, make_recorder
from episode_control import Conflict
from washer_tasks import (
    HOMECARE_POLICY_HOST, HOMECARE_ROBOT_HOST, MOTION_UNKNOWN_EXIT, OPEN_WASHER_TASK, OPERATOR_FINISHED_EXIT,
    OPERATOR_FINISH_BYTE, TASK_IDS, managed_max_duration, single_task_config,
)

takeover = Runtime(vr_proxy.fetch_status, WasherAdapter, recorder_factory=make_recorder)
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from remote_records import RemoteRecordBrowser
from clothes_task import (CLOTHES_TASKS, CLOSE_WASHER_TASK_ID, MEM_CLOSE_WASHER_TASK_ID, HANG_TASK_ID, PUT_WASHER_TASK_ID, SOFA_GRASP_HANDS, SOFA_TASK_ID, TASK_ID as CLOTHES_TASK_ID,
                          load_task_spec,
                          ClothesPolicy, clothes_policy_slot, validation_report_path)
from clothes_robot import (clothes_motion_allowed, ensure_remote_env_running,
                           execution_readiness as clothes_readiness, robot_report_path,
                           RobotConnection, load_robot_setup)


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
    **{task_id: {"name": name, "type": "washer" if task_id in {PUT_WASHER_TASK_ID, CLOSE_WASHER_TASK_ID, MEM_CLOSE_WASHER_TASK_ID} else "clothes",
                 "inferenceOnly": True, "robotFlowConnected": True, "managedClothes": True,
                 "managedSofa": task_id == SOFA_TASK_ID}
       for task_id, name in CLOTHES_TASKS.items()},
}

run_lock = threading.Lock()
process_lock = threading.Lock()
clothes_policies = {task_id: ClothesPolicy(task_id) for task_id in CLOTHES_TASKS}
current_process = None
current_task_id = None
current_operation = None
current_sofa_grasp_hand = None
current_policy_cancel = None
current_washer_request = None
current_washer_cancel_fd = None
current_washer_finish = None
washer_motion_unknown = False
washer_request_history = {}
WASHER_CANCEL_PROTOCOL = "request-id-v1"
WASHER_REQUEST_TTL_S = 300
WASHER_REQUEST_HISTORY_LIMIT = 256
WASHER_OWNER_POLL_S = 0.1


def tail_output(text: str, max_lines: int = 80) -> str:
    return "\n".join(text.splitlines()[-max_lines:])


def managed_washer_readiness(washer_config):
    try:
        import homecare_washer_execution as managed
    except ImportError:
        return None, "WASHER_MODEL_MANAGER_UNAVAILABLE"
    return managed, managed.managed_readiness(washer_config)


def washer_refusal(task_id, code, *, status=HTTPStatus.CONFLICT, motion_state="not_started"):
    return status, {
        "success": False, "taskId": task_id, "returnCode": 1,
        "motionState": motion_state, "message": code,
    }


def valid_washer_request_id(request_id):
    return isinstance(request_id, str) and re.fullmatch(r"[0-9a-f]{32}", request_id) is not None


def prune_washer_requests():
    # Only under process_lock. Never evict a live attempt or an unexpired cancellation.
    now = monotonic()
    for request_id, (_, _, expires) in tuple(washer_request_history.items()):
        if request_id != current_washer_request and expires <= now:
            del washer_request_history[request_id]


def remember_washer_request(request_id, task_id, state):
    washer_request_history[request_id] = (task_id, state, monotonic() + WASHER_REQUEST_TTL_S)


def prior_washer_request(task_id, request_id):
    prior = washer_request_history.get(request_id)
    if prior is None:
        return None
    if prior[:2] == (task_id, "cancelled"):
        return washer_refusal(task_id, "WASHER_CANCELLED_BEFORE_SUBMIT")
    return washer_refusal(task_id, "WASHER_REQUEST_ID_REUSED", motion_state="unknown")


def close_washer_cancel_fd():
    global current_washer_cancel_fd
    if current_washer_cancel_fd is not None:
        os.close(current_washer_cancel_fd)
        current_washer_cancel_fd = None


def request_active_washer_cancel():
    # Called with process_lock held and the active request already matched.
    if not current_policy_cancel.is_set():
        current_policy_cancel.set()
        close_washer_cancel_fd()
        if current_process is not None:
            try:
                current_process.send_signal(signal.SIGINT)
            except OSError:
                # The closed pipe also prevents the next cooperative submission.
                pass


def enable_washer_owner_keepalive(connection):
    # Detect a vanished peer even if the connection did not close cleanly.
    connection.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    for option, value in ((socket.TCP_KEEPIDLE, 5), (socket.TCP_KEEPINTVL, 2), (socket.TCP_KEEPCNT, 3)):
        connection.setsockopt(socket.IPPROTO_TCP, option, value)


def washer_owner_lost(connection, wait=0):
    try:
        readable, _, failed = select.select([connection], [], [connection], wait)
    except (OSError, ValueError):
        return True
    # After the POST body, the client must only wait for its response.
    # EOF, an error or any unexpected inbound bytes ends this ownership.
    return bool(readable or failed)


def watch_washer_owner(connection, request_id, done):
    while not done.is_set():
        if washer_owner_lost(connection, WASHER_OWNER_POLL_S):
            with process_lock:
                entry = washer_request_history.get(request_id)
                if request_id == current_washer_request and entry and entry[1] == "running":
                    request_active_washer_cancel()
            return


def communicate_client_owned(process, cancelled):
    # Polling has no action deadline. Only loss of ownership or a scoped
    # cancellation starts the bounded local subprocess cleanup.
    while not cancelled.is_set():
        try:
            stdout, stderr = process.communicate(timeout=WASHER_OWNER_POLL_S)
        except subprocess.TimeoutExpired:
            continue
        return stdout, stderr, process.returncode
    try:
        stdout, stderr = process.communicate(timeout=1)
    except subprocess.TimeoutExpired:
        process.kill()
        try:
            stdout, stderr = process.communicate(timeout=1)
        except subprocess.TimeoutExpired:
            stdout, stderr = "", ""
    return stdout, stderr, MOTION_UNKNOWN_EXIT


def cancel_washer_task(task_id, request_id, client_ip):
    if client_ip != get_robot_host():
        return washer_refusal(task_id, "HOMECARE_WASHER_CLIENT_NOT_AUTHORIZED", status=HTTPStatus.FORBIDDEN)
    if task_id not in TASK_IDS or not valid_washer_request_id(request_id):
        return washer_refusal(task_id, "WASHER_CANCEL_REQUEST_INVALID", status=HTTPStatus.BAD_REQUEST)
    try:
        config = single_task_config()
    except (OSError, ValueError):
        return washer_refusal(task_id, "WASHER_CONFIG_INVALID")
    if not config or not config.get("managedPolicy", {}).get("enabled"):
        return washer_refusal(task_id, "MANAGED_WASHER_DISABLED")
    with process_lock:
        prune_washer_requests()
        entry = washer_request_history.get(request_id)
        if entry and (entry[0] != task_id or entry[1] == "finished"):
            return washer_refusal(task_id, "WASHER_CANCEL_REQUEST_MISMATCH_OR_FINISHED")
        if request_id == current_washer_request:
            motion_state = "unknown" if current_process is not None else "not_started"
            request_active_washer_cancel()
        else:
            if entry is None and len(washer_request_history) >= WASHER_REQUEST_HISTORY_LIMIT:
                return washer_refusal(task_id, "WASHER_REQUEST_HISTORY_FULL")
            remember_washer_request(request_id, task_id, "cancelled")
            motion_state = "not_started"
    return HTTPStatus.ACCEPTED, {
        "success": True, "taskId": task_id, "requestId": request_id,
        "motionState": motion_state, "message": "WASHER_CANCEL_REQUESTED",
    }


def finish_washer_task(task_id, request_id, client_ip, completion_source=None):
    if client_ip != get_robot_host():
        return washer_refusal(task_id, "HOMECARE_WASHER_CLIENT_NOT_AUTHORIZED", status=HTTPStatus.FORBIDDEN)
    if task_id not in TASK_IDS or not valid_washer_request_id(request_id):
        return washer_refusal(task_id, "WASHER_FINISH_REQUEST_INVALID", status=HTTPStatus.BAD_REQUEST)
    expected_source = "client_deadline" if task_id == OPEN_WASHER_TASK else None
    if completion_source != expected_source:
        return washer_refusal(task_id, "WASHER_FINISH_SOURCE_MISMATCH")
    with process_lock:
        entry = washer_request_history.get(request_id)
        if (request_id != current_washer_request or not entry or entry[:2] != (task_id, "running")
                or current_washer_finish is None or current_process is None
                or current_policy_cancel.is_set()):
            return washer_refusal(task_id, "WASHER_FINISH_REQUEST_MISMATCH_OR_FINISHED")
        if not current_washer_finish.is_set():
            try:
                os.write(current_washer_cancel_fd, OPERATOR_FINISH_BYTE)
            except OSError:
                return washer_refusal(task_id, "WASHER_FINISH_RUNNER_UNAVAILABLE", motion_state="unknown")
            current_washer_finish.set()
            close_washer_cancel_fd()
    return HTTPStatus.ACCEPTED, {
        "success": True, "taskId": task_id, "requestId": request_id,
        "message": "WASHER_FINISH_REQUESTED",
    }


def model_validation_only():
    return os.environ.get("VLA_MODEL_VALIDATION_ONLY") == "1"


def robot_actions_disabled():
    return HTTPStatus.FORBIDDEN, {"success": False, "message": "当前为模型验证模式，机器人动作已禁用"}


def run_clothes_validation(task_id):
    return _run_clothes_process(task_id, real_robot=False)


def run_clothes_robot_task(task_id=CLOTHES_TASK_ID, *, grasp_hand=None):
    return _run_clothes_process(task_id, real_robot=True, grasp_hand=grasp_hand)


def set_clothes_model(task_id, start):
    global current_task_id, current_operation
    if not run_lock.acquire(blocking=False):
        return HTTPStatus.CONFLICT, {"success": False, "message": "请等当前任务结束"}
    try:
        if takeover.active():
            return HTTPStatus.CONFLICT, {"success": False, "message": "请等当前任务结束"}
        with process_lock:
            current_task_id = task_id
            current_operation = "clothes-model-start" if start else "clothes-model-stop"
        with clothes_policy_slot():
            if start:
                if clothes_motion_allowed(task_id):
                    readiness = clothes_readiness(task_id=task_id)
                    if not readiness["can_execute"]:
                        return HTTPStatus.CONFLICT, {"success": False, "message": "；".join(readiness["blockers"])}
                    ensure_remote_env_running(readiness["robot"])
                clothes_policies[task_id].start()
            else:
                clothes_policies[task_id].close()
        return HTTPStatus.OK, {"success": True, "message": "模型已就绪" if start else "模型已关闭"}
    except Exception as exc:
        return HTTPStatus.INTERNAL_SERVER_ERROR, {"success": False, "message": str(exc)}
    finally:
        with process_lock:
            current_task_id = current_operation = None
        run_lock.release()


def send_gripper_command(robot, side, action):
    command = {
        "base_link": "base_link",
        "trajectory_reference_time": 0.8,
        f"{side}_effector": [[-0.785 if action == "open" else 0.0]],
    }
    with closing(RobotConnection(robot)) as connection:
        connection.execute(command)
        sleep(command["trajectory_reference_time"])


def control_gripper(body):
    global current_operation
    side, action = body.get("side"), body.get("action")
    if set(body) != {"side", "action"} or side not in ("left", "right") or action not in ("open", "close"):
        return HTTPStatus.BAD_REQUEST, {"success": False, "message": "请选择左夹爪或右夹爪，以及张开或闭合"}
    if model_validation_only():
        return robot_actions_disabled()
    if not run_lock.acquire(blocking=False):
        return HTTPStatus.CONFLICT, {"success": False, "message": "任务正在运行，请结束任务后再控制夹爪"}
    try:
        if takeover.active():
            return HTTPStatus.CONFLICT, {"success": False, "message": "请先结束当前任务"}
        robot = {"host": get_robot_host(), "management_port": 8890, "control_port": 8891}
        with process_lock:
            current_operation = f"gripper-{side}-{action}"
        send_gripper_command(robot, side, action)
        label = "左夹爪" if side == "left" else "右夹爪"
        verb = "张开" if action == "open" else "闭合"
        return HTTPStatus.OK, {"success": True, "message": f"机器人已接收{label}{verb}指令",
                               "robotActionsSubmitted": 1}
    except Exception as exc:
        return HTTPStatus.BAD_GATEWAY, {"success": False, "message": f"夹爪指令未确认：{exc}"}
    finally:
        with process_lock:
            current_operation = None
        run_lock.release()


def control_clothes_gripper(task_id, side, action):
    global current_task_id, current_operation
    if task_id not in {CLOTHES_TASK_ID, HANG_TASK_ID, PUT_WASHER_TASK_ID, CLOSE_WASHER_TASK_ID, MEM_CLOSE_WASHER_TASK_ID} or side not in {"left", "right"} or action not in {"open", "close"}:
        return HTTPStatus.BAD_REQUEST, {"success": False, "message": "衣物任务夹爪操作无效"}
    if not clothes_motion_allowed(task_id):
        return robot_actions_disabled()
    if not run_lock.acquire(blocking=False):
        return HTTPStatus.CONFLICT, {"success": False, "message": "已有任务运行，请等待完成"}

    try:
        if takeover.active():
            return HTTPStatus.CONFLICT, {"success": False, "message": "请先结束当前任务"}
        readiness = clothes_readiness(task_id=task_id)
        if not readiness["can_execute"]:
            return HTTPStatus.CONFLICT, {"success": False, "message": "；".join(readiness["blockers"])}
        setup = load_robot_setup(readiness["setup_path"])
        with process_lock:
            current_task_id = task_id
            current_operation = f"clothes-gripper-{side}-{action}"
        ensure_remote_env_running(setup["robot"])
        send_gripper_command(setup["robot"], side, action)
        hand_label = "左" if side == "left" else "右"
        action_label = "张开" if action == "open" else "闭合"
        return HTTPStatus.OK, {
            "success": True, "taskId": task_id,
            "message": f"已发送{hand_label}夹爪{action_label}指令",
            "robotActionsSubmitted": 1,
        }
    except Exception as exc:
        return HTTPStatus.INTERNAL_SERVER_ERROR, {"success": False, "message": str(exc)}
    finally:
        with process_lock:
            current_task_id = current_operation = None
        run_lock.release()


def _kill_clothes_process(process):
    if os.name == "nt":
        process.kill()
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _run_clothes_process(task_id, *, real_robot, restore_pose=False, grasp_hand=None):
    global current_process, current_task_id, current_operation, current_sofa_grasp_hand
    name = CLOTHES_TASKS[task_id]
    if not run_lock.acquire(blocking=False):
        return HTTPStatus.CONFLICT, {"success": False, "message": "已有任务运行，请等待完成"}
    process = None
    try:
        if takeover.active():
            return HTTPStatus.CONFLICT, {"success": False, "message": "请先结束当前任务"}
        if real_robot:
            readiness = clothes_readiness(task_id=task_id)
            if not readiness["can_execute"]:
                return HTTPStatus.CONFLICT, {"success": False, "message": "；".join(readiness["blockers"]),
                                             "preflight": readiness, "robotActionsSubmitted": 0}
            if restore_pose and not readiness["can_reset_pose"]:
                return HTTPStatus.CONFLICT, {"success": False, "message": "尚未记录本任务的复位姿态",
                                             "robotActionsSubmitted": 0}
        use_loaded_model = real_robot and not restore_pose
        if use_loaded_model and not clothes_policies[task_id].ready:
            return HTTPStatus.CONFLICT, {"success": False, "message": f"请先启动{name}模型", "robotActionsSubmitted": 0}
        run_until_stopped = use_loaded_model and load_task_spec(task_id=task_id).get("run_until_stopped") is True
        script = "run_clothes_robot.py" if real_robot else "run_clothes.py"
        command = [sys.executable, "-u", str(BASE_DIR / script), "--task", task_id]
        if real_robot:
            command.append("--restore-pose" if restore_pose else "--execute")
        if use_loaded_model:
            command.append("--use-loaded-model")
        if use_loaded_model and task_id == SOFA_TASK_ID:
            grasp_hand = grasp_hand or "right"
            command.extend(("--grasp-hand", grasp_hand))
        operation = "clothes-restore" if restore_pose else ("clothes-robot" if real_robot else "model-validation")
        with process_lock:
            process = subprocess.Popen(
                command, cwd=BASE_DIR,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                start_new_session=(os.name != "nt"),
            )
            current_process, current_task_id, current_operation = process, task_id, operation
            current_sofa_grasp_hand = grasp_hand if use_loaded_model and task_id == SOFA_TASK_ID else None
        try:
            timeout = None if run_until_stopped else (360 if real_robot else 210)
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_clothes_process(process)
            stdout, stderr = process.communicate()
            stderr += "\n任务超时，任务进程已结束"
        ok = process.returncode == 0
        report_path = robot_report_path(task_id, restore_pose=restore_pose) if real_robot else validation_report_path(task_id)
        report = json.loads(report_path.read_text(encoding="utf-8")) if ok else None
        if report and report["task_id"] != task_id:
            raise ValueError("任务结果与所选衣物任务不匹配")
        if restore_pose:
            message = f"{name}姿态复位未完成，请核对轨迹状态"
            if ok:
                message = {"pose_already_set": "双臂和双夹爪已处于记录姿态",
                           "pose_restored": "双臂和双夹爪已确认复位"}.get(report["status"], "已中止姿态复位")
        elif real_robot:
            message = f"{name}实机流程未完成，请核对轨迹状态"
            if ok:
                message = "运行结束，等待现场确认抓取结果" if task_id == SOFA_TASK_ID else f"{name}执行流程已结束，等待人工核对实机结果"
                if report["status"] in {"cancelled_before_connect", "stopped_submitting"}:
                    message = "已中止" if task_id == SOFA_TASK_ID else f"{name}已中止后续动作下发"
        else:
            message = f"{name}模型推理验证通过（合成输入）" if ok else f"{name}模型推理验证未通过"
        return HTTPStatus.OK if ok else HTTPStatus.INTERNAL_SERVER_ERROR, {
            "success": ok, "taskId": task_id, "taskName": name,
            "inferenceOnly": not real_robot, "robotActionsSubmitted": (report or {}).get("robot_actions_submitted", 0) if not real_robot or ok else None,
            "physicalEffectVerified": False,
            "message": message,
            "output": tail_output((stdout or "") + "\n" + (stderr or "")),
            "validation": {k: report[k] for k in (
                "input_kind", "action_shape", "all_finite", "prompt", "policy_timing",
                "robot_connected", "robot_actions_submitted", "checkpoint", "asset_id", "model_metadata",
            )} if report and not real_robot else None,
            "execution": report if real_robot else None,
        }
    except Exception as exc:
        return HTTPStatus.INTERNAL_SERVER_ERROR, {
            "success": False, "taskId": task_id, "inferenceOnly": not real_robot,
            "robotActionsSubmitted": None if real_robot and process is not None else 0,
            "message": f"{name}流程未完成：{exc}",
        }
    finally:
        if process is not None and process.poll() is None:
            _kill_clothes_process(process)
            process.communicate()
        with process_lock:
            current_process = current_task_id = current_operation = None
            current_sofa_grasp_hand = None
        run_lock.release()


def run_robot_task(
    task_id: str, *, single_task=False, client_ip=None, request_id=None, owner_connection=None,
) -> tuple[int, dict]:
    global current_process, current_task_id, current_operation, current_policy_cancel
    global current_washer_request, current_washer_cancel_fd, current_washer_finish
    global washer_motion_unknown

    if not isinstance(task_id, str) or task_id not in TASKS:
        return HTTPStatus.BAD_REQUEST, {"success": False, "message": f"未知任务: {task_id}"}
    if task_id in CLOTHES_TASKS:
        if single_task or request_id is not None:
            return HTTPStatus.BAD_REQUEST, {"success": False, "message": "衣物任务使用独立的模型验证和实机入口"}
        return run_clothes_validation(task_id)
    if model_validation_only():
        return robot_actions_disabled()
    if request_id is not None and (not single_task or not valid_washer_request_id(request_id)):
        return washer_refusal(task_id, "WASHER_REQUEST_ID_INVALID", status=HTTPStatus.BAD_REQUEST)
    if request_id is not None and client_ip != get_robot_host():
        return washer_refusal(task_id, "HOMECARE_WASHER_CLIENT_NOT_AUTHORIZED", status=HTTPStatus.FORBIDDEN)
    if request_id is not None:
        with process_lock:
            prune_washer_requests()
            prior = prior_washer_request(task_id, request_id)
        if prior is not None:
            return prior
    if not run_lock.acquire(blocking=False):
        if request_id is not None:
            with process_lock:
                prior = prior_washer_request(task_id, request_id)
            if prior is not None:
                return prior
            return washer_refusal(task_id, "WASHER_SERVICE_BUSY")
        return HTTPStatus.CONFLICT, {"success": False, "message": "已有测试正在运行，请等待完成"}

    managed = None
    policy_cancel = None
    client_owned = False
    try:
        config = web_config.load()
        if takeover.active():
            run_lock.release()
            if request_id is not None:
                return washer_refusal(task_id, "WASHER_SERVICE_BUSY")
            return HTTPStatus.CONFLICT, {"success": False, "message": "人工接管任务尚未结束"}
        washer_config = single_task_config() if single_task else None
        managed_configured = washer_config is not None and "managedPolicy" in washer_config
        if request_id is not None and not managed_configured:
            run_lock.release()
            return washer_refusal(task_id, "WASHER_CANCEL_PROTOCOL_UNAVAILABLE")
        if single_task and (task_id not in TASK_IDS or not washer_config or washer_motion_unknown
                            or (config["human_takeover_enabled"] and not managed_configured)):
            run_lock.release()
            return washer_refusal(task_id, "限时洗衣机模式不可用，请检查配置或确认上一次轨迹已停止")
        if managed_configured:
            if client_ip != get_robot_host():
                run_lock.release()
                return washer_refusal(task_id, "HOMECARE_WASHER_CLIENT_NOT_AUTHORIZED",
                                      status=HTTPStatus.FORBIDDEN)
            if "runSeconds" not in washer_config and request_id is None:
                run_lock.release()
                return washer_refusal(task_id, "WASHER_REQUEST_ID_INVALID", status=HTTPStatus.BAD_REQUEST)
            client_owned = "maxDurationS" not in washer_config
            if client_owned:
                if owner_connection is None:
                    run_lock.release()
                    return washer_refusal(task_id, "WASHER_OWNER_CONNECTION_REQUIRED")
                enable_washer_owner_keepalive(owner_connection)
                if washer_owner_lost(owner_connection):
                    run_lock.release()
                    return washer_refusal(task_id, "WASHER_OWNER_CONNECTION_LOST")
            managed, status = managed_washer_readiness(washer_config)
            if status != "READY":
                run_lock.release()
                return washer_refusal(task_id, status)
            policy_cancel = threading.Event()
        elif config["human_takeover_enabled"]:
            try:
                return HTTPStatus.OK, takeover.start(task_id, config)
            except (Conflict, ValueError, OSError) as exc:
                return HTTPStatus.CONFLICT, {"success": False, "message": str(exc)}
            finally:
                run_lock.release()
    except (OSError, ValueError) as exc:
        run_lock.release()
        if request_id is not None:
            return washer_refusal(task_id, "WASHER_CONFIG_INVALID", status=HTTPStatus.BAD_REQUEST)
        return HTTPStatus.BAD_REQUEST, {"success": False, "message": f"无法读取配置: {exc}"}
    finish_requested = threading.Event() if client_owned else None
    with process_lock:
        if request_id is not None:
            prune_washer_requests()
            prior = prior_washer_request(task_id, request_id)
            if prior is not None:
                run_lock.release()
                return prior
            if len(washer_request_history) >= WASHER_REQUEST_HISTORY_LIMIT:
                run_lock.release()
                return washer_refusal(task_id, "WASHER_REQUEST_HISTORY_FULL")
            remember_washer_request(request_id, task_id, "running")
            current_washer_request = request_id
        current_task_id = task_id
        current_operation = "washer-single-task" if single_task else "run"
        current_policy_cancel = policy_cancel
        current_washer_finish = finish_requested

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
        if single_task and "runSeconds" in washer_config:
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

    model = None
    motion_claimed = False
    owner_done = None
    owner_watcher = None
    try:
        if client_owned:
            owner_done = threading.Event()
            owner_watcher = threading.Thread(
                target=watch_washer_owner, args=(owner_connection, request_id, owner_done),
                name="washer-owner-connection", daemon=True,
            )
            owner_watcher.start()
        if managed is not None:
            model = managed.model_runner(washer_config)
            try:
                model.start(
                    task_id, (task_id,), HOMECARE_POLICY_HOST,
                    monotonic() + washer_config["managedPolicy"]["startupTimeoutS"], policy_cancel,
                    check_gpu_memory=False,
                )
            except managed.WasherPolicyError as exc:
                return washer_refusal(task_id, str(exc))
            if not managed.wait_for_stopped(cancelled=policy_cancel.is_set):
                return washer_refusal(task_id, "WASHER_CANCELLED_BEFORE_SUBMIT" if policy_cancel.is_set()
                                      else "WASHER_PREPARATION_MOTION_NOT_SETTLED")
        print(f"[INFO] 开始测试: {task['name']} ({task_id})", flush=True)
        with process_lock:
            if managed is not None:
                if client_owned and washer_owner_lost(owner_connection):
                    request_active_washer_cancel()
                if policy_cancel.is_set():
                    return washer_refusal(task_id, "WASHER_CANCELLED_BEFORE_SUBMIT")
                managed.claim_motion(washer_config, task_id)
                motion_claimed = True
            try:
                inherited_fds = ()
                if request_id is not None:
                    read_fd, current_washer_cancel_fd = os.pipe()
                    inherited_fds = (read_fd,)
                    command += ["--cancel-fd", str(read_fd)]
                process = subprocess.Popen(
                    command, cwd=BASE_DIR, text=True,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    start_new_session=(os.name != "nt"),
                    pass_fds=inherited_fds,
                )
            except OSError:
                if motion_claimed:
                    managed.clear_motion(washer_config)
                    motion_claimed = False
                raise
            finally:
                for fd in inherited_fds:
                    os.close(fd)
            current_process = process
            current_task_id = task_id

        completion_source = None
        if single_task:
            if client_owned:
                stdout, stderr, return_code = communicate_client_owned(process, policy_cancel)
                if return_code == OPERATOR_FINISHED_EXIT:
                    with process_lock:
                        finished = finish_requested.is_set()
                    if finished:
                        completion_source = "client_deadline" if task_id == OPEN_WASHER_TASK else "operator"
                        return_code = 0
            else:
                try:
                    stdout, stderr = process.communicate(timeout=washer_config["maxDurationS"])
                except subprocess.TimeoutExpired:
                    process.kill()
                    try:
                        stdout, stderr = process.communicate(timeout=1)
                    except subprocess.TimeoutExpired:
                        stdout, stderr = "", ""
                    return_code = MOTION_UNKNOWN_EXIT
                else:
                    return_code = process.returncode
            if managed is not None:
                if return_code == 0 and managed.wait_for_stopped(cancelled=policy_cancel.is_set):
                    cleanup_ok = model.close()
                    model = None
                    with process_lock:
                        if cleanup_ok and not policy_cancel.is_set():
                            managed.clear_motion(washer_config)
                            motion_claimed = False
                            if request_id is not None:
                                remember_washer_request(request_id, task_id, "finished")
                        else:
                            return_code = MOTION_UNKNOWN_EXIT
                else:
                    return_code = MOTION_UNKNOWN_EXIT
            if return_code == MOTION_UNKNOWN_EXIT or return_code < 0:
                with process_lock:
                    washer_motion_unknown = True
                output = tail_output((stdout or "") + "\n" + (stderr or ""))
                diagnostic = tail_output(output, max_lines=12)[-2400:]
                print(f"[WASHER] {task_id} motion unknown (exit={return_code})\n{diagnostic}", flush=True)
                return HTTPStatus.GATEWAY_TIMEOUT, {
                    "success": False, "taskId": task_id, "returnCode": return_code,
                    "motionState": "unknown", "message": "未确认轨迹结束，后续洗衣机限时任务已锁定",
                    "output": output,
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
                **({"motionState": "finished", "physicalEffectVerified": False} if single_task and ok else {}),
                **({"completionSource": completion_source} if completion_source and ok else {}),
                "message": ("轨迹执行已结束，等待后态验证" if single_task and ok
                            else f"{task['name']} 测试{'成功' if ok else '失败'}"),
                "output": output,
            },
        )
    except Exception as exc:
        if single_task and (current_process is not None or motion_claimed):
            with process_lock:
                washer_motion_unknown = True
            if current_process is not None and current_process.poll() is None:
                try:
                    current_process.kill()
                    current_process.wait(timeout=1)
                except (OSError, subprocess.SubprocessError):
                    pass
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
        try:
            if model is not None:
                model.close()
        finally:
            if owner_done is not None:
                owner_done.set()
                if owner_watcher is not None and owner_watcher.ident is not None:
                    owner_watcher.join(timeout=2 * WASHER_OWNER_POLL_S)
            with process_lock:
                close_washer_cancel_fd()
                if request_id is not None:
                    remember_washer_request(request_id, task_id, "finished")
                current_washer_request = None
                current_process = None
                current_task_id = None
                current_operation = None
                current_policy_cancel = None
                current_washer_finish = None
            run_lock.release()


def stop_current_task(clothes_task=None) -> tuple[int, dict]:
    if clothes_task is None and takeover.active():
        snapshot = takeover.status()['episode']
        try:
            return HTTPStatus.OK, takeover.command({'command': 'pause', 'episodeId': snapshot['id'],
                'revision': snapshot['revision'], 'requestId': 'interrupt-'+str(snapshot['revision'])})
        except (Conflict, ValueError) as exc:
            return HTTPStatus.CONFLICT, {"success": False, "message": str(exc)}
    with process_lock:
        process = current_process
        task_id = current_task_id
        if clothes_task is not None and (task_id != clothes_task or current_operation != "clothes-robot"):
            return HTTPStatus.CONFLICT, {"success": False, "message": "当前没有正在运行的对应测试"}
        single_task = current_operation == "washer-single-task"
        clothes_motion = current_operation in {"clothes-robot", "clothes-restore"}
        policy_cancel = current_policy_cancel
        if policy_cancel is not None:
            policy_cancel.set()
            close_washer_cancel_fd()
    if process is None or process.poll() is not None:
        if policy_cancel is not None:
            return HTTPStatus.ACCEPTED, {
                "success": True, "message": "已请求停止模型加载或后态核验",
            }
        return HTTPStatus.CONFLICT, {"success": False, "message": "当前没有正在运行的测试"}

    task_name = TASKS.get(task_id, {}).get("name", task_id or "未知任务")
    try:
        if os.name == "nt" or clothes_motion:
            process.send_signal(signal.SIGINT)
        else:
            os.killpg(process.pid, signal.SIGINT)
        if single_task or clothes_motion:
            return HTTPStatus.ACCEPTED, {
                "success": True, "message": f"已请求 {task_name} 停止继续下发，轨迹状态待确认",
            }
        return HTTPStatus.OK, {"success": True, "message": f"已向 {task_name} 发送 Ctrl+C"}
    except Exception as exc:
        return HTTPStatus.INTERNAL_SERVER_ERROR, {"success": False, "message": f"发送 Ctrl+C 失败: {exc}"}


def restore_robot_task(task_id: str) -> tuple[int, dict]:
    """执行机器人复位操作"""
    global current_task_id, current_operation
    if isinstance(task_id, str) and task_id in CLOTHES_TASKS:
        return HTTPStatus.CONFLICT, {"success": False, "message": "衣物任务请人工切到预设姿态和导航点"}
    if model_validation_only():
        return robot_actions_disabled()
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


def configuration_status(client_ip=None):
    with process_lock:
        task_id, operation = current_task_id, current_operation
        sofa_grasp_hand = current_sofa_grasp_hand
        motion_unknown = washer_motion_unknown
        active_request = None
        if (client_ip in ("127.0.0.1", "::1", get_robot_host())
                and current_washer_request is not None and current_process is not None
                and current_policy_cancel is not None
                and not current_policy_cancel.is_set()):
            active_request = {"taskId": task_id, "requestId": current_washer_request}
    config = web_config.load()
    capabilities = dict(web_config.CAPABILITIES)
    clothes_status = {task_id: clothes_readiness(task_id=task_id) for task_id in CLOTHES_TASKS}
    clothes_models = {key: "ready" if model.ready else "off" for key, model in clothes_policies.items()}
    model_transition = {"clothes-model-start": "loading", "clothes-model-stop": "closing"}.get(operation)
    if model_transition is not None:
        clothes_models[task_id] = model_transition
    if model_validation_only():
        return {
            "success": True, "config": config, "robotActionsEnabled": False, "clothesReadiness": clothes_status,
            "clothesModels": clothes_models, "sofaModel": clothes_models[SOFA_TASK_ID], "sofaGraspHand": sofa_grasp_hand,
            "washerMotionUnknown": False, "capabilities": {**capabilities, "human_takeover": False},
            "locked": run_lock.locked(), "taskId": task_id, "operation": operation,
            "takeover_readiness": {"ready": False, "blockers": ["当前网页未开放人工接管"]},
            "episode": None, "recording_error": None, "managedWasher": None,
        }
    washer_config = single_task_config()
    managed_washer_status = None
    if washer_config and "managedPolicy" in washer_config:
        if motion_unknown:
            status = "WASHER_MOTION_UNKNOWN"
        elif run_lock.locked() or takeover.active():
            status = "WASHER_SERVICE_BUSY"
        else:
            managed, status = managed_washer_readiness(washer_config)
            if status == "READY":
                support = {
                    "robotHost": get_robot_host(),
                    "taskIds": list(TASK_IDS),
                    "timingBasis": washer_config["managedPolicy"]["timingBasis"],
                    "cancelProtocol": WASHER_CANCEL_PROTOCOL,
                }
                if "maxDurationS" in washer_config:
                    support["maxDurationS"] = managed_max_duration(
                        washer_config, cleanup_budget_s=managed.CLEANUP_BUDGET_S,
                    )
                else:
                    support["deadlineOwner"] = "client"
                    support["openDeadlineFinish"] = True
                capabilities["washer_single_task_v1"] = support
        motion_unknown = motion_unknown or status == "WASHER_MOTION_UNKNOWN"
        managed_washer_status = {
            "status": status, "timingBasis": washer_config["managedPolicy"]["timingBasis"],
            "taskIds": list(TASK_IDS),
        }
        if active_request is not None:
            managed_washer_status["activeRequest"] = active_request
    elif washer_config and not motion_unknown and not config["human_takeover_enabled"]:
        capabilities["washer_single_task_v1"] = {
            "robotHost": get_robot_host(),
            "maxDurationS": washer_config["maxDurationS"],
        }
    runtime = takeover.status()
    episode = runtime['episode']
    active = takeover.active()
    return {"success": True, "config": config, "washerMotionUnknown": motion_unknown, "capabilities": {**capabilities,"human_takeover":runtime["readiness"]["ready"],"action_chunk_recording":runtime["readiness"]["persistent_recording_ready"],"timeline":True,"inline_review_v1":True},
            "locked": run_lock.locked() or active,
            "taskId": episode['taskId'] if active else task_id,
            "operation": 'takeover' if active else operation,
            "takeover_readiness": runtime['readiness'], "episode": episode,"recording_error":runtime.get("recording_error"),
            "managedWasher": managed_washer_status, "clothesReadiness": clothes_status, "clothesModels": clothes_models,
            "sofaModel": clothes_models[SOFA_TASK_ID],
            "sofaGraspHand": sofa_grasp_hand}


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
        if path in {"/data_browser.js", "/data_browser.css", "/episode_review.js"}:
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
            if model_validation_only():
                self.send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"success": False, "reason": "模型验证模式"})
                return
            status, payload = vr_proxy.fetch_status()
            self.send_json(status, payload)
            return
        if path == "/api/episode":
            if model_validation_only():
                self.send_json(HTTPStatus.OK, {"success": True, "episode": None})
                return
            self.send_json(HTTPStatus.OK, takeover.status())
            return
        if path == "/api/config":
            try:
                self.send_json(HTTPStatus.OK, configuration_status(self.client_address[0]))
            except (OSError, ValueError) as exc:
                self.send_json(HTTPStatus.BAD_REQUEST, {"success": False, "message": f"无法读取配置: {exc}"})
            return
        if path == "/api/tasks":
            self.send_json(HTTPStatus.OK, {"success": True, "tasks": TASKS})
            return
        if path == "/api/clothes/preflight":
            query = parse_qs(urlparse(self.path).query, keep_blank_values=True)
            task_id = query.get("taskId", [CLOTHES_TASK_ID])[0]
            if task_id not in CLOTHES_TASKS:
                self.send_json(HTTPStatus.BAD_REQUEST, {"success": False, "message": "衣物任务标识无效"})
                return
            self.send_json(HTTPStatus.OK, {"success": True, "preflight": clothes_readiness(task_id=task_id)})
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
        clothes_model_paths = {"/api/clothes/model/start", "/api/clothes/model/stop", "/api/clothes/stop-test"}
        clothes_gripper_path = "/api/clothes/gripper"
        if path not in {"/api/run-task", "/api/run-washer-task", "/api/cancel-washer-task",
                        "/api/finish-washer-task", "/api/restore-task", "/api/config", "/api/gripper",
                        "/api/episode/command", "/api/episode/review", "/api/run-clothes-robot", "/api/clothes/restore-pose",
                        clothes_gripper_path, *clothes_model_paths}:
            self.send_json(HTTPStatus.NOT_FOUND, {"success": False, "message": "Not found"})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if not 0 <= content_length <= (262144 if path=='/api/episode/review' else 65536):
                raise ValueError("请求过大")
            raw_body = self.rfile.read(content_length).decode("utf-8") if content_length else "{}"
            body = json.loads(raw_body or "{}")
            if not isinstance(body, dict):
                raise ValueError("JSON对象必需")
        except (ValueError, UnicodeDecodeError):
            self.send_json(HTTPStatus.BAD_REQUEST, {"success": False, "message": "JSON格式不正确"})
            return

        if path == "/api/gripper":
            if self.headers.get_content_type() != "application/json":
                self.send_json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"success": False, "message": "夹爪操作需要 JSON 请求"})
            else:
                self.send_json(*control_gripper(body))
            return

        if path in clothes_model_paths:
            task_id = body.get("taskId")
            if set(body) != {"taskId"} or not isinstance(task_id, str) or task_id not in CLOTHES_TASKS:
                self.send_json(HTTPStatus.BAD_REQUEST, {"success": False, "message": "衣物任务标识无效"})
            elif path == "/api/clothes/stop-test":
                status, payload = (stop_current_task(clothes_task=task_id)
                                   if clothes_motion_allowed(task_id) else robot_actions_disabled())
                self.send_json(status, payload)
            else:
                self.send_json(*set_clothes_model(task_id, start=path == "/api/clothes/model/start"))
            return

        clothes_control_paths = {"/api/run-clothes-robot", "/api/clothes/restore-pose", clothes_gripper_path}
        if model_validation_only() and path not in {"/api/run-task", "/api/config", *clothes_control_paths}:
            status, payload = robot_actions_disabled()
            self.send_json(status, payload)
            return
        if path == clothes_gripper_path:
            task_id, side, action = body.get("taskId"), body.get("side"), body.get("action")
            if (set(body) != {"taskId", "side", "action"} or task_id not in {CLOTHES_TASK_ID, HANG_TASK_ID, PUT_WASHER_TASK_ID, CLOSE_WASHER_TASK_ID, MEM_CLOSE_WASHER_TASK_ID}
                    or side not in {"left", "right"} or action not in {"open", "close"}):
                self.send_json(HTTPStatus.BAD_REQUEST, {"success": False, "message": "衣物任务夹爪操作无效"})
            else:
                self.send_json(*control_clothes_gripper(task_id, side, action))
            return
        if path in clothes_control_paths:
            task_id = body.get("taskId")
            grasp_hand = body.get("graspHand")
            valid_fields = {"taskId"}
            if path == "/api/run-clothes-robot" and task_id == SOFA_TASK_ID and grasp_hand in SOFA_GRASP_HANDS:
                valid_fields.add("graspHand")
            if set(body) != valid_fields or not isinstance(task_id, str) or task_id not in CLOTHES_TASKS:
                self.send_json(HTTPStatus.BAD_REQUEST, {"success": False, "message": "衣物任务标识无效"})
            elif not clothes_motion_allowed(task_id):
                status, payload = robot_actions_disabled()
                self.send_json(status, payload)
            else:
                if path == "/api/clothes/restore-pose":
                    status, payload = _run_clothes_process(task_id, real_robot=True, restore_pose=True)
                elif grasp_hand is None:
                    status, payload = run_clothes_robot_task(task_id)
                else:
                    status, payload = run_clothes_robot_task(task_id, grasp_hand=grasp_hand)
                self.send_json(status, payload)
            return
        if path in {"/api/episode/command","/api/episode/review"}:
            try:
                self.send_json(HTTPStatus.OK, takeover.save_review(body) if path.endswith('/review') else takeover.command(body))
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
                normalized = web_config.normalize(body)
                changed_robot = normalized["robot_host"] != get_robot_host()
                uploader = chunk_log._uploader
                if changed_robot and uploader is not None and uploader.status()["pending_records"]:
                    raise ValueError("采集数据尚未传完，请传输完成后再修改机器人 IP")
                config = web_config.save(body)
                configure_proxy_bypass()
                if changed_robot and uploader is not None:
                    uploader.transport = chunk_log.Transport(host=config["robot_host"])
                    uploader.anchor_value = None
                    uploader.anchor_at = 0
                self.send_json(HTTPStatus.OK, {"success": True, "config": config,
                    "message": f"配置已保存，后续任务使用机器人 {config['robot_host']}"})
            except (OSError, ValueError) as exc:
                self.send_json(HTTPStatus.BAD_REQUEST, {"success": False, "message": f"配置未保存: {exc}"})
            finally:
                run_lock.release()
            return
        if path in {"/api/run-washer-task", "/api/cancel-washer-task", "/api/finish-washer-task"}:
            task_id, request_id = body.get("taskId"), body.get("requestId")
            request_fields = {"taskId", "requestId"}
            if path == "/api/finish-washer-task" and body.get("completionSource") == "client_deadline":
                request_fields.add("completionSource")
            if path == "/api/run-washer-task" and set(body) == {"taskId"}:
                status, payload = run_robot_task(task_id, single_task=True, client_ip=self.client_address[0])
            elif set(body) != request_fields or not valid_washer_request_id(request_id):
                status, payload = washer_refusal(
                    task_id, "WASHER_REQUEST_ID_INVALID", status=HTTPStatus.BAD_REQUEST,
                )
            elif path == "/api/cancel-washer-task":
                status, payload = cancel_washer_task(task_id, request_id, self.client_address[0])
            elif path == "/api/finish-washer-task":
                status, payload = finish_washer_task(
                    task_id, request_id, self.client_address[0], body.get("completionSource"),
                )
            else:
                status, payload = run_robot_task(
                    task_id, single_task=True, client_ip=self.client_address[0], request_id=request_id,
                    owner_connection=self.connection,
                )
            if valid_washer_request_id(request_id):
                payload["requestId"] = request_id
        elif path == "/api/restore-task":
            status, payload = restore_robot_task(body.get("taskId"))
        else:
            status, payload = run_robot_task(body.get("taskId"))
        try:
            self.send_json(status, payload)
        except (BrokenPipeError, ConnectionResetError):
            if path != "/api/run-washer-task":
                raise

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
        try:
            print(f"[HTTP] {self.address_string()} - {fmt % args}", flush=True)
        except BrokenPipeError:
            pass


def main():
    if not model_validation_only():
        from chunk_log import shared_uploader
        shared_uploader()  # recover only file transfers; never resume a robot task
    print("=" * 60)
    print("Agibot 综合任务测试服务")
    print("=" * 60)
    print(f"测试页面: http://localhost:{PORT}/intergrated_test.html")
    print("按 Ctrl+C 停止服务")
    print("=" * 60)
    def exit_server(_signum, _frame):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, exit_server)
    try:
        with ThreadingHTTPServer((HOST, PORT), IntegratedHandler) as server:
            server.serve_forever()
    finally:
        with run_lock:
            for model in clothes_policies.values():
                model.close()


if __name__ == "__main__":
    main()
