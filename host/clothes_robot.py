"""Clothing tasks start from the robot's current observation after manual setup."""
from contextlib import closing, suppress
import ipaddress
import json
import math
import os
from pathlib import Path
import tempfile
import threading
import time
from urllib.request import ProxyHandler, Request, build_opener

import numpy as np

from clothes_task import BOTH_ARMS, CONFIG_PATH, SOFA_TASK_ID, TASK_ID, load_task_spec, open_policy, validate_actions
from network_config import configured_robot_host

SETUP_PATH = CONFIG_PATH.with_name("clothes_robot_setup.json")
REPORT_PATH = Path(__file__).with_name("model_logs") / "clothes_robot_latest.json"
ARM_COLUMNS = {"left_arm": slice(0, 7), "right_arm": slice(8, 15)}


def controlled_arms(arm_type):
    return tuple(ARM_COLUMNS) if arm_type == BOTH_ARMS else (arm_type,)


def robot_report_path(task_id, *, restore_pose=False):
    if restore_pose:
        return REPORT_PATH.with_name(f"{task_id}_pose_reset_latest.json")
    if task_id == TASK_ID:
        return REPORT_PATH
    return REPORT_PATH.with_name(f"{task_id}_robot_latest.json")


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def vector(value, size):
    return (isinstance(value, list) and len(value) == size
            and all(type(v) in (float, int) and math.isfinite(v) for v in value))


def check_setup(setup):
    """Check connection and execution timing settings."""
    blockers = []
    if not isinstance(setup, dict):
        return ["摘衣服实机配置必须为JSON对象"]
    if setup.get("schema") != "clothes-robot-setup/1":
        blockers.append("衣物任务实机配置格式不匹配")
    robot = setup.get("robot") if isinstance(setup.get("robot"), dict) else {}
    try:
        address = ipaddress.IPv4Address(robot.get("host"))
        if address.is_unspecified or address.is_multicast or address.is_loopback:
            raise ValueError("invalid robot address")
        if not isinstance(robot.get("model"), str) or not robot["model"].strip():
            raise ValueError("missing model")
        for name in ("management_port", "control_port"):
            if type(robot.get(name)) is not int or not 1 <= robot[name] <= 65535:
                raise ValueError("invalid port")
    except (ipaddress.AddressValueError, TypeError, ValueError):
        blockers.append("目标机器人地址与型号尚未核实")
    control = setup.get("control") if isinstance(setup.get("control"), dict) else {}
    ranges = {"trajectory_reference_time_s": (.1, 5),
              "max_duration_s": (1, 120),
              "max_observation_age_s": (.05, 2), "arrival_tolerance_rad": (.001, .1),
              "arrival_timeout_s": (1, 30)}
    if (any(type(control.get(k)) not in (int, float) or not math.isfinite(control[k])
            or not low <= control[k] <= high for k, (low, high) in ranges.items())
            or type(control.get("max_chunks")) is not int or not 1 <= control["max_chunks"] <= 300):
        blockers.append("实机执行时长或到位检查参数无效")
    return blockers


def load_robot_setup(path):
    setup = json.loads(Path(path).read_text(encoding="utf-8"))
    host = configured_robot_host()
    if host and isinstance(setup, dict) and isinstance(setup.get("robot"), dict):
        setup["robot"] = {**setup["robot"], "host": host}
    return setup


def clothes_motion_allowed(task_id):
    return (os.environ.get("VLA_MODEL_VALIDATION_ONLY") != "1"
            or task_id in os.environ.get("VLA_CLOTHES_LIVE_TASK", "").split(","))


def saved_reset_pose(setup, task_id):
    poses = setup.get("reset_poses")
    pose = poses.get(task_id) if isinstance(poses, dict) else None
    if pose is None and task_id == SOFA_TASK_ID:
        pose = setup.get("sofa_reset_pose")
    if (not isinstance(pose, dict) or pose.get("robot") != setup["robot"]
            or not isinstance(pose.get("recorded_at"), str)):
        return None
    limits = setup.get("joint_limits")
    step = setup["control"].get("max_joint_step_rad")
    if (not isinstance(limits, dict) or limits.get("robot_model") != setup["robot"]["model"]
            or limits.get("units") != "rad" or type(step) not in (int, float) or not .001 <= step <= 1):
        return None
    for arm in ARM_COLUMNS:
        gripper = arm.replace("_arm", "_gripper")
        if not vector(pose.get(arm), 7) or not vector([pose.get(gripper)], 1):
            return None
        if any(not isinstance(limits.get(part), list) or len(limits[part]) != size
               or any(not vector(pair, 2) or pair[0] >= pair[1] for pair in limits[part])
               for part, size in ((arm, 7), (gripper, 1))):
            return None
        target = np.asarray(pose[arm] + [pose[gripper]])
        lower, upper = np.asarray(limits[arm] + limits[gripper]).T
        if not ((target >= lower) & (target <= upper)).all():
            return None
    return pose


def execution_readiness(config_path=CONFIG_PATH, setup_path=None, *, task_id=TASK_ID):
    setup = {}
    try:
        spec = load_task_spec(config_path, task_id=task_id)
        setup_path = Path(setup_path or spec.get("robot_setup_file") or SETUP_PATH)
        setup = load_robot_setup(setup_path)
        setup_blockers = check_setup(setup)
        if not isinstance(setup, dict):
            setup = {}
    except (KeyError, OSError, ValueError, TypeError) as exc:
        spec = {}
        setup_blockers = ["无法读取衣物任务实机配置：" + str(exc)]
    execution = []
    if not clothes_motion_allowed(task_id):
        execution.append("本任务当前仅开放模型验证")
    if spec.get("robot_execution_enabled") is not True:
        execution.append("任务的实机执行开关保持关闭")
    if setup.get("motion_enabled") is not True:
        execution.append("实机动作总开关保持关闭")
    pose = saved_reset_pose(setup, task_id) if not setup_blockers and spec["arm_type"] == BOTH_ARMS else None
    can_execute = not (setup_blockers or execution)
    return {"task_id": task_id, "config_ready": not setup_blockers, "can_execute": can_execute,
            "can_reset_pose": can_execute and pose is not None, "pose_recorded_at": pose["recorded_at"] if pose else None,
            "setup_blockers": setup_blockers, "execution_blockers": execution,
            "blockers": setup_blockers + execution, "robot": setup.get("robot"),
            "run_until_stopped": spec.get("run_until_stopped") is True,
            "temporal_memory": spec.get("temporal_memory") is True,
            "start_mode": "current_observation",
            "robot_actions_submitted": 0, "robot_connected": False,
            "setup_path": str(setup_path)}


def chunk_command(response, setup, arm_type):
    actions = validate_actions(response)
    command = {"base_link": "base_link", "trajectory_reference_time": setup["control"]["trajectory_reference_time_s"]}
    for arm_name in controlled_arms(arm_type):
        columns = ARM_COLUMNS[arm_name]
        command[arm_name] = {"kind": "JOINT_ABS", "values": actions[:, columns].tolist()}
        command[arm_name.replace("_arm", "_effector")] = actions[:, columns.stop:columns.stop + 1].tolist()
    return command


def check_observation(raw, setup, previous_timestamp=None):
    states = raw["states"]
    if not vector(states.get("arm_joint_states"), 14) or not vector(states.get("gripper_states"), 2):
        raise ValueError("机器人关节观测无效")
    stamp = raw.get("timestamps", {}).get("states")
    if (type(stamp) is not int or (previous_timestamp is not None and stamp <= previous_timestamp)
            or not -.25 <= (time.time_ns() - stamp) / 1e9 <= setup["control"]["max_observation_age_s"]):
        raise ValueError("机器人状态时间戳缺失、重复或过期")
    return stamp


def state_vector(raw):
    states = raw["states"]
    arm, grip = states["arm_joint_states"], states["gripper_states"]
    return arm[:7] + [grip[0]] + arm[7:14] + [grip[1]]


def model_observation(raw, prompt):
    import cv2
    images = {}
    for source, target in (("head", "cam_head"), ("hand_left", "cam_left"), ("hand_right", "cam_right")):
        item = raw["images"][source]
        if item.get("encoding") != "JPEG":
            raise ValueError("衣物任务实机图像要求 JPEG 编码：" + source)
        image = cv2.imdecode(np.frombuffer(item["image_data"], dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("相机图像解码失败：" + source)
        images[target] = image
    observation = {"state": state_vector(raw), "images": images, "prompt": prompt}
    if 'temporal_history' in raw:
        from temporal_washer import timestamp_ns
        observation['observation/timestamp_ns'] = timestamp_ns(raw)
        observation['history_samples'] = []
        for sample in raw['temporal_history']:
            item = model_observation(sample, prompt)
            item['observation/timestamp_ns'] = timestamp_ns(sample)
            observation['history_samples'].append(item)
    return observation


def robot_management(robot, path, *, start=False):
    request = Request(f"http://{robot['host']}:{robot['management_port']}/system/{path}")
    if start:
        request.data = b"{}"
        request.add_header("Content-Type", "application/json")
    with build_opener(ProxyHandler({})).open(request, timeout=10 if start else 3) as response:
        result = json.load(response)
    if result.get("success") is not True:
        raise RuntimeError(result.get("message") or result.get("msg") or f"机器人管理接口 {path} 返回失败")
    return result.get("data", {})


def ensure_remote_env_running(robot):
    status = robot_management(robot, "status")
    if status.get("current_mode") != "remote_env":
        raise RuntimeError("请在机器人端选中 remote_env 模式后重试")
    if status.get("instance_running") is True:
        return
    if status.get("instance_running") is not False or status.get("app_status") != "idle":
        raise RuntimeError("机器人控制服务正在切换状态，请稍后重试")
    robot_management(robot, "start", start=True)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        status = robot_management(robot, "status")
        if status.get("current_mode") == "remote_env" and status.get("instance_running") is True:
            return
        time.sleep(.25)
    raise TimeoutError("机器人 remote_env 控制服务启动超时")


class RobotConnection:
    """Read-only callers connect to an already running service by default."""
    def __init__(self, robot, *, start_if_needed=False):
        import msgpack
        from websockets.sync.client import connect
        self.codec = msgpack
        if start_if_needed:
            ensure_remote_env_running(robot)
        else:
            data = robot_management(robot, "status")
            if data.get("current_mode") != "remote_env" or data.get("instance_running") is not True:
                raise RuntimeError("目标机器人的 remote_env 控制服务尚未就绪")
        self.connection = connect(f"ws://{robot['host']}:{robot['control_port']}",
                                  proxy=None, compression=None, open_timeout=3, close_timeout=2, max_size=20_000_000)

    def _rpc(self, method, params):
        self.connection.send(self.codec.packb({"method": method, "params": params}, use_bin_type=True))
        reply = self.codec.unpackb(self.connection.recv(timeout=5), raw=False)
        if reply.get("error"):
            raise RuntimeError(str(reply["error"]))
        return reply.get("result")

    def observe(self, include_images=True):
        return self._rpc("get_observation", {"include_images": include_images, "include_states": True})

    def execute(self, command):
        result = self._rpc("execute_action", command)
        if result is not True and not (isinstance(result, dict) and result.get("success") is True):
            raise RuntimeError("未确认机器人接收轨迹；不重试")

    def close(self):
        self.connection.close()


def wait_for_arrival(robot, command, setup, arm_type, *, check_gripper=True, pause=time.sleep, clock=time.monotonic):
    pause(command["trajectory_reference_time"])
    deadline = clock() + setup["control"]["arrival_timeout_s"]
    previous, consecutive = None, 0
    positions, target = [], []
    for arm in controlled_arms(arm_type):
        columns = ARM_COLUMNS[arm]
        positions.extend(range(columns.start, columns.stop))
        target.extend(command[arm]["values"][-1])
        if check_gripper:
            positions.append(columns.stop)
            target.extend(command[arm.replace("_arm", "_effector")][-1])
    while clock() < deadline:
        raw = robot.observe(include_images=False)
        timestamp = check_observation(raw, setup)
        if previous is not None and timestamp <= previous:
            pause(.05)
            continue
        previous = timestamp
        observed = np.asarray(state_vector(raw))[positions]
        arrived = np.abs(observed - target).max() <= setup["control"]["arrival_tolerance_rad"]
        consecutive = consecutive + 1 if arrived else 0
        if consecutive >= 3:
            return
        pause(.05)
    raise RuntimeError("轨迹已提交但未确认到位；停止后续动作，不自动重发或复位")


def reset_command(raw, setup, task_id):
    pose = saved_reset_pose(setup, task_id)
    if pose is None:
        raise ValueError("没有与目标机器人匹配的复位姿态")
    start = np.asarray(state_vector(raw))
    target = np.asarray(pose["left_arm"] + [pose["left_gripper"]] + pose["right_arm"] + [pose["right_gripper"]])
    travel = float(np.max(np.abs(target - start)))
    if travel <= setup["control"]["arrival_tolerance_rad"]:
        return None
    steps = max(30, math.ceil(travel / setup["control"]["max_joint_step_rad"]))
    values = np.linspace(start, target, steps + 1)[1:]
    command = {"base_link": "base_link", "trajectory_reference_time": max(2.0, travel / 0.5)}
    for arm, columns in ARM_COLUMNS.items():
        gripper = arm.replace("_arm", "_gripper")
        lower, upper = np.asarray(setup["joint_limits"][arm] + setup["joint_limits"][gripper]).T
        track = np.clip(values[:, columns.start:columns.stop + 1], lower, upper)
        command[arm] = {"kind": "JOINT_ABS", "values": track[:, :7].tolist()}
        command[arm.replace("_arm", "_effector")] = track[:, 7:8].tolist()
    return command


def execute_flow(config_path=CONFIG_PATH, setup_path=None,
                 report_path=None, *, task_id=TASK_ID, restore_pose=False, stop=None, model_factory=None,
                 require_loaded_model=False, grasp_hand=None, robot_factory=RobotConnection,
                 pause=time.sleep, clock=time.monotonic, wait=None):
    """Explicit live entry. Tests inject both transports; defaults remain motion-disabled."""
    stop = stop if stop is not None else threading.Event()
    wait = stop.wait if wait is None else wait
    readiness = execution_readiness(config_path, setup_path, task_id=task_id)
    if not readiness["can_execute"]:
        raise ValueError("；".join(readiness["blockers"]))
    if restore_pose and not readiness["can_reset_pose"]:
        raise ValueError("没有与目标机器人匹配的复位姿态")
    setup_path = Path(readiness["setup_path"])
    spec = load_task_spec(config_path, task_id=task_id)
    prompt = spec["prompt"]
    if task_id == SOFA_TASK_ID and grasp_hand == "left" and not restore_pose:
        prompt = spec["left_grasp_prompt"]
    setup = load_robot_setup(setup_path)
    run_until_stopped = not restore_pose and spec.get("run_until_stopped") is True
    report_path = report_path if report_path is not None else robot_report_path(task_id, restore_pose=restore_pose)
    arm_type = spec["arm_type"]
    arrival_checked_parts = list(controlled_arms(arm_type)) if restore_pose else []
    if restore_pose:
        arrival_checked_parts.extend(arm.replace("_arm", "_gripper") for arm in controlled_arms(arm_type))
    robot = None
    submitted = chunks = 0
    settled = True if restore_pose else None
    last_stamp = None

    def finish(status):
        report = {"task_id": task_id, "status": status,
                  "start_mode": "saved_pose_reset" if restore_pose else "current_observation",
                  "physical_effect_verified": False, "robot_connected": robot is not None,
                  "robot_actions_submitted": submitted, "model_chunks_submitted": chunks,
                  "last_target_arrival_confirmed": settled, "arrival_checked_parts": arrival_checked_parts,
                  "execution_progression": "arrival_confirmation" if restore_pose else "trajectory_time",
                  "run_until_stopped": run_until_stopped,
                  "prompt": prompt}
        write_json(report_path, report)
        return report

    def submit(command):
        nonlocal submitted, settled
        if stop.is_set() or clock() + command["trajectory_reference_time"] > deadline:
            return False
        fresh = execution_readiness(config_path, setup_path, task_id=task_id)
        if not fresh["can_execute"]:
            raise ValueError("；".join(fresh["blockers"]))
        if (load_task_spec(config_path, task_id=task_id) != spec
                or load_robot_setup(setup_path) != setup):
            raise ValueError("衣物模型或实机配置已变更，停止后续下发")
        if stop.is_set():
            return False
        settled = False
        submitted += 1
        robot.execute(command)
        if restore_pose:
            wait_for_arrival(robot, command, setup, arm_type, check_gripper=True, pause=pause, clock=clock)
            settled = True
        else:
            # Keep the trajectory cadence, then let the model react to the next
            # actual observation. The operator can interrupt this timed wait.
            wait(command["trajectory_reference_time"])
            settled = None  # Test execution does not check target arrival.
        return True

    try:
        if stop.is_set():
            return finish("cancelled_before_connect")
        if restore_pose:
            robot = robot_factory(setup["robot"])
            deadline = clock() + setup["control"]["max_duration_s"]
            raw = robot.observe(include_images=False)
            check_observation(raw, setup)
            command = reset_command(raw, setup, task_id)
            if command is None:
                return finish("pose_already_set")
            if submit(command):
                return finish("pose_restored")
            if stop.is_set():
                return finish("stopped_submitting")
            raise ValueError("复位轨迹超出配置允许的执行时长")
        model_session = (open_policy(spec, cancelled=stop.is_set, require_loaded=require_loaded_model)
                         if model_factory is None else closing(model_factory(spec)))
        with model_session as model:
            if stop.is_set():
                return finish("cancelled_before_connect")
            robot = robot_factory(setup["robot"])
            deadline = math.inf if run_until_stopped else clock() + setup["control"]["max_duration_s"]
            chunk_limit = math.inf if run_until_stopped else setup["control"]["max_chunks"]
            while not stop.is_set() and chunks < chunk_limit and clock() < deadline:
                raw = robot.observe()
                last_stamp = check_observation(raw, setup, last_stamp)
                observation = model_observation(raw, prompt)
                response = model.infer(observation)
                if stop.is_set() or clock() >= deadline:
                    break
                check_observation(raw, setup)
                if not submit(chunk_command(response, setup, arm_type)):
                    break
                chunks += 1
        return finish("stopped_submitting" if stop.is_set() else "awaiting_physical_verification")
    except BaseException as exc:
        if isinstance(exc, InterruptedError) and stop.is_set() and robot is None:
            return finish("cancelled_before_connect")
        finish("motion_unknown" if settled is False else "failed")
        raise
    finally:
        if robot is not None:
            with suppress(Exception):
                robot.close()
