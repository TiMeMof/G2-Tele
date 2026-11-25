"""Clothing model selection and offline inference. Contains no robot client."""
from contextlib import ExitStack, closing, contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np

TASK_ID = "right_arm_take_off_clothes"
HANG_TASK_ID = "right_arm_hang_clothes"
PUT_WASHER_TASK_ID = "左臂放浅蓝色衣服进洗衣机"
CLOSE_WASHER_TASK_ID = "双臂关洗衣机门"
MEM_CLOSE_WASHER_TASK_ID = "双臂关洗衣机门（历史帧MEM）"
SOFA_TASK_ID = "left_arm_pick_blue_clothes_from_sofa"
SOFA_GRASP_HANDS = ("left", "right")
BOTH_ARMS = "both_arms"
CLOTHES_TASKS = {
    HANG_TASK_ID: "挂衣服",
    SOFA_TASK_ID: "沙发拿浅蓝色衣服",
    TASK_ID: "摘衣服",
    PUT_WASHER_TASK_ID: "左臂放浅蓝色衣服进洗衣机",
    CLOSE_WASHER_TASK_ID: "双臂关洗衣机门",
    MEM_CLOSE_WASHER_TASK_ID: "双臂关洗衣机门（历史帧MEM）",
}
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config_V2_0.json"
REPORT_PATH = Path(__file__).resolve().parent / "model_logs" / "clothes_validation_latest.json"
MODEL_LAUNCHER = Path(__file__).resolve().parents[3] / "openpi" / "serve_take_off_clothes.sh"


def validation_report_path(task_id):
    if task_id == TASK_ID:
        return REPORT_PATH
    return REPORT_PATH.with_name(f"{task_id}_validation_latest.json")


def load_task_spec(config_path=CONFIG_PATH, *, task_id=TASK_ID):
    if task_id not in CLOTHES_TASKS:
        raise ValueError("未知衣物任务")
    spec = json.loads(Path(config_path).read_text(encoding="utf-8"))[task_id]
    if spec.get("policy_host") != "127.0.0.1":
        raise ValueError("衣物任务仅连接本机推理服务")
    asset_id = spec.get("asset_id")
    if not isinstance(asset_id, str) or asset_id in ("", ".", "..") or Path(asset_id).name != asset_id:
        raise ValueError("衣物模型归一化参数标识无效")
    if not isinstance(spec.get("policy_config"), str) or not spec["policy_config"].strip():
        raise ValueError("缺少衣物模型加载配置")
    if type(spec.get("policy_port")) is not int or not 1 <= spec["policy_port"] <= 65535:
        raise ValueError("衣物模型端口无效")
    if not isinstance(spec.get("prompt"), str) or not spec["prompt"].strip():
        raise ValueError("缺少衣物任务英文指令")
    if spec.get("arm_type") not in ("left_arm", "right_arm", BOTH_ARMS):
        raise ValueError("衣物任务需指定左臂、右臂或双臂")
    checkpoint = Path(spec["checkpoint"]).resolve(strict=True)
    for path in (checkpoint / "model.safetensors", checkpoint / "assets" / asset_id / "norm_stats.json"):
        if not path.is_file():
            raise ValueError("模型文件不完整：" + str(path))
    return {**spec, "task_id": task_id, "checkpoint": str(checkpoint)}


def synthetic_observation(spec):
    """Use normalization means and generated pixels, without reading robot sensors."""
    stats_path = Path(spec["checkpoint"]) / "assets" / spec["asset_id"] / "norm_stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))["norm_stats"]
    state = np.asarray(stats["state"]["mean"], dtype=np.float32)
    if state.shape != (16,) or not np.isfinite(state).all():
        raise ValueError("衣物任务状态归一化参数无效")
    y, x = np.indices((224, 224), dtype=np.uint16)
    image = np.stack((x % 256, y % 256, (x + y) % 256), axis=-1).astype(np.uint8)
    observation = {
        "state": state,
        "images": {
            "cam_head": image,
            "cam_left": np.roll(image, 37, axis=1).copy(),
            "cam_right": np.flip(image, axis=1).copy(),
        },
        "prompt": spec["prompt"],
    }
    if spec.get('temporal_memory'):
        stamp = time.time_ns()
        observation['observation/timestamp_ns'] = stamp
        observation['history_samples'] = [
            {**observation, 'observation/timestamp_ns': stamp + round(offset * 1e9)}
            for offset in (-1.0, -0.5)
        ]
    return observation


def validate_metadata(metadata, spec):
    expected = {
        "task_id": spec["task_id"],
        "model_config": spec["policy_config"],
        "asset_id": spec["asset_id"],
        "checkpoint": spec["checkpoint"],
        "prompt": spec["prompt"],
        "action_space": "joint_absolute",
        "action_shape": [30, 16],
        "inference_only": True,
    }
    if not isinstance(metadata, dict) or any(metadata.get(k) != v for k, v in expected.items()):
        raise ValueError("本机端口上的模型身份与所选衣物任务不匹配")
    if spec.get('temporal_memory'):
        model_input = metadata.get('model_input', {})
        if (model_input.get('temporal_memory') is not True
                or model_input.get('history_seconds') != [-1.0, -0.5]
                or model_input.get('temporal_operator_version') != 'mem_composed_v1'):
            raise ValueError('关门模型的时序记忆配置不匹配')


def validate_actions(response):
    actions = np.asarray(response["actions"])
    if actions.shape != (30, 16) or actions.dtype.kind not in "iuf" or not np.isfinite(actions).all():
        raise ValueError(f"模型输出无效：{actions.shape}，要求30×16且全部为有限数值")
    return actions


class ModelConnection:
    def __init__(self, spec):
        from openpi_client import msgpack_numpy
        from websockets.sync.client import connect

        self.codec = msgpack_numpy
        self.temporal_memory = spec.get('temporal_memory') is True
        self.connection = connect(
            f"ws://{spec['policy_host']}:{spec['policy_port']}",
            proxy=None, compression=None, max_size=20_000_000, open_timeout=3, close_timeout=2,
        )
        try:
            self.metadata = self.codec.unpackb(self.connection.recv(timeout=5))
            validate_metadata(self.metadata, spec)
        except BaseException:
            self.close()
            raise

    def infer(self, observation, *, timeout=40):
        if self.temporal_memory:
            from temporal_washer import policy_payload
            observation = policy_payload(observation)
        self.connection.send(self.codec.Packer().pack(observation))
        raw = self.connection.recv(timeout=timeout)
        if isinstance(raw, str):
            raise RuntimeError("模型推理失败：" + raw[-3000:])
        return self.codec.unpackb(raw)

    def close(self):
        self.connection.close()


@contextmanager
def clothes_policy_slot():
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.with_name(".clothes_policy.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("已有衣物模型在运行，请等待当前任务结束") from exc
        yield


def check_model_cancelled(cancelled):
    if cancelled is not None and cancelled():
        raise InterruptedError("已中止模型加载")


@contextmanager
def launch_policy(spec, *, cancelled=None):
    """Own the subprocess until the caller closes the session."""
    with tempfile.TemporaryFile() as log:
        process = subprocess.Popen(
            ["bash", str(MODEL_LAUNCHER), "--task", spec["task_id"]],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                check_model_cancelled(cancelled)
                if process.poll() is not None:
                    size = os.fstat(log.fileno()).st_size
                    tail = os.pread(log.fileno(), 3500, max(0, size - 3500)).decode(errors="replace")
                    raise RuntimeError(f"模型启动失败（退出码 {process.returncode}）\n{tail}")
                try:
                    with closing(ModelConnection(spec)):
                        break
                except ConnectionRefusedError:
                    time.sleep(.25)
            else:
                raise TimeoutError("等待本机模型加载超时")
            yield process
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


@contextmanager
def open_policy(spec, *, cancelled=None, require_loaded=False):
    """Reuse a matching listener or start a temporary process for this run."""
    with clothes_policy_slot():
        check_model_cancelled(cancelled)
        try:
            model = ModelConnection(spec)
        except ConnectionRefusedError as exc:
            if require_loaded:
                raise RuntimeError(f"请先启动{CLOTHES_TASKS[spec['task_id']]}模型") from exc
        else:
            with closing(model):
                yield model
            return
        with launch_policy(spec, cancelled=cancelled), closing(ModelConnection(spec)) as model:
            yield model


class ClothesPolicy:
    """Keep the web-owned model loaded between test requests."""
    def __init__(self, task_id):
        self.task_id = task_id
        self.session = ExitStack()
        self.process = None

    @property
    def ready(self):
        process = self.process
        return process is not None and process.poll() is None

    def start(self):
        if self.ready:
            return
        spec = load_task_spec(task_id=self.task_id)
        self.close()
        try:
            with closing(ModelConnection(spec)):
                raise RuntimeError(f"{CLOTHES_TASKS[self.task_id]}模型端口已有独立服务")
        except ConnectionRefusedError:
            pass
        self.process = self.session.enter_context(launch_policy(spec))

    def close(self):
        self.session.close()
        self.process = None


def run_validation(config_path=CONFIG_PATH, report_path=None, *, task_id=TASK_ID):
    spec = load_task_spec(config_path, task_id=task_id)
    observation = synthetic_observation(spec)
    started = time.perf_counter()
    with open_policy(spec) as model:
        metadata = model.metadata
        response = model.infer(observation, timeout=60)
    actions = validate_actions(response)
    report = {
        "success": True,
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "task_id": task_id,
        "prompt": spec["prompt"],
        "checkpoint": spec["checkpoint"],
        "policy_config": spec["policy_config"],
        "asset_id": spec["asset_id"],
        "input_kind": "synthetic_images_and_normalization_mean_state",
        "camera_keys": list(observation["images"]),
        "state_shape": list(observation["state"].shape),
        "action_shape": list(actions.shape),
        "all_finite": True,
        "robot_connected": False,
        "robot_actions_submitted": 0,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        "policy_timing": response.get("policy_timing", {}),
        "temporal_history_mask": response.get("temporal_history_mask"),
        "model_metadata": metadata,
        "actions": actions.tolist(),
    }
    target = Path(report_path if report_path is not None else validation_report_path(task_id))
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return report
