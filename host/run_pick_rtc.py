#!/usr/bin/env python3
# -- coding: UTF-8
"""Agibot G2 双臂拿取任务的异步 RTC（Real-Time Chunking）推理脚本。

架构仿照 galbot_rtc_inference.py（基于 threading + 队列的异步 RTC），
但机器人 I/O、图像获取、动作下发、任务配置全部改用 run_pick.py 的方式：
    - 机器人环境: core.corobot_env.g2_corobot_env.G2CorobotEnv
    - 观测读取:   G2CorobotEnv.get_state()  一次返回关节+夹爪+三路图像(JPEG)
    - 动作下发:   G2CorobotEnv.execute({left_arm,left_gripper,right_arm,right_gripper}, ref_times)
    - 复位:       G2CorobotEnv.reset_pos(position)
    - 推理端:     openpi_client.websocket_client_policy (127.0.0.1:8181)
    - 任务配置:   intergrated_test 下的 right/left_add_drawer_config.json

与 run_pick.py（同步串行 6~7Hz、动作块被过早覆盖）的区别：
    * 采集线程持续把 get_state() 写入缓存；
    * 推理线程尽量高频推理，把新动作块按"观测时刻 + 帧序号 * dt"splice 进
      时间索引的执行队列，用新规划覆盖未来帧，但通过 RTC 前缀引导保持连续；
    * 执行器线程按固定频率从队列 pop 出到期的单帧动作下发。
这样推理与执行真正重叠，且动作切换点由 RTC 平滑衔接，而不是被硬覆盖。

RTC 原理与 galbot 版一致：每次推理把"距上次推理已流逝的帧数"作为 index
传给推理端，推理端用上一次的动作缓存作为前缀引导新 chunk 的前若干帧，
使拼接处动作连续。若推理端不支持 RTC，可用 --no-rtc 退化为纯时间索引拼接。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import cv2
import numpy as np

# 获取脚本所在目录的父目录（g2_policy_ctl）的绝对路径，并添加到 sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from openpi_client import websocket_client_policy
from core.corobot_env.g2_corobot_env import G2CorobotEnv

logger = logging.getLogger("run_pick_rtc")

# 动作维度：左臂7 + 左夹爪1 + 右臂7 + 右夹爪1 = 16
ARM_DOF = 7
ACTION_DOF = 16

# 抓取判定：夹爪合拢稳定后的等待时间（秒），与 run_pick.py 保持一致
GRIPPER_SETTLE_SECONDS = 1.5
# 复位/张开夹爪的目标值，与 run_pick.py 的 reset_action 一致
GRIPPER_OPEN_VALUE = -0.785

# ---------------------------------------------------------------------------
# 任务配置加载（复用 run_pick.py 的 TaskConfigLoader）
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TaskSpec:
    task_name: str
    prompt: str
    init_position: Any
    end_position: Any
    arm_type: str


class TaskConfigLoader:
    def __init__(self, config_paths: Sequence[Path | str]):
        self.config_paths = [Path(path) for path in config_paths]
        self._configs = []
        for path in self.config_paths:
            with path.open("r", encoding="utf-8") as handle:
                self._configs.append((path, json.load(handle)))

    def list_available_tasks(self) -> list[str]:
        tasks = []
        for _, config in self._configs:
            for task_name in config.keys():
                if task_name == "task":
                    continue
                if isinstance(config[task_name], dict):
                    tasks.append(task_name)
        return sorted(tasks)

    def resolve_task(self, task_name: str) -> TaskSpec:
        normalized_name = task_name.strip()
        if not normalized_name:
            raise ValueError("task_name 不能为空")

        for path, config in self._configs:
            if normalized_name in config:
                task_config = config[normalized_name]
                if not isinstance(task_config, dict):
                    continue
                if not {"prompt", "init_position", "end_position"}.issubset(task_config.keys()):
                    continue
                arm_type = self._infer_arm_type(path, normalized_name)
                return TaskSpec(
                    task_name=normalized_name,
                    prompt=task_config["prompt"],
                    init_position=task_config["init_position"],
                    end_position=task_config["end_position"],
                    arm_type=arm_type,
                )

        available = ", ".join(self.list_available_tasks())
        raise KeyError(f"未找到任务 '{task_name}'。可用任务：{available}")

    @staticmethod
    def _infer_arm_type(path: Path, task_name: str) -> str:
        lower_name = f"{path.name} {task_name}".lower()
        if "left" in lower_name:
            return "left_arm"
        if "right" in lower_name:
            return "right_arm"
        raise ValueError(f"无法从任务 '{task_name}' 推断机械臂类型")


# ---------------------------------------------------------------------------
# 线程安全队列（复用 galbot 的 StateBuffer 和 RTCActionQueue 设计）
# ---------------------------------------------------------------------------

class StateBuffer:
    """线程安全的状态缓存，保留近期带时间戳的观测帧。"""
    def __init__(self, maxlen: int):
        self._deque = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def append(self, ts: float, data: Any):
        with self._lock:
            self._deque.append((ts, data))

    def latest(self) -> Optional[Tuple[float, Any]]:
        with self._lock:
            return self._deque[-1] if self._deque else None

    def clear(self):
        with self._lock:
            self._deque.clear()


class RTCActionQueue:
    """RTC 动作队列：每条 (target_time, action_16d)，按目标墙钟时间执行。

    推理线程把新 chunk splice 进来时，丢弃所有目标时间晚于本 chunk 起点的
    旧条目（用新规划覆盖未来），实现 real-time chunking 的"拼接"。
    """
    def __init__(self):
        self._items = []  # [(target_time, action_16d), ...] 按 target_time 递增
        self._last_action = None
        self._last_action_time = None
        self._lock = threading.Lock()

    def splice(self, chunk: np.ndarray, T_obs: float, dt: float, now: float,
               skip_frames: int = 0):
        """把新 chunk 写入队列。

        chunk[i] 的目标执行时间 = T_obs + (i+1)*dt（相对该 chunk 观测时刻）。
        跳过已经过期（target_time <= now）的帧，其余写入；同时丢弃队列里
        target_time >= 新帧起点的旧条目，让新规划覆盖未来。

        skip_frames: 跳过新 chunk 的前 skip_frames 帧不写入（对齐 RTC 语义：
            这 inference_delay 帧由旧 chunk 对应位置执行，新 chunk 前缀只用作
            引导，不是硬约束，直接执行易在切换瞬间抖动）。
        """
        n = len(chunk)
        start_idx = min(max(int(skip_frames), 0), n)
        if start_idx >= n:
            return

        model_times = T_obs + (np.arange(n, dtype=np.float64) + 1.0) * dt
        new_items = [
            (float(model_times[i]), chunk[i].copy())
            for i in range(start_idx, n)
            if model_times[i] > now
        ]
        if not new_items:
            return

        cutoff = new_items[0][0]
        with self._lock:
            kept_items = [x for x in self._items if x[0] < cutoff]
            self._items = kept_items
            self._items.extend(new_items)

    def pop_due(self, now: float, max_lag: Optional[float] = None) -> Optional[np.ndarray]:
        """弹出所有 target_time <= now 的条目，返回最新一条 action。

        max_lag 不为 None 且最新 due 帧也已过期超过 max_lag 时，视为旧 chunk
        残留（例如 grace period 后），丢弃不执行，等待新鲜 chunk。
        """
        with self._lock:
            due = [x for x in self._items if x[0] <= now]
            self._items = [x for x in self._items if x[0] > now]
            if not due:
                return None
            target_time, action = due[-1]
            if max_lag is not None and (now - target_time) > max_lag:
                return None
            self._last_action = action.copy()
            self._last_action_time = target_time
            return action

    def clear(self):
        with self._lock:
            self._items = []
            self._last_action = None
            self._last_action_time = None


# ---------------------------------------------------------------------------
# G2 控制器封装（对齐 run_pick.py 的 G2CorobotEnv 接口）
# ---------------------------------------------------------------------------

class G2Controller:
    """对 G2CorobotEnv 的轻量封装（异步 RTC 版）。

    与 run_pick.py 共享同一套关键设计：
      - G2CorobotEnv.get_state() 一次返回关节+夹爪+三路图像（JPEG）；
      - G2CorobotEnv.execute({left_arm,left_gripper,right_arm,right_gripper}, ref_times)；
      - G2CorobotEnv.reset_pos(position) 复位到指定位姿。
    异步版不维护 obs 滑窗（观测由采集线程写入队列），只暴露线程安全的
    单次读取接口。
    """

    def __init__(self, task_spec: TaskSpec):
        self.task_spec = task_spec
        self.arm_type = task_spec.arm_type
        self.g2_env = G2CorobotEnv()
        # SDK 调用串行化：采集线程与执行线程都会碰 g2_env，用一把锁避免并发
        self.env_lock = threading.Lock()

        # 加载关节限位（用于裁剪动作到安全范围）
        (
            self.left_joint_lower,
            self.left_joint_upper,
            self.right_joint_lower,
            self.right_joint_upper,
        ) = self.load_joint_limits(ROOT_DIR / "config" / "robot_config_ori.json")

        logger.info("G2Controller 初始化完成，任务：%s，使用：%s", task_spec.task_name, self.arm_type)

    def load_joint_limits(self, file_path: Path):
        """加载左右臂关节限位，返回左右臂各自的 (lower数组, upper数组)。"""
        with file_path.open("r", encoding="utf-8") as handle:
            robot_config = json.load(handle)
        left_limits = robot_config["left_arm"]
        right_limits = robot_config["right_arm"]
        left_lower = np.array(left_limits["lower"])
        left_upper = np.array(left_limits["upper"])
        right_lower = np.array(right_limits["lower"])
        right_upper = np.array(right_limits["upper"])
        return left_lower, left_upper, right_lower, right_upper

    def clip_to_limits(self, arm_actions: np.ndarray, arm_side: str) -> np.ndarray:
        """将机械臂动作裁剪到对应臂的关节限位范围内。"""
        if arm_side == "left_arm":
            lower = self.left_joint_lower
            upper = self.left_joint_upper
        else:
            lower = self.right_joint_lower
            upper = self.right_joint_upper

        below = arm_actions < lower
        above = arm_actions > upper
        if below.any() or above.any():
            for j in range(arm_actions.shape[1]):
                if below[:, j].any() or above[:, j].any():
                    n_out = int(below[:, j].sum() + above[:, j].sum())
                    logger.warning(
                        f"{arm_side} joint {j + 1} 有 {n_out} 帧超限，已裁剪到边界"
                    )
        return np.clip(arm_actions, lower, upper)

    # ------------------------- 观测读取 -------------------------

    def read_state(self) -> Tuple[float, dict]:
        """读取当前机器人状态，返回 (ts, obs_dict)。

        obs_dict 包含：
          - "arm_joint_states": 左臂7 + 右臂7，共14维
          - "gripper_states": 左夹爪 + 右夹爪，共2维
          - "images": {"cam_head": BGR, "cam_left": BGR, "cam_right": BGR}
        """
        with self.env_lock:
            obs = self.g2_env.get_state()
        ts = time.time()

        arm_states = obs["observation.states.arm_joint_states"][:14]
        gripper_states = obs["observation.states.gripper_states"][:2]

        # 解码三路图像（JPEG -> BGR）
        images = {
            "cam_head": cv2.imdecode(
                np.frombuffer(obs["observation.images.head"]["image_data"], dtype=np.uint8),
                cv2.IMREAD_COLOR
            ),
            "cam_left": cv2.imdecode(
                np.frombuffer(obs["observation.images.hand_left"]["image_data"], dtype=np.uint8),
                cv2.IMREAD_COLOR
            ),
            "cam_right": cv2.imdecode(
                np.frombuffer(obs["observation.images.hand_right"]["image_data"], dtype=np.uint8),
                cv2.IMREAD_COLOR
            ),
        }

        return ts, {
            "arm_joint_states": arm_states,
            "gripper_states": gripper_states,
            "images": images,
        }

    # ------------------------- 整体运动 -------------------------

    def reset_pos(self, position: Any):
        """阻塞地把机器人移动到指定位姿。"""
        with self.env_lock:
            self.g2_env.reset_pos(position)

    # ------------------------- 单步动作下发 -------------------------

    def send_step(self, action_16d: np.ndarray, ref_time: float):
        """把单帧 16 维动作（左臂7+左爪1+右臂7+右爪1）下发给机器人。

        action_16d: [left_arm_7, left_gripper_1, right_arm_7, right_gripper_1]
        ref_time: 轨迹参考时间（秒），控制执行速度
        """
        left_arm = action_16d[:7].reshape(1, -1).tolist()
        left_gripper = action_16d[7:8].reshape(1, -1).tolist()
        right_arm = action_16d[8:15].reshape(1, -1).tolist()
        right_gripper = action_16d[15:16].reshape(1, -1).tolist()

        with self.env_lock:
            self.g2_env.execute(
                {
                    "left_arm": left_arm,
                    "left_gripper": left_gripper,
                    "right_arm": right_arm,
                    "right_gripper": right_gripper,
                },
                ref_times=ref_time,
            )


# ---------------------------------------------------------------------------
# RTC 异步推理引擎
# ---------------------------------------------------------------------------

class RTCInferenceEngine:
    """RTC 异步推理引擎：3 线程 + 2 队列 + grace period。"""

    def __init__(self, controller: G2Controller, policy, cfg: dict):
        self.controller = controller
        self.policy = policy
        self.cfg = cfg
        self.policy_lock = threading.Lock()

        # 2 个队列：Q1 观测缓存 / Q2 动作队列
        self.q1_state = StateBuffer(maxlen=cfg["collect_rate"])
        self.q2_action = RTCActionQueue()

        # 线程控制
        self.stop_event = threading.Event()
        self.grasp_event = threading.Event()  # 执行器检测到闭合指令时置位
        self.threads = []
        self._run_generation = 0

        # Grace period: 启动后这段时间内执行器不下发动作
        self.grace_period_s = cfg.get("grace_period_s", 1.0)
        self.start_time = None

        # 时间步长（模型动作帧间隔）
        self.dt = 1.0 / cfg["publish_rate"]

        # RTC 参数（传给推理端 infer_args）
        self.use_rtc = cfg.get("use_rtc", True)
        self.chunk_size = cfg["chunk_size"]
        self.inference_delay = cfg.get("inference_delay", 4)
        self.max_guidance_weight = cfg.get("max_guidance_weight", 5.0)
        self.num_flow_steps = cfg.get("num_flow_steps", 10)

        # 执行器参数
        self.executor_rate = cfg["executor_rate"]
        self.step_ref_time = 1.0 / cfg["publish_rate"]  # 单帧执行的 ref_time

        # 抓取检测
        self.close_threshold = cfg.get("grasp_close_cmd_threshold", -0.05)
        self.fail_threshold = cfg.get("fail_threshold", -0.01)
        self.success_threshold = cfg.get("success_threshold", -0.06)

    def _run_active(self, generation: int) -> bool:
        return (not self.stop_event.is_set()) and generation == self._run_generation

    # ------------------------- 采集线程 -------------------------

    def _collect_thread(self, generation: int):
        """按 collect_rate 采集机器人状态写入 Q1。"""
        dt = 1.0 / self.cfg["collect_rate"]
        while self._run_active(generation):
            t0 = time.time()
            try:
                ts, obs_dict = self.controller.read_state()
                if not self._run_active(generation):
                    break
                self.q1_state.append(ts, obs_dict)
            except Exception as e:
                logger.error(f"[collect] 读取状态失败: {e}")
            time.sleep(max(0, dt - (time.time() - t0)))

    # ------------------------- 推理线程 -------------------------

    def _inference_thread(self, generation: int):
        """读取 Q1 -> 推理 -> splice 到 Q2。

        RTC 模式（use_rtc=True）：
            通过 obs['infer_args'] 传 RTC 参数（index, inference_delay 等），
            推理端用 self._last_raw_actions 自动缓存上一次模型输出当 prefix。
        标准模式（use_rtc=False）：
            不传 infer_args，纯时间索引拼接，但仍通过 splice 避免动作硬覆盖。
        """
        max_rate = self.cfg.get("max_inference_rate", 10)
        min_interval = 1.0 / max_rate
        prompt = self.controller.task_spec.prompt

        # 上一次推理的观测时间戳，用于换算 index（=已流逝帧数）
        last_infer_obs_ts = None

        while self._run_active(generation):
            t0 = time.time()

            state_entry = self.q1_state.latest()
            if state_entry is None:
                time.sleep(0.01)
                continue
            ts_obs, obs_dict = state_entry

            arm_states = obs_dict["arm_joint_states"]
            gripper_states = obs_dict["gripper_states"]
            images = obs_dict["images"]

            # 构建 payload（对齐 run_pick.py 的格式）
            state = list(arm_states[:7]) + [gripper_states[0]] + list(arm_states[7:14]) + [gripper_states[1]]
            payload = {
                "state": state,
                "images": images,
                "prompt": prompt,
            }

            # 计算 index: 距上次推理观测时刻流逝了多少模型帧
            if self.use_rtc:
                if last_infer_obs_ts is None:
                    index = -1  # 首次推理走标准模式
                else:
                    elapsed_frames = int(round((ts_obs - last_infer_obs_ts) / self.dt))
                    # clamp 到 [0, chunk_size-1]，超过 chunk_size 视为旧规划失效退回标准模式
                    if elapsed_frames >= self.chunk_size:
                        index = -1
                    else:
                        index = max(0, elapsed_frames)

                payload["infer_args"] = {
                    "index": index,
                    "inference_delay": self.inference_delay,
                    "max_guidance_weight": self.max_guidance_weight,
                    "num_flow_steps": self.num_flow_steps,
                }

            try:
                with self.policy_lock:
                    if not self._run_active(generation):
                        break
                    response = self.policy.infer(payload)
            except Exception as e:
                if not self._run_active(generation):
                    break
                logger.error(f"[inference] 推理失败: {e}")
                time.sleep(min_interval)
                continue

            if not self._run_active(generation):
                break
            chunk = np.asarray(response["actions"], dtype=np.float32)  # (chunk_size, 16)

            # 裁剪到关节限位
            left_arm = self.controller.clip_to_limits(chunk[:, :7], "left_arm")
            right_arm = self.controller.clip_to_limits(chunk[:, 8:15], "right_arm")
            chunk[:, :7] = left_arm
            chunk[:, 8:15] = right_arm

            last_infer_obs_ts = ts_obs

            # splice 到执行队列（以观测时刻为时间基准）
            # RTC 模式（index>=0）：跳过新 chunk 前 inference_delay 帧
            # 标准模式（index<0，首帧/重试）：整段都要写入，不跳帧
            skip = self.inference_delay if (self.use_rtc and payload.get("infer_args", {}).get("index", -1) >= 0) else 0
            self.q2_action.splice(chunk, ts_obs, self.dt, time.time(), skip_frames=skip)

            elapsed = time.time() - t0
            mode_str = f"rtc(index={payload.get('infer_args', {}).get('index', -1)})" if self.use_rtc else "standard"
            logger.info(f"[inference] {elapsed*1000:.1f}ms mode={mode_str}")
            time.sleep(max(0, min_interval - elapsed))

    # ------------------------- 执行器线程 -------------------------

    def _executor_thread(self, generation: int):
        """按 executor_rate 打拍: grace period -> pop_due -> 下发 -> 抓取检测。"""
        dt = 1.0 / self.executor_rate
        max_lag = self.cfg.get("executor_max_lag_s", 0.2)

        while self._run_active(generation):
            t0 = time.time()
            now = time.time()

            # Grace period: 防止重试时模型看到旧状态立即闭合
            if self.start_time and (now - self.start_time) < self.grace_period_s:
                time.sleep(dt)
                continue

            action = self.q2_action.pop_due(now, max_lag=max_lag)
            if action is None:
                time.sleep(dt)
                continue

            # 单帧下发
            try:
                self.controller.send_step(action, self.step_ref_time)
            except Exception as e:
                logger.error(f"[executor] 下发失败: {e}")

            # 抓取检测: policy 下发闭合指令 -> 通知主线程做鲁棒夹取判定
            arm_type = self.controller.arm_type
            gripper_index = 0 if arm_type == "left_arm" else 1
            gripper_cmd = float(action[7 if gripper_index == 0 else 15])
            if gripper_cmd > self.close_threshold:
                logger.info(f"[executor] 检测到闭合指令: {gripper_cmd:.4f}")
                self.grasp_event.set()
                break

            time.sleep(max(0, dt - (time.time() - t0)))

    # ------------------------- 线程编排 -------------------------

    def _start_all_threads(self):
        self.stop_event.clear()
        self.grasp_event.clear()
        self.start_time = time.time()
        self._run_generation += 1
        generation = self._run_generation

        # 清空队列，防上一轮残留
        self.q1_state.clear()
        self.q2_action.clear()

        self.threads = [
            threading.Thread(target=self._collect_thread, args=(generation,), name="collect", daemon=True),
            threading.Thread(target=self._inference_thread, args=(generation,), name="inference", daemon=True),
            threading.Thread(target=self._executor_thread, args=(generation,), name="executor", daemon=True),
        ]
        for t in self.threads:
            t.start()
        logger.info(f"[RTC engine] 3 线程已启动, grace period = {self.grace_period_s}s")

    def _stop_all_threads(self):
        self.stop_event.set()
        for t in self.threads:
            t.join(timeout=2.0)
        alive = [t.name for t in self.threads if t.is_alive()]
        if alive:
            logger.warning(f"[RTC engine] 线程仍在退出中: {alive}")
        else:
            logger.info("[RTC engine] 所有线程已停止")
        self.threads = []

    def run_one_attempt(self) -> bool:
        """执行一次推理+抓取，返回是否成功。"""
        self._start_all_threads()

        max_steps_time = self.cfg["max_steps"] * self.dt
        triggered = self.grasp_event.wait(timeout=max_steps_time)
        self._stop_all_threads()

        if not triggered:
            logger.warning("[RTC engine] 超时未触发闭合指令")
            return False

        # 等待夹爪稳定后重新取样
        time.sleep(GRIPPER_SETTLE_SECONDS)
        _, obs_dict = self.controller.read_state()
        gripper_states = obs_dict["gripper_states"]
        arm_type = self.controller.arm_type
        gripper_index = 0 if arm_type == "left_arm" else 1
        current_gripper = float(gripper_states[gripper_index])

        logger.info(f"[RTC engine] 夹爪当前值: {current_gripper:.4f}")
        if current_gripper > self.fail_threshold:
            logger.warning("[RTC engine] 抓取失败（夹爪为空）")
            return False
        if current_gripper <= self.success_threshold:
            logger.info("[RTC engine] 抓取成功")
            return True

        logger.warning("[RTC engine] 抓取失败（夹爪值在中间区域）")
        return False


# ---------------------------------------------------------------------------
# 主入口（支持夹取失败后重新推理重试）
# ---------------------------------------------------------------------------

def run_inference(controller: G2Controller, policy, cfg: dict):
    """RTC 异步推理主入口，支持夹取失败后重新推理重试。

    流程：
        1. 移动到起始位姿；
        2. 每轮调 engine.run_one_attempt() 跑推理+抓取；
        3. 失败则打开夹爪，从当前状态重新推理，最多 max_retries 次；
        4. 任一轮成功即退出并移动到结束位姿。
    """
    max_retries = cfg.get("max_retries", 3)
    retry_ref_time = cfg.get("retry_ref_time", 0.2)

    engine = RTCInferenceEngine(controller, policy, cfg)

    # 第 0 轮首次执行，之后最多 max_retries 次重试，共 max_retries+1 次机会
    for attempt in range(max_retries + 1):
        if attempt == 0:
            logger.info("--- 第 1 次尝试 ---")
        else:
            logger.info(f"--- 夹取重试 {attempt}/{max_retries}: 打开夹爪，从当前状态重新推理 ---")
            # 打开夹爪
            arm_type = controller.arm_type
            gripper_index = 0 if arm_type == "left_arm" else 1
            reset_action = np.zeros(16, dtype=np.float32)
            reset_action[7 if gripper_index == 0 else 15] = GRIPPER_OPEN_VALUE
            controller.send_step(reset_action, retry_ref_time)
            time.sleep(2)

        success = engine.run_one_attempt()
        if success:
            logger.info(f"夹取成功!（第 {attempt + 1} 次尝试）任务完成。")
            controller.reset_pos(controller.task_spec.end_position)
            return True

        logger.warning(f"第 {attempt + 1} 次夹取失败。")

    logger.error(f"已达最大重试次数上限 ({max_retries + 1} 次)，任务失败。")
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agibot G2 RTC 异步推理拿取任务")
    parser.add_argument("--task", default="left_arm_drawer_pick_omeprazole_bottle",
                        help="任务名称")
    parser.add_argument("--chunk-size", type=int, default=30,
                        help="模型输出的动作块大小（帧数）")
    parser.add_argument("--publish-rate", type=float, default=30.0,
                        help="模型输出的帧率（Hz），决定 dt = 1/publish_rate")
    parser.add_argument("--executor-rate", type=float, default=30.0,
                        help="执行器打拍频率（Hz）")
    parser.add_argument("--collect-rate", type=int, default=30,
                        help="状态采集频率（Hz）")
    parser.add_argument("--max-inference-rate", type=float, default=10.0,
                        help="推理线程最大频率（Hz）")
    parser.add_argument("--inference-delay", type=int, default=4,
                        help="RTC 推理延迟（帧数），新 chunk 前这些帧跳过不写入")
    parser.add_argument("--max-guidance-weight", type=float, default=5.0,
                        help="RTC 前缀引导权重")
    parser.add_argument("--num-flow-steps", type=int, default=10,
                        help="RTC flow 步数")
    parser.add_argument("--grace-period", type=float, default=1.0,
                        help="启动后 grace period（秒），期间不执行动作")
    parser.add_argument("--max-steps", type=int, default=150,
                        help="单次尝试最大步数")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="夹取失败后最大重试次数")
    parser.add_argument("--no-rtc", action="store_true",
                        help="禁用 RTC 模式，退化为纯时间索引拼接")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = parse_args()

    # 加载任务配置
    loader = TaskConfigLoader([
        ROOT_DIR / "config" / "right_add_drawer_config.json",
        ROOT_DIR / "config" / "left_add_drawer_config.json",
    ])
    task_spec = loader.resolve_task(args.task)

    # 构建配置字典
    cfg = {
        "chunk_size": args.chunk_size,
        "publish_rate": args.publish_rate,
        "executor_rate": args.executor_rate,
        "collect_rate": args.collect_rate,
        "max_inference_rate": args.max_inference_rate,
        "inference_delay": args.inference_delay,
        "max_guidance_weight": args.max_guidance_weight,
        "num_flow_steps": args.num_flow_steps,
        "grace_period_s": args.grace_period,
        "max_steps": args.max_steps,
        "max_retries": args.max_retries,
        "use_rtc": not args.no_rtc,
        "grasp_close_cmd_threshold": -0.05,
        "fail_threshold": -0.01,
        "success_threshold": -0.06,
        "executor_max_lag_s": 0.2,
        "retry_ref_time": 0.2,
    }

    # 根据任务调整夹取阈值（对齐 run_pick.py）
    if task_spec.task_name in {"right_arm_pick_tissue", "right_arm_pick_glasses"}:
        cfg["fail_threshold"] = -0.0008
        cfg["success_threshold"] = 0.001

    logger.info(f"任务: {task_spec.task_name}")
    logger.info(f"RTC 模式: {'启用' if cfg['use_rtc'] else '禁用'}")
    logger.info(f"chunk_size={cfg['chunk_size']} publish_rate={cfg['publish_rate']}Hz "
                f"executor_rate={cfg['executor_rate']}Hz")
    if cfg['use_rtc']:
        logger.info(f"inference_delay={cfg['inference_delay']} "
                    f"max_guidance_weight={cfg['max_guidance_weight']}")

    # 初始化控制器和推理端
    controller = G2Controller(task_spec)
    policy = websocket_client_policy.WebsocketClientPolicy("127.0.0.1", 8181)
    logger.info(f"推理服务器: 127.0.0.1:8181")

    # 复位到初始位置
    logger.info("移动到初始位置...")
    controller.reset_pos(task_spec.init_position)
    input("按回车开始 RTC 推理...")

    try:
        success = run_inference(controller, policy, cfg)
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("用户中断")
        sys.exit(1)
    finally:
        logger.info("程序结束")


if __name__ == "__main__":
    main()
