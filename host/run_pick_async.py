#!/usr/bin/env python3
"""完全异步版本的拿取任务实验。

使用 asyncio 实现真正的异步执行：
- 视觉推理、机器人控制、状态读取均异步并发
- 推理和执行可以重叠，提高整体吞吐量
- 非阻塞IO操作，充分利用等待时间
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np


# 获取脚本所在目录的父目录（g2_policy_ctl）的绝对路径，并添加到 sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from openpi_client import websocket_client_policy
from core.corobot_env.g2_corobot_env import G2CorobotEnv

logger = logging.getLogger(__name__)

# 记录文件的存放目录（agibot_policy_ctl/g2_policy_ctl/record）
RECORD_DIR = ROOT_DIR / "record"

# 是否记录推理动作并生成 record 目录下的 JSON 文件
ENABLE_ACTION_RECORDING = False


class ActionRecorder:
    """记录每次任务执行过程中从推理端获取的动作，按任务名和时间戳存为 .json 文件。"""

    def __init__(self, task_name: str, prompt: str, arm_type: str, record_dir: Path = RECORD_DIR):
        self.task_name = task_name
        self.prompt = prompt
        self.arm_type = arm_type
        self.record_dir = Path(record_dir)
        # 任务开始时间戳，作为文件名和记录元数据
        self.start_time = datetime.now()
        self.timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        self.steps: list[dict] = []
        self._lock = asyncio.Lock()

    async def record_step(self, step: int, actions: np.ndarray) -> None:
        """异步记录单步推理返回的原始动作。

        actions: 推理端返回的动作数组，形状通常为 (chunk_size, 16)。
        """
        async with self._lock:
            self.steps.append(
                {
                    "step": step,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                    # 转成嵌套 list 以便 JSON 序列化，完整保留推理端返回的动作块
                    "actions": np.asarray(actions).tolist(),
                }
            )

    async def save(self, success: bool | None = None) -> Path | None:
        """异步将记录写入 record 目录下的 .json 文件，返回文件路径。"""
        try:
            await asyncio.to_thread(self.record_dir.mkdir, parents=True, exist_ok=True)
            file_path = self.record_dir / f"{self.task_name}_{self.timestamp}.json"
            payload = {
                "task_name": self.task_name,
                "prompt": self.prompt,
                "arm_type": self.arm_type,
                "start_time": self.start_time.strftime("%Y-%m-%d %H:%M:%S"),
                "end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "success": success,
                "step_count": len(self.steps),
                "steps": self.steps,
            }

            def write_json():
                with file_path.open("w", encoding="utf-8") as handle:
                    json.dump(payload, handle, ensure_ascii=False, indent=2)

            await asyncio.to_thread(write_json)
            logger.info("动作记录已保存: %s (共 %d 步)", file_path, len(self.steps))
            return file_path
        except Exception as exc:  # 记录失败不应影响任务本身
            logger.error("保存动作记录失败: %s", exc)
            return None


@dataclass(frozen=True)
class TaskSpec:
    task_name: str
    prompt: str
    init_position: Any
    end_position: Any
    arm_type: str


class TaskConfigLoader:
    # 任务配置加载器，用于从 JSON 文件中加载任务配置，并提供任务解析功能
    def __init__(self, config_paths: Sequence[Path | str]):
        self.config_paths = [Path(path) for path in config_paths]
        self._configs = []
        for path in self.config_paths:
            with path.open("r", encoding="utf-8") as handle:
                self._configs.append((path, json.load(handle)))

    # 列出所有可用的任务名称
    def list_available_tasks(self) -> list[str]:
        tasks = []
        for _, config in self._configs:
            for task_name in config.keys():
                if task_name == "task":
                    continue
                if isinstance(config[task_name], dict):
                    tasks.append(task_name)
        return sorted(tasks)

    # 根据任务名称解析任务配置，返回 TaskSpec 对象
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

    # 根据任务配置文件路径和任务名称推断机械臂类型（左臂或右臂）
    @staticmethod
    def _infer_arm_type(path: Path, task_name: str) -> str:
        lower_name = f"{path.name} {task_name}".lower()
        if "left" in lower_name:
            return "left_arm"
        if "right" in lower_name:
            return "right_arm"
        raise ValueError(f"无法从任务 '{task_name}' 推断机械臂类型，请使用包含 left 或 right 的任务名")


@dataclass(frozen=True)
class ControlConfig:
    max_step: int = 150            # 最大执行步数
    try_num: int = 3               # 最大尝试次数
    arm_type: str = "left_arm"    # 机械臂类型
    fail_thehold: float = -0.01    # 夹取失败阈值 -0.05  -0.0008    -0.01
    success_thehold: float = -0.06  # 夹取成功阈值  -0.1   -0.001    -0.06
    ref_time: float = 1.0          # 轨迹参考时间（单位：秒），值越大动作执行越慢
    retry_ref_time: float = 0.2    # 重试时使用的轨迹参考时间（通常比正常推理更短以快速复位）

    ALLOW_ARM_TYPES = {"left_arm", "right_arm"}   # 目前没有双臂协作任务

    def __post_init__(self):
        if self.arm_type not in self.ALLOW_ARM_TYPES:
            raise ValueError(
                f"arm_type 取值非法！可选值：{self.ALLOW_ARM_TYPES}，当前传入：{self.arm_type}"
            )


class AsyncControl:
    def __init__(self, task_spec: TaskSpec):
        self.task_name = task_spec.task_name
        self.prompt = task_spec.prompt
        self.init_position = task_spec.init_position
        self.end_position = task_spec.end_position
        self.arm_type = task_spec.arm_type
        if self.task_name in {"right_arm_pick_tissue", "right_arm_pick_glasses"}:
            self.control_config = ControlConfig(
                arm_type=self.arm_type,
                fail_thehold=-0.0008,
                success_thehold=0.001,
            )
        else:
            self.control_config = ControlConfig(arm_type=self.arm_type)

        logger.info("加载任务 '%s'，使用 %s", self.task_name, self.arm_type)
        print("self.prompt:", self.prompt)

        # 动作记录器：记录每步从推理端获取的动作
        self.recorder = ActionRecorder(self.task_name, self.prompt, self.arm_type)

        self.web_policy = websocket_client_policy.WebsocketClientPolicy("127.0.0.1", 8181)
        self.g2_control = G2CorobotEnv()

        (
            self.left_joint_lower,
            self.left_joint_upper,
            self.right_joint_lower,
            self.right_joint_upper,
        ) = self.load_joint_limits(ROOT_DIR / "config" / "robot_config_ori.json")

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
                        f"{arm_side} joint {j + 1} 有 {n_out} 帧超限"
                        f"(范围 [{lower[j]:.4f}, {upper[j]:.4f}])，已裁剪到边界"
                    )
        return np.clip(arm_actions, lower, upper)

    async def get_state_async(self) -> dict:
        """异步获取机器人状态"""
        return await asyncio.to_thread(self.g2_control.get_state)

    async def execute_async(self, action: dict, ref_times: float) -> None:
        """异步执行机器人动作"""
        await asyncio.to_thread(self.g2_control.execute, action, ref_times)

    async def reset_pos_async(self, position: Any) -> None:
        """异步复位机器人位置"""
        await asyncio.to_thread(self.g2_control.reset_pos, position)

    async def infer_async(self, payload: dict) -> dict:
        """异步调用推理服务"""
        return await asyncio.to_thread(self.web_policy.infer, payload)

    # 推理主循环 - 完全异步版本
    async def inf_loop(self):
        try_num = 0
        for i in range(self.control_config.max_step):
            # 异步获取状态
            obs = await self.get_state_async()
            arm_states = obs["observation.states.arm_joint_states"][:14]
            gripper_states = obs["observation.states.gripper_states"][:2]

            state = list(arm_states[:7]) + [gripper_states[0]] + list(arm_states[7:14]) + [gripper_states[1]]

            # 异步解码图像
            def decode_images():
                return {
                    "cam_head": cv2.imdecode(np.frombuffer(obs["observation.images.head"]["image_data"], dtype=np.uint8), cv2.IMREAD_COLOR),
                    "cam_left": cv2.imdecode(np.frombuffer(obs["observation.images.hand_left"]["image_data"], dtype=np.uint8), cv2.IMREAD_COLOR),
                    "cam_right": cv2.imdecode(np.frombuffer(obs["observation.images.hand_right"]["image_data"], dtype=np.uint8), cv2.IMREAD_COLOR),
                }

            images = await asyncio.to_thread(decode_images)

            payload = {
                "state": state,
                "images": images,
                "prompt": self.prompt,
            }

            infer_start = time.time()
            # 异步调用推理
            response = await self.infer_async(payload)
            response = np.asarray(response["actions"])
            logger.info(f"第{i}步推理耗时: {(time.time() - infer_start) * 1000:.1f} ms")

            # 记录本步从推理端获取的原始动作
            if ENABLE_ACTION_RECORDING:
                await self.recorder.record_step(i, response)

            left_arm = response[:, :7]
            left_gripper = response[:, 7:8]
            right_arm = response[:, 8:15]
            right_gripper = response[:, 15:16]

            # 调试：当执行左臂拿奥美拉唑瓶子任务时，监控左夹爪值的变化
            if "omeprazole" in self.task_name.lower() and "bottle" in self.task_name.lower() and self.control_config.arm_type == "left_arm":
                left_gripper_values = left_gripper[:, 0]  # 获取所有帧的左夹爪值
                # 检查是否有任何一帧不是-0.785
                changed_indices = np.where(np.abs(left_gripper_values - (-0.785)) > 0.01)[0]
                if len(changed_indices) > 0:
                    # 找到第一个变化的帧
                    first_change_idx = changed_indices[0]
                    logger.info(f"⚠️ 第{i}步 - 左夹爪值开始变化: 第{first_change_idx}帧={left_gripper_values[first_change_idx]:.4f}, "
                               f"共{len(changed_indices)}帧变化, 末帧={left_gripper_values[-1]:.4f}")

            left_arm = self.clip_to_limits(left_arm, "left_arm")
            right_arm = self.clip_to_limits(right_arm, "right_arm")

            # 异步执行动作
            await self.execute_async(
                {
                    "left_arm": left_arm.tolist(),
                    "left_gripper": left_gripper.tolist(),
                    "right_arm": right_arm.tolist(),
                    "right_gripper": right_gripper.tolist(),
                },
                ref_times=self.control_config.ref_time,
            )

            if self.control_config.arm_type == "left_arm":
                response_gripper_mean = np.mean(left_gripper[-5:])
                reset_action = {
                    "left_gripper": [[-0.785]],
                }
            else:
                response_gripper_mean = np.mean(right_gripper[-5:])
                reset_action = {
                    "right_gripper": [[-0.785]],
                }

            if response_gripper_mean > -0.05:
                # 异步等待夹爪稳定
                await asyncio.sleep(1.5)
                obs = await self.get_state_async()
                current_gripper = obs["observation.states.gripper_states"][0] if self.control_config.arm_type == "left_arm" else obs["observation.states.gripper_states"][1]
                logger.info("当前夹爪值: %.3f, 指令期望闭合: %.3f", current_gripper, response_gripper_mean)
                if current_gripper > self.control_config.fail_thehold:
                    if try_num < self.control_config.try_num:
                        try_num += 1
                        logger.warning(
                            f"第{try_num}次夹取失败(夹爪值: {current_gripper:.3f}), 回到初始位置并打开夹爪重试"
                        )
                        await self.execute_async(reset_action, ref_times=self.control_config.retry_ref_time)
                        await asyncio.sleep(2)
                        continue
                    logger.error(f"尝试{self.control_config.try_num}次后仍未夹取成功，任务失败")
                    return False

                if current_gripper <= self.control_config.success_thehold:
                    logger.info(
                        f"夹取成功! 指令期望闭合: {response_gripper_mean:.3f}, 实际值(有物体阻挡): {current_gripper:.3f}"
                    )
                    return True

        return False

    async def run(self):
        await self.reset_pos_async(self.init_position)
        try:
            loop_result = await self.inf_loop()
        finally:
            # 无论成功、失败还是中途异常，都保存已采集到的动作记录
            if ENABLE_ACTION_RECORDING:
                await self.recorder.save(success=locals().get("loop_result"))
        if loop_result:
            await self.reset_pos_async(self.end_position)
        else:
            logger.error("任务失败")
        return loop_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行拿取模型测试任务（完全异步版本）")
    parser.add_argument(
        "--task",
        default="left_arm_drawer_pick_omeprazole_bottle",
        help=(
            "任务名称，可选：left_arm_drawer_pick_omeprazole_bottle、left_arm_drawer_pick_omeprazole_box、"
            "left_arm_drawer_pick_kangfuxin_bottle、left_arm_drawer_pick_kangfuxin_box"
        ),
    )
    return parser.parse_args()


async def main_async() -> int:
    args = parse_args()
    task_name = args.task

    loader = TaskConfigLoader([
        ROOT_DIR / "config" / "right_add_drawer_config.json",
        ROOT_DIR / "config" / "left_add_drawer_config.json",
    ])

    task_spec = loader.resolve_task(task_name)
    async_control = AsyncControl(task_spec)
    success = await async_control.run()
    return 0 if success else 1


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    exit_code = asyncio.run(main_async())
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
