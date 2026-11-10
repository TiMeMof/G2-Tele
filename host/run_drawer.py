#!/usr/bin/env python3
"""抽屉开关控制脚本 - 支持拉抽屉和推抽屉操作

本脚本实现了机器人抽屉开关操作的智能控制，包含：
1. 基于关节角度变化的位移计算
2. 智能的成功/失败判断逻辑
3. 完整的关节限位保护
4. 详细的运行日志输出
"""

import numpy as np
import cv2
import json
import logging
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
import sys
import time

# 获取脚本所在目录的绝对路径，并添加父目录到 sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from openpi_client import websocket_client_policy
from core.corobot_env.g2_corobot_env import G2CorobotEnv

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TaskSpec:
    """任务规格定义"""
    task_name: str
    prompt: str
    init_position: Any
    end_position: Any
    arm_type: str


class TaskConfigLoader:
    """任务配置加载器"""

    def __init__(self, config_paths: Sequence[Path | str]):
        self.config_paths = [Path(path) for path in config_paths]
        self._configs = []
        for path in self.config_paths:
            with path.open("r", encoding="utf-8") as handle:
                self._configs.append((path, json.load(handle)))

    def list_available_tasks(self) -> list[str]:
        """列出所有可用的任务名称"""
        tasks = []
        for _, config in self._configs:
            for task_name in config.keys():
                if task_name == "task":
                    continue
                if isinstance(config[task_name], dict):
                    tasks.append(task_name)
        return sorted(tasks)

    def resolve_task(self, task_name: str) -> TaskSpec:
        """根据任务名称解析任务配置"""
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
        """根据任务配置文件路径和任务名称推断机械臂类型"""
        lower_name = f"{path.name} {task_name}".lower()
        if "left" in lower_name:
            return "left_arm"
        if "right" in lower_name:
            return "right_arm"
        raise ValueError(f"无法从任务 '{task_name}' 推断机械臂类型")


@dataclass(frozen=True)
class DrawerControlConfig:
    """抽屉控制配置参数"""
    max_step: int = 150              # 最长执行步数
    arm_type: str = "right_arm"      # 机械臂类型（抽屉操作通常用右臂）
    ref_time: float = 1.0            # 轨迹参考时间（单位：秒）

    # 夹爪状态判断阈值
    gripper_open_threshold: float = -0.6         # 夹爪完全张开的阈值（接近-0.785）
    push_gripper_open_threshold: float = -0.4    # 关抽屉二次张开阈值（无需完全张开，张开到一定程度即可）
    gripper_closed_threshold: float = -0.3       # 夹爪闭合的阈值
    min_steps_before_success: int = 10           # 判定成功前的最小执行步数

    # 夹空检测参数
    empty_grasp_consecutive_steps: int = 10       # 连续多少步夹爪为 0 判定为夹空
    max_retries: int = 3                          # 夹空最大重试次数

    ALLOW_ARM_TYPES = {"left_arm", "right_arm"}

    def __post_init__(self):
        if self.arm_type not in self.ALLOW_ARM_TYPES:
            raise ValueError(f"arm_type 非法！可选：{self.ALLOW_ARM_TYPES}")


class DrawerControl:
    """抽屉控制类 - 处理拉抽屉和推抽屉操作"""

    def __init__(self, task_spec: TaskSpec):
        self.task_name = task_spec.task_name
        self.prompt = task_spec.prompt
        self.init_position = task_spec.init_position
        self.end_position = task_spec.end_position
        self.arm_type = task_spec.arm_type
        self.control_config = DrawerControlConfig(arm_type=self.arm_type)

        # 判断是拉抽屉还是推抽屉
        self.is_pull = "pull" in self.task_name.lower() or "拉" in self.prompt
        self.is_push = "push" in self.task_name.lower() or "关" in self.prompt or "推" in self.prompt

        logger.info("=== 抽屉操作任务 ===")
        logger.info("任务名称: %s", self.task_name)
        logger.info("使用机械臂: %s", self.arm_type)
        logger.info("任务提示: %s", self.prompt)
        logger.info("操作类型: %s", "拉抽屉" if self.is_pull else "推抽屉")

        # 初始化策略客户端和机器人环境
        # self.web_policy = websocket_client_policy.WebsocketClientPolicy("127.0.0.1", 8181)
        self.web_policy = websocket_client_policy.WebsocketClientPolicy("192.168.50.114", 8181)
        self.g2_control = G2CorobotEnv()

        # 加载关节限位
        (
            self.left_joint_lower,
            self.left_joint_upper,
            self.right_joint_lower,
            self.right_joint_upper,
        ) = self.load_joint_limits(ROOT_DIR / "config" / "robot_config_ori.json")

        # 记录初始位置用于计算位移
        self.initial_arm_position = None

    def load_joint_limits(self, file_path: Path):
        """加载左右臂关节限位"""
        with file_path.open("r", encoding="utf-8") as handle:
            robot_config = json.load(handle)
        left_limits = robot_config["left_arm"]
        right_limits = robot_config["right_arm"]
        return (
            np.array(left_limits["lower"]),
            np.array(left_limits["upper"]),
            np.array(right_limits["lower"]),
            np.array(right_limits["upper"]),
        )

    def clip_to_limits(self, arm_actions: np.ndarray, arm_side: str) -> np.ndarray:
        """将机械臂动作裁剪到关节限位范围内"""
        if arm_side == "left_arm":
            lower, upper = self.left_joint_lower, self.left_joint_upper
        else:
            lower, upper = self.right_joint_lower, self.right_joint_upper

        below = arm_actions < lower
        above = arm_actions > upper
        if below.any() or above.any():
            for j in range(arm_actions.shape[1]):
                if below[:, j].any() or above[:, j].any():
                    n_out = int(below[:, j].sum() + above[:, j].sum())
                    logger.warning(
                        f"{arm_side} 关节{j+1} 有{n_out}帧超限，已裁剪"
                    )
        return np.clip(arm_actions, lower, upper)

    def calculate_arm_displacement(self, current_joints: np.ndarray, initial_joints: np.ndarray) -> float:
        """
        计算机械臂末端位移（简化计算）

        实际应用中应使用正运动学计算末端位置
        这里使用关节角度变化的加权和作为近似
        前三个关节对末端位置影响较大，给予更高权重
        """
        weights = np.array([0.3, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02])
        current = np.asarray(current_joints, dtype=float)
        initial = np.asarray(initial_joints, dtype=float)
        joint_diff = np.abs(current - initial)
        return float(np.sum(joint_diff * weights))

    def inf_loop(self):
        """推理循环 - 执行抽屉开关操作"""
        # 记录初始手臂位置
        initial_obs = self.g2_control.get_state()
        arm_states = initial_obs["observation.states.arm_joint_states"][:14]

        if self.control_config.arm_type == "left_arm":
            self.initial_arm_position = arm_states[:7].copy()
            initial_gripper = initial_obs["observation.states.gripper_states"][0]
        else:
            self.initial_arm_position = arm_states[7:14].copy()
            initial_gripper = initial_obs["observation.states.gripper_states"][1]

        logger.info("初始手臂关节: %s", self.initial_arm_position[:3])
        logger.info("初始夹爪状态: %.3f", initial_gripper)
        logger.info("使用 prompt: %s", self.prompt)

        # 夹爪状态追踪
        gripper_history = []  # 记录夹爪状态历史
        has_closed = False    # 是否已经闭合过（对于拉抽屉：抓住拉环）

        for i in range(self.control_config.max_step):
            # 获取当前状态
            obs = self.g2_control.get_state()
            arm_states = obs["observation.states.arm_joint_states"][:14]
            gripper_states = obs["observation.states.gripper_states"][:2]

            # 构造状态向量
            state = list(arm_states[:7]) + [gripper_states[0]] + list(arm_states[7:14]) + [gripper_states[1]]

            # 准备推理输入
            payload = {
                "state": state,
                "images": {
                    "cam_head": cv2.imdecode(np.frombuffer(obs["observation.images.head"]["image_data"], dtype=np.uint8), cv2.IMREAD_COLOR),
                    "cam_left": cv2.imdecode(np.frombuffer(obs["observation.images.hand_left"]["image_data"], dtype=np.uint8), cv2.IMREAD_COLOR),
                    "cam_right": cv2.imdecode(np.frombuffer(obs["observation.images.hand_right"]["image_data"], dtype=np.uint8), cv2.IMREAD_COLOR),
                },
                "prompt": self.prompt,
            }

            # 调试：在第0步打印 payload 的 prompt 和 state 信息
            if i == 0:
                logger.info(f"Payload prompt: {payload['prompt']}")
                logger.info(f"State 维度: {len(state)}, 右臂状态: {state[8:15]}, 右夹爪: {state[15]}")

            # 调用推理
            infer_start = time.time()
            response = np.asarray(self.web_policy.infer(payload)["actions"])
            logger.info(f"第{i}步推理耗时: {(time.time() - infer_start) * 1000:.1f} ms")

            # 解析动作
            left_arm = self.clip_to_limits(response[:, :7], "left_arm")
            left_gripper = response[:, 7:8]
            right_arm = self.clip_to_limits(response[:, 8:15], "right_arm")
            right_gripper = response[:, 15:16]

            # 调试：打印右手臂动作和夹爪动作
            if i < 5 or i % 20 == 0:
                logger.info(f"第{i}步 右臂动作前3关节: {right_arm[0, :3]}, 右夹爪: {right_gripper[0, 0]:.3f}")

            # 执行动作
            self.g2_control.execute(
                {
                    "left_arm": left_arm.tolist(),
                    "left_gripper": left_gripper.tolist(),
                    "right_arm": right_arm.tolist(),
                    "right_gripper": right_gripper.tolist(),
                },
                ref_times=self.control_config.ref_time,
            )

            # 获取执行后的夹爪位置
            obs = self.g2_control.get_state()
            arm_states = obs["observation.states.arm_joint_states"][:14]

            if self.control_config.arm_type == "left_arm":
                current_arm_position = arm_states[:7]
                current_gripper = obs["observation.states.gripper_states"][0]
            else:
                current_arm_position = arm_states[7:14]
                current_gripper = obs["observation.states.gripper_states"][1]

            # 记录夹爪历史
            gripper_history.append(current_gripper)

            # 计算位移（用于日志）
            displacement = self.calculate_arm_displacement(
                current_arm_position,
                self.initial_arm_position
            )

            logger.info(f"第{i}步 - 位移: {displacement:.4f}, 夹爪: {current_gripper:.3f}")

            # 判断夹爪状态变化
            if i >= self.control_config.min_steps_before_success:
                # 拉抽屉判断逻辑：
                # 1. 初始张开 -> 2. 闭合抓住拉环 -> 3. 再次张开（松开拉环，抽屉已拉开）
                if self.is_pull:
                    # 检查是否已经闭合过
                    if not has_closed and current_gripper > self.control_config.gripper_closed_threshold:
                        has_closed = True
                        logger.info(f"第{i}步 - 夹爪已闭合（抓住拉环）: {current_gripper:.3f}")

                    # 如果已经闭合过，现在再次张开，判定成功
                    if has_closed and current_gripper < self.control_config.gripper_open_threshold:
                        logger.info(f"✓ 拉抽屉成功！夹爪二次张开: {current_gripper:.3f}")
                        time.sleep(1.0)
                        return True

                # 推抽屉判断逻辑：
                # 1. 初始张开 -> 2. 闭合推抽屉 -> 3. 再次张开（推完松开）
                elif self.is_push:
                    # 检查是否已经闭合过
                    if not has_closed and current_gripper > self.control_config.gripper_closed_threshold:
                        has_closed = True
                        logger.info(f"第{i}步 - 夹爪已闭合（准备推）: {current_gripper:.3f}")

                    # 如果已经闭合过，现在再次张开到一定程度即可判定成功（无需完全张开到-0.785）
                    if has_closed and current_gripper < self.control_config.push_gripper_open_threshold:
                        logger.info(f"✓ 推抽屉成功！夹爪二次张开: {current_gripper:.3f}")
                        time.sleep(1.0)
                        return True

                # 夹空检测：闭合后若夹爪值连续多步保持在 0 附近，说明夹空，返回 "retry"
                if has_closed:
                    n = self.control_config.empty_grasp_consecutive_steps
                    recent = gripper_history[-n:]
                    if (len(recent) >= n
                            and all(v == 0 for v in recent)):
                        logger.warning(
                            f"第{i}步 - 检测到夹空！夹爪值连续 {n} 步近似为 0，"
                            f"最近夹爪值: {recent[-1]:.3f}"
                        )
                        return "retry"

            # 每20步输出进度
            if (i + 1) % 20 == 0:
                logger.info(f"进度: {i+1}/{self.control_config.max_step}, 夹爪状态: {current_gripper:.3f}, has_closed: {has_closed}")

        # 达到最大步数，检查最终状态
        logger.error(f"✗ 抽屉操作失败，未检测到夹爪二次张开")
        logger.error(f"   has_closed: {has_closed}, 最终夹爪: {gripper_history[-1]:.3f}")
        return False

    def run(self):
        """执行完整的抽屉操作任务，支持夹空自动重试"""
        # 移动到初始位置
        self.g2_control.reset_pos(self.init_position)

        for attempt in range(self.control_config.max_retries + 1):
            if attempt > 0:
                logger.info(
                    f"=== 夹空重试 第{attempt}次 / 共{self.control_config.max_retries}次 ==="
                )
                # 返回初始位置再次尝试
                self.g2_control.reset_pos(self.init_position)

            try:
                loop_result = self.inf_loop()
            except Exception as e:
                logger.error(f"抽屉操作异常: {e}", exc_info=True)
                return False

            if loop_result == "retry":
                if attempt < self.control_config.max_retries:
                    logger.warning(
                        f"检测到夹空，准备返回初始位置重试"
                        f"（第{attempt + 1}次重试）..."
                    )
                    continue
                else:
                    logger.error(
                        f"已达到最大重试次数（{self.control_config.max_retries}次），任务失败"
                    )
                    return False
            elif loop_result:
                # 成功：移动到结束位置
                self.g2_control.reset_pos(self.end_position)
                logger.info("=== 抽屉操作成功完成 ===")
                return True
            else:
                logger.error("=== 抽屉操作失败 ===")
                return False

        return False


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="抽屉开关测试")
    parser.add_argument(
        "--task",
        default="right_arm_push_drawer",
        help="任务名称：right_arm_pull_drawer（拉）、right_arm_push_drawer（推）",
    )
    return parser.parse_args()


def main() -> None:
    """主函数"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = parse_args()
    task_name = args.task

    # 加载任务配置
    loader = TaskConfigLoader([
        ROOT_DIR / "config" / "config_V2_0.json",
        # ROOT_DIR / "config" / "right_add_drawer_config.json",
        # ROOT_DIR / "config" / "left_add_drawer_config.json",
    ])

    try:
        task_spec = loader.resolve_task(task_name)
    except KeyError as e:
        logger.error(str(e))
        sys.exit(1)

    # 执行任务
    drawer_control = DrawerControl(task_spec)
    success = drawer_control.run()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
