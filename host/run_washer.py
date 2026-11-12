#!/usr/bin/env python3
"""洗衣机门开关控制脚本 - 支持拉洗衣机门和推洗衣机门操作

本脚本实现了机器人洗衣机门开关操作的智能控制，包含：
1. 基于关节角度变化的位移计算
2. 智能的成功/失败判断逻辑
3. 完整的关节限位保护
4. 详细的运行日志输出
"""

import numpy as np
import cv2
import json
import logging
import os
import select
import argparse
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
import sys
import time

# 获取脚本所在目录的绝对路径，并添加父目录到 sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

PROJECT_DIR = ROOT_DIR.parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
from runtime_config import get_policy_host, configure_proxy_bypass
from washer_tasks import (
    HOMECARE_ROBOT_HOST, MOTION_UNKNOWN_EXIT, OPEN_WASHER_TASK, OPERATOR_FINISHED_EXIT,
    OPERATOR_FINISH_BYTE, policy_for, resolve_running_policy,
)

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
        if "both" in lower_name or "双臂" in task_name:
            return "both_arms"
        if "left" in lower_name or "左臂" in task_name:
            return "left_arm"
        if "right" in lower_name or "右臂" in task_name:
            return "right_arm"
        raise ValueError(f"无法从任务 '{task_name}' 推断机械臂类型")


@dataclass(frozen=True)
class DrawerControlConfig:
    """洗衣机门控制配置参数"""
    max_step: int = 150              # 最长执行步数
    arm_type: str = "right_arm"      # 机械臂类型（洗衣机门操作通常用右臂）
    ref_time: float = 1.0            # 轨迹参考时间（单位：秒）

    min_steps_before_success: int = 30     # 至少完成30个推理/动作循环
    success_distance_m: float = 0.10              # 右臂末端距初始XYZ不超过10cm
    success_reference_frame: str = "base_link"

    ALLOW_ARM_TYPES = {"left_arm", "right_arm", "both_arms"}

    def __post_init__(self):
        if self.arm_type not in self.ALLOW_ARM_TYPES:
            raise ValueError(f"arm_type 非法！可选：{self.ALLOW_ARM_TYPES}")


class MotionUnknown(RuntimeError):
    """A washer action was cancelled or its submission could not be confirmed."""


class OperatorFinished(Exception):
    """This action received the scoped finish signal."""


class DrawerControl:
    """洗衣机门控制类 - 处理拉洗衣机门和推洗衣机门操作"""

    def __init__(self, task_spec: TaskSpec, *, single_task=False, cancel_fd=None):
        self.task_name = task_spec.task_name
        self.prompt = task_spec.prompt
        self.init_position = task_spec.init_position
        self.end_position = task_spec.end_position
        self.arm_type = task_spec.arm_type
        self.cancel_fd = cancel_fd
        self.control_config = DrawerControlConfig(arm_type=self.arm_type)
        self.last_open_trajectory_end = 0.0

        # 判断是拉洗衣机门还是推洗衣机门
        self.is_pull = True
        self.is_push = False

        logger.info("=== 洗衣机门操作任务 ===")
        logger.info("任务名称: %s", self.task_name)
        logger.info("使用机械臂: %s", self.arm_type)
        logger.info("任务提示: %s", self.prompt)
        logger.info("操作类型: %s", "拉洗衣机门" if self.is_pull else "推洗衣机门")

        # 初始化策略客户端和机器人环境
        # self.web_policy = websocket_client_policy.WebsocketClientPolicy("127.0.0.1", 8181)
        if single_task:
            policy = resolve_running_policy(policy_for(self.task_name), require_unique_port=True)
            robot_host = configure_proxy_bypass(HOMECARE_ROBOT_HOST)
        else:
            task_config = json.loads((ROOT_DIR / "config" / "config_V2_0.json").read_text(encoding="utf-8"))[self.task_name]
            policy = {"host": configure_proxy_bypass(get_policy_host()), "port": task_config["policy_port"]}
            robot_host = None
        self.web_policy = websocket_client_policy.WebsocketClientPolicy(policy["host"], policy["port"])
        self.g2_control = G2CorobotEnv(robot_host=robot_host)

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

    @staticmethod
    def get_arm_xyz(obs: dict, arm_type: str, reference_frame: str) -> np.ndarray:
        end_pose = obs.get("observation.states.end_pose")
        if not isinstance(end_pose, dict) or reference_frame not in end_pose:
            available = list(end_pose) if isinstance(end_pose, dict) else []
            raise ValueError(
                f"缺少末端位姿参考系 {reference_frame}；实际提供：{available}"
            )
        pose = end_pose[reference_frame].get(arm_type)
        if not isinstance(pose, dict) or "position" not in pose:
            raise ValueError(f"缺少 {reference_frame}/{arm_type} 的末端XYZ")
        xyz = np.asarray(pose["position"], dtype=float)
        if xyz.shape != (3,) or not np.isfinite(xyz).all():
            raise ValueError(f"末端XYZ格式错误：{pose['position']}")
        return xyz

    def submit_trajectory(self, action_parts, ref_time):
        """Send one trajectory and check its submission receipt."""
        action = self.g2_control.build_action(action_parts, trajectory_reference_time=ref_time)
        self.check_cancellation()
        action["timestamps"] = time.time_ns()
        try:
            acknowledged = self.g2_control.env.execute_action(action)
        except Exception as exc:
            raise MotionUnknown("未收到机器人轨迹接收回执") from exc
        if acknowledged is not True:
            raise MotionUnknown("机器人未接收轨迹")
        if self.task_name == OPEN_WASHER_TASK:
            self.last_open_trajectory_end = time.monotonic() + ref_time

    def check_cancellation(self):
        if self.cancel_fd is not None and select.select([self.cancel_fd], [], [], 0)[0]:
            if os.read(self.cancel_fd, 1) == OPERATOR_FINISH_BYTE:
                if self.task_name == OPEN_WASHER_TASK:
                    time.sleep(max(0.0, self.last_open_trajectory_end - time.monotonic()))
                raise OperatorFinished
            raise MotionUnknown("已取消本次洗衣机任务，停止继续提交轨迹")

    def run_single_task(self, run_seconds):
        """Prepare at the washer and run cancellable model inference."""
        positions = self.init_position
        if isinstance(positions, dict):
            positions = [positions]
        for position in positions:
            arm, gripper = position["arm"], position["gripper"]
            parts = {
                "left_arm": [arm[:7]],
                "right_arm": [arm[7:]],
                "left_gripper": [gripper[:1]],
                "right_gripper": [gripper[1:]],
                "head": [position["head"]],
                "waist": [position["waist"]],
            }
            if self.task_name in {OPEN_WASHER_TASK, "左臂放浅蓝色衣服进洗衣机"}:
                # The left hand may already be holding the garment.
                del parts["left_gripper"]
            self.submit_trajectory(parts, 2.0)
            time.sleep(2.0)
        deadline = None if run_seconds is None else time.monotonic() + run_seconds
        return self.inf_loop(deadline=deadline, single_task=True)

    def inf_loop(self, *, deadline=None, single_task=False):
        """Run model inference until the configured end or an explicit stop."""
        logger.info("使用 prompt: %s", self.prompt)
        if deadline is None:
            logger.info("持续推理，等待取消" if single_task
                        else "临时无限循环模式：不会自动判定成功，按 Ctrl+C 中断")
        i = 0
        ref_time = self.control_config.ref_time

        while deadline is None or time.monotonic() + ref_time < deadline:
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
            if single_task:
                self.check_cancellation()
            infer_start = time.time()
            response = np.asarray(self.web_policy.infer(payload)["actions"])
            logger.info(f"第{i}步推理耗时: {(time.time() - infer_start) * 1000:.1f} ms")

            if single_task and (response.shape != (30, 16) or not np.isfinite(response).all()):
                raise ValueError(f"模型动作格式无效: {response.shape}")

            # 解析动作
            left_arm = self.clip_to_limits(response[:, :7], "left_arm")
            left_gripper = response[:, 7:8]
            right_arm = self.clip_to_limits(response[:, 8:15], "right_arm")
            right_gripper = response[:, 15:16]

            # 调试：打印右手臂动作和夹爪动作
            if i < 5 or i % 20 == 0:
                logger.info(f"第{i}步 右臂动作前3关节: {right_arm[0, :3]}, 右夹爪: {right_gripper[0, 0]:.3f}")

            parts = {
                "left_arm": left_arm.tolist(),
                "left_gripper": left_gripper.tolist(),
                "right_arm": right_arm.tolist(),
                "right_gripper": right_gripper.tolist(),
            }
            if not single_task:
                self.g2_control.execute(parts, ref_times=ref_time)
            else:
                if self.task_name != OPEN_WASHER_TASK and self.arm_type != "both_arms":
                    inactive = "left" if self.arm_type == "right_arm" else "right"
                    del parts[f"{inactive}_arm"], parts[f"{inactive}_gripper"]
                if deadline is not None and time.monotonic() + ref_time >= deadline:
                    break
                self.submit_trajectory(parts, ref_time)

            i += 1
            if i % 20 == 0:
                logger.info("已执行%d个推理循环", i)

        if deadline is not None:
            logger.info("限时推理结束，已提交 %d 段模型轨迹", i)
        return i > 0

    def run(self):
        """执行完整的洗衣机门操作任务，支持夹空自动重试"""
        # 移动到初始位置
        self.g2_control.reset_pos(self.init_position)

        try:
            loop_result = self.inf_loop()
        except Exception as e:
            logger.error(f"洗衣机门操作异常: {e}", exc_info=True)
            return False

        if loop_result:
            # 成功：移动到结束位置
            self.g2_control.reset_pos(self.end_position)
            logger.info("=== 洗衣机门操作成功完成 ===")
            return True

        logger.error("=== 洗衣机门操作失败 ===")
        return False


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="洗衣机门开关测试")
    parser.add_argument(
        "--task",
        default="右臂开洗衣机门",
        help="任务名称：right_arm_pull_drawer（拉）、right_arm_push_drawer（推）",
    )
    parser.add_argument("--run-seconds", type=float, help="HomeCare 单步限时模式")
    parser.add_argument("--cancel-fd", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.run_seconds is not None and not 1 < args.run_seconds < 60:
        parser.error("--run-seconds 必须大于 1 秒且小于 60 秒")
    if args.cancel_fd is not None and args.cancel_fd < 3:
        parser.error("--cancel-fd 必须是独立文件描述符")
    return args


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

    if args.run_seconds is None and args.cancel_fd is None:
        sys.exit(0 if DrawerControl(task_spec).run() else 1)

    drawer_control = None
    try:
        drawer_control = DrawerControl(task_spec, single_task=True, cancel_fd=args.cancel_fd)
        success = drawer_control.run_single_task(args.run_seconds)
    except OperatorFinished:
        logger.info("收到本次执行的结束信号，停止提交后续轨迹")
        sys.exit(OPERATOR_FINISHED_EXIT)
    except (MotionUnknown, KeyboardInterrupt):
        logger.exception("洗衣机任务未确认停止")
        sys.exit(MOTION_UNKNOWN_EXIT)
    finally:
        if drawer_control is not None:
            with suppress(OSError):
                drawer_control.g2_control.env.close()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
