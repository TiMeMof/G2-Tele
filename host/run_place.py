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
class ContrlConfig:
    max_step: int = 100              # 最长步数
    arm_type: str = "left_arm"      # 机械臂类型
    # 轨迹参考时间（单位：秒），值越大动作执行越慢
    ref_time: float = 1.0

    # 定义允许的选项
    ALLOW_ARM_TYPES = {"left_arm", "right_arm"}

    def __post_init__(self):
        if self.arm_type not in self.ALLOW_ARM_TYPES:
            raise ValueError(
                f"arm_type 取值非法！可选值：{self.ALLOW_ARM_TYPES}，当前传入：{self.arm_type}"
            )


class SyncControlPlace():
    def __init__(self, task_spec: TaskSpec):
        self.task_name = task_spec.task_name
        self.prompt = task_spec.prompt
        self.init_positon = task_spec.init_position
        self.end_position = task_spec.end_position
        self.arm_type = task_spec.arm_type
        self.contrl_config = ContrlConfig(arm_type=self.arm_type)

        logger.info("加载任务 '%s'，使用 %s", self.task_name, self.arm_type)

        # self.web_policy = websocket_client_policy.WebsocketClientPolicy("127.0.0.1", 8181)
        self.web_policy = websocket_client_policy.WebsocketClientPolicy("192.168.50.114", 8181)

        self.g2_control = G2CorobotEnv()

        # 加载左右臂关节限位，用于裁剪推理结果防止超限
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
        left_lower = np.array(left_limits["lower"])  # (7,)
        left_upper = np.array(left_limits["upper"])  # (7,)
        right_lower = np.array(right_limits["lower"])  # (7,)
        right_upper = np.array(right_limits["upper"])  # (7,)
        return left_lower, left_upper, right_lower, right_upper

    def clip_to_limits(self, arm_actions: np.ndarray, arm_side: str) -> np.ndarray:
        """将机械臂动作裁剪到对应臂的关节限位范围内。"""
        if arm_side == "left_arm":
            lower = self.left_joint_lower
            upper = self.left_joint_upper
        else:
            lower = self.right_joint_lower
            upper = self.right_joint_upper
        # 找出超限的位置并打印日志
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


    def inf_loop(self):
        for i in range(self.contrl_config.max_step):
            obs = self.g2_control.get_state()
            # 按顺序构造 state：左臂 7 自由度 + 左夹爪 + 右臂 7 自由度 + 右夹爪
            arm_states = obs["observation.states.arm_joint_states"][:14]  # 前7 左臂，后7 右臂
            gripper_states = obs["observation.states.gripper_states"][:2]  # 左夹爪，右夹爪

            state = list(arm_states[:7]) + [gripper_states[0]] + list(arm_states[7:14]) + [gripper_states[1]]
            payload = {
                "state": state,
                "images": {
                    "cam_head": cv2.imdecode(np.frombuffer(obs["observation.images.head"]["image_data"], dtype=np.uint8), cv2.IMREAD_COLOR),
                    "cam_left": cv2.imdecode(np.frombuffer(obs["observation.images.hand_left"]["image_data"], dtype=np.uint8), cv2.IMREAD_COLOR),
                    "cam_right": cv2.imdecode(np.frombuffer(obs["observation.images.hand_right"]["image_data"], dtype=np.uint8), cv2.IMREAD_COLOR),
                },
                "prompt": self.prompt,
            }
            # response: (chunk, 16) -> 左臂7 + 左爪1 + 右臂7 + 右爪1
            infer_start = time.time()
            response = np.asarray(self.web_policy.infer(payload)["actions"])
            logger.info(f"第{i}步推理耗时: {(time.time() - infer_start) * 1000:.1f} ms")

            left_arm = response[:, :7]        # (chunk, 7)
            left_gripper = response[:, 7:8]   # (chunk, 1)
            right_arm = response[:, 8:15]     # (chunk, 7)
            right_gripper = response[:, 15:16] # (chunk, 1)

            # 裁剪到左右臂关节限位，防止下发超限值
            left_arm = self.clip_to_limits(left_arm, "left_arm")
            right_arm = self.clip_to_limits(right_arm, "right_arm")

            # 下发动作，使用配置的轨迹时间来控制速度
            self.g2_control.execute(
                {
                    "left_arm": left_arm.tolist(),
                    "left_gripper": left_gripper.tolist(),
                    "right_arm": right_arm.tolist(),
                    "right_gripper": right_gripper.tolist(),
                },
                ref_times=self.contrl_config.ref_time,
            )

            # 获取当前夹爪状态
            if self.contrl_config.arm_type == "left_arm":
                response_gripper_mean = np.mean(left_gripper[-5:])
                obs = self.g2_control.get_state()
                current_gripper = obs["observation.states.gripper_states"][0]
            else:
                response_gripper_mean = np.mean(right_gripper[-5:])
                obs = self.g2_control.get_state()
                current_gripper = obs["observation.states.gripper_states"][1]

            # 记录夹爪状态
            logger.info(
                f"第{i}步 - 指令夹爪值: {response_gripper_mean:.3f}, 实际夹爪值: {current_gripper:.3f}"
            )

            # 停止条件：当夹爪打开到 -0.7 以下时，认为物体已放下
            if current_gripper < -0.7:
                logger.info(f"检测到夹爪已充分打开 ({current_gripper:.3f} < -0.7)，放下完成")
                return True

        logger.info("推理循环完成")
        return True


    def run(self):
        # 开始任务前机器人需要移动到初始化位置
        self.g2_control.reset_pos(self.init_positon)
        # 开持续推理
        loop_result = self.inf_loop()
        # 推理完成后，直接运动到end_position
        self.g2_control.reset_pos(self.end_position)
        if loop_result:
            logger.info("任务完成")
        else:
            logger.error("任务失败")
        return loop_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行放置模型测试任务")
    parser.add_argument(
        "--task",
        default="right_arm_place_glasses",
        help="任务名称，可选：left_arm_place_omeprazole_box、right_arm_place_water/right_arm_place_glasses",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # 初始化日志，确保各模块的 logger 能输出到终端
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = parse_args()
    task_name = args.task

    loader = TaskConfigLoader([
        ROOT_DIR / "config" / "config_V2_0.json",
        # ROOT_DIR / "config" / "left_add_drawer_config.json",
        # ROOT_DIR / "config" / "right_add_drawer_config.json",
    ])

    task_spec = loader.resolve_task(task_name)
    sync_control = SyncControlPlace(task_spec)
    success = sync_control.run()
    sys.exit(0 if success else 1)
