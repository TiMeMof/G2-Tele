#!/usr/bin/env python3
"""Run hot- and cold-water tasks with feedback-based completion checks."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

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
    """Load tasks with the same config structure used by the existing scripts."""

    def __init__(self, config_paths: Sequence[Path | str]):
        self._configs: list[tuple[Path, dict[str, Any]]] = []
        for raw_path in config_paths:
            path = Path(raw_path)
            with path.open("r", encoding="utf-8") as handle:
                self._configs.append((path, json.load(handle)))

    def resolve_task(self, task_name: str) -> TaskSpec:
        for path, config in self._configs:
            task = config.get(task_name)
            if not isinstance(task, dict):
                continue
            required = {"prompt", "init_position", "end_position"}
            if not required.issubset(task):
                continue
            name = f"{path.name} {task_name}".lower()
            if "left" in name or "左臂" in name:
                arm_type = "left_arm"
            elif "right" in name or "右臂" in name:
                arm_type = "right_arm"
            else:
                raise ValueError(f"Cannot infer arm type for task: {task_name}")
            return TaskSpec(
                task_name=task_name,
                prompt=task["prompt"],
                init_position=task["init_position"],
                end_position=task["end_position"],
                arm_type=arm_type,
            )
        raise KeyError(f"Task not found or incomplete: {task_name}")


@dataclass(frozen=True)
class WaterControlConfig:
    max_step: int = 150
    ref_time: float = 1.0

    # Gripper convention: -0.785 is fully open and 0 is fully closed.
    open_threshold: float = -0.70
    closed_threshold: float = -0.05
    grasp_min: float = -0.50
    grasp_max: float = -0.02
    debounce_frames: int = 3
    cold_open_delta: float = 0.20
    cold_open_limit: float = -0.70
    final_grasp_frames: int = 5
    hot_button_hold_seconds: float = 4.0

    # Cold-water joint-range settings.
    approach_min_distance: float = 0.15
    press_enter_tolerance: float = 0.18
    press_exit_tolerance: float = 0.25
    press_enter_frames: int = 2
    jitter_grace_seconds: float = 0.80
    cold_hold_seconds: float = 3.0
    withdraw_min_distance: float = 0.08
    withdraw_progress: float = 0.04
    withdraw_frames: int = 2


def get_arm_joints(obs: dict[str, Any], arm_type: str) -> np.ndarray:
    joints = np.asarray(
        obs["observation.states.arm_joint_states"][:14], dtype=float
    )
    return joints[:7].copy() if arm_type == "left_arm" else joints[7:14].copy()


def get_gripper(obs: dict[str, Any], arm_type: str) -> float:
    values = np.asarray(
        obs["observation.states.gripper_states"][:2], dtype=float
    )
    return float(values[0] if arm_type == "left_arm" else values[1])


class PaddleWaterJudge:
    """Judge a cup-held black-paddle press followed by withdrawal behavior."""

    def __init__(
        self,
        config: WaterControlConfig,
        initial_joints: np.ndarray,
        initial_gripper: float,
    ):
        self.config = config
        self.initial_joints = initial_joints.copy()
        self.initial_gripper = initial_gripper
        self.stage = "wait_approach"
        self.stable_samples: deque[tuple[float, np.ndarray]] = deque(
            maxlen=config.press_enter_frames
        )
        self.press_joints: np.ndarray | None = None
        self.press_start: float | None = None
        self.outside_since: float | None = None
        self.withdraw_count = 0
        self.open_count = 0
        self.failed = False

    def update(self, joints: np.ndarray, gripper: float, now: float) -> bool:
        c = self.config
        open_failure_threshold = max(
            c.cold_open_limit,
            self.initial_gripper - c.cold_open_delta,
        )
        if gripper <= open_failure_threshold:
            self.open_count += 1
            if self.open_count >= c.debounce_frames:
                self.failed = True
                logger.error(
                    "Water failure: left gripper opened "
                    "(actual %.3f <= threshold %.3f)",
                    gripper,
                    open_failure_threshold,
                )
                return False
        else:
            self.open_count = 0

        initial_distance = float(np.linalg.norm(joints - self.initial_joints))
        if self.stage == "wait_approach":
            if initial_distance < c.approach_min_distance:
                self.stable_samples.clear()
                return False
            self.stage = "wait_stable_press"
            logger.info(
                "Water: arm left initial range (joint distance %.3f)",
                initial_distance,
            )

        if self.stage == "wait_stable_press":
            self.stable_samples.append((now, joints.copy()))
            if len(self.stable_samples) < c.press_enter_frames:
                return False
            samples = np.stack([sample for _, sample in self.stable_samples])
            if not np.all(np.ptp(samples, axis=0) <= c.press_enter_tolerance):
                return False
            self.press_joints = np.median(samples, axis=0)
            self.press_start = self.stable_samples[0][0]
            self.outside_since = None
            self.stage = "hold_press"
            logger.info("Water: black paddle press range detected; timer started")
            return False

        if self.stage == "hold_press":
            assert self.press_joints is not None and self.press_start is not None
            inside = bool(
                np.all(np.abs(joints - self.press_joints) <= c.press_exit_tolerance)
            )
            if inside:
                self.outside_since = None
            elif self.outside_since is None:
                self.outside_since = now
            elif now - self.outside_since > c.jitter_grace_seconds:
                logger.info("Water: press range interrupted; detecting again")
                self.stage = "wait_stable_press"
                self.stable_samples.clear()
                self.press_joints = None
                self.press_start = None
                self.outside_since = None
                return False
            if now - self.press_start >= c.cold_hold_seconds:
                self.stage = "wait_withdraw"
                logger.info(
                    "Water: black paddle held for %.2f seconds",
                    now - self.press_start,
                )
            return False

        assert self.press_joints is not None
        press_to_initial = self.initial_joints - self.press_joints
        direction_norm = float(np.linalg.norm(press_to_initial))
        if direction_norm < 1e-6:
            return False
        withdrawal_direction = press_to_initial / direction_norm
        movement = joints - self.press_joints
        moved_from_press = float(np.linalg.norm(movement))
        moved_toward_initial = float(np.dot(movement, withdrawal_direction))
        withdrawn = (
            moved_from_press >= c.withdraw_min_distance
            and moved_toward_initial >= c.withdraw_progress
        )
        self.withdraw_count = self.withdraw_count + 1 if withdrawn else 0
        if self.withdraw_count >= c.withdraw_frames:
            logger.info(
                "Water: withdrawal behavior detected for %d frames",
                self.withdraw_count,
            )
            return True
        return False

class WaterControl:
    def __init__(self, task_spec: TaskSpec, config: WaterControlConfig):
        self.task_name = task_spec.task_name
        self.prompt = task_spec.prompt
        self.init_position = task_spec.init_position
        self.end_position = task_spec.end_position
        self.arm_type = task_spec.arm_type
        self.config = config

        task_text = f"{self.task_name} {task_spec.prompt}".lower()
        self.is_hot = "hot" in task_text or "热" in task_text
        self.is_cold = "cold" in task_text or "冷" in task_text or "凉" in task_text

        if self.is_hot == self.is_cold:
            raise ValueError("Task must identify exactly one of hot water or cold water")
        if self.arm_type != "left_arm":
            raise ValueError("Water tasks must use the left arm")

        self.web_policy = websocket_client_policy.WebsocketClientPolicy(
            "192.168.50.114", 8182
        )
        self.g2_control = G2CorobotEnv()
        (
            self.left_joint_lower,
            self.left_joint_upper,
            self.right_joint_lower,
            self.right_joint_upper,
        ) = self.load_joint_limits(ROOT_DIR / "config" / "robot_config_ori.json")

    @staticmethod
    def load_joint_limits(file_path: Path) -> tuple[np.ndarray, ...]:
        with file_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        return (
            np.asarray(config["left_arm"]["lower"]),
            np.asarray(config["left_arm"]["upper"]),
            np.asarray(config["right_arm"]["lower"]),
            np.asarray(config["right_arm"]["upper"]),
        )

    def clip_to_limits(self, actions: np.ndarray, arm_type: str) -> np.ndarray:
        if arm_type == "left_arm":
            lower, upper = self.left_joint_lower, self.left_joint_upper
        else:
            lower, upper = self.right_joint_lower, self.right_joint_upper
        if np.any(actions < lower) or np.any(actions > upper):
            logger.warning("%s action exceeded joint limits and was clipped", arm_type)
        return np.clip(actions, lower, upper)

    @staticmethod
    def build_payload(obs: dict[str, Any], prompt: str) -> dict[str, Any]:
        arm = obs["observation.states.arm_joint_states"][:14]
        gripper = obs["observation.states.gripper_states"][:2]
        state = list(arm[:7]) + [gripper[0]] + list(arm[7:14]) + [gripper[1]]
        return {
            "state": state,
            "images": {
                "cam_head": cv2.imdecode(
                    np.frombuffer(
                        obs["observation.images.head"]["image_data"],
                        dtype=np.uint8,
                    ),
                    cv2.IMREAD_COLOR,
                ),
                "cam_left": cv2.imdecode(
                    np.frombuffer(
                        obs["observation.images.hand_left"]["image_data"],
                        dtype=np.uint8,
                    ),
                    cv2.IMREAD_COLOR,
                ),
                "cam_right": cv2.imdecode(
                    np.frombuffer(
                        obs["observation.images.hand_right"]["image_data"],
                        dtype=np.uint8,
                    ),
                    cv2.IMREAD_COLOR,
                ),
            },
            "prompt": prompt,
        }

    def get_fast_state(self) -> dict[str, Any]:
        """Read joint and gripper feedback without decoding camera images."""
        raw = self.g2_control.env.get_observation()
        states = raw["states"]
        return {
            "observation.states.arm_joint_states": states["arm_joint_states"],
            "observation.states.gripper_states": states["gripper_states"],
        }

    def inf_loop(self) -> bool:
        initial_obs = self.get_fast_state()
        judge = PaddleWaterJudge(
            self.config,
            get_arm_joints(initial_obs, self.arm_type),
            get_gripper(initial_obs, self.arm_type),
        )

        for step in range(self.config.max_step):
            obs = self.g2_control.get_state()
            infer_start = time.monotonic()
            response = np.asarray(
                self.web_policy.infer(self.build_payload(obs, self.prompt))["actions"]
            )
            logger.info(
                "Step %d inference: %.1f ms",
                step,
                (time.monotonic() - infer_start) * 1000,
            )

            left_arm = self.clip_to_limits(response[:, :7], "left_arm")
            left_gripper = response[:, 7:8]
            right_arm = self.clip_to_limits(response[:, 8:15], "right_arm")
            right_gripper = response[:, 15:16]
            self.g2_control.execute(
                {
                    "left_arm": left_arm.tolist(),
                    "left_gripper": left_gripper.tolist(),
                    "right_arm": right_arm.tolist(),
                    "right_gripper": right_gripper.tolist(),
                },
                ref_times=self.config.ref_time,
            )

            actual = self.get_fast_state()
            actual_gripper = get_gripper(actual, self.arm_type)
            command_gripper = float(np.mean(left_gripper[-5:]))
            logger.info(
                "Step %d: actual left gripper %.3f, command mean %.3f, stage %s",
                step,
                actual_gripper,
                command_gripper,
                judge.stage,
            )

            success = judge.update(
                get_arm_joints(actual, self.arm_type),
                actual_gripper,
                time.monotonic(),
            )
            if judge.failed:
                return False
            if success:
                return True

        logger.error("Task timed out after %d steps", self.config.max_step)
        return False

    def run(self) -> bool:
        self.g2_control.reset_pos(self.init_position)
        try:
            success = self.inf_loop()
        except Exception:
            logger.exception("Water task raised an exception")
            return False
        if success:
            self.g2_control.reset_pos(self.end_position)
            logger.info("=== Water task completed successfully ===")
        else:
            logger.error("=== Water task failed ===")
        return success


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a water-dispenser task")
    parser.add_argument(
        "--task",
        default="left_arm_to_get_cold_water",
        help="left_arm_to_get_hot_water or left_arm_to_get_cold_water",
    )
    parser.add_argument("--max-step", type=int, default=150)
    parser.add_argument("--ref-time", type=float, default=1.0)
    parser.add_argument("--hot-button-hold-seconds", type=float, default=4.0)
    parser.add_argument("--grasp-min", type=float, default=-0.50)
    parser.add_argument("--grasp-max", type=float, default=-0.02)
    parser.add_argument("--cold-hold-seconds", type=float, default=3.0)
    parser.add_argument("--press-enter-tolerance", type=float, default=0.18)
    parser.add_argument("--press-exit-tolerance", type=float, default=0.25)
    parser.add_argument("--withdraw-min-distance", type=float, default=0.08)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_args()
    try:
        loader = TaskConfigLoader([ROOT_DIR / "config" / "config_V2_0.json"])
        task_spec = loader.resolve_task(args.task)
        config = WaterControlConfig(
            max_step=args.max_step,
            ref_time=args.ref_time,
            hot_button_hold_seconds=args.hot_button_hold_seconds,
            grasp_min=args.grasp_min,
            grasp_max=args.grasp_max,
            cold_hold_seconds=args.cold_hold_seconds,
            press_enter_tolerance=args.press_enter_tolerance,
            press_exit_tolerance=args.press_exit_tolerance,
            withdraw_min_distance=args.withdraw_min_distance,
        )
        success = WaterControl(task_spec, config).run()
    except Exception:
        logger.exception("Unable to start water task")
        success = False
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
