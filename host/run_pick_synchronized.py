#!/usr/bin/env python3
"""按完整动作块同步执行的拿取任务实验。

与 run_pick.py 的核心区别：每次下发 30 帧轨迹后，等待轨迹参考时间结束，
读取真实关节状态并检查末帧跟踪误差，然后才进行下一次视觉推理。默认不
保存运行记录；只有显式传入 --record 才保存图片、JSON 和分析报告。

注意：这会把原来约 6~7 Hz 的闭环频率降到约 1 Hz，是用于验证"轨迹被
后续动作块过早覆盖"假设的独立实验，不覆盖原程序。
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import queue
import statistics
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DRAWER_TEST_DIR = Path(__file__).resolve().parent
ROOT_DIR = DRAWER_TEST_DIR.parent
for path in (DRAWER_TEST_DIR, ROOT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_pick
from run_pick import SyncControl, TaskConfigLoader


logger = logging.getLogger("run_pick_synchronized")
OUTPUT_ROOT = ROOT_DIR / "record" / "synchronized"
DEFAULT_BASELINE = ROOT_DIR / "record" / "right_arm_pick_yibao_20260714_152710.json"
GRIPPER_SETTLE_SECONDS = 1.5
CAMERAS = {
    "cam_head": "observation.images.head",
    "cam_left": "observation.images.hand_left",
    "cam_right": "observation.images.hand_right",
}


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")


def floats(values: Any) -> list[float]:
    return [float(value) for value in values]


def max_error(left: list[float], right: list[float]) -> float:
    return max(abs(a - b) for a, b in zip(left, right))


class AsyncImageWriter:
    """后台保存原始 JPEG，避免磁盘写入阻塞控制线程。"""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.jobs: queue.Queue[tuple[Path, bytes] | None] = queue.Queue()
        self.error: str | None = None
        self.thread = threading.Thread(target=self._worker, name="image-writer", daemon=True)
        self.thread.start()

    def submit(self, relative_path: Path, data: Any) -> None:
        self.jobs.put((relative_path, bytes(data)))

    def _worker(self) -> None:
        while True:
            job = self.jobs.get()
            try:
                if job is None:
                    return
                try:
                    path, data = job
                    output = self.output_dir / path
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_bytes(data)
                except Exception as exc:
                    self.error = str(exc)
                    logger.exception("后台图片保存失败")
            finally:
                self.jobs.task_done()

    def close(self) -> None:
        self.jobs.put(None)
        self.jobs.join()
        self.thread.join(timeout=5)
        if self.error:
            raise RuntimeError(f"后台图片保存失败：{self.error}")


class ExperimentRecorder:
    def __init__(self, task_name: str, prompt: str, arm_type: str, args: argparse.Namespace):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = OUTPUT_ROOT / f"{task_name}_{stamp}"
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.images = AsyncImageWriter(self.output_dir)
        self.steps: list[dict[str, Any]] = []
        self.payload: dict[str, Any] = {
            "schema_version": 1,
            "experiment": "full_action_chunk_synchronized",
            "task_name": task_name,
            "prompt": prompt,
            "arm_type": arm_type,
            "start_time": now_text(),
            "settings": {
                "trajectory_reference_time": args.ref_time,
                "minimum_wait_seconds": args.min_wait,
                "tracking_timeout_seconds": args.tracking_timeout,
                "tracking_tolerance_rad": args.tracking_tolerance,
                "poll_interval_seconds": args.poll_interval,
                "max_steps": args.max_steps,
            },
            "steps": self.steps,
        }

    def record_images(self, step_index: int, obs: dict[str, Any]) -> dict[str, str]:
        paths: dict[str, str] = {}
        for name, key in CAMERAS.items():
            message = obs[key]
            relative = Path(f"step_{step_index:04d}") / f"{name}.jpg"
            self.images.submit(relative, message["image_data"])
            paths[name] = str(relative)
        return paths

    def finish(self, success: bool | None) -> Path:
        self.images.close()
        self.payload["success"] = success
        self.payload["step_count"] = len(self.steps)
        self.payload["end_time"] = now_text()
        output = self.output_dir / "synchronized_diagnostic.json"
        output.write_text(json.dumps(self.payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return output


class SynchronizedPickControl(SyncControl):
    def __init__(self, task_spec: Any, args: argparse.Namespace):
        super().__init__(task_spec)
        self.args = args
        self.control_config = type(self.control_config)(
            max_step=args.max_steps,
            try_num=self.control_config.try_num,
            arm_type=self.control_config.arm_type,
            fail_thehold=self.control_config.fail_thehold,
            success_thehold=self.control_config.success_thehold,
            ref_time=args.ref_time,
            retry_ref_time=self.control_config.retry_ref_time,
        )
        self.record: ExperimentRecorder | None = (
            ExperimentRecorder(self.task_name, self.prompt, self.arm_type, args)
            if args.record
            else None
        )

    @staticmethod
    def snapshot(obs: dict[str, Any]) -> dict[str, Any]:
        arm = floats(obs["observation.states.arm_joint_states"][:14])
        gripper = floats(obs["observation.states.gripper_states"][:2])
        return {
            "arm_joint_states": arm,
            "left_arm": arm[:7],
            "right_arm": arm[7:14],
            "gripper_states": gripper,
        }

    def active_arm(self, state: dict[str, Any]) -> list[float]:
        return state["left_arm" if self.arm_type == "left_arm" else "right_arm"]

    def execute_and_wait(
        self, action: dict[str, list[list[float]]], target_arm: list[float]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        sent_at = time.monotonic()
        self.g2_control.execute(action, ref_times=self.args.ref_time)

        # 不早于参考轨迹时间检查完成，避免把"接口已接收"误当作"轨迹已执行"。
        minimum_wait = max(self.args.min_wait, self.args.ref_time)
        if minimum_wait > 0:
            time.sleep(minimum_wait)

        deadline = sent_at + max(minimum_wait, self.args.tracking_timeout)
        samples = 0
        state_after: dict[str, Any] | None = None
        error = math.inf
        while True:
            state_after = self.snapshot(self.g2_control.get_state())
            samples += 1
            error = max_error(self.active_arm(state_after), target_arm)
            if error <= self.args.tracking_tolerance or time.monotonic() >= deadline:
                break
            time.sleep(self.args.poll_interval)

        timing = {
            "execution_wait_seconds": time.monotonic() - sent_at,
            "tracking_error_rad": error,
            "tracking_reached": error <= self.args.tracking_tolerance,
            "tracking_samples": samples,
        }
        return state_after, timing

    def inf_loop(self) -> bool:
        try_num = 0
        previous_cycle = time.monotonic()
        for step_index in range(self.control_config.max_step):
            cycle_started = time.monotonic()
            obs = self.g2_control.get_state()
            state_before = self.snapshot(obs)
            image_paths = self.record.record_images(step_index, obs) if self.record else None

            arm = state_before["arm_joint_states"]
            gripper = state_before["gripper_states"]
            model_state = arm[:7] + [gripper[0]] + arm[7:14] + [gripper[1]]
            images = {
                name: cv2.imdecode(
                    np.frombuffer(obs[key]["image_data"], dtype=np.uint8), cv2.IMREAD_COLOR
                )
                for name, key in CAMERAS.items()
            }

            inference_started = time.monotonic()
            response = np.asarray(
                self.web_policy.infer(
                    {"state": model_state, "images": images, "prompt": self.prompt}
                )["actions"],
                dtype=float,
            )
            inference_seconds = time.monotonic() - inference_started
            if response.ndim != 2 or response.shape[1] != 16 or not np.isfinite(response).all():
                raise ValueError(f"模型 actions 非法：shape={response.shape}")

            left_arm = self.clip_to_limits(response[:, :7], "left_arm")
            left_gripper = response[:, 7:8]
            right_arm = self.clip_to_limits(response[:, 8:15], "right_arm")
            right_gripper = response[:, 15:16]
            action = {
                "left_arm": left_arm.tolist(),
                "left_gripper": left_gripper.tolist(),
                "right_arm": right_arm.tolist(),
                "right_gripper": right_gripper.tolist(),
            }
            target_arm = (left_arm if self.arm_type == "left_arm" else right_arm)[-1].tolist()
            state_after, execution = self.execute_and_wait(action, target_arm)

            if self.record:
                self.record.steps.append(
                    {
                        "step": step_index,
                        "timestamp": now_text(),
                        "images": image_paths,
                        "model_input_state": model_state,
                        "state_before": state_before,
                        "state_after": state_after,
                        "actions": response.tolist(),
                        "timing": {
                            "inference_seconds": inference_seconds,
                            "cycle_seconds": time.monotonic() - cycle_started,
                            "seconds_since_previous_cycle": cycle_started - previous_cycle,
                            **execution,
                        },
                    }
                )
            previous_cycle = cycle_started
            logger.info(
                "step=%d inference=%.3fs wait=%.3fs tracking_error=%.4frad reached=%s",
                step_index,
                inference_seconds,
                execution["execution_wait_seconds"],
                execution["tracking_error_rad"],
                execution["tracking_reached"],
            )

            if self.arm_type == "left_arm":
                gripper_mean = float(np.mean(left_gripper[-5:]))
                gripper_index = 0
                reset_action = {"left_gripper": [[-0.785]]}
            else:
                gripper_mean = float(np.mean(right_gripper[-5:]))
                gripper_index = 1
                reset_action = {"right_gripper": [[-0.785]]}

            if gripper_mean > -0.05:
                # 手臂跟踪到位不代表夹爪已完成闭合；等待夹爪稳定后重新取样，
                # 避免将闭合途中的中间位置误判为"物体阻挡"。
                time.sleep(GRIPPER_SETTLE_SECONDS)
                state_after = self.snapshot(self.g2_control.get_state())
                current_gripper = float(state_after["gripper_states"][gripper_index])
                logger.info("夹爪期望闭合 %.3f，实际 %.3f", gripper_mean, current_gripper)
                if current_gripper > self.control_config.fail_thehold:
                    if try_num < self.control_config.try_num:
                        try_num += 1
                        logger.warning("第%d次夹取失败，打开夹爪后继续", try_num)
                        self.g2_control.execute(
                            reset_action, ref_times=self.control_config.retry_ref_time
                        )
                        time.sleep(2)
                        continue
                    return False
                if current_gripper <= self.control_config.success_thehold:
                    return True
        return False

    def run(self) -> tuple[bool, Path | None]:
        result: bool | None = None
        try:
            self.g2_control.reset_pos(self.init_position)
            result = self.inf_loop()
            if result:
                self.g2_control.reset_pos(self.end_position)
            return bool(result), self.record.output_dir if self.record else None
        finally:
            if self.record:
                output = self.record.finish(result)
                logger.info("同步实验记录：%s", output)
            remote = getattr(self.g2_control, "env", None)
            close = getattr(remote, "close", None)
            if callable(close):
                close()


def tail_motion(sequence: list[list[float]], window: int = 30) -> dict[str, float]:
    selected = sequence[-min(window, len(sequence)) :]
    if len(selected) < 2:
        return {"max_step_rad": 0.0, "median_step_rad": 0.0, "max_span_rad": 0.0}
    changes = [max_error(a, b) for a, b in zip(selected, selected[1:])]
    spans = [max(row[j] for row in selected) - min(row[j] for row in selected) for j in range(7)]
    return {
        "max_step_rad": max(changes),
        "median_step_rad": statistics.median(changes),
        "max_span_rad": max(spans),
    }


def analyze(output_dir: Path, baseline_path: Path | None) -> Path:
    record_path = output_dir / "synchronized_diagnostic.json"
    data = json.loads(record_path.read_text(encoding="utf-8"))
    steps = data["steps"]
    actual_before = [step["state_before"]["right_arm"] for step in steps]
    action_first = [step["actions"][0][8:15] for step in steps]
    action_last = [step["actions"][-1][8:15] for step in steps]
    tracking = [float(step["timing"]["tracking_error_rad"]) for step in steps]
    cycles = [float(step["timing"]["cycle_seconds"]) for step in steps]

    report: dict[str, Any] = {
        "step_count": len(steps),
        "success": data.get("success"),
        "actual_state_tail_motion": tail_motion(actual_before),
        "action_first_frame_tail_motion": tail_motion(action_first),
        "action_last_frame_tail_motion": tail_motion(action_last),
        "tracking_error_rad": {
            "median": statistics.median(tracking) if tracking else None,
            "maximum": max(tracking) if tracking else None,
            "reached_ratio": (
                sum(bool(step["timing"]["tracking_reached"]) for step in steps) / len(steps)
                if steps else None
            ),
        },
        "cycle_seconds": {
            "median": statistics.median(cycles) if cycles else None,
            "minimum": min(cycles) if cycles else None,
            "maximum": max(cycles) if cycles else None,
        },
    }
    if baseline_path and baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        baseline_first = [step["actions"][0][8:15] for step in baseline.get("steps", [])]
        report["baseline"] = {
            "path": str(baseline_path),
            "step_count": len(baseline_first),
            "action_first_frame_tail_motion": tail_motion(baseline_first),
        }

    output = output_dir / "analysis.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("自动分析报告：%s", output)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="完整执行动作块后再推理的拿取实验")
    parser.add_argument("--task", default="left_arm_drawer_pick_omeprazole_bottle")
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--ref-time", type=float, default=1.0)
    parser.add_argument("--min-wait", type=float, default=1.0)
    parser.add_argument("--tracking-timeout", type=float, default=1.5)
    parser.add_argument("--tracking-tolerance", type=float, default=0.05)
    parser.add_argument("--poll-interval", type=float, default=0.05)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument(
        "--record",
        action="store_true",
        help="保存图片、同步诊断 JSON 和分析报告；默认不保存",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_args()
    if min(
        args.max_steps,
        args.ref_time,
        args.min_wait,
        args.tracking_timeout,
        args.tracking_tolerance,
        args.poll_interval,
    ) <= 0:
        raise ValueError("所有时间、阈值和 max-steps 参数都必须为正")
    run_pick.ENABLE_ACTION_RECORDING = False
    loader = TaskConfigLoader([
        ROOT_DIR / "config" / "right_add_drawer_config.json",
        ROOT_DIR / "config" / "left_add_drawer_config.json",
    ])
    task = loader.resolve_task(args.task)
    success, output_dir = SynchronizedPickControl(task, args).run()
    if output_dir is not None:
        analyze(output_dir, args.baseline)
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
