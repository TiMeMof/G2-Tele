#!/usr/bin/env python3
# -- coding: UTF-8
"""run_pick.py 的时序诊断版本。

用途：
1. 测量观测 RPC、图像解码、模型推理、动作后处理和 execute_action RPC 耗时；
2. 记录相邻模型动作块的提交间隔与可观测提交频率；
3. 使用独立 RemoteEnvClient 连接只读采样机器人状态，估计状态观测频率、
   运动开始延迟，以及动作块末端目标是否在下一动作块提交前到达；
4. 保存 CSV/JSON 日志，明确整个动作块如何通过一次 RPC 下发。

重要说明：
- execute_action RPC 返回只表示远端服务返回结果，不等同于物理轨迹执行完成；
- trajectory_reference_time 的精确定义和底层控制周期仍由远端 Corobot 服务决定；
- 本脚本测得的是客户端可观测的命令提交频率和状态反馈频率，不应直接等同于
  电机控制器内部频率。

示例：
    python intergrated_test/run_pick_timing.py --task right_arm_pick_wahaha
    python intergrated_test/run_pick_timing.py --task right_arm_pick_wahaha --probe-rate 50
    python intergrated_test/run_pick_timing.py --task right_arm_pick_wahaha --record-full-actions
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import threading
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import msgpack
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from corobot_client_cpp.remote_env_client import RemoteEnvClient
from openpi_client import websocket_client_policy
from run_pick import SyncControl, TaskConfigLoader


logger = logging.getLogger("run_pick_timing")
TIMING_RECORD_DIR = ROOT_DIR / "record" / "timing"


def _wall_time() -> str:
    return datetime.now().isoformat(timespec="milliseconds")


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _rate_stats(timestamps: list[float]) -> dict[str, float | int | None]:
    if len(timestamps) < 2:
        return {
            "count": len(timestamps),
            "duration_s": 0.0,
            "average_hz": None,
            "median_interval_ms": None,
            "min_interval_ms": None,
            "max_interval_ms": None,
        }

    intervals = np.diff(np.asarray(timestamps, dtype=np.float64))
    duration = float(timestamps[-1] - timestamps[0])
    return {
        "count": len(timestamps),
        "duration_s": duration,
        "average_hz": float((len(timestamps) - 1) / duration) if duration > 0 else None,
        "median_interval_ms": float(np.median(intervals) * 1000.0),
        "min_interval_ms": float(np.min(intervals) * 1000.0),
        "max_interval_ms": float(np.max(intervals) * 1000.0),
    }


class StateProbe:
    """通过独立 WebSocket 连接只读采样关节和夹爪状态。

    不能与主控制循环共用 RemoteEnvClient：同步 WebSocket 不支持一个连接上并发
    发送 get_observation 和 execute_action RPC。
    """

    def __init__(self, host: str, port: int, target_hz: float):
        if target_hz <= 0:
            raise ValueError("probe rate 必须大于 0")
        self.host = host
        self.port = port
        self.target_hz = target_hz
        self.samples: list[dict[str, Any]] = []
        self.errors: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run,
            name="corobot-state-probe",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logger.warning("状态采样线程未在 5 秒内退出")

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self.samples)

    def _run(self) -> None:
        client = RemoteEnvClient(host=self.host, port=self.port, log_level=logging.ERROR)
        interval = 1.0 / self.target_hz
        next_deadline = time.perf_counter()

        try:
            if not client.setup():
                raise RuntimeError("状态采样连接建立失败")

            while not self._stop_event.is_set():
                rpc_start = time.perf_counter()
                try:
                    observation = client.get_observation(
                        decode_image=False,
                        include_images=False,
                        include_states=True,
                    )
                    rpc_end = time.perf_counter()
                    states = observation.get("states", {}) if observation else {}
                    arm = np.asarray(states.get("arm_joint_states", []), dtype=np.float64)[:14]
                    gripper = np.asarray(states.get("gripper_states", []), dtype=np.float64)[:2]

                    if arm.size != 14 or gripper.size != 2:
                        raise ValueError(
                            f"状态维度异常: arm={arm.shape}, gripper={gripper.shape}"
                        )

                    sample = {
                        "monotonic_s": rpc_end,
                        "wall_time": _wall_time(),
                        "rpc_ms": (rpc_end - rpc_start) * 1000.0,
                        "server_timestamps": _jsonable(observation.get("timestamps", {})),
                        "arm": arm,
                        "gripper": gripper,
                    }
                    with self._lock:
                        self.samples.append(sample)
                except Exception as exc:
                    with self._lock:
                        self.errors.append(
                            {
                                "monotonic_s": time.perf_counter(),
                                "wall_time": _wall_time(),
                                "error": repr(exc),
                            }
                        )
                    logger.warning("状态采样失败: %s", exc)

                next_deadline += interval
                wait_s = next_deadline - time.perf_counter()
                if wait_s > 0:
                    self._stop_event.wait(wait_s)
                else:
                    # RPC 已经慢于目标周期，从当前时刻重新计时，避免持续追赶。
                    next_deadline = time.perf_counter()
        except Exception as exc:
            with self._lock:
                self.errors.append(
                    {
                        "monotonic_s": time.perf_counter(),
                        "wall_time": _wall_time(),
                        "error": repr(exc),
                    }
                )
            logger.exception("状态采样线程退出")
        finally:
            try:
                client.close()
            except Exception:
                pass


class TimingRecorder:
    def __init__(
        self,
        task_name: str,
        arm_type: str,
        prompt: str,
        joint_tolerance: float,
        motion_epsilon: float,
        record_full_actions: bool,
    ):
        self.task_name = task_name
        self.arm_type = arm_type
        self.prompt = prompt
        self.joint_tolerance = joint_tolerance
        self.motion_epsilon = motion_epsilon
        self.record_full_actions = record_full_actions
        self.start_wall_time = _wall_time()
        self.start_monotonic = time.perf_counter()
        self.iterations: list[dict[str, Any]] = []
        self.commands: list[dict[str, Any]] = []
        self.chunks: list[dict[str, Any]] = []
        self.reset_events: list[dict[str, Any]] = []
        self.exception: str | None = None

    def add_iteration(self, row: dict[str, Any]) -> None:
        self.iterations.append(_jsonable(row))

    def add_command(self, row: dict[str, Any]) -> None:
        self.commands.append(_jsonable(row))

    def add_chunk(self, row: dict[str, Any]) -> None:
        self.chunks.append(_jsonable(row))

    def add_reset(self, phase: str, elapsed_s: float) -> None:
        self.reset_events.append(
            {"phase": phase, "elapsed_s": elapsed_s, "wall_time": _wall_time()}
        )

    def _analyse_state_probe(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        if not samples:
            return {
                "available": False,
                "note": "未获得独立状态采样；无法估计状态反馈频率或末端目标到达时间。",
            }

        timestamps = [float(sample["monotonic_s"]) for sample in samples]
        arms = np.asarray([sample["arm"] for sample in samples], dtype=np.float64)
        probe_stats = _rate_stats(timestamps)
        probe_stats["available"] = True
        probe_stats["mean_rpc_ms"] = float(
            np.mean([float(sample["rpc_ms"]) for sample in samples])
        )

        if len(samples) >= 2:
            state_deltas = np.max(np.abs(np.diff(arms, axis=0)), axis=1)
            changed_indices = np.flatnonzero(state_deltas > self.motion_epsilon)
            duration = timestamps[-1] - timestamps[0]
            probe_stats["distinct_state_change_events"] = int(len(changed_indices))
            probe_stats["observed_state_change_event_hz"] = (
                float(len(changed_indices) / duration) if duration > 0 else None
            )
            probe_stats["state_change_epsilon_rad"] = self.motion_epsilon
        else:
            probe_stats["distinct_state_change_events"] = 0
            probe_stats["observed_state_change_event_hz"] = None

        return probe_stats

    def _analyse_chunk_targets(
        self,
        samples: list[dict[str, Any]],
        task_end_monotonic: float,
    ) -> list[dict[str, Any]]:
        if not samples:
            return []

        model_chunks = [chunk for chunk in self.chunks if chunk["kind"] == "model_chunk"]
        analyses: list[dict[str, Any]] = []
        active_slice = slice(0, 7) if self.arm_type == "left_arm" else slice(7, 14)

        for index, chunk in enumerate(model_chunks):
            submit_time = float(chunk["submit_monotonic_s"])
            next_submit = (
                float(model_chunks[index + 1]["submit_monotonic_s"])
                if index + 1 < len(model_chunks)
                else task_end_monotonic
            )
            window_samples = [
                sample
                for sample in samples
                if submit_time <= float(sample["monotonic_s"]) < next_submit
            ]
            target = np.asarray(chunk["last_action"], dtype=np.float64)
            active_target = target[:7] if self.arm_type == "left_arm" else target[8:15]

            result: dict[str, Any] = {
                "step": chunk["step"],
                "submit_to_next_submit_ms": (next_submit - submit_time) * 1000.0,
                "state_sample_count": len(window_samples),
                "joint_tolerance_rad": self.joint_tolerance,
                "reached_final_joint_target_before_next_chunk": False,
                "reach_latency_ms": None,
                "motion_start_latency_ms": None,
                "min_final_target_max_abs_error_rad": None,
            }
            if not window_samples:
                analyses.append(result)
                continue

            sample_arms = np.asarray(
                [np.asarray(sample["arm"], dtype=np.float64)[active_slice] for sample in window_samples]
            )
            sample_times = np.asarray(
                [float(sample["monotonic_s"]) for sample in window_samples], dtype=np.float64
            )
            final_errors = np.max(np.abs(sample_arms - active_target), axis=1)
            result["min_final_target_max_abs_error_rad"] = float(np.min(final_errors))

            reached = np.flatnonzero(final_errors <= self.joint_tolerance)
            if reached.size:
                result["reached_final_joint_target_before_next_chunk"] = True
                result["reach_latency_ms"] = float(
                    (sample_times[reached[0]] - submit_time) * 1000.0
                )

            start_state = sample_arms[0]
            motion = np.flatnonzero(
                np.max(np.abs(sample_arms - start_state), axis=1) > self.motion_epsilon
            )
            if motion.size:
                result["motion_start_latency_ms"] = float(
                    (sample_times[motion[0]] - submit_time) * 1000.0
                )

            analyses.append(result)

        return analyses

    def build_summary(
        self,
        success: bool,
        samples: list[dict[str, Any]],
        task_end_monotonic: float,
        probe_errors: list[dict[str, Any]],
    ) -> dict[str, Any]:
        inference_ms = [
            float(row["inference_ms"])
            for row in self.iterations
            if row.get("inference_ms") is not None
        ]
        model_commands = [
            command for command in self.commands if command.get("kind") == "model_chunk"
        ]
        submit_times = [float(command["submit_monotonic_s"]) for command in model_commands]
        execute_rpc_ms = [float(command["execute_rpc_ms"]) for command in model_commands]
        chunk_analysis = self._analyse_chunk_targets(samples, task_end_monotonic)
        reached_count = sum(
            bool(row["reached_final_joint_target_before_next_chunk"])
            for row in chunk_analysis
        )

        return {
            "task_name": self.task_name,
            "arm_type": self.arm_type,
            "prompt": self.prompt,
            "success": success,
            "start_wall_time": self.start_wall_time,
            "end_wall_time": _wall_time(),
            "task_elapsed_s": task_end_monotonic - self.start_monotonic,
            "exception": self.exception,
            "iteration_count": len(self.iterations),
            "model_chunk_count": len(model_commands),
            "mean_inference_ms": float(np.mean(inference_ms)) if inference_ms else None,
            "p50_inference_ms": float(np.percentile(inference_ms, 50)) if inference_ms else None,
            "p95_inference_ms": float(np.percentile(inference_ms, 95)) if inference_ms else None,
            "mean_execute_action_rpc_ms": (
                float(np.mean(execute_rpc_ms)) if execute_rpc_ms else None
            ),
            "model_chunk_submission": _rate_stats(submit_times),
            "state_probe": self._analyse_state_probe(samples),
            "chunk_target_analysis": {
                "joint_tolerance_rad": self.joint_tolerance,
                "reached_before_next_count": reached_count,
                "analysed_chunk_count": len(chunk_analysis),
                "details": chunk_analysis,
            },
            "probe_errors": _jsonable(probe_errors),
            "reset_events": self.reset_events,
            "interpretation": {
                "execute_action_rpc_ms": (
                    "客户端发送整个动作块并等待 RPC 响应的时间；不是物理动作执行时间。"
                ),
                "model_chunk_submission.average_hz": (
                    "模型动作块提交频率，不是动作点频率或电机控制频率。"
                ),
                "state_probe.average_hz": (
                    "独立客户端可观测的状态 RPC 采样频率，受网络和服务端影响。"
                ),
                "observed_state_change_event_hz": (
                    "采样中检测到关节数值变化的事件频率，只是实际控制频率的下界/近似。"
                ),
                "reach_latency_ms": (
                    "提交动作块后，状态首次进入末帧关节目标容差的时间；只有到达时才可作为"
                    "可观测动作完成时间。"
                ),
                "not_reached_before_next": (
                    "若下一块提交前未到达上一块末帧目标，说明上一块没有在该窗口完成；"
                    "但仅凭客户端日志不能证明底层一定采用了覆盖而不是重新规划/排队。"
                ),
            },
        }

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                encoded = {
                    key: (
                        json.dumps(_jsonable(value), ensure_ascii=False)
                        if isinstance(value, (dict, list, tuple, np.ndarray))
                        else value
                    )
                    for key, value in row.items()
                }
                writer.writerow(encoded)

    def save(
        self,
        success: bool,
        samples: list[dict[str, Any]],
        probe_errors: list[dict[str, Any]],
        task_end_monotonic: float,
    ) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = TIMING_RECORD_DIR / f"{self.task_name}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=False)

        sample_rows = []
        for index, sample in enumerate(samples):
            arm = np.asarray(sample["arm"], dtype=np.float64)
            gripper = np.asarray(sample["gripper"], dtype=np.float64)
            row = {
                "sample_index": index,
                "elapsed_s": float(sample["monotonic_s"] - self.start_monotonic),
                "wall_time": sample["wall_time"],
                "rpc_ms": sample["rpc_ms"],
                "server_timestamps": sample["server_timestamps"],
            }
            row.update({f"arm_{joint}": float(value) for joint, value in enumerate(arm)})
            row.update(
                {f"gripper_{joint}": float(value) for joint, value in enumerate(gripper)}
            )
            sample_rows.append(row)

        self._write_csv(output_dir / "iterations.csv", self.iterations)
        self._write_csv(output_dir / "command_events.csv", self.commands)
        self._write_csv(output_dir / "state_samples.csv", sample_rows)

        with (output_dir / "action_chunks.jsonl").open("w", encoding="utf-8") as handle:
            for chunk in self.chunks:
                handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")

        summary = self.build_summary(
            success=success,
            samples=samples,
            task_end_monotonic=task_end_monotonic,
            probe_errors=probe_errors,
        )
        with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)

        return output_dir


class TimingSyncControl(SyncControl):
    def __init__(self, task_spec, args: argparse.Namespace):
        super().__init__(task_spec)
        self.args = args
        if args.max_step is not None or args.ref_time is not None:
            self.control_config = replace(
                self.control_config,
                max_step=(
                    args.max_step
                    if args.max_step is not None
                    else self.control_config.max_step
                ),
                ref_time=(
                    args.ref_time
                    if args.ref_time is not None
                    else self.control_config.ref_time
                ),
            )

        self.web_policy = websocket_client_policy.WebsocketClientPolicy(
            args.policy_host,
            args.policy_port,
        )
        self.timing = TimingRecorder(
            task_name=self.task_name,
            arm_type=self.arm_type,
            prompt=self.prompt,
            joint_tolerance=args.joint_tolerance,
            motion_epsilon=args.motion_epsilon,
            record_full_actions=args.record_full_actions,
        )
        self._last_model_submit: float | None = None
        self._last_chunk_final_action: np.ndarray | None = None

        self.state_probe: StateProbe | None = None
        if not args.no_state_probe:
            self.state_probe = StateProbe(
                host=self.g2_control.env.host,
                port=self.g2_control.env.port,
                target_hz=args.probe_rate,
            )

    def _execute_action_parts(
        self,
        action_parts: dict[str, Any],
        ref_time: float,
        kind: str,
        step: int,
    ) -> tuple[bool, dict[str, Any]]:
        """复现 G2CorobotEnv.execute，但保留 RPC 返回值和完整计时。"""
        action = self.g2_control.build_action(
            action_parts,
            trajectory_reference_time=ref_time,
        )
        packed_bytes = len(msgpack.packb(action))
        point_counts = {
            key: len(value)
            for key, value in action_parts.items()
            if value is not None and hasattr(value, "__len__")
        }

        submit_start = time.perf_counter()
        submit_gap_ms = (
            (submit_start - self._last_model_submit) * 1000.0
            if kind == "model_chunk" and self._last_model_submit is not None
            else None
        )
        success = self.g2_control.env.execute_action(action)
        rpc_end = time.perf_counter()

        if kind == "model_chunk":
            self._last_model_submit = submit_start

        event = {
            "step": step,
            "kind": kind,
            "wall_time": _wall_time(),
            "submit_monotonic_s": submit_start,
            "rpc_return_monotonic_s": rpc_end,
            "execute_rpc_ms": (rpc_end - submit_start) * 1000.0,
            "submit_gap_ms": submit_gap_ms,
            "observable_chunk_submit_hz": (
                1000.0 / submit_gap_ms if submit_gap_ms and submit_gap_ms > 0 else None
            ),
            "rpc_success": bool(success),
            "trajectory_reference_time_s": ref_time,
            "point_counts": point_counts,
            "msgpack_payload_bytes": packed_bytes,
            "one_rpc_contains_whole_chunk": True,
        }
        self.timing.add_command(event)
        return bool(success), event

    def _previous_target_error(self, arm_states: np.ndarray) -> float | None:
        if self._last_chunk_final_action is None:
            return None
        if self.arm_type == "left_arm":
            current = arm_states[:7]
            target = self._last_chunk_final_action[:7]
        else:
            current = arm_states[7:14]
            target = self._last_chunk_final_action[8:15]
        return float(np.max(np.abs(np.asarray(current) - np.asarray(target))))

    def inf_loop(self):
        try_num = 0
        for step in range(self.control_config.max_step):
            loop_start = time.perf_counter()

            observation_start = time.perf_counter()
            obs = self.g2_control.get_state()
            observation_end = time.perf_counter()

            arm_states = np.asarray(
                obs["observation.states.arm_joint_states"][:14], dtype=np.float64
            )
            gripper_states = np.asarray(
                obs["observation.states.gripper_states"][:2], dtype=np.float64
            )
            previous_target_error = self._previous_target_error(arm_states)

            decode_start = time.perf_counter()
            state = (
                list(arm_states[:7])
                + [gripper_states[0]]
                + list(arm_states[7:14])
                + [gripper_states[1]]
            )
            payload = {
                "state": state,
                "images": {
                    "cam_head": cv2.imdecode(
                        np.frombuffer(
                            obs["observation.images.head"]["image_data"], dtype=np.uint8
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
                "prompt": self.prompt,
            }
            decode_end = time.perf_counter()

            inference_start = time.perf_counter()
            response = np.asarray(self.web_policy.infer(payload)["actions"], dtype=np.float64)
            inference_end = time.perf_counter()

            if response.ndim != 2 or response.shape[1] < 16:
                raise ValueError(f"模型动作形状异常，期望 (T, >=16)，实际 {response.shape}")

            postprocess_start = time.perf_counter()
            left_arm = self.clip_to_limits(response[:, :7], "left_arm")
            left_gripper = response[:, 7:8]
            right_arm = self.clip_to_limits(response[:, 8:15], "right_arm")
            right_gripper = response[:, 15:16]
            submitted_actions = np.concatenate(
                [left_arm, left_gripper, right_arm, right_gripper], axis=1
            )
            postprocess_end = time.perf_counter()

            rpc_success, command_event = self._execute_action_parts(
                {
                    "left_arm": left_arm.tolist(),
                    "left_gripper": left_gripper.tolist(),
                    "right_arm": right_arm.tolist(),
                    "right_gripper": right_gripper.tolist(),
                },
                ref_time=self.control_config.ref_time,
                kind="model_chunk",
                step=step,
            )

            chunk_row = {
                "step": step,
                "kind": "model_chunk",
                "submit_monotonic_s": command_event["submit_monotonic_s"],
                "shape": list(submitted_actions.shape),
                "first_action": submitted_actions[0].tolist(),
                "last_action": submitted_actions[-1].tolist(),
                "trajectory_reference_time_s": self.control_config.ref_time,
                "full_actions": (
                    submitted_actions.tolist() if self.args.record_full_actions else None
                ),
            }
            self.timing.add_chunk(chunk_row)
            self._last_chunk_final_action = submitted_actions[-1].copy()

            inference_ms = (inference_end - inference_start) * 1000.0
            logger.info(
                "[step=%d] obs_rpc=%.1f ms | decode=%.1f ms | infer=%.1f ms | "
                "chunk=%s | execute_rpc=%.1f ms | submit_gap=%s | chunk_submit_hz=%s | "
                "ref_time=%.3f s | previous_final_error=%s",
                step,
                (observation_end - observation_start) * 1000.0,
                (decode_end - decode_start) * 1000.0,
                inference_ms,
                tuple(submitted_actions.shape),
                command_event["execute_rpc_ms"],
                (
                    f"{command_event['submit_gap_ms']:.1f} ms"
                    if command_event["submit_gap_ms"] is not None
                    else "N/A"
                ),
                (
                    f"{command_event['observable_chunk_submit_hz']:.2f} Hz"
                    if command_event["observable_chunk_submit_hz"] is not None
                    else "N/A"
                ),
                self.control_config.ref_time,
                (
                    f"{previous_target_error:.4f} rad"
                    if previous_target_error is not None
                    else "N/A"
                ),
            )

            row = {
                "step": step,
                "wall_time": _wall_time(),
                "loop_start_monotonic_s": loop_start,
                "observation_rpc_ms": (observation_end - observation_start) * 1000.0,
                "payload_decode_ms": (decode_end - decode_start) * 1000.0,
                "inference_ms": inference_ms,
                "postprocess_ms": (postprocess_end - postprocess_start) * 1000.0,
                "execute_rpc_ms": command_event["execute_rpc_ms"],
                "chunk_size": int(submitted_actions.shape[0]),
                "action_dim": int(submitted_actions.shape[1]),
                "trajectory_reference_time_s": self.control_config.ref_time,
                "submit_gap_ms": command_event["submit_gap_ms"],
                "observable_chunk_submit_hz": command_event[
                    "observable_chunk_submit_hz"
                ],
                "previous_final_target_error_at_observation_rad": previous_target_error,
                "previous_final_target_within_tolerance": (
                    previous_target_error <= self.args.joint_tolerance
                    if previous_target_error is not None
                    else None
                ),
                "rpc_success": rpc_success,
                "outcome": "continue",
            }

            if self.control_config.arm_type == "left_arm":
                response_gripper_mean = float(np.mean(left_gripper[-5:]))
                reset_action = {"left_gripper": [[-0.785]]}
            else:
                response_gripper_mean = float(np.mean(right_gripper[-5:]))
                reset_action = {"right_gripper": [[-0.785]]}

            if response_gripper_mean > -0.05:
                settle_start = time.perf_counter()
                time.sleep(1.5)
                gripper_obs_start = time.perf_counter()
                grasp_obs = self.g2_control.get_state()
                gripper_obs_end = time.perf_counter()
                current_gripper = (
                    grasp_obs["observation.states.gripper_states"][0]
                    if self.control_config.arm_type == "left_arm"
                    else grasp_obs["observation.states.gripper_states"][1]
                )
                row["gripper_settle_ms"] = (gripper_obs_start - settle_start) * 1000.0
                row["gripper_observation_rpc_ms"] = (
                    gripper_obs_end - gripper_obs_start
                ) * 1000.0
                row["commanded_gripper_mean"] = response_gripper_mean
                row["observed_gripper"] = float(current_gripper)
                logger.info(
                    "当前夹爪值: %.3f, 指令期望闭合: %.3f",
                    current_gripper,
                    response_gripper_mean,
                )

                if current_gripper > self.control_config.fail_thehold:
                    if try_num < self.control_config.try_num:
                        try_num += 1
                        logger.warning(
                            "第%d次夹取失败(夹爪值: %.3f)，回到初始位置并打开夹爪重试",
                            try_num,
                            current_gripper,
                        )
                        self._execute_action_parts(
                            reset_action,
                            ref_time=self.control_config.retry_ref_time,
                            kind="gripper_retry_reset",
                            step=step,
                        )
                        time.sleep(2)
                        row["outcome"] = "grasp_failed_retry"
                        row["loop_ms"] = (time.perf_counter() - loop_start) * 1000.0
                        self.timing.add_iteration(row)
                        continue

                    logger.error("尝试%d次后仍未夹取成功", self.control_config.try_num)
                    row["outcome"] = "grasp_failed_final"
                    row["loop_ms"] = (time.perf_counter() - loop_start) * 1000.0
                    self.timing.add_iteration(row)
                    return False

                if current_gripper <= self.control_config.success_thehold:
                    logger.info("夹取成功，实际夹爪值: %.3f", current_gripper)
                    row["outcome"] = "success"
                    row["loop_ms"] = (time.perf_counter() - loop_start) * 1000.0
                    self.timing.add_iteration(row)
                    return True

            row["loop_ms"] = (time.perf_counter() - loop_start) * 1000.0
            self.timing.add_iteration(row)

        return False

    def run(self):
        initial_reset_start = time.perf_counter()
        self.g2_control.reset_pos(self.init_position)
        self.timing.add_reset(
            "initial_reset",
            time.perf_counter() - initial_reset_start,
        )

        if self.state_probe is not None:
            self.state_probe.start()
            # 给第二条 WebSocket 连接一个很短的建连窗口；不影响动作下发语义。
            time.sleep(0.2)

        success = False
        try:
            success = bool(self.inf_loop())
        except Exception as exc:
            self.timing.exception = repr(exc)
            logger.exception("任务执行异常")
        finally:
            task_end_monotonic = time.perf_counter()
            if self.state_probe is not None:
                self.state_probe.stop()

        if success:
            end_reset_start = time.perf_counter()
            self.g2_control.reset_pos(self.end_position)
            self.timing.add_reset("end_reset", time.perf_counter() - end_reset_start)
        else:
            logger.error("任务失败")

        samples = self.state_probe.snapshot() if self.state_probe is not None else []
        probe_errors = self.state_probe.errors if self.state_probe is not None else []
        output_dir = self.timing.save(
            success=success,
            samples=samples,
            probe_errors=probe_errors,
            task_end_monotonic=task_end_monotonic,
        )
        logger.info("时序诊断日志已保存: %s", output_dir)
        logger.info("优先查看: %s", output_dir / "summary.json")
        logger.info("逐轮计时: %s", output_dir / "iterations.csv")
        logger.info("动作提交: %s", output_dir / "command_events.csv")
        logger.info("状态采样: %s", output_dir / "state_samples.csv")
        return success


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="拿取任务动作下发与时序诊断")
    parser.add_argument("--task", default="right_arm_pick_yibao", help="任务名称")
    parser.add_argument("--policy-host", default="192.168.50.114")
    parser.add_argument("--policy-port", type=int, default=8182)
    parser.add_argument(
        "--probe-rate",
        type=float,
        default=50.0,
        help="独立状态采样目标频率；真实频率以 summary.json 为准",
    )
    parser.add_argument(
        "--no-state-probe",
        action="store_true",
        help="关闭独立状态采样，只记录主循环与动作 RPC 时序",
    )
    parser.add_argument(
        "--record-full-actions",
        action="store_true",
        help="在 action_chunks.jsonl 中保存每个完整动作块，而不仅是首末帧",
    )
    parser.add_argument("--max-step", type=int, default=None)
    parser.add_argument(
        "--ref-time",
        type=float,
        default=None,
        help="覆盖原始 trajectory_reference_time；默认沿用 run_pick.py 的 1.0 秒",
    )
    parser.add_argument(
        "--joint-tolerance",
        type=float,
        default=0.03,
        help="判定动作块末帧关节目标到达的最大绝对误差，单位 rad",
    )
    parser.add_argument(
        "--motion-epsilon",
        type=float,
        default=0.002,
        help="状态变化/运动开始判定阈值，单位 rad",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_args()
    loader = TaskConfigLoader(
        [
            ROOT_DIR / "config" / "config_V2_0.json",
        ]
    )
    task_spec = loader.resolve_task(args.task)
    control = TimingSyncControl(task_spec, args)
    success = control.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
