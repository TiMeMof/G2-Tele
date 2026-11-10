#!/usr/bin/env python3
"""Run place inference after mosaicing the task-gripper object in all cameras.

This is an independent entry point. It imports run_place.py but does not modify
it. The robot process stays in the policy environment, while a child process in
the sam31 environment owns SAM 3.1 and returns binary masks over a JSON-lines
pipe. Object selection is appearance-agnostic: the initial mask must be tied to
the configured left/right gripper, and temporal tracking follows it after release.
The policy is never called when the masking pipeline fails closed.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
from pathlib import Path
import select
import subprocess
import sys
import time
import traceback
import warnings
from typing import Any

import cv2
import numpy as np

IS_MASK_WORKER = "--mask-worker" in sys.argv
if IS_MASK_WORKER:
    ROOT_DIR = None
    SyncControlPlace = object
    TaskConfigLoader = None
    TaskSpec = Any
else:
    from run_place import ROOT_DIR, SyncControlPlace, TaskConfigLoader, TaskSpec


LOGGER = logging.getLogger(__name__)
CAMERAS = ("cam_head", "cam_left", "cam_right")
CAMERA_OBSERVATIONS = {
    "cam_head": "observation.images.head",
    "cam_left": "observation.images.hand_left",
    "cam_right": "observation.images.hand_right",
}
HELD_OBJECT_PROMPTS = {
    "left": (
        "object held by a robot gripper.",
        "held item.",
        "object.",
    ),
    "right": (
        "object held by a robot gripper.",
        "held item.",
        "object.",
    ),
}
GRIPPER_PROMPTS = {
    "left": ("left robot gripper.", "robot gripper.", "robot hand."),
    "right": ("right robot gripper.", "robot gripper.", "robot hand."),
}
ARM_CAMERA = {"left": "cam_left", "right": "cam_right"}
# The task wrist camera is rigidly mounted behind the gripper.  This box covers
# the grasp gap, not the complete hand.  It is only a SAM visual prompt; the
# returned instance mask (never the rectangle itself) is mosaiced.
OWN_WRIST_OBJECT_BOX = [0.50, 0.58, 0.34, 0.52]
HEAD_HELD_OBJECT_BOX = {
    "left": [0.21, 0.76, 0.28, 0.38],
    "right": [0.79, 0.76, 0.28, 0.38],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run place policy with SAM 3.1 mosaics on all camera inputs."
    )
    parser.add_argument("--task", default="right_arm_place_glasses")
    parser.add_argument(
        "--sam-python",
        type=Path,
        default=Path("/home/lindenbot/miniconda3/envs/sam31/bin/python"),
    )
    parser.add_argument(
        "--sam3-source",
        type=Path,
        default=Path("/home/lindenbot/qianzhang43/place/vendor/sam3"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/home/lindenbot/qianzhang43/place/models/sam3.1_multiplex.pt"
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--confidence-threshold", type=float, default=0.10)
    parser.add_argument("--min-mask-fraction", type=float, default=0.0002)
    parser.add_argument("--max-mask-fraction", type=float, default=0.45)
    parser.add_argument("--mask-dilate", type=int, default=1)
    parser.add_argument("--mosaic-block", type=int, default=12)
    parser.add_argument("--gripper-protect-dilate", type=int, default=1)
    parser.add_argument("--max-gripper-overlap", type=float, default=0.55)
    parser.add_argument("--min-initial-association", type=float, default=0.12)
    parser.add_argument(
        "--minimum-masked-cameras",
        type=int,
        default=1,
        help="Fail closed when fewer cameras have a target mask in one step.",
    )
    parser.add_argument("--max-flow-fallbacks", type=int, default=4)
    parser.add_argument("--worker-start-timeout", type=float, default=90.0)
    parser.add_argument("--worker-request-timeout", type=float, default=45.0)
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Capture one observation and save raw/masked previews; do not move robot.",
    )
    parser.add_argument(
        "--preview-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "mask_preview",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=None,
        help="Optional directory for periodic masked frames and masks.",
    )
    parser.add_argument("--debug-every", type=int, default=0)
    parser.add_argument("--mask-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--task-arm", choices=("left", "right"), default=None, help=argparse.SUPPRESS
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.task.strip():
        raise ValueError("--task must not be empty")
    if args.mask_worker and args.task_arm not in {"left", "right"}:
        raise ValueError("The mask worker requires --task-arm left or right")
    if not 0.0 < args.confidence_threshold < 1.0:
        raise ValueError("--confidence-threshold must be between 0 and 1")
    if not 0.0 <= args.min_mask_fraction < args.max_mask_fraction <= 1.0:
        raise ValueError("Invalid min/max mask fractions")
    if args.mask_dilate < 0 or args.mosaic_block < 2:
        raise ValueError("Invalid mosaic parameters")
    if args.gripper_protect_dilate < 0:
        raise ValueError("--gripper-protect-dilate must be >= 0")
    if args.gripper_protect_dilate < args.mask_dilate:
        raise ValueError(
            "--gripper-protect-dilate must be >= --mask-dilate so mosaic "
            "dilation cannot grow back onto the gripper"
        )
    if not 0.0 <= args.max_gripper_overlap < 1.0:
        raise ValueError("--max-gripper-overlap must be in [0, 1)")
    if not 0.0 <= args.min_initial_association <= 1.0:
        raise ValueError("--min-initial-association must be in [0, 1]")
    if not 0 <= args.minimum_masked_cameras <= len(CAMERAS):
        raise ValueError("--minimum-masked-cameras must be in [0, 3]")
    if args.max_flow_fallbacks < 0:
        raise ValueError("--max-flow-fallbacks must be >= 0")


def _encode_jpeg(frame: np.ndarray) -> str:
    ok, encoded = cv2.imencode(
        ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95]
    )
    if not ok:
        raise RuntimeError("Failed to encode camera frame")
    return base64.b64encode(encoded).decode("ascii")


def _decode_image(data: str, flags: int) -> np.ndarray:
    encoded = np.frombuffer(base64.b64decode(data), dtype=np.uint8)
    image = cv2.imdecode(encoded, flags)
    if image is None:
        raise RuntimeError("Failed to decode worker image")
    return image


def _mosaic(frame: np.ndarray, mask: np.ndarray, block: int, dilate: int) -> np.ndarray:
    if mask.shape != frame.shape[:2]:
        raise ValueError(f"Mask shape {mask.shape} != frame shape {frame.shape[:2]}")
    binary = mask.astype(np.uint8) > 0
    if dilate:
        size = dilate * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        binary = cv2.dilate(binary.astype(np.uint8), kernel) > 0
    if not binary.any():
        return frame.copy()
    height, width = frame.shape[:2]
    y_blocks = np.arange(height, dtype=np.int32)[:, None] // block
    x_blocks = np.arange(width, dtype=np.int32)[None, :] // block
    levels = np.asarray([96, 128, 160, 120], dtype=np.uint8)
    gray = levels[(x_blocks * 17 + y_blocks * 29) % len(levels)]
    pattern = np.repeat(gray[..., None], 3, axis=2)
    result = frame.copy()
    result[binary] = pattern[binary]
    return result


class Sam31MaskClient:
    def __init__(self, args: argparse.Namespace, arm_type: str):
        if arm_type not in {"left_arm", "right_arm"}:
            raise ValueError(f"Unsupported placement arm type: {arm_type}")
        task_arm = arm_type.removesuffix("_arm")
        command = [
            str(args.sam_python),
            "-u",
            str(Path(__file__).resolve()),
            "--mask-worker",
            "--task",
            args.task,
            "--task-arm",
            task_arm,
            "--sam3-source",
            str(args.sam3_source),
            "--checkpoint",
            str(args.checkpoint),
            "--device",
            args.device,
            "--confidence-threshold",
            str(args.confidence_threshold),
            "--min-mask-fraction",
            str(args.min_mask_fraction),
            "--max-mask-fraction",
            str(args.max_mask_fraction),
            "--max-flow-fallbacks",
            str(args.max_flow_fallbacks),
            "--gripper-protect-dilate",
            str(args.gripper_protect_dilate),
            "--max-gripper-overlap",
            str(args.max_gripper_overlap),
            "--min-initial-association",
            str(args.min_initial_association),
        ]
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        environment["PYTHONWARNINGS"] = "ignore"
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            encoding="utf-8",
            bufsize=1,
            env=environment,
        )
        self.request_timeout = args.worker_request_timeout
        ready = self._read_response(args.worker_start_timeout)
        if ready.get("status") != "ready":
            self.close()
            raise RuntimeError(f"SAM 3.1 worker failed to start: {ready}")
        LOGGER.info(
            "SAM 3.1 mask worker ready: %.0f MiB GPU memory",
            ready.get("gpu_allocated_mb", 0.0),
        )

    def _read_response(self, timeout: float) -> dict[str, Any]:
        if self.process.stdout is None:
            raise RuntimeError("Mask worker stdout is unavailable")
        readable, _, _ = select.select([self.process.stdout], [], [], timeout)
        if not readable:
            raise TimeoutError(f"SAM 3.1 worker timed out after {timeout:.1f}s")
        line = self.process.stdout.readline()
        if not line:
            code = self.process.poll()
            raise RuntimeError(f"SAM 3.1 worker exited unexpectedly (code={code})")
        response = json.loads(line)
        if response.get("status") == "error":
            raise RuntimeError(response.get("error", "Unknown SAM 3.1 worker error"))
        return response

    def mask_images(
        self, images: dict[str, np.ndarray]
    ) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]], dict[str, np.ndarray]]:
        if set(images) != set(CAMERAS):
            raise ValueError(f"Expected cameras {CAMERAS}, got {sorted(images)}")
        request = {
            "type": "mask",
            "images": {camera: _encode_jpeg(images[camera]) for camera in CAMERAS},
        }
        if self.process.stdin is None:
            raise RuntimeError("Mask worker stdin is unavailable")
        self.process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        response = self._read_response(self.request_timeout)
        if response.get("status") != "ok":
            raise RuntimeError(f"Invalid SAM 3.1 worker response: {response}")
        masks = {
            camera: _decode_image(response["masks"][camera], cv2.IMREAD_GRAYSCALE)
            for camera in CAMERAS
        }
        metadata = response["metadata"]
        return images, metadata, masks

    def close(self) -> None:
        process = getattr(self, "process", None)
        if process is None or process.poll() is not None:
            return
        try:
            if process.stdin is not None:
                process.stdin.write('{"type":"stop"}\n')
                process.stdin.flush()
            process.wait(timeout=5)
        except Exception:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


class MaskingPolicyProxy:
    def __init__(
        self,
        delegate: Any,
        mask_client: Sam31MaskClient,
        args: argparse.Namespace,
    ):
        self.delegate = delegate
        self.mask_client = mask_client
        self.args = args
        self.step = 0
        self.last_metadata: dict[str, dict[str, Any]] = {}
        self.last_masks: dict[str, np.ndarray] = {}

    def prepare_images(
        self, images: dict[str, np.ndarray]
    ) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]], dict[str, np.ndarray]]:
        _, metadata, masks = self.mask_client.mask_images(images)
        self.last_metadata = metadata
        self.last_masks = masks
        masked_count = sum(bool(np.any(mask)) for mask in masks.values())
        missed_visible_views = [
            camera
            for camera in CAMERAS
            if (
                bool(metadata[camera].get("expected_visible", False))
                or int(metadata[camera].get("gripper_pixels", 0)) > 0
            )
            and not np.any(masks[camera])
        ]
        if missed_visible_views:
            details = ", ".join(
                f"{camera}={metadata[camera]['mode']}" for camera in missed_visible_views
            )
            raise RuntimeError(
                "Masking failed closed: the task gripper is visible but its held "
                f"object was not masked in {details}"
            )
        if masked_count < self.args.minimum_masked_cameras:
            details = ", ".join(
                f"{camera}={metadata[camera]['mode']}" for camera in CAMERAS
            )
            raise RuntimeError(
                "Masking failed closed: only "
                f"{masked_count}/{len(CAMERAS)} cameras have a target mask; {details}"
            )
        masked_images = {
            camera: _mosaic(
                images[camera],
                masks[camera],
                self.args.mosaic_block,
                self.args.mask_dilate,
            )
            for camera in CAMERAS
        }
        if (
            self.args.debug_dir is not None
            and self.args.debug_every > 0
            and self.step % self.args.debug_every == 0
        ):
            self._save_debug(masked_images, masks)
        return masked_images, metadata, masks

    def _save_debug(
        self, masked_images: dict[str, np.ndarray], masks: dict[str, np.ndarray]
    ) -> None:
        self.args.debug_dir.mkdir(parents=True, exist_ok=True)
        for camera in CAMERAS:
            stem = f"step_{self.step:04d}_{camera}"
            cv2.imwrite(str(self.args.debug_dir / f"{stem}_masked.jpg"), masked_images[camera])
            cv2.imwrite(str(self.args.debug_dir / f"{stem}_mask.png"), masks[camera])

    def infer(self, payload: dict[str, Any]) -> Any:
        images = payload.get("images")
        if not isinstance(images, dict):
            raise TypeError("Policy payload does not contain an image dictionary")
        masked_images, metadata, _ = self.prepare_images(images)
        modes = ", ".join(
            f"{camera}:{metadata[camera]['mode']}" for camera in CAMERAS
        )
        LOGGER.info("Mask step %d: %s", self.step, modes)
        masked_payload = dict(payload)
        masked_payload["images"] = masked_images
        self.step += 1
        return self.delegate.infer(masked_payload)


class MaskedSyncControlPlace(SyncControlPlace):
    def __init__(
        self,
        task_spec: TaskSpec,
        mask_client: Sam31MaskClient,
        args: argparse.Namespace,
    ):
        super().__init__(task_spec)
        self.masking_policy = MaskingPolicyProxy(self.web_policy, mask_client, args)
        self.web_policy = self.masking_policy

    def preview_once(self, output_dir: Path) -> None:
        observation = self.g2_control.get_state()
        images: dict[str, np.ndarray] = {}
        for camera, observation_key in CAMERA_OBSERVATIONS.items():
            image_data = observation[observation_key]["image_data"]
            image = cv2.imdecode(
                np.frombuffer(image_data, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if image is None:
                raise RuntimeError(f"Failed to decode {camera}")
            images[camera] = image
        output_dir.mkdir(parents=True, exist_ok=True)
        for camera in CAMERAS:
            cv2.imwrite(str(output_dir / f"{camera}_raw.jpg"), images[camera])
        try:
            masked_images, metadata, masks = self.masking_policy.prepare_images(images)
        except Exception as error:
            metadata = self.masking_policy.last_metadata
            masks = self.masking_policy.last_masks
            for camera in CAMERAS:
                mask = masks.get(
                    camera, np.zeros(images[camera].shape[:2], dtype=np.uint8)
                )
                cv2.imwrite(str(output_dir / f"{camera}_mask.png"), mask)
                failed_preview = _mosaic(
                    images[camera],
                    mask,
                    self.masking_policy.args.mosaic_block,
                    self.masking_policy.args.mask_dilate,
                )
                cv2.imwrite(
                    str(output_dir / f"{camera}_masked.jpg"), failed_preview
                )
            diagnostic = dict(metadata)
            diagnostic["_error"] = str(error)
            with (output_dir / "metadata.json").open(
                "w", encoding="utf-8"
            ) as stream:
                json.dump(diagnostic, stream, indent=2, ensure_ascii=False)
            LOGGER.error("Failed preview diagnostics saved to %s", output_dir)
            raise
        for camera in CAMERAS:
            cv2.imwrite(str(output_dir / f"{camera}_masked.jpg"), masked_images[camera])
            cv2.imwrite(str(output_dir / f"{camera}_mask.png"), masks[camera])
        with (output_dir / "metadata.json").open("w", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2, ensure_ascii=False)
        LOGGER.info("Preview saved to %s; no robot motion was executed", output_dir)


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    y, x = np.nonzero(mask)
    if not len(x):
        return None
    return int(x.min()), int(y.min()), int(x.max() + 1), int(y.max() + 1)


def _bbox_iou(
    first: tuple[int, int, int, int], second: tuple[int, int, int, int]
) -> float:
    x0 = max(first[0], second[0])
    y0 = max(first[1], second[1])
    x1 = min(first[2], second[2])
    y1 = min(first[3], second[3])
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    first_area = max(1, first[2] - first[0]) * max(1, first[3] - first[1])
    second_area = max(1, second[2] - second[0]) * max(1, second[3] - second[1])
    return intersection / max(1, first_area + second_area - intersection)


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    binary = mask.astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        return binary
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = int(areas.max())
    minimum = max(12, int(largest * 0.015))
    keep = np.flatnonzero(areas >= minimum) + 1
    return np.isin(labels, keep).astype(np.uint8)


class CameraTrack:
    def __init__(self) -> None:
        self.gray: np.ndarray | None = None
        self.mask: np.ndarray | None = None
        self.bbox: tuple[int, int, int, int] | None = None
        self.flow_fallbacks = 0

    def choose(
        self,
        frame: np.ndarray,
        candidates: list[tuple[np.ndarray, float, float]],
        max_flow_fallbacks: int,
    ) -> tuple[np.ndarray, str, float, float]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if candidates and self.mask is not None:
            previous_pixels = max(1, int(self.mask.sum()))
            # A held object cannot change its image area by several times in a
            # single control step.  This rejects a nearby sofa/table candidate
            # without tying tracking to any object appearance.
            size_consistent = [
                item
                for item in candidates
                if 0.25 <= int(item[0].sum()) / previous_pixels <= 4.0
            ]
            candidates = size_consistent
        if candidates:
            selected_mask, selected_score, selected_association = max(
                candidates,
                key=lambda item: self._candidate_score(
                    item[0], item[1], item[2], frame.shape
                ),
            )
            selected_mask = _clean_mask(selected_mask)
            selected_bbox = _bbox_from_mask(selected_mask)
            plausible = selected_bbox is not None
            if plausible and self.bbox is not None and selected_association < 0.20:
                old_center = (
                    (self.bbox[0] + self.bbox[2]) / 2,
                    (self.bbox[1] + self.bbox[3]) / 2,
                )
                new_center = (
                    (selected_bbox[0] + selected_bbox[2]) / 2,
                    (selected_bbox[1] + selected_bbox[3]) / 2,
                )
                diagonal = max(1.0, float(np.hypot(frame.shape[1], frame.shape[0])))
                center_distance = float(
                    np.hypot(
                        new_center[0] - old_center[0], new_center[1] - old_center[1]
                    )
                    / diagonal
                )
                plausible = not (
                    _bbox_iou(self.bbox, selected_bbox) < 0.01
                    and center_distance > 0.25
                )
            if plausible:
                self.gray = gray
                self.mask = selected_mask
                self.bbox = selected_bbox
                self.flow_fallbacks = 0
                return (
                    selected_mask,
                    "sam",
                    float(selected_score),
                    float(selected_association),
                )

        if (
            self.gray is not None
            and self.mask is not None
            and self.flow_fallbacks < max_flow_fallbacks
        ):
            tracked = self._track_mask(self.gray, gray, self.mask)
            if tracked is not None:
                self.gray = gray
                self.mask = tracked
                self.bbox = _bbox_from_mask(tracked)
                self.flow_fallbacks += 1
                return tracked, "flow", 0.0, 0.0

        self.gray = gray
        self.mask = None
        self.bbox = None
        self.flow_fallbacks += 1
        return np.zeros(frame.shape[:2], dtype=np.uint8), "absent", 0.0, 0.0

    def _candidate_score(
        self,
        mask: np.ndarray,
        semantic_score: float,
        association_score: float,
        shape: tuple[int, ...],
    ) -> float:
        bbox = _bbox_from_mask(mask)
        if bbox is None:
            return -1e9
        quality = float(semantic_score) + 2.8 * float(association_score)
        if self.bbox is None:
            area_fraction = float(mask.mean())
            return quality - max(0.0, area_fraction - 0.12)
        quality += 2.0 * _bbox_iou(self.bbox, bbox)
        old_center = ((self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2)
        new_center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        diagonal = max(1.0, float(np.hypot(shape[1], shape[0])))
        distance = np.hypot(new_center[0] - old_center[0], new_center[1] - old_center[1])
        quality += max(0.0, 1.0 - distance / (0.35 * diagonal))
        return quality

    @staticmethod
    def _track_mask(
        previous_gray: np.ndarray, current_gray: np.ndarray, previous_mask: np.ndarray
    ) -> np.ndarray | None:
        points = cv2.goodFeaturesToTrack(
            previous_gray,
            maxCorners=120,
            qualityLevel=0.01,
            minDistance=5,
            mask=(previous_mask > 0).astype(np.uint8) * 255,
        )
        if points is None or len(points) < 6:
            return None
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            previous_gray,
            current_gray,
            points,
            None,
            winSize=(31, 31),
            maxLevel=3,
        )
        if next_points is None or status is None:
            return None
        valid = status.reshape(-1) > 0
        if int(valid.sum()) < 6:
            return None
        transform, inliers = cv2.estimateAffinePartial2D(
            points.reshape(-1, 2)[valid],
            next_points.reshape(-1, 2)[valid],
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
        )
        if transform is None or inliers is None or int(inliers.sum()) < 4:
            return None
        tracked = cv2.warpAffine(
            previous_mask.astype(np.uint8),
            transform,
            (current_gray.shape[1], current_gray.shape[0]),
            flags=cv2.INTER_NEAREST,
        )
        if not tracked.any():
            return None
        return tracked


class Sam31FrameMasker:
    def __init__(self, args: argparse.Namespace):
        args.sam3_source = args.sam3_source.expanduser().resolve()
        args.checkpoint = args.checkpoint.expanduser().resolve()
        if not args.sam3_source.is_dir():
            raise FileNotFoundError(args.sam3_source)
        if not args.checkpoint.is_file():
            raise FileNotFoundError(args.checkpoint)
        sys.path.insert(0, str(args.sam3_source))
        import torch
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model.sam3_multiplex_detector import Sam3MultiplexDetector
        from sam3.model.vl_combiner import SAM3VLBackboneTri
        from sam3.model_builder import (
            _create_dot_product_scoring,
            _create_geometry_encoder,
            _create_multiplex_tri_backbone,
            _create_sam3_transformer,
            _create_segmentation_head,
            _create_text_encoder,
        )

        if not torch.cuda.is_available() or not args.device.startswith("cuda"):
            raise RuntimeError("SAM 3.1 live masking requires a CUDA device")
        checkpoint_data = torch.load(
            args.checkpoint, map_location="cpu", weights_only=True, mmap=True
        )
        checkpoint_state = checkpoint_data.get("model", checkpoint_data)
        tri_neck = _create_multiplex_tri_backbone(
            compile_mode=None, use_fa3=False, use_rope_real=False
        )
        bpe_path = args.sam3_source / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
        detector = Sam3MultiplexDetector(
            num_feature_levels=1,
            backbone=SAM3VLBackboneTri(
                scalp=0,
                visual=tri_neck,
                text=_create_text_encoder(str(bpe_path)),
            ),
            transformer=_create_sam3_transformer(use_fa3=False),
            segmentation_head=_create_segmentation_head(use_fa3=False),
            semantic_segmentation_head=None,
            input_geometry_encoder=_create_geometry_encoder(),
            use_early_fusion=True,
            use_dot_prod_scoring=True,
            dot_prod_scoring=_create_dot_product_scoring(),
            supervise_joint_box_scores=True,
            is_multiplex=True,
        )
        detector_state = {
            key.removeprefix("detector."): value
            for key, value in checkpoint_state.items()
            if key.startswith("detector.")
        }
        missing, unexpected = detector.load_state_dict(
            detector_state, strict=False, assign=True
        )
        if missing or unexpected:
            raise RuntimeError(
                f"SAM detector checkpoint mismatch: missing={missing[:5]}, "
                f"unexpected={unexpected[:5]}"
            )
        detector.to(args.device).eval()
        self.torch = torch
        self.processor = Sam3Processor(
            detector,
            device=args.device,
            confidence_threshold=args.confidence_threshold,
        )
        self.device = args.device
        if args.task_arm not in HELD_OBJECT_PROMPTS:
            raise ValueError(f"Invalid task arm for mask worker: {args.task_arm}")
        self.task_arm = args.task_arm
        self.prompts = HELD_OBJECT_PROMPTS[self.task_arm]
        self.gripper_prompts = GRIPPER_PROMPTS[self.task_arm]
        self.min_fraction = args.min_mask_fraction
        self.max_fraction = args.max_mask_fraction
        self.max_flow_fallbacks = args.max_flow_fallbacks
        self.gripper_protect_dilate = args.gripper_protect_dilate
        self.max_gripper_overlap = args.max_gripper_overlap
        self.min_initial_association = args.min_initial_association
        self.tracks = {camera: CameraTrack() for camera in CAMERAS}

    def mask_frame(self, camera: str, frame: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        from PIL import Image

        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        valid_candidates: list[tuple[np.ndarray, float, float]] = []
        gripper_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        gripper_score = 0.0
        used_prompt = ""
        used_gripper_prompt = ""
        with self.torch.inference_mode(), self.torch.autocast(
            "cuda", dtype=self.torch.bfloat16
        ):
            state = self.processor.set_image(pil_image)
            camera_gripper_prompts = self._gripper_prompts_for_camera(camera)
            for prompt_index, gripper_prompt in enumerate(camera_gripper_prompts):
                if prompt_index:
                    self.processor.reset_all_prompts(state)
                state = self.processor.set_text_prompt(gripper_prompt, state)
                gripper_mask, gripper_score = self._select_gripper_mask(
                    state, frame.shape[:2], camera
                )
                if gripper_mask.any():
                    used_gripper_prompt = gripper_prompt
                    break

            # The own wrist always looks through the task gripper's grasp gap.
            # A visual box gives SAM a location cue that does not depend on the
            # held item's category, colour, text or orientation.  It is tried
            # before generic text to avoid selecting furniture in the scene.
            if camera == ARM_CAMERA[self.task_arm]:
                self.processor.reset_all_prompts(state)
                state = self.processor.add_geometric_prompt(
                    OWN_WRIST_OBJECT_BOX, True, state
                )
                valid_candidates = self._valid_candidates(
                    state,
                    frame.shape[:2],
                    gripper_mask,
                    camera,
                    geometric_anchor=True,
                )
                if valid_candidates:
                    used_prompt = "visual grasp-gap box"
            elif camera == "cam_head":
                self.processor.reset_all_prompts(state)
                # First ask for independent object instances.  The strict head
                # grasp-zone gate below rejects the wrist camera, gripper base
                # and furniture, while remaining agnostic to item appearance.
                state = self.processor.set_text_prompt("object.", state)
                valid_candidates = self._valid_candidates(
                    state,
                    frame.shape[:2],
                    gripper_mask,
                    camera,
                    geometric_anchor=True,
                )
                if valid_candidates:
                    used_prompt = "object + head grasp-zone gate"

            if camera == "cam_head" and not valid_candidates:
                self.processor.reset_all_prompts(state)
                # Position alone can make a head-view box select the sofa or
                # table behind the hand.  Keep the category generic, but combine
                # the location with the explicit held-object relationship.
                state = self.processor.set_text_prompt(self.prompts[0], state)
                state = self.processor.add_geometric_prompt(
                    HEAD_HELD_OBJECT_BOX[self.task_arm], True, state
                )
                valid_candidates = self._valid_candidates(
                    state,
                    frame.shape[:2],
                    gripper_mask,
                    camera,
                    geometric_anchor=True,
                )
                if valid_candidates:
                    used_prompt = "held-object text + head task-arm box"
            for prompt in self.prompts:
                if valid_candidates:
                    break
                self.processor.reset_all_prompts(state)
                state = self.processor.set_text_prompt(prompt, state)
                valid_candidates = self._valid_candidates(
                    state, frame.shape[:2], gripper_mask, camera
                )
                if valid_candidates:
                    used_prompt = prompt
                    break
        mask, mode, score, association = self.tracks[camera].choose(
            frame, valid_candidates, self.max_flow_fallbacks
        )
        candidate_summary = []
        for candidate_mask, candidate_score, candidate_association in valid_candidates:
            bbox = _bbox_from_mask(candidate_mask)
            candidate_summary.append(
                {
                    "score": round(float(candidate_score), 4),
                    "association": round(float(candidate_association), 4),
                    "pixels": int(candidate_mask.sum()),
                    "bbox": list(bbox) if bbox is not None else None,
                }
            )
        joint_count = 0
        if mask.any() and camera == ARM_CAMERA[self.task_arm]:
            mask, joint_count = self._protect_own_wrist_joints(frame, mask)
        if mask.any() and gripper_mask.any():
            mask = self._remove_gripper(mask, gripper_mask)
        if mask.any():
            self.tracks[camera].mask = mask
            self.tracks[camera].bbox = _bbox_from_mask(mask)
        elif mode != "absent":
            mode = "gripper_rejected"
            self.tracks[camera].mask = None
            self.tracks[camera].bbox = None
        return mask, {
            "mode": mode,
            "score": round(score, 4),
            "association": round(association, 4),
            "candidate_count": len(valid_candidates),
            "candidates": candidate_summary,
            "mask_pixels": int(mask.sum()),
            "gripper_pixels": int(gripper_mask.sum()),
            "gripper_score": round(gripper_score, 4),
            "gripper_prompt": used_gripper_prompt,
            "protected_joint_count": joint_count,
            "task_arm": self.task_arm,
            "prompt": used_prompt,
            "expected_visible": camera == ARM_CAMERA[self.task_arm]
            or self.tracks[camera].bbox is not None,
        }

    def _gripper_prompts_for_camera(self, camera: str) -> tuple[str, ...]:
        """Avoid confusing the opposite wrist's own hand with the task hand."""
        if camera == ARM_CAMERA[self.task_arm]:
            return self.gripper_prompts
        # Handedness is the only safe semantic cue in head/opposite-wrist views.
        # Generic "robot hand" selected the empty left hand as the right task
        # hand in real preview data and consequently mosaiced the table.
        return self.gripper_prompts[:1]

    @staticmethod
    def _protect_own_wrist_joints(
        frame: np.ndarray, object_mask: np.ndarray
    ) -> tuple[np.ndarray, int]:
        """Remove only visually confirmed circular gripper joints.

        The wrist camera is rigid, so the six joint neighbourhoods are stable.
        Hough confirmation is required before removing pixels: when a held
        object occludes a joint, a fixed exclusion hole is therefore not cut
        into the object mask.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 1.5)
        height, width = gray.shape
        scale = min(height, width)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(18, int(round(scale * 0.024))),
            param1=120,
            param2=32,
            minRadius=max(8, int(round(scale * 0.014))),
            maxRadius=max(18, int(round(scale * 0.050))),
        )
        if circles is None:
            return object_mask, 0

        expected = (
            (0.36, 0.62),
            (0.30, 0.74),
            (0.36, 0.80),
            (0.62, 0.61),
            (0.69, 0.74),
            (0.63, 0.80),
        )
        protection = np.zeros_like(object_mask, dtype=np.uint8)
        accepted = 0
        for circle_x, circle_y, radius in circles[0]:
            normalized_x = float(circle_x / width)
            normalized_y = float(circle_y / height)
            if not any(
                np.hypot(normalized_x - expected_x, normalized_y - expected_y)
                <= 0.075
                for expected_x, expected_y in expected
            ):
                continue
            # Only the lower linkage joints caused false object components in
            # real previews.  Upper circles can sit directly behind wide held
            # objects, so protecting those would carve holes in the object.
            if normalized_y < 0.68:
                continue
            cv2.circle(
                protection,
                (int(round(circle_x)), int(round(circle_y))),
                max(1, int(round(radius * 1.30))),
                1,
                thickness=-1,
            )
            accepted += 1
        if not accepted:
            return object_mask, 0
        protected = np.logical_and(object_mask > 0, protection == 0).astype(np.uint8)
        return (_clean_mask(protected) if protected.any() else protected), accepted

    def _select_gripper_mask(
        self, state: dict[str, Any], shape: tuple[int, int], camera: str
    ) -> tuple[np.ndarray, float]:
        masks = state["masks"][:, 0].detach().cpu().numpy().astype(np.uint8)
        scores = state["scores"].detach().float().cpu().numpy()
        ranked: list[tuple[float, np.ndarray, float]] = []
        for mask, score in zip(masks, scores, strict=True):
            if mask.shape != shape:
                continue
            cleaned = _clean_mask(mask)
            fraction = float(cleaned.mean())
            if not self.min_fraction <= fraction <= 0.65:
                continue
            rank = (
                float(score)
                + 0.20 * self._spatial_prior(cleaned, camera, shape)
                - 0.35 * max(0.0, fraction - 0.22)
            )
            ranked.append((rank, cleaned, float(score)))
        if not ranked:
            return np.zeros(shape, dtype=np.uint8), 0.0
        _, mask, score = max(ranked, key=lambda item: item[0])
        return mask, score

    def _valid_candidates(
        self,
        state: dict[str, Any],
        shape: tuple[int, int],
        gripper_mask: np.ndarray,
        camera: str,
        geometric_anchor: bool = False,
    ) -> list[tuple[np.ndarray, float, float]]:
        masks = state["masks"][:, 0].detach().cpu().numpy().astype(np.uint8)
        scores = state["scores"].detach().float().cpu().numpy()
        valid: list[tuple[np.ndarray, float, float]] = []
        track_is_locked = self.tracks[camera].bbox is not None
        for mask, score in zip(masks, scores, strict=True):
            if mask.shape != shape:
                continue
            raw_mask = _clean_mask(mask)
            raw_pixels = int(raw_mask.sum())
            if raw_pixels == 0:
                continue
            raw_fraction = raw_pixels / float(shape[0] * shape[1])
            if not self.min_fraction <= raw_fraction <= self.max_fraction:
                continue
            # Outside the task wrist, a first lock still has to be tied to the
            # task-side gripper.  In the task wrist the rigid camera mount gives
            # us an appearance-independent grasp-gap prior instead.
            anchored_association = 0.0
            if not gripper_mask.any() and not track_is_locked:
                if geometric_anchor:
                    anchored_association = self._geometric_initial_score(
                        raw_mask, camera, shape
                    )
                elif camera == ARM_CAMERA[self.task_arm]:
                    anchored_association = self._own_wrist_initial_score(
                        raw_mask, shape
                    )
                else:
                    continue
                if anchored_association < 0.52:
                    continue
            if gripper_mask.any():
                gripper_overlap = float(
                    np.logical_and(raw_mask > 0, gripper_mask > 0).sum() / raw_pixels
                )
                if gripper_overlap > self.max_gripper_overlap:
                    continue
            association = self._association_score(
                raw_mask, gripper_mask, camera, shape
            )
            if anchored_association:
                association = anchored_association
            if (
                gripper_mask.any()
                and not track_is_locked
                and association < self.min_initial_association
            ):
                continue
            protected = self._remove_gripper(raw_mask, gripper_mask)
            protected_pixels = int(protected.sum())
            if protected_pixels < max(1, int(raw_pixels * 0.35)):
                continue
            fraction = protected_pixels / float(shape[0] * shape[1])
            if self.min_fraction <= fraction <= self.max_fraction:
                valid.append((protected, float(score), association))
        return valid

    def _geometric_initial_score(
        self, mask: np.ndarray, camera: str, shape: tuple[int, int]
    ) -> float:
        if camera == ARM_CAMERA[self.task_arm]:
            return self._own_wrist_initial_score(mask, shape)
        if camera != "cam_head":
            return 0.0
        bbox = _bbox_from_mask(mask)
        if bbox is None:
            return 0.0
        height, width = shape
        x0, y0, x1, y1 = bbox
        normalized_x0, normalized_x1 = x0 / width, x1 / width
        normalized_y0, normalized_y1 = y0 / height, y1 / height
        # The wrist camera and gripper base lie outside/below this envelope in
        # the rigid head-camera frame. Rejecting the whole candidate is safer
        # than clipping it, which could silently keep a partial robot mask.
        if self.task_arm == "left":
            if normalized_x0 < 0.11 or normalized_x1 > 0.35:
                return 0.0
        else:
            if normalized_x0 < 0.65 or normalized_x1 > 0.89:
                return 0.0
        # At the initial place pose the held item is above the front plate; the
        # plate/camera begins below roughly 82% image height.  A candidate that
        # crosses this contact line has merged the robot into the object.  This
        # gate is only used before CameraTrack is locked.
        if normalized_y0 < 0.38 or normalized_y1 > 0.82:
            return 0.0
        center_x = (x0 + x1) / (2.0 * width)
        center_y = (y0 + y1) / (2.0 * height)
        expected_x = 0.21 if self.task_arm == "left" else 0.79
        expected_y = 0.70
        distance = float(np.hypot(center_x - expected_x, center_y - expected_y))
        center_score = max(0.0, 1.0 - distance / 0.36)

        if self.task_arm == "left":
            core_x0, core_x1 = int(0.08 * width), int(0.34 * width)
        else:
            core_x0, core_x1 = int(0.66 * width), int(0.92 * width)
        core_y0, core_y1 = int(0.58 * height), int(0.82 * height)
        core_pixels = int(mask[core_y0:core_y1, core_x0:core_x1].sum())
        if core_pixels == 0:
            return 0.0
        core_coverage = core_pixels / max(
            1, (core_y1 - core_y0) * (core_x1 - core_x0)
        )
        fraction = float(mask.mean())
        if fraction > 0.28 or y1 / height < 0.58:
            return 0.0
        return float(
            np.clip(
                0.68 * center_score
                + 0.32 * min(1.0, core_coverage / 0.10),
                0.0,
                1.0,
            )
        )

    @staticmethod
    def _own_wrist_initial_score(
        mask: np.ndarray, shape: tuple[int, int]
    ) -> float:
        """Score an instance by whether it occupies the fixed grasp gap.

        This is deliberately based on geometry rather than appearance.  The
        constraints reject the table/background above the gripper and the robot
        housing at the lower image edge, while accepting tall bottles, flat
        boxes, tissue packs, glasses and irregular objects held at any angle.
        """
        bbox = _bbox_from_mask(mask)
        if bbox is None:
            return 0.0
        height, width = shape
        x0, y0, x1, y1 = bbox
        nx0, nx1 = x0 / width, x1 / width
        ny0, ny1 = y0 / height, y1 / height
        center_x = (nx0 + nx1) * 0.5
        center_y = (ny0 + ny1) * 0.5

        # Every grasped object must cross this small region between the fingers.
        core_x0, core_x1 = int(0.40 * width), int(0.60 * width)
        core_y0, core_y1 = int(0.42 * height), int(0.78 * height)
        core_pixels = int(mask[core_y0:core_y1, core_x0:core_x1].sum())
        object_pixels = max(1, int(mask.sum()))
        core_coverage = core_pixels / max(
            1, (core_y1 - core_y0) * (core_x1 - core_x0)
        )
        object_in_core = core_pixels / object_pixels

        if core_pixels == 0 or not (0.18 <= center_x <= 0.82):
            return 0.0
        if ny1 < 0.54 or ny0 > 0.82 or ny1 > 0.97:
            return 0.0
        if nx0 <= 0.01 or nx1 >= 0.99:
            return 0.0

        center_distance = float(np.hypot(center_x - 0.50, center_y - 0.59))
        center_score = max(0.0, 1.0 - center_distance / 0.48)
        lower_reach = float(np.clip((ny1 - 0.54) / 0.25, 0.0, 1.0))
        return float(
            np.clip(
                0.46 * center_score
                + 0.26 * min(1.0, core_coverage / 0.16)
                + 0.18 * min(1.0, object_in_core / 0.10)
                + 0.10 * lower_reach,
                0.0,
                1.0,
            )
        )

    def _association_score(
        self,
        object_mask: np.ndarray,
        gripper_mask: np.ndarray,
        camera: str,
        shape: tuple[int, int],
    ) -> float:
        spatial = self._spatial_prior(object_mask, camera, shape)
        if not gripper_mask.any():
            return 0.25 * spatial
        object_pixels = max(1, int(object_mask.sum()))
        radius = max(4, int(round(min(shape) * 0.035)))
        diameter = radius * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
        near_gripper = cv2.dilate(gripper_mask.astype(np.uint8), kernel) > 0
        contact_fraction = float(
            np.logical_and(object_mask > 0, near_gripper).sum() / object_pixels
        )
        contact = min(1.0, contact_fraction / 0.08)
        object_box = _bbox_from_mask(object_mask)
        gripper_box = _bbox_from_mask(gripper_mask)
        proximity = 0.0
        if object_box is not None and gripper_box is not None:
            dx = max(
                gripper_box[0] - object_box[2],
                object_box[0] - gripper_box[2],
                0,
            )
            dy = max(
                gripper_box[1] - object_box[3],
                object_box[1] - gripper_box[3],
                0,
            )
            diagonal = max(1.0, float(np.hypot(shape[1], shape[0])))
            proximity = max(0.0, 1.0 - float(np.hypot(dx, dy)) / (0.22 * diagonal))
        overlap = float(
            np.logical_and(object_mask > 0, gripper_mask > 0).sum() / object_pixels
        )
        return float(
            np.clip(
                0.65 * contact + 0.27 * proximity + 0.08 * spatial - overlap,
                0.0,
                1.0,
            )
        )

    def _spatial_prior(
        self, mask: np.ndarray, camera: str, shape: tuple[int, int]
    ) -> float:
        bbox = _bbox_from_mask(mask)
        if bbox is None:
            return 0.0
        center_x = (bbox[0] + bbox[2]) / (2.0 * shape[1])
        center_y = (bbox[1] + bbox[3]) / (2.0 * shape[0])
        if camera == "cam_head":
            expected_x = 0.28 if self.task_arm == "left" else 0.72
            expected_y = 0.62
        elif camera == ARM_CAMERA[self.task_arm]:
            expected_x, expected_y = 0.50, 0.48
        else:
            expected_x, expected_y = 0.50, 0.55
        distance = float(np.hypot(center_x - expected_x, center_y - expected_y))
        return max(0.0, 1.0 - distance / 0.75)

    def _remove_gripper(
        self, object_mask: np.ndarray, gripper_mask: np.ndarray
    ) -> np.ndarray:
        protected = gripper_mask.astype(np.uint8)
        if protected.any() and self.gripper_protect_dilate:
            diameter = self.gripper_protect_dilate * 2 + 1
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (diameter, diameter)
            )
            protected = cv2.dilate(protected, kernel)
        result = np.logical_and(object_mask > 0, protected == 0).astype(np.uint8)
        return _clean_mask(result) if result.any() else result


def worker_main(args: argparse.Namespace) -> int:
    warnings.filterwarnings("ignore")
    try:
        validate_args(args)
        masker = Sam31FrameMasker(args)
        print(
            json.dumps(
                {
                    "status": "ready",
                    "gpu_allocated_mb": masker.torch.cuda.memory_allocated()
                    / 1024**2,
                }
            ),
            flush=True,
        )
    except Exception as error:
        print(json.dumps({"status": "error", "error": str(error)}), flush=True)
        traceback.print_exc(file=sys.stderr)
        return 1

    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("type") == "stop":
                return 0
            if request.get("type") != "mask":
                raise ValueError(f"Unknown request type: {request.get('type')}")
            encoded_masks: dict[str, str] = {}
            metadata: dict[str, dict[str, Any]] = {}
            for camera in CAMERAS:
                frame = _decode_image(request["images"][camera], cv2.IMREAD_COLOR)
                mask, metadata[camera] = masker.mask_frame(camera, frame)
                ok, encoded = cv2.imencode(".png", mask.astype(np.uint8) * 255)
                if not ok:
                    raise RuntimeError(f"Failed to encode {camera} mask")
                encoded_masks[camera] = base64.b64encode(encoded).decode("ascii")
            print(
                json.dumps(
                    {"status": "ok", "masks": encoded_masks, "metadata": metadata},
                    separators=(",", ":"),
                ),
                flush=True,
            )
        except Exception as error:
            print(json.dumps({"status": "error", "error": str(error)}), flush=True)
            traceback.print_exc(file=sys.stderr)
    return 0


def main(args: argparse.Namespace) -> int:
    validate_args(args)
    if ROOT_DIR is None or TaskConfigLoader is None:
        raise RuntimeError("Robot control modules are unavailable in the mask worker")
    loader = TaskConfigLoader([ROOT_DIR / "config" / "config_V2_0.json"])
    task_spec = loader.resolve_task(args.task)
    mask_client = Sam31MaskClient(args, task_spec.arm_type)
    try:
        controller = MaskedSyncControlPlace(task_spec, mask_client, args)
        if args.preview_only:
            controller.preview_once(args.preview_dir)
            return 0
        return 0 if controller.run() else 1
    finally:
        mask_client.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parsed_args = parse_args()
    if parsed_args.mask_worker:
        sys.exit(worker_main(parsed_args))
    sys.exit(main(parsed_args))
