"""Read sparse camera/state history for the temporal washer model."""
from collections import deque
import threading

import cv2
import numpy as np


HISTORY_SECONDS = (-1.0, -0.5)
HISTORY_TOLERANCE_S = 0.15


def timestamp_ns(raw):
    return int(raw['timestamps'].get('head', raw['timestamps']['states']))


def select_history(samples, current_stamp):
    candidates = [sample for sample in samples if timestamp_ns(sample) < current_stamp]
    selected = {}
    for offset in HISTORY_SECONDS:
        target = current_stamp + round(offset * 1e9)
        if not candidates:
            continue
        sample = min(candidates, key=lambda value: abs(timestamp_ns(value) - target))
        stamp = timestamp_ns(sample)
        if abs(stamp - target) <= round(HISTORY_TOLERANCE_S * 1e9):
            selected[stamp] = sample
    return [selected[stamp] for stamp in sorted(selected)]


class TemporalRobotConnection:
    """Use a separate read-only connection to collect history while actions run."""
    def __init__(self, robot, *, connection_factory, start_reader=True):
        self.control = connection_factory(robot)
        try:
            self.reader = connection_factory(robot)
        except BaseException:
            self.control.close()
            raise
        self.samples = deque(maxlen=40)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._read, daemon=True, name='washer-observation-history')
        if start_reader:
            self.thread.start()

    def record(self, raw):
        with self.lock:
            if self.samples and timestamp_ns(raw) < timestamp_ns(self.samples[-1]):
                self.samples.clear()
            if self.samples and timestamp_ns(raw) == timestamp_ns(self.samples[-1]):
                self.samples[-1] = raw
            else:
                self.samples.append(raw)

    def _read(self):
        try:
            while not self.stop.is_set():
                self.record(self.reader.observe())
                self.stop.wait(0.1)
        except Exception as exc:
            if not self.stop.is_set():
                self.error = exc

    def observe(self, include_images=True):
        if include_images and self.error is not None:
            raise RuntimeError(f'历史观测读取失败：{self.error}') from self.error
        raw = self.control.observe(include_images=include_images)
        if include_images:
            with self.lock:
                history = select_history(tuple(self.samples), timestamp_ns(raw))
            raw = {**raw, 'temporal_history': history}
        return raw

    def execute(self, command):
        return self.control.execute(command)

    def close(self):
        self.stop.set()
        try:
            self.reader.close()
        finally:
            if self.thread.ident is not None:
                self.thread.join(timeout=6)
            self.control.close()


def policy_payload(observation):
    """Use the same RGB/raw field convention as 76's MEM washer client."""
    images = observation['images']
    result = {
        'observation/image': cv2.cvtColor(np.asarray(images['cam_head']), cv2.COLOR_BGR2RGB),
        'observation/wrist_image_left': cv2.cvtColor(np.asarray(images['cam_left']), cv2.COLOR_BGR2RGB),
        'observation/wrist_image_right': cv2.cvtColor(np.asarray(images['cam_right']), cv2.COLOR_BGR2RGB),
        'observation/joint_position': np.asarray(observation['state'], dtype=np.float32),
        'observation/timestamp_ns': int(observation['observation/timestamp_ns']),
        'prompt': observation['prompt'],
    }
    result['history_samples'] = [policy_payload(sample) for sample in observation.get('history_samples', ())]
    return result
