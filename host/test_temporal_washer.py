"""Offline history transport tests; all robot connections are substitutes."""
import copy
import unittest
from unittest.mock import Mock

import numpy as np

from clothes_robot import model_observation
from clothes_task import CLOSE_WASHER_TASK_ID, MEM_CLOSE_WASHER_TASK_ID, load_task_spec, synthetic_observation
from temporal_washer import TemporalRobotConnection, policy_payload, select_history
from test_clothes_robot import FakeRobot


class TemporalWasherTests(unittest.TestCase):
    def test_synthetic_temporal_payload_has_both_past_frames_in_rgb(self):
        for task_id in (CLOSE_WASHER_TASK_ID, MEM_CLOSE_WASHER_TASK_ID):
            with self.subTest(task_id=task_id):
                spec = load_task_spec(task_id=task_id)
                raw = synthetic_observation(spec)
                packet = policy_payload(raw)
                stamp = packet['observation/timestamp_ns']
                self.assertEqual([sample['observation/timestamp_ns'] for sample in packet['history_samples']],
                                 [stamp - 1_000_000_000, stamp - 500_000_000])
                np.testing.assert_array_equal(packet['observation/image'], raw['images']['cam_head'][..., ::-1])
                self.assertEqual(packet['observation/joint_position'].shape, (16,))

    def test_history_uses_past_sensor_times_and_masks_missing_slots(self):
        samples = [{'timestamps': {'states': stamp}} for stamp in (1_000_000_000, 1_510_000_000, 2_100_000_000)]
        selected = select_history(samples, 2_000_000_000)
        self.assertEqual([sample['timestamps']['states'] for sample in selected], [1_000_000_000, 1_510_000_000])
        self.assertEqual(select_history(samples, 900_000_000), [])

    def test_history_reader_is_separate_and_only_control_connection_sends_actions(self):
        current = FakeRobot().observe()
        current['timestamps']['head'] = current['timestamps']['states']
        stamp = current['timestamps']['states']
        control = Mock()
        reader = Mock()
        control.observe.return_value = current
        factory = Mock(side_effect=[control, reader])
        robot = TemporalRobotConnection({'host': '192.0.2.10'}, connection_factory=factory, start_reader=False)
        try:
            for offset in (-1_000_000_000, -500_000_000):
                past = copy.deepcopy(current)
                past['timestamps'] = {'states': stamp + offset, 'head': stamp + offset}
                robot.record(past)
            observation = model_observation(robot.observe(), 'close the door')
            self.assertEqual(len(observation['history_samples']), 2)
            self.assertEqual(len(policy_payload(observation)['history_samples']), 2)
            command = {'left_arm': 'fixture command'}
            robot.execute(command)
            control.execute.assert_called_once_with(command)
            reader.execute.assert_not_called()
        finally:
            robot.close()
        control.close.assert_called_once()
        reader.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
