"""Regression tests for inference-only operation; never use a robot connection."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import clothes_task
import intergrated_test_server as server


class ClothesValidationTests(unittest.TestCase):
    def test_synthetic_payload_uses_exact_operator_prompt_and_no_live_sensors(self):
        prompts = {
            "right_arm_hang_clothes": "Use the right arm to hang the clothes on the clothes rack using the hanger.",
            "left_arm_pick_blue_clothes_from_sofa": "Use both arms to pick up the light blue clothes from the sofa.",
            clothes_task.TASK_ID: "Use the right arm to remove  the clothes and hanger from the clothes rack.",
            clothes_task.PUT_WASHER_TASK_ID: "Use the left arm to put the light blue clothes into the washing machine",
            clothes_task.CLOSE_WASHER_TASK_ID: "Use both arms to close the washing machine door.",
            clothes_task.MEM_CLOSE_WASHER_TASK_ID: "Use both arms to close the washing machine door.",
        }
        for task_id, prompt in prompts.items():
            with self.subTest(task_id=task_id):
                spec = clothes_task.load_task_spec(task_id=task_id)
                observation = clothes_task.synthetic_observation(spec)
                self.assertEqual(observation["prompt"], prompt)
                self.assertEqual(observation["state"].shape, (16,))
                self.assertEqual(set(observation["images"]), {"cam_head", "cam_left", "cam_right"})
                self.assertTrue(all(image.shape == (224, 224, 3) and image.dtype == np.uint8
                                    for image in observation["images"].values()))

    def test_validation_refuses_a_nonlocal_model_host_before_connecting(self):
        data = json.loads(clothes_task.CONFIG_PATH.read_text())
        data[clothes_task.TASK_ID]["policy_host"] = "192.0.2.42"
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "tasks.json"
            config.write_text(json.dumps(data))
            with patch.object(clothes_task, "open_policy") as connect:
                with self.assertRaisesRegex(ValueError, "仅连接本机"):
                    clothes_task.run_validation(config_path=config)
                connect.assert_not_called()

    def test_wrong_model_at_the_configured_port_is_rejected(self):
        spec = clothes_task.load_task_spec()
        metadata = {"task_id": clothes_task.TASK_ID, "model_config": spec["policy_config"],
                    "asset_id": spec["asset_id"], "checkpoint": spec["checkpoint"],
                    "prompt": spec["prompt"], "action_space": "joint_absolute",
                    "action_shape": [30, 16], "inference_only": True}
        clothes_task.validate_metadata(metadata, spec)
        wrong = {**metadata, "checkpoint": "/different/checkpoint"}
        with self.assertRaisesRegex(ValueError, "身份"):
            clothes_task.validate_metadata(wrong, spec)

    def test_wrong_shape_or_nonfinite_actions_are_not_reported_as_success(self):
        for actions in (np.zeros((30, 32)), np.full((30, 16), np.nan), np.full((30, 16), np.inf)):
            with self.subTest(shape=actions.shape):
                with self.assertRaises(ValueError):
                    clothes_task.validate_actions({"actions": actions})

    def test_clothes_button_bypasses_robot_and_takeover_dispatch(self):
        expected = (200, {"success": True, "inferenceOnly": True})
        with patch.object(server, "run_clothes_validation", return_value=expected) as validation, \
                patch.object(server.takeover, "start", side_effect=AssertionError("Robot takeover must not start")), \
                patch.object(server.web_config, "load", side_effect=AssertionError("Robot dispatch must not load")):
            for task_id in clothes_task.CLOTHES_TASKS:
                with self.subTest(task_id=task_id):
                    self.assertEqual(server.run_robot_task(task_id), expected)
                    validation.assert_called_with(task_id)

    def test_clothes_restore_never_launches_a_robot_script(self):
        with patch.object(server.subprocess, "run", side_effect=AssertionError("Robot process must not start")):
            for task_id in clothes_task.CLOTHES_TASKS:
                status, payload = server.restore_robot_task(task_id)
                self.assertEqual(status, 409)
                self.assertFalse(payload["success"])

    def test_model_validation_mode_blocks_existing_robot_tasks(self):
        with patch.dict(os.environ, {"VLA_MODEL_VALIDATION_ONLY": "1"}), \
                patch.object(server.subprocess, "Popen", side_effect=AssertionError("Robot process must not start")), \
                patch.object(server.takeover, "start", side_effect=AssertionError("Robot takeover must not start")):
            for task_id in server.TASKS:
                if task_id in clothes_task.CLOTHES_TASKS:
                    continue
                with self.subTest(task=task_id):
                    self.assertEqual(server.run_robot_task(task_id)[0], 403)
                    self.assertEqual(server.restore_robot_task(task_id)[0], 403)


if __name__ == "__main__":
    unittest.main()
