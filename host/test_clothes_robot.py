"""Execution tests use temporary fixtures and injected transports, never a real robot."""
import copy
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, call, patch

import cv2
import numpy as np

import clothes_robot as flow
from clothes_task import CLOTHES_TASKS, CLOSE_WASHER_TASK_ID, MEM_CLOSE_WASHER_TASK_ID, CONFIG_PATH, PUT_WASHER_TASK_ID, SOFA_TASK_ID, TASK_ID


class FakeClock:
    def __init__(self): self.now = 0.0
    def __call__(self): return self.now
    def pause(self, seconds): self.now += seconds


class FakeRobot:
    def __init__(self):
        self.commands = []
        self.closed = False
        self.reject = self.stale = self.no_arrival = False
        self.gripper_contact = False
        self.stalled_arm = None
        self.states = {"arm_joint_states": [.02] * 7 + [-.02] * 7, "gripper_states": [-.6, -.7]}
        ok, image = cv2.imencode('.jpg', np.zeros((224, 224, 3), dtype=np.uint8))
        assert ok
        self.image = image.tobytes()

    def observe(self, include_images=True):
        result = {"states": copy.deepcopy(self.states), "timestamps": {"states": 0 if self.stale else time.time_ns()}}
        if include_images:
            result["images"] = {name: {"encoding": "JPEG", "image_data": self.image}
                                for name in ("head", "hand_left", "hand_right")}
        return result

    def execute(self, command):
        self.commands.append(copy.deepcopy(command))
        if self.reject: raise RuntimeError("fixture rejected acknowledgment")
        if self.no_arrival: return
        for side, i in (("left", 0), ("right", 1)):
            arm = f"{side}_arm"
            if arm in command:
                if arm != self.stalled_arm:
                    self.states["arm_joint_states"][i * 7:(i + 1) * 7] = command[arm]["values"][-1]
                if not self.gripper_contact:
                    self.states["gripper_states"][i] = command[f"{side}_effector"][-1][0]

    def close(self): self.closed = True


class FakeModel:
    def __init__(self):
        self.observations = []
        self.callback = None
        self.actions = np.zeros((30, 16), dtype=np.float32)
        self.actions[:, 8:15] = .1
        self.actions[:, 15] = -.4
        self.closed = False

    def infer(self, observation):
        self.observations.append(copy.deepcopy(observation))
        if self.callback: self.callback()
        return {"actions": self.actions}

    def close(self): self.closed = True


class ClothesRobotTests(unittest.TestCase):
    def setUp(self):
        target = patch.object(flow, "configured_robot_host", return_value=None)
        target.start()
        self.addCleanup(target.stop)
        self.temp = tempfile.TemporaryDirectory(prefix="clothes-fake-only-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        env = patch.dict(os.environ, {"VLA_MODEL_VALIDATION_ONLY": "0"})
        env.start(); self.addCleanup(env.stop)
        self.config = self.root / 'tasks.json'
        self.setup_path = self.root / 'setup.json'
        self.report = self.root / 'report.json'
        self.task_data = json.loads(CONFIG_PATH.read_text())
        self.task_data[TASK_ID]["robot_execution_enabled"] = True
        self.task_data[TASK_ID]["arm_type"] = "right_arm"  # Retain coverage of single-arm configurations.
        self.config.write_text(json.dumps(self.task_data))
        self.setup = {"schema": "clothes-robot-setup/1", "motion_enabled": True,
                      "robot": {"host": "192.0.2.42", "management_port": 8890, "control_port": 8891, "model": "TEST_FIXTURE"},
                      "joint_limits": {"robot_model": "TEST_FIXTURE", "units": "rad",
                                       "right_arm": [[-3., 3.]] * 7, "right_gripper": [[-.785, 0.]],
                                       "left_arm": [[-3., 3.]] * 7, "left_gripper": [[-.785, 0.]]},
                      "control": {"trajectory_reference_time_s": 1., "max_chunks": 2,
                                  "max_duration_s": 60., "max_joint_step_rad": .15, "max_observation_age_s": 1.,
                                  "arrival_tolerance_rad": .02, "arrival_timeout_s": 1.}}
        self.save_setup()
        self.clock = FakeClock()
        self.robot, self.model = FakeRobot(), FakeModel()

    def save_setup(self): self.setup_path.write_text(json.dumps(self.setup))

    def test_saved_robot_ip_is_used_by_execution_without_relabeling_reset_poses(self):
        factory = Mock(return_value=self.robot)
        with patch.object(flow, "configured_robot_host", return_value="192.0.2.70"):
            flow.execute_flow(self.config, self.setup_path, self.report,
                              model_factory=lambda _: self.model, robot_factory=factory,
                              pause=self.clock.pause, clock=self.clock, wait=self.clock.pause)
        self.assertEqual(factory.call_args.args[0]["host"], "192.0.2.70")
        self.assertEqual(json.loads(self.setup_path.read_text())["robot"]["host"], "192.0.2.42")

    def run_flow(self, **kwargs):
        kwargs.setdefault('wait', self.clock.pause)
        return flow.execute_flow(self.config, self.setup_path, self.report,
                                 model_factory=lambda _: self.model, robot_factory=lambda _: self.robot,
                                 pause=self.clock.pause, clock=self.clock, **kwargs)

    def test_strict_validation_mode_disables_execution_for_all_three(self):
        with patch.dict(os.environ, {"VLA_MODEL_VALIDATION_ONLY": "1", "VLA_CLOTHES_LIVE_TASK": ""}):
            for task_id in CLOTHES_TASKS:
                status = flow.execution_readiness(task_id=task_id)
                self.assertFalse(status['can_execute'])
                self.assertEqual(status['start_mode'], 'current_observation')
                self.assertNotIn('initial_pose', status)
                self.assertNotIn('rack_relative_pose', status)

    def test_disabled_gates_refuse_before_constructing_transports(self):
        for gate in ('task', 'setup', 'web'):
            with self.subTest(gate=gate):
                self.task_data[TASK_ID]['robot_execution_enabled'] = gate != 'task'
                self.config.write_text(json.dumps(self.task_data))
                self.setup['motion_enabled'] = gate != 'setup'; self.save_setup()
                with patch.dict(os.environ, {'VLA_MODEL_VALIDATION_ONLY': '1' if gate == 'web' else '0'}):
                    factory = Mock(side_effect=AssertionError('Must not open any connection'))
                    with self.assertRaises(ValueError):
                        flow.execute_flow(self.config, self.setup_path, self.report,
                                          model_factory=factory, robot_factory=factory)
                    factory.assert_not_called()

    def test_local_preflight_never_opens_a_socket(self):
        with patch('socket.socket', side_effect=AssertionError('Preflight is local only')):
            self.assertTrue(flow.execution_readiness(self.config, self.setup_path)['can_execute'])

    def test_prepare_remote_env_starts_only_the_stopped_selected_mode(self):
        robot = self.setup["robot"]
        running = {"current_mode": "remote_env", "instance_running": True, "app_status": "running"}
        idle = {**running, "instance_running": False, "app_status": "idle"}
        with patch.object(flow, "robot_management", return_value=running) as manage:
            flow.ensure_remote_env_running(robot)
            manage.assert_called_once_with(robot, "status")
        with patch.object(flow, "robot_management", side_effect=[idle, {}, running]) as manage:
            flow.ensure_remote_env_running(robot)
            self.assertEqual(manage.call_args_list, [
                call(robot, "status"), call(robot, "start", start=True), call(robot, "status"),
            ])
        with patch.object(flow, "robot_management", return_value={**idle, "current_mode": "manual"}) as manage:
            with self.assertRaisesRegex(RuntimeError, "remote_env 模式"):
                flow.ensure_remote_env_running(robot)
            manage.assert_called_once_with(robot, "status")

    def test_missing_reset_limits_and_step_only_disable_reset(self):
        self.setup['sofa_reset_pose'] = {
            'robot': dict(self.setup['robot']), 'recorded_at': '2026-11-18T16:50:35+08:00',
            'left_arm': [.4] * 7, 'left_gripper': -.75,
            'right_arm': [-1.52] * 7, 'right_gripper': -.2,
        }
        self.save_setup()
        self.assertTrue(flow.execution_readiness(self.config, self.setup_path, task_id=SOFA_TASK_ID)['can_reset_pose'])
        del self.setup['joint_limits']['right_arm']
        del self.setup['control']['max_joint_step_rad']
        self.save_setup()
        for task_id in (TASK_ID, SOFA_TASK_ID):
            with self.subTest(task_id=task_id):
                status = flow.execution_readiness(self.config, self.setup_path, task_id=task_id)
                self.assertTrue(status['config_ready'])
                self.assertTrue(status['can_execute'])
                self.assertFalse(status['can_reset_pose'])

    def test_first_command_comes_from_live_observation_and_moves_only_the_task_arm(self):
        self.model.actions[:, :8] = 999.  # A left-arm prediction must never be dispatched.
        def check_inference_precedes_commands():
            self.assertEqual(len(self.robot.commands), len(self.model.observations) - 1)
        self.model.callback = check_inference_precedes_commands
        result = self.run_flow()
        self.assertEqual(len(self.robot.commands), 2)
        self.assertEqual(result['model_chunks_submitted'], 2)
        self.assertFalse(result['physical_effect_verified'])
        self.assertEqual(result['status'], 'awaiting_physical_verification')
        for command in self.robot.commands:
            self.assertEqual(set(command), {'base_link', 'trajectory_reference_time', 'right_arm', 'right_effector'})
            self.assertEqual(len(command['right_arm']['values']), 30)
        self.assertEqual(self.model.observations[0]['prompt'], self.task_data[TASK_ID]['prompt'])
        self.assertEqual(self.model.observations[0]['state'], [.02] * 7 + [-.6] + [-.02] * 7 + [-.7])
        self.assertEqual(set(self.model.observations[0]['images']), {'cam_head', 'cam_left', 'cam_right'})
        self.assertTrue(self.robot.closed and self.model.closed)

    def test_rack_and_put_washer_send_raw_output_for_both_arms_and_grippers(self):
        configured = json.loads(CONFIG_PATH.read_text())
        for task_id in ("right_arm_hang_clothes", TASK_ID, PUT_WASHER_TASK_ID, CLOSE_WASHER_TASK_ID, MEM_CLOSE_WASHER_TASK_ID):
            with self.subTest(task_id=task_id):
                self.task_data[task_id] = configured[task_id]
                self.task_data[task_id]['run_until_stopped'] = False
                self.config.write_text(json.dumps(self.task_data))
                self.robot, self.model = FakeRobot(), FakeModel()
                self.model.actions = self.model.actions.astype(np.float64)
                self.model.actions[0, [0, 7, 8, 15]] = [4.123456789, .02, -4.123456789, -.92]
                report = self.run_flow(task_id=task_id)
                self.assertEqual(report["arrival_checked_parts"], [])
                self.assertIsNone(report['last_target_arrival_confirmed'])
                self.assertEqual(self.model.observations[0]["prompt"], configured[task_id]["prompt"])
                self.assertEqual(len(self.robot.commands), 2)
                command = self.robot.commands[0]
                self.assertEqual(command["left_arm"]["values"], self.model.actions[:, :7].tolist())
                self.assertEqual(command["left_effector"], self.model.actions[:, 7:8].tolist())
                self.assertEqual(command["right_arm"]["values"], self.model.actions[:, 8:15].tolist())
                self.assertEqual(command["right_effector"], self.model.actions[:, 15:16].tolist())

    def test_all_managed_tasks_advance_by_time_and_use_actual_unreached_state(self):
        configured = json.loads(CONFIG_PATH.read_text())
        for task_id in CLOTHES_TASKS:
            with self.subTest(task_id=task_id):
                self.task_data[task_id] = {**configured[task_id], 'run_until_stopped': False}
                self.config.write_text(json.dumps(self.task_data))
                self.clock = FakeClock()
                self.robot, self.model = FakeRobot(), FakeModel()
                self.robot.no_arrival = True
                initial = flow.state_vector(self.robot.observe())
                self.robot.observe = Mock(wraps=self.robot.observe)
                started = []
                execute = self.robot.execute

                def dispatch(command):
                    started.append(self.clock())
                    execute(command)

                self.robot.execute = dispatch
                wait = Mock(side_effect=self.clock.pause)
                with patch.object(flow, 'wait_for_arrival', side_effect=AssertionError('Tests must not wait for arrival')) as arrival:
                    report = self.run_flow(task_id=task_id, wait=wait)
                arrival.assert_not_called()
                self.assertEqual(started, [0., 1.])
                self.assertEqual(wait.call_args_list, [call(1.), call(1.)])
                self.assertEqual(self.robot.observe.call_args_list, [call(), call()])
                self.assertEqual([obs['state'] for obs in self.model.observations], [initial, initial])
                self.assertEqual(report['model_chunks_submitted'], 2)
                self.assertEqual(report['execution_progression'], 'trajectory_time')
                self.assertEqual(report['arrival_checked_parts'], [])
                self.assertIsNone(report['last_target_arrival_confirmed'])

    def test_stop_interrupts_mem_reference_time_wait_without_another_dispatch(self):
        stop = threading.Event()

        def interrupt(duration):
            self.clock.pause(.2)
            stop.set()
            return True

        with patch.object(stop, 'wait', side_effect=interrupt) as wait:
            report = self.run_flow(task_id=MEM_CLOSE_WASHER_TASK_ID, stop=stop, wait=None)
        wait.assert_called_once_with(1.)
        self.assertLess(self.clock(), 1.)
        self.assertEqual(len(self.robot.commands), 1)
        self.assertEqual(len(self.model.observations), 1)
        self.assertEqual(report['model_chunks_submitted'], 1)
        self.assertEqual(report['status'], 'stopped_submitting')
        self.assertIsNone(report['last_target_arrival_confirmed'])

    def test_put_washer_runs_beyond_chunk_and_time_limits_until_operator_stop(self):
        self.task_data[PUT_WASHER_TASK_ID]['run_until_stopped'] = True
        self.config.write_text(json.dumps(self.task_data))
        self.setup['control']['max_chunks'] = 30
        self.save_setup()
        stop = threading.Event()

        def infer_progress():
            self.clock.pause(10)
            if len(self.model.observations) == 42:
                stop.set()

        self.model.callback = infer_progress
        report = self.run_flow(task_id=PUT_WASHER_TASK_ID, stop=stop)
        self.assertGreater(self.clock.now, 360)
        self.assertEqual(len(self.robot.commands), 41)
        self.assertEqual(len(self.model.observations), 42)
        self.assertEqual(report['model_chunks_submitted'], 41)
        self.assertTrue(report['run_until_stopped'])
        self.assertEqual(report['status'], 'stopped_submitting')

    def test_put_washer_continuous_mode_honors_stop_before_first_submission(self):
        self.task_data[PUT_WASHER_TASK_ID]['run_until_stopped'] = True
        self.config.write_text(json.dumps(self.task_data))
        stop = threading.Event()
        self.model.callback = stop.set
        report = self.run_flow(task_id=PUT_WASHER_TASK_ID, stop=stop)
        self.assertEqual(self.robot.commands, [])
        self.assertEqual(report['status'], 'stopped_submitting')
        self.assertTrue(report['run_until_stopped'])

    def test_put_washer_stop_after_submission_prevents_the_next_segment(self):
        self.task_data[PUT_WASHER_TASK_ID]['run_until_stopped'] = True
        self.config.write_text(json.dumps(self.task_data))
        stop = threading.Event()
        execute = self.robot.execute

        def submit_and_stop(command):
            execute(command)
            stop.set()

        self.robot.execute = submit_and_stop
        report = self.run_flow(task_id=PUT_WASHER_TASK_ID, stop=stop)
        self.assertEqual(len(self.robot.commands), 1)
        self.assertEqual(len(self.model.observations), 1)
        self.assertEqual(report['status'], 'stopped_submitting')

    def test_put_washer_pose_reset_remains_a_single_operation_in_continuous_mode(self):
        self.task_data[PUT_WASHER_TASK_ID]['run_until_stopped'] = True
        self.config.write_text(json.dumps(self.task_data))
        self.setup['reset_poses'] = {PUT_WASHER_TASK_ID: {
            'robot': dict(self.setup['robot']), 'recorded_at': '2026-11-23T13:53:19+08:00',
            'left_arm': [.4] * 7, 'left_gripper': -.5,
            'right_arm': [-.4] * 7, 'right_gripper': -.4,
        }}
        self.save_setup()
        report = self.run_flow(task_id=PUT_WASHER_TASK_ID, restore_pose=True)
        self.assertEqual(len(self.robot.commands), 1)
        self.assertEqual(self.model.observations, [])
        self.assertEqual(report['status'], 'pose_restored')
        self.assertFalse(report['run_until_stopped'])

    def test_sofa_grasp_choices_keep_all_model_columns_and_select_the_prompt(self):
        task_id = SOFA_TASK_ID
        self.task_data[task_id]['robot_execution_enabled'] = True
        self.config.write_text(json.dumps(self.task_data))
        lower = -np.pi / 4
        self.setup['joint_limits']['left_gripper'] = [[lower, 0.]]
        self.setup['joint_limits']['right_gripper'] = [[lower, 0.]]
        self.save_setup()
        self.model.actions = self.model.actions.astype(np.float64)
        self.model.actions[:, :7] = np.linspace(.04, .10, 7)
        self.model.actions[0, 0] = 4.123456789
        self.model.actions[:, 7] = -.4
        self.model.actions[:2, 7] = [lower - .005927, .005927]
        self.model.actions[:, 8:15] = np.linspace(-.07, .05, 7)
        self.model.actions[0, 8] = -4.123456789
        self.model.actions[:, 15] = -.3
        self.model.actions[:2, 15] = [.005927, lower - .005927]
        left_prompt = "Use the left arm to pick up the light blue clothes from the sofa."
        right_prompt = "Use both arms to pick up the light blue clothes from the sofa."
        model_factory = Mock(return_value=self.model)
        for hand, prompt in (('left', left_prompt), ('right', right_prompt), (None, right_prompt)):
            with self.subTest(grasp_hand=hand):
                self.robot = FakeRobot()
                self.model.observations.clear()
                kwargs = {} if hand is None else {'grasp_hand': hand}
                result = flow.execute_flow(self.config, self.setup_path, self.report, task_id=task_id,
                                           model_factory=model_factory, robot_factory=lambda _: self.robot,
                                           pause=self.clock.pause, clock=self.clock, wait=self.clock.pause, **kwargs)
                self.assertEqual(result['robot_actions_submitted'], 2)
                self.assertEqual(result['prompt'], prompt)
                self.assertEqual(model_factory.call_args.args[0]['prompt'], right_prompt)
                self.assertEqual([obs['prompt'] for obs in self.model.observations], [prompt, prompt])
                for command in self.robot.commands:
                    self.assertEqual(set(command), {'base_link', 'trajectory_reference_time', 'left_arm',
                                                    'left_effector', 'right_arm', 'right_effector'})
                    self.assertEqual(command['left_arm']['values'], self.model.actions[:, :7].tolist())
                    self.assertEqual(command['right_arm']['values'], self.model.actions[:, 8:15].tolist())
                    self.assertEqual(command['left_effector'], self.model.actions[:, 7:8].tolist())
                    self.assertEqual(command['right_effector'], self.model.actions[:, 15:16].tolist())

    def test_sofa_continues_with_measured_gripper_contact(self):
        self.task_data[SOFA_TASK_ID]['robot_execution_enabled'] = True
        self.config.write_text(json.dumps(self.task_data))
        self.robot.gripper_contact = True
        self.model.actions[:, :7] = .08
        self.model.actions[:, 7] = 0.
        self.model.actions[:, 8:15] = -.09
        self.model.actions[:, 15] = 0.
        result = self.run_flow(task_id=SOFA_TASK_ID)
        self.assertEqual(result['model_chunks_submitted'], 2)
        self.assertEqual(result['arrival_checked_parts'], [])
        np.testing.assert_allclose(self.model.observations[1]['state'][:7], .08)
        np.testing.assert_allclose(self.model.observations[1]['state'][8:15], -.09)
        self.assertEqual(self.model.observations[1]['state'][7], -.6)
        self.assertEqual(self.model.observations[1]['state'][15], -.7)

    def test_sofa_uses_actual_right_arm_observation_without_waiting_for_arrival(self):
        self.task_data[SOFA_TASK_ID]['robot_execution_enabled'] = True
        self.config.write_text(json.dumps(self.task_data))
        self.robot.stalled_arm = 'right_arm'
        self.model.actions[:, :7] = .08
        self.model.actions[:, 8:15] = -.09
        report = self.run_flow(task_id=SOFA_TASK_ID)
        self.assertEqual(len(self.robot.commands), 2)
        np.testing.assert_allclose(self.robot.states['arm_joint_states'][:7], .08)
        np.testing.assert_allclose(self.robot.states['arm_joint_states'][7:], -.02)
        np.testing.assert_allclose(self.model.observations[1]['state'][8:15], -.02)
        self.assertEqual(report['status'], 'awaiting_physical_verification')
        self.assertIsNone(report['last_target_arrival_confirmed'])

    def test_sofa_pose_reset_moves_all_four_tracks_together_without_loading_model(self):
        self.task_data[SOFA_TASK_ID]['robot_execution_enabled'] = True
        self.config.write_text(json.dumps(self.task_data))
        pose = {'robot': dict(self.setup['robot']), 'recorded_at': '2026-11-18T16:50:35+08:00',
                'left_arm': [.4] * 7, 'left_gripper': -.75,
                'right_arm': [-1.52] * 7, 'right_gripper': -.2}
        self.setup['sofa_reset_pose'] = pose
        self.save_setup()
        initial = copy.deepcopy(self.robot.states)
        factory = Mock(side_effect=AssertionError('Pose reset must not load the VLA model'))
        kwargs = dict(task_id=SOFA_TASK_ID, restore_pose=True, model_factory=factory, robot_factory=lambda _: self.robot,
                      pause=self.clock.pause, clock=self.clock)
        report = flow.execute_flow(self.config, self.setup_path, self.report, **kwargs)
        self.assertEqual(report['status'], 'pose_restored')
        self.assertEqual(report['model_chunks_submitted'], 0)
        self.assertEqual(report['arrival_checked_parts'], ['left_arm', 'right_arm', 'left_gripper', 'right_gripper'])
        command, = self.robot.commands
        self.assertEqual(set(command), {'base_link', 'trajectory_reference_time', 'left_arm', 'left_effector',
                                        'right_arm', 'right_effector'})
        self.assertGreater(command['trajectory_reference_time'], 2.)
        left = np.asarray(command['left_arm']['values'])
        right = np.asarray(command['right_arm']['values'])
        left_gripper = np.asarray(command['left_effector'])
        right_gripper = np.asarray(command['right_effector'])
        self.assertEqual(command['left_arm']['kind'], 'JOINT_ABS')
        self.assertEqual(command['right_arm']['kind'], 'JOINT_ABS')
        self.assertEqual(left.shape, (len(right), 7))
        self.assertEqual(right.shape, (len(left), 7))
        self.assertEqual(left_gripper.shape, (len(left), 1))
        self.assertEqual(right_gripper.shape, (len(left), 1))
        for track, start, target in (
            (left, initial['arm_joint_states'][:7], pose['left_arm']),
            (right, initial['arm_joint_states'][7:], pose['right_arm']),
            (left_gripper, initial['gripper_states'][:1], [pose['left_gripper']]),
            (right_gripper, initial['gripper_states'][1:], [pose['right_gripper']]),
        ):
            np.testing.assert_allclose(track[-1], target)
            progress = np.linspace(0., 1., len(track) + 1)[1:, None]
            np.testing.assert_allclose(track, np.asarray(start) + progress * (np.asarray(target) - start))
        report = flow.execute_flow(self.config, self.setup_path, self.report, **kwargs)
        self.assertEqual(report['status'], 'pose_already_set')
        self.assertEqual(len(self.robot.commands), 1)
        factory.assert_not_called()

    def test_sofa_pose_reset_requires_a_recorded_right_side(self):
        self.task_data[SOFA_TASK_ID]['robot_execution_enabled'] = True
        self.config.write_text(json.dumps(self.task_data))
        self.setup['sofa_reset_pose'] = {
            'robot': dict(self.setup['robot']), 'recorded_at': '2026-11-18T16:50:35+08:00',
            'left_arm': [.4] * 7, 'left_gripper': -.75,
        }
        self.save_setup()
        ready = flow.execution_readiness(self.config, self.setup_path, task_id=SOFA_TASK_ID)
        self.assertTrue(ready['can_execute'])
        self.assertFalse(ready['can_reset_pose'])
        factory = Mock(side_effect=AssertionError('Incomplete pose reset must not open any connection'))
        with self.assertRaises(ValueError):
            flow.execute_flow(self.config, self.setup_path, self.report, task_id=SOFA_TASK_ID, restore_pose=True,
                              model_factory=factory, robot_factory=factory)
        factory.assert_not_called()

    def test_stop_before_start_never_constructs_a_connection(self):
        stop = threading.Event(); stop.set()
        factory = Mock(side_effect=AssertionError('No connection after cancellation'))
        result = flow.execute_flow(self.config, self.setup_path, self.report,
                                   stop=stop, model_factory=factory, robot_factory=factory)
        self.assertEqual(result['robot_actions_submitted'], 0)
        factory.assert_not_called()

    def test_stop_during_first_inference_prevents_any_trajectory(self):
        stop = threading.Event(); self.model.callback = stop.set
        result = self.run_flow(stop=stop)
        self.assertEqual(self.robot.commands, [])
        self.assertEqual(result['model_chunks_submitted'], 0)
        self.assertEqual(result['status'], 'stopped_submitting')

    def test_disabled_gate_rechecked_after_inference(self):
        def disable():
            self.task_data[TASK_ID]['robot_execution_enabled'] = False
            self.config.write_text(json.dumps(self.task_data))
        self.model.callback = disable
        with self.assertRaises(ValueError): self.run_flow()
        self.assertEqual(self.robot.commands, [])

    def test_robot_configuration_change_stops_dispatch(self):
        def change_setup():
            self.setup['control']['max_chunks'] = 3
            self.save_setup()
        self.model.callback = change_setup
        with self.assertRaisesRegex(ValueError, '配置已变更'): self.run_flow()
        self.assertEqual(self.robot.commands, [])

    def test_rejected_ack_ends_run_and_keeps_readiness(self):
        self.robot.reject = True
        with self.assertRaises(RuntimeError): self.run_flow()
        self.assertEqual(len(self.robot.commands), 1)
        self.assertEqual(json.loads(self.report.read_text())['status'], 'motion_unknown')
        self.assertTrue(flow.execution_readiness(self.config, self.setup_path)['can_execute'])

    def test_pose_reset_arrival_timeout_does_not_retry_or_reset(self):
        self.task_data[SOFA_TASK_ID]['robot_execution_enabled'] = True
        self.config.write_text(json.dumps(self.task_data))
        self.setup['sofa_reset_pose'] = {
            'robot': dict(self.setup['robot']), 'recorded_at': '2026-11-18T16:50:35+08:00',
            'left_arm': [.4] * 7, 'left_gripper': -.5,
            'right_arm': [-.4] * 7, 'right_gripper': -.4,
        }
        self.save_setup()
        self.robot.no_arrival = True
        with self.assertRaisesRegex(RuntimeError, '未确认到位'):
            self.run_flow(task_id=SOFA_TASK_ID, restore_pose=True)
        self.assertEqual(len(self.robot.commands), 1)
        self.assertEqual(self.model.observations, [])
        self.assertEqual(json.loads(self.report.read_text())['status'], 'motion_unknown')
        self.assertTrue(flow.execution_readiness(self.config, self.setup_path)['can_execute'])

    def test_stale_current_state_prevents_inference_and_dispatch(self):
        self.robot.stale = True
        with self.assertRaisesRegex(ValueError, '时间戳'): self.run_flow()
        self.assertEqual(self.robot.commands, [])
        self.assertEqual(self.model.observations, [])

    def test_invalid_model_chunk_does_not_dispatch_or_reset(self):
        for failure, actions in (
            ('shape', np.zeros((30, 15))),
            ('nan', np.full((30, 16), np.nan)),
            ('infinite', np.full((30, 16), np.inf)),
        ):
            with self.subTest(failure=failure):
                self.model.actions = actions
                with self.assertRaises(ValueError): self.run_flow()
                self.assertEqual(self.robot.commands, [])

    def test_inference_overrunning_time_budget_does_not_submit_another_chunk(self):
        self.model.callback = lambda: self.clock.pause(120)
        result = self.run_flow()
        self.assertEqual(self.robot.commands, [])
        self.assertEqual(result['model_chunks_submitted'], 0)

    def test_server_execute_entry_refuses_disabled_tasks_before_spawning(self):
        import intergrated_test_server as server
        with patch.dict(os.environ, {"VLA_MODEL_VALIDATION_ONLY": "1", "VLA_CLOTHES_LIVE_TASK": ""}), \
                patch.object(server.subprocess, 'Popen', side_effect=AssertionError('Must not spawn motion runner')) as process:
            for task_id in CLOTHES_TASKS:
                status, payload = server.run_clothes_robot_task(task_id)
                self.assertEqual(status, 409)
                self.assertEqual(payload['robotActionsSubmitted'], 0)
            process.assert_not_called()


if __name__ == '__main__': unittest.main()
