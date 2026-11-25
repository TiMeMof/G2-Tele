"""A1 HTTP/configuration/dispatch tests. All robot subprocesses are mocked."""
import contextlib
import http.client
import io
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import clothes_task
import web_config as config
import intergrated_test_server as app
import network_config
import robot_target
from chunk_log import Transport


class QuietHandler(app.IntegratedHandler):
    def log_message(self, *args):
        pass


class A1Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="a1_test_")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.patch = mock.patch.multiple(config, BASE_DIR=self.root, CONFIG_PATH=self.root / "settings.json")
        self.patch.start()
        self.addCleanup(self.patch.stop)
        target_config = mock.patch.object(robot_target, "WEB_CONFIG_PATH", config.CONFIG_PATH)
        target_config.start()
        self.addCleanup(target_config.stop)
        # These HTTP tests must not probe the live robot's collection service.
        readiness = mock.patch.object(app.takeover, "readiness_check", return_value={
            "ready": False, "persistent_recording_ready": False, "blockers": ["offline test"],
        })
        readiness.start()
        self.addCleanup(readiness.stop)
        app.current_process = None
        app.current_task_id = None
        app.current_operation = None
        self.server = app.ThreadingHTTPServer(("127.0.0.1", 0), QuietHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.close_server)

    def close_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def request(self, path, body=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        try:
            connection.request("GET" if body is None else "POST", path,
                               None if body is None else json.dumps(body), {"Content-Type": "application/json"})
            response = connection.getresponse()
            return response.status, response.read().decode("utf-8")
        finally:
            connection.close()

    def test_validation_mode_exposes_only_the_selected_clothing_robot_route(self):
        sofa = "left_arm_pick_blue_clothes_from_sofa"
        with mock.patch.dict(os.environ, {"VLA_MODEL_VALIDATION_ONLY": "1", "VLA_CLOTHES_LIVE_TASK": sofa}), \
                mock.patch.object(app, "run_clothes_robot_task", return_value=(200, {"success": True})) as run, \
                mock.patch.object(app, "_run_clothes_process", return_value=(200, {"success": True})) as reset:
            self.assertEqual(self.request("/api/run-clothes-robot", {"taskId": sofa})[0], 200)
            self.assertEqual(self.request("/api/clothes/restore-pose", {"taskId": sofa})[0], 200)
            for other in ("right_arm_take_off_clothes", "right_arm_hang_clothes"):
                self.assertEqual(self.request("/api/run-clothes-robot", {"taskId": other})[0], 403)
                self.assertEqual(self.request("/api/run-clothes-robot", {"taskId": other, "graspHand": "left"})[0], 400)
                self.assertEqual(self.request("/api/clothes/restore-pose", {"taskId": other})[0], 403)
            self.assertEqual(self.request("/api/run-clothes-robot", {"taskId": sofa, "graspHand": "both"})[0], 400)
            self.assertEqual(self.request("/api/clothes/restore-pose", {"taskId": sofa, "left_arm": [0]*7})[0], 400)
            self.assertEqual(self.request("/api/restore-task", {"taskId": sofa})[0], 403)
            run.assert_called_once_with(sofa)
            reset.assert_called_once_with(sofa, real_robot=True, restore_pose=True)

    def test_validation_mode_can_open_takeoff_alongside_sofa(self):
        sofa, takeoff = clothes_task.SOFA_TASK_ID, clothes_task.TASK_ID
        with mock.patch.dict(os.environ, {"VLA_MODEL_VALIDATION_ONLY": "1", "VLA_CLOTHES_LIVE_TASK": f"{sofa},{takeoff}"}), \
                mock.patch.object(app, "run_clothes_robot_task", return_value=(200, {"success": True})) as run:
            for task_id in (sofa, takeoff):
                self.assertEqual(self.request("/api/run-clothes-robot", {"taskId": task_id})[0], 200)
            self.assertEqual(self.request("/api/run-clothes-robot", {"taskId": "right_arm_hang_clothes"})[0], 403)
            with mock.patch.object(app, "clothes_readiness", return_value={"can_execute": True, "can_reset_pose": False}):
                self.assertEqual(self.request("/api/clothes/restore-pose", {"taskId": takeoff})[0], 409)
            self.assertEqual(run.call_args_list, [mock.call(sofa), mock.call(takeoff)])

    def test_rack_tasks_control_their_own_model(self):
        tasks = ("right_arm_hang_clothes", clothes_task.TASK_ID, clothes_task.PUT_WASHER_TASK_ID, clothes_task.CLOSE_WASHER_TASK_ID, clothes_task.MEM_CLOSE_WASHER_TASK_ID)
        models = {task_id: mock.Mock(ready=False) for task_id in tasks}
        robot = {"host": "192.0.2.42"}
        with mock.patch.dict(os.environ, {"VLA_MODEL_VALIDATION_ONLY": "1", "VLA_CLOTHES_LIVE_TASK": ",".join(tasks)}), \
                mock.patch.dict(app.clothes_policies, models), \
                mock.patch.object(app, "clothes_readiness", return_value={"can_execute": True, "can_reset_pose": False, "robot": robot}), \
                mock.patch.object(app, "ensure_remote_env_running") as prepare, \
                mock.patch.object(app, "clothes_policy_slot", side_effect=contextlib.nullcontext):
            for task_id, model in models.items():
                body = {"taskId": task_id}
                model.start.side_effect = lambda model=model: setattr(model, "ready", True)
                model.close.side_effect = lambda model=model: setattr(model, "ready", False)
                self.assertEqual(self.request("/api/clothes/model/start", body)[0], 200)
                self.assertEqual(json.loads(self.request("/api/config")[1])["clothesModels"][task_id], "ready")
                self.assertEqual(self.request("/api/clothes/model/stop", body)[0], 200)
                self.assertEqual(json.loads(self.request("/api/config")[1])["clothesModels"][task_id], "off")
                model.start.assert_called_once_with()
                model.close.assert_called_once_with()
            self.assertEqual(prepare.call_count, len(tasks))

    def test_managed_sofa_model_survives_run_and_stop_and_reset_works_after_close(self):
        task = {"taskId": clothes_task.SOFA_TASK_ID}
        robot = {"host": "192.0.2.42", "management_port": 8890, "control_port": 8891}
        ready = {"can_execute": True, "can_reset_pose": True, "robot": robot}
        report_file = self.root / "sofa-result.json"
        model_process = mock.Mock()
        model_process.poll.return_value = None
        model = clothes_task.ClothesPolicy(task["taskId"])
        self.addCleanup(model.close)
        releases = []
        commands = []

        @contextlib.contextmanager
        def launch(spec):
            self.assertEqual(spec["task_id"], task["taskId"])
            yield model_process
            releases.append(model_process)

        def runner(command, **kwargs):
            commands.append(command)
            process = mock.Mock(returncode=None)
            process.poll.side_effect = lambda: process.returncode

            def finish(timeout=None):
                if "--restore-pose" in command:
                    self.assertNotIn("--use-loaded-model", command)
                    self.assertFalse(model.ready)
                    status = "pose_restored"
                else:
                    self.assertIn("--use-loaded-model", command)
                    self.assertIs(model.process, model_process)
                    hand = "left" if len(commands) == 1 else "right"
                    self.assertEqual(command[command.index("--grasp-hand") + 1], hand)
                    self.assertEqual(json.loads(self.request("/api/config")[1])["sofaGraspHand"], hand)
                    self.assertEqual(self.request("/api/clothes/model/stop", task)[0], 409)
                    status = "awaiting_physical_verification"
                    if len(commands) == 2:
                        self.assertEqual(self.request("/api/clothes/stop-test", task)[0], 202)
                        process.send_signal.assert_called_once_with(app.signal.SIGINT)
                        status = "stopped_submitting"
                report_file.write_text(json.dumps({"task_id": task["taskId"], "status": status}))
                process.returncode = 0
                return "", ""

            process.communicate.side_effect = finish
            return process

        with mock.patch.dict(os.environ, {"VLA_MODEL_VALIDATION_ONLY": "1", "VLA_CLOTHES_LIVE_TASK": task["taskId"]}), \
                mock.patch.dict(app.clothes_policies, {task["taskId"]: model}), \
                mock.patch.object(app, "clothes_readiness", return_value=ready), \
                mock.patch.object(app, "ensure_remote_env_running") as prepare, \
                mock.patch.object(app, "clothes_policy_slot", side_effect=contextlib.nullcontext), \
                mock.patch.object(app, "robot_report_path", return_value=report_file), \
                mock.patch.object(app.subprocess, "Popen", side_effect=runner), \
                mock.patch.object(clothes_task, "launch_policy", side_effect=launch) as load, \
                mock.patch.object(clothes_task, "ModelConnection", side_effect=ConnectionRefusedError):
            self.assertEqual(self.request("/api/run-clothes-robot", task)[0], 409)
            self.assertEqual(self.request("/api/clothes/model/start", task)[0], 200)
            prepare.assert_called_once_with(robot)
            self.assertEqual(json.loads(self.request("/api/config")[1])["sofaModel"], "ready")
            self.assertEqual(self.request("/api/run-clothes-robot", {**task, "graspHand": "left"})[0], 200)
            self.assertEqual(self.request("/api/run-clothes-robot", {**task, "graspHand": "right"})[0], 200)
            self.assertIsNone(json.loads(self.request("/api/config")[1])["sofaGraspHand"])
            self.assertTrue(model.ready)
            self.assertEqual(releases, [])
            load.assert_called_once()
            self.assertEqual(self.request("/api/clothes/model/stop", task)[0], 200)
            self.assertEqual(releases, [model_process])
            self.assertEqual(self.request("/api/clothes/restore-pose", task)[0], 200)

    def test_put_washer_uses_managed_card_and_keeps_model_after_test_or_stop(self):
        self._check_managed_washer_card(clothes_task.PUT_WASHER_TASK_ID)

    def test_close_washer_uses_managed_card_and_keeps_model_after_test_or_stop(self):
        self._check_managed_washer_card(clothes_task.CLOSE_WASHER_TASK_ID)

    def test_mem_close_washer_uses_managed_card_and_keeps_model_after_test_or_stop(self):
        self._check_managed_washer_card(clothes_task.MEM_CLOSE_WASHER_TASK_ID)

    def _check_managed_washer_card(self, task_id):
        model = mock.Mock(ready=True)
        report_path = self.root / 'put-washer-report.json'
        commands = []

        def runner(command, **kwargs):
            commands.append(command)
            process = mock.Mock(returncode=None)
            process.poll.side_effect = lambda: process.returncode

            def complete(timeout=None):
                if '--restore-pose' in command:
                    self.assertEqual(timeout, 360)
                    self.assertNotIn('--use-loaded-model', command)
                    result = 'pose_restored'
                else:
                    self.assertIsNone(timeout)
                    self.assertIn('--use-loaded-model', command)
                    self.assertNotIn('--grasp-hand', command)
                    self.assertEqual(self.request('/api/clothes/stop-test', {'taskId': task_id})[0], 202)
                    process.send_signal.assert_called_once_with(app.signal.SIGINT)
                    result = 'stopped_submitting'
                report_path.write_text(json.dumps({'task_id': task_id, 'status': result}))
                process.returncode = 0
                return '', ''

            process.communicate.side_effect = complete
            return process

        with mock.patch.dict(os.environ, {'VLA_MODEL_VALIDATION_ONLY': '1', 'VLA_CLOTHES_LIVE_TASK': task_id}), \
                mock.patch.dict(app.clothes_policies, {task_id: model}), \
                mock.patch.object(app, 'clothes_readiness', return_value={'can_execute': True, 'can_reset_pose': True}), \
                mock.patch.object(app, 'robot_report_path', return_value=report_path), \
                mock.patch.object(app.subprocess, 'Popen', side_effect=runner):
            tasks = json.loads(self.request('/api/tasks')[1])['tasks']
            self.assertEqual(tasks[task_id]['type'], 'washer')
            self.assertTrue(tasks[task_id]['managedClothes'])
            for _ in range(2):
                self.assertEqual(self.request('/api/run-clothes-robot', {'taskId': task_id})[0], 200)
                self.assertTrue(model.ready)
                model.close.assert_not_called()
            self.assertEqual(self.request('/api/clothes/restore-pose', {'taskId': task_id})[0], 200)
            model.start.assert_not_called()
            model.close.assert_not_called()

    def test_put_washer_grippers_use_the_selected_robot_without_arm_commands(self):
        task_id = clothes_task.PUT_WASHER_TASK_ID
        config.save({**config.defaults(), 'robot_host': '192.0.2.70'})
        with mock.patch.dict(os.environ, {'VLA_MODEL_VALIDATION_ONLY': '0'}), \
                mock.patch.object(app, 'ensure_remote_env_running') as prepare, \
                mock.patch.object(app, 'RobotConnection') as connection:
            for side, action in (('left', 'open'), ('right', 'close')):
                status, raw = self.request('/api/clothes/gripper', {'taskId': task_id, 'side': side, 'action': action})
                self.assertEqual(status, 200, raw)
                command = connection.return_value.execute.call_args.args[0]
                self.assertEqual(set(command), {'base_link', 'trajectory_reference_time', side + '_effector'})
                self.assertEqual(connection.call_args.args[0]['host'], network_config.get_robot_host())
            self.assertEqual(prepare.call_count, 2)

    def test_standalone_gripper_matches_reference_values_and_preserves_loaded_models(self):
        config.save({**config.defaults(), 'robot_host': '192.0.2.70'})
        loaded_model = mock.Mock(ready=True)
        with mock.patch.dict(os.environ, {'VLA_MODEL_VALIDATION_ONLY': '0'}), \
                mock.patch.dict(app.clothes_policies, {clothes_task.PUT_WASHER_TASK_ID: loaded_model}), \
                mock.patch.object(app, 'RobotConnection') as connection, \
                mock.patch.object(app, 'ensure_remote_env_running') as prepare, \
                mock.patch.object(app, 'sleep') as pause:
            for side in ('left', 'right'):
                for action, target in (('open', -0.785), ('close', 0.0)):
                    status, raw = self.request('/api/gripper', {'side': side, 'action': action})
                    self.assertEqual(status, 200, raw)
                    self.assertEqual(connection.call_args.args[0], {
                        'host': '192.0.2.70', 'management_port': 8890, 'control_port': 8891})
                    self.assertEqual(connection.return_value.execute.call_args.args[0], {
                        'base_link': 'base_link', 'trajectory_reference_time': 0.8,
                        side + '_effector': [[target]]})
                    self.assertFalse(app.run_lock.locked())
            self.assertEqual(connection.return_value.close.call_count, 4)
            self.assertEqual(pause.call_args_list, [mock.call(0.8)] * 4)
            prepare.assert_not_called()
            loaded_model.start.assert_not_called()
            loaded_model.close.assert_not_called()

    def test_standalone_gripper_rejects_bad_requests_and_busy_state_without_connection(self):
        with mock.patch.dict(os.environ, {'VLA_MODEL_VALIDATION_ONLY': '0'}), \
                mock.patch.object(app, 'RobotConnection') as connection:
            for body in ({}, [], {'side': 'both', 'action': 'close'},
                         {'side': [], 'action': 'close'}, {'side': 'left', 'action': {}},
                         {'side': 'left', 'action': 'close', 'target': 0.0}):
                status, raw = self.request('/api/gripper', body)
                self.assertEqual(status, 400, raw)
            app.run_lock.acquire()
            try:
                self.assertEqual(self.request('/api/gripper', {'side': 'left', 'action': 'close'})[0], 409)
            finally:
                app.run_lock.release()
            with mock.patch.dict(os.environ, {'VLA_MODEL_VALIDATION_ONLY': '1'}):
                self.assertEqual(self.request('/api/gripper', {'side': 'right', 'action': 'close'})[0], 403)
            connection.assert_not_called()

    def test_standalone_gripper_connection_error_releases_runtime_state(self):
        with mock.patch.dict(os.environ, {'VLA_MODEL_VALIDATION_ONLY': '0'}), \
                mock.patch.object(app, 'RobotConnection') as connection, \
                mock.patch.object(app, 'sleep'):
            connection.return_value.execute.side_effect = RuntimeError('fixture acknowledgement lost')
            self.assertEqual(self.request('/api/gripper', {'side': 'right', 'action': 'close'})[0], 502)
            connection.return_value.execute.assert_called_once()
            connection.return_value.close.assert_called_once()
            self.assertFalse(app.run_lock.locked())
            self.assertIsNone(app.current_operation)
            connection.return_value.execute.side_effect = None
            self.assertEqual(self.request('/api/gripper', {'side': 'left', 'action': 'open'})[0], 200)

    def test_defaults_on_three_no_data_created(self):
        status, raw = self.request("/api/config")
        result = json.loads(raw)
        self.assertEqual(status, 200)
        self.assertTrue(result["config"]["human_takeover_enabled"])
        self.assertEqual(result["config"]["max_interventions"], 3)
        self.assertEqual(result["config"]["data_dir"], "/data/yfmo")
        self.assertFalse(any(self.root.iterdir()))
        self.assertFalse(result["capabilities"]["data_collection"])

    def test_save_roundtrip_zero_unlimited_and_custom_path(self):
        for limit in (0, 3, 10000, None):
            body = {**config.defaults(), "max_interventions": limit, "data_dir": "/data/yfmo/chosen"}
            status, raw = self.request("/api/config", body)
            self.assertEqual(status, 200, raw)
            self.assertEqual(config.load(), body)
        self.assertFalse((self.root / "chosen").exists())
        self.assertEqual(len(list(self.root.glob(".web-settings-*"))), 0)

    def test_bad_config_does_not_overwrite_previous(self):
        config.save(config.defaults())
        before = config.CONFIG_PATH.read_bytes()
        for bad in (-1, 0.5, True, "3"):
            status, raw = self.request("/api/config", {**config.defaults(), "max_interventions": bad})
            self.assertEqual(status, 400, raw)
        self.assertEqual(self.request("/api/config", {**config.defaults(), "data_dir": "relative/path"})[0], 400)
        self.assertEqual(self.request("/api/config", [1, 2])[0], 400)
        self.assertEqual(config.CONFIG_PATH.read_bytes(), before)

    def test_robot_ip_save_routes_clothes_grippers_and_records_to_saved_target(self):
        target = "192.0.2.70"
        transport = Transport()
        with mock.patch.object(app, "RobotConnection") as connection, \
                mock.patch.object(app, "ensure_remote_env_running") as prepare, \
                mock.patch.object(app.subprocess, "Popen") as spawn:
            body = {**config.defaults(), "robot_host": target}
            status, raw = self.request("/api/config", body)
            self.assertEqual(status, 200, raw)
            self.assertEqual(config.load()["robot_host"], target)
            self.assertEqual(network_config.get_robot_host(), target)
            self.assertEqual(transport.host, target)
            state = json.loads(self.request("/api/config")[1])
            for readiness in state["clothesReadiness"].values():
                self.assertEqual(readiness["robot"]["host"], target)
                self.assertFalse(readiness["can_reset_pose"])
            prepare.assert_not_called()
            connection.assert_not_called()
            spawn.assert_not_called()
            status, raw = self.request("/api/clothes/gripper", {
                "taskId": clothes_task.HANG_TASK_ID, "side": "left", "action": "open"})
            self.assertEqual(status, 200, raw)
            self.assertEqual(connection.call_args.args[0]["host"], target)
            command = connection.return_value.execute.call_args.args[0]
            self.assertIn("left_effector", command)
            self.assertNotIn("right_effector", command)
            self.assertNotIn("left_arm", command)

    def test_invalid_robot_ips_leave_saved_target_unchanged(self):
        config.save(config.defaults())
        before = config.CONFIG_PATH.read_bytes()
        for address in ("", "not-an-ip", "192.168.50.999", "http://192.168.50.74", "0.0.0.0", "127.0.0.1", "224.0.0.1"):
            status, raw = self.request("/api/config", {**config.defaults(), "robot_host": address})
            self.assertEqual(status, 400, raw)
            self.assertEqual(config.CONFIG_PATH.read_bytes(), before)

    def test_robot_ip_can_be_saved_in_model_validation_mode(self):
        with mock.patch.dict(os.environ, {"VLA_MODEL_VALIDATION_ONLY": "1"}):
            body = {**config.defaults(), "robot_host": "192.0.2.71"}
            status, raw = self.request("/api/config", body)
            self.assertEqual(status, 200, raw)
            self.assertEqual(json.loads(self.request("/api/config")[1])["config"]["robot_host"], "192.0.2.71")

    def test_remote_directory_is_not_created_on_web_host(self):
        with mock.patch.object(Path, "mkdir", side_effect=AssertionError("must not create robot directory on web host")):
            status, raw = self.request("/api/config", {**config.defaults(), "data_dir": "/data/yfmo/custom"})
            self.assertEqual(status, 200, raw)
        self.assertEqual(config.load()["data_dir"], "/data/yfmo/custom")
        before = config.CONFIG_PATH.read_bytes()
        self.assertEqual(self.request("/api/config", {**config.defaults(), "data_dir": "C:/local/path"})[0], 400)
        self.assertEqual(config.CONFIG_PATH.read_bytes(), before)

    def test_busy_backend_rejects_settings_and_second_task(self):
        app.run_lock.acquire()
        try:
            self.assertEqual(self.request("/api/config", config.defaults())[0], 409)
            self.assertTrue(json.loads(self.request("/api/config")[1])["locked"])
            self.assertEqual(self.request("/api/run-task", {"taskId": next(iter(app.TASKS))})[0], 409)
        finally:
            app.run_lock.release()
        self.assertFalse(config.CONFIG_PATH.exists())

    def test_unavailable_mode_never_starts_robot(self):
        config.save({**config.defaults(), "human_takeover_enabled": True})
        with mock.patch.object(app.subprocess, "Popen") as popen:
            self.assertEqual(self.request("/api/run-task", {"taskId": next(iter(app.TASKS))})[0], 409)
            popen.assert_not_called()
        self.assertFalse(app.run_lock.locked())
        self.assertIsNone(app.current_task_id)

    def test_routes_and_removed_record_page(self):
        for route in ("/", "/intergrated_test.html"):
            status, html = self.request(route)
            self.assertEqual(status, 200)
            self.assertIn('id="runPanel"', html)
            self.assertIn('id="dataPanel"', html)
            self.assertNotIn("打开测试记录", html)
            self.assertNotIn("localStorage", html)
        for route in ("/drawer_test.html", "/drawer_test_record.html", "/drawer_test_record.html.backup"):
            self.assertEqual(self.request(route)[0], 404)
        self.assertEqual(json.loads(self.request("/api/tasks")[1])["tasks"], app.TASKS)

    def test_all_task_dispatch_paths_preserved_without_robot(self):
        config.save({**config.defaults(), "human_takeover_enabled": False})
        expected = {"pick": "run_pick.py", "place": "run_place.py", "water": "run_water.py", "washer": "run_washer.py", "drawer": "run_drawer.py"}
        for task_id, task in app.TASKS.items():
            if task.get("inferenceOnly"):
                continue  # The model-only dispatch has separate robot-free regression tests.
            process = mock.Mock(returncode=0)
            process.communicate.return_value = ("offline test", "")
            with mock.patch.object(app.subprocess, "Popen", return_value=process) as popen, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(app.run_robot_task(task_id)[0], 200)
                command = popen.call_args.args[0]
                self.assertEqual(Path(command[1]), app.BASE_DIR / expected[task["type"]])
                self.assertTrue(Path(command[1]).is_file())
                self.assertEqual(command[2:], ["--task", task_id])
        self.assertFalse(app.run_lock.locked())
        self.assertFalse((self.root / "datas").exists())

    def test_running_and_restoring_lock_status(self):
        config.save({**config.defaults(), "human_takeover_enabled": False})
        task_id = next(iter(app.TASKS))
        def inspect_running():
            state = app.configuration_status()
            self.assertTrue(state["locked"])
            self.assertEqual(state["taskId"], task_id)
            self.assertEqual(state["operation"], "run")
            self.assertEqual(self.request("/api/config", config.defaults())[0], 409)
            return "", ""
        process = mock.Mock(returncode=0)
        process.communicate.side_effect = inspect_running
        with mock.patch.object(app.subprocess, "Popen", return_value=process), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(app.run_robot_task(task_id)[0], 200)
        def inspect_restore(command, **kwargs):
            self.assertEqual(app.configuration_status()["operation"], "restore")
            self.assertEqual(self.request("/api/config", config.defaults())[0], 409)
            self.assertEqual(Path(command[1]), app.BASE_DIR / "restore.py")
            return mock.Mock(returncode=0, stdout="", stderr="")
        with mock.patch.object(app.subprocess, "run", side_effect=inspect_restore), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(app.restore_robot_task(task_id)[0], 200)
        self.assertFalse(app.run_lock.locked())
        self.assertIsNone(app.current_operation)

    def test_launch_failure_releases_configuration_lock(self):
        config.save({**config.defaults(), "human_takeover_enabled": False})
        with mock.patch.object(app.subprocess, "Popen", side_effect=OSError("fixture")), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(app.run_robot_task(next(iter(app.TASKS)))[0], 500)
        self.assertFalse(app.run_lock.locked())
        self.assertIsNone(app.current_task_id)

    def test_invalid_task_never_spawns(self):
        with mock.patch.object(app.subprocess, "Popen") as popen:
            for task in ("unknown", [], None):
                self.assertEqual(self.request("/api/run-task", {"taskId": task})[0], 400)
            popen.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
