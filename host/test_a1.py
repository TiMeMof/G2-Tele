"""A1 HTTP/configuration/dispatch tests. All robot subprocesses are mocked."""
import contextlib
import http.client
import io
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import web_config as config
import intergrated_test_server as app


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
        config.save({**config.defaults(), "human_takeover_enabled":False})
        expected = {"pick": "run_pick.py", "place": "run_place.py", "water": "run_water.py", "washer": "run_washer.py", "drawer": "run_drawer.py"}
        for task_id, task in app.TASKS.items():
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
        config.save({**config.defaults(), "human_takeover_enabled":False})
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
        config.save({**config.defaults(), "human_takeover_enabled":False})
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
