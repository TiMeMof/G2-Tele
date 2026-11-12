"""Linux process tests on isolated ports; never import or contact robot code."""
import json
import os
from pathlib import Path
import select
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from urllib.request import build_opener, ProxyHandler

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "host/web_restart.py"
FIXTURE = r'''
import json, os, signal, time
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
if os.environ.get("WEB_TEST_IGNORE_TERM"):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
time.sleep(float(os.environ.get("WEB_TEST_DELAY", "0")))
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def do_GET(self):
        value = json.loads(Path("state.json").read_text())
        value["fixture_pid"] = os.getpid()
        value["fixture_robot"] = os.environ.get("VLA_ROBOT_HOST")
        raw = json.dumps(value).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)
ThreadingHTTPServer(("127.0.0.1", int(os.environ["WEB_TEST_PORT"])), Handler).serve_forever()
'''


@unittest.skipUnless(sys.platform == "linux" and shutil.which("ss"), "requires Linux /proc and ss")
class WebRestartTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="vla_web_restart_")
        self.app = Path(self.temp.name) / "app with spaces"
        self.app.mkdir()
        (self.app / "intergrated_test_server.py").write_text(FIXTURE)
        self.state = {"success": True, "locked": False, "taskId": None,
                      "operation": None, "episode": None, "recording_error": None}
        self.save()
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]
        self.procs = []
        self.seen = set()
        self.logs = []
        self.env = {**os.environ, "WEB_TEST_PORT": str(self.port), "VLA_ROBOT_HOST": "192.168.50.118"}
        self.addCleanup(self.cleanup)

    def save(self):
        (self.app / "state.json").write_text(json.dumps(self.state))

    def cleanup(self):
        for proc in reversed(self.procs):
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2)
            if proc.stdout is not None:
                proc.stdout.close()
        for pid in self.seen:
            proc = Path("/proc") / str(pid)
            try:
                if (proc / "cwd").resolve() == self.app:
                    os.kill(pid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass
        for stream in self.logs:
            stream.close()
        self.temp.cleanup()

    def start_old(self, directory=None, script="intergrated_test_server.py", extra=None):
        directory = directory or self.app
        log = (Path(self.temp.name) / ("old-" + str(len(self.logs)) + ".log")).open("w+")
        self.logs.append(log)
        proc = subprocess.Popen([sys.executable, "-u", str(directory / script)],
                                cwd=directory, env={**self.env, **(extra or {})},
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        self.procs.append(proc)
        self.ready()
        return proc

    def launcher(self, extra=None):
        proc = subprocess.Popen([sys.executable, "-u", str(LAUNCHER),
                                 "--app-dir", str(self.app), "--port", str(self.port),
                                 "--stop-timeout", ".5", "--startup-timeout", "5"],
                                env={**self.env, **(extra or {})},
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, start_new_session=True)
        self.procs.append(proc)
        return proc

    def status(self):
        with build_opener(ProxyHandler({})).open(f"http://127.0.0.1:{self.port}/api/config", timeout=.5) as r:
            value = json.load(r)
        self.seen.add(value["fixture_pid"])
        return value

    def ready(self, different_from=None):
        deadline = time.monotonic() + 6
        while time.monotonic() < deadline:
            try:
                value = self.status()
                if value["fixture_pid"] != different_from:
                    return value
            except (OSError, ValueError):
                pass
            time.sleep(.05)
        details = []
        for proc in self.procs:
            details.append("pid=" + str(proc.pid) + " returncode=" + str(proc.poll()))
            if proc.stdout is not None:
                chunks = []
                while select.select([proc.stdout], [], [], 0)[0]:
                    raw = os.read(proc.stdout.fileno(), 4096)
                    if not raw:
                        break
                    chunks.append(raw)
                details.append(b"".join(chunks).decode(errors="replace"))
        for stream in self.logs:
            stream.flush()
            stream.seek(0)
            details.append(stream.read())
        self.fail("isolated fixture did not start on " + str(self.port) + ": " + " | ".join(details))

    def refused(self, old):
        proc = self.launcher()
        output = proc.communicate(timeout=8)[0]
        self.assertNotEqual(proc.returncode, 0, output)
        self.assertIsNone(old.poll(), output)
        self.assertEqual(self.status()["fixture_pid"], old.pid)
        return output

    def startup_completed(self, proc):
        # HTTP can bind before the launcher releases its startup lock. Wait for
        # the same completion message an operator sees before starting again.
        deadline = time.monotonic() + 5
        output = b""
        while time.monotonic() < deadline:
            if select.select([proc.stdout], [], [], .2)[0]:
                part = os.read(proc.stdout.fileno(), 4096)
                if not part:
                    break
                output += part
                if b"Ctrl+C" in output:
                    return
            if proc.poll() is not None:
                break
        self.fail("launcher did not complete startup: " + output.decode(errors="replace"))

    def test_idle_server_replaced_and_new_robot_environment_inherited(self):
        old = self.start_old()
        launched = self.launcher()
        value = self.ready(different_from=old.pid)
        self.assertEqual(old.wait(timeout=2), -signal.SIGTERM)
        self.assertEqual(value["fixture_robot"], "192.168.50.118")
        self.assertIsNone(launched.poll())
        self.startup_completed(launched)
        # A subsequent startup may replace the new instance as well.
        next_launch = self.launcher()
        newest = self.ready(different_from=value["fixture_pid"])
        self.assertNotEqual(newest["fixture_pid"], old.pid)
        self.assertIsNone(next_launch.poll())
        launched.wait(timeout=2)

    def test_running_paused_and_awaiting_result_are_not_killed(self):
        old = self.start_old()
        for phase in ["running", "paused", "awaiting_result"]:
            with self.subTest(phase=phase):
                self.state.update(locked=True, episode={"phase": phase})
                self.save()
                self.refused(old)

    def test_recording_and_pending_transfers_block_restart_after_result(self):
        old = self.start_old()
        for stopped, pending in [(False, 0), (True, 2)]:
            with self.subTest(stopped=stopped, pending=pending):
                self.state["episode"] = {"phase": "succeeded", "persistent_recording": True,
                    "recording": {"collection_stopped": stopped, "pending_records": pending}}
                self.save()
                self.refused(old)

    def test_other_script_on_port_is_not_killed(self):
        (self.app / "unrelated_service.py").write_text(FIXTURE)
        old = self.start_old(script="unrelated_service.py")
        self.refused(old)

    def test_same_script_name_in_other_directory_is_not_killed(self):
        foreign = Path(self.temp.name) / "other app"
        foreign.mkdir()
        (foreign / "intergrated_test_server.py").write_text(FIXTURE)
        (foreign / "state.json").write_text(json.dumps(self.state))
        old = self.start_old(directory=foreign)
        self.refused(old)

    def test_unknown_status_is_not_treated_as_idle(self):
        self.state = {"success": False, "locked": False}
        self.save()
        old = self.start_old()
        self.refused(old)

    def test_nonresponsive_sigterm_does_not_force_kill_old_server(self):
        old = self.start_old(extra={"WEB_TEST_IGNORE_TERM": "1"})
        self.refused(old)

    def test_concurrent_start_does_not_race_or_kill_starting_server(self):
        first = self.launcher(extra={"WEB_TEST_DELAY": "1"})
        deadline = time.monotonic() + 2
        while not (self.app / ".web_start.lock").exists() and time.monotonic() < deadline:
            time.sleep(.01)
        second = self.launcher()
        output = second.communicate(timeout=4)[0]
        self.assertNotEqual(second.returncode, 0, output)
        self.ready()
        self.assertIsNone(first.poll())

    def test_ctrl_c_reaches_foreground_server(self):
        proc = self.launcher()
        self.ready()
        proc.send_signal(signal.SIGINT)
        proc.communicate(timeout=4)
        with self.assertRaises(OSError):
            self.status()

    def test_invalid_new_source_leaves_existing_server_running(self):
        old = self.start_old()
        (self.app / "intergrated_test_server.py").write_text("this is not valid Python :")
        self.refused(old)


if __name__ == "__main__":
    unittest.main()
