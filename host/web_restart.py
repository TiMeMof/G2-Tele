"""Replace only an identified, idle instance of this application's web server.

Linux launcher. Never signals a model, robot service, or an unknown port owner.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from urllib.request import build_opener, ProxyHandler


class StartError(RuntimeError):
    pass


class OwnerUnknown(StartError):
    pass


def listener_pid(port):
    # ss snapshots file descriptors and TCP sockets separately. A newly bound or
    # closing socket can briefly have no PID; retry before treating it as unknown.
    for attempt in range(3):
        try:
            return _listener_pid(port)
        except OwnerUnknown:
            if attempt == 2:
                raise
            time.sleep(.05)


def _listener_pid(port):
    result = subprocess.run(
        ["ss", "-H", "-lntp", "sport = :" + str(port)],
        capture_output=True, text=True, timeout=3, check=True,
    )
    rows = [row for row in result.stdout.splitlines() if row.strip()]
    if not rows:
        return None
    pids = set()
    for row in rows:
        found = re.findall(r"\bpid=(\d+)\b", row)
        if not found:
            raise OwnerUnknown(f"端口 {port} 已占用，但无法确认所属 PID；请手动检查，不自动结束进程。")
        pids.update(map(int, found))
    if len(pids) != 1:
        raise StartError(f"端口 {port} 涉及多个进程 {sorted(pids)}；请手动检查。")
    return pids.pop()


def process_signature(pid, app_dir):
    proc = Path("/proc") / str(pid)
    try:
        if proc.stat().st_uid != os.geteuid():
            raise StartError(f"PID {pid} 属于其他用户，不自动结束。")
        cwd = (proc / "cwd").resolve(strict=True)
        argv = (proc / "cmdline").read_bytes().decode(errors="replace").split("\0")
        if cwd != app_dir or not re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", Path(argv[0]).name):
            raise StartError(f"端口占用者 PID {pid} 不是本目录的网页服务，不自动结束。")
        index = 1
        while index < len(argv) and argv[index] in {"-u", "-B", "-E", "-s", "-S", "-I", "-O", "-OO", "-q"}:
            index += 1
        if index >= len(argv) or not argv[index] or argv[index].startswith("-"):
            raise StartError(f"无法确认 PID {pid} 的网页入口，不自动结束。")
        script = Path(argv[index])
        if not script.is_absolute():
            script = cwd / script
        if script.resolve() != app_dir / "intergrated_test_server.py":
            raise StartError(f"端口占用者 PID {pid} 是其他程序，不自动结束。")
        # Field 22 is start time; parentheses in comm may contain spaces.
        started = (proc / "stat").read_text().rsplit(")", 1)[1].split()[19]
        return pid, started
    except (OSError, IndexError) as exc:
        raise StartError(f"无法核实 PID {pid} 的完整进程信息：{exc}") from exc


def require_idle(port):
    try:
        opener = build_opener(ProxyHandler({}))
        with opener.open(f"http://127.0.0.1:{port}/api/config", timeout=5) as response:
            data = json.load(response)
    except (OSError, ValueError) as exc:
        raise StartError(f"无法确认旧网页是否空闲，不自动结束：{exc}") from exc
    if not isinstance(data, dict) or data.get("success") is not True or data.get("locked") is not False:
        raise StartError("旧网页状态不可确认，或已有任务占用；请先在网页结束任务。")
    if data.get("taskId") or data.get("operation") or data.get("washerMotionUnknown"):
        raise StartError("旧网页仍有任务、复位或未确认的动作状态；请先处理当前任务。")
    episode = data.get("episode")
    if episode is not None:
        if not isinstance(episode, dict) or episode.get("phase") not in {"succeeded", "failed"}:
            raise StartError("当前采集任务尚未结束，包括暂停或等待结果；请先完成结果判定。")
        if episode.get("inflight") or episode.get("inference_busy"):
            raise StartError("当前任务仍有动作请求或推理在处理中，暂不能重启。")
        recording = episode.get("recording")
        if recording is not None and not isinstance(recording, dict):
            raise StartError("无法读取当前录制状态，暂不能重启。")
        if episode.get("persistent_recording") and not isinstance(recording, dict):
            raise StartError("无法确认当前录制已结束，暂不能重启。")
        if recording and (recording.get("collection_stopped") is not True or recording.get("pending_records", 0)):
            raise StartError("录制尚未确认结束或仍有数据待传输，请等待收尾后重启。")
    if data.get("recording_error"):
        raise StartError("当前录制存在错误，请先核对录制与数据状态后重启。")


def stop_previous(app_dir, port, timeout):
    pid = listener_pid(port)
    if pid is None:
        return
    identity = process_signature(pid, app_dir)
    # A stable process handle prevents accidentally signalling a reused PID.
    pidfd = os.pidfd_open(pid) if hasattr(os, "pidfd_open") else None
    try:
        require_idle(port)
        if listener_pid(port) != pid or process_signature(pid, app_dir) != identity:
            raise StartError("端口占用者在检查中发生变化，请重新执行启动命令。")
        print(f"端口 {port} 旧网页已确认空闲，正在结束 PID {pid}（SIGTERM）…", flush=True)
        if pidfd is not None and hasattr(signal, "pidfd_send_signal"):
            signal.pidfd_send_signal(pidfd, signal.SIGTERM)
        else:
            os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + timeout
        while True:
            try:
                owner = listener_pid(port)
            except OwnerUnknown:
                if time.monotonic() >= deadline:
                    raise StartError(f"端口 {port} 未在 {timeout:g} 秒内释放，且无法确认占用者；不强制结束任何进程。")
                time.sleep(.1)
                continue
            if owner is None:
                return
            if owner != pid:
                raise StartError("旧服务退出后端口被另一个进程占用；不结束新占用者。")
            if time.monotonic() >= deadline:
                raise StartError(f"旧网页 PID {pid} 未在 {timeout:g} 秒内退出；未强制 kill，也未启动第二个实例。")
            time.sleep(.1)
    finally:
        if pidfd is not None:
            os.close(pidfd)


@contextmanager
def startup_lock(app_dir):
    import fcntl
    with (app_dir / ".web_start.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise StartError("另一个启动命令正在处理，请稍后重试。") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def launch(app_dir, port, stop_timeout=8, startup_timeout=20):
    app_dir = Path(app_dir).resolve(strict=True)
    server = app_dir / "intergrated_test_server.py"
    compile(server.read_text(encoding="utf-8"), str(server), "exec")
    child = None

    def forward(signum, _frame):
        if child is None:
            raise KeyboardInterrupt
        if child.poll() is None:
            try:
                child.send_signal(signum)
            except ProcessLookupError:
                pass

    handlers = {sig: signal.signal(sig, forward) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        with startup_lock(app_dir):
            stop_previous(app_dir, port, stop_timeout)
            child = subprocess.Popen(
                [sys.executable, "-u", str(server)], cwd=app_dir, start_new_session=True,
            )
            deadline = time.monotonic() + startup_timeout
            while True:
                if child.poll() is not None:
                    raise StartError(f"新网页启动失败，退出码 {child.returncode}；请查看上方错误。")
                try:
                    owner = listener_pid(port)
                except OwnerUnknown:
                    owner = None  # Wait for stable metadata; never signal this owner.
                if owner == child.pid:
                    break
                if owner is not None:
                    raise StartError(f"端口 {port} 被其他进程抢占；请检查 PID {owner}。")
                if time.monotonic() >= deadline:
                    raise StartError(f"新网页未在 {startup_timeout:g} 秒内开始监听。")
                time.sleep(.1)
        print(f"网页服务已启动，PID {child.pid}；按 Ctrl+C 停止。", flush=True)
        result = child.wait()
        return result if result >= 0 else 128 - result
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Only the new process created here, never an existing port owner.
                child.kill()
                child.wait()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--port", type=int, default=8084)
    parser.add_argument("--stop-timeout", type=float, default=8)
    parser.add_argument("--startup-timeout", type=float, default=20)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or min(args.stop_timeout, args.startup_timeout) <= 0:
        parser.error("port and timeouts must be positive and valid")
    try:
        return launch(args.app_dir, args.port, args.stop_timeout, args.startup_timeout)
    except KeyboardInterrupt:
        return 130
    except (StartError, OSError, subprocess.SubprocessError, SyntaxError) as exc:
        print("启动已停止：" + str(exc), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
