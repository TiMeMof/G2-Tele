#!/usr/bin/env python3
"""Run a clothes task, restore its saved pose, or print the local preflight."""
import argparse
from functools import partial
import json
import signal
import sys
import threading

from clothes_robot import RobotConnection, execution_readiness, execute_flow
from clothes_task import CLOTHES_TASKS, SOFA_GRASP_HANDS, TASK_ID, load_task_spec


def main():
    parser = argparse.ArgumentParser(description="衣物任务实机预检与执行")
    parser.add_argument("--task", choices=CLOTHES_TASKS, default=TASK_ID)
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument("--execute", action="store_true", help="从人工就位后的当前状态开始实机执行")
    operation.add_argument("--restore-pose", action="store_true", help="回到已保存的沙发任务姿态")
    parser.add_argument("--use-loaded-model", action="store_true", help="只连接已经启动的模型")
    parser.add_argument("--grasp-hand", choices=SOFA_GRASP_HANDS, help="选择沙发任务的抓取提示词")
    args = parser.parse_args()
    if not (args.execute or args.restore_pose):
        print(json.dumps(execution_readiness(task_id=args.task), ensure_ascii=False, indent=2))
        return 0
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    robot_factory = partial(RobotConnection, start_if_needed=True)
    if not args.restore_pose and load_task_spec(task_id=args.task).get('temporal_memory'):
        from temporal_washer import TemporalRobotConnection
        robot_factory = partial(TemporalRobotConnection, connection_factory=robot_factory)
    try:
        report = execute_flow(task_id=args.task, restore_pose=args.restore_pose, stop=stop,
                              require_loaded_model=args.use_loaded_model, grasp_hand=args.grasp_hand,
                              robot_factory=robot_factory)
    except Exception as exc:
        print(f"{CLOTHES_TASKS[args.task]}实机流程未完成：{exc}", file=sys.stderr, flush=True)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not args.restore_pose:
        print("执行流程已结束，物理任务结果仍需人工确认；中止不会撤销已提交轨迹。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
