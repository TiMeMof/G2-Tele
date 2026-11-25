#!/usr/bin/env python3
"""Validate a clothing model with synthetic inputs."""
import argparse
import sys

from clothes_task import CLOTHES_TASKS, TASK_ID, run_validation, validation_report_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=CLOTHES_TASKS, default=TASK_ID)
    task_id = parser.parse_args().task
    name = CLOTHES_TASKS[task_id]
    try:
        report = run_validation(task_id=task_id)
    except Exception as exc:
        print(f"{name}模型验证失败：{exc}", file=sys.stderr, flush=True)
        return 1
    print(f"{name}模型推理验证通过（合成输入）")
    print("任务指令：" + report["prompt"])
    print("动作输出：30 × 16，全部为有限数值")
    print(f"GPU 加载 {report['model_metadata']['load_ms'] / 1000:.2f} 秒，推理 {report['policy_timing']['infer_ms']:.1f} 毫秒")
    print("机器人连接：无；机器人动作下发：0")
    print("完整验证结果：" + str(validation_report_path(task_id)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
