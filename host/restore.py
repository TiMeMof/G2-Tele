import json
import sys
from pathlib import Path
from typing import Any, Sequence
from dataclasses import dataclass

# 保证从仓库根或 g2_policy_ctl 目录运行时都能导入 core 包
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from core.corobot_env.g2_corobot_env import G2CorobotEnv


@dataclass(frozen=True)
class TaskSpec:
    task_name: str
    prompt: str
    init_position: Any
    end_position: Any
    arm_type: str


class TaskConfigLoader:
    """任务配置加载器，用于从 JSON 文件中加载任务配置"""

    def __init__(self, config_paths: Sequence[Path | str]):
        self.config_paths = [Path(path) for path in config_paths]
        self._configs = []
        for path in self.config_paths:
            with path.open("r", encoding="utf-8") as handle:
                self._configs.append((path, json.load(handle)))

    def list_available_tasks(self) -> list[str]:
        """列出所有可用的任务名称"""
        tasks = []
        for _, config in self._configs:
            for task_name in config.keys():
                if task_name == "task":
                    continue
                if isinstance(config[task_name], dict):
                    tasks.append(task_name)
        return sorted(tasks)

    def resolve_task(self, task_name: str) -> TaskSpec:
        """根据任务名称解析任务配置，返回 TaskSpec 对象"""
        normalized_name = task_name.strip()
        if not normalized_name:
            raise ValueError("task_name 不能为空")

        for path, config in self._configs:
            if normalized_name in config:
                task_config = config[normalized_name]
                if not isinstance(task_config, dict):
                    continue
                if not {"prompt", "init_position", "end_position"}.issubset(task_config.keys()):
                    continue
                arm_type = self._infer_arm_type(path, normalized_name)
                return TaskSpec(
                    task_name=normalized_name,
                    prompt=task_config["prompt"],
                    init_position=task_config["init_position"],
                    end_position=task_config["end_position"],
                    arm_type=arm_type,
                )

        available = ", ".join(self.list_available_tasks())
        raise KeyError(f"未找到任务 '{task_name}'。可用任务：{available}")

    @staticmethod
    def _infer_arm_type(path: Path, task_name: str) -> str:
        """根据任务配置文件路径和任务名称推断机械臂类型"""
        lower_name = f"{path.name} {task_name}".lower()
        if "both" in lower_name or "双臂" in task_name:
            return "both_arms"
        if "left" in lower_name or "左臂" in task_name:
            return "left_arm"
        if "right" in lower_name or "右臂" in task_name:
            return "right_arm"
        raise ValueError(f"无法从任务 '{task_name}' 推断机械臂类型，请使用包含 left 或 right 的任务名")


def restore_to_init_position(task_name: str):
    """恢复到指定任务的初始位置"""

    # 加载任务配置（从 g2_policy_ctl/config 下的左右臂抽屉配置读取）
    loader = TaskConfigLoader([
        project_root / "config" / "config_V2_0.json",
        # project_root / "config" / "left_add_drawer_config.json",
        # project_root / "config" / "right_add_drawer_config.json",
        # project_root  / "config" / "left_config_new.json",
        # project_root  / "config" / "right_config_new.json",
    ])

    try:
        task_spec = loader.resolve_task(task_name)
    except KeyError as e:
        print(f"错误: {e}")
        return False

    try:
        env = G2CorobotEnv()
        print("连接成功")
    except Exception as e:
        print(f"连接失败: {e}")
        return False

    try:
        print("开始恢复")
        env.reset_pos(task_spec.init_position)
        print("复位成功")
        return True
    except Exception as e:
        print(f"复位失败: {e}")
        return False


def main():
    """主函数"""
    # 支持命令行参数
    if len(sys.argv) > 1:
        task_name = sys.argv[1]
    else:
        # 默认任务：left_arm_pick_omeprazole 或 left_arm_pick_vitamin_c
        task_name = "left_arm_to_get_cold_water"

    success = restore_to_init_position(task_name)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
