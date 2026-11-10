"""Explicit washer allowlist; each episode resolves its own configured policy."""
import json
import os
import socket
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[3]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
from network_config import get_policy_host, configure_proxy_bypass

TASK_IDS = ('双臂关洗衣机门', '左臂放浅蓝色衣服进洗衣机', '右臂开洗衣机门', '左臂从洗衣机拿衣服')
HOMECARE_ROBOT_HOST = '192.168.50.61'
MOTION_UNKNOWN_EXIT = 2
CONFIG_PATH = Path(__file__).resolve().parent.parent / 'config' / 'config_V2_0.json'


def single_task_config():
    """Opt in only after per-task timings and blocking control are validated."""
    raw = os.environ.get("WASHER_SINGLE_TASK_CONFIG")
    if raw is None:
        return None
    config = json.loads(raw)
    if not isinstance(config, dict) or set(config) != {"runSeconds", "maxDurationS"}:
        raise ValueError("限时洗衣机配置必须包含 runSeconds 和 maxDurationS")
    durations = config["runSeconds"]
    limit = config["maxDurationS"]
    if (not isinstance(durations, dict) or set(durations) != set(TASK_IDS)
            or any(type(seconds) not in (int, float) or not 1 < seconds < 60
                   for seconds in durations.values())
            or type(limit) not in (int, float) or not max(durations.values()) + 2 < limit < 60):
        raise ValueError("须配置四项运行时间，并为准备、通信和收尾留出余量；总时限须小于 60 秒")
    return config


def policy_for(task_id, config_path=None):
    if task_id not in TASK_IDS:
        raise ValueError('人工接管模式当前仅开放洗衣机分类任务')
    tasks = json.loads(Path(config_path or CONFIG_PATH).read_text(encoding='utf-8'))
    task = tasks.get(task_id)
    if not isinstance(task, dict):
        raise ValueError('缺少任务配置：' + task_id)
    host, port = configure_proxy_bypass(get_policy_host()), task.get('policy_port')
    if not isinstance(host, str) or not host.strip() or type(port) is not int or not 1 <= port <= 65535:
        raise ValueError('任务模型地址或端口无效：' + task_id)
    if not isinstance(task.get('prompt'), str) or not task['prompt'].strip() or not task.get('init_position'):
        raise ValueError('任务缺少prompt或初始位姿：' + task_id)
    return {'host':host, 'port':port, 'config':task.get('policy_config'),
            'checkpoint':task.get('checkpoint'), 'asset_id':task.get('asset_id'),
            'action_shape':[30,16], 'task_id':task_id, 'action_space':'joint_absolute',
            'configuration_source':str(Path(config_path or CONFIG_PATH))}


def _option(argv, name):
    for i, value in enumerate(argv):
        if value == name and i+1 < len(argv):
            return argv[i+1]
        if value.startswith(name+'='):
            return value[len(name)+1:]
    return None


def _local_models(proc_root=Path('/proc')):
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            argv = [part.decode('utf-8', errors='replace') for part in
                    (entry/'cmdline').read_bytes().split(b'\0') if part]
        except (OSError, PermissionError):
            continue
        if not any(Path(arg).name == 'serve_policy.py' for arg in argv):
            continue
        try:
            port = int(_option(argv, '--port'))
        except (TypeError, ValueError):
            continue
        yield {'port':port, 'checkpoint':_option(argv, '--policy.dir'),
               'asset_id':_option(argv, '--policy.asset-id'),
               'config':_option(argv, '--policy.config')}


def resolve_running_policy(
    policy, proc_root=Path('/proc'), connect=socket.create_connection, *, require_unique_port=False,
):
    """Select a locally running model with this task's exact identity, before an episode exists."""
    if policy['host'] not in {get_policy_host(), '127.0.0.1', 'localhost'}:
        return policy
    checkpoint = policy.get('checkpoint')
    asset = policy.get('asset_id')
    if not checkpoint or not asset:
        raise ValueError('任务缺少模型 checkpoint 或 asset_id：'+policy['task_id'])
    models = list(_local_models(proc_root))
    matches = [m for m in models if m['checkpoint'] and
               os.path.realpath(m['checkpoint']) == os.path.realpath(checkpoint) and
               m['asset_id'] == asset and
               (not policy.get('config') or m['config'] == policy['config'])]
    if not matches:
        ports = ', '.join(str(m['port']) for m in models) or '无'
        raise ValueError(policy['host']+' 未运行该任务的匹配模型：'+policy['task_id']+
                         '；当前模型端口：'+ports+'；请检查 checkpoint 和 asset_id')
    selected = next((m for m in matches if m['port'] == policy['port']), None)
    if selected is None:
        if len(matches) != 1:
            raise ValueError('匹配模型同时运行在多个端口，无法自动选择：'+policy['task_id'])
        selected = matches[0]
    if require_unique_port and sum(model['port'] == selected['port'] for model in models) != 1:
        raise ValueError(f"多个模型进程声明端口 {selected['port']}，请为各模型分配独立端口")
    try:
        connection = connect((policy['host'], selected['port']), timeout=.7)
        connection.close()
    except OSError as exc:
        raise ValueError('匹配模型进程存在，但端口 '+str(selected['port'])+' 不可连接') from exc
    result = dict(policy)
    result['configured_port'] = policy['port']
    result['port'] = selected['port']
    return result
