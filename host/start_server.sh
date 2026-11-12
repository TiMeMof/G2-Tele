#!/bin/bash
# Replace this application's idle web server, then keep the new one in foreground.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# websockets 16 默认会读取系统代理。Python 的代理绕过匹配不可靠地支持
# 192.168.0.0/16 这类 CIDR，因此显式加入推理服务器地址，避免内网连接被
# 转发到本机 HTTP/SOCKS 代理后在握手阶段被关闭。
# Shared with openpi/scripts/serve_policy.py and washer clients.
PROJECT_DIR="$(cd ../../.. && pwd)"
if ! INFERENCE_HOST="$(python3 "${PROJECT_DIR}/runtime_config.py")"; then
    echo "Failed to read policy_host from ${PROJECT_DIR}/runtime_config.json" >&2
    exit 1
fi
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${INFERENCE_HOST}"
export no_proxy="${no_proxy:+${no_proxy},}${INFERENCE_HOST}"

# Include the selected robot, including VLA_ROBOT_HOST overrides.
NETWORK_HOSTS="$(python3 -c 'from network_config import get_robot_host; print(get_robot_host()+",127.0.0.1,localhost")')"
export NO_PROXY="${NO_PROXY},${NETWORK_HOSTS}"
export no_proxy="${no_proxy},${NETWORK_HOSTS}"

echo "=================================="
echo "Agibot 综合任务测试服务"
echo "=================================="
echo "机器人地址：${NETWORK_HOSTS%%,*}"
echo "检查 8084；空闲的本项目旧网页服务会自动替换。"

exec python3 -u web_restart.py
