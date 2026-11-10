#!/bin/bash
# Agibot 综合任务测试服务启动脚本
# http_proxy="" https_proxy="" python run_place.py
cd "$(dirname "$0")"

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

echo "=================================="
echo "Agibot 综合任务测试服务"
echo "=================================="
echo "正在启动服务器..."
echo ""

python3 intergrated_test_server.py
