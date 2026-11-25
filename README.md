# VLA 人工接管与采集系统

当前部署：推理主机 **192.168.50.113**；机器人当前地址 **192.168.50.118**（原 61）。113 当前通过 192.168.50 网段访问机器人，启动网页时设置 `VLA_ROBOT_HOST=192.168.50.118`。历史直连配置 10.42.0.113 → 10.42.0.101 保留在 network.json，当前用环境变量覆盖。网页沿用 [综合测试页面](http://192.168.50.113:8084/intergrated_test.html)。

## 源码同步状态（2026-11-25）

已从 113 当前部署目录同步网页、调度与执行模块、静态资源、任务配置和公共地址模块 robot_target.py；66 个文件均与远端逐文件 SHA256 核对一致，清单见 deploy/sync_snapshot_20261125.json。Git 按仓库规则将文本换行规范为 LF，清单同时记录相关文件的规范化哈希。

113 网页当前选择机器人 192.168.50.223；已使用原机器人可信主机密钥成功 SSH 登录，但几个已知部署目录中未找到这套采集服务，8086/8087 未监听。原地址 118 从本机及 113 均未连通，因此 robot/ 保留此前源码版本，本次未确认其与机器人当前部署一致。此次只更新本地，未重启远端服务、修改远端部署或发送机器人动作。

本仓库仍不包含 checkpoint、采集数据、认证文件以及原项目完整 core、openpi、AgentOS、机器人 SDK 与运行环境。原部署脚本的文件白名单保持原范围，未扩展为新增服装任务和公共模块的完整迁移包；迁移整套系统时须单独核对依赖。

## 操作流程

1. 打开 PICO 数采程序，连接机器人，松开左右接管键；网页“测试 VR”应显示新鲜数据和未接管。
2. 在“运行测试”选择任务。统一采集模式默认开启；当前人工模式支持四项洗衣机任务，其他分类保留原有普通运行流程。
3. 点击开始：检查模型身份、机器人和 VR → 原生录制器保存原始消息及相机数据 → 确认文件已有负载后，开始模型任务及其原有初始化。
4. 按住任一接管键进入人工纠正；双手全部释放且状态连续有效 300 ms 后，重新读取观测并恢复模型，不复用旧预测。
5. 网页手动暂停不会因松键解除。断连、未知、观察服务重启、故障、未完成初始化或接管超限均不会自动恢复。
6. 操作完成后，在运行页中止并点击“失败”。先结束原始录制，当前窗口随即列出本次所有接管段落；填写期间不恢复模型，也不继续录制。
7. 点击各段红色按钮，选择接管原因和/或填写具体说明。这里仅保留当前页面草稿，不自动提交；没有接管段时填写整条失败说明。
8. 全部段落填写完整后，点击“保存本条数据”：提交失败结果与全部段落标注，确认模型记录传输完整后归档。保存失败可重试，已提交的部分不会重复生成录制。历史草稿仍可在“采集数据”页补标归档；成功按钮沿用原有结果判定流程。

数据列表为历史记录保留独立的轻量摘要缓存，不再因只缓存八条完整时间轴而反复解析所有动作文件；文件变化会重新校验。机器人列表连接异常时先返回可读的历史记录并说明原因。列表与详情的前端读取窗口为30秒，读取超时不会被显示为零条数据。

原生录制窗口最大一小时。系统在窗口到期前停止新的模型发送并结束录制，转入人工结果标注；已经接受的轨迹不因此撤回。

## 数据保存

~~~text
机器人 /data/yfmo/sessions/
  类别_稳定ID/
    任务_稳定ID/
      YYYY-MM-DD/
        episode_id/
          raw/
            record/*.pbdat
            camera/<相机>/*.h265 / *.zst / 时间戳文件
          model/<record_id>.json
          annotations/round_N.json
          session.json
~~~

- VR 输入本来就是原生 DDS 录制的一部分，不复制第二份 VR 文件。话题读取机器人原有 /home/agi/app/conf/record/record_topic.json。
- 相机包括八路彩色视频和 head_depth，沿用原生文件格式。
- H5 是对齐后的派生格式。新链路保留原始消息，不重复生成 H5；训练导出可基于同一会话后处理。
- 模型预测、裁剪后命令、输入状态、源时间戳、发送状态和事件保存在机器人。模型输入 state 只是离散快照。
- 113 仅保留 pending_spool 中尚未确认的临时文件，机器人哈希确认后删除本地 payload；掉线时保留并重试。重启不恢复机器人动作。
- 历史 action_spool 可只读查看，未移动或重写历史采集。
- 标注按接管轮次保存。“其他”需填写说明，也支持只写自由文本原因。结束录制与完成归档为两个阶段。
- 纳秒时间戳保持十进制字符串；保留源时间、两端 wall/monotonic/boot_id 和跨机测量。预测点不是实际执行时刻。为兼容旧浏览器，部分字段保留旧的 _114/_226 名称，实际 host 与时间域以内容为准。

## 分工与频率

~~~mermaid
flowchart LR
  subgraph H["113：推理和调度"]
    W["网页：开始、结束、结果与逐次标注"]
    V["VLA：观测输入、动作输出"]
  end
  subgraph R["机器人"]
    G["状态、VR观察与动作接口"]
    C["统一采集服务"]
    D[("按类别和任务组织的本地数据")]
    C -->|"源频率持续保存"| D
  end
  P["PICO"] -->|"原始VR流：历史约90Hz"| G
  W -->|"开始/结束：事件触发"| C
  G -->|"接管状态：名义20Hz轮询"| H
  G -->|"每轮推理一次：图像、关节、源时间"| V
  V -->|"每轮一次：动作块和推理记录"| R
~~~

当前关节模型为 30×16 动作块、参考时长 1 秒；30 点不代表每秒推理 30 次。推理/下发频率取决于实际耗时。连续关节约 100 Hz，VR 频率以设备实测为准。原始采集在模型和人工阶段持续运行。

## 代码与部署

- host/：113 当前应用基线与改造模块、页面、时间轴和网络配置。
- robot/：只读 VR 观察器、原生录制桥、分类存储和标注 API。
- config/：任务配置参考。部署不覆盖任务配置、checkpoint、prompt 或位姿。
- scripts/build_release.py：白名单打包与 SHA256 清单，不包含数据和凭据。
- scripts/apply_release.py：仅在网页空闲时部署，检查当前源码哈希并备份；不启动模型、切换机器人模式或发送动作。
- deploy/source_baseline.json：从 113 下载的源码基线哈希。
- 宿主应用仍位于 /home/lindenbot/qianzhang43/agibot_policy_ctl/g2_policy_ctl/intergrated_test，依赖原项目 core、模型环境与机器人 SDK。
- 机器人服务位于 /data/yfmo/takeover_system。8086 是只读 VR 观察器；8087 是鉴权采集和标注服务。
- 地址集中在 host/network.json；新凭据 session_receiver.token 与旧 receiver.token 独立，不提交 Git。
- PICO 原生遥操作及本体控制保持原链路。新采集服务不导入 Robot，不发运动请求。

机器人辅助服务重启命令（原服务未运行时）：

~~~bash
cd /data/yfmo/takeover_system
export LD_LIBRARY_PATH=/home/agi/app/lib:/home/agi/app/gdk/lib
python3 vr_monitor.py --host 0.0.0.0
# 在另一个终端：
python3 collection_server.py --host 0.0.0.0 --token-file session_receiver.token
~~~

未配置开机自启；重启后先查端口和日志，避免重复启动。原 remote_env 由既有流程维护。

### 网页重复启动与机器人新地址（2026-11-12）

机器人原 192.168.50.61 现为 **192.168.50.118**，SSH 用户仍为 agi，程序目录未变。
113 当前经 192.168.50 网段访问机器人；启动网页前使用 VLA_ROBOT_HOST 覆盖旧的直连地址：

~~~bash
export VLA_ROBOT_HOST=192.168.50.118
cd /home/lindenbot/qianzhang43/agibot_policy_ctl/g2_policy_ctl/intergrated_test
source /home/lindenbot/miniconda3/etc/profile.d/conda.sh
conda activate pi_05
bash start_server.sh
~~~

start_server.sh 会检查 8084。如果占用者是当前目录的 intergrated_test_server.py，且网页确认没有任务、未结束的录制或待传数据，先发送 SIGTERM，等待旧服务释放端口，再启动新服务。启动时使用当前终端的机器人地址和环境。
其他目录/程序占用、正在运行/暂停/等待结果、状态无法读取时会拒绝替换并说明原因；旧进程未按时退出时也不会强制结束。文件锁避免并发替换，Ctrl+C 会传给新网页进程。
此流程只重启网页，不启动模型或机器人执行实例。当前普通模式机器人地址同为118，原有开门任务的取消和结束接口均保留；上述环境变量覆盖用于人工接管采集链路。

Linux 下可单独运行隔离端口的启动/替换验证（不连接机器人）：

~~~bash
python -m unittest discover -s tests -p 'test_web_restart.py' -v
~~~

## 验证

~~~bash
python -m unittest discover -s tests -p 'test_*.py' -q
python -m unittest discover -s host -p 'test_a1.py' -q
python -m compileall -q host robot
node --check host/data_browser.js
~~~

浏览器替身：先运行 python tests/preview.py，再运行 node tests/browser.cjs。可用 PLAYWRIGHT_MODULE 和 PLAYWRIGHT_CHANNEL 指定本机安装。替身仅绑定 127.0.0.1，不连接机器人。

已验证分类、逐段标注、漏标归档拦截、跨区间草稿、刷新持久化、松键恢复、断连和手动暂停、迟到推理、鉴权、哈希确认和重试。真实短录制已验证 DDS、八路视频、深度和两类结束成功回执；当时 VR 无输入帧，原始 VR 文件为空，诊断数据不作为正式任务。没有执行初始化、复位或模型动作。
