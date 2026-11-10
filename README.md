# VLA 人工接管与采集系统

当前部署：推理主机 113；机器人维护地址 192.168.50.61，新接管链路通过已核实的有线地址 **10.42.0.101** 通信。113 本地有线地址为 10.42.0.113。网页沿用 [综合测试页面](http://192.168.50.113:8084/intergrated_test.html)。

## 操作流程

1. 打开 PICO 数采程序，连接机器人，松开左右接管键；网页“测试 VR”应显示新鲜数据和未接管。
2. 在“运行测试”选择任务。统一采集模式默认开启；当前人工模式支持四项洗衣机任务，其他分类保留原有普通运行流程。
3. 点击开始：检查模型身份、机器人和 VR → 原生录制器保存原始消息及相机数据 → 确认文件已有负载后，开始模型任务及其原有初始化。
4. 按住任一接管键进入人工纠正；双手全部释放且状态连续有效 300 ms 后，重新读取观测并恢复模型，不复用旧预测。
5. 网页手动暂停不会因松键解除。断连、未知、观察服务重启、故障、未完成初始化或接管超限均不会自动恢复。
6. 操作完成后，在运行页中止并选择成功或失败，结束本条录制。数据先保存；后续填写原因不会延长录制。
7. 打开“采集数据”，点击每段红色区间，选择接管原因和/或填写具体说明。编辑后约 500 ms 自动保存，也可以点击保存。
8. 全部接管区间已标注、整条结果已填写、录制器确认结束且模型记录传输完整后，点击“完成归档”。

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

## 验证

~~~bash
python -m unittest discover -s tests -p 'test_*.py' -q
python -m unittest discover -s host -p 'test_a1.py' -q
python -m compileall -q host robot
node --check host/data_browser.js
~~~

浏览器替身：先运行 python tests/preview.py，再运行 node tests/browser.cjs。可用 PLAYWRIGHT_MODULE 和 PLAYWRIGHT_CHANNEL 指定本机安装。替身仅绑定 127.0.0.1，不连接机器人。

已验证分类、逐段标注、漏标归档拦截、跨区间草稿、刷新持久化、松键恢复、断连和手动暂停、迟到推理、鉴权、哈希确认和重试。真实短录制已验证 DDS、八路视频、深度和两类结束成功回执；当时 VR 无输入帧，原始 VR 文件为空，诊断数据不作为正式任务。没有执行初始化、复位或模型动作。
