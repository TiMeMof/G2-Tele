# Agibot 综合测试

入口已从 drawer_test 迁移为用户指定拼写 intergrated_test。A1完成页面入口和配置；实时VR接管、采集和时间轴浏览尚未开放。

## 启动

```bash
conda activate pi_05
cd /home/lindenbot/qianzhang43/agibot_policy_ctl/g2_policy_ctl/intergrated_test
bash start_server.sh
```

访问 http://192.168.50.114:8084/ 或 /intergrated_test.html。页面分“运行测试”“采集数据”两个页签。原手填测试记录页面及专属接口已删除；快照保留旧文件，不清除浏览器历史数据。

## 运行配置

- 在「运行配置」中填写「机器人 IP 地址」并点击「保存配置」，后续测试、复位、夹爪控制和 VR 读取使用该地址，保存后无需重启网页或模型。设置保存在 `web_settings.json`。保存只更新本机配置。已有复位记录仍保留采集时的机器人地址，切换到其他地址后需为目标机器人记录复位位姿。
- 人工接管模式默认关闭：保持原普通测试脚本，不创建本方案的采集记录。
- 可保存接管模式偏好，但当前模式未就绪，开启后前后端均阻止启动测试；关闭后可运行普通测试。不能把这个开关当作已能接管机器人。
- 根据最新约束，默认采集位置为226机器人的 /data/yfmo。A1只持久化这一设置，不在114创建该目录；226实时连接及可写性检查留到A2接入后完成。
- 接管上限默认3，可填非负整数或选择不限；当前只保存配置，轮次执行逻辑在后续A3实现。
- 配置保存在 web_settings.json，服务重启后保留。测试或复位中后端拒绝修改，页面同步禁用配置。先保存更改再运行。
- 采集数据页目前为明确的未开放提示，不生成示例或虚假采集数据。

## 文件与接口

- intergrated_test_server.py：标准库HTTP服务，默认8084；启动服务本身不连接机器人。
- intergrated_test.html：页面和交互。
- web_config.py：设置格式验证与原子保存。
- run_pick/run_place/run_water/run_washer/run_drawer.py：保留原任务分发，业务控制逻辑未改。
- restore.py：保留原复位入口。
- GET /api/tasks：任务列表；GET /api/config：配置、功能可用状态与运行锁。
- POST /api/config：保存完整配置 human_takeover_enabled / data_dir / max_interventions；不限为null。
- POST /api/run-task、/api/restore-task、/api/stop-task：沿用原流程。
- 运行页的「夹爪控制」区域参考 76 的页面，提供左、右夹爪独立张开和闭合。`POST /api/gripper` 接收 `side`（`left`/`right`）和 `action`（`open`/`close`），使用页面保存的机器人 IP，不依赖模型加载。张开值为 `-0.785`、闭合值为 `0.0`，参考时间 `0.8` 秒，只发送所选夹爪的 `effector` 字段。任务卡片的夹爪开关复用这些参数。

A1没有实现A3的新中断状态机：普通测试的中断仍发送SIGINT，不能据此认定已取消机器人已接收轨迹；结果仍来自原执行脚本，不能直接证明物理任务成功。

## 离线验证

```bash
python -m unittest discover -s . -p test_a1.py -v
```

测试使用临时目录和进程替身，不连接机器人、不下发动作。

## 衣物任务模型

以下任务使用各自的模型和英文指令。卡片上的「一键启动」加载对应模型，测试结束或中止后可继续使用。离线验证报告保存在 `model_logs`。

| 任务 | 任务 ID | checkpoint 目录 | 归一化资产 |
|---|---|---|---|
| 挂衣服 | `right_arm_hang_clothes` | `xinnie_hang_on_cloth/s90000_date1118` | `hang_on_cloth_90000_261118` |
| 沙发拿浅蓝色衣服 | `left_arm_pick_blue_clothes_from_sofa` | `xinnie_sofa_pick_up_blue_cloth/s30000_data1118` | `sofa_pick_up_close_20261118` |
| 摘衣服 | `right_arm_take_off_clothes` | `xinnie_take_off_cloth/s80000_date117` | `clothes_takeoff_20261108` |
| 左臂放浅蓝色衣服进洗衣机 | `左臂放浅蓝色衣服进洗衣机` | `xinnie_put_in_washer/s32500_1122` | `s32500_1120_put_in_washer` |
| 双臂关洗衣机门 | `双臂关洗衣机门` | `xinnie_close_washer_door/s27500_1123_event` | `s27500_1123_event_wighted` |
| 双臂关洗衣机门（历史帧MEM） | `双臂关洗衣机门（历史帧MEM）` | `xinnie_close_washer_door/s30000_1123_mem` | `washer_close_261115_261116_872_20261118` |

「左臂放浅蓝色衣服进洗衣机」保留在洗衣机分类，复用摘衣服的卡片和执行流程。一键启动加载本机 8188 端口的模型，运行测试使用 `Use the left arm to put the light blue clothes into the washing machine`，双臂和两个夹爪使用模型原始输出。从人工就位后的当前观测开始，测试结束或中止后保留模型。卡片提供左右夹爪开关。复位使用 `reset_poses` 中本任务单独记录的姿态，现场记录前按钮禁用。

放衣服进洗衣机配置了 `run_until_stopped: true`，在观测、推理和执行正常时持续运行，直到操作员点击「中止」。该任务的测试循环不受 `max_chunks`、`max_duration_s` 或网页端 360 秒进程超时限制。中止会停止后续轨迹下发，模型保持加载，可直接再次运行测试。「复位」仍是一次恢复已记录姿态的操作。

双臂关洗衣机门的事件加权版在本机 8189 端口提供推理；从 76 同步的独立「双臂关洗衣机门（历史帧MEM）」卡片使用 `s30000_1123_mem`，端口为 8190。两张卡片均使用训练指令 `Use both arms to close the washing machine door.`，各自启动、关闭模型，模型和测试结果相互独立。

时序 OpenPI 源码同步自 76 的关门 MEM 模块，保存在 `/home/lindenbot/qianzhang43/openpi_mem_washer_close_76`。保留训练时的 `mem_composed_v1` 时序算子及全部历史状态权重，使用约 1 秒前和 0.5 秒前的图像、状态，采样容差为 0.15 秒。独立的只读连接在动作执行期间采集历史观测，主连接下发双臂和双夹爪原始输出。模型运行到人工中止，中止后保留加载。卡片与放衣服进洗衣机一致，复位待现场记录本任务姿态后启用。推理服务采用直接执行方式并在线程中推理，使 WebSocket 保活不受首次推理影响。

以模型验证模式启动网页后，访问 [评测界面](http://192.168.50.113:8084/)。

```bash
source /home/lindenbot/miniconda3/etc/profile.d/conda.sh
conda activate pi_05
cd /home/lindenbot/qianzhang43/agibot_policy_ctl/g2_policy_ctl/intergrated_test
VLA_MODEL_VALIDATION_ONLY=1 bash start_server.sh
```

## 衣物任务实机预检

操作员先把机器人切到对应的预设姿态和导航点。实机流程从当前三路图像和 16 维关节状态开始推理。三个衣物任务每段都同时使用双臂和两个夹爪的模型输出。

`../config/clothes_robot_setup.json` 保存目标机器人地址、型号和关节限位。机器人端选中 `remote_env` 模式，评测按需启动该模式的控制服务。限位单位为 `rad`，每侧机械臂填 7 组 `[lower, upper]`、夹爪填 1 组，并标明 `robot_model` 和 `units`。预检接口只检查本机配置。

启用三个衣物任务的实机入口。

```bash
VLA_MODEL_VALIDATION_ONLY=1 VLA_CLOTHES_LIVE_TASK=left_arm_pick_blue_clothes_from_sofa,right_arm_take_off_clothes,right_arm_hang_clothes bash start_server.sh
```

模型输出通过 `30×16` 形状和有限实数检查后，原样拆分成双臂和双夹爪指令。运行测试采用「读取当前观测 → 推理 → 下发轨迹 → 等待轨迹参考执行时间 → 重新读取观测」的循环；每段后不比较目标关节误差，也不因未精确到位而等待或结束测试。时间等待使用可中断的事件等待，点击「中止」后停止后续下发，模型保持加载；已经提交的轨迹不会由此撤销。共用该流程的挂衣服、沙发、摘衣服、放洗衣机、关门 event 和历史帧 MEM 任务均采用此节奏。

「复位」单独保留双臂和双夹爪到位确认。运行报告通过 `execution_progression` 区分两种模式；运行测试的 `arrival_checked_parts` 为空，`last_target_arrival_confirmed` 为 `null`，表示未进行到位判定。

沙发卡片可分别选择左手和右手抓取。

- 「一键启动」按需启动机器人控制服务并加载沙发模型。加载完成后按钮变成「关闭模型」，只卸载 VLA。模型在测试结束或中止后继续运行，可用于下一轮。
- 「左手抓取」使用配置中的 `left_grasp_prompt`。「右手抓取」使用默认的 `prompt`。两者共用已加载的模型，从当前姿态执行双臂和双夹爪轨迹。测试运行时所选按钮变成「中止」。
- 「复位」在人工确认后，按 `sofa_reset_pose` 里的记录恢复双臂和双夹爪。关节限位和步长配置用于生成复位轨迹。四路共用一段轨迹和参考时间。导航点由现场人员设置。

摘衣服和挂衣服卡片的「运行测试」复用各自已加载的模型，运行中显示「中止」。两项的复位姿态分别记录到 `reset_poses` 下对应的任务 ID。记录后「复位」可用。

在评测目录运行离线回归。实机流程测试用机器人和模型替身。

```bash
/home/lindenbot/miniconda3/envs/pi_05/bin/python -B -m unittest test_a1 test_clothes_task test_clothes_robot -v
```
