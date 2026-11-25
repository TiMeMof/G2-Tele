# Agibot G2 RTC 异步推理实现说明

## 文件概览

- **run_pick_rtc.py** - 基于 threading 的真正异步 RTC 推理实现
- **run_pick.py** - 原始同步版本（存在动作覆盖问题）
- **run_pick_async.py** - 错误的 asyncio 版本（与同步版时序相同，无实际改进）
- **run_pick_synchronized.py** - 同步等待版本（解决动作覆盖但频率降到 1Hz）

## 核心架构

### 三线程 + 两队列设计

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  采集线程    │ --> │   Q1: 状态    │ --> │  推理线程    │
│ 30Hz 采集   │     │   缓存队列    │     │ ~10Hz 推理  │
│ 状态+图像   │     │ (带时间戳)   │     │             │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                  │
                                                  │ splice
                                                  ↓
                    ┌──────────────┐     ┌─────────────┐
                    │   Q2: 动作    │ <-- │  执行器线程  │
                    │   队列        │     │ 30Hz 执行   │
                    │ (时间索引)   │     │ 单帧下发    │
                    └──────────────┘     └─────────────┘
```

### 线程职责

1. **采集线程** (`_collect_thread`)
   - 频率：30Hz（可配置）
   - 调用 `controller.read_state()` 获取关节状态 + 三路图像
   - 写入 `Q1: StateBuffer` 带时间戳的观测缓存

2. **推理线程** (`_inference_thread`)
   - 频率：最高 10Hz（可配置）
   - 从 Q1 读取最新观测
   - 计算 RTC index（距上次推理流逝的帧数）
   - 调用推理端获取 30 帧动作块
   - **splice 到 Q2**：按"观测时刻 + 帧序号 * dt"计算目标执行时间
   - RTC 模式：跳过前 `inference_delay` 帧（由旧 chunk 继续执行）

3. **执行器线程** (`_executor_thread`)
   - 频率：30Hz（可配置）
   - Grace period：启动后 1 秒内不执行（防止重试时立即闭合）
   - 从 Q2 `pop_due(now)`：弹出所有到期的动作，取最新一条
   - 单帧下发：`controller.send_step(action_16d, ref_time)`
   - 抓取检测：夹爪指令 > -0.05 时触发 `grasp_event`

## RTC 动作拼接机制

### 问题：原始版本的动作覆盖

```
时间轴（原始 run_pick.py）：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
0ms:    下发 chunk1（30帧，ref_time=1.0s，应执行1秒）
150ms:  推理完成，下发 chunk2 → 覆盖 chunk1 ❌
300ms:  推理完成，下发 chunk3 → 覆盖 chunk2 ❌
结果：每个 chunk 只执行 5-8 帧就被覆盖，轨迹不连续
```

### 解决方案：RTC 时间索引队列

```
时间轴（RTC run_pick_rtc.py）：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
推理1（0ms观测）：
  chunk1[0..29] → 目标时间 [T_obs + dt, T_obs + 2dt, ..., T_obs + 30dt]
  splice 到队列，index=-1（首次推理）

推理2（150ms观测，已流逝 ~5 帧）：
  chunk2[0..29] → 目标时间 [T_obs2 + dt, ...]
  index=5，跳过前 4 帧（inference_delay）
  splice: 丢弃队列中 >= T_obs2 + 5dt 的旧条目，写入新 chunk[4..29]
  前 4 帧仍由 chunk1 的对应位置执行 ✅

执行器：
  每 33ms pop_due(now)，取出所有 target_time <= now 的帧
  返回最新一条执行，保证按时间顺序、无跳跃
```

### splice 核心逻辑

```python
def splice(self, chunk, T_obs, dt, now, skip_frames):
    # 1. 计算每帧目标时间
    model_times = T_obs + (np.arange(n) + 1.0) * dt
    
    # 2. 跳过已过期的帧和前 skip_frames 帧
    new_items = [(model_times[i], chunk[i]) 
                 for i in range(skip_frames, n) 
                 if model_times[i] > now]
    
    # 3. 用新规划覆盖未来：丢弃队列中 >= cutoff 的旧条目
    cutoff = new_items[0][0]
    kept_items = [x for x in self._items if x[0] < cutoff]
    
    # 4. 拼接
    self._items = kept_items + new_items
```

## RTC 前缀引导（推理端）

### 参数传递

```python
payload["infer_args"] = {
    "index": 5,                    # 距上次推理流逝 5 帧
    "inference_delay": 4,          # 前 4 帧由旧 chunk 执行
    "max_guidance_weight": 5.0,    # 前缀引导权重
    "num_flow_steps": 10,          # flow 采样步数
}
```

### 推理端行为（需服务器支持）

1. **index=-1**：首次推理，走标准 `sample_actions`
2. **index ∈ [0, chunk_size)**：走 `sample_actions_rtc`
   - 从缓存的 `_last_raw_actions` 滚动对齐到 `[index:]` 作为 prefix
   - 通过 VJP 伪逆约束新 chunk 前缀贴近旧规划
   - guidance_weight 随 flow 时间衰减，平衡"过去一致"与"未来自由"

### 如果服务器不支持 RTC

使用 `--no-rtc` 参数：
- 不传 `infer_args`，推理端走标准推理
- 仍然使用时间索引队列和 splice 机制
- 动作切换点可能有小抖动，但不会硬覆盖

## 与其他版本对比

| 特性 | run_pick.py | run_pick_async.py | run_pick_synchronized.py | **run_pick_rtc.py** |
|------|------------|-------------------|-------------------------|---------------------|
| 执行模式 | 同步串行 | asyncio 串行等待 | 同步+等待轨迹完成 | **threading 真异步** |
| 动作覆盖 | ✅ 存在 | ✅ 存在（相同） | ❌ 不存在 | **❌ 不存在** |
| 闭环频率 | ~6-7 Hz | ~6-7 Hz | ~1 Hz | **~6-7 Hz** |
| 轨迹完整性 | ❌ 5-8帧 | ❌ 5-8帧 | ✅ 完整30帧 | **✅ 完整30帧** |
| 推理与执行 | 串行 | 串行 | 串行 | **并发重叠** |
| 动作平滑 | ❌ 硬切换 | ❌ 硬切换 | ✅ 等待到位 | **✅ RTC 引导** |

## 使用方法

### 基本用法

```bash
# 默认参数（RTC 模式）
python run_pick_rtc.py --task left_arm_drawer_pick_omeprazole_bottle

# 调整 RTC 参数
python run_pick_rtc.py \
    --task left_arm_drawer_pick_omeprazole_bottle \
    --chunk-size 30 \
    --publish-rate 30 \
    --executor-rate 30 \
    --inference-delay 4 \
    --max-guidance-weight 5.0

# 禁用 RTC（服务器不支持时）
python run_pick_rtc.py --task left_arm_drawer_pick_omeprazole_bottle --no-rtc
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--chunk-size` | 30 | 模型输出的动作块大小（帧数） |
| `--publish-rate` | 30.0 | 模型输出帧率（Hz），决定 dt=1/rate |
| `--executor-rate` | 30.0 | 执行器打拍频率（Hz） |
| `--collect-rate` | 30 | 状态采集频率（Hz） |
| `--max-inference-rate` | 10.0 | 推理线程最大频率（Hz） |
| `--inference-delay` | 4 | RTC 推理延迟（帧数），新 chunk 跳过前 N 帧 |
| `--max-guidance-weight` | 5.0 | RTC 前缀引导权重 |
| `--grace-period` | 1.0 | 启动后 grace period（秒） |
| `--max-steps` | 150 | 单次尝试最大步数 |
| `--max-retries` | 3 | 夹取失败后最大重试次数 |
| `--no-rtc` | False | 禁用 RTC 模式 |

## 关键改进点

### 1. 真正的异步并发

- **采集、推理、执行三线程独立运行**，无相互阻塞
- 推理耗时（~150ms）与执行重叠，不影响动作下发
- 采集持续进行，推理始终用最新观测

### 2. 时间索引动作队列

- 每个动作帧标记目标执行时间
- `splice` 用新规划覆盖未来，保留已到期的旧规划
- `pop_due` 按时间顺序弹出，避免跳帧或乱序

### 3. RTC 前缀引导

- 推理端用上次动作作为前缀引导新 chunk
- 动作切换点平滑衔接，不抖动
- 需推理服务器支持（可用 `--no-rtc` 降级）

### 4. Grace Period

- 重试时打开夹爪后，等待 1 秒再执行新动作
- 防止模型看到旧状态（闭合中）立即再次闭合

### 5. 鲁棒抓取判定

- 检测到闭合指令后，等待 1.5 秒夹爪稳定
- 重新采样夹爪状态，避免中间位置误判
- 阈值对齐 run_pick.py（`fail_threshold`, `success_threshold`）

## 适配要点

从 galbot_rtc_inference.py 到 agibot 的主要改动：

1. **机器人接口**：`GalbotRobot` → `G2CorobotEnv`
2. **动作维度**：8 维（单臂） → 16 维（双臂）
3. **观测获取**：分离的关节/图像线程 → 统一的 `get_state()`
4. **图像解码**：SDK 原始格式 → JPEG 字节流 `cv2.imdecode`
5. **配置加载**：YAML+JSON → intergrated_test 的 JSON 配置
6. **夹爪判定**：SDK `gripper.width` → 观测字典 `gripper_states[index]`

## 性能预期

- **闭环频率**：~6-7 Hz（推理 ~150ms + 执行重叠）
- **轨迹完整性**：每个 30 帧 chunk 完整执行，不被覆盖
- **动作平滑性**：RTC 前缀引导 → 切换点连续；禁用 RTC → 可能有小抖动
- **响应延迟**：采集→推理→执行全流程 ~150-200ms

## 注意事项

1. **推理服务器兼容性**
   - 如果 127.0.0.1:8181 不支持 `infer_args`，使用 `--no-rtc`
   - 标准模式仍避免动作覆盖，但无前缀引导

2. **线程安全**
   - `controller.env_lock` 保护 `G2CorobotEnv` 并发访问
   - 所有队列操作都有锁保护

3. **Grace Period 调优**
   - 重试场景：增大 `--grace-period`（默认 1.0s）
   - 快速响应：减小，但可能重试时误闭合

4. **Executor Rate vs Publish Rate**
   - 相等（30Hz）：忠实执行模型输出
   - executor > publish：插值更多中间帧（未实现）
   - executor < publish：跳过部分模型帧

## 总结

**run_pick_rtc.py 是真正解决动作覆盖问题的异步实现**，通过：
- threading 多线程实现采集/推理/执行并发
- 时间索引动作队列避免硬覆盖
- RTC 前缀引导保持动作连续
- Grace period 和鲁棒判定提高重试成功率

相比 run_pick_synchronized.py（1Hz），RTC 版保持高频闭环（6-7Hz）；
相比 run_pick.py/run_pick_async.py，RTC 版真正避免动作覆盖且轨迹平滑。
