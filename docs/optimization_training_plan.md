# OdorSearch_MobileArm 策略优化训练方案

## 1. 目标与现状

### 1.1 当前算法在 OdorSim/GADEN 中的验证结果
使用 `scripts/run_odorsim_verification.py` 在 8×8 m 空仓库（GADEN 实时气体扩散）中测试 5 个随机种子：

| seed | 结果 | 步数 | 说明 |
|------|------|------|------|
| 1    | ✅ 成功 | 41   | 源较近，快速找到 |
| 2    | ❌ 失败 | 2000 | 基座停在远端，机械臂陷入高浓度 filament 但无法靠近源 |
| 3    | ✅ 成功 | 70   | — |
| 4    | ✅ 成功 | 128  | — |
| 5    | ✅ 成功 | 226  | — |

成功率 **80%**（小空间、无障碍）。失败模式主要是：
1. **基座推进不足**：ARM_TRACKING 阶段 base 移动增益过小（0.03/0.005），机械臂够到高浓度区后基座不再前进。
2. **随机游走无风向偏置**：探索阶段未利用已知风场，远距离源发现慢。
3. **机械臂 preset 分层粗糙**：低/中/高三档 preset 无法连续适应源高度。
4. **丢失后无记忆**：LOST 后直接回到 RANDOM_WALK，未回到历史高浓度点。

### 1.2 优化训练目标
在保持零碰撞的前提下，把 GADEN 后端上的端到端成功率提升到 **>95%**，并缩短平均搜索步数。训练得到的策略应可直接替换当前的有限状态机（FSM），或作为 FSM 中各子模块的神经网络替代。

---

## 2. 训练范式选择

推荐 **分阶段训练**：

| 阶段 | 名称 | 内容 | 目的 |
|------|------|------|------|
| Phase A | 行为克隆（BC） | 用当前 FSM / 人工示教生成专家轨迹，训练策略初始策略 | 获得安全、可解释的起点 |
| Phase B | 强化学习微调 | 在 GADEN 环境中用 PPO/SAC 继续训练 | 克服 FSM 的局部最优 |
| Phase C | 课程学习 | 逐步增大房间尺寸、增加障碍物、改变风速 | 提升泛化能力 |

> 如果算力有限，可只做 Phase B（end-to-end RL），用零初始化或轻量 CNN/MLP 策略。

---

## 3. MDP 形式化定义

### 3.1 状态空间 `s_t`
建议维度 30~40，包含：

```text
# 本体感知（9 维）
base_pose: [x, y, z, yaw]                 (4)
joint_angles: 6-DOF                       (6)
base_vel / joint_vel: 可选                (up to 7)

# 气味观测（核心，10 维）
ee_ppm (current + history 4 步)           (5)
corner_ppm: front_left, front_right,       (4)
             rear_left, rear_right
wind_estimate: [wx, wy, wz]               (3)

# 任务/历史信息（6 维）
step_norm: t / max_steps                   (1)
visited_map_feature: 当前格 + 前/左/右是否访问 (4)
last_action: [dx, dy, dyaw, dJ1..dJ6]     (7)
```

设计要点：
- **ppm 历史窗**：MOX/GADEN 信号有噪声和时滞，4~8 步历史能稳定策略。
- **风场估计**：可用 GADEN `/wind_value` 服务查询当前位置风场，或从机器人运动历史推断。
- **归一化**：所有位置除以房间半长，ppm 做对数压缩 `log(1 + ppm)`。

### 3.2 动作空间 `a_t`
保持与原算法一致的车体-机械臂解耦控制，便于 sim-to-real：

```text
a_t = [dx, dy, dyaw, dJ1, dJ2, dJ3, dJ4, dJ5, dJ6]   (9 维)
```

- 基座：`dx, dy ∈ [-0.10, 0.10] m/步`，`dyaw ∈ [-0.15, 0.15] rad/步`
- 机械臂：`dJi ∈ [-5°, 5°]`/步
- 可在策略输出后加与原代码相同的安全裁剪/碰撞预测。

### 3.3 奖励函数 `r_t`

```python
r_t = r_close + r_detect + r_finish + r_time + r_collision + r_explore
```

| 分量 | 公式 | 说明 |
|------|------|------|
| `r_close` | `+ η1 * (d_{t-1} - d_t)` | 距离源变近则正奖励 |
| `r_detect` | `+ η2 * I[ppm_ee > θ]` | 首次或持续检测到气味 |
| `r_finish` | `+100`（成功），`-50`（碰撞/超时） | 稀疏事件奖励 |
| `r_time` | `-0.01` / step | 鼓励尽快完成 |
| `r_collision` | `-10`（预测碰撞被裁剪），`-50`（实际碰撞） | 安全 |
| `r_explore` | `+0.02 * (新访问网格)` | 防止原地打转 |

建议系数：
- `η1 = 5.0`（距离奖励按米缩放）
- `η2 = 0.5`
- 成功奖励随步数衰减：`100 * (1 - steps/max_steps)`

### 3.4 终止条件
- `done_success`: 末端距源 < 0.55 m 且 ee_ppm > 50
- `done_timeout`: 达到 `max_steps`
- `done_collision`: 实际发生碰撞（应通过安全层尽量避免）

---

## 4. 网络结构

### 4.1 纯 MLP（推荐先尝试）
```
输入 (obs_dim) -> [256, 256, 128] -> 输出 mean/std (PPO) 或 Q/V (SAC)
激活函数：ReLU / Swish
输出层：Tanh 缩放至动作空间
```

### 4.2 带历史窗的 MLP+LSTM
若 ppm 时序对策略很重要，可在第二层后接 64 维 LSTM：
```
obs_t -> FC256 -> FC256 -> LSTM64 -> FC128 -> action
```

### 4.3 视觉增强（可选）
若后续接入 OdorSim 的 RGB 观测（`agentview_image`），可并一路 CNN：
```
image -> Conv(32,3) -> Conv(64,3) -> Flatten -> FC256
                          concat with proprio/odor obs
```

---

## 5. 训练算法与超参

### 5.1 推荐：PPO（Proximal Policy Optimization）
原因：
- 连续动作空间表现稳定
- 与 on-policy 数据收集兼容，易于在 GADEN 锁步环境中实现
- 超参不敏感，适合作为 baseline

超参建议：
```yaml
algo: PPO
lr: 3e-4
gamma: 0.99
lambda: 0.95
clip_eps: 0.2
value_loss_coef: 0.5
entropy_coef: 0.01
batch_size: 256
n_epochs: 10
rollout_steps: 2048
max_grad_norm: 0.5
```

### 5.2 替代：SAC（Soft Actor-Critic）
若样本效率优先、可以接受 off-policy，SAC 在气味搜索这种稀疏奖励任务中可能更快，但需要更多调参（temperature auto-tuning）。

---

## 6. 课程学习（Curriculum）

为逐步提升难度，按以下顺序生成训练环境：

| 课程阶段 | 房间尺寸 | 障碍物 | 风速 | 源距离起点 | 目标成功率 |
|----------|----------|--------|------|------------|------------|
| L1 简单 | 4×4 m | 无 | 均匀 0.3 m/s | 1.5~2.5 m | >95% |
| L2 中等 | 6×6 m | 无 | 均匀 0.3 m/s | 2.5~4.0 m | >90% |
| L3 当前 | 8×8 m | 无 | 均匀 0.3~0.6 m/s | 3.0~5.5 m | >85% |
| L4 大房间 | 12×12 m | 无 | 随机风向 | 4.0~7.0 m | >80% |
| L5 障碍 | 8×8/12×12 | 少量 box/cylinder | 随机 | 3.0~6.0 m | >75% |

自动升级条件：最近 20 个 episode 成功率超过阈值，或平均回报连续 50 个 episode 不再提升。

---

## 7. 数据收集与训练流程

### 7.1 专家数据（Phase A）
运行当前 FSM 在 GADEN 中生成轨迹：
```bash
bash scripts/run_odorsim_benchmark.sh 1 2 ... 50
```
每个 episode 保存 `(s_t, a_t, r_t, s_{t+1}, done)`。BC loss：
```python
L_bc = E[(π_θ(s) - a_expert)^2]
```
训练 50~100 epoch 得到安全初始策略。

### 7.2 RL 训练循环（Phase B）
```python
for iteration:
    # 1. 收集 rollout
    for step in range(rollout_steps):
        a = policy.act(obs)
        obs_next, reward, done, info = env.step(a)
        buffer.store(obs, a, reward, value, log_prob, done)
        if done:
            obs = env.reset()

    # 2. 计算 GAE
    buffer.compute_returns_and_advantage()

    # 3. 更新策略
    for epoch in range(n_epochs):
        for batch in buffer.mini_batches():
            loss = ppo_loss(batch)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
```

### 7.3 并行化（可选）
GADEN 当前版本为单进程 lockstep，不易直接并行。可采用：
- **串行多 episode**：每次 rollout 一个 episode，累积后更新。
- **ROS_DOMAIN_ID 隔离**：若未来 OdorSim 支持，可起多个 GADEN server 并行。
- **离线合成数据**：用 OdorSim 的 `odor_sim.recording.synthesize` 批量生成不同源/风条件下的浓度数据，做 model-based RL。

---

## 8. 与现有代码的接口设计（最小侵入）

新增/修改文件规划：

```text
OdorSearch_MobileArm/
├── src/
│   ├── rl/
│   │   ├── __init__.py
│   │   ├── env_wrapper.py        # Gymnasium wrapper around GadenBackedWarehouseEnv
│   │   ├── policy.py             # MLP / LSTM actor-critic
│   │   ├── ppo.py                # PPO 训练循环
│   │   ├── reward_shaper.py      # 奖励函数
│   │   └── train_ppo.py          # 入口脚本
│   └── ... (已有文件保持不动)
├── scripts/
│   ├── run_odorsim_verification.py   # 已完成：仿真验证 + 视频
│   ├── run_odorsim_benchmark.sh      # 已完成：多种子 batch
│   └── train_rl.sh                   # 训练入口
└── docs/
    └── optimization_training_plan.md   # 本文档
```

`rl/env_wrapper.py` 要点：
- 封装 `OdorSearchSessionOdorSim` 为 `gymnasium.Env`
- 在 `reset()` 中随机化源位置、风场、起点
- 在 `step()` 中调用 `session.step()` 并返回 `(obs, reward, terminated, truncated, info)`
- 使用 GADEN-aware 的 agent 做训练对手

---

## 9. 评估指标

| 指标 | 定义 | 目标 |
|------|------|------|
| 成功率 | `FINISHED` / 总 episode | >95%（L1），>85%（L3） |
| 平均步数 | 成功 episode 的步数均值 | 相比 FSM 降低 30% |
| 碰撞率 | 发生实际碰撞的 episode 比例 | <1% |
| 平均最终距离 | 所有 episode 末时刻到源距离 | <0.6 m |
| GADEN 查询数 | 每 episode 调用 `/odor_value` 次数 | 尽量低，反映样本效率 |
| 风/源泛化 | 在未见源位置/风场上的成功率 | >80% |

---

## 10. 风险与缓解

| 风险 | 缓解措施 |
|------|----------|
| GADEN 浓度稀疏，奖励稀疏导致训练困难 | 加入密集奖励（距离、检测、探索）；使用 BC 初始化 |
| 策略在 filament 局部高浓度区震荡 | 观测中加入 ppm 历史窗；奖励中加入“基座靠近源”项 |
|  sim-to-real 差距 | 动作空间保持与真实控制器一致；训练时加入关节延迟、传感器噪声 |
|  训练时间过长 | 先用小房间/低 ppm 复杂度快速迭代；再课程放大 |
|  碰撞 | 保留原 `_ensure_action_safe` 安全层作为策略输出后处理 |

---

## 11. 下一步可执行动作

1. **实现 `src/rl/env_wrapper.py`**：把当前 OdorSim 验证环境包装成 Gymnasium 接口。
2. **实现 BC 预训练**：用 FSM 在 8×8 房间生成 100 条轨迹，训练初始策略。
3. **实现 PPO 训练循环**：在 L1（4×4）课程上训练至成功率 >95%。
4. **逐步提升课程难度**：L2 → L3 → L4 → L5。
5. **与 OdorSim 的 robosuite 渲染对接**：未来把移动小车+机械臂模型导入 OdorSim，获得真实 MuJoCo 视频。
