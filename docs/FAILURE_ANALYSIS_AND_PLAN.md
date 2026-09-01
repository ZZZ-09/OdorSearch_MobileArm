# 失败案例分析及下一步改进方案

> 基于 2026-09-01 第三批 30 新种子 OdorSim/GADEN 有障碍 8×8 仓库验证结果（成功率 28/30 = 93.3%）。

---

## 一、失败种子概览

| seed | 结果 | 步数 | 最终状态 | 最终 base→source xy 距离 | 最终 ee→source 距离 | 历史最小 base→source xy | 历史最小 ee→source |
|------|------|------|----------|--------------------------|---------------------|--------------------------|---------------------|
| 39958 | ❌ 失败 | 2000 | ARM_TRACKING | 1.190 m | 1.511 m | 0.715 m | 0.484 m |
| 68933 | ❌ 失败 | 2000 | ARM_TRACKING | 1.573 m | 1.891 m | 0.102 m | 0.151 m |

两个种子均满足以下条件：
- 机器人**曾经到达过源附近**（最小距离 < 0.5 m）。
- 最终因为**未触发结束条件**而耗尽 2000 步。
- 最终停留在 `ARM_TRACKING` 状态，且末端高度明显低于源高度。

---

## 二、关键现象：机械臂在 ARM_TRACKING 中逐渐折叠

### 2.1 末端高度变化

| seed | 源高度 (m) | 早期 ARM_TRACKING 末端高度 (step 350) | 后期末端高度 (step 500~2000) |
|------|------------|--------------------------------------|------------------------------|
| 39958 | 0.880 | 0.668 m | 0.15–0.20 m |
| 68933 | 0.611 | 0.686 m | 0.13–0.20 m |

### 2.2 关节角变化（以 seed 39958 为例）

| 时刻 | j1 | j2 | j3 | j4 | j5 | j6 |
|------|----|----|----|----|----|----|
| step 350（健康姿态） | -3.4° | -66.0° | 135.0° | -43.1° | 128.0° | 0° |
| step 500（已折叠） | -178.0° | **-130.0°** | -10.4° | -178.0° | -0.6° | 0° |

j2 达到下限 -130°，j4 达到下限 -178°，手臂被“卷”到车体下方，导致：
- 末端高度从 0.6 m+ 降到 0.2 m 以下。
- 结束条件中的 `ee_pos[2] >= 0.40 m` 永远无法满足。
- 即使浓度很高，也无法声明找到源。

### 2.3 状态机循环

两个种子都呈现相同的循环：

```text
ARM_TRACKING (251 steps) → LOST (1 step) → ARM_TRACKING (251 steps) → ...
```

原因：`_arm_tracking_max_steps = 250` 触发超时进入 `LOST`，但此时末端浓度仍高，立即在下一步回到 `ARM_TRACKING`。机器人没有真正逃离，只是在状态机里空转。

---

## 三、根因分析

### 3.1 末端梯度追踪无高度约束

`_arm_tracking_action()` 中：

```python
self._target_point = ee_pos + self.gradient_step * direction
```

`direction` 是 3D 梯度方向。如果局部烟羽的 z 向梯度指向下方（或数值噪声导致），`_target_point` 的 z 坐标会降低。`_point_ee_to_target()` 使用纯雅可比伪逆 IK，没有任何**高度保持**或**姿态正则化**，会跟随目标点向下折叠。

### 3.2 数值 IK 缺乏关节限位与姿态保护

`_point_ee_to_target()`：
- 只最小化末端位置误差。
- 没有二次任务使关节保持在中位角或预设姿态。
- 没有主动远离关节极限。
- 在长时间 ARM_TRACKING 中，j2/j4 容易被推到极限，形成折叠姿态。

### 3.3 结束条件过严且未与姿态健康度联动

当前结束条件：
1. `ee_ppm >= 1000 ppm` 持续 50 步；
2. 末端位于局部浓度峰值；
3. 基座位移 < 0.15 m；
4. `ee_pos[2] >= 0.40 m`。

条件 4 是合理的（防止地面 filament 误结束），但当机械臂已经折叠到 0.2 m 时，条件 4 永远无法满足。算法没有**在条件不满足时主动抬升手臂**的机制。

### 3.4 ARM_TRACKING 超时策略过于机械

`_arm_tracking_max_steps = 250` 在浓度仍然很高时强制切到 `LOST`，但下一步又因高浓度切回 `ARM_TRACKING`。这种切换：
- 没有帮助机器人摆脱局部区域；
- 浪费了步数；
- 没有区分“高浓度但姿态错误”和“真正丢失”。

---

## 四、下一步改进方案

### 方案 A：ARM_TRACKING 中强制保持最小末端高度（推荐，改动小）

在 `_arm_tracking_action()` 中，根据当前高度层预设或源高度先验，限制 `_target_point[2]` 不低于最小工作高度：

```python
_min_arm_tracking_height = 0.45  # m
self._target_point[2] = max(self._target_point[2], _min_arm_tracking_height)
```

理由：
- 8×8 有障碍仓库中源高度在 0.6–0.9 m 之间。
- 保持末端 z ≥ 0.45 m 可避免折叠到地面 filament 区，同时保留一定搜索范围。
- 实现简单，不破坏现有梯度逻辑。

### 方案 B：数值 IK 增加姿态正则化与关节限位惩罚

在 `_point_ee_to_target()` 中：

1. **二次任务**：让关节尽量靠近当前高度层预设姿态。
   ```python
   preset = self.robot.arm_cfg["preset_poses"][self._height_presets[self._current_height_level]]
   posture_error = current_joints_deg - preset
   delta_theta_posture = -k_posture * posture_error
   ```
2. **关节限位惩罚**：对接近限位的关节减小该方向的增量。
   ```python
   for i, limit in enumerate(joint_limits):
       margin = 10.0  # 度
       if current_deg[i] < limit[0] + margin:
           delta_deg[i] = max(0, delta_deg[i])  # 只允许离开下限
       elif current_deg[i] > limit[1] - margin:
           delta_deg[i] = min(0, delta_deg[i])  # 只允许离开上限
   ```
3. **零空间投影**：将姿态任务投影到末端任务零空间，优先保证位置追踪。

### 方案 C：高浓度时主动抬升/切换高度层

当 `ee_ppm` 持续较高但 `ee_pos[2] < 0.45 m` 时：
- 临时提高 `_current_height_level` 到 `mid` 或 `high`；
- 或在 `_target_point` 上叠加向上的偏置；
- 直到高度满足结束条件。

可与方案 A 结合使用。

### 方案 D：改进 ARM_TRACKING 超时逻辑

当前超时切 `LOST` 的方式应改为“智能重试”：

```python
if self._arm_tracking_start_step > self._arm_tracking_max_steps:
    if ee_ppm >= 0.8 * self._odor_finish_min_ppm and not self._height_ok_recently():
        # 高浓度但高度不够：先尝试抬升，不逃离
        self._target_point[2] += 0.10
    else:
        # 真正可能局部最优：逃离
        self.state = SearchState.LOST
        ...
```

避免无意义的状态抖动。

### 方案 E：放宽结束条件的高度检查或引入“接近源”辅助判据

由于独立评估阈值是 1.0 m，结束条件可以稍微放宽：
- `height_ok = ee_pos[2] >= 0.35`（从 0.40 降到 0.35）；
- 或增加“基座与末端水平距离很小且浓度高”的辅助判据。

但治标不治本，建议优先实施方案 A/B/C。

---

## 五、推荐实施顺序

| 优先级 | 方案 | 预期效果 | 改动量 |
|--------|------|----------|--------|
| P0 | A：最小末端高度约束 | 阻止折叠，2 个失败种子大概率转为成功 | 小（~5 行） |
| P1 | B：IK 姿态正则化 | 长期 ARM_TRACKING 更稳定，减少关节到限 | 中（~30 行） |
| P2 | D：智能超时 | 减少状态抖动，节省步数 | 小（~10 行） |
| P3 | C：主动抬升 | 进一步提高不同源高度的适应能力 | 中 |
| P4 | E：放宽结束条件 | 作为兜底，但可能引入误结束 | 小 |

---

## 六、验证计划

1. 实施 P0（方案 A）后，重新跑 2 个失败种子：
   ```bash
   bash scripts/generate_odorsim_videos_triple_v3.sh 39958 68933
   ```
2. 若通过，跑 30 新种子全量回归：
   ```bash
   bash scripts/run_odorsim_obstacle_30seeds_v3.sh
   ```
3. 观察：
   - 失败种子是否转为成功；
   - 原成功种子是否保持成功；
   - 是否出现新的碰撞或误结束。

---

## 七、结论

第三批验证的 2 个失败案例并非“找不到源”，而是**找到了源附近但未能在正确姿态下触发结束条件**。核心根因是 `ARM_TRACKING` 中的 3D 梯度追踪缺乏高度约束，导致机械臂折叠到地面附近。通过引入最小末端高度约束、IK 姿态正则化和更智能的超时策略，有望将成功率从 93.3% 进一步提升，同时保持 30/30 无碰撞。
