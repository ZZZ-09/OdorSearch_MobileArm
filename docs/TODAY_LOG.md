# 2026-09-01 第三批：算法改进 + 新 30 种子 OdorSim 验证 + 三视角视频

## 一、今日目标

1. 针对 2026-09-01 第二批有障碍 OdorSim 30 种子测试中的两类失败模式（障碍物绕过困难、高浓度丢失后复捕失败）改进搜索算法。
2. 使用改进后的算法在 8×8 m 有障碍仓库中运行 **30 个全新的随机种子**，进行正确率评估与失败原因分析。
3. 从新 30 个种子中随机挑选 5 个生成三视角视频，保存到新的独立文件夹。

## 二、核心代码改动（`src/search_algorithm.py`）

### 1. 障碍物感知覆盖路径规划
- `_build_coverage_tracks()` 现在利用已知的障碍物地图，按**最近邻顺序**连接各条牛耕法扫描线，扫描线之间用 A* 规划绕障转移路径。
- 这样机器人从当前位置出发优先探索附近区域，避免固定 y 顺序导致先远赴仓库另一端，同时保证能绕过障碍物到达被遮挡区域。

### 2. 降低过度保守的惩罚区域
- `_no_track_zone_radius`：5.0 m → **2.0 m**（在 8×8 仓库中不再覆盖整个源区）。
- `_escape_cooldown_steps`：250 → **120**（更快允许复捕）。

### 3. 障碍物感知的逃离路径
- `_set_escape_target()` 与 `_escape_action()` 改为使用 A* 规划到逃离目标的路径，而不是直奔目标。
- 逃离过程中若重新闻到 `ee_ppm ≥ 50` ppm 的连续高浓度，立即中断逃离并回到 `ARM_TRACKING`。

### 4. ARM_TRACKING 障碍物规避
- `_arm_tracking_action()` 在应用基座命令前用 `_predict_collision` 检查下一步是否会碰撞；若会碰撞则降低速度或做侧向偏移。

### 5. 随机游走覆盖增强
- `_steps_before_level_switch`：400 → **250**，加快机械臂高度层切换。
- 随机游走时增加机械臂末端小幅 z 方向扫描（±5 cm），提高与不同高度烟羽的匹配概率。
- 卡住恢复结束后自动重建覆盖路径。

### 6. 嗅觉结束条件微调
- `_odor_finish_required`：40 → **50**。
- `_odor_finish_displacement_threshold`：0.20 m → **0.15 m**。
- 新增 `ee_pos[2] >= 0.40 m` 的高度检查，避免在地面 filament 处误结束。

### 7. 新增/修改脚本
- `scripts/run_odorsim_obstacle_30seeds_v3.sh`：30 个新种子的批量验证（无视频）。
- `scripts/generate_odorsim_videos_triple_v3.sh`：为指定种子生成三视角视频，输出到 `outputs/odorsim_obstacle_new30_v3_videos/`。

## 三、新 30 种子有障碍 OdorSim 运行结果

种子由 `numpy.random.default_rng(2028)` 生成，范围 [0, 100000)，且与之前批次不重复：

```
56541 38675 35694 59255 23488 73733 33511 98492 96264 84224
30991 39958 41850 82929 64497 47843 95012 39036 22771 7343
34564 1415 68933 27165 7960 89797 81069 59446 71823 64124
```

运行命令（WSL Ubuntu-24.04，已 source OdorSim/setup/activate.sh）：

```bash
cd /mnt/c/Users/lenovo/Desktop/OSL_mechanic_arm/OdorSearch_MobileArm
bash scripts/run_odorsim_obstacle_30seeds_v3.sh
```

| 指标 | 数值 |
|------|------|
| 成功率 | 28/30 = **93.3%** |
| 碰撞-free | 30/30 |
| 成功平均步数 | 750.4 |
| 成功平均误差 | 0.568 m |
| 成功最大误差 | 0.692 m |
| 全体平均误差 | 0.644 m |
| 全体平均步数 | 833.7 |

### 失败案例分析

仅 2 个种子失败：

| seed | 结果 | 步数 | 误差 (m) | 最终状态 | 关键现象 |
|------|------|------|----------|----------|----------|
| 39958 | ❌ 失败 | 2000 | 1.511 | ARM_TRACKING | 局部高浓度但未满足结束条件，最终距源较远 |
| 68933 | ❌ 失败 | 2000 | 1.891 | ARM_TRACKING | 局部高浓度但未满足结束条件，基座漂移远离源 |

失败模式均为：机器人进入 `ARM_TRACKING` 后检测到高浓度，但由于 GADEN 湍流烟羽的局部 filament 特性，基座未能进一步收敛到源附近，最终超时。

## 四、三视角视频生成

从新 30 个种子中随机抽取 5 个（picker seed = 2028）：**30991、39036、64497、84224、98492**，全部为成功案例。

运行命令：

```bash
bash scripts/generate_odorsim_videos_triple_v3.sh 30991 39036 64497 84224 98492
```

| seed | 结果 | 帧数 | 时长 (@20fps) | 说明 |
|------|------|------|---------------|------|
| 30991 | ✅ 成功 | 127 | 6.35 s | RANDOM_WALK → BASE_TRACKING → ARM_TRACKING → FINISHED |
| 39036 | ✅ 成功 | 153 | 7.65 s | 较快结束 |
| 64497 | ✅ 成功 | 255 | 12.75 s | 含 BASE_TRACKING 阶段 |
| 84224 | ✅ 成功 | 99 | 4.95 s | 较快结束 |
| 98492 | ✅ 成功 | 32 | 1.60 s | 快速结束 |

视频保存路径：`outputs/odorsim_obstacle_new30_v3_videos/video_seed<SEED>.mp4`。

## 五、关键发现与下一步

1. **成功率显著提升**：从第二批的 23/30（76.7%）提升到本批次的 28/30（93.3%），验证了障碍物感知覆盖路径与复捕策略改进的有效性。
2. **碰撞-free 保持**：30/30 无碰撞，障碍物地图与 A* 规划有效。
3. **失败模式集中**：剩余失败均属于“ARM_TRACKING 中高浓度但未能收敛到源”，可进一步优化高浓度下的基座推进策略或结束条件。
4. **视频验证**：5 个随机种子均成功生成三视角视频，新文件夹与已有 `odorsim_obstacle_new30_videos/` 不重叠。

---

# 2026-09-01 第二批：三视角视频 + 30 种子有障碍 OdorSim 验证

## 一、今日目标

1. 将 OdorSim/GADEN 仿真视频从单一三维视图升级为**三视角同步画面**（3D + 俯视 XY + 前视 XZ）。
2. 在 8×8 m 有障碍仓库中使用 OdorSim/GADEN 后端运行 **30 个新的随机种子**，统计成功率并做失败案例分析（不同种子障碍物布局不同）。
3. 从 30 个新种子中随机挑选 5 个生成三视角视频，种子编号与项目已有文件夹均不重复。

## 二、代码与脚本改动

### 1. `scripts/run_odorsim_verification.py`

- 重构 `VideoRecorder`：
  - 新增 `--video-layout {3d|triple}` 命令行参数，默认 `triple`。
  - `triple` 模式下，一个视频画面同时显示：
    - **左侧 3D 视图**：机器人、机械臂、障碍物、气味源、末端轨迹。
    - **中间俯视图（XY）**：障碍物 footprint、气味源、底盘平动路径、末端路径、当前位置。
    - **右侧前视图（XZ）**：障碍物侧影、气味源、机械臂投影、末端高度变化轨迹。
  - `3d` 模式下保持原有单视图，便于回退。

### 2. 新增脚本

- `scripts/run_odorsim_obstacle_30seeds.sh`：30 种子有障碍 OdorSim 批量验证（无视频，节省时间），自动调用评估与分析脚本。
- `scripts/generate_odorsim_videos_triple.sh`：为指定种子批量生成三视角视频。
- `scripts/inspect_seed.py`：快速查看 `history_seed<N>.npz` 末尾若干步，用于失败案例根因分析。

### 3. `scripts/analyze_odorsim_runs.py`

- 泛化报告模板，支持任意数量种子。
- 改进失败模式分类，新增“机器人声明结束但独立评估误差超过阈值（误结束）”类别。
- 产出文件列表中仅在视频真实存在时才列出 `video_seed<N>.mp4`。

### 4. `scripts/pick_random_seeds_for_video.py`

- 支持读取 `evaluate_odorsim_runs.py` 生成的 `aggregate_evaluation.json`（`runs` 字段），从而可以直接从 OdorSim 聚合结果中挑选视频种子。

## 三、30 种子有障碍 OdorSim 运行结果

```bash
# WSL Ubuntu-24.04，已 source OdorSim/setup/activate.sh
cd /mnt/c/Users/lenovo/Desktop/OSL_mechanic_arm/OdorSearch_MobileArm
bash scripts/run_odorsim_obstacle_30seeds.sh
```

| 指标 | 数值 |
|------|------|
| 成功率 | 23/30 = **76.7%** |
| 碰撞-free | 30/30 |
| 成功平均步数 | 174.4 |
| 成功平均误差 | 0.645 m |
| 成功最大误差 | 0.961 m |
| 全体平均误差 | 1.396 m |
| 全体平均步数 | 543.8 |

### 各种子结果（部分）

| seed | 结果 | 步数 | 误差 (m) | 最终状态 | 关键现象 |
|------|------|------|----------|----------|----------|
| 63219 | ✅ 成功 | 291 | 0.636 | FINISHED | 随机游走后追踪并结束 |
| 800 | ❌ 失败 | 302 | 1.207 | FINISHED | 机器人声明结束，但独立评估误差 > 1 m（误结束） |
| 14849 | ✅ 成功 | 102 | 0.672 | FINISHED | 快速结束 |
| 8252 | ❌ 失败 | 2000 | 4.345 | RANDOM_WALK | 高浓度后丢失，随机游走未复捕 |
| 91814 | ❌ 失败 | 2000 | 4.138 | RANDOM_WALK | 全程几乎无气味检测 |
| 44874 | ❌ 失败 | 2000 | 3.629 | RANDOM_WALK | 全程几乎无气味检测 |
| 62126 | ❌ 失败 | 2000 | 4.994 | RANDOM_WALK | 高浓度后丢失 |
| 8984 | ❌ 失败 | 2000 | 4.676 | RANDOM_WALK | 高浓度后丢失 |
| 72330 | ❌ 失败 | 2000 | 4.076 | RANDOM_WALK | 高浓度后丢失 |

完整结果见 `outputs/odorsim_obstacle_new30/aggregate_evaluation.json` 与 `outputs/odorsim_obstacle_new30/analysis.md`。

### 失败案例分析

失败种子共 7 个，可分为三类：

1. **误结束（1 个：seed 800）**
   - 源位于 (0.36, 2.91, 0.80) m。
   - step 302 进入 FINISHED，但独立评估误差 1.207 m > 1.0 m 阈值。
   - 原因：嗅觉主导结束条件在局部高浓度平台被触发，但末端实际尚未到达源附近。

2. **追踪后丢失 / 超时（4 个：8252、62126、8984、72330）**
   - 均在运行早期检测到较高浓度并进入 ARM_TRACKING（约 250 步），随后触发 LOST/逃离。
   - 之后随机游走未能复捕烟羽，最终 2000 步超时失败。
   - 这是与 2026-08-31/09-01 早期批次一致的典型失败模式：GADEN 湍流烟羽局部高浓度后丢失，复捕策略不足。

3. **全程无气味检测（2 个：91814、44874）**
   - 全程 RANDOM_WALK，max EE ppm 极低（< 0.15 ppm）。
   - 源位置距起点较远或烟羽被障碍物/风场切割，机器人未经过有效烟羽区域。

## 四、三视角视频生成

从 30 个新种子中随机抽取 5 个（picker seed = 2027）：

```bash
python scripts/pick_random_seeds_for_video.py \
    --results outputs/odorsim_obstacle_new30/aggregate_evaluation.json \
    --num-pick 5 --output outputs/odorsim_obstacle_new30/seeds_for_video.txt --seed 2027
```

选中种子：**28488、50541、63219、70542、91814**（含 1 个失败案例，便于对比）。

生成命令：

```bash
bash scripts/generate_odorsim_videos_triple.sh 28488 50541 63219 70542 91814
```

| seed | 结果 | 帧数 | 时长 (@20fps) | 说明 |
|------|------|------|---------------|------|
| 28488 | ✅ 成功 | 145 | 7.25 s | 含 RANDOM_WALK → ARM_TRACKING → FINISHED 完整过程 |
| 50541 | ✅ 成功 | 37 | 1.85 s | 快速结束 |
| 63219 | ✅ 成功 | 146 | 7.30 s | 含较长 ARM_TRACKING |
| 70542 | ✅ 成功 | 83 | 4.15 s | 中等步数结束 |
| 91814 | ❌ 失败 | 1001 | 50.05 s | 完整记录 2000 步无检测随机游走 |

视频保存路径：`outputs/odorsim_obstacle_new30_videos/video_seed<SEED>.mp4`。

## 五、关键发现与下一步

1. **三视角视频有效**：单画面中同时呈现 3D 全局、俯视避障路径、前视机械臂升降，便于分析机器人行为。
2. **有障碍 OdorSim 成功率 76.7%**：高于 2026-08-28 空仓库 5 种子的 20%，与 2026-08-31 有障碍 30 种子的 73.3% 接近，说明算法在当前配置下相对稳定。
3. **失败模式仍以“高浓度后丢失”为主**：4/7 失败种子均因 ARM_TRACKING 中检测到高浓度后触发 LOST，后续随机游走未能复捕烟羽。
4. **误结束案例（seed 800）**：需进一步收紧嗅觉主导结束条件，或要求视觉辅助确认，避免局部高浓度平台触发 FINISHED。
5. **无检测案例（91814、44874）**：可能与源位置、风场方向、障碍物遮挡有关，可考虑增强覆盖策略或在长时间无检测时主动换高度层/换方向搜索。

---

# 2026-09-01 修改与运行日志

## 一、今日目标

使用 OdorSim/GADEN 后端，在 8×8 m 有障碍仓库中运行 **5 个随机种子**的仿真，生成运行视频，并对成功率、成功案例与失败案例进行详细分析。

## 二、运行方式

### 1. 5 种子 OdorSim 有障碍仿真（含视频）

```bash
# WSL Ubuntu-24.04，已 source OdorSim/setup/activate.sh
cd /mnt/c/Users/lenovo/Desktop/OSL_mechanic_arm/OdorSearch_MobileArm
bash scripts/run_odorsim_obstacle_5seeds.sh
```

脚本默认种子：`42 123 456 789 101112`；环境配置 `config/warehouse_obstacle_8x8.yaml`；最大步数 2000；视频帧率 **20 fps**，每 **2 步**渲染一帧；最后自动调用 `evaluate_odorsim_runs.py` 生成聚合评估 JSON。

### 2. 详细成功/失败案例分析

```bash
python scripts/analyze_odorsim_runs.py \
    --out-dir outputs/odorsim_obstacle_5seeds \
    --seeds 42 123 456 789 101112
```

生成 `outputs/odorsim_obstacle_5seeds/analysis.md`。

## 三、实验结果

| 指标 | 数值 |
|------|------|
| 成功率 | 4/5 = 80.0% |
| 碰撞-free | 5/5 |
| 成功平均步数 | 416.0 |
| 成功平均误差 | 0.501 m |
| 成功最大误差 | 0.666 m |
| 全体平均步数 | 732.8 |
| 全体平均误差 | 1.383 m |

### 各种子结果

| seed | 结果 | 步数 | 误差 (m) | 最终状态 | 关键现象 |
|------|------|------|----------|----------|----------|
| 42 | ✅ 成功 | 68 | 0.495 | FINISHED | 快速通过嗅觉主导结束 |
| 123 | ✅ 成功 | 303 | 0.666 | FINISHED | 高浓度 22000+ ppm 后结束 |
| 456 | ❌ 失败 | 2000 | 4.911 | BASE_TRACKING | 早期 ARM_TRACKING 检测到 7000+ ppm 后丢失，后续未复捕 |
| 789 | ✅ 成功 | 1198 | 0.238 | FINISHED | 长时间随机游走后成功定位 |
| 101112 | ✅ 成功 | 95 | 0.604 | FINISHED | 高浓度 22000+ ppm 后快速结束 |

### 成功案例分析

- **seed 42**：源位于 (-1.36, 0.25, 0.65)，机器人在 68 步内通过 RANDOM_WALK → BASE_TRACKING → VISUAL_APPROACH → ARM_TRACKING → FINISHED 完成定位。
- **seed 123**：源位于 (2.10, 2.45, 0.71)，236 步随机游走后进入追踪，303 步结束，最大 EE ppm 超过 22000。
- **seed 789**：源位于 (1.87, -1.60, 0.74)，耗时 1198 步，最终误差仅 0.238 m，精度最高。
- **seed 101112**：源位于 (0.08, -1.13, 0.86)，95 步快速结束，高浓度触发嗅觉主导结束条件。

### 失败案例分析

- **seed 456**：源位于 (0.01, 2.91, 0.59)。step 400 左右 ARM_TRACKING 检测到 6629 ppm，但随后丢失并回到 RANDOM_WALK；后续多次触发 BASE_TRACKING 但浓度始终较低，最终 2000 步超时失败。失败模式与 2026-08-31 30 种子 OdorSim 测试一致：高浓度后丢失，随机游走未能复捕烟羽。

## 四、新增/补充文件

- `scripts/run_odorsim_obstacle_5seeds.sh`：此前已存在但未在日志中记录，现补充说明；用于 5 种子有障碍 OdorSim 批量验证+视频生成。
- `scripts/analyze_odorsim_runs.py`：新增 OdorSim 运行详细 Markdown 分析报告生成脚本。
- `outputs/odorsim_obstacle_5seeds/aggregate_evaluation.json`：聚合评估结果。
- `outputs/odorsim_obstacle_5seeds/analysis.md`：成功/失败案例分析报告。
- `outputs/odorsim_obstacle_5seeds/video_seed<SEED>.mp4`（5 个）。
- `outputs/odorsim_obstacle_5seeds/summary_seed<SEED>.png`（5 个）。
- `outputs/odorsim_obstacle_5seeds/history_seed<SEED>.npz`（5 个）。

## 五、视频渲染问题修复

在首次生成视频后发现：
1. **机器人“穿越”障碍物**：旧版 `VideoRecorder` 仅用车体包围盒的线框（wireframe）绘制机器人，而障碍物使用半透明表面；matplotlib 3D 中线条无法被表面正确遮挡，导致视觉上机器人穿过障碍物。
2. **视频卡顿/缺尾帧**：旧版 `--video-stride=5 --fps=10`，每 5 步才渲染一帧，机器人跳跃明显；且若结束步数不落在 stride 整数倍上，会丢失最终 FINISHED 帧。

### 修复内容

- `scripts/run_odorsim_verification.py`：
  - `VideoRecorder.capture()` 中为机器人底盘同时绘制**线框 + 半透明表面**，使其能被障碍物表面正确遮挡；
  - 仿真循环结束后**强制补录一帧最终状态**，确保视频包含 FINISHED；
  - 默认参数改为 `--video-stride=2 --fps=20`。
- `scripts/run_odorsim_obstacle_5seeds.sh`、`scripts/generate_odorsim_videos_env.sh`、`run_odorsim_obstacle_5seeds.bat`、`run_odorsim_from_windows.ps1`：同步更新为 `--video-stride 2 --fps 20`。

### 修复后视频规格

| seed | 帧数 | 时长 (@20fps) | 说明 |
|------|------|---------------|------|
| 42 | 35 | 1.8 s | 包含最终 FINISHED 帧 |
| 123 | 152 | 7.6 s | 机器人运动更连贯 |
| 456 | 1001 | 50.0 s | 完整记录 2000 步 |
| 789 | 600 | 30.0 s | 不再缺失结尾 |
| 101112 | 48 | 2.4 s | 包含最终 FINISHED 帧 |

## 六、关键发现与下一步

1. **嗅觉主导结束条件在 OdorSim 中表现更好**：5 种子成功率 80%，高于 2026-08-28 视觉主导版本的 20%（同 5 种子对比）。
2. **失败模式仍集中在“高浓度后丢失”**：seed 456 在检测到高浓度后进入 LOST/逃离，后续未能复捕烟羽；可考虑在高浓度时降低逃离概率或增强复捕策略。
3. **障碍物避障有效**：全部 5 轮均无碰撞；修复后的视频渲染与轨迹图一致。
4. **GADEN 查询量较大**：seed 456 因长时间运行产生 13374 次查询；seed 789 为 6046 次，成功种子平均查询量更低。

---

# 2026-08-31 修改与运行日志

## 一、今日目标

按上一次日志中的第一条高优先级建议，将搜索算法的**结束条件从“视觉主导”改为“嗅觉主导”**，同时保留独立评估体系（真实源与声明源距离 < 1 m）。随后**在 8×8 m 仓库中加入障碍物**，验证机器人在不碰撞的前提下完成搜索。

## 二、核心修改

### 1. `src/search_algorithm.py`

- 重写 `_should_declare_finished`：
  - **嗅觉主导**：`ee_ppm ≥ 1000 ppm` 连续 40 步、末端为局部浓度高点（±x/±y/±z 采样）、基座最近窗口内水平位移 < 0.20 m；
  - **视觉辅助**：相机看到源实体且距离 < 0.80 m、浓度足够时，连续 3 步可结束，仅作为加速条件，不再唯一。
- 新增局部峰值判断 `_is_local_concentration_peak` 与基座稳定性 `_recent_base_displacement`。
- 调整 `_is_local_maximum_stuck`：当浓度接近嗅觉结束阈值（≥800 ppm）时不再触发 LOST/逃离，避免把真实源附近高浓度平台误判为 filament。

### 2. 新配置文件 `config/warehouse_obstacle_8x8.yaml`

- 基于 `warehouse_empty_8x8.yaml`，在 8×8 m 仓库中加入 3 个 box 障碍物和 1 个竖直 cylinder 管道；
- 保持 `randomize: true`，源随机范围 [-2, 3] × [-2, 3] × [0.5, 0.9]，与障碍物、起点保持安全间距。

### 3. 脚本增强

- `scripts/run_search.py`、`scripts/run_random_seed_benchmark.py` 新增 `--env-config` 参数，可指定自定义环境配置；
- 新增 `scripts/analyze_random_seed_results.py`，自动读取 benchmark JSON 并生成 Markdown 分析报告。

## 三、实验结果

### 1. 结束条件修改后：解析环境 30 随机种子

```powershell
python scripts/run_random_seed_benchmark.py --num-runs 30 --max-steps 2000 --output-dir outputs/random_seed_runs_odor_term --seed-offset 2026
```

| 指标 | 数值 |
|------|------|
| 成功率 | 17/30 = 56.7% |
| 碰撞-free | 30/30 |
| 成功平均步数 | 261.8 |
| 成功平均误差 | 0.477 m |
| 成功最大误差 | 0.759 m |

失败模式：8 次完全无气味检测，3 次低浓度未有效追踪，2 次追踪后丢失/被困。无明显的嗅觉误结束案例。

### 2. 环境复杂化后：带障碍 8×8 仓库 30 随机种子

```powershell
python scripts/run_random_seed_benchmark.py --num-runs 30 --max-steps 2000 --output-dir outputs/random_seed_runs_obstacle --seed-offset 2026 --env-config config/warehouse_obstacle_8x8.yaml
```

| 指标 | 数值 |
|------|------|
| 成功率 | 30/30 = 100.0% |
| 碰撞-free | 30/30 |
| 成功平均步数 | 133.4 |
| 成功平均误差 | 0.432 m |
| 成功最大误差 | 0.759 m |

### 3. OdorSim/GADEN 30 种子验证（带障碍 8×8）

```bash
bash scripts/run_odorsim_obstacle_benchmark.sh
```

| 指标 | 数值 |
|------|------|
| 成功率 | 22/30 = 73.3% |
| 碰撞-free | 30/30 |
| 成功平均步数 | 447.0 |
| 成功平均误差 | 0.642 m |
| 全体平均步数 | 702.3 |
| 全体平均误差 | 1.441 m |

失败种子：17975、79072、85859、65313、29900、96699、60457、75297。主要失败模式为“检测到高浓度后丢失”，后续随机游走未能复捕烟羽。

### 4. OdorSim/GADEN 视频验证

#### 带障碍物环境（5 个种子）

```bash
bash scripts/generate_odorsim_videos_env.sh config/warehouse_obstacle_8x8.yaml outputs/odorsim_obstacle 42 2739 36611 9936 51563
```

全部成功、0 碰撞，视频保存在 `outputs/odorsim_obstacle/video_seed<SEED>.mp4`。

#### 无障碍环境（5 个种子，同种子对比）

```bash
bash scripts/generate_odorsim_videos_env.sh config/warehouse_empty_8x8.yaml outputs/odorsim_empty 42 2739 36611 9936 51563
```

全部成功、0 碰撞，视频保存在 `outputs/odorsim_empty/video_seed<SEED>.mp4`。

## 四、产出文件

- `outputs/random_seed_runs_odor_term/random_seed_benchmark_results.json`
- `outputs/random_seed_runs_odor_term/analysis_odor_term.md`
- `outputs/random_seed_runs_obstacle/random_seed_benchmark_results.json`
- `outputs/random_seed_runs_obstacle/analysis_obstacle.md`
- `outputs/odorsim_obstacle_30seeds/aggregate_evaluation.json`
- `outputs/odorsim_obstacle/video_seed*.mp4`（5 个）
- `outputs/odorsim_obstacle/history_seed*.npz`、`summary_seed*.png`
- `outputs/odorsim_empty/video_seed*.mp4`（5 个）
- `outputs/odorsim_empty/history_seed*.npz`、`summary_seed*.png`
- `config/warehouse_obstacle_8x8.yaml`

## 五、下一步建议

1. 在更大场景（如 20×16 m）或障碍物更密集的配置下重复 30 种子测试，验证远距离/复杂布局下的成功率；
2. 如果要在 OdorSim/GADEN 中取消视觉辅助，可进一步降低嗅觉阈值（如 500 ppm）并配合更严格的局部峰值/更大空间窗口；
3. 将当前 8×8 障碍物场景对应的 GADEN 场景 CAD 模型也加入障碍物，使气体扩散与可视化一致。

---

# 2026-08-28 修改与运行日志

## 一、今日改动总览

### 1. 核心算法 `src/search_algorithm.py`

进行了大幅重构，主要解决三个问题：
- **结束条件误判**：原版本会在局部高浓度丝（filament）处提前声明成功；
- **使用真实源位置作弊**：原算法在 `_arm_tracking_action`、`_should_declare_finished`、`_visual_approach_action` 中直接使用 `env.source_pos`；
- **随机游走覆盖不足 / 绕圈**：原随机游走缺乏系统性， Seed 2 视频中可见大片区域未被覆盖。

具体改动：
- **移除真实源位置依赖**：
  - `_arm_tracking_action` 不再使用 `env.source_pos`，改为仅根据局部浓度梯度移动；
  - `_should_declare_finished` 不再使用 `env.source_pos` 计算距离；
  - `_visual_approach_action` 不再直接朝向 `env.source_pos`，而是根据视觉输出的 `image_xy` 计算相机射线方向；
  - `_base_tracking_action` 的末端目标高度改用当前 EE 高度，而非源高度。
- **结束条件改为视觉主导**：
  - 必须满足 `source_visible == True` 且视觉测距 `< source_distance_threshold`（默认 0.4 m）且浓度 `> 10×detection_threshold`，连续 3 步；
  - 完全移除了基于嗅觉平台的结束判据，避免局部高浓度丝误触发。
- **局部极值逃离机制**：
  - 新增 `_is_local_maximum_stuck`（基于基座位移和浓度平台）；
  - 检测到局部极值或 ARM_TRACKING 400 步未视觉确认时，进入 `LOST`；
  - 设置 `_escape_cooldown = 120` 步，期间不再重新进入 `ARM_TRACKING` / `BASE_TRACKING`；
  - 添加 `_no_track_zones`（半径 2.0 m），避免反复回到同一 filament；
  - 添加 `_escape_target`，根据梯度反方向强制基座远离 3 m。
- **随机游走改为牛耕法（boustrophedon）覆盖**：
  - 每个机械臂高度层生成平行于 x 轴的往复轨道（间距 1.2 m）；
  - 机器人按轨道系统性地扫过仓库；
  - 轨道被堵时回退到前沿方向选择；
  - 完成一层覆盖后自动切换高度层并重build轨迹；
  - 保留 `visited` 网格和高度层清除逻辑。

### 2. 独立评估体系 `src/evaluation.py`（新增）

- `SourceSearchEvaluator`：离线计算机器人声明源位置 / 最终末端位置与真实源位置的距离；
- 默认成功阈值 1.0 m，独立于机器人决策逻辑。

### 3. 仿真会话 `src/simulation.py`、`src/simulation_odorsim.py`

- 每条历史记录中加入 `declared_source_position`；
- `get_summary()` 返回 `declared_source_position`。

### 4. 运行脚本

- `scripts/run_search.py`：新增 `--success-threshold`，运行后打印独立评估结果；
- `scripts/run_odorsim_verification.py`：新增 `--success-threshold`，运行后打印独立评估结果；
- `scripts/evaluate_odorsim_runs.py`（新增）：批量聚合评估，输出 JSON；
- `scripts/generate_odorsim_videos.sh`（新增）：批量生成视频。

### 5. 文档更新

- `docs/ALGORITHM.md`：按新状态机和行为重写；
- `docs/ODORSIM_INTEGRATION.md`：更新文件清单、当前结果、设计要点；
- `docs/USAGE.md`：补充 `--success-threshold` 和聚合评估说明；
- `docs/README.md`：更新项目结构和算法流程；
- `src/__init__.py`：导出 `SourceSearchEvaluator`。

## 二、当前运行方式

### 解析环境快速测试

```powershell
cd C:\Users\lenovo\Desktop\OSL_mechanic_arm\OdorSearch_MobileArm
python scripts/run_search.py --seed 42 --max-steps 1000 --success-threshold 1.0
```

### OdorSim/GADEN 单种子带视频

```bash
# WSL Ubuntu-24.04
cd /home/odorsim/OdorSim
source setup/activate.sh
cd /mnt/c/Users/lenovo/Desktop/OSL_mechanic_arm/OdorSearch_MobileArm
python scripts/run_odorsim_verification.py --seed 2 --max-steps 2000 --video-stride 20 --fps 10 --success-threshold 1.0
```

### OdorSim 批量验证

```bash
bash scripts/run_odorsim_benchmark.sh 1 2 3 4 5
```

### 批量生成视频

```bash
bash scripts/generate_odorsim_videos.sh 1 2 3 4 5 42
```

### 聚合评估

```bash
python scripts/evaluate_odorsim_runs.py --out-dir outputs/odorsim --seeds 1 2 3 4 5 42 --save-json outputs/odorsim/aggregate_evaluation.json
```

## 三、当前结果

### 解析环境

| seed | 结果 | 误差 |
|------|------|------|
| 42   | ✅ 成功 | 0.145 m |

结束位置精度较旧版本（~0.6 m）有明显提升。

### OdorSim/GADEN

- **Seed 42**：成功，误差约 0.6 m（视觉确认生效）。
- **Seed 2**：仍然失败。机器人能在 2000 步内多次接近源附近（例如 step 1200 到达 `(0.79, 0.91)`，距源仅 ~1.4 m），但：
  - 未能在该处检测到足够浓度（可能高度不对或 GADEN 烟羽极窄）；
  - 随后被局部高浓度丝拉向仓库右侧/上侧，最终卡在 `(3.02, 3.70)` 等区域；
  - 牛耕法轨迹在局部被阻挡/绕圈后，仍会出现长时间不移动的情况。

上一次 Seed 2 运行的状态分布（step 2000）：
- RANDOM_WALK 为主，后期停滞在 `(3.02, 3.70)`。

## 四、遗留问题

1. **Seed 2 失败：覆盖率与检测问题**
   - 机器人理论上已按牛耕法扫过源附近，但实际未触发气味检测；
   - 需要确认：机械臂高度预设是否匹配源高度、GADEN 烟羽是否极窄、视觉/嗅觉触发阈值是否合适。

2. **随机游走仍可能“卡住”**
   - 牛耕法在空旷 8×8 仓库应能覆盖，但当机器人处于局部高浓度区域时，状态切换/no-track zone/escape 之间的交互导致基座长时间不移动；
   - `_follow_waypoint` 的前方碰撞检测可能过于保守，导致频繁回退到前沿选择；
   - 需要检查 `_escape_action` 是否真正把机器人带离 filament，而不是带到另一个角落。

3. **结束条件依赖视觉**
   - 当前仅视觉确认可结束。若源在视觉盲区或被遮挡，机器人会超时失败；
   - 是否需要引入“多方向确认”或“气味峰值 + 近距离视觉”的折中方案，待讨论。

4. **超参数待调**
   - `_arm_tracking_max_steps = 400`、`_escape_cooldown_steps = 120`、no-track radius 2.0 m、track spacing 1.2 m、visual distance threshold 0.4 m 等均为经验值；
   - 需要在多个 seed 上系统测试。

5. **GADEN 查询量大**
   - ARM_TRACKING 每步调用 `_estimate_gradient`（6 次 `concentration_at`），单步查询量大；
   - 已在 `odor_sim_adapter.py` 中通过批量查询优化，但若频繁在 ARM/RANDOM 间切换，总量仍高。

## 五、下一步建议

1. **调试 Seed 2 的覆盖轨迹**
   - 打印或可视化 `_coverage_tracks` 与实际基座轨迹，确认牛耕法是否被正确执行；
   - 检查 `_follow_waypoint` 中 `front_clear` 判定是否导致机器人过早放弃轨道；
   - 考虑在轨道跟随失败时，直接跳过当前路点而不是完全回退到前沿选择。

2. **增强“被卡住”检测与恢复**
   - 在 RANDOM_WALK 中加入基于基座位移的卡住检测（参考 `_check_stuck`）；
   - 卡住时强制转向仓库中心或最近未访问区域，而不是继续当前偏航。

3. **视觉与嗅觉融合再设计**
   - 若视觉确认不可用，考虑使用“高浓度 + 梯度从四周指向中心”作为辅助结束条件；
   - 或要求视觉确认作为唯一结束条件，同时优化视觉扫描行为，确保机器人在高浓度区会主动旋转相机寻找源实体。

4. **机械臂高度扫描**
   - 当前每个高度层覆盖完整仓库后再切换，可能错过源；
   - 可改为“在每个 xy 位置快速扫过三个高度”或在 RANDOM_WALK 中让 EE 做小幅上下扫摆。

5. **多 seed 批量回归测试**
   - 对 seeds 1,2,3,4,5,42 重新运行并统计成功率；
   - 使用 `scripts/evaluate_odorsim_runs.py` 生成 aggregate JSON，观察误差分布。

## 六、关键文件版本

- `src/search_algorithm.py`：当前为视觉主导 + 牛耕法覆盖 + 局部极值逃离版本；
- `src/evaluation.py`：独立评估器，未改动；
- `src/simulation.py`、`src/simulation_odorsim.py`：记录 declared_source_position；
- `scripts/run_odorsim_verification.py`：支持 `--success-threshold`；
- `docs/TODAY_LOG.md`：本文件。


---

# 2026-08-28 后续补充：普适性测试、OdorSim 视频生成与综合评估

## 一、本次补充所做工作

### 1. 解析环境 30 次随机种子批量测试

新增脚本 `scripts/run_random_seed_benchmark.py`，使用 30 个随机种子（`seed_offset=2026`）在解析环境中测试算法普适性：

```powershell
cd C:\Users\lenovo\Desktop\OSL_mechanic_arm\OdorSearch_MobileArm
python scripts/run_random_seed_benchmark.py --num-runs 30 --max-steps 2000 --output-dir outputs/random_seed_runs
```

评估方式使用独立评估器 `SourceSearchEvaluator`，以**机器人声明的源位置 / 最终末端位置与真实源位置的距离**是否小于 1.0 m 作为成功标准。

### 2. OdorSim/GADEN 5 次随机种子视频生成

从 30 个随机种子中按 `picker seed=2026` 随机抽取 5 个：

- seeds：`9936, 36611, 72823, 83123, 85200`

使用修改后的 `scripts/generate_odorsim_videos.sh`（单个 seed 失败后继续运行）生成视频与仿真文件：

```bash
wsl -d Ubuntu-24.04 bash -c "cd /mnt/c/Users/lenovo/Desktop/OSL_mechanic_arm/OdorSearch_MobileArm && source /home/odorsim/OdorSim/setup/activate.sh && bash scripts/generate_odorsim_videos.sh 9936 36611 72823 83123 85200"
```

### 3. 失败种子分析

对解析环境与 OdorSim 的失败种子分别进行了分类与根因推断。

---

## 二、测试结果

### 1. 解析环境 30 次随机种子

| 指标 | 数值 |
|------|------|
| 成功率 | **17/30 = 56.7%** |
| 碰撞-free | 30/30 |
| 成功平均步数 | 456.5 步 |
| 成功平均误差 | 0.221 m |
| 成功最大误差 | 0.350 m |

**成功源位置分布**：平均距起始点 8.66 m（±4.51 m），最近 0.92 m，最远 16.78 m。  
**失败源位置分布**：平均距起始点 13.93 m（±2.95 m），最近 10.53 m，最远 19.30 m。  
**象限分布**：
- NE（x>0, y>0）：3/11 成功
- NW（x<0, y>0）：3/6 成功
- SW（x<0, y<0）：8/8 成功
- SE（x>0, y<0）：3/5 成功

**失败模式分类（13 个失败种子）**：

| 类型 | 数量 | 典型特征 |
|------|------|----------|
| 完全无气味检测 | 8 | 全部 RANDOM_WALK，max EE ppm < 5 |
| 低浓度检测但被困 | 3 | max EE ppm 35–122，有 BASE/ARM_TRACKING 但进入 LOST |
| 中等浓度未确认 | 1 | max EE ppm ~279，反复 BASE/ARM/LOST |
| 高浓度但无视觉确认 | 1 | max EE ppm 58，状态复杂但未 FINISHED |

**核心发现**：失败种子几乎都集中在**距离起始点较远（>10 m）且位于东北/西北象限**的区域；机器人在这些场景下容易被局部烟羽/障碍物困住，或根本检测不到足够浓度。

### 2. OdorSim/GADEN 5 次随机种子

| seed | 结果 | 最终距离 | 关键现象 |
|------|------|----------|----------|
| 9936 | ❌ 失败 | 3.278 m | 多次检测到 75–873 ppm，但始终未进入追踪/结束 |
| 36611 | ✅ 成功 | 0.104 m | 760 步 FINISHED |
| 72823 | ❌ 失败 | 3.493 m | step 400 检测到 5916 ppm，随后回到 RANDOM_WALK |
| 83123 | ❌ 失败 | 3.156 m | step 200 检测到 8134 ppm，随后回到 RANDOM_WALK |
| 85200 | ❌ 失败 | 4.301 m | step 400 检测到 16011 ppm，随后回到 RANDOM_WALK |

**OdorSim 成功率：1/5 = 20%**

**关键发现**：
- GADEN 烟羽在局部可产生极高浓度（>10000 ppm），但机器人经常在检测到高浓度后回到 RANDOM_WALK，未能持续追踪到源；
- 视觉主导结束条件在 OdorSim 中很难满足：相机视场/遮挡导致即使末端浓度极高，也无法看到源实体；
- 逃离/局部极值机制在 GADEN 的湍流烟羽中过于敏感，容易把真实高浓度区误判为局部 filament 而逃离。

---

## 三、综合评估

### 当前算法优势

1. **无真实源位置作弊**：结束条件、追踪动作均只使用传感器与视觉输出；
2. **成功时精度高**：解析环境成功平均误差 0.22 m，最大 0.35 m；
3. **碰撞-free**：30/30 随机种子无碰撞；
4. **障碍物感知覆盖**：引入 `OccupancyGrid2D` 与 A* 后，牛耕法轨迹能避开静态障碍物；
5. **局部极值逃离**：对解析环境中的远距离低浓度 filament 有一定抑制作用。

### 当前算法不足

1. **远距离/复杂方位成功率低**：解析环境 56.7%，OdorSim 仅 20%；
2. **视觉主导结束条件在 OdorSim 中过严**：GADEN 高浓度区经常无法同时满足视觉确认；
3. **牛耕法覆盖仍不够鲁棒**：在障碍物密集或边界附近容易卡住、反复震荡；
4. **高度层扫描效率低**：源高度匹配问题在 OdorSim 中更明显；
5. **状态切换敏感**：高浓度但无视觉确认时容易触发 LOST/逃离，导致错过真实源。

---

## 四、下一步可修改建议

### 高优先级

1. **放宽结束条件，引入嗅觉主导兜底**
   - 当前仅视觉可结束。建议在末端浓度持续高于阈值、梯度指向中心、且多方向浓度从四周指向中心时，允许以嗅觉为主的结束；
   - 或保留视觉结束，但将视觉测距阈值从 0.4 m 放宽到 0.8 m，并增强视觉扫描行为。

2. **改进 OdorSim 中的高浓度保持机制**
   - 当 ee_ppm > 1000 时，降低逃离概率，优先做原地视觉扫描/基座旋转寻找源实体；
   - 只有浓度下降且持续无视觉时才允许逃离。

3. **修复“检测到高浓度后回退 RANDOM_WALK”问题**
   - 检查状态机：为何 ARM_TRACKING 中高浓度会退到 RANDOM_WALK（可能是 `_arm_tracking_max_steps` 超时或 `_is_local_maximum_stuck` 误判）；
   - 增加“高浓度锁定”：连续多步高浓度时，禁止回到 RANDOM_WALK。

### 中优先级

4. **障碍物地图与覆盖路径再优化**
   - 占用网格膨胀半径 1.0 m 导致某些可行通道被过度阻塞，可改为根据车体尺寸动态膨胀；
   - 牛耕法轨道生成时，避免轨道太靠近边界或障碍物；轨道间转移使用 A* 全局路径而非局部探测。

5. **高度扫描策略**
   - 每个 xy 位置快速扫过 low/mid/high 三个高度，而不是每个高度层完整覆盖仓库；
   - 在 BASE_TRACKING/ARM_TRACKING 中让末端做小幅上下扫摆，提高 z 方向匹配概率。

6. **引入烟羽追踪经典策略**
   - 当浓度高但梯度不明显时，采用 cross-wind cast（横向扫描）寻找烟羽中心；
   - 当浓度低丢失时，沿估计风向做 surge（快速前进）重新捕获烟羽。

### 低优先级 / 工程化

7. **OdorSim 批量回归测试框架**
   - 扩展 `scripts/run_odorsim_benchmark.sh` 支持 30 个随机种子，自动统计成功率与查询量；
   - 将 OdorSim 结果与解析环境结果对比，建立“解析环境通过但 OdorSim 失败”的种子清单。

8. **减少 GADEN 查询量**
   - ARM_TRACKING 每步 7 次查询仍较高；可降低梯度估计频率（每 3–5 步一次）或使用历史梯度插值。

---

## 五、生成文件清单

- `outputs/random_seed_runs/random_seed_benchmark_results.json`
- `outputs/random_seed_runs/summary_seed<SEED>.png`（30 张）
- `outputs/random_seed_runs/history_seed<SEED>.npz`（30 个）
- `outputs/random_seed_runs/seeds_for_video.txt`
- `outputs/odorsim/video_seed<SEED>.mp4`（9936, 36611, 72823, 83123, 85200）
- `outputs/odorsim/summary_seed<SEED>.png`（同上）
- `outputs/odorsim/history_seed<SEED>.npz`（同上）

---

## 六、关键文件版本（更新后）

- `src/search_algorithm.py`：视觉主导 + 牛耕法覆盖 + OccupancyGrid2D + 局部极值逃离 + 强制卡住恢复；
- `src/evaluation.py`：独立评估器，以 declared/final EE 位置与真实源距离判定成功；
- `scripts/run_random_seed_benchmark.py`：新增 30 随机种子解析环境批量测试；
- `scripts/pick_random_seeds_for_video.py`：新增从结果中抽取 OdorSim 视频种子；
- `scripts/generate_odorsim_videos.sh` / `scripts/run_odorsim_benchmark.sh`：单个 seed 失败后继续运行；
- `docs/TODAY_LOG.md`：本文件。
