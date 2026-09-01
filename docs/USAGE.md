# OdorSearch_MobileArm 运行说明文档

本文档详细介绍如何运行 `OdorSearch_MobileArm` 项目中的算法验证、OdorSim/GADEN 仿真验证以及策略优化训练。

---

## 目录

1. [环境准备](#1-环境准备)
2. [原始算法仿真（无 OdorSim）](#2-原始算法仿真无-odorsim)
3. [OdorSim/GADEN 仿真验证](#3-odorsimgaden-仿真验证)
4. [多种子批量验证](#4-多种子批量验证)
5. [结果查看与可视化](#5-结果查看与可视化)
6. [策略优化训练（PPO）](#6-策略优化训练ppo)
7. [参数速查表](#7-参数速查表)
8. [常见问题](#8-常见问题)

---

## 1. 环境准备

### 1.1 项目位置

```text
C:\Users\lenovo\Desktop\OSL_mechanic_arm\OdorSearch_MobileArm
```

在 WSL 中对应路径：

```text
/mnt/c/Users/lenovo/Desktop/OSL_mechanic_arm/OdorSearch_MobileArm
```

### 1.2 Windows 端（查看代码、结果）

无需额外安装，使用文件浏览器或 PowerShell 即可。

### 1.3 WSL 端（运行仿真）

必须进入 OdorSim 环境：

```bash
cd /home/odorsim/OdorSim
source setup/activate.sh
```

激活成功后会显示：

```text
[activate] OdorSim env ready (ROS jazzy + GADEN + venv).
```

然后进入项目目录：

```bash
cd /mnt/c/Users/lenovo/Desktop/OSL_mechanic_arm/OdorSearch_MobileArm
```

### 1.4 RL 训练额外依赖

如果运行 PPO 训练，需要安装：

```bash
pip install gymnasium stable-baselines3 tensorboard
```

---

## 2. 原始算法仿真（无 OdorSim）

使用简化高斯烟羽模型，不启动 GADEN，运行最快，适合快速调试算法逻辑。

### 2.1 基本运行

```bash
python scripts/run_search.py --seed 42 --max-steps 3000
```

### 2.2 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--seed` | `42` | 随机种子，决定障碍物、源位置、风场 |
| `--max-steps` | `3000` | 单轮最大仿真步数 |
| `--start-x` | `-7.0` | 机器人初始 X 位置（m） |
| `--start-y` | `-6.5` | 机器人初始 Y 位置（m） |
| `--visualize` | 无 | 加此参数会弹出 3D matplotlib 窗口 |
| `--save-plot` | `None` | 保存摘要图路径，如 `outputs/summary.png` |
| `--save-history` | `None` | 保存历史 NPZ 路径，如 `outputs/history.npz` |

### 2.3 示例

```bash
# 保存历史、保存摘要图、不弹窗
python scripts/run_search.py --seed 1 --max-steps 2000 --save-history outputs/history.npz --save-plot outputs/summary.png

# 可视化（带弹窗）
python scripts/run_search.py --seed 7 --max-steps 1000 --visualize
```

### 2.4 输出解读

```text
Simulation finished.
  Success: True
  Total steps: 264
  Final state: FINISHED
  Final distance to source: 0.523 m
  Max EE ppm: 2847.32
  State counts: {'RANDOM_WALK': 120, 'BASE_TRACKING': 45, 'ARM_TRACKING': 99}
  Collision count: 0
```

---

## 3. OdorSim/GADEN 仿真验证

使用 GADEN 真实气体扩散模型替换高斯烟羽，验证算法在更真实烟羽下的表现，并可保存 MP4 视频。

### 3.1 基本运行

```bash
python scripts/run_odorsim_verification.py --seed 42 --max-steps 2000
```

### 3.2 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--seed` | `42` | 随机种子 |
| `--max-steps` | `3000` | 单轮最大步数（上限，找到源会提前结束） |
| `--scenario` | `odorsim_scenarios/warehouse_8x8/environment_configurations/config1` | GADEN 场景路径 |
| `--scene-id` | `scene1` | GADEN scene id |
| `--gaden-dt` | `0.05` | GADEN 仿真步长，需与 `sim.yaml` 中 `deltaTime` 一致 |
| `--env-config` | `config/warehouse_empty_8x8.yaml` | 仓库环境配置文件（常用 `config/warehouse_obstacle_8x8.yaml`） |
| `--out-dir` | `outputs/odorsim` | 输出目录 |
| `--fps` | `10` | 视频帧率 |
| `--video-stride` | `5` | 每 N 步保存一帧视频 |
| `--no-video` | 无 | 不生成视频，只保存历史/摘要图 |
| `--start-x` | `-3.0` | 初始 X |
| `--start-y` | `-3.0` | 初始 Y |
| `--success-threshold` | `1.0` | 独立评估成功距离阈值（m） |

### 3.3 示例

```bash
# 生成视频（默认 8x8 场景）
python scripts/run_odorsim_verification.py --seed 42 --max-steps 2000 --video-stride 10 --fps 15

# 不生成视频，只看统计结果
python scripts/run_odorsim_verification.py --seed 5 --max-steps 2000 --no-video

# 使用更大的 20x16 场景
python scripts/run_odorsim_verification.py \
    --scenario ./odorsim_scenarios/warehouse_20x16/environment_configurations/config1 \
    --env-config ./config/warehouse_empty_20x16.yaml \
    --start-x -7.0 --start-y -6.5 \
    --seed 42 --max-steps 5000 --no-video

# 自定义起点，观察不同探索路径
python scripts/run_odorsim_verification.py --seed 3 --start-x -2.0 --start-y -3.5 --max-steps 2000
```

### 3.4 输出文件

运行后会在 `outputs/odorsim/` 下生成：

```text
outputs/odorsim/
├── history_seed<N>.npz      # 搜索历史（base_pose, ee_pos, joint_angles, sensor_readings 等）
├── summary_seed<N>.png      # 轨迹俯视图 + 高度曲线 + 浓度曲线
├── video_seed<N>.mp4        # 三维搜索过程视频（matplotlib 渲染）
└── gaden_logs/              # GADEN 服务器日志
```

### 3.5 重要提示

- `--max-steps` 只是**上限**。如果机器人提前找到源（进入 `FINISHED`），仿真会提前结束。
- 视频是用 **matplotlib** 渲染的，不是 OdorSim 的 MuJoCo 画面。OdorSim 在这里只负责**气体扩散计算（GADEN）**。
- 若要看 OdorSim 的 3D 物理画面，需要把 UMR+RM65 模型导入 OdorSim/robosuite，这是另一个阶段的工作。

---

## 4. 多种子批量验证

### 4.1 运行脚本

```bash
bash scripts/run_odorsim_benchmark.sh 1 2 3 4 5

# 30 新种子有障碍验证（当前最佳算法 v3，成功率 28/30 = 93.3%）
bash scripts/run_odorsim_obstacle_30seeds_v3.sh
```

### 4.2 参数说明

脚本接受一个种子列表，默认是 `1 2 3 4 5`。

```bash
# 自定义种子
bash scripts/run_odorsim_benchmark.sh 10 20 30

# 使用默认种子 1~5
bash scripts/run_odorsim_benchmark.sh
```

### 4.3 输出

为每个种子生成 `history_seed<N>.npz` 和 `summary_seed<N>.png`，方便批量统计成功率。

### 4.4 独立聚合评估

批量运行后，使用独立评估脚本统计成功率：

```bash
python scripts/evaluate_odorsim_runs.py --out-dir outputs/odorsim --seeds 1 2 3 4 5 42 --save-json outputs/odorsim/aggregate_evaluation.json
```

输出包含每个种子的误差、是否成功、平均步数等。

### 4.5 解析环境 30 次随机种子批量测试

用于快速评估算法在解析高斯烟羽下的普适性：

```powershell
python scripts/run_random_seed_benchmark.py --num-runs 30 --max-steps 2000 --output-dir outputs/random_seed_runs
```

结果会保存到 `outputs/random_seed_runs/random_seed_benchmark_results.json`，并打印失败种子列表。

### 4.6 从批量结果中抽取 OdorSim 视频种子

```powershell
python scripts/pick_random_seeds_for_video.py --num-pick 5 --output outputs/random_seed_runs/seeds_for_video.txt --seed 2026
```

然后在 WSL 中运行：

```bash
source /home/odorsim/OdorSim/setup/activate.sh
cd /mnt/c/Users/lenovo/Desktop/OSL_mechanic_arm/OdorSearch_MobileArm
SEEDS=$(cat outputs/random_seed_runs/seeds_for_video.txt)
bash scripts/generate_odorsim_videos.sh $SEEDS

# 生成三视角视频（3D + 俯视 XY + 前视 XZ，算法 v3）
bash scripts/generate_odorsim_videos_triple_v3.sh 30991 39036 64497 84224 98492
```

---

## 5. 结果查看与可视化

### 5.1 查看保存的 NPZ 历史

```python
import numpy as np

hist = np.load("outputs/odorsim/history_seed42.npz", allow_pickle=True)
print(hist.files)
# ['base_poses', 'ee_positions', 'joint_angles', 'steps', 'states', 'source_pos']

print("源位置:", hist["source_pos"])
print("总步数:", len(hist["steps"]))
```

### 5.2 生成静态 3D 图

```bash
python scripts/plot_3d.py --history outputs/odorsim/history_seed42.npz --output outputs/odorsim/warehouse_3d.png
```

参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--history` | `outputs/final_demo.npz` | 历史文件路径 |
| `--output` | `outputs/warehouse_3d.png` | 输出图片路径 |

### 5.3 查看视频

视频保存在 `outputs/odorsim/video_seed<N>.mp4`，直接用播放器打开即可。

---

## 6. 策略优化训练（PPO）

### 6.1 安装依赖

```bash
pip install gymnasium stable-baselines3 tensorboard
```

### 6.2 基本运行

```bash
python scripts/train_ppo.py --seed 42 --total-timesteps 200000 --max-episode-steps 1000
```

### 6.3 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--seed` | `42` | 训练随机种子 |
| `--total-timesteps` | `200000` | 总训练步数 |
| `--room-half` | `4.0` | 房间半长（用于观测归一化） |
| `--max-episode-steps` | `1000` | 每回合最大步数 |
| `--scenario` | `odorsim_scenarios/warehouse_8x8/...` | GADEN 场景 |
| `--env-config` | `config/warehouse_empty_8x8.yaml` | 环境配置 |
| `--save-dir` | `outputs/rl` | 模型保存目录 |

### 6.4 输出

```text
outputs/rl/
├── ppo_odor_search_final.zip   # 最终模型
├── checkpoints/                # 每 50000 步的检查点
├── tensorboard/                # TensorBoard 日志
├── monitor.csv                 # 训练监控数据
└── gaden_logs/                 # GADEN 日志
```

### 6.5 TensorBoard 查看

```bash
tensorboard --logdir outputs/rl/tensorboard
```

然后在浏览器打开 `http://localhost:6006`。

### 6.6 加载训练好的模型继续运行

```python
from stable_baselines3 import PPO

model = PPO.load("outputs/rl/ppo_odor_search_final")
```

---

## 7. 参数速查表

### 7.1 仓库环境配置参数（warehouse_empty_8x8.yaml）

| 配置项 | 说明 |
|--------|------|
| `environment.bounds` | 房间边界 `[x_min, x_max]`, `[y_min, y_max]`, `[z_min, z_max]` |
| `environment.dt` | 仿真步长，需与 GADEN `deltaTime` 一致 |
| `environment.max_steps` | 默认最大步数 |
| `environment.randomize` | 是否每次 reset 随机化源/障碍物/风场 |
| `random_generation.source` | 源位置随机范围 |
| `odor_source.position` | 固定源位置（`randomize=false` 时生效） |
| `odor_source.wind` | 环境风速向量（m/s） |
| `plume.detection_threshold` | 传感器触发阈值（ppm） |

### 7.2 GADEN 场景参数（sim.yaml）

| 参数 | 说明 |
|------|------|
| `deltaTime` | GADEN 每步推进时间，需与 `env.dt` 一致 |
| `filamentPPMcenter` | 单个 filament 中心浓度 |
| `filamentInitialSigma` | filament 初始标准差 |
| `filamentGrowthGamma` | filament 增长系数 |
| `numFilaments_sec` | 每秒释放 filament 数量 |

### 7.3 机器人配置参数（robot.yaml）

| 参数 | 说明 |
|------|------|
| `mobile_base.length/width/height` | 小车尺寸（m） |
| `mobile_base.max_position_delta` | 每步最大平移量（m） |
| `mobile_base.max_yaw_delta` | 每步最大偏航角变化（rad） |
| `corner_sensors` | 四个角传感器相对车体中心位置 |
| `arm.joint_limits_deg` | 各关节角度限位 |
| `arm.max_joint_speed` | 各关节最大角速度（°/s） |
| `arm.preset_poses` | 随机游走阶段使用的低/中/高姿态 |

---

## 8. 常见问题

### Q1: 为什么 `--max-steps` 从 2000 改成 5000，视频还是一样？

**A:** `--max-steps` 只是上限。如果机器人在 473 步时就找到源并进入 `FINISHED` 状态，仿真会提前结束。因此 2000 和 5000 的实际运行步数相同，视频也相同。要得到不同视频，可以换 `--seed` 或起点位置。

### Q2: 视频为什么像 matplotlib 画的，不像 OdorSim？

**A:** 当前视频确实是 matplotlib 渲染的。OdorSim 在这里只负责**气体扩散（GADEN）**计算，机器人和仓库的可视化仍用原项目的 matplotlib 可视化器。要获得真正的 OdorSim 3D 画面，需要把移动小车+机械臂模型导入 OdorSim/robosuite。

### Q3: 运行时报 `TimeoutError: /odor_value did not respond` 怎么办？

**A:** 这是因为原算法的梯度估计会发起 6 次串行 GADEN 查询。本项目已通过 `GadenAwareSearchAgent` 把 6 次查询合并为 1 次批量查询。如果仍超时，请检查：

- 是否已用 `GadenAwareSearchAgent`（在 `simulation_odorsim.py` 中）
- 是否房间过大、cell_size 过小导致 GADEN 响应慢
- 可尝试缩小场景到 `warehouse_8x8`

### Q4: 如何更换 GADEN 场景？

**A:** 修改 `--scenario` 参数指向新的场景目录，并确保 `--env-config` 中的 `bounds` 与该场景一致。例如：

```bash
python scripts/run_odorsim_verification.py \
    --scenario ./odorsim_scenarios/warehouse_20x16/environment_configurations/config1 \
    --env-config ./config/warehouse_empty_20x16.yaml
```

### Q5: 如何修改源位置和风速？

**A:** 有两种方式：

1. **固定源**：修改 `config/warehouse_empty_8x8.yaml` 中的 `odor_source.position` 和 `odor_source.wind`，并设置 `randomize: false`。
2. **随机源范围**：修改 `random_generation.source` 中的 `x_range/y_range/z_range`，保持 `randomize: true`。

GADEN 场景中的 `sim.yaml` 初始源位置会被运行时通过 `/gaden/source_poses` 覆盖，所以不需要改 `sim.yaml`。

### Q6: 训练时提示 `No module named 'gymnasium'`？

**A:** 在激活 OdorSim 环境后运行：

```bash
pip install gymnasium stable-baselines3 tensorboard
```

### Q7: 如何在 Windows 端直接运行 Python？

**A:** 原始算法（`run_search.py`、`plot_3d.py`）可以在 Windows Python 环境直接运行，只需安装 `numpy`、`matplotlib`、`pyyaml`。但 OdorSim/GADEN 相关脚本必须在 WSL 中运行，因为需要 ROS 2 和 GADEN。

---

## 附录：文件结构速览

```text
OdorSearch_MobileArm/
├── config/                          # 环境/机器人配置
│   ├── robot.yaml
│   ├── warehouse.yaml               # 原 20x16 配置（含障碍物）
│   ├── warehouse_empty_8x8.yaml     # 8x8 空仓库配置
│   ├── warehouse_empty_20x16.yaml   # 20x16 空仓库配置
│   └── warehouse_obstacle_8x8.yaml  # 8x8 有障碍仓库配置（当前主验证场景）
├── odorsim_scenarios/               # GADEN 场景
│   ├── warehouse_8x8/
│   └── warehouse_20x16/
├── src/                             # 核心代码
│   ├── environment.py               # 解析环境 WarehouseEnv
│   ├── robot.py                     # UMR + RM65 机器人模型
│   ├── search_algorithm.py          # 搜索算法（视觉主导 + 牛耕法 + OccupancyGrid2D）
│   ├── simulation.py                # 解析环境仿真会话
│   ├── evaluation.py                # 独立评估器
│   ├── odor_sim_adapter.py          # GADEN 适配器
│   ├── simulation_odorsim.py        # OdorSim 版仿真会话
│   └── rl/                          # RL 训练工具
│       ├── env_wrapper.py
│       └── reward_shaper.py
├── scripts/                         # 运行脚本
│   ├── run_search.py
│   ├── run_seed_benchmark.py
│   ├── run_random_seed_benchmark.py
│   ├── pick_random_seeds_for_video.py
│   ├── plot_3d.py
│   ├── run_odorsim_verification.py
│   ├── run_odorsim_benchmark.sh
│   ├── run_odorsim_obstacle_30seeds_v3.sh   # 当前最佳 30 种子批量验证
│   ├── generate_odorsim_videos.sh
│   ├── generate_odorsim_videos_triple_v3.sh # 当前最佳三视角视频
│   ├── evaluate_odorsim_runs.py
│   └── train_ppo.py
├── docs/                            # 文档
│   ├── USAGE.md                     # 本文档
│   ├── ODORSIM_INTEGRATION.md
│   └── optimization_training_plan.md
└── outputs/                         # 运行输出
    ├── odorsim/
    └── rl/
```
