# 项目结构说明

> 本文档描述 `OdorSearch_MobileArm` 的目录与关键文件用途，方便新用户快速定位代码与结果。

---

## 目录总览

```text
OdorSearch_MobileArm/
├── config/                      # 环境/机器人参数配置
├── docs/                        # 项目文档
├── odorsim_scenarios/           # GADEN 仿真场景（气体扩散数据）
├── outputs/                     # 运行输出（视频、历史、评估报告）
├── scripts/                     # 可执行脚本（运行、分析、训练）
├── src/                         # 核心源代码
│   └── rl/                      # 强化学习相关
├── .gitignore                   # Git 忽略规则
├── requirements.txt             # Python 依赖
├── run_demo.bat                 # Windows 快速演示入口
├── run_odorsim_from_windows.bat # Windows → WSL OdorSim 入口
├── run_odorsim_from_windows.ps1
├── run_odorsim_obstacle_5seeds.bat
├── 机器人参数.txt               # 原始机器人参数（中文）
└── 基础算法设计.docx            # 原始算法设计文档
```

---

## 1. `config/` — 配置

| 文件 | 说明 |
|------|------|
| `robot.yaml` | UMR 移动底盘 + RM65-6F-V 机械臂的运动学、限位、传感器布局、预设姿态。 |
| `warehouse.yaml` | 原始 20×16 m 有障碍仓库配置（项目早期基准）。 |
| `warehouse_empty_8x8.yaml` | 8×8 m 空仓库，用于快速算法验证。 |
| `warehouse_empty_20x16.yaml` | 20×16 m 空仓库。 |
| `warehouse_obstacle_8x8.yaml` | **当前主要验证场景**：8×8 m 有障碍仓库（3 box + 1 cylinder）。 |

---

## 2. `docs/` — 文档

| 文件 | 说明 |
|------|------|
| `README.md` | 项目简介与快速开始。 |
| `USAGE.md` | **主要运行说明**：如何运行解析仿真、OdorSim/GADEN 验证、批量测试、视频生成、RL 训练。 |
| `SETUP.md` | 环境安装与 WSL/OdorSim 配置。 |
| `ALGORITHM.md` | 算法设计说明（状态机、梯度追踪、牛耕法覆盖等）。 |
| `BENCHMARK.md` | 各批次 benchmark 结果汇总。 |
| `ODORSIM_INTEGRATION.md` | OdorSim/GADEN 集成细节。 |
| `TODAY_LOG.md` | 按日期记录的实验与迭代日志（含第二批、第三批完整记录）。 |
| `FAILURE_ANALYSIS_AND_PLAN.md` | 第三批 2 个失败种子分析及下一步改进方案。 |
| `PROJECT_STRUCTURE.md` | 本文档。 |
| `optimization_training_plan.md` | RL/PPO 训练计划。 |

---

## 3. `odorsim_scenarios/` — GADEN 场景

```text
odorsim_scenarios/
├── warehouse_8x8/
│   ├── cad_models/              # 仓库/障碍物 CAD 模型（DAE）
│   └── environment_configurations/
│       └── config1/             # GADEN 配置文件（sim.yaml、occupancy、wind 等）
└── warehouse_20x16/
    ├── cad_models/
    └── environment_configurations/
        └── config1/
```

> 注意：这些场景文件通常较大，Git 上传时建议用 [Git LFS](https://git-lfs.github.com/) 管理 `.dae`、`.npz` 等二进制文件，或在 `.gitignore` 中排除 `outputs/` 与 GADEN 临时日志。

---

## 4. `outputs/` — 运行输出

```text
outputs/
├── odorsim/                              # 默认单种子 OdorSim 输出
├── odorsim_empty/                        # 空仓库早期输出
├── odorsim_obstacle/                     # 有障碍早期输出
├── odorsim_obstacle_new30/               # 第二批 30 新种子输出
├── odorsim_obstacle_new30_v2/            # 算法改进 v2 输出
├── odorsim_obstacle_new30_v3/            # **第三批 30 新种子输出（当前最佳）**
│   ├── aggregate_evaluation.json         # 聚合评估结果
│   ├── analysis.md                       # 详细分析报告
│   ├── history_seed<N>.npz               # 每种子历史
│   ├── summary_seed<N>.png               # 每种子摘要图
│   └── gaden_logs/                       # GADEN 日志
├── odorsim_obstacle_new30_v3_videos/     # **第三批 5 个种子三视角视频**
├── odorsim_obstacle_30seeds/             # 原始 30 种子输出
├── odorsim_obstacle_5seeds/              # 5 种子演示输出
├── odorsim_obstacle_test/                # 临时测试输出
├── random_seed_runs/                     # 解析高斯烟羽随机种子 benchmark
├── rl/                                   # RL 训练输出
└── ...
```

> 建议：将 `outputs/` 加入 `.gitignore`，避免上传大量视频/历史文件。

---

## 5. `scripts/` — 可执行脚本

### 5.1 解析环境仿真（无 OdorSim，Windows 可直接运行）

| 脚本 | 说明 |
|------|------|
| `run_search.py` | 单种子运行简化高斯烟羽仿真。 |
| `run_seed_benchmark.py` | 多指定种子 benchmark。 |
| `run_random_seed_benchmark.py` | 随机生成 30 种子并批量运行。 |
| `plot_3d.py` | 从历史 NPZ 生成静态 3D 轨迹图。 |
| `inspect_seed.py` | 查看单个种子的运行结果摘要。 |
| `pick_random_seeds_for_video.py` | 从批量结果中随机挑选用于生成视频的种子。 |

### 5.2 OdorSim/GADEN 验证（需在 WSL 中运行）

| 脚本 | 说明 |
|------|------|
| `run_odorsim_verification.py` | 单种子 OdorSim/GADEN 验证，可生成视频。 |
| `run_odorsim_benchmark.sh` | 基础多种子批量验证。 |
| `run_odorsim_obstacle_30seeds.sh` | 原始 30 种子有障碍验证。 |
| `run_odorsim_obstacle_30seeds_v2.sh` | 算法改进 v2 批量验证。 |
| `run_odorsim_obstacle_30seeds_v3.sh` | **算法改进 v3 批量验证（当前最佳）**。 |
| `run_odorsim_obstacle_5seeds.sh` | 5 种子快速演示。 |
| `run_odorsim_obstacle_benchmark.sh` | 有障碍 benchmark。 |
| `generate_odorsim_videos.sh` | 单视角视频生成。 |
| `generate_odorsim_videos_env.sh` | 环境视角视频生成。 |
| `generate_odorsim_videos_triple.sh` | 第二批三视角视频生成。 |
| `generate_odorsim_videos_triple_v2.sh` | v2 三视角视频生成。 |
| `generate_odorsim_videos_triple_v3.sh` | **v3 三视角视频生成**。 |
| `evaluate_odorsim_runs.py` | 对指定输出目录聚合评估成功率/误差。 |
| `analyze_odorsim_runs.py` | 生成详细分析报告 `analysis.md`。 |

### 5.3 强化学习训练

| 脚本 | 说明 |
|------|------|
| `train_ppo.py` | PPO 训练入口。 |

### 5.4 辅助分析

| 脚本 | 说明 |
|------|------|
| `analyze_odorsim_runs.py` | OdorSim 批量结果分析。 |
| `analyze_random_seed_results.py` | 解析环境随机种子结果分析。 |

---

## 6. `src/` — 核心源代码

| 文件 | 说明 |
|------|------|
| `__init__.py` | 包初始化。 |
| `environment.py` | `WarehouseEnv`：解析高斯烟羽环境、边界、障碍物、传感器模型。 |
| `robot.py` | `MobileArmRobot`：UMR 底盘 + RM65 机械臂正运动学、关节控制、碰撞检测。 |
| `sensors.py` | 气味传感器模型（MOS 响应、电压转换、噪声）。 |
| `vision.py` | 末端相机模型（可见性判断、深度测距）。 |
| `search_algorithm.py` | `OdorSearchAgent`：**核心搜索算法**（状态机、梯度追踪、牛耕法覆盖、A* 绕障、结束条件）。 |
| `simulation.py` | 解析环境仿真会话。 |
| `odor_sim_adapter.py` | `GadenAwareSearchAgent`：继承 `OdorSearchAgent`，批量查询 GADEN 浓度；GADEN 环境包装器。 |
| `simulation_odorsim.py` | OdorSim/GADEN 版仿真会话。 |
| `evaluation.py` | 独立评估器（不依赖算法自身结束条件，按最终距离评估）。 |
| `visualization.py` | matplotlib 可视化（3D 仓库、轨迹、三视角视频）。 |
| `utils.py` | 通用工具函数。 |

### 6.1 `src/rl/` — 强化学习

| 文件 | 说明 |
|------|------|
| `__init__.py` | 包初始化。 |
| `env_wrapper.py` | `gymnasium` 环境包装器。 |
| `reward_shaper.py` | 奖励塑造函数。 |

---

## 7. 根目录辅助文件

| 文件 | 说明 |
|------|------|
| `.gitignore` | Git 忽略规则（应包含 `outputs/`、`__pycache__/`、`.npz`、`.mp4` 等）。 |
| `requirements.txt` | Python 依赖（numpy、matplotlib、pyyaml 等）。 |
| `run_demo.bat` | Windows 下运行一次简化仿真演示。 |
| `run_odorsim_from_windows.bat` / `.ps1` | 从 Windows 侧启动 WSL OdorSim 验证。 |
| `run_odorsim_obstacle_5seeds.bat` | Windows 下运行 5 种子有障碍验证。 |
| `机器人参数.txt` | 原始中文机器人参数（供参考）。 |
| `基础算法设计.docx` | 原始中文算法设计文档（供参考）。 |

---

## 8. GitHub 上传建议

1. **保留**：`config/`、`docs/`、`scripts/`、`src/`、`requirements.txt`、`.gitignore`、根目录 `.bat/.ps1`。
2. **忽略**：`outputs/`、`__pycache__/`、`.pyc`、`.npz`、`.mp4`、GADEN 日志等大文件。
3. **可选保留/忽略**：`odorsim_scenarios/` 中的 CAD 模型较大，建议用 Git LFS；若不上传，需在 README 中说明场景需单独下载。
4. **编码注意**：根目录下的 `机器人参数.txt` 和 `基础算法设计.docx` 为中文名，Windows 默认编码可能导致跨平台显示问题，建议保留但注意 UTF-8 设置。

---

## 9. 核心数据流

```text
config/  ──┬──> src/search_algorithm.py  ──┬──> src/simulation.py          ──> outputs/
           │                                │
           └──> src/robot.py  ──────────────┘   (解析高斯烟羽)
           │
           └──> src/odor_sim_adapter.py ──> src/simulation_odorsim.py  ──> outputs/
                                              (GADEN 真实烟羽)
```

---

## 10. 快速定位

| 我想做… | 看/运行… |
|------|----------|
| 了解算法 | `docs/ALGORITHM.md`、`src/search_algorithm.py` |
| 运行一次验证 | `docs/USAGE.md` 第 3 节、`scripts/run_odorsim_verification.py` |
| 跑 30 种子批量测试 | `scripts/run_odorsim_obstacle_30seeds_v3.sh` |
| 生成三视角视频 | `scripts/generate_odorsim_videos_triple_v3.sh` |
| 看最新结果 | `outputs/odorsim_obstacle_new30_v3/` |
| 分析失败案例 | `docs/FAILURE_ANALYSIS_AND_PLAN.md` |
| 配置机器人和环境 | `config/robot.yaml`、`config/warehouse_obstacle_8x8.yaml` |
