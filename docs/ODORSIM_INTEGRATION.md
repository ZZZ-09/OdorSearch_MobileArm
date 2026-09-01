# OdorSim/GADEN 集成说明

本文档说明如何在 OdorSim（GADEN 实时气体扩散后端）中验证和训练 `OdorSearch_MobileArm` 的搜索算法，同时尽量不对原有代码做修改。

## 1. 新增文件清单

```text
OdorSearch_MobileArm/
├── config/
│   ├── warehouse_empty_8x8.yaml     # 8×8 m 空仓库配置（用于快速验证）
│   └── warehouse_empty_20x16.yaml   # 20×16 m 空仓库配置（大空间验证）
├── odorsim_scenarios/
│   ├── warehouse_8x8/               # 8×8 m GADEN 场景（STL、wind、config、scene）
│   └── warehouse_20x16/             # 20×16 m GADEN 场景
├── src/
│   ├── odor_sim_adapter.py          # GadenBackedWarehouseEnv + GadenAwareSearchAgent
│   ├── simulation_odorsim.py        # OdorSearchSessionOdorSim（注入自定义 env）
│   ├── search_algorithm.py          # 多模态搜索算法（视觉优先、局部极值逃离）
│   ├── evaluation.py                # 独立评估体系
│   └── rl/
│       ├── __init__.py
│       ├── env_wrapper.py           # Gymnasium Env 包装器
│       └── reward_shaper.py         # RL 奖励函数
├── scripts/
│   ├── run_odorsim_verification.py  # 仿真验证 + 视频
│   ├── run_odorsim_benchmark.sh     # 多种子批量验证
│   ├── evaluate_odorsim_runs.py     # 离线聚合评估
│   ├── generate_odorsim_videos.sh   # 批量生成视频
│   └── train_ppo.py                 # PPO 训练入口
└── docs/
    ├── ODORSIM_INTEGRATION.md       # 本文档
    └── optimization_training_plan.md # 策略优化训练方案
```

原有 `src/*.py`（environment.py、robot.py、search_algorithm.py、simulation.py 等）**未做任何修改**。

## 2. 运行环境

必须在 WSL2 Ubuntu-24.04 中，且已安装 OdorSim 并激活其环境：

```bash
cd /home/odorsim/OdorSim
source setup/activate.sh
```

所有脚本都需要在激活后的 shell 中运行。

## 3. 仿真验证（保存视频）

```bash
cd /mnt/c/Users/lenovo/Desktop/OSL_mechanic_arm/OdorSearch_MobileArm
python scripts/run_odorsim_verification.py --seed 42 --max-steps 2000
```

输出：
- `outputs/odorsim/history_seed<N>.npz` — 搜索历史
- `outputs/odorsim/summary_seed<N>.png` — 轨迹/浓度摘要图
- `outputs/odorsim/video_seed<N>.mp4` — 三维搜索过程视频

多种子批量验证：

```bash
bash scripts/run_odorsim_benchmark.sh 1 2 3 4 5
```

## 4. 当前验证结果（8×8 m 空仓库）

### 4.1 早期固定种子结果（已过时，仅供参考）

在算法较早期版本中，对固定 seeds 1–5、42 的测试曾达到 100% 成功率（阈值 1.0 m）。随着算法约束收紧（不再使用真实源位置、视觉主导结束条件、 OccupancyGrid2D 避障覆盖），该结果已不能代表当前版本。

### 4.2 最新 5 次随机种子 OdorSim 结果

从解析环境 30 次随机种子中按 `picker seed=2026` 抽取 5 个种子，在 8×8 m 空仓库 GADEN 场景中运行并生成视频：

| seed | 解析环境 | OdorSim/GADEN | 步数 | 评估误差 | 关键现象 |
|------|----------|---------------|------|----------|----------|
| 9936  | ✅ 成功 | ❌ 失败 | 2000 | 3.278 m | 多次检测到 75–873 ppm，但未进入追踪/结束 |
| 36611 | ✅ 成功 | ✅ 成功 | 760  | 0.104 m | 视觉确认生效 |
| 72823 | ✅ 成功 | ❌ 失败 | 2000 | 3.493 m | step 400 检测到 5916 ppm，随后回到 RANDOM_WALK |
| 83123 | ✅ 成功 | ❌ 失败 | 2000 | 3.156 m | step 200 检测到 8134 ppm，随后回到 RANDOM_WALK |
| 85200 | ✅ 成功 | ❌ 失败 | 2000 | 4.301 m | step 400 检测到 16011 ppm，随后回到 RANDOM_WALK |

**OdorSim 成功率：1/5 = 20%**（阈值 1.0 m）。

主要问题：
1. **视觉主导结束条件在 GADEN 中过严**：即使末端浓度极高，相机也常因视场/遮挡无法看到源实体；
2. **高浓度后回退 RANDOM_WALK**：ARM_TRACKING 中检测到高浓度后，因 `_arm_tracking_max_steps` 超时或局部极值误判而逃离，未能持续追踪；
3. **GADEN 烟羽湍流导致梯度不稳定**：局部 filament 变化快，梯度方向抖动大，基座跟进困难。

独立评估：
```bash
python scripts/evaluate_odorsim_runs.py --out-dir outputs/odorsim --seeds 9936 36611 72823 83123 85200
```

## 5. 策略优化训练

详见 `docs/optimization_training_plan.md`。 starter 代码已提供：

```bash
# 安装依赖
pip install gymnasium stable-baselines3 tensorboard

# 启动训练（示例 20 万步）
python scripts/train_ppo.py --seed 42 --total-timesteps 200000 --max-episode-steps 1000
```

## 6. 设计要点

- **GADEN-backed 环境**：`GadenBackedWarehouseEnv` 继承原 `WarehouseEnv`，仅重写 `concentration_at` / `query_sensors` / `step`，把气味查询转发到 GADEN 的 `/odor_value` 服务。
- **批量梯度估计**：`GadenAwareSearchAgent` 重写 `_estimate_gradient`，把原本 6 次串行浓度查询合并为 1 次批量查询，避免 ROS 服务超时。
- **坐标对齐**：自定义 GADEN 场景与仓库坐标系原点对齐，无需额外坐标变换。
- **多模态搜索算法**：`OdorSearchAgent` 集成视觉优先、移动探测器向最高浓度固定探测器靠拢、局部极值逃离、分层覆盖地图清除等策略。
- **独立评估体系**：`SourceSearchEvaluator` 在机器人声明找到源后计算声明位置与实际源位置的距离，独立于决策逻辑，用于仿真成功率统计。
- **Gymnasium 接口**：`OdorSearchRLEnv` 封装为 RL 标准接口，可直接接入 PPO/SAC 等算法。
