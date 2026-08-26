[README.md](https://github.com/user-attachments/files/31459811/README.md)
# OdorSearch_MobileArm

基于 **AgileX UMR 移动小车 + RealMan RM65-6F-V 机械臂** 的三维仓库气味源搜索算法与仿真平台。

## 项目背景与目标

本项目为“机械臂+移动小车”组合机器人设计了一套三维环境下的气味源搜索算法，并在自主构建的仓库仿真环境中验证效果。

核心目标：
1. 移动小车搭载 4 个固定气味传感器在仓库内探索；
2. 机械臂末端集成第 5 个气味传感器与视觉相机，可在三维空间中指向气味源；
3. 机器人在不碰撞货架、管道等障碍物的前提下，最终定位并指向气味泄漏源。

## 项目结构

```text
OdorSearch_MobileArm/
├── config/
│   ├── robot.yaml          # UMR + RM65-6F-V 机器人参数
│   └── warehouse.yaml      # 三维仓库环境与气味源配置
├── src/
│   ├── __init__.py
│   ├── robot.py            # 机器人运动学与状态管理
│   ├── sensors.py          # 气味传感器阵列模型
│   ├── vision.py           # 末端视觉相机模型
│   ├── environment.py      # 仓库环境、障碍物与气味扩散
│   ├── search_algorithm.py # 气味源搜索算法
│   ├── simulation.py       # 仿真会话（参考 OdorSim 结构）
│   ├── visualization.py    # 三维与二维可视化
│   └── utils.py            # 配置读取、坐标变换、碰撞检测
├── scripts/
│   └── run_search.py       # 仿真运行入口
├── outputs/                # 仿真输出（历史轨迹、摘要图）
└── docs/
    ├── README.md           # 本文件
    ├── ALGORITHM.md        # 算法详细说明
    └── SETUP.md            # 安装、运行与 OdorSim 对接
```

## 随机环境生成

在 `config/warehouse.yaml` 中设置 `environment.randomize: true`，每次运行（不同 `seed`）都会：

- 随机生成气味源位置（高度限制在机械臂可达范围内）；
- 随机生成 8 个 box 障碍物 + 4 个 cylinder 管道；
- 随机生成环境风场。

这允许在大量不同布局下测试算法的鲁棒性。

## 批量基准测试

运行 10 个随机种子并保存每轮结果：

```bash
python scripts/run_seed_benchmark.py --seeds 1 2 3 4 5 6 7 8 9 10 --max-steps 3000
```

结果保存在 `outputs/seed_runs/`，包含每轮的轨迹图、三维图、历史数据与汇总 JSON。

测试结果显示：在 100% 零碰撞、严格遵守关节限制的前提下，当前算法在 10 个随机环境中的成功率约为 **40%**。失败原因主要是随机游走未能覆盖到远距离/上风向气味源的烟羽。详细分析与改进方向见 [docs/BENCHMARK.md](BENCHMARK.md)。

## 快速开始

### 环境要求

- Python >= 3.9
- NumPy
- Matplotlib
- PyYAML

当前代码使用纯 Python + NumPy/Matplotlib 实现三维可视化，无需安装 PyBullet/MuJoCo 即可运行算法仿真。

### 运行一次仿真

在项目根目录执行：

```bash
python scripts/run_search.py --max-steps 2000 --seed 42
```

常用参数：

```bash
python scripts/run_search.py \
  --max-steps 3000 \
  --seed 42 \
  --visualize \
  --save-history outputs/history.npz \
  --save-plot outputs/summary.png \
  --start-x -7.0 \
  --start-y -6.5
```

### 查看结果

运行结束后会在终端打印摘要：

```text
Success: True
Total steps: 483
Final state: FINISHED
Final distance to source: 0.446 m
Max EE ppm: 1198.02
State counts: {'RANDOM_WALK': 171, 'BASE_TRACKING': 14, 'ARM_TRACKING': 297, 'FINISHED': 1}
Collision count: 22
```

同时生成 `outputs/summary.png` 轨迹与浓度曲线图。

## 关键设计

### 硬件模型

- **移动底盘 AgileX UMR**：尺寸 0.83×0.54×0.41 m，全向移动，四角各安装一个固定气味传感器。
- **机械臂 RealMan RM65-6F-V**：6-DOF，工作半径约 0.6385 m，末端安装气味传感器与视觉相机。
- 机器人 MDH 参数、关节限位、最大速度均来自官方公开规格。

### 算法流程

1. **随机游走（Random Walk）**：未检测到气味时，小车在仓库内自主探索，机械臂按预设姿态分层抬升，覆盖不同高度。
2. **基座追踪（Base Tracking）**：当四角传感器检测到气味时，小车向浓度高的方向小步移动，同时机械臂伸向该角。
3. **末端追踪（Arm Tracking）**：当机械臂末端传感器检测到气味时，小车停止大幅移动，通过局部浓度梯度估计控制机械臂指向气味源。
4. **视觉确认（Visual Confirmation）**：末端相机看到气味源实体且浓度足够高时，提前确认成功。
5. **终止（Finished）**：满足几何指向判据或浓度饱和/视觉确认判据时结束。

详见 [docs/ALGORITHM.md](ALGORITHM.md)。

### 仿真结构（参考 OdorSim）

参考 `OdorSim-master` 的架构，本项目同样采用“会话（Session）”作为顶层接口：

```python
from src.simulation import OdorSearchSession

session = OdorSearchSession(seed=42)
obs = session.reset()
summary = session.run(max_steps=3000)
print(summary)
```

与 OdorSim 的对应关系：

| OdorSim 组件 | 本项目组件 | 说明 |
|--------------|------------|------|
| `OdorCosimSession` | `OdorSearchSession` | 顶层会话 |
| `OdorManipulationEnv` | `WarehouseEnv` | 环境/物理/气味场 |
| `MOX/PID e-nose` | `OdorSensorArray` | 气味传感器模型 |
| `SceneBuilder` | `WarehouseEnv.obstacles` | 场景构建 |
| `GadenBridge` | 简化浓度场 | 本项目使用解析高斯烟羽 |
| 末端相机 | `VisualSensor` | 视觉确认 |

## 与 OdorSim 平台对接

本项目提供了可在 Windows/无 ROS 环境下独立运行的算法与仿真；当部署到 Ubuntu + ROS 2 + GADEN 的 OdorSim 平台时，只需替换 `environment.py` 中的气味浓度查询为真实的 `/odor_value` 服务调用，其余搜索逻辑保持不变。

详细对接步骤见 [docs/SETUP.md](SETUP.md)。

## 扩展建议

1. **多气味源**：在 `warehouse.yaml` 中添加多个 `odor_source`，并在搜索算法中维护“已找到源列表”。
2. **动态障碍物**：在 `WarehouseEnv.step()` 中更新障碍物位置。
3. **真实 GADEN 集成**：将 `concentration_at()` 替换为 ROS 服务调用。
4. **强化学习策略**：将当前脚本策略作为专家策略，生成数据集后训练端到端策略。

## 文档索引

| 文档 | 内容 |
|------|------|
| [README.md](README.md) | 项目总览、快速开始、结构说明 |
| [ALGORITHM.md](ALGORITHM.md) | 搜索算法详细说明 |
| [SETUP.md](SETUP.md) | 安装、运行、与 OdorSim 对接 |
| [BENCHMARK.md](BENCHMARK.md) | 随机环境基准测试结果与改进方向 |

## 作者与许可

本项目为定制化研究开发代码，参数与算法可根据真实机器人接口进一步调整。
