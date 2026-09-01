# 安装、运行与 OdorSim 对接指南

## 1. 当前代码运行环境

### 1.1 已验证依赖

本项目使用纯 Python 实现，依赖以下包：

- Python >= 3.9
- NumPy
- Matplotlib
- PyYAML

### 1.2 快速检查

在项目根目录执行：

```bash
python -c "import numpy, matplotlib, yaml; print('OK')"
```

若提示缺失，使用 pip 安装：

```bash
python -m pip install numpy matplotlib pyyaml
```

### 1.3 运行仿真

```bash
python scripts/run_search.py --max-steps 2000 --seed 42
```

### 1.4 生成可视化

```bash
python scripts/run_search.py --max-steps 2000 --seed 42 --visualize --save-plot outputs/summary.png
```

> 注意：`--visualize` 会调用 matplotlib 的 3D 交互窗口。在 Windows 上若使用某些后端无法弹出窗口，可改用 `--save-plot` 保存图片后查看。

## 2. 可选：安装更真实的物理仿真器

当前代码为了在无 Ubuntu/ROS 环境下也能运行，使用 Matplotlib 进行三维可视化，几何碰撞检测由纯 Python 实现。若希望获得更真实的物理、渲染效果，可安装以下仿真器之一：

### 2.1 PyBullet（推荐，跨平台）

```bash
python -m pip install pybullet
```

安装成功后，可参考 `src/` 中的机器人模型，将 `MobileArmRobot` 替换为 PyBullet 中的 URDF 加载，保留搜索算法逻辑不变。

### 2.2 MuJoCo + robosuite（OdorSim 使用）

OdorSim 使用 MuJoCo 3.9.0 + robosuite 1.5.2。该组合在 Windows 上安装较复杂，官方更推荐 Ubuntu 24.04 + ROS 2 Jazzy。

```bash
python -m pip install mujoco==3.9.0
python -m pip install robosuite==1.5.2
```

> 安装失败可先跳过，当前算法仿真不依赖 MuJoCo。

### 2.3 OpenCV / imageio（用于视频录制）

```bash
python -m pip install opencv-python imageio
```

## 3. 与 OdorSim 平台对接

本项目参考了 `OdorSim-master` 的仿真结构，便于后续迁移到真实 OdorSim 环境。

### 3.1 当前代码与 OdorSim 的对应关系

| 本项目 | OdorSim | 替换方式 |
|--------|---------|----------|
| `OdorSearchSession` | `OdorCosimSession` | 顶层接口可直接复用 |
| `WarehouseEnv` | `OdorManipulationEnv` | 替换为 OdorSim 的 env |
| `WarehouseEnv.concentration_at()` | GADEN `/odor_value` | 改为 ROS 服务调用 |
| `OdorSensorArray` | `mox_pid.py` | 可直接使用 OdorSim 的 MOX 模型 |
| `VisualSensor` | 末端相机 | 使用 OdorSim/robosuite 的相机观测 |

### 3.2 对接步骤

#### 步骤 1：准备 Ubuntu 24.04 + ROS 2 Jazzy

按照 OdorSim 的 `AGENTS.md` 执行：

```bash
bash setup/install_ros_gaden.sh
bash setup/install_sim_env.sh
source setup/activate.sh
```

#### 步骤 2：创建 OdorSim 任务（可选）

若需要把当前仓库作为 OdorSim 的一个任务：

1. 在 `OdorSim-master/odor_sim/envs/` 中新建 `odor_search_mobilearm.py`；
2. 继承 `OdorManipulationEnv`；
3. 复用本项目 `search_algorithm.py` 中的 `OdorSearchAgent`；
4. 在 `registry.py` 中注册新任务，例如 `"OdorSearchMobileArm"`。

#### 步骤 3：替换气味浓度查询

将 `src/environment.py` 中的 `concentration_at()` 替换为调用 OdorSim 的 GADEN bridge：

```python
# 示例伪代码（需结合 OdorSim 的 GadenBridge 实际接口）
def concentration_at(self, point):
    from odor_sim.bridge.gaden_bridge import GadenBridge
    # 将点坐标转换到 GADEN 坐标系
    ppm = self.gaden_bridge.query_ppm(point)
    return ppm
```

#### 步骤 4：复用搜索算法

`OdorSearchAgent` 只依赖以下输入：

- `sensor_readings`：5 个传感器的浓度/电压；
- `vision_result`：视觉检测结果（可选）。

因此可直接把 `OdorSearchAgent` 集成到 OdorSim 的 `OdorCosimSession.step()` 中，替换原有的策略或 teleop。

#### 步骤 5：扩展场景

OdorSim 的 `scenarios/` 目录下存放 GADEN 环境配置。可参照 `scenarios/10x6_uniform` 创建更大的仓库场景：

```text
scenarios/warehouse_large/
  cad_models/                    # STL 仓库几何
  wind_simulations/<wind>/       # 风场 CSV
  environment_configurations/<config>/
    config.yaml                  # GADEN 环境配置
    scenes/scene1.yaml
    simulations/
```

然后在 `odor_sim.make(..., scenario="warehouse_large")` 中加载。

### 3.3 运行 OdorSim 版搜索

```python
import odor_sim as odorsim
from src.search_algorithm import OdorSearchAgent

with odorsim.make(
    "OdorSearchMobileArm",
    objects=["leak_source"],
    odor_mode="continuous",
) as cosim:
    obs = cosim.reset()
    agent = OdorSearchAgent(cosim.env, cosim.env)
    for _ in range(3000):
        sensor_readings = cosim.env.get_sensor_readings()  # 需自行实现接口
        vision_result = cosim.env.get_vision_result()
        base_cmd, joint_delta, _ = agent.decide_action(sensor_readings, vision_result)
        action = compose_action(base_cmd, joint_delta)
        obs, reward, done, info = cosim.step(action)
        if done:
            break
```

## 4. 常见问题

### Q1：Windows 上无法弹出 3D 窗口？

A：使用 `--save-plot` 保存图片，或用 `Agg` 后端：

```python
import matplotlib
matplotlib.use("Agg")
```

### Q2：机器人初始位置与障碍物碰撞？

A：修改 `config/warehouse.yaml` 中障碍物位置，或调整 `scripts/run_search.py` 的 `--start-x/--start-y`。

### Q3：机械臂够不到气味源？

A：检查源高度是否超过机械臂最大 reach。RM65 基座高 0.41 m，最大垂直 reach 约 1.05 m，因此源高度建议 ≤ 0.9 m。

### Q4：搜索失败或步数过多？

A：
- 提高 `source_strength`；
- 降低 `detection_threshold`；
- 增大仓库中空旷区域；
- 调整风场使烟羽覆盖更广。

## 5. 后续工作建议

1. 在 Ubuntu + ROS 2 + GADEN 上复现 OdorSim 环境；
2. 将本项目的 `OdorSearchAgent` 移植为 OdorSim 的收集策略/teleop；
3. 使用真实 GADEN 烟羽替代解析高斯模型，验证算法在更真实湍流中的鲁棒性；
4. 加入更多视觉模型（如 YOLO/语义分割）识别管道、阀门等潜在泄漏实体。
