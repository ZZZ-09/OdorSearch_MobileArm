"""三维仓库环境与气味扩散模型。

参考 OdorSim 的 `odor_sim.envs.base.OdorManipulationEnv` 与 `odor_sim.config.objects`，
本模块提供：
- 仓库边界与障碍物（box / cylinder）管理
- 简化高斯烟羽 + 湍流的三维气味浓度场
- 支持随机生成障碍物与气味源位置
- 支持多位置同时查询的传感器接口
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.utils import config_dir, load_yaml


class WarehouseEnv:
    """3D 仓库环境。

    Args:
        config: 环境配置字典；若为 None 则自动加载 config/warehouse.yaml。
        seed: 随机种子；当 ``randomize=true`` 时，不同 seed 生成不同布局。
    """

    def __init__(self, config: "dict[str, Any] | None" = None, seed: int = 0):
        if config is None:
            config = load_yaml(config_dir() / "warehouse.yaml")
        self.cfg = config
        self.rng = np.random.default_rng(seed)

        bounds = config["environment"]["bounds"]
        self.bounds_low = np.array(
            [bounds["x"][0], bounds["y"][0], bounds["z"][0]], dtype=float
        )
        self.bounds_high = np.array(
            [bounds["x"][1], bounds["y"][1], bounds["z"][1]], dtype=float
        )
        self.dt = float(config["environment"]["dt"])
        self.max_steps = int(config["environment"]["max_steps"])
        self.randomize = bool(config["environment"].get("randomize", False))

        # 随机生成参数
        self.rand_cfg = config.get("random_generation", {})

        # 静态障碍物（如外墙）始终保留
        self.static_obstacles: list[dict[str, Any]] = []
        for obs in config.get("obstacles", []):
            if obs["name"].endswith("_wall"):
                self.static_obstacles.append(self._parse_obstacle(obs))

        # 动态障碍物（可随机生成）
        self.dynamic_obstacle_templates = [
            self._parse_obstacle(obs)
            for obs in config.get("obstacles", [])
            if not obs["name"].endswith("_wall")
        ]

        # 气味源参数（位置可能随机）
        src = config["odor_source"]
        self.source_pos = np.array(src["position"], dtype=float)
        self.source_radius = float(src.get("radius", 0.08))
        self.gas_type = src["gas"]
        self.source_strength = float(src["strength"])
        self.wind = np.array(src["wind"], dtype=float)
        self.wind_speed = float(np.linalg.norm(self.wind))

        # 烟羽模型参数
        plume = config["plume"]
        self.background = float(plume["background"])
        self.detection_threshold = float(plume["detection_threshold"])
        self.diffusion_coefficient = float(plume["diffusion_coefficient"])
        self.turbulence = float(plume["turbulence"])
        self.turbulence_update_interval = float(plume["turbulence_update_interval"])

        # 障碍物列表（运行时生成）
        self.obstacles: list[dict[str, Any]] = []

        # 湍流相位（时间相关）
        self._turbulence_phase = self.rng.uniform(0.0, 2.0 * math.pi, size=4)
        self._phase_update_rate = self.rng.uniform(0.5, 1.5, size=4)
        self._elapsed_time = 0.0

        # 执行一次生成
        self.reset(seed=seed)

    # ---------------------------------------------------------------------- #
    # 配置解析
    # ---------------------------------------------------------------------- #
    @staticmethod
    def _parse_obstacle(obs: dict[str, Any]) -> dict[str, Any]:
        return {
            "name": obs["name"],
            "type": obs["type"],
            "center": np.array(obs["center"], dtype=float),
            "size": np.array(obs["size"], dtype=float) if obs["type"] == "box" else None,
            "axis": np.array(obs["axis"], dtype=float)
            if obs["type"] == "cylinder"
            else None,
            "radius": float(obs["radius"]) if obs["type"] == "cylinder" else None,
            "length": float(obs["length"]) if obs["type"] == "cylinder" else None,
        }

    # ---------------------------------------------------------------------- #
    # 随机生成
    # ---------------------------------------------------------------------- #
    def reset(self, seed: "int | None" = None) -> None:
        """重置环境状态，并在 randomize=true 时重新生成源与障碍物。"""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._turbulence_phase = self.rng.uniform(0.0, 2.0 * math.pi, size=4)
        self._phase_update_rate = self.rng.uniform(0.5, 1.5, size=4)
        self._elapsed_time = 0.0

        if self.randomize:
            self._generate_random_layout()

    def _generate_random_layout(self) -> None:
        """随机生成气味源位置和障碍物布局。"""
        # 1. 随机气味源位置
        self.source_pos = self._random_source_position()

        # 2. 随机风场方向（保持较小风速）
        angle = self.rng.uniform(0.0, 2.0 * math.pi)
        speed = self.rng.uniform(0.3, 0.8)
        self.wind = np.array([speed * math.cos(angle), speed * math.sin(angle), 0.0])
        self.wind_speed = float(speed)

        # 3. 生成动态障碍物
        self.obstacles = list(self.static_obstacles)
        self._generate_random_obstacles()

    def _random_source_position(self) -> np.ndarray:
        """生成不与静态障碍物碰撞的气味源位置。"""
        src_cfg = self.rand_cfg.get("source", {})
        x_range = src_cfg.get("x_range", [self.bounds_low[0] + 1.0, self.bounds_high[0] - 1.0])
        y_range = src_cfg.get("y_range", [self.bounds_low[1] + 1.0, self.bounds_high[1] - 1.0])
        z_range = src_cfg.get("z_range", [0.6, 0.9])

        min_gap = float(self.rand_cfg.get("min_source_to_obstacle", 1.5))

        for _ in range(200):
            pos = np.array(
                [
                    self.rng.uniform(x_range[0], x_range[1]),
                    self.rng.uniform(y_range[0], y_range[1]),
                    self.rng.uniform(z_range[0], z_range[1]),
                ],
                dtype=float,
            )
            if self._position_clear_of_obstacles(pos, min_gap):
                return pos

        # 回退：使用配置中的固定位置
        return np.array(self.cfg["odor_source"]["position"], dtype=float)

    def _generate_random_obstacles(self) -> None:
        """随机生成 box/cylinder 障碍物，并保证与源、初始位置有足够间距。"""
        num_boxes = int(self.rand_cfg.get("num_boxes", 6))
        num_cylinders = int(self.rand_cfg.get("num_cylinders", 3))
        min_gap = float(self.rand_cfg.get("min_obstacle_gap", 1.2))

        # 默认起始位置附近需要空旷
        default_start = np.array(
            self.rand_cfg.get("default_start", [-7.0, -6.5, 0.0]), dtype=float
        )
        start_clearance = float(self.rand_cfg.get("min_start_to_obstacle", 1.5))

        def add_if_valid(obs: dict[str, Any]) -> bool:
            # 与已有动态障碍物间距（外墙不参与间距检查）
            for existing in self.obstacles:
                if existing["name"].endswith("_wall"):
                    continue
                if self._obstacle_distance(obs, existing) < min_gap:
                    return False
            # 与源间距
            if self._obstacle_point_distance(obs, self.source_pos) < self.rand_cfg.get(
                "min_source_to_obstacle", 1.5
            ):
                return False
            # 与默认起始位置间距
            if self._obstacle_point_distance(obs, default_start) < start_clearance:
                return False
            self.obstacles.append(obs)
            return True

        # 生成 box
        box_cfg = self.rand_cfg.get("box_size_ranges", {})
        for i in range(num_boxes):
            for _ in range(100):
                size = np.array(
                    [
                        self.rng.uniform(*box_cfg.get("length", [1.0, 2.5])),
                        self.rng.uniform(*box_cfg.get("width", [0.6, 1.5])),
                        self.rng.uniform(*box_cfg.get("height", [1.0, 2.2])),
                    ],
                    dtype=float,
                )
                center = np.array(
                    [
                        self.rng.uniform(self.bounds_low[0] + 1.0, self.bounds_high[0] - 1.0),
                        self.rng.uniform(self.bounds_low[1] + 1.0, self.bounds_high[1] - 1.0),
                        size[2] / 2.0,
                    ],
                    dtype=float,
                )
                obs = {"name": f"random_box_{i}", "type": "box", "center": center, "size": size}
                if add_if_valid(obs):
                    break

        # 生成 cylinder
        cyl_cfg = self.rand_cfg.get("cylinder", {})
        for i in range(num_cylinders):
            for _ in range(100):
                radius = self.rng.uniform(*cyl_cfg.get("radius_range", [0.12, 0.30]))
                length = self.rng.uniform(*cyl_cfg.get("length_range", [2.0, 6.0]))
                axis = self.rng.choice(
                    [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])]
                )
                center = np.array(
                    [
                        self.rng.uniform(self.bounds_low[0] + 1.0, self.bounds_high[0] - 1.0),
                        self.rng.uniform(self.bounds_low[1] + 1.0, self.bounds_high[1] - 1.0),
                        self.rng.uniform(length / 2.0 + 0.5, self.bounds_high[2] - 0.5),
                    ],
                    dtype=float,
                )
                obs = {
                    "name": f"random_cyl_{i}",
                    "type": "cylinder",
                    "center": center,
                    "axis": axis,
                    "radius": radius,
                    "length": length,
                }
                if add_if_valid(obs):
                    break

    def _position_clear_of_obstacles(self, point: np.ndarray, clearance: float) -> bool:
        """判断某点与所有动态障碍物保持 clearance 距离（外墙除外）。"""
        for obs in self.obstacles:
            if obs["name"].endswith("_wall"):
                continue
            if self._obstacle_point_distance(obs, point) < clearance:
                return False
        return True

    @staticmethod
    def _obstacle_distance(a: dict[str, Any], b: dict[str, Any]) -> float:
        """两个障碍物中心之间的距离减去各自的半尺寸/半径近似。"""
        center_dist = float(np.linalg.norm(a["center"] - b["center"]))
        # 粗略估计：取各自包围球半径
        if a["type"] == "box":
            ra = float(np.linalg.norm(a["size"]) / 2.0)
        else:
            ra = float(max(a["radius"], a["length"] / 2.0))
        if b["type"] == "box":
            rb = float(np.linalg.norm(b["size"]) / 2.0)
        else:
            rb = float(max(b["radius"], b["length"] / 2.0))
        return center_dist - ra - rb

    @staticmethod
    def _obstacle_point_distance(obs: dict[str, Any], point: np.ndarray) -> float:
        """障碍物到某点的近似距离。"""
        d = float(np.linalg.norm(obs["center"] - np.asarray(point)))
        if obs["type"] == "box":
            r = float(np.linalg.norm(obs["size"]) / 2.0)
        else:
            r = float(max(obs["radius"], obs["length"] / 2.0))
        return max(0.0, d - r)

    # ---------------------------------------------------------------------- #
    # 环境查询
    # ---------------------------------------------------------------------- #
    def inside_bounds(self, point: np.ndarray) -> bool:
        """判断点是否在仓库有效范围内。"""
        p = np.asarray(point, dtype=float)
        return bool(np.all(p >= self.bounds_low) and np.all(p <= self.bounds_high))

    def get_obstacles(self) -> list[dict[str, Any]]:
        return self.obstacles

    # ---------------------------------------------------------------------- #
    # 气味浓度场
    # ---------------------------------------------------------------------- #
    def concentration_at(self, point: np.ndarray) -> float:
        """计算某点的气味浓度（ppm）。

        模型：基于下风向拉伸的高斯烟羽 + 湍流扰动 + 背景浓度。
        """
        p = np.asarray(point, dtype=float)
        if not self.inside_bounds(p):
            return self.background

        wind_dir = self.wind / (self.wind_speed + 1e-9)
        v = p - self.source_pos

        downwind = float(np.dot(v, wind_dir))
        cross = v - downwind * wind_dir
        cross_dist = float(np.linalg.norm(cross))

        sigma_0 = 1.2
        k = 0.60
        sigma = sigma_0 + k * max(downwind, 0.0)

        gaussian = (
            self.source_strength
            / (sigma ** 2 + 1e-6)
            * math.exp(-(cross_dist ** 2) / (2.0 * sigma ** 2 + 1e-9))
        )

        upwind_decay = 1.5
        if downwind < 0.0:
            gaussian *= math.exp(-abs(downwind) / upwind_decay)

        z_diff = p[2] - self.source_pos[2]
        gaussian *= math.exp(-(z_diff ** 2) / (2.0 * sigma ** 2 + 1e-9))

        turb = 1.0
        if self.turbulence > 0.0:
            for i in range(4):
                phase = self._turbulence_phase[i] + self._phase_update_rate[i] * self._elapsed_time
                turb += self.turbulence * 0.25 * math.sin(
                    phase + 0.3 * i * (p[0] + p[1] + p[2])
                )
            turb = max(0.3, turb)

        conc = self.background + gaussian * turb
        return max(0.0, conc)

    def query_sensors(self, sensor_positions: dict[str, np.ndarray]) -> dict[str, float]:
        """批量查询多个传感器位置的气味浓度。"""
        return {name: self.concentration_at(pos) for name, pos in sensor_positions.items()}

    def step(self) -> None:
        """推进环境时间（湍流演化）。"""
        self._elapsed_time += self.dt

    # ---------------------------------------------------------------------- #
    # 障碍物碰撞
    # ---------------------------------------------------------------------- #
    def check_collision(
        self,
        base_boxes: list[tuple[np.ndarray, np.ndarray]],
        arm_points: list[np.ndarray],
    ) -> bool:
        """检测机器人体积是否与任何障碍物发生碰撞。"""
        from src.utils import (
            box_box_collision,
            point_in_box,
            point_in_cylinder,
        )

        for obs in self.obstacles:
            if obs["type"] == "box":
                for center, size in base_boxes:
                    if box_box_collision(center, size, obs["center"], obs["size"]):
                        return True
                for p in arm_points:
                    if point_in_box(p, obs["center"], obs["size"]):
                        return True

            elif obs["type"] == "cylinder":
                axis = obs["axis"] / (np.linalg.norm(obs["axis"]) + 1e-12)
                for center, size in base_boxes:
                    for dx in [-1, 1]:
                        for dy in [-1, 1]:
                            for dz in [-1, 1]:
                                pt = center + 0.5 * size * np.array([dx, dy, dz])
                                if point_in_cylinder(
                                    pt, obs["center"], axis, obs["radius"], obs["length"]
                                ):
                                    return True
                for p in arm_points:
                    if point_in_cylinder(
                        p, obs["center"], axis, obs["radius"], obs["length"]
                    ):
                        return True
        return False

    def source_direction_hint(self, point: np.ndarray) -> np.ndarray:
        """返回从某点指向气味源的单位向量（考虑风场做简单修正）。"""
        p = np.asarray(point, dtype=float)
        to_source = self.source_pos - p
        dist = np.linalg.norm(to_source)
        if dist < 1e-6:
            return np.zeros(3)
        to_source = to_source / dist

        wind_norm = self.wind / (self.wind_speed + 1e-9)
        hint = 0.8 * to_source - 0.2 * wind_norm
        norm = np.linalg.norm(hint)
        return hint / norm if norm > 1e-6 else to_source
