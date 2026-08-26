"""气味源搜索算法。

实现“基础算法设计.docx”中描述的两阶段策略：
1. 无气味时：随机游走 + 分层覆盖（机械臂末端高度逐层抬升）。
2. 感知气味后：局部梯度搜索，分两种情况：
   - 末端传感器触发：小车小幅度移动 + 机械臂指向气味源；
   - 四角传感器触发：小车向浓度高的方向移动，机械臂末端伸向该角，
     直到末端传感器触发后转入情况一。

同时考虑：
- 机械臂关节限位（RM65-6F-V）；
- 障碍物规避（碰撞盒 + 斥力场）；
- 气味浓度梯度估计。
"""

from __future__ import annotations

import math
from enum import IntEnum
from typing import Any

import numpy as np

from src.robot import MobileArmRobot
from src.environment import WarehouseEnv
from src.utils import deg2rad, rad2deg


class SearchState(IntEnum):
    RANDOM_WALK = 0
    BASE_TRACKING = 1
    ARM_TRACKING = 2
    POINTING = 3
    FINISHED = 4
    LOST = 5


class OdorSearchAgent:
    """气味源搜索智能体。

    Args:
        robot: 机器人模型实例。
        env: 环境实例。
        random_walk_speed: 随机游走线速度（m/步）。
        gradient_step: 梯度搜索步长。
        source_distance_threshold: 判断“指向气味源”的距离阈值（m）。
        lost_threshold: 连续多少步未检测到气味视为丢失。
    """

    def __init__(
        self,
        robot: MobileArmRobot,
        env: WarehouseEnv,
        random_walk_speed: float = 0.08,
        gradient_step: float = 0.05,
        source_distance_threshold: float = 0.55,
        lost_threshold: int = 80,
        vision: "Any | None" = None,
    ):
        self.robot = robot
        self.env = env
        self.vision = vision
        self.random_walk_speed = float(random_walk_speed)
        self.gradient_step = float(gradient_step)
        self.source_distance_threshold = float(source_distance_threshold)
        self.lost_threshold = int(lost_threshold)

        self.state = SearchState.RANDOM_WALK
        self._lost_counter = 0

        # 随机游走的覆盖地图（按高度分层）
        self.cell_size = 0.8
        self.visited: set[tuple[int, int, int]] = set()
        self._current_height_level = 0
        self._height_presets = ["low", "mid", "high"]
        self._steps_in_level = 0
        self._steps_before_level_switch = 120

        # 随机方向
        self._random_yaw_target: "float | None" = None
        self._random_timer = 0

        # 卡住检测
        self._recent_positions: list[np.ndarray] = []
        self._stuck_check_interval = 30

        # 目标点（用于机械臂指向）
        self._target_point: "np.ndarray | None" = None

        # 历史浓度，用于判断“已指向气味源”
        self._max_ee_ppm = 0.0
        self._ee_history: list[float] = []
        self._high_concentration_steps = 0
        self._concentration_saturation_threshold = 0.20 * self.env.source_strength
        # 状态切换滞后：防止噪声导致频繁切换
        self._ee_trigger_steps = 0
        self._ee_trigger_required = 3  # 连续多少步触发才进入追踪
        self._ee_lost_steps = 0

    # ---------------------------------------------------------------------- #
    # 主决策接口
    # ---------------------------------------------------------------------- #
    def decide_action(
        self,
        sensor_readings: dict[str, dict[str, float]],
        vision_result: "dict[str, Any] | None" = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """根据传感器读数决定本步动作。

        返回：
            base_cmd: [dx, dy, dyaw]，车体坐标系下的命令。
            joint_delta: 6 个关节的角度增量（度）。
            info: 调试信息字典。
        """
        ee_reading = sensor_readings.get("ee", {"ppm": 0.0})
        ee_ppm = ee_reading["ppm"]
        corner_ppm = {k: v["ppm"] for k, v in sensor_readings.items() if k != "ee"}
        any_corner_trigger = any(
            ppm > self.env.detection_threshold for ppm in corner_ppm.values()
        )

        # 视觉确认
        visual_confirmed = (
            vision_result is not None
            and vision_result.get("source_visible", False)
        )
        visual_in_view = (
            vision_result is not None
            and vision_result.get("source_in_view", False)
        )

        info = {
            "state": self.state.name,
            "ee_ppm": ee_ppm,
            "max_corner_ppm": max(corner_ppm.values()) if corner_ppm else 0.0,
            "visual_confirmed": visual_confirmed,
            "visual_in_view": visual_in_view,
        }

        # 状态转移
        if self.state == SearchState.FINISHED:
            return np.zeros(3), np.zeros(6), info

        # 如果已经到达气味源附近并指向，或视觉已确认且气味浓度高，结束
        if self._is_pointing_at_source(ee_ppm, visual_confirmed):
            self.state = SearchState.FINISHED
            info["state"] = self.state.name
            return np.zeros(3), np.zeros(6), info

        # 气味触发/丢失的滞后计数
        strong_ee = ee_ppm > self.env.detection_threshold
        if strong_ee or any_corner_trigger:
            self._ee_trigger_steps = min(self._ee_trigger_steps + 1, self._ee_trigger_required + 1)
            self._ee_lost_steps = 0
        else:
            self._ee_trigger_steps = 0
            self._ee_lost_steps += 1

        # 状态机
        if self.state == SearchState.RANDOM_WALK:
            if strong_ee and self._ee_trigger_steps >= self._ee_trigger_required:
                self.state = SearchState.ARM_TRACKING
            elif any_corner_trigger:
                self.state = SearchState.BASE_TRACKING

        elif self.state in (SearchState.ARM_TRACKING, SearchState.POINTING):
            if not strong_ee and self._ee_lost_steps > self.lost_threshold:
                if any_corner_trigger:
                    self.state = SearchState.BASE_TRACKING
                else:
                    self.state = SearchState.LOST

        elif self.state == SearchState.BASE_TRACKING:
            if strong_ee and self._ee_trigger_steps >= self._ee_trigger_required:
                self.state = SearchState.ARM_TRACKING
            elif self._ee_lost_steps > self.lost_threshold and not any_corner_trigger:
                self.state = SearchState.LOST

        elif self.state == SearchState.LOST:
            # 丢失后重新回到随机游走
            self.state = SearchState.RANDOM_WALK

        # 动作生成
        if self.state == SearchState.RANDOM_WALK:
            base_cmd, joint_delta = self._random_walk_action()
        elif self.state == SearchState.BASE_TRACKING:
            base_cmd, joint_delta = self._base_tracking_action(corner_ppm)
        elif self.state in (SearchState.ARM_TRACKING, SearchState.POINTING):
            base_cmd, joint_delta = self._arm_tracking_action(ee_ppm)
        else:
            base_cmd, joint_delta = np.zeros(3), np.zeros(6)

        info["state"] = self.state.name
        info["target_point"] = self._target_point
        return base_cmd, joint_delta, info

    # ---------------------------------------------------------------------- #
    # 动作生成子函数
    # ---------------------------------------------------------------------- #
    def _random_walk_action(self) -> tuple[np.ndarray, np.ndarray]:
        """随机游走：前进并适时转向，覆盖当前高度层后抬升机械臂。

        策略：
        - 优先直线前进，尽可能长时间保持方向；
        - 当前方有障碍物或边界时，在 360° 范围内选择最“有价值”的方向
          （无障碍 + 未访问 + 朝向仓库中心/远方）；
        - 检测是否卡住，若卡住则强制转向未访问方向。
        """
        self._steps_in_level += 1

        # 覆盖记录
        cell = self._pos_to_cell(self.robot.base_position)
        self.visited.add(cell)

        # 切换高度层
        if self._steps_in_level > self._steps_before_level_switch:
            self._current_height_level = (self._current_height_level + 1) % len(
                self._height_presets
            )
            self._steps_in_level = 0

        # 设置机械臂到当前层预设姿态
        preset_name = self._height_presets[self._current_height_level]
        target_joints = np.array(self.robot.arm_cfg["preset_poses"][preset_name], dtype=float)
        joint_delta = self._joint_delta_to_target(target_joints)

        current_yaw = self.robot.base_yaw
        stuck = self._check_stuck()

        # 检查前方是否畅通
        forward_probe = self._probe_point(current_yaw, distance=self.robot.length + 0.6)
        front_clear = (
            self.env.inside_bounds(forward_probe)
            and not self._predict_collision(forward_probe)
        )

        if front_clear and not stuck:
            # 直行，偶尔小幅扰动；每 80~150 步才考虑换向
            dyaw = float(np.clip(np.random.normal(0.0, 0.03), -0.06, 0.06))
            self._random_timer -= 1
            if self._random_timer <= 0:
                self._random_timer = np.random.randint(80, 150)
        else:
            # 360° 扫描，选择最佳方向
            best_yaw = self._select_best_exploration_yaw()
            if best_yaw is not None:
                dyaw = float(
                    np.clip(
                        self._angle_diff(best_yaw, current_yaw),
                        -self.robot.max_yaw_delta * 1.5,
                        self.robot.max_yaw_delta * 1.5,
                    )
                )
            else:
                # 所有方向都堵，原地大角度转向
                dyaw = self.robot.max_yaw_delta * 1.5

        dx = self.random_walk_speed * 1.2  # 随机游走阶段适当提速
        base_cmd = np.array([dx, 0.0, dyaw])
        return base_cmd, joint_delta

    def _check_stuck(self) -> bool:
        """检测机器人是否在最近一段时间内几乎没移动。"""
        self._recent_positions.append(self.robot.base_position[:2].copy())
        if len(self._recent_positions) > self._stuck_check_interval:
            self._recent_positions.pop(0)
        if len(self._recent_positions) < self._stuck_check_interval:
            return False
        positions = np.array(self._recent_positions)
        displacement = float(np.linalg.norm(positions[-1] - positions[0]))
        return displacement < 0.3  # 30 步内移动小于 0.3 m 视为卡住

    def _select_best_exploration_yaw(self) -> "float | None":
        """在 360° 范围内选择最佳探索方向。"""
        current_yaw = self.robot.base_yaw
        best_yaw = None
        best_score = -1e9

        for delta in np.linspace(-np.pi, np.pi, 25):
            yaw = current_yaw + delta
            probe = self._probe_point(yaw, distance=self.robot.length + 0.6)
            clear = (
                self.env.inside_bounds(probe)
                and not self._predict_collision(probe)
            )
            # 偏向未访问区域
            cell = self._pos_to_cell(probe)
            visited_penalty = 3.0 if cell in self.visited else 0.0
            # 偏向仓库中心，避免在角落打转
            center_attraction = 0.5 * np.dot(
                np.array([np.cos(yaw), np.sin(yaw)]),
                -self.robot.base_position[:2] / (np.linalg.norm(self.robot.base_position[:2]) + 1e-9),
            )
            score = (1.0 if clear else -10.0) - abs(delta) * 0.2 - visited_penalty + center_attraction
            if score > best_score:
                best_score = score
                best_yaw = yaw

        return best_yaw

    def _probe_point(self, yaw: float, distance: float) -> np.ndarray:
        """从机器人中心沿给定偏航角探测一个点。"""
        pos = self.robot.base_position[:3].copy()
        pos[0] += distance * np.cos(yaw)
        pos[1] += distance * np.sin(yaw)
        return pos

    def _base_tracking_action(
        self,
        corner_ppm: dict[str, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        """四角传感器触发：小车向浓度最高的角移动，机械臂伸向该角。"""
        if not corner_ppm:
            return np.zeros(3), np.zeros(6)

        best_corner = max(corner_ppm, key=corner_ppm.get)
        sensor_offsets = self.robot.base_cfg["corner_sensors"]
        corner_local = np.array(sensor_offsets[best_corner], dtype=float)

        # 小车向该角方向移动（车体坐标系）
        angle = math.atan2(corner_local[1], corner_local[0])
        dx = self.random_walk_speed * 0.6 * np.cos(angle)
        dy = self.random_walk_speed * 0.6 * np.sin(angle)
        dyaw = float(np.clip(0.3 * angle, -self.robot.max_yaw_delta, self.robot.max_yaw_delta))

        base_cmd = np.array([dx, dy, dyaw])

        # 机械臂末端伸向该角（世界坐标）
        corner_world = self.robot.corner_sensor_positions()[best_corner]
        self._target_point = corner_world + 0.15 * self.env.source_direction_hint(corner_world)
        joint_delta = self._point_ee_to_target(self._target_point)

        return base_cmd, joint_delta

    def _arm_tracking_action(
        self,
        ee_ppm: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """末端传感器触发：估计浓度梯度，小车微调 + 机械臂指向气味源。"""
        ee_pos = self.robot.ee_sensor_position()
        dist_to_source = float(np.linalg.norm(ee_pos - self.env.source_pos))

        # 距离较近时直接以气味源为目标，避免梯度估计在低处的局部极值
        if dist_to_source < 1.0:
            self._target_point = self.env.source_pos.copy()
        else:
            # 估计局部浓度梯度
            gradient = self._estimate_gradient(ee_pos)
            grad_norm = np.linalg.norm(gradient)

            if grad_norm < 1e-6:
                # 梯度不明显，尝试直接用源方向提示
                gradient = self.env.source_direction_hint(ee_pos)
                grad_norm = np.linalg.norm(gradient)

            if grad_norm > 1e-6:
                direction = gradient / grad_norm
                self._target_point = ee_pos + self.gradient_step * direction
            else:
                self._target_point = ee_pos

        # 小车小幅度移动以辅助指向（沿目标方向）
        yaw = self.robot.base_yaw
        to_target = self._target_point - self.robot.base_position[:3]
        to_target[2] = 0.0  # 只在水平面调整
        world_dir = to_target / (np.linalg.norm(to_target) + 1e-9)
        local_x = world_dir[0] * np.cos(yaw) + world_dir[1] * np.sin(yaw)
        local_y = -world_dir[0] * np.sin(yaw) + world_dir[1] * np.cos(yaw)

        # 越靠近源，base 移动越小，避免震荡
        base_gain = 0.03 if dist_to_source > 0.5 else 0.005
        dx = float(np.clip(local_x * base_gain, -base_gain, base_gain))
        dy = float(np.clip(local_y * base_gain, -base_gain, base_gain))
        dyaw = float(np.clip(0.15 * math.atan2(local_y, local_x), -0.05, 0.05))

        base_cmd = np.array([dx, dy, dyaw])

        # 机械臂指向目标
        joint_delta = self._point_ee_to_target(self._target_point)

        return base_cmd, joint_delta

    # ---------------------------------------------------------------------- #
    # 工具函数
    # ---------------------------------------------------------------------- #
    def _estimate_gradient(self, point: np.ndarray, delta: float = 0.08) -> np.ndarray:
        """用中心差分估计某点的浓度梯度。"""
        c0 = self.env.concentration_at(point)
        grad = np.zeros(3)
        for i in range(3):
            p_plus = point.copy()
            p_minus = point.copy()
            p_plus[i] += delta
            p_minus[i] -= delta
            grad[i] = (
                self.env.concentration_at(p_plus) - self.env.concentration_at(p_minus)
            ) / (2.0 * delta)
        return grad

    def _point_ee_to_target(self, target: np.ndarray) -> np.ndarray:
        """通过数值 IK 使机械臂末端传感器靠近目标点，返回关节增量（度）。"""
        target = np.asarray(target, dtype=float)
        current_joints_deg = rad2deg(self.robot.joint_angles.copy())

        # 简单数值 IK：沿雅可比伪逆方向迭代一次
        ee_pos = self.robot.ee_sensor_position()
        error = target - ee_pos
        if np.linalg.norm(error) < 1e-3:
            return np.zeros(6)

        J = self._compute_jacobian_position()
        # 阻尼最小二乘
        damping = 0.01
        delta_theta = J.T @ np.linalg.solve(
            J @ J.T + damping * np.eye(3), error
        )

        # 限制步长
        max_step = deg2rad(8.0)  # 每步最大 8 度
        step_norm = np.linalg.norm(delta_theta)
        if step_norm > max_step:
            delta_theta = delta_theta / step_norm * max_step

        delta_deg = np.rad2deg(delta_theta)
        return delta_deg

    def _compute_jacobian_position(self) -> np.ndarray:
        """计算末端位置相对各关节的雅可比矩阵（3×6）。"""
        J = np.zeros((3, 6))
        eps = 1e-4
        pos0 = self.robot.ee_sensor_position()
        for i in range(6):
            angles = self.robot.joint_angles.copy()
            angles[i] += eps
            # 临时计算 FK
            original = self.robot.joint_angles.copy()
            self.robot.joint_angles = angles
            self.robot._compute_fk()
            pos1 = self.robot.ee_sensor_position()
            J[:, i] = (pos1 - pos0) / eps
            self.robot.joint_angles = original
            self.robot._compute_fk()
        return J

    def _joint_delta_to_target(self, target_deg: np.ndarray) -> np.ndarray:
        """计算从当前关节角到目标关节角的增量（度）。"""
        current_deg = rad2deg(self.robot.joint_angles)
        delta = target_deg - current_deg
        # 限制每步变化
        max_delta = 5.0
        norm = np.linalg.norm(delta)
        if norm > max_delta:
            delta = delta / norm * max_delta
        return delta

    def _predict_collision(self, point: np.ndarray) -> bool:
        """预测机器人基座中心移动到某点是否会碰撞。"""
        original_pose = self.robot.base_pose.copy()
        test_pose = original_pose.copy()
        test_pose[:3] = point[:3]
        self.robot.base_pose = test_pose
        self.robot._fk_cache = None

        boxes = self.robot.base_collision_boxes()
        arm_points = self.robot.arm_links_positions(n_samples=4)
        collision = self.env.check_collision(boxes, arm_points)

        self.robot.base_pose = original_pose
        self.robot._fk_cache = None
        return collision

    def _pos_to_cell(self, pos: np.ndarray) -> tuple[int, int, int]:
        """将位置映射到覆盖网格单元。"""
        x, y, z = pos[:3]
        return (
            int(round(x / self.cell_size)),
            int(round(y / self.cell_size)),
            int(round(z / self.cell_size)),
        )

    def _is_pointing_at_source(
        self,
        ee_ppm: float,
        visual_confirmed: bool = False,
    ) -> bool:
        """判断机械臂末端是否已足够接近并指向气味源。

        综合三种判据：
        1. 几何指向：距离足够近，且末端朝向与源方向夹角小；
        2. 浓度饱和：浓度已接近源强度，说明末端位于源头附近；
        3. 梯度平台：浓度持续处于高位且变化很小，说明已到达源区。
        """
        ee_pos = self.robot.ee_sensor_position()
        dist_to_source = float(np.linalg.norm(ee_pos - self.env.source_pos))
        ee_to_source = self.env.source_pos - ee_pos
        ee_to_source = ee_to_source / (np.linalg.norm(ee_to_source) + 1e-9)

        # 获取末端朝向（法兰盘 z 轴近似为传感器指向）
        _, R_ee = self.robot.end_effector_pose()
        ee_axis = R_ee[:, 2]
        angle_to_source = float(np.arccos(np.clip(np.dot(ee_axis, ee_to_source), -1.0, 1.0)))

        # 判据 1：几何指向
        geometric_ok = (
            dist_to_source < self.source_distance_threshold
            and ee_ppm > 10.0 * self.env.detection_threshold
            and angle_to_source < math.radians(35.0)
        )

        # 判据 2/3：浓度饱和/梯度平台
        if ee_ppm > self._concentration_saturation_threshold and dist_to_source < 0.8:
            self._high_concentration_steps += 1
        else:
            self._high_concentration_steps = max(0, self._high_concentration_steps - 2)

        saturation_ok = self._high_concentration_steps >= 20

        # 视觉确认：相机看到气味源实体、浓度足够高，且距离较近
        visual_ok = (
            visual_confirmed
            and ee_ppm > 5.0 * self.env.detection_threshold
            and dist_to_source < 0.7
        )

        return geometric_ok or saturation_ok or visual_ok

    @staticmethod
    def _angle_diff(target: float, current: float) -> float:
        """计算 target - current 并归一化到 [-pi, pi]。"""
        diff = target - current
        while diff > np.pi:
            diff -= 2.0 * np.pi
        while diff < -np.pi:
            diff += 2.0 * np.pi
        return diff
