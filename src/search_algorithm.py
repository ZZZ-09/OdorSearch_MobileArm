"""气味源搜索算法（无真实源位置先验版）。

本版本遵循“源的位置与浓度场对机器人未知”的约束：
- 机器人只能使用自身传感器读数（四角 + 末端气味传感器）和视觉相机输出；
- 不使用 env.source_pos、env.source_strength 等全局真实信息；
- 结束条件改为视觉测距 + 嗅觉平台双重严格确认；
- 随机游走采用前沿（frontier）方向引导，并在每个机械臂高度层尽量覆盖空间。
"""

from __future__ import annotations

import heapq
import math
from enum import IntEnum
from typing import Any

import numpy as np

from src.robot import MobileArmRobot
from src.environment import WarehouseEnv
from src.utils import deg2rad, point_in_box, point_in_cylinder, rad2deg


class OccupancyGrid2D:
    """基于环境障碍物的二维占用网格地图。

    机器人可以提前获知仓库的静态障碍物地图（例如通过 SLAM 或预建图），
    本类用于在随机游走阶段生成避开障碍物的覆盖轨迹，并在局部被堵时
    通过 A* 规划绕行路径。

    Args:
        env: 环境实例。
        resolution: 网格分辨率（米/格）。
        inflation: 障碍物膨胀半径（米），用于考虑机器人半径和安全距离。
    """

    def __init__(
        self,
        env: WarehouseEnv,
        resolution: float = 0.25,
        inflation: float = 0.55,
    ):
        self.resolution = float(resolution)
        self.bounds_low = env.bounds_low[:2].copy()
        self.bounds_high = env.bounds_high[:2].copy()
        self.shape = (
            max(1, int(np.ceil((self.bounds_high[0] - self.bounds_low[0]) / self.resolution)) + 1),
            max(1, int(np.ceil((self.bounds_high[1] - self.bounds_low[1]) / self.resolution)) + 1),
        )
        self.grid = np.zeros(self.shape, dtype=bool)
        self._build(env, inflation)

    def _build(self, env: WarehouseEnv, inflation: float) -> None:
        for obs in env.obstacles:
            if obs["name"].endswith("_wall"):
                continue
            if obs["type"] == "box":
                self._inflate_box(obs, inflation)
            elif obs["type"] == "cylinder":
                self._inflate_cylinder(obs, inflation)

    def _inflate_box(self, obs: dict[str, Any], inflation: float) -> None:
        center = obs["center"][:2]
        size = obs["size"][:2] + 2.0 * inflation
        half = size / 2.0
        min_p = center - half
        max_p = center + half
        i_min, j_min = self.world_to_grid(min_p)
        i_max, j_max = self.world_to_grid(max_p)
        i_min = max(0, i_min)
        j_min = max(0, j_min)
        i_max = min(self.shape[0] - 1, i_max)
        j_max = min(self.shape[1] - 1, j_max)
        self.grid[i_min : i_max + 1, j_min : j_max + 1] = True

    def _inflate_cylinder(self, obs: dict[str, Any], inflation: float) -> None:
        center = obs["center"][:2]
        radius = obs["radius"] + inflation
        axis = np.asarray(obs["axis"], dtype=float)
        # 只考虑近似直立的圆柱体或水平圆柱体在地面上的投影
        # 保守做法：以圆柱中心为圆心、半径为 radius 的圆膨胀
        i_c, j_c = self.world_to_grid(center)
        r_cells = int(np.ceil(radius / self.resolution))
        i_min = max(0, i_c - r_cells)
        i_max = min(self.shape[0] - 1, i_c + r_cells)
        j_min = max(0, j_c - r_cells)
        j_max = min(self.shape[1] - 1, j_c + r_cells)
        for i in range(i_min, i_max + 1):
            for j in range(j_min, j_max + 1):
                p = self.grid_to_world(i, j)
                if float(np.linalg.norm(p - center)) <= radius:
                    self.grid[i, j] = True

    def world_to_grid(self, p: np.ndarray) -> tuple[int, int]:
        p = np.asarray(p, dtype=float)
        i = int(np.floor((p[0] - self.bounds_low[0]) / self.resolution))
        j = int(np.floor((p[1] - self.bounds_low[1]) / self.resolution))
        return i, j

    def grid_to_world(self, i: int, j: int) -> np.ndarray:
        x = self.bounds_low[0] + (i + 0.5) * self.resolution
        y = self.bounds_low[1] + (j + 0.5) * self.resolution
        return np.array([x, y], dtype=float)

    def is_free(self, p: np.ndarray) -> bool:
        i, j = self.world_to_grid(p)
        if i < 0 or i >= self.shape[0] or j < 0 or j >= self.shape[1]:
            return False
        return not self.grid[i, j]

    def nearest_free(self, p: np.ndarray) -> np.ndarray:
        """返回离 p 最近的自由网格中心。"""
        i, j = self.world_to_grid(p)
        if self.is_free(p):
            return self.grid_to_world(i, j)
        for radius in range(1, max(self.shape)):
            for di in range(-radius, radius + 1):
                for dj in range(-radius, radius + 1):
                    if abs(di) != radius and abs(dj) != radius:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self.shape[0] and 0 <= nj < self.shape[1]:
                        if not self.grid[ni, nj]:
                            return self.grid_to_world(ni, nj)
        return np.asarray(p, dtype=float)

    def find_path_a_star(
        self,
        start: np.ndarray,
        goal: np.ndarray,
    ) -> "list[np.ndarray] | None":
        """在网格上从 start 到 goal 执行 A* 搜索，返回路径点列表。"""
        start = np.asarray(start, dtype=float)
        goal = np.asarray(goal, dtype=float)
        si, sj = self.world_to_grid(start)
        gi, gj = self.world_to_grid(goal)

        si = int(np.clip(si, 0, self.shape[0] - 1))
        sj = int(np.clip(sj, 0, self.shape[1] - 1))
        gi = int(np.clip(gi, 0, self.shape[0] - 1))
        gj = int(np.clip(gj, 0, self.shape[1] - 1))

        if self.grid[si, sj] or self.grid[gi, gj]:
            return None

        open_set: list[tuple[float, int, int]] = [(0.0, si, sj)]
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        g_score = np.full(self.shape, float("inf"))
        g_score[si, sj] = 0.0
        f_score = np.full(self.shape, float("inf"))
        f_score[si, sj] = float(np.linalg.norm(goal - start))

        closed: set[tuple[int, int]] = set()

        while open_set:
            _, ci, cj = heapq.heappop(open_set)
            if (ci, cj) == (gi, gj):
                path = []
                cur = (ci, cj)
                while cur in came_from:
                    path.append(self.grid_to_world(cur[0], cur[1]))
                    cur = came_from[cur]
                path.append(self.grid_to_world(si, sj))
                path.reverse()
                return path

            if (ci, cj) in closed:
                continue
            closed.add((ci, cj))

            for di, dj in [
                (1, 0),
                (-1, 0),
                (0, 1),
                (0, -1),
                (1, 1),
                (1, -1),
                (-1, 1),
                (-1, -1),
            ]:
                ni, nj = ci + di, cj + dj
                if ni < 0 or ni >= self.shape[0] or nj < 0 or nj >= self.shape[1]:
                    continue
                if self.grid[ni, nj]:
                    continue
                move_cost = math.hypot(di, dj) * self.resolution
                tentative_g = g_score[ci, cj] + move_cost
                if tentative_g < g_score[ni, nj]:
                    came_from[(ni, nj)] = (ci, cj)
                    g_score[ni, nj] = tentative_g
                    h = float(np.linalg.norm(self.grid_to_world(ni, nj) - goal))
                    f_score[ni, nj] = tentative_g + h
                    heapq.heappush(open_set, (f_score[ni, nj], ni, nj))

        return None


class SearchState(IntEnum):
    RANDOM_WALK = 0
    VISUAL_APPROACH = 1
    BASE_TRACKING = 2
    ARM_TRACKING = 3
    POINTING = 4
    FINISHED = 5
    LOST = 6


class OdorSearchAgent:
    """气味源搜索智能体。

    Args:
        robot: 机器人模型实例。
        env: 环境实例。
        random_walk_speed: 随机游走线速度（m/步）。
        gradient_step: 梯度搜索步长。
        source_distance_threshold: 视觉测距成功阈值（m）。
        lost_threshold: 连续多少步未检测到气味视为丢失。
        success_distance_threshold: 独立评估体系中判定成功的距离阈值（m）。
    """

    def __init__(
        self,
        robot: MobileArmRobot,
        env: WarehouseEnv,
        random_walk_speed: float = 0.08,
        gradient_step: float = 0.05,
        source_distance_threshold: float = 0.40,
        lost_threshold: int = 80,
        vision: "Any | None" = None,
        success_distance_threshold: float = 1.0,
    ):
        self.robot = robot
        self.env = env
        self.vision = vision
        self.rng = np.random.default_rng(0)
        self.random_walk_speed = float(random_walk_speed)
        self.gradient_step = float(gradient_step)
        self.source_distance_threshold = float(source_distance_threshold)
        self.lost_threshold = int(lost_threshold)
        self.success_distance_threshold = float(success_distance_threshold)

        self.state = SearchState.RANDOM_WALK
        self._last_state = self.state
        self._lost_counter = 0

        # 随机游走的覆盖地图（按高度层索引区分）
        self.cell_size = 0.8
        self.visited: set[tuple[int, int, int]] = set()
        self._current_height_level = 0
        self._height_presets = ["low", "mid", "high"]
        self._steps_in_level = 0
        self._steps_before_level_switch = 250  # 每个高度层至少尝试 250 步
        self._last_height_level = 0
        self._layer_visited_count = 0
        self._min_layer_coverage_cells = 40  # 当前层至少覆盖 40 个网格才允许切换
        self._layer_switch_coverage_ratio = 0.40  # 或覆盖率达到 40%

        # 随机方向
        self._random_yaw_target: "float | None" = None
        self._random_timer = 0

        # 卡住检测
        self._recent_positions: list[np.ndarray] = []
        self._stuck_check_interval = 30
        self._stuck_recovery_counter = 0
        self._stuck_recovery_duration = 40
        self._stuck_recovery_yaw = 0.0

        # 目标点（用于机械臂指向）
        self._target_point: "np.ndarray | None" = None
        self._visual_target: "np.ndarray | None" = None

        # 历史浓度，用于判断“已指向气味源”
        self._ee_history: list[float] = []
        self._high_concentration_steps = 0
        # 状态切换滞后：防止噪声导致频繁切换
        self._ee_trigger_steps = 0
        self._ee_trigger_required = 3  # 连续多少步触发才进入追踪
        self._ee_lost_steps = 0

        # 结束条件：嗅觉主导 + 视觉辅助
        self._visual_confirm_steps = 0
        self._visual_confirm_required = 3
        self._visual_distance_threshold = 0.80  # 视觉辅助结束的距离阈值

        # 嗅觉主导结束条件参数
        self._odor_finish_required = 50
        self._odor_finish_window = 40
        self._odor_finish_min_ppm = 1000.0
        self._odor_finish_peak_delta = 0.20
        self._odor_finish_displacement_threshold = 0.15
        self._odor_finish_counter = 0
        self._odor_finish_base_positions: list[np.ndarray] = []

        # ARM_TRACKING 超时：长时间未视觉确认则重新探索
        self._arm_tracking_start_step = 0
        self._arm_tracking_max_steps = 250
        self._arm_tracking_prev_ppm = 0.0

        # BASE_TRACKING 超时：长时间未触发末端高浓度则认为该角传感器线索不可靠
        self._base_tracking_start_step = 0
        self._base_tracking_max_steps = 120

        # 主动视觉扫描：ARM_TRACKING 高浓度但未看到源时旋转搜索
        self._visual_scan_steps = 0

        # 局部极值/平台检测（用于逃离局部高浓度区）
        self._recent_ee_ppm: list[float] = []
        self._recent_ee_positions: list[np.ndarray] = []
        self._recent_base_positions: list[np.ndarray] = []
        self._local_max_window = 30
        self._local_max_ppm_threshold = 5.0 * self.env.detection_threshold

        # 逃离模式：检测到局部极值或 ARM_TRACKING 超时后，暂时不重新进入追踪
        self._escape_cooldown = 0
        self._escape_cooldown_steps = 120

        # 不重新进入追踪的区域（避免反复回到同一局部高浓度丝）
        self._no_track_zones: list[tuple[np.ndarray, float]] = []
        self._no_track_zone_radius = 2.0

        # 进入末端追踪的最小浓度阈值，防止被远距离低浓度烟羽边缘误导
        self._arm_tracking_min_ppm = 50.0

        # 逃离目标：检测到局部极值后，强制基座先离开该区域
        self._escape_target: "np.ndarray | None" = None
        self._escape_steps_remaining = 0

        # 牛耕法（boustrophedon）覆盖轨迹：每个高度层系统性地扫过空间
        self._coverage_track_spacing = 0.8
        self._coverage_tracks: list[list[np.ndarray]] = []
        self._coverage_track_idx = 0
        self._coverage_waypoint_idx = 0
        self._coverage_reached_threshold = 0.3

        # 全局覆盖路径：利用已知障碍物地图按最近邻顺序连接各扫描线
        self._global_coverage_path: "list[np.ndarray]" = []
        self._global_waypoint_idx = 0

        # 基于障碍物地图的二维占用网格（仅使用 env.obstacles，不使用源信息）
        # 膨胀半径取机器人半宽 + 安全余量，避免通道被过度阻塞
        self._occupancy_grid = OccupancyGrid2D(self.env, resolution=0.25, inflation=0.50)
        self._coverage_path: "list[np.ndarray]" = []

        # 逃离路径（A* 规划）
        self._escape_path: "list[np.ndarray]" = []

        # 机械臂末端上下扫描相位（随机游走时增加 z 覆盖）
        self._arm_scan_phase = 0.0

        # 上一帧浓度，用于判断浓度趋势
        self._last_ee_ppm = 0.0

        # 覆盖路径是否需要因卡住而重建
        self._coverage_needs_rebuild = False

        self._build_coverage_tracks()

        # ARM_TRACKING 梯度记忆，避免连续重估
        self._last_gradient = np.zeros(3)

        # 机器人声明的源位置（用于独立评估）
        self._declared_source_position: "np.ndarray | None" = None

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
        self._last_ee_ppm = float(ee_ppm)
        corner_ppm = {k: v["ppm"] for k, v in sensor_readings.items() if k != "ee"}
        any_corner_trigger = any(
            ppm > self.env.detection_threshold for ppm in corner_ppm.values()
        )
        max_corner_ppm = max(corner_ppm.values()) if corner_ppm else 0.0

        # 视觉确认
        visual_confirmed = (
            vision_result is not None
            and vision_result.get("source_visible", False)
        )
        visual_in_view = (
            vision_result is not None
            and vision_result.get("source_in_view", False)
        )
        visual_distance = (
            vision_result.get("distance", float("inf"))
            if vision_result is not None
            else float("inf")
        )

        # 记录基座位置，用于嗅觉结束条件的稳定性判断
        self._odor_finish_base_positions.append(self.robot.base_position[:2].copy())
        if len(self._odor_finish_base_positions) > self._odor_finish_window:
            self._odor_finish_base_positions.pop(0)

        info = {
            "state": self.state.name,
            "ee_ppm": ee_ppm,
            "max_corner_ppm": max_corner_ppm,
            "visual_confirmed": visual_confirmed,
            "visual_in_view": visual_in_view,
            "visual_distance": visual_distance,
        }

        # 状态转移
        if self.state == SearchState.FINISHED:
            return np.zeros(3), np.zeros(6), info

        # 如果满足严格结束条件，结束
        if self._should_declare_finished(ee_ppm, visual_confirmed, visual_distance):
            self.state = SearchState.FINISHED
            self._declared_source_position = self.robot.ee_sensor_position().copy()
            info["state"] = self.state.name
            info["declared_source_position"] = self._declared_source_position
            return np.zeros(3), np.zeros(6), info

        # 气味触发/丢失的滞后计数
        strong_ee = ee_ppm > self.env.detection_threshold
        if strong_ee or any_corner_trigger:
            self._ee_trigger_steps = min(self._ee_trigger_steps + 1, self._ee_trigger_required + 1)
            self._ee_lost_steps = 0
        else:
            self._ee_trigger_steps = 0
            self._ee_lost_steps += 1

        # 视觉优先：任何状态下看到潜在气味源都优先进入视觉接近
        if visual_in_view and self.state not in (
            SearchState.VISUAL_APPROACH,
            SearchState.FINISHED,
        ):
            self.state = SearchState.VISUAL_APPROACH
            self._visual_target = None  # 将在动作中根据相机射线实时计算

        # 逃离冷却倒计时
        if self._escape_cooldown > 0:
            self._escape_cooldown -= 1

        # 状态机
        prev_state = self._last_state
        self._last_state = self.state

        if self.state == SearchState.RANDOM_WALK:
            in_no_track_zone = self._is_in_no_track_zone(self.robot.base_position[:2])
            if (
                strong_ee
                and ee_ppm >= self._arm_tracking_min_ppm
                and self._ee_trigger_steps >= self._ee_trigger_required
                and self._escape_cooldown == 0
                and not in_no_track_zone
            ):
                self.state = SearchState.ARM_TRACKING
            elif (
                any_corner_trigger
                and self._escape_cooldown == 0
                and not in_no_track_zone
            ):
                self.state = SearchState.BASE_TRACKING

        elif self.state == SearchState.VISUAL_APPROACH:
            if (
                strong_ee
                and ee_ppm >= self._arm_tracking_min_ppm
                and self._ee_trigger_steps >= self._ee_trigger_required
            ):
                self.state = SearchState.ARM_TRACKING
            elif not visual_in_view and self._ee_lost_steps > self.lost_threshold // 2:
                self.state = SearchState.RANDOM_WALK
                self._visual_target = None

        elif self.state in (SearchState.ARM_TRACKING, SearchState.POINTING):
            # 刚进入 ARM_TRACKING 时重置局部极值跟踪窗口
            if prev_state not in (SearchState.ARM_TRACKING, SearchState.POINTING):
                self._recent_ee_ppm.clear()
                self._recent_ee_positions.clear()
                self._last_gradient = np.zeros(3)
                self._arm_tracking_start_step = 0  # 由动作函数自增
                self._arm_tracking_prev_ppm = 0.0
                self._visual_scan_steps = 0

            self._arm_tracking_start_step += 1
            self._arm_tracking_prev_ppm = float(ee_ppm)
            if not strong_ee and self._ee_lost_steps > self.lost_threshold:
                if any_corner_trigger:
                    self.state = SearchState.BASE_TRACKING
                else:
                    self.state = SearchState.LOST
            elif self._is_local_maximum_stuck(ee_ppm):
                # 陷入局部高浓度平台，标记当前位置并启动逃离
                self.state = SearchState.LOST
                self._add_no_track_zone(self.robot.base_position[:2])
                self.visited.add(self._pos_to_cell(self.robot.base_position, use_height_level=True))
                self._escape_cooldown = self._escape_cooldown_steps
                self._set_escape_target()
            elif (
                self._arm_tracking_start_step > self._arm_tracking_max_steps
                and not visual_confirmed
            ):
                # 长时间未视觉确认，认为可能陷入局部区域，启动逃离
                self.state = SearchState.LOST
                self._add_no_track_zone(self.robot.base_position[:2])
                self.visited.add(self._pos_to_cell(self.robot.base_position, use_height_level=True))
                self._escape_cooldown = self._escape_cooldown_steps
                self._set_escape_target()

        elif self.state == SearchState.BASE_TRACKING:
            if prev_state != SearchState.BASE_TRACKING:
                self._base_tracking_start_step = 0
            self._base_tracking_start_step += 1
            if (
                strong_ee
                and ee_ppm >= self._arm_tracking_min_ppm
                and self._ee_trigger_steps >= self._ee_trigger_required
            ):
                self.state = SearchState.ARM_TRACKING
            elif self._ee_lost_steps > self.lost_threshold and not any_corner_trigger:
                self.state = SearchState.LOST
            elif self._base_tracking_start_step > self._base_tracking_max_steps:
                # 长时间未触发高浓度末端读数，认为该固定传感器线索不可靠，启动逃离
                self.state = SearchState.LOST
                self._add_no_track_zone(self.robot.base_position[:2])
                self.visited.add(self._pos_to_cell(self.robot.base_position, use_height_level=True))
                self._escape_cooldown = self._escape_cooldown_steps
                self._set_escape_target()

        elif self.state == SearchState.LOST:
            # 丢失后重新回到随机游走
            self.state = SearchState.RANDOM_WALK

        # 动作生成
        if self.state == SearchState.RANDOM_WALK:
            base_cmd, joint_delta = self._random_walk_action()
        elif self.state == SearchState.VISUAL_APPROACH:
            base_cmd, joint_delta = self._visual_approach_action(vision_result)
        elif self.state == SearchState.BASE_TRACKING:
            base_cmd, joint_delta = self._base_tracking_action(corner_ppm)
        elif self.state in (SearchState.ARM_TRACKING, SearchState.POINTING):
            base_cmd, joint_delta = self._arm_tracking_action(ee_ppm, visual_confirmed)
        else:
            base_cmd, joint_delta = np.zeros(3), np.zeros(6)

        info["state"] = self.state.name
        info["target_point"] = self._target_point
        info["visual_target"] = self._visual_target
        return base_cmd, joint_delta, info

    @property
    def declared_source_position(self) -> "np.ndarray | None":
        """机器人声明的气味源位置（进入 FINISHED 时的末端传感器位置）。"""
        return self._declared_source_position

    # ---------------------------------------------------------------------- #
    # 动作生成子函数
    # ---------------------------------------------------------------------- #
    def _random_walk_action(self) -> tuple[np.ndarray, np.ndarray]:
        """随机游走 / 覆盖搜索：牛耕法轨迹 + 前沿修正，确保每个高度层基本覆盖空间。

        策略：
        - 在当前机械臂高度层，按牛耕法（boustrophedon）轨迹系统性地扫过仓库；
        - 当前方遇到障碍物或卡住时，临时切换为前沿方向选择；
        - 同一高度层内维护已访问网格；高度层切换后清除并重新扫描；
        - 若存在逃离目标（局部极值逃离后），优先驶向该目标；
        - 若长时间未移动（卡住），强制进入恢复模式，随机转向并直线脱离。
        """
        self._steps_in_level += 1

        # 逃离目标优先；但逃离过程中若重新闻到高浓度，应优先复捕
        if self._escape_target is not None:
            return self._escape_action()

        # 切换高度层判断
        should_switch_height = False
        if self._steps_in_level > self._steps_before_level_switch:
            should_switch_height = True
        elif self._is_coverage_complete():
            should_switch_height = True

        if should_switch_height:
            self._current_height_level = (self._current_height_level + 1) % len(
                self._height_presets
            )
            self._steps_in_level = 0
            self._layer_visited_count = 0
            self.visited.clear()
            self._build_coverage_tracks()  # 为新高度层重建轨迹

        # 高度层发生变化时清除已访问标记，重新扫描新高度
        if self._current_height_level != self._last_height_level:
            self.visited.clear()
            self._last_height_level = self._current_height_level
            self._layer_visited_count = 0
            self._build_coverage_tracks()

        # 卡住恢复后，从当前位置重新生成覆盖路径
        if self._coverage_needs_rebuild:
            self._build_coverage_tracks()
            self._coverage_needs_rebuild = False

        # 覆盖记录
        cell = self._pos_to_cell(self.robot.base_position, use_height_level=True)
        if cell not in self.visited:
            self._layer_visited_count += 1
        self.visited.add(cell)

        # 设置机械臂到当前层预设姿态，并叠加小幅 z 方向扫描，以增加烟羽高度匹配概率
        preset_name = self._height_presets[self._current_height_level]
        target_joints = np.array(self.robot.arm_cfg["preset_poses"][preset_name], dtype=float)
        joint_delta = self._joint_delta_to_target_with_scan(target_joints)

        # 优先跟随牛耕法覆盖轨迹；若当前路点不可达，尝试后续路点
        waypoint = self._current_coverage_waypoint()
        attempts = 0
        while waypoint is not None and attempts < 8:
            base_cmd = self._follow_waypoint(waypoint, target_speed=self.random_walk_speed * 1.2)
            if base_cmd is not None:
                # 即使跟随轨迹成功，也要检查是否卡住
                if self._check_stuck():
                    return self._stuck_recovery_action(joint_delta)
                return base_cmd, joint_delta
            # 当前路点被堵，跳过并尝试下一个
            self._advance_coverage_waypoint()
            waypoint = self._current_coverage_waypoint()
            attempts += 1

        # 轨迹无法跟随或卡住时，检查是否进入强制恢复
        stuck = self._check_stuck()
        if stuck or self._stuck_recovery_counter > 0:
            return self._stuck_recovery_action(joint_delta)

        # 回退到前沿方向选择
        current_yaw = self.robot.base_yaw
        forward_probe = self._probe_point(current_yaw, distance=self.robot.length + 0.6)
        front_clear = (
            self.env.inside_bounds(forward_probe)
            and not self._predict_collision(forward_probe)
        )

        if front_clear:
            dyaw = float(np.clip(np.random.normal(0.0, 0.03), -0.06, 0.06))
            # 每 30 步根据前沿方向做一次修正，避免绕圈
            if self._steps_in_level % 30 == 0:
                frontier_yaw = self._nearest_frontier_yaw()
                if frontier_yaw is not None:
                    dyaw += float(
                        np.clip(
                            self._angle_diff(frontier_yaw, current_yaw) * 0.3,
                            -0.15,
                            0.15,
                        )
                    )
        else:
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
                dyaw = self.robot.max_yaw_delta * 1.5

        dx = self.random_walk_speed * 1.2
        base_cmd = np.array([dx, 0.0, dyaw])
        return base_cmd, joint_delta

    def _stuck_recovery_action(
        self,
        joint_delta: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """强制脱离卡住状态：随机转向并朝空旷方向直线前进。

        持续 _stuck_recovery_duration 步；期间不再跟随覆盖轨迹，
        而是朝一个选定的恢复方向前进。优先选择占用网格上最长的
        自由射线方向，避免再次撞到障碍物。
        """
        if self._stuck_recovery_counter == 0:
            # 首次进入恢复：选择一个远离障碍物的方向
            self._stuck_recovery_counter = self._stuck_recovery_duration
            self._coverage_needs_rebuild = True  # 恢复结束后重新规划覆盖路径
            best_yaw = self._best_free_direction()
            if best_yaw is None:
                # 随机方向，偏向仓库中心
                to_center = -self.robot.base_position[:2]
                center_yaw = math.atan2(to_center[1], to_center[0])
                best_yaw = center_yaw + self.rng.uniform(-math.pi / 3, math.pi / 3)
            self._stuck_recovery_yaw = best_yaw

        self._stuck_recovery_counter -= 1
        current_yaw = self.robot.base_yaw
        dyaw = float(
            np.clip(
                self._angle_diff(self._stuck_recovery_yaw, current_yaw),
                -self.robot.max_yaw_delta * 1.5,
                self.robot.max_yaw_delta * 1.5,
            )
        )
        # 朝向恢复方向时前进，否则先原地转向
        angle_error = abs(self._angle_diff(self._stuck_recovery_yaw, current_yaw))
        dx = self.random_walk_speed * 1.2 if angle_error < math.radians(30.0) else 0.0
        base_cmd = np.array([dx, 0.0, dyaw])
        return base_cmd, joint_delta

    def _best_free_direction(self) -> "float | None":
        """在 360° 范围内选择占用网格上自由距离最长的方向，并偏向未访问区域。"""
        current_yaw = self.robot.base_yaw
        pos_xy = self.robot.base_position[:2]
        max_free_dist = 0.0
        best_yaw = None
        frontier_yaw = self._nearest_frontier_yaw()

        for delta in np.linspace(-np.pi, np.pi, 36, endpoint=False):
            yaw = current_yaw + delta
            free_dist = self._free_distance(yaw)
            if free_dist < 0.4:
                continue
            score = free_dist
            # 偏向未访问前沿方向
            if frontier_yaw is not None:
                angle_diff = abs(self._angle_diff(frontier_yaw, yaw))
                score += 2.0 * max(0.0, 1.0 - angle_diff / math.radians(60.0))
            # 偏向仓库中心
            to_center = -pos_xy
            to_center_norm = float(np.linalg.norm(to_center))
            if to_center_norm > 1e-6:
                center_dir = to_center / to_center_norm
                score += 1.5 * np.dot(
                    np.array([np.cos(yaw), np.sin(yaw)]),
                    center_dir,
                )
            if score > max_free_dist:
                max_free_dist = score
                best_yaw = yaw

        return best_yaw

    def _free_distance(self, yaw: float) -> float:
        """沿给定偏航角在占用网格上返回自由距离（直到障碍物或边界）。"""
        pos_xy = self.robot.base_position[:2]
        direction = np.array([np.cos(yaw), np.sin(yaw)], dtype=float)
        max_dist = 5.0
        step = self._occupancy_grid.resolution
        dist = step  # 从当前位置前方一步开始，避免当前位置被膨胀网格误判
        while dist < max_dist:
            p = pos_xy + dist * direction
            if not self._occupancy_grid.is_free(p):
                return max(0.0, dist - step)
            dist += step
        return max_dist

    def _follow_waypoint(
        self,
        waypoint: np.ndarray,
        target_speed: float,
    ) -> "np.ndarray | None":
        """向覆盖路点行驶，到达后自动推进路点。

        返回 None 表示前方被障碍物严重阻挡，需要上层回退到前沿选择。
        本实现使用朝向路点的方向进行多距离探测，并在正前方被堵时
        尝试小角度绕行。
        """
        base_xy = self.robot.base_position[:2]
        to_wp = waypoint[:2] - base_xy
        dist = float(np.linalg.norm(to_wp))

        if dist < self._coverage_reached_threshold:
            self._advance_coverage_waypoint()
            # 返回一个继续向下一目标前进的命令
            next_wp = self._current_coverage_waypoint()
            if next_wp is None:
                return np.zeros(3)
            to_wp = next_wp[:2] - base_xy
            dist = float(np.linalg.norm(to_wp))
            if dist < 1e-6:
                return np.zeros(3)

        yaw = self.robot.base_yaw
        target_yaw = math.atan2(to_wp[1], to_wp[0])
        direction = to_wp / dist

        # 沿朝向路点的方向在多个距离上探测
        probe_clear = True
        for probe_dist in [self.robot.length + 0.3, self.robot.length + 0.6, self.robot.length + 0.9]:
            probe = self._probe_point(target_yaw, distance=probe_dist)
            if not self.env.inside_bounds(probe) or self._predict_collision(probe):
                probe_clear = False
                break

        if probe_clear:
            local_x = direction[0] * np.cos(yaw) + direction[1] * np.sin(yaw)
            local_y = -direction[0] * np.sin(yaw) + direction[1] * np.cos(yaw)

            dx = float(np.clip(local_x * target_speed, -target_speed, target_speed))
            dy = float(np.clip(local_y * target_speed * 0.5, -target_speed, target_speed))
            dyaw = float(np.clip(0.5 * math.atan2(local_y, local_x), -0.12, 0.12))
            return np.array([dx, dy, dyaw])

        # 正前方被堵：尝试向左/右小角度绕行
        for side_offset in [-1.0, 1.0]:
            probe_yaw = target_yaw + side_offset * math.radians(30.0)
            probe = self._probe_point(probe_yaw, distance=self.robot.length + 0.6)
            if self.env.inside_bounds(probe) and not self._predict_collision(probe):
                side_dir = np.array([np.cos(probe_yaw), np.sin(probe_yaw)])
                local_x = float(side_dir[0] * np.cos(yaw) + side_dir[1] * np.sin(yaw))
                local_y = float(-side_dir[0] * np.sin(yaw) + side_dir[1] * np.cos(yaw))
                dx = float(np.clip(local_x * target_speed, -target_speed, target_speed))
                dy = float(np.clip(local_y * target_speed * 0.5, -target_speed, target_speed))
                dyaw = float(
                    np.clip(
                        self._angle_diff(probe_yaw, yaw),
                        -0.12,
                        0.12,
                    )
                )
                return np.array([dx, dy, dyaw])

        # 所有直接方向都被堵：使用已知障碍物地图进行 A* 局部路径规划
        path = self._occupancy_grid.find_path_a_star(base_xy, waypoint[:2])
        if path is not None and len(path) >= 2:
            self._coverage_path = path
            next_wp = path[1]
            to_next = next_wp - base_xy
            dist_next = float(np.linalg.norm(to_next))
            if dist_next < 1e-6:
                return np.zeros(3)
            direction = to_next / dist_next
            local_x = float(direction[0] * np.cos(yaw) + direction[1] * np.sin(yaw))
            local_y = float(-direction[0] * np.sin(yaw) + direction[1] * np.cos(yaw))
            dx = float(np.clip(local_x * target_speed, -target_speed, target_speed))
            dy = float(np.clip(local_y * target_speed * 0.5, -target_speed, target_speed))
            dyaw = float(np.clip(0.5 * math.atan2(local_y, local_x), -0.12, 0.12))
            return np.array([dx, dy, dyaw])

        # 无法规划路径，回退到前沿选择
        return None

    def _estimate_layer_total_cells(self) -> int:
        """粗略估计当前高度层需要覆盖的网格总数。"""
        bounds = self.env.bounds_high - self.env.bounds_low
        area = float(bounds[0] * bounds[1])
        return max(1, int(area / (self.cell_size ** 2)))

    def _build_coverage_tracks(self) -> None:
        """生成牛耕法覆盖轨迹（平行于 x 轴的往复轨道），并避开已知障碍物。

        利用已知障碍物地图，本方法构建一条从当前位置出发、按最近邻顺序
        连接各扫描线的全局覆盖路径，扫描线之间用 A* 规划绕障转移路径。
        """
        margin = 1.5
        x_min = self.env.bounds_low[0] + margin
        x_max = self.env.bounds_high[0] - margin
        y_min = self.env.bounds_low[1] + margin
        y_max = self.env.bounds_high[1] - margin

        y_values = np.arange(y_min, y_max + 1e-9, self._coverage_track_spacing)
        if len(y_values) < 2:
            y_values = np.array([y_min, y_max])

        tracks: list[list[np.ndarray]] = []
        for i, y in enumerate(y_values):
            # 在该 y 值上找到所有自由 x 区间
            free_segments = self._free_x_segments_at_y(y, x_min, x_max)
            for seg_x_min, seg_x_max in free_segments:
                if seg_x_max - seg_x_min < 0.6:
                    continue
                n_points = max(2, int(round((seg_x_max - seg_x_min) / 0.6)) + 1)
                # 偶数轨道从左到右，奇数轨道从右到左，形成往复
                if i % 2 == 0:
                    xs = np.linspace(seg_x_min, seg_x_max, n_points)
                else:
                    xs = np.linspace(seg_x_max, seg_x_min, n_points)
                track = [np.array([x, y, 0.0], dtype=float) for x in xs]
                tracks.append(track)
        self._coverage_tracks = tracks
        self._coverage_track_idx = 0
        self._coverage_waypoint_idx = 0

        # 构建全局覆盖路径：从当前位置出发，按最近邻顺序连接各 track，
        # 优先探索机器人附近区域，避免固定 y 顺序导致先远赴仓库另一端。
        self._global_coverage_path = []
        self._global_waypoint_idx = 0
        if not tracks:
            return

        current_xy = self.robot.base_position[:2].copy()
        if not self._occupancy_grid.is_free(current_xy):
            current_xy = self._occupancy_grid.nearest_free(current_xy)
        remaining = list(range(len(tracks)))
        visited_order: list[int] = []

        while remaining:
            best_idx = None
            best_dist = float("inf")
            best_endpoint = None
            best_reversed = False
            for idx in remaining:
                track = tracks[idx]
                d_first = float(np.linalg.norm(track[0][:2] - current_xy))
                d_last = float(np.linalg.norm(track[-1][:2] - current_xy))
                if d_first <= d_last:
                    d, ep, rev = d_first, track[0], False
                else:
                    d, ep, rev = d_last, track[-1], True
                if d < best_dist:
                    best_dist = d
                    best_idx = idx
                    best_endpoint = ep
                    best_reversed = rev

            if best_idx is None:
                break
            remaining.remove(best_idx)
            visited_order.append(best_idx)

            # 从当前位置到入口点用 A* 规划转移路径
            transfer = self._plan_transfer_path(current_xy, best_endpoint[:2])
            if transfer:
                self._global_coverage_path.extend(transfer)
            else:
                self._global_coverage_path.append(best_endpoint.copy())

            ordered_track = list(reversed(tracks[best_idx])) if best_reversed else tracks[best_idx]
            self._global_coverage_path.extend(ordered_track)
            current_xy = ordered_track[-1][:2].copy()

        # 去重：相邻重复点合并
        self._global_coverage_path = self._deduplicate_path(self._global_coverage_path)

    def _deduplicate_path(self, path: "list[np.ndarray]") -> "list[np.ndarray]":
        """合并相邻重复或过于接近的路径点。"""
        if not path:
            return path
        result = [path[0].copy()]
        for p in path[1:]:
            if float(np.linalg.norm(p[:2] - result[-1][:2])) > 0.05:
                result.append(p.copy())
        return result

    def _plan_transfer_path(
        self,
        start_xy: np.ndarray,
        goal_xy: np.ndarray,
    ) -> "list[np.ndarray]":
        """在占用网格上用 A* 规划从 start_xy 到 goal_xy 的转移路径。"""
        start_xy = np.asarray(start_xy, dtype=float)
        goal_xy = np.asarray(goal_xy, dtype=float)
        if np.linalg.norm(start_xy - goal_xy) < 1e-3:
            return []
        path = self._occupancy_grid.find_path_a_star(start_xy, goal_xy)
        if path is None:
            return []
        # 将路径点转为 3D 路点
        return [np.array([p[0], p[1], 0.0], dtype=float) for p in path]

    def _free_x_segments_at_y(
        self,
        y: float,
        x_min: float,
        x_max: float,
    ) -> list[tuple[float, float]]:
        """返回给定 y 值上，机器人在 x 方向上可以自由通行的区间列表。"""
        # 沿 x 轴采样，标记每个采样点是否在障碍物自由空间内
        n_samples = max(3, int(round((x_max - x_min) / self._occupancy_grid.resolution)) + 1)
        xs = np.linspace(x_min, x_max, n_samples)
        free = [self._occupancy_grid.is_free(np.array([x, y])) for x in xs]

        segments = []
        start_idx = None
        for idx, is_f in enumerate(free):
            if is_f and start_idx is None:
                start_idx = idx
            elif not is_f and start_idx is not None:
                segments.append((xs[start_idx], xs[idx - 1]))
                start_idx = None
        if start_idx is not None:
            segments.append((xs[start_idx], xs[-1]))
        return segments

    def _current_coverage_waypoint(self) -> "np.ndarray | None":
        """返回当前全局覆盖目标路点。"""
        if not self._global_coverage_path:
            return None
        if self._global_waypoint_idx >= len(self._global_coverage_path):
            return None
        return self._global_coverage_path[self._global_waypoint_idx]

    def _advance_coverage_waypoint(self) -> None:
        """推进到下一个全局覆盖路点。"""
        if not self._global_coverage_path:
            return
        self._global_waypoint_idx += 1

    def _is_coverage_complete(self) -> bool:
        """当前高度层全局覆盖路径是否已完成。"""
        return self._global_waypoint_idx >= len(self._global_coverage_path)

    def _set_escape_target(self) -> None:
        """根据当前浓度梯度反方向设置一个逃离目标点，并用 A* 规划绕障路径。"""
        base_xy = self.robot.base_position[:2].copy()
        ee_pos = self.robot.ee_sensor_position()
        gradient = self._estimate_gradient(ee_pos)
        grad_xy = gradient[:2].copy()
        grad_norm = float(np.linalg.norm(grad_xy))
        if grad_norm > 1e-6:
            away = -grad_xy / grad_norm
        else:
            # 梯度不明显时随机选一个远离当前方向
            yaw = self.robot.base_yaw + math.pi
            away = np.array([np.cos(yaw), np.sin(yaw)])

        # 尝试 1.5 m 外的点，确保在边界内且无障碍
        for dist in [3.0, 2.5, 2.0, 1.5, 1.0]:
            target_xy = base_xy + dist * away
            target_3d = np.array([target_xy[0], target_xy[1], 0.0])
            if (
                self.env.inside_bounds(target_3d)
                and not self._predict_collision(target_3d)
                and not self._is_in_no_track_zone(target_xy)
            ):
                self._escape_target = target_xy.copy()
                self._escape_steps_remaining = int(dist / (self.random_walk_speed * 1.2)) + 80
                # 使用 A* 规划从当前位置到逃离目标的路径
                self._escape_path = self._plan_transfer_path(base_xy, target_xy)
                if not self._escape_path:
                    self._escape_path = [target_3d]
                return
        # 找不到合适目标，使用当前位置前方 0.8 m
        yaw = self.robot.base_yaw
        fallback_target = base_xy + 0.8 * np.array([np.cos(yaw), np.sin(yaw)])
        self._escape_target = fallback_target.copy()
        self._escape_steps_remaining = 100
        self._escape_path = self._plan_transfer_path(base_xy, fallback_target)
        if not self._escape_path:
            self._escape_path = [np.array([fallback_target[0], fallback_target[1], 0.0])]

    def _escape_action(self) -> tuple[np.ndarray, np.ndarray]:
        """沿 A* 逃离路径行驶；若中途重新闻到高浓度，则中断逃离并复捕。"""
        # 逃离过程中若检测到足够浓度，立即中断并复捕
        if (
            self._last_ee_ppm >= self._arm_tracking_min_ppm
            and self._ee_trigger_steps >= self._ee_trigger_required
        ):
            self._escape_target = None
            self._escape_path = []
            self._escape_cooldown = 0
            self.state = SearchState.ARM_TRACKING
            return self._arm_tracking_action(self._last_ee_ppm, visual_confirmed=False)

        if self._escape_target is None:
            return self._random_walk_action()

        self._escape_steps_remaining -= 1
        if self._escape_steps_remaining <= 0:
            self._escape_target = None
            self._escape_path = []
            return self._random_walk_action()

        base_xy = self.robot.base_position[:2]
        to_target = self._escape_target - base_xy
        dist = float(np.linalg.norm(to_target))
        if dist < 0.2:
            self._escape_target = None
            self._escape_path = []
            return self._random_walk_action()

        # 沿 A* 路径跟随；若路径为空则直接朝目标前进
        if self._escape_path:
            while self._escape_path:
                wp = self._escape_path[0]
                if float(np.linalg.norm(wp[:2] - base_xy)) < self._coverage_reached_threshold:
                    self._escape_path.pop(0)
                else:
                    break
            if self._escape_path:
                wp = self._escape_path[0]
                base_cmd = self._follow_waypoint(wp, target_speed=self.random_walk_speed * 1.2)
                if base_cmd is not None:
                    joint_delta = self._joint_delta_to_target_with_scan(
                        np.array(
                            self.robot.arm_cfg["preset_poses"][
                                self._height_presets[self._current_height_level]
                            ],
                            dtype=float,
                        )
                    )
                    return base_cmd, joint_delta

        # A* 跟随失败或路径为空时直接朝目标前进
        yaw = self.robot.base_yaw
        direction = to_target / dist
        local_x = direction[0] * np.cos(yaw) + direction[1] * np.sin(yaw)
        local_y = -direction[0] * np.sin(yaw) + direction[1] * np.cos(yaw)

        dx = self.random_walk_speed * 1.2
        dy = 0.0
        dyaw = float(np.clip(0.4 * math.atan2(local_y, local_x), -0.15, 0.15))

        joint_delta = self._joint_delta_to_target_with_scan(
            np.array(
                self.robot.arm_cfg["preset_poses"][self._height_presets[self._current_height_level]],
                dtype=float,
            )
        )
        return np.array([dx, dy, dyaw]), joint_delta

    def _nearest_frontier_yaw(self) -> "float | None":
        """返回最近未访问前沿单元相对于当前位置的方向角。"""
        current_cell = self._pos_to_cell(self.robot.base_position, use_height_level=True)
        pos_xy = self.robot.base_position[:2]

        # 按螺旋顺序搜索最近未访问单元
        for radius in range(1, 20):
            best_cell = None
            best_dist = float("inf")
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) + abs(dy) > radius * 1.5:
                        continue
                    cell = (current_cell[0] + dx, current_cell[1] + dy, current_cell[2])
                    if cell in self.visited:
                        continue
                    cell_xy = np.array([cell[0] * self.cell_size, cell[1] * self.cell_size])
                    dist = float(np.linalg.norm(cell_xy - pos_xy))
                    if dist < best_dist:
                        best_dist = dist
                        best_cell = cell_xy
            if best_cell is not None:
                return float(math.atan2(best_cell[1] - pos_xy[1], best_cell[0] - pos_xy[0]))
        return None

    def _visual_approach_action(
        self,
        vision_result: "dict[str, Any] | None",
    ) -> tuple[np.ndarray, np.ndarray]:
        """摄像头看到潜在气味源时，沿相机射线方向接近。

        不使用 env.source_pos；仅利用视觉输出的图像坐标 (u, v) 和相机外参
        估计源在机器人前方的大致方向，并前进/旋转以对齐目标。
        """
        yaw = self.robot.base_yaw
        cam_pos, R_cam = self.robot.end_effector_pose()

        # 默认：沿当前相机光轴前进
        default_dir_world = R_cam[:, 2]
        default_dir_world = default_dir_world / (np.linalg.norm(default_dir_world) + 1e-9)
        target_dir = default_dir_world.copy()

        if vision_result is not None and vision_result.get("image_xy") is not None:
            u, v = vision_result["image_xy"]
            # 相机坐标系下指向目标的射线（z 为光轴）
            ray_cam = np.array([u, v, 1.0], dtype=float)
            ray_cam = ray_cam / (np.linalg.norm(ray_cam) + 1e-9)
            target_dir = R_cam @ ray_cam
            target_dir = target_dir / (np.linalg.norm(target_dir) + 1e-9)

        self._visual_target = cam_pos + 0.5 * target_dir
        self._target_point = self._visual_target.copy()

        # 机械臂指向目标
        joint_delta = self._point_ee_to_target(self._target_point)

        # 基座沿目标方向在地面投影移动，并旋转以对齐
        target_dir_xy = target_dir.copy()
        target_dir_xy[2] = 0.0
        norm_xy = float(np.linalg.norm(target_dir_xy))
        if norm_xy < 1e-6:
            base_cmd = np.zeros(3)
            return base_cmd, joint_delta

        target_dir_xy = target_dir_xy / norm_xy
        local_x = target_dir_xy[0] * np.cos(yaw) + target_dir_xy[1] * np.sin(yaw)
        local_y = -target_dir_xy[0] * np.sin(yaw) + target_dir_xy[1] * np.cos(yaw)

        # 越近越慢
        speed = min(self.random_walk_speed, 0.03 + 0.05 * float(vision_result.get("distance", 2.0)))
        dx = float(np.clip(local_x * speed, -speed, speed))
        dy = float(np.clip(local_y * speed * 0.5, -speed, speed))
        dyaw = float(np.clip(0.25 * math.atan2(local_y, local_x), -0.10, 0.10))

        base_cmd = np.array([dx, dy, dyaw])
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
        """在 360° 范围内选择最佳探索方向，强烈偏向最近未访问前沿。"""
        current_yaw = self.robot.base_yaw
        best_yaw = None
        best_score = -1e9

        frontier_yaw = self._nearest_frontier_yaw()

        for delta in np.linspace(-np.pi, np.pi, 25):
            yaw = current_yaw + delta
            probe = self._probe_point(yaw, distance=self.robot.length + 0.6)
            clear = (
                self.env.inside_bounds(probe)
                and not self._predict_collision(probe)
            )
            # 偏向未访问区域
            cell = self._pos_to_cell(probe, use_height_level=True)
            visited_penalty = 5.0 if cell in self.visited else 0.0
            # 偏向最近未访问前沿方向
            frontier_bonus = 0.0
            if frontier_yaw is not None:
                angle_diff = abs(self._angle_diff(frontier_yaw, yaw))
                frontier_bonus = 3.0 * max(0.0, 1.0 - angle_diff / math.radians(60.0))
            # 偏向仓库中心，避免在角落打转
            to_center = -self.robot.base_position[:2]
            to_center_norm = float(np.linalg.norm(to_center))
            if to_center_norm > 1e-6:
                center_dir = to_center / to_center_norm
                center_attraction = 2.0 * np.dot(
                    np.array([np.cos(yaw), np.sin(yaw)]),
                    center_dir,
                )
            else:
                center_attraction = 0.0
            # 边界排斥：越接近边界越不喜欢朝边界走
            boundary_margin = 0.6
            probe_dist_to_bounds = min(
                self.env.bounds_high[0] - probe[0],
                probe[0] - self.env.bounds_low[0],
                self.env.bounds_high[1] - probe[1],
                probe[1] - self.env.bounds_low[1],
            )
            if probe_dist_to_bounds < boundary_margin:
                boundary_penalty = 2.0 * (1.0 - probe_dist_to_bounds / boundary_margin)
            else:
                boundary_penalty = 0.0
            score = (1.0 if clear else -10.0) - abs(delta) * 0.2 - visited_penalty + frontier_bonus + center_attraction - boundary_penalty
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
        """四角传感器触发：移动探测器（末端传感器）向浓度最高的固定探测器移动。

        小车仅做辅助性平移/旋转，主要依靠机械臂将末端传感器伸向浓度最高的
        角传感器方向，直到末端传感器也检测到气味。
        """
        if not corner_ppm:
            return np.zeros(3), np.zeros(6)

        best_corner = max(corner_ppm, key=corner_ppm.get)
        sensor_offsets = self.robot.base_cfg["corner_sensors"]
        corner_local = np.array(sensor_offsets[best_corner], dtype=float)

        # 移动探测器目标：最高浓度固定探测器的世界坐标，并沿该方向外推一段
        corner_world = self.robot.corner_sensor_positions()[best_corner]
        outward = corner_world - self.robot.base_position[:3]
        outward[2] = 0.0
        outward_norm = float(np.linalg.norm(outward))
        if outward_norm > 1e-6:
            outward = outward / outward_norm
        else:
            outward = np.array([1.0, 0.0, 0.0])

        # 末端目标：固定探测器位置再向外延伸 0.2 m，高度使用当前扫描高度
        current_ee_z = self.robot.ee_sensor_position()[2]
        target_z = max(0.3, min(1.3, current_ee_z))
        self._target_point = corner_world + 0.2 * outward
        self._target_point[2] = target_z

        # 机械臂指向目标
        joint_delta = self._point_ee_to_target(self._target_point)

        # 小车辅助：向该角方向平移并调整朝向（速度提高以更快接近源）
        yaw = self.robot.base_yaw
        angle = math.atan2(corner_local[1], corner_local[0])

        # 将角传感器局部方向转到世界坐标系
        world_angle = yaw + angle
        world_dir = np.array([np.cos(world_angle), np.sin(world_angle)])

        # 探测前方是否畅通
        probe = self._probe_point(world_angle, distance=self.robot.length + 0.5)
        front_clear = (
            self.env.inside_bounds(probe) and not self._predict_collision(probe)
        )

        speed = self.random_walk_speed * 1.0 if front_clear else self.random_walk_speed * 0.3
        dx = speed * np.cos(angle)
        dy = speed * np.sin(angle)
        dyaw = float(np.clip(0.3 * angle, -self.robot.max_yaw_delta, self.robot.max_yaw_delta))

        base_cmd = np.array([dx, dy, dyaw])
        return base_cmd, joint_delta

    def _arm_tracking_action(
        self,
        ee_ppm: float,
        visual_confirmed: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """末端传感器触发：纯局部梯度搜索，小车主动跟进。

        约束：不使用真实源位置。仅根据当前末端浓度梯度决定移动方向；
        基座跟进增益随浓度升高而降低，避免冲过头；
        高浓度但长时间未视觉确认时，基座会缓慢旋转以主动扫描视觉源。
        """
        ee_pos = self.robot.ee_sensor_position()

        # 记录历史用于局部极值检测
        self._recent_ee_ppm.append(float(ee_ppm))
        self._recent_ee_positions.append(ee_pos.copy())
        self._recent_base_positions.append(self.robot.base_position[:2].copy())
        if len(self._recent_ee_ppm) > self._local_max_window:
            self._recent_ee_ppm.pop(0)
            self._recent_ee_positions.pop(0)
            self._recent_base_positions.pop(0)

        # 估计局部浓度梯度
        gradient = self._estimate_gradient(ee_pos)
        grad_norm = float(np.linalg.norm(gradient))
        if grad_norm > 1e-6:
            direction = gradient / grad_norm
            self._last_gradient = direction.copy()
        else:
            # 梯度不明显时保持上一步方向，避免原地抖动
            direction = self._last_gradient.copy()
            if np.linalg.norm(direction) < 1e-6:
                # 初始无方向时沿当前偏航前进
                yaw = self.robot.base_yaw
                direction = np.array([np.cos(yaw), np.sin(yaw), 0.0])

        self._target_point = ee_pos + self.gradient_step * direction

        # 小车主动向梯度方向跟进（只在水平面）
        yaw = self.robot.base_yaw
        grad_xy = gradient.copy()
        grad_xy[2] = 0.0
        grad_xy_norm = float(np.linalg.norm(grad_xy))

        if grad_xy_norm > 1e-6:
            world_dir = grad_xy / grad_xy_norm
        else:
            world_dir = np.array([np.cos(yaw), np.sin(yaw), 0.0])

        local_x = world_dir[0] * np.cos(yaw) + world_dir[1] * np.sin(yaw)
        local_y = -world_dir[0] * np.sin(yaw) + world_dir[1] * np.cos(yaw)

        # 根据浓度动态调整基座增益：浓度越高越谨慎
        if ee_ppm > 5000.0:
            base_gain = 0.01
        elif ee_ppm > 1000.0:
            base_gain = 0.02
        elif ee_ppm > 100.0:
            base_gain = 0.04
        else:
            base_gain = 0.07

        dx = float(np.clip(local_x * base_gain, -base_gain, base_gain))
        dy = float(np.clip(local_y * base_gain, -base_gain, base_gain))
        dyaw = float(np.clip(0.2 * math.atan2(local_y, local_x), -0.08, 0.08))

        # 高浓度但未视觉确认时，主动旋转基座扫描视觉源
        if (
            ee_ppm > 50.0 * self.env.detection_threshold
            and not visual_confirmed
        ):
            self._visual_scan_steps += 1
            scan_period = 80
            scan_phase = (self._visual_scan_steps % scan_period) / scan_period
            # 先顺时针再逆时针，幅值逐渐增大
            scan_yaw = 0.12 * math.sin(2.0 * math.pi * scan_phase) * (1.0 + 0.5 * scan_phase)
            dyaw += float(np.clip(scan_yaw, -0.15, 0.15))
        else:
            self._visual_scan_steps = 0

        base_cmd = np.array([dx, dy, dyaw])

        # 障碍物规避：检查按当前命令移动后的基座位置是否会碰撞
        if self._predict_arm_tracking_collision(base_cmd):
            # 尝试降低速度或侧向偏移
            base_cmd = self._avoid_obstacle_in_arm_tracking(base_cmd, world_dir)

        # 机械臂指向目标
        joint_delta = self._point_ee_to_target(self._target_point)

        return base_cmd, joint_delta

    def _predict_arm_tracking_collision(self, base_cmd: np.ndarray) -> bool:
        """预测 ARM_TRACKING 中执行 base_cmd 后是否会碰撞。"""
        yaw = self.robot.base_yaw
        dx_world = (
            base_cmd[0] * np.cos(yaw) - base_cmd[1] * np.sin(yaw),
            base_cmd[0] * np.sin(yaw) + base_cmd[1] * np.cos(yaw),
        )
        next_pos = self.robot.base_position[:3].copy()
        next_pos[0] += dx_world[0]
        next_pos[1] += dx_world[1]
        next_pos[2] = 0.0
        return self._predict_collision(next_pos)

    def _avoid_obstacle_in_arm_tracking(
        self,
        base_cmd: np.ndarray,
        world_dir: np.ndarray,
    ) -> np.ndarray:
        """ARM_TRACKING 中遇到障碍物时，尝试降低速度或做侧向偏移。"""
        # 先尝试降低速度
        slowed = base_cmd.copy()
        slowed[0] *= 0.3
        slowed[1] *= 0.3
        if not self._predict_arm_tracking_collision(slowed):
            return slowed

        # 再尝试向左右侧向偏移
        yaw = self.robot.base_yaw
        for side in [-1.0, 1.0]:
            offset = side * 0.15
            test_cmd = base_cmd.copy()
            test_cmd[0] = base_cmd[0] * 0.3
            test_cmd[1] = offset
            if not self._predict_arm_tracking_collision(test_cmd):
                return test_cmd

        # 仍然碰撞，则只保留原地旋转
        return np.array([0.0, 0.0, base_cmd[2]])

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

    def _joint_delta_to_target_with_scan(
        self,
        target_deg: np.ndarray,
    ) -> np.ndarray:
        """在目标关节角基础上叠加小幅末端 z 方向扫描，提高烟羽高度匹配概率。

        计算当前层预设姿态对应的末端位置，在其 z 方向叠加正弦扫描后，
        用数值 IK 求关节增量。扫描幅值 ±5 cm，周期约 42 步。
        """
        target_deg = np.asarray(target_deg, dtype=float)
        original_joints = self.robot.joint_angles.copy()

        # 临时设置关节为目标预设，计算对应末端位置
        self.robot.joint_angles = deg2rad(target_deg)
        self.robot._compute_fk()
        preset_ee = self.robot.ee_sensor_position().copy()

        # 恢复原始关节角
        self.robot.joint_angles = original_joints
        self.robot._compute_fk()

        # 末端 z 方向小幅扫描
        self._arm_scan_phase += 0.15
        z_scan = 0.05 * math.sin(self._arm_scan_phase)
        scan_target = preset_ee.copy()
        scan_target[2] = max(
            self.env.bounds_low[2] + 0.05,
            min(self.env.bounds_high[2] - 0.05, scan_target[2] + z_scan),
        )

        # 混合：70% 趋向预设关节角 + 30% 趋向扫描后的末端位置
        delta_preset = self._joint_delta_to_target(target_deg)
        delta_scan = self._point_ee_to_target(scan_target)
        return 0.7 * delta_preset + 0.3 * delta_scan

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

    def _pos_to_cell(
        self,
        pos: np.ndarray,
        use_height_level: bool = False,
    ) -> tuple[int, int, int]:
        """将位置映射到覆盖网格单元。

        当 use_height_level=True 时，z 维度使用当前高度层索引，
        以便在不同机械臂高度层区分访问记录。
        """
        x, y, z = pos[:3]
        if use_height_level:
            z = float(self._current_height_level)
        return (
            int(round(x / self.cell_size)),
            int(round(y / self.cell_size)),
            int(round(z / self.cell_size)),
        )

    def _should_declare_finished(
        self,
        ee_ppm: float,
        visual_confirmed: bool = False,
        visual_distance: float = float("inf"),
    ) -> bool:
        """判断是否满足结束条件并声明找到气味源。

        本版本改为嗅觉主导：当末端浓度持续高于阈值、末端位于局部浓度高点、
        且基座已收敛（位移很小）时，即可声明成功。视觉仅在相机明确看到源实体
        且距离很近时作为辅助加速条件。不使用真实源位置或源强度。
        """
        # 嗅觉主导条件
        if ee_ppm >= self._odor_finish_min_ppm:
            self._odor_finish_counter += 1
        else:
            self._odor_finish_counter = 0

        odor_ok = False
        if self._odor_finish_counter >= self._odor_finish_required:
            ee_pos = self.robot.ee_sensor_position()
            peak = self._is_local_concentration_peak(ee_pos)
            stable = self._recent_base_displacement() < self._odor_finish_displacement_threshold
            # 源高度通常在 0.5 m 以上，避免在低处 filament 误结束
            height_ok = ee_pos[2] >= 0.40
            if peak and stable and height_ok:
                odor_ok = True

        # 视觉辅助条件（放宽距离阈值，不作为唯一条件）
        visual_ok = (
            visual_confirmed
            and visual_distance < self._visual_distance_threshold
            and ee_ppm > 10.0 * self.env.detection_threshold
        )
        if visual_ok:
            self._visual_confirm_steps += 1
        else:
            self._visual_confirm_steps = max(0, self._visual_confirm_steps - 1)

        return odor_ok or (
            visual_ok and self._visual_confirm_steps >= self._visual_confirm_required
        )

    def _is_local_concentration_peak(self, ee_pos: np.ndarray) -> bool:
        """判断末端位置是否为局部浓度高点（不使用真实源位置）。

        在 ±x、±y、±z 六个方向采样，若 ee 处浓度不低于所有相邻点
        （允许小幅噪声容差），则认为当前位置位于局部峰值。
        """
        delta = self._odor_finish_peak_delta
        positions: dict[str, np.ndarray] = {"center": np.asarray(ee_pos, dtype=float)}
        for axis in range(3):
            for sign in (-1.0, 1.0):
                p = np.asarray(ee_pos, dtype=float).copy()
                p[axis] += sign * delta
                positions[f"{axis}_{sign}"] = p

        conc = self.env.query_sensors(positions)
        c_center = conc["center"]
        # 容差：取中心浓度的 5% 或 1.0 ppm 中较大者，避免噪声导致误判
        tol = max(1.0, 0.05 * abs(c_center))
        for name, c in conc.items():
            if name == "center":
                continue
            if c_center < c - tol:
                return False
        return True

    def _recent_base_displacement(self) -> float:
        """返回最近窗口期内基座的水平位移。"""
        if len(self._odor_finish_base_positions) < self._odor_finish_window:
            return float("inf")
        positions = np.array(self._odor_finish_base_positions)
        return float(np.linalg.norm(positions[-1] - positions[0]))

    def _is_local_maximum_stuck(self, ee_ppm: float) -> bool:
        """检测是否陷入局部高浓度平台但并未真正接近源。

        条件：
        - 末端浓度持续高于阈值；
        - 在最近窗口期内，浓度变化很小（平台）；
        - 末端位置变化很小（机械臂够到高浓度区但基座没跟上）。
        """
        if len(self._recent_ee_ppm) < self._local_max_window:
            return False
        if ee_ppm < self._local_max_ppm_threshold:
            return False

        # 若浓度已接近嗅觉结束阈值，给嗅觉主导结束条件留出时间，
        # 避免把真实源附近的高浓度平台误判为 filament。
        if ee_ppm >= 0.8 * self._odor_finish_min_ppm:
            return False

        ppm_values = np.array(self._recent_ee_ppm)
        positions = np.array(self._recent_ee_positions)
        # 浓度平台：最大值与最小值之差小于均值的 20%
        ppm_range = float(np.max(ppm_values) - np.min(ppm_values))
        ppm_mean = float(np.mean(ppm_values))
        plateau = ppm_range < 0.2 * max(ppm_mean, 1.0)

        # 基座位置几乎没动（用末端位置会受机械臂摆动干扰）
        base_positions = np.array(self._recent_base_positions)
        base_displacement = float(np.linalg.norm(base_positions[-1] - base_positions[0]))
        stuck = base_displacement < 0.20

        return plateau and stuck

    def _add_no_track_zone(self, base_xy: np.ndarray) -> None:
        """将当前基座位置加入不重新进入追踪的区域列表。"""
        self._no_track_zones.append((np.asarray(base_xy, dtype=float).copy(), self._no_track_zone_radius))

    def _is_in_no_track_zone(self, base_xy: np.ndarray) -> bool:
        """判断基座位置是否位于某个不追踪区域内。"""
        p = np.asarray(base_xy, dtype=float)
        for center, radius in self._no_track_zones:
            if float(np.linalg.norm(p - center)) < radius:
                return True
        return False

    @staticmethod
    def _angle_diff(target: float, current: float) -> float:
        """计算 target - current 并归一化到 [-pi, pi]。"""
        diff = target - current
        while diff > np.pi:
            diff -= 2.0 * np.pi
        while diff < -np.pi:
            diff += 2.0 * np.pi
        return diff
