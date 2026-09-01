"""UMR 移动小车 + RM65-6F-V 机械臂模型。

包含：
- 车体与关节状态管理
- 基于 Modified D-H 参数的正运动学
- 五个气味传感器（4 角固定 + 末端跟随）的世界坐标计算
- 车体/机械臂与障碍物的碰撞检测
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.utils import (
    config_dir,
    deg2rad,
    euler_to_rotation,
    load_yaml,
    transform_point,
)


class MobileArmRobot:
    """UMR + RM65-6F-V 复合机器人模型。

    Args:
        config: 机器人参数字典；若为 None 则自动加载 config/robot.yaml。
    """

    def __init__(self, config: "dict[str, Any] | None" = None):
        if config is None:
            config = load_yaml(config_dir() / "robot.yaml")
        self.cfg = config
        self.base_cfg = config["mobile_base"]
        self.arm_cfg = config["arm"]

        # 车体尺寸
        self.length = float(self.base_cfg["length"])
        self.width = float(self.base_cfg["width"])
        self.height = float(self.base_cfg["height"])
        self.ground_clearance = float(self.base_cfg["ground_clearance"])

        # 运动限制
        self.max_pos_delta = float(self.base_cfg["max_position_delta"])
        self.max_yaw_delta = float(self.base_cfg["max_yaw_delta"])

        # 机械臂 MDH 参数（转换为米和弧度）
        self.mdh = []
        for row in self.arm_cfg["mdh"]:
            self.mdh.append(
                {
                    "a": row["a"] / 1000.0,
                    "alpha": deg2rad(row["alpha"]),
                    "d": row["d"] / 1000.0,
                    "theta_offset": deg2rad(row["theta_offset"]),
                }
            )

        # 关节限位（弧度）与速度限制（度/秒）
        self.joint_limits = {}
        self.joint_speed_limits = {}
        for i, (k, v) in enumerate(self.arm_cfg["joint_limits_deg"].items(), start=1):
            self.joint_limits[i] = (deg2rad(v[0]), deg2rad(v[1]))
            self.joint_speed_limits[i] = float(self.arm_cfg["max_joint_speed"][k])

        # 末端传感器偏移
        self.ee_sensor_offset = np.array(self.arm_cfg["ee_sensor_offset"], dtype=float)

        # 初始化状态
        self.reset()

    # ---------------------------------------------------------------------- #
    # 状态管理
    # ---------------------------------------------------------------------- #
    def reset(
        self,
        base_pose: "np.ndarray | None" = None,
        joint_angles: "np.ndarray | None" = None,
    ) -> None:
        """重置机器人为初始状态。"""
        if base_pose is None:
            base_pose = np.array([-7.0, -6.5, 0.0, 0.0], dtype=float)
        if joint_angles is None:
            joint_angles = np.array([0.0, 60.0, 70.0, 0.0, 110.0, 0.0], dtype=float)

        self.base_pose = np.asarray(base_pose, dtype=float).copy()
        self.joint_angles = np.deg2rad(
            np.asarray(joint_angles, dtype=float).copy()
        )

        # 缓存正运动学结果
        self._fk_cache: "dict[str, Any] | None" = None
        self._compute_fk()

    @property
    def base_position(self) -> np.ndarray:
        return self.base_pose[:3]

    @property
    def base_yaw(self) -> float:
        return float(self.base_pose[3])

    @property
    def num_joints(self) -> int:
        return self.arm_cfg["dof"]

    # ---------------------------------------------------------------------- #
    # 正运动学
    # ---------------------------------------------------------------------- #
    def _compute_fk(self) -> None:
        """计算机械臂各关节在世界坐标系中的位姿。"""
        yaw = self.base_yaw
        R_base = euler_to_rotation(0.0, 0.0, yaw)
        t_base = self.base_position + np.array(
            self.arm_cfg["base_offset"], dtype=float
        )

        # 齐次变换矩阵列表
        T_list = []
        T = np.eye(4)
        T[:3, :3] = R_base
        T[:3, 3] = t_base

        for i, mdh in enumerate(self.mdh, start=1):
            theta = self.joint_angles[i - 1] + mdh["theta_offset"]
            a = mdh["a"]
            alpha = mdh["alpha"]
            d = mdh["d"]

            ct, st = np.cos(theta), np.sin(theta)
            ca, sa = np.cos(alpha), np.sin(alpha)

            A_i = np.array(
                [
                    [ct, -st, 0.0, a],
                    [st * ca, ct * ca, -sa, -sa * d],
                    [st * sa, ct * sa, ca, ca * d],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=float,
            )
            T = T @ A_i
            T_list.append(T.copy())

        # 法兰盘位姿
        T_flange = T_list[-1]
        # 末端传感器位姿
        T_ee = T_flange.copy()
        T_ee[:3, 3] = T_flange[:3, :3] @ self.ee_sensor_offset + T_flange[:3, 3]

        self._fk_cache = {
            "R_base": R_base,
            "t_base": t_base,
            "T_list": T_list,
            "T_flange": T_flange,
            "T_ee": T_ee,
        }

    # ---------------------------------------------------------------------- #
    # 传感器位置
    # ---------------------------------------------------------------------- #
    def corner_sensor_positions(self) -> dict[str, np.ndarray]:
        """返回四个角固定气味传感器的世界坐标。"""
        R_base = euler_to_rotation(0.0, 0.0, self.base_yaw)
        t_base = self.base_position
        positions = {}
        for name, offset in self.base_cfg["corner_sensors"].items():
            positions[name] = transform_point(offset, t_base, R_base)
        return positions

    def ee_sensor_position(self) -> np.ndarray:
        """返回机械臂末端气味传感器的世界坐标。"""
        if self._fk_cache is None:
            self._compute_fk()
        return self._fk_cache["T_ee"][:3, 3].copy()

    def end_effector_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """返回末端法兰盘的 (位置, 旋转矩阵)。"""
        if self._fk_cache is None:
            self._compute_fk()
        T = self._fk_cache["T_flange"]
        return T[:3, 3].copy(), T[:3, :3].copy()

    def all_sensor_positions(self) -> dict[str, np.ndarray]:
        """返回全部 5 个传感器的世界坐标。"""
        out = self.corner_sensor_positions()
        out["ee"] = self.ee_sensor_position()
        return out

    # ---------------------------------------------------------------------- #
    # 控制接口
    # ---------------------------------------------------------------------- #
    def apply_base_command(
        self,
        dx: float,
        dy: float,
        dyaw: float,
        *,
        check_bounds: "tuple[np.ndarray, np.ndarray] | None" = None,
    ) -> bool:
        """根据车体坐标系下的速度命令更新小车位置。

        返回值：True 表示命令已执行；False 表示因越界被裁剪。
        """
        dx = float(np.clip(dx, -self.max_pos_delta, self.max_pos_delta))
        dy = float(np.clip(dy, -self.max_pos_delta, self.max_pos_delta))
        dyaw = float(np.clip(dyaw, -self.max_yaw_delta, self.max_yaw_delta))

        yaw = self.base_yaw
        # 将车体坐标系下的 delta 转到世界坐标系
        world_dx = dx * np.cos(yaw) - dy * np.sin(yaw)
        world_dy = dx * np.sin(yaw) + dy * np.cos(yaw)

        new_pose = self.base_pose.copy()
        new_pose[0] += world_dx
        new_pose[1] += world_dy
        new_pose[3] = self._normalize_angle(new_pose[3] + dyaw)

        # 边界检查
        if check_bounds is not None:
            low, high = check_bounds
            if np.any(new_pose[:3] < low) or np.any(new_pose[:3] > high):
                return False

        self.base_pose = new_pose
        self._fk_cache = None
        return True

    def apply_joint_command(self, delta_deg: np.ndarray) -> bool:
        """按给定的关节角度增量（度）更新机械臂。"""
        delta_deg = np.asarray(delta_deg, dtype=float)
        if delta_deg.size != self.num_joints:
            raise ValueError(
                f"joint command size {delta_deg.size} != {self.num_joints}"
            )

        new_angles = self.joint_angles.copy()
        for i in range(self.num_joints):
            jid = i + 1
            d = deg2rad(delta_deg[i])
            # 按最大速度裁剪
            max_d = deg2rad(self.joint_speed_limits[jid] * 0.1)  # 假设 dt=0.1s
            d = float(np.clip(d, -max_d, max_d))
            new_angles[i] = float(
                np.clip(new_angles[i] + d, self.joint_limits[jid][0], self.joint_limits[jid][1])
            )

        self.joint_angles = new_angles
        self._fk_cache = None
        return True

    def set_joint_angles(self, angles_deg: np.ndarray) -> None:
        """直接设置关节角度（度），并裁剪到关节限位。"""
        angles_deg = np.asarray(angles_deg, dtype=float)
        new = np.deg2rad(angles_deg)
        for i in range(self.num_joints):
            jid = i + 1
            new[i] = float(np.clip(new[i], self.joint_limits[jid][0], self.joint_limits[jid][1]))
        self.joint_angles = new
        self._fk_cache = None

    # ---------------------------------------------------------------------- #
    # 碰撞检测
    # ---------------------------------------------------------------------- #
    def base_collision_boxes(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """返回车体简化碰撞盒：[(center, size), ...]。

        将车体近似为两个 box：底盘 + 安装立柱（arm base）。
        """
        yaw = self.base_yaw
        R_base = euler_to_rotation(0.0, 0.0, yaw)
        center = self.base_position.copy()
        center[2] = self.height / 2.0

        # 主体底盘
        chassis = (center, np.array([self.length, self.width, self.height]))

        # 机械臂安装立柱（圆柱/方柱），简化成包围盒
        arm_base_offset = np.array(self.arm_cfg["base_offset"], dtype=float)
        arm_base_world = transform_point(arm_base_offset, self.base_position, R_base)
        arm_base_world[2] = self.height + 0.05
        pillar = (arm_base_world, np.array([0.15, 0.15, 0.10]))

        return [chassis, pillar]

    def arm_links_positions(self, n_samples: int = 8) -> list[np.ndarray]:
        """沿机械臂连杆采样若干点，用于粗略碰撞检测。"""
        if self._fk_cache is None:
            self._compute_fk()
        T_list = self._fk_cache["T_list"]
        points = []
        for T in T_list:
            points.append(T[:3, 3].copy())
        # 在线段上插值
        sampled = []
        for i in range(len(points) - 1):
            for t in np.linspace(0.0, 1.0, n_samples):
                sampled.append(points[i] + t * (points[i + 1] - points[i]))
        sampled.extend(points)
        return sampled

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """将角度归一化到 [-pi, pi]。"""
        while angle > np.pi:
            angle -= 2.0 * np.pi
        while angle < -np.pi:
            angle += 2.0 * np.pi
        return angle
