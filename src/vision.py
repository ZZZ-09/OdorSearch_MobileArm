"""简化视觉传感器模型。

参考 OdorSim 中机械臂末端相机的使用方式，本模块提供一个基于几何的
虚拟视觉检测器：
- 判断气味源（实体）是否在相机视野内；
- 判断是否在有效探测距离内；
- 判断视线是否被障碍物遮挡。

输出：
    {
        "source_in_view": bool,
        "source_visible": bool,
        "distance": float,
        "bbox": (u, v) 图像坐标或 None,
    }
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.environment import WarehouseEnv
from src.robot import MobileArmRobot
from src.utils import segment_box_collision, segment_cylinder_collision


class VisualSensor:
    """机械臂末端视觉相机。

    Args:
        robot: 机器人模型。
        env: 环境模型。
        config: 视觉配置字典；若为 None 则使用 robot.arm["camera"]。
    """

    def __init__(
        self,
        robot: MobileArmRobot,
        env: WarehouseEnv,
        config: "dict[str, Any] | None" = None,
    ):
        self.robot = robot
        self.env = env
        if config is None:
            config = robot.arm_cfg["camera"]
        self.cfg = config

        self.offset = np.array(config["offset"], dtype=float)
        self.fov_h = math.radians(float(config["fov_horizontal_deg"]))
        self.fov_v = math.radians(float(config["fov_vertical_deg"]))
        self.max_range = float(config["max_range"])
        self.source_radius = float(config["source_visible_radius"])

    def get_camera_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """返回相机在世界坐标系中的 (位置, 旋转矩阵)。"""
        flange_pos, R_flange = self.robot.end_effector_pose()
        cam_pos = flange_pos + R_flange @ self.offset
        return cam_pos, R_flange

    def detect_source(self) -> dict[str, Any]:
        """检测气味源是否在视野内且可见。"""
        cam_pos, R_cam = self.get_camera_pose()
        to_source = self.env.source_pos - cam_pos
        distance = float(np.linalg.norm(to_source))

        result = {
            "source_in_view": False,
            "source_visible": False,
            "distance": distance,
            "image_xy": None,
            "occluded": False,
        }

        if distance > self.max_range or distance < 1e-6:
            return result

        # 相机坐标系：z 轴为光轴，x 向右，y 向下（pinhole 惯例）
        # R_cam 的列是世界坐标系轴在相机坐标系中的表示，所以
        # 点在相机坐标系中的坐标为 R_cam^T @ (point - cam_pos)
        p_cam = R_cam.T @ to_source
        if p_cam[2] <= 0.0:
            return result  # 在相机后方

        # 检查视野
        u = p_cam[0] / p_cam[2]
        v = p_cam[1] / p_cam[2]
        in_horizontal = abs(u) <= math.tan(self.fov_h / 2.0)
        in_vertical = abs(v) <= math.tan(self.fov_v / 2.0)
        result["source_in_view"] = in_horizontal and in_vertical
        result["image_xy"] = (float(u), float(v))

        if not result["source_in_view"]:
            return result

        # 检查遮挡：从相机到气味源连线是否与障碍物相交
        # 同时考虑源实体半径，检查连线上多个点
        occluded = self._is_occluded(cam_pos, self.env.source_pos)
        result["occluded"] = occluded
        result["source_visible"] = not occluded

        return result

    def _is_occluded(
        self,
        cam_pos: np.ndarray,
        source_pos: np.ndarray,
    ) -> bool:
        """判断 camera->source 的视线是否被任何障碍物遮挡。

        对源实体表面附近采样若干点，若任一点被遮挡则视为遮挡。
        """
        # 采样源实体表面上的点
        samples = [source_pos]
        # 在源周围加几个偏移点
        dirs = [
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1],
        ]
        for d in dirs:
            samples.append(source_pos + self.source_radius * np.array(d, dtype=float))

        for target in samples:
            if self._segment_occluded(cam_pos, target):
                return True
        return False

    def _segment_occluded(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
    ) -> bool:
        """判断线段 p1-p2 是否与任何障碍物相交。

        若相交点在 source 附近（<= source_radius），则忽略（那是源本身）。
        """
        p1 = np.asarray(p1, dtype=float)
        p2 = np.asarray(p2, dtype=float)
        seg_len = np.linalg.norm(p2 - p1)
        if seg_len < 1e-9:
            return False

        for obs in self.env.obstacles:
            # 忽略靠近源本身的障碍物（源不是障碍物）
            if np.linalg.norm(obs["center"] - self.env.source_pos) < self.source_radius * 2.0:
                continue

            if obs["type"] == "box":
                if segment_box_collision(p1, p2, obs["center"], obs["size"]):
                    return True
            elif obs["type"] == "cylinder":
                axis = obs["axis"] / (np.linalg.norm(obs["axis"]) + 1e-12)
                if segment_cylinder_collision(
                    p1, p2, obs["center"], axis, obs["radius"], obs["length"]
                ):
                    return True
        return False
