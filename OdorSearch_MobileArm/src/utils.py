"""通用工具函数：配置读取、坐标变换、碰撞检测。

与 OdorSim 的 `odor_sim.config.frame_map` 类似，本模块提供世界坐标系与
机器人本体坐标系之间的刚体变换，以及仓库障碍物（box / cylinder）的碰撞检测。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml


# --------------------------------------------------------------------------- #
# 配置加载
# --------------------------------------------------------------------------- #
def load_yaml(path: "str | Path") -> dict[str, Any]:
    """加载 YAML 配置文件。"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def config_dir() -> Path:
    """返回项目 config 目录。"""
    return Path(__file__).resolve().parent.parent / "config"


# --------------------------------------------------------------------------- #
# 角度/旋转工具
# --------------------------------------------------------------------------- #
def deg2rad(d: float) -> float:
    return d * math.pi / 180.0


def rad2deg(r: float) -> float:
    return r * 180.0 / math.pi


def clip_angle_deg(a: float, limit: tuple[float, float]) -> float:
    """将角度（度）裁剪到 [low, high] 区间。"""
    return float(np.clip(a, limit[0], limit[1]))


def rot_x(theta: float) -> np.ndarray:
    """绕 X 轴的 3×3 旋转矩阵。"""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def rot_y(theta: float) -> np.ndarray:
    """绕 Y 轴的 3×3 旋转矩阵。"""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def rot_z(theta: float) -> np.ndarray:
    """绕 Z 轴的 3×3 旋转矩阵。"""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def euler_to_rotation(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Z-Y-X 欧拉角（rpy）-> 旋转矩阵。"""
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def transform_point(
    point: np.ndarray,
    translation: np.ndarray,
    rotation: np.ndarray,
) -> np.ndarray:
    """将点从局部坐标系变换到世界坐标系：p_w = R * p_l + t。"""
    return rotation @ np.asarray(point, dtype=float) + np.asarray(translation, dtype=float)


def inverse_transform(
    translation: np.ndarray,
    rotation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """返回逆变换 (R^T, -R^T*t)。"""
    R_inv = rotation.T
    t_inv = -R_inv @ np.asarray(translation, dtype=float)
    return t_inv, R_inv


# --------------------------------------------------------------------------- #
# 碰撞检测
# --------------------------------------------------------------------------- #
def point_in_box(point: np.ndarray, center: np.ndarray, size: np.ndarray) -> bool:
    """判断点是否在轴对齐包围盒内。"""
    p = np.asarray(point, dtype=float)
    c = np.asarray(center, dtype=float)
    s = np.asarray(size, dtype=float)
    return bool(np.all(np.abs(p - c) <= s / 2.0))


def point_in_cylinder(
    point: np.ndarray,
    center: np.ndarray,
    axis: np.ndarray,
    radius: float,
    length: float,
) -> bool:
    """判断点是否在圆柱体内（axis 已归一化）。"""
    p = np.asarray(point, dtype=float)
    c = np.asarray(center, dtype=float)
    a = np.asarray(axis, dtype=float)
    a = a / (np.linalg.norm(a) + 1e-12)
    v = p - c
    proj = np.dot(v, a)
    if abs(proj) > length / 2.0:
        return False
    perp = v - proj * a
    return float(np.linalg.norm(perp)) <= radius


def box_box_collision(
    center_a: np.ndarray,
    size_a: np.ndarray,
    center_b: np.ndarray,
    size_b: np.ndarray,
) -> bool:
    """两个轴对齐包围盒的碰撞检测。"""
    a = np.asarray(center_a, dtype=float)
    b = np.asarray(center_b, dtype=float)
    sa = np.asarray(size_a, dtype=float)
    sb = np.asarray(size_b, dtype=float)
    return bool(np.all(np.abs(a - b) <= (sa + sb) / 2.0))


def segment_box_collision(
    p1: np.ndarray,
    p2: np.ndarray,
    center: np.ndarray,
    size: np.ndarray,
) -> bool:
    """线段与轴对齐包围盒的粗略碰撞检测。

    采用 AABB 包围盒快速排除，再对线段中点采样。
    """
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    c = np.asarray(center, dtype=float)
    s = np.asarray(size, dtype=float)

    # 先判断线段 AABB 与盒子是否相交
    seg_min = np.minimum(p1, p2)
    seg_max = np.maximum(p1, p2)
    box_min = c - s / 2.0
    box_max = c + s / 2.0
    if np.any(seg_max < box_min) or np.any(seg_min > box_max):
        return False

    # 采样检查
    for t in np.linspace(0.0, 1.0, 10):
        if point_in_box(p1 + t * (p2 - p1), c, s):
            return True
    return False


def segment_cylinder_collision(
    p1: np.ndarray,
    p2: np.ndarray,
    center: np.ndarray,
    axis: np.ndarray,
    radius: float,
    length: float,
) -> bool:
    """线段与圆柱体的采样碰撞检测。"""
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    for t in np.linspace(0.0, 1.0, 10):
        if point_in_cylinder(p1 + t * (p2 - p1), center, axis, radius, length):
            return True
    return False
