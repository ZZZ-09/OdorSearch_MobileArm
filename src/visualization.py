"""三维可视化工具。

使用 matplotlib 的 3D 绘图能力绘制：
- 仓库边界、障碍物、管道；
- 气味源与气味浓度等值面（可选）；
- 机器人轨迹、小车底盘、机械臂连杆、传感器位置。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D, proj3d

from src.environment import WarehouseEnv
from src.robot import MobileArmRobot


class WarehouseVisualizer:
    """仓库环境 + 机器人可视化器。"""

    def __init__(self, env: WarehouseEnv, robot: MobileArmRobot):
        self.env = env
        self.robot = robot
        self.fig = plt.figure(figsize=(14, 10))
        self.ax: Axes3D = self.fig.add_subplot(111, projection="3d")
        self._setup_axes()

    def _setup_axes(self) -> None:
        low, high = self.env.bounds_low, self.env.bounds_high
        self.ax.set_xlim(low[0], high[0])
        self.ax.set_ylim(low[1], high[1])
        self.ax.set_zlim(low[2], high[2])
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_zlabel("Z (m)")
        self.ax.set_title("3D Warehouse Odor Source Search")

    def draw_static_environment(self) -> None:
        """绘制仓库静态结构（只调用一次）。"""
        self.ax.clear()
        self._setup_axes()

        # 绘制障碍物（使用更明显的颜色/透明度，确保在视频中清晰可见）
        for obs in self.env.obstacles:
            if obs["name"].endswith("_wall"):
                continue
            if obs["type"] == "box":
                self._draw_box(obs["center"], obs["size"], color="darkred", alpha=0.55, linewidth=1.5)
                self._draw_box_surface(obs["center"], obs["size"], color="darkred", alpha=0.18)
            elif obs["type"] == "cylinder":
                self._draw_cylinder(
                    obs["center"], obs["axis"], obs["radius"], obs["length"],
                    color="darkblue", alpha=0.55, linewidth=1.5,
                )
                self._draw_cylinder_surface(
                    obs["center"], obs["axis"], obs["radius"], obs["length"],
                    color="darkblue", alpha=0.18,
                )

        # 绘制气味源实体（红色球体）
        self._draw_sphere(self.env.source_pos, self.env.source_radius, color="red")
        self.ax.scatter(
            *self.env.source_pos,
            color="red",
            s=200,
            marker="*",
            label="odor source",
        )

        # 绘制气味浓度等值面（在 source 附近采样）
        self._draw_concentration_cloud()

    def draw_robot(self, trajectory: "np.ndarray | None" = None) -> None:
        """绘制机器人当前状态与历史轨迹。"""
        # 车体
        self._draw_robot_base()

        # 机械臂连杆
        self._draw_arm()

        # 传感器
        self._draw_sensors()

        # 轨迹
        if trajectory is not None and len(trajectory) > 0:
            self.ax.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                trajectory[:, 2],
                "g-",
                alpha=0.5,
                linewidth=1.5,
                label="base trajectory",
            )

    def _draw_robot_base(self) -> None:
        """绘制小车底盘包围盒。"""
        center = self.robot.base_position.copy()
        center[2] = self.robot.height / 2.0
        self._draw_box(
            center,
            np.array([self.robot.length, self.robot.width, self.robot.height]),
            color="darkorange",
            alpha=0.7,
        )

    def _draw_arm(self) -> None:
        """绘制机械臂连杆。"""
        if self.robot._fk_cache is None:
            self.robot._compute_fk()
        points = [self.robot._fk_cache["t_base"]]
        for T in self.robot._fk_cache["T_list"]:
            points.append(T[:3, 3].copy())
        points = np.array(points)
        self.ax.plot(points[:, 0], points[:, 1], points[:, 2], "b-o", linewidth=3, markersize=4)

    def _draw_sensors(self) -> None:
        """绘制五个传感器位置。"""
        positions = self.robot.all_sensor_positions()
        pts = np.array(list(positions.values()))
        self.ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color="purple", s=80, label="sensors")

    def _draw_box(
        self,
        center: np.ndarray,
        size: np.ndarray,
        color: str = "gray",
        alpha: float = 0.4,
        linewidth: float = 1.0,
    ) -> None:
        """绘制轴对齐包围盒。"""
        c = np.asarray(center, dtype=float)
        s = np.asarray(size, dtype=float)
        # 8 个角点
        corners = np.array(
            [
                [-1, -1, -1],
                [1, -1, -1],
                [1, 1, -1],
                [-1, 1, -1],
                [-1, -1, 1],
                [1, -1, 1],
                [1, 1, 1],
                [-1, 1, 1],
            ],
            dtype=float,
        )
        corners = c + 0.5 * s * corners

        # 12 条边
        edges = [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ]
        for e in edges:
            pts = corners[e]
            self.ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, alpha=alpha, linewidth=linewidth)

    def _draw_box_surface(
        self,
        center: np.ndarray,
        size: np.ndarray,
        color: str = "gray",
        alpha: float = 0.15,
    ) -> None:
        """绘制轴对齐包围盒的半透明表面，增强视频中障碍物的可见性。"""
        c = np.asarray(center, dtype=float)
        s = np.asarray(size, dtype=float)
        # 六个面的角点索引（逆时针，保证朝外法向大致一致）
        faces = [
            [0, 1, 5, 4],
            [1, 2, 6, 5],
            [2, 3, 7, 6],
            [3, 0, 4, 7],
            [0, 1, 2, 3],
            [4, 5, 6, 7],
        ]
        corners = c + 0.5 * s * np.array(
            [
                [-1, -1, -1],
                [1, -1, -1],
                [1, 1, -1],
                [-1, 1, -1],
                [-1, -1, 1],
                [1, -1, 1],
                [1, 1, 1],
                [-1, 1, 1],
            ],
            dtype=float,
        )
        for face in faces:
            pts = corners[face]
            self.ax.plot_surface(
                pts[:, 0].reshape(2, 2),
                pts[:, 1].reshape(2, 2),
                pts[:, 2].reshape(2, 2),
                color=color,
                alpha=alpha,
                shade=False,
            )

    def _draw_cylinder(
        self,
        center: np.ndarray,
        axis: np.ndarray,
        radius: float,
        length: float,
        color: str = "steelblue",
        alpha: float = 0.4,
        linewidth: float = 1.0,
    ) -> None:
        """绘制圆柱体。"""
        c = np.asarray(center, dtype=float)
        a = np.asarray(axis, dtype=float)
        a = a / (np.linalg.norm(a) + 1e-12)

        # 构造局部坐标系
        if abs(a[2]) < 0.99:
            tmp = np.array([0.0, 0.0, 1.0])
        else:
            tmp = np.array([0.0, 1.0, 0.0])
        u = np.cross(a, tmp)
        u = u / (np.linalg.norm(u) + 1e-12)
        v = np.cross(a, u)

        theta = np.linspace(0.0, 2.0 * np.pi, 30)
        circle = np.array([np.cos(theta), np.sin(theta), np.zeros_like(theta)])
        # 局部到世界
        R = np.column_stack([u, v, a])

        for z in [-length / 2.0, length / 2.0]:
            local = circle.copy()
            local[2, :] = z
            world = (R @ local).T + c
            self.ax.plot(world[:, 0], world[:, 1], world[:, 2], color=color, alpha=alpha, linewidth=linewidth)

        # 画几条母线
        for i in [0, 10, 20]:
            local = circle[:, i]
            p1 = c + R @ np.array([local[0], local[1], -length / 2.0])
            p2 = c + R @ np.array([local[0], local[1], length / 2.0])
            self.ax.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                [p1[2], p2[2]],
                color=color,
                alpha=alpha,
                linewidth=linewidth,
            )

    def _draw_cylinder_surface(
        self,
        center: np.ndarray,
        axis: np.ndarray,
        radius: float,
        length: float,
        color: str = "steelblue",
        alpha: float = 0.15,
    ) -> None:
        """绘制圆柱体半透明侧面，增强视频中圆柱障碍物的可见性。"""
        c = np.asarray(center, dtype=float)
        a = np.asarray(axis, dtype=float)
        a = a / (np.linalg.norm(a) + 1e-12)
        if abs(a[2]) < 0.99:
            tmp = np.array([0.0, 0.0, 1.0])
        else:
            tmp = np.array([0.0, 1.0, 0.0])
        u = np.cross(a, tmp)
        u = u / (np.linalg.norm(u) + 1e-12)
        v = np.cross(a, u)
        R = np.column_stack([u, v, a])

        theta = np.linspace(0.0, 2.0 * np.pi, 30)
        z_vals = np.linspace(-length / 2.0, length / 2.0, 8)
        theta_grid, z_grid = np.meshgrid(theta, z_vals)
        local = np.stack(
            [
                radius * np.cos(theta_grid),
                radius * np.sin(theta_grid),
                z_grid,
            ],
            axis=-1,
        )
        world = np.einsum("ij,mnj->mni", R, local) + c
        self.ax.plot_surface(
            world[..., 0],
            world[..., 1],
            world[..., 2],
            color=color,
            alpha=alpha,
            shade=False,
        )

    def _draw_sphere(
        self,
        center: np.ndarray,
        radius: float,
        color: str = "red",
        alpha: float = 0.6,
    ) -> None:
        """绘制球体（用于表示气味源实体）。"""
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
        self.ax.plot_surface(x, y, z, color=color, alpha=alpha)

    def draw_vision(self, vision_result: dict[str, Any]) -> None:
        """绘制视觉检测结果：相机到气味源的视线。"""
        if vision_result and vision_result.get("source_visible"):
            cam_pos, _ = self.robot.end_effector_pose()
            # 简化：从法兰盘中心到源画一条绿线
            pts = np.array([cam_pos, self.env.source_pos])
            self.ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "g-", linewidth=2, label="vision ray")

    def _draw_concentration_cloud(self) -> None:
        """在气味源附近绘制半透明低浓度点云，展示烟羽大致范围。"""
        src = self.env.source_pos
        # 在以源为中心的网格上采样
        xs = np.linspace(src[0] - 4.0, src[0] + 4.0, 15)
        ys = np.linspace(src[1] - 4.0, src[1] + 4.0, 15)
        zs = np.linspace(max(src[2] - 1.5, 0.1), src[2] + 1.5, 8)
        pts = []
        vals = []
        for x in xs:
            for y in ys:
                for z in zs:
                    p = np.array([x, y, z])
                    c = self.env.concentration_at(p)
                    if c > 5.0 * self.env.detection_threshold:
                        pts.append(p)
                        vals.append(c)
        if pts:
            pts = np.array(pts)
            vals = np.array(vals)
            # 归一化透明度
            alphas = np.clip((vals - vals.min()) / (vals.max() - vals.min() + 1e-9) * 0.3, 0.05, 0.35)
            self.ax.scatter(
                pts[:, 0],
                pts[:, 1],
                pts[:, 2],
                c=vals,
                cmap="Reds",
                s=20,
                alpha=np.mean(alphas),
                label="plume cloud",
            )

    def show(self) -> None:
        self.ax.legend()
        plt.tight_layout()
        plt.show()

    def save(self, path: "str | Path") -> None:
        self.ax.legend()
        plt.tight_layout()
        plt.savefig(path, dpi=150)


class TrajectoryPlotter:
    """离线绘制搜索轨迹与浓度曲线。"""

    def __init__(self, history: list[dict[str, Any]], env: WarehouseEnv):
        self.history = history
        self.env = env

    def plot_summary(self, save_path: "str | Path | None" = None) -> None:
        """绘制轨迹俯视图 + 各传感器浓度曲线。"""
        fig = plt.figure(figsize=(16, 6))

        # 1. 俯视图
        ax1 = fig.add_subplot(131)
        ee_positions = np.array([rec["ee_pos"] for rec in self.history])
        base_positions = np.array([rec["base_pose"][:3] for rec in self.history])

        ax1.plot(base_positions[:, 0], base_positions[:, 1], "b-", label="base")
        ax1.plot(ee_positions[:, 0], ee_positions[:, 1], "r-", label="EE sensor")
        ax1.scatter(*self.env.source_pos[:2], color="red", s=200, marker="*", label="source")

        # 在俯视图中绘制障碍物，便于直观验证路径是否避障
        for obs in self.env.obstacles:
            if obs["name"].endswith("_wall"):
                continue
            if obs["type"] == "box":
                cx, cy = obs["center"][0], obs["center"][1]
                sx, sy = obs["size"][0], obs["size"][1]
                rect = plt.Rectangle(
                    (cx - sx / 2.0, cy - sy / 2.0),
                    sx,
                    sy,
                    color="darkred",
                    alpha=0.35,
                    label="obstacle" if obs["name"] == "random_box_0" else "",
                )
                ax1.add_patch(rect)
            elif obs["type"] == "cylinder":
                circle = plt.Circle(
                    (obs["center"][0], obs["center"][1]),
                    obs["radius"],
                    color="darkblue",
                    alpha=0.35,
                    label="obstacle" if obs["name"] == "random_cyl_0" else "",
                )
                ax1.add_patch(circle)

        ax1.set_xlabel("X (m)")
        ax1.set_ylabel("Y (m)")
        ax1.set_title("Top-down trajectory")
        ax1.legend()
        ax1.axis("equal")
        ax1.grid(True)

        # 2. 高度曲线
        ax2 = fig.add_subplot(132)
        steps = [rec["step"] for rec in self.history]
        ax2.plot(steps, ee_positions[:, 2], "g-", label="EE height")
        ax2.axhline(self.env.source_pos[2], color="red", linestyle="--", label="source height")
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Z (m)")
        ax2.set_title("End-effector height")
        ax2.legend()
        ax2.grid(True)

        # 3. 浓度曲线
        ax3 = fig.add_subplot(133)
        ee_ppm = [rec["sensor_readings"].get("ee", {}).get("ppm", 0.0) for rec in self.history]
        fl_ppm = [rec["sensor_readings"].get("front_left", {}).get("ppm", 0.0) for rec in self.history]
        ax3.plot(steps, ee_ppm, "r-", label="EE")
        ax3.plot(steps, fl_ppm, "c-", label="front_left")
        ax3.axhline(self.env.detection_threshold, color="gray", linestyle="--", label="threshold")
        ax3.set_xlabel("Step")
        ax3.set_ylabel("Concentration (ppm)")
        ax3.set_title("Sensor readings")
        ax3.legend()
        ax3.grid(True)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        else:
            plt.show()
