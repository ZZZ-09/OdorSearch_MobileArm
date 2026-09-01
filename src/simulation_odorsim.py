"""OdorSim/GADEN 版仿真会话管理。

本文件是 ``src/simulation.py`` 的拷贝，唯一改动是在 ``__init__`` 中允许
外部传入已经构造好的环境实例（例如 ``GadenBackedWarehouseEnv``），从而
在不修改原始仿真代码的前提下把 GADEN 气味扩散后端接入进来。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.environment import WarehouseEnv
from src.robot import MobileArmRobot
from src.sensors import OdorSensorArray
from src.search_algorithm import SearchState
from src.odor_sim_adapter import GadenAwareSearchAgent
from src.vision import VisualSensor


class OdorSearchSessionOdorSim:
    """气味源搜索仿真会话（OdorSim/GADEN 后端）。

    Args:
        env: 已构造的环境实例。若提供，则 ``env_config`` 被忽略。
        robot_config: 机器人配置字典或路径。
        env_config: 环境配置字典或路径（仅在 ``env`` 为 None 时使用）。
        seed: 随机种子。
    """

    def __init__(
        self,
        env: "WarehouseEnv | None" = None,
        robot_config: "dict[str, Any] | str | Path | None" = None,
        env_config: "dict[str, Any] | str | Path | None" = None,
        seed: int = 42,
    ):
        from src.utils import load_yaml

        if robot_config is None or isinstance(robot_config, (str, Path)):
            robot_cfg = load_yaml(robot_config or Path(__file__).resolve().parent.parent / "config" / "robot.yaml")
        else:
            robot_cfg = robot_config

        self.seed = int(seed)
        if env is not None:
            self.env = env
        else:
            if env_config is None or isinstance(env_config, (str, Path)):
                env_cfg = load_yaml(env_config or Path(__file__).resolve().parent.parent / "config" / "warehouse.yaml")
            else:
                env_cfg = env_config
            self.env = WarehouseEnv(env_cfg, seed=seed)

        self.robot = MobileArmRobot(robot_cfg)
        self.sensors = OdorSensorArray(seed=seed)
        self.vision = VisualSensor(self.robot, self.env)
        # Use the GADEN-aware agent so gradient estimation is batched into one
        # service call instead of six sequential concentration queries.
        self.agent = GadenAwareSearchAgent(self.robot, self.env, vision=self.vision)

        self.step_count = 0
        self.done = False
        self.history: list[dict[str, Any]] = []

    # ---------------------------------------------------------------------- #
    # 生命周期
    # ---------------------------------------------------------------------- #
    def reset(
        self,
        base_pose: "np.ndarray | None" = None,
        joint_angles: "np.ndarray | None" = None,
    ) -> dict[str, Any]:
        """重置仿真会话。

        若 ``randomize=true``，则每次 reset 会重新生成障碍物与气味源。
        当传入的 base_pose 与障碍物碰撞时，会自动在默认起始区域搜索安全位置。
        """
        self.env.reset(seed=self.seed)

        if base_pose is None:
            base_pose = self._find_safe_start_pose()
        else:
            base_pose = np.asarray(base_pose, dtype=float).copy()
            if not self._is_base_pose_safe(base_pose):
                print(f"[WARN] Provided start pose {base_pose[:3]} collides; searching safe pose...")
                base_pose = self._find_safe_start_pose()

        # 保证是 4 维 [x, y, z, yaw]
        if base_pose.size == 3:
            base_pose = np.array([base_pose[0], base_pose[1], base_pose[2], 0.0], dtype=float)

        self.robot.reset(base_pose=base_pose, joint_angles=joint_angles)
        self.sensors.reset()
        self.vision = VisualSensor(self.robot, self.env)
        self.agent = GadenAwareSearchAgent(self.robot, self.env, vision=self.vision)
        self.step_count = 0
        self.done = False
        self.history = []
        return self._get_observation()

    def step(self) -> tuple[dict[str, Any], bool, dict[str, Any]]:
        """推进仿真一步。

        返回：
            observation: 当前观测。
            done: 是否结束。
            info: 调试信息。
        """
        if self.done:
            return self._get_observation(), True, {"reason": "already_done"}

        # 1. 查询环境浓度
        sensor_positions = self.robot.all_sensor_positions()
        concentrations = self.env.query_sensors(sensor_positions)

        # 2. 传感器读数（含噪声与动态响应）
        sensor_readings = self.sensors.read(concentrations, dt=self.env.dt)

        # 2.5 视觉检测
        vision_result = self.vision.detect_source()

        # 3. 搜索算法决策
        base_cmd, joint_delta, agent_info = self.agent.decide_action(
            sensor_readings, vision_result
        )

        # 4. 执行动作前进行安全性检查：预测执行后是否会碰撞或超限
        safe_base_cmd, safe_joint_delta = self._ensure_action_safe(base_cmd, joint_delta)

        base_ok = self.robot.apply_base_command(
            safe_base_cmd[0],
            safe_base_cmd[1],
            safe_base_cmd[2],
            check_bounds=(self.env.bounds_low, self.env.bounds_high),
        )
        self.robot.apply_joint_command(safe_joint_delta)

        # 5. 碰撞检测（二次确认）
        boxes = self.robot.base_collision_boxes()
        arm_points = self.robot.arm_links_positions()
        collision = self.env.check_collision(boxes, arm_points)

        # 如果仍发生碰撞，说明预测有漏检，执行安全回退
        if collision:
            self.robot.base_pose -= np.array(
                [safe_base_cmd[0], safe_base_cmd[1], 0.0, safe_base_cmd[2]], dtype=float
            )
            self.robot._fk_cache = None

        # 6. 环境时间推进（湍流演化）
        self.env.step()

        self.step_count += 1

        obs = self._get_observation()
        info = {
            "base_cmd": base_cmd,
            "joint_delta": joint_delta,
            "collision": collision,
            "base_ok": base_ok,
            **agent_info,
        }

        # 记录历史
        record = {
            "step": self.step_count,
            "base_pose": self.robot.base_pose.copy(),
            "joint_angles_deg": np.rad2deg(self.robot.joint_angles).copy(),
            "ee_pos": self.robot.ee_sensor_position().copy(),
            "sensor_readings": {
                k: {"ppm": v["ppm"], "voltage": v["voltage"]}
                for k, v in sensor_readings.items()
            },
            "vision": vision_result,
            "state": self.agent.state.name,
            "collision": collision,
            "info": {k: v for k, v in info.items() if k not in ("target_point", "visual_target")},
            "declared_source_position": self.agent.declared_source_position.copy()
            if self.agent.declared_source_position is not None
            else None,
        }
        self.history.append(record)

        # 终止条件
        if self.agent.state == SearchState.FINISHED:
            self.done = True
            info["reason"] = "source_located"
        elif self.step_count >= self.env.max_steps:
            self.done = True
            info["reason"] = "max_steps_reached"
        elif collision:
            # 连续碰撞不直接结束，仅记录；避免小抖动导致任务失败
            pass

        return obs, self.done, info

    def run(self, max_steps: "int | None" = None) -> dict[str, Any]:
        """运行完整搜索任务，直到结束。"""
        if max_steps is None:
            max_steps = self.env.max_steps
        while not self.done and self.step_count < max_steps:
            _, done, _ = self.step()
            if done:
                break
        return self.get_summary()

    # ---------------------------------------------------------------------- #
    # 观测与摘要
    # ---------------------------------------------------------------------- #
    def _get_observation(self) -> dict[str, Any]:
        """构造当前观测。"""
        return {
            "base_pose": self.robot.base_pose.copy(),
            "joint_angles_deg": np.rad2deg(self.robot.joint_angles).copy(),
            "ee_pos": self.robot.ee_sensor_position().copy(),
            "source_pos": self.env.source_pos.copy(),
            "step": self.step_count,
            "state": self.agent.state.name,
        }

    def get_summary(self) -> dict[str, Any]:
        """返回任务摘要。"""
        if not self.history:
            return {"steps": 0, "success": False, "reason": "no_history"}

        final = self.history[-1]
        ee_pos = final["ee_pos"]
        dist = float(np.linalg.norm(ee_pos - self.env.source_pos))
        success = self.agent.state == SearchState.FINISHED

        # 统计各阶段步数
        state_counts = {}
        for rec in self.history:
            state_counts[rec["state"]] = state_counts.get(rec["state"], 0) + 1

        max_ee_ppm = max(
            (rec["sensor_readings"]["ee"]["ppm"] for rec in self.history if "ee" in rec["sensor_readings"]),
            default=0.0,
        )

        return {
            "success": success,
            "steps": self.step_count,
            "final_distance_to_source": dist,
            "final_state": self.agent.state.name,
            "state_counts": state_counts,
            "max_ee_ppm": max_ee_ppm,
            "final_ee_ppm": final["sensor_readings"].get("ee", {}).get("ppm", 0.0),
            "collision_count": sum(1 for rec in self.history if rec["collision"]),
            "declared_source_position": self.agent.declared_source_position.copy()
            if self.agent.declared_source_position is not None
            else None,
        }

    # ---------------------------------------------------------------------- #
    # 安全辅助
    # ---------------------------------------------------------------------- #
    def _is_base_pose_safe(self, base_pose: np.ndarray) -> bool:
        """判断给定基座姿态 [x, y, z, yaw] 是否与障碍物碰撞。"""
        original = self.robot.base_pose.copy()
        pose = np.asarray(base_pose, dtype=float).copy()
        if pose.size == 3:
            pose = np.array([pose[0], pose[1], pose[2], 0.0], dtype=float)
        self.robot.base_pose = pose
        self.robot._fk_cache = None
        boxes = self.robot.base_collision_boxes()
        arm_points = self.robot.arm_links_positions(n_samples=4)
        collision = self.env.check_collision(boxes, arm_points)
        self.robot.base_pose = original
        self.robot._fk_cache = None
        return not collision

    def _find_safe_start_pose(self) -> np.ndarray:
        """在默认起始区域搜索一个不与障碍物碰撞的位置。"""
        default_xy = np.array(
            self.env.rand_cfg.get("default_start", [-7.0, -6.5, 0.0]), dtype=float
        )[:3]
        default = np.array([default_xy[0], default_xy[1], 0.0, 0.0], dtype=float)
        if self._is_base_pose_safe(default):
            return default

        # 在默认位置附近随机搜索
        rng = np.random.default_rng(self.seed)
        for _ in range(500):
            candidate = default.copy()
            candidate[0] += rng.uniform(-2.0, 2.0)
            candidate[1] += rng.uniform(-2.0, 2.0)
            candidate[2] = 0.0
            if (
                self.env.inside_bounds(candidate[:3])
                and self._is_base_pose_safe(candidate)
            ):
                return candidate

        # 若仍找不到，返回默认（后续可能碰撞，但至少能启动）
        print("[WARN] Could not find a fully safe start pose; using default.")
        return default

    def _ensure_action_safe(
        self,
        base_cmd: np.ndarray,
        joint_delta: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """预测 base_cmd + joint_delta 执行后是否安全，若不安全则返回安全动作。

        策略：
        - 先检查原始动作；
        - 若预测碰撞，则尝试只执行机械臂动作（基座不动）；
        - 若仍碰撞，则返回零动作。
        """
        base_cmd = np.asarray(base_cmd, dtype=float).copy()
        joint_delta = np.asarray(joint_delta, dtype=float).copy()

        if self._predict_action_collision(base_cmd, joint_delta):
            # 尝试仅机械臂运动
            if not self._predict_action_collision(np.zeros(3), joint_delta):
                return np.zeros(3), joint_delta
            # 尝试仅基座运动
            if not self._predict_action_collision(base_cmd, np.zeros(6)):
                return base_cmd, np.zeros(6)
            # 都不行，则保持静止
            return np.zeros(3), np.zeros(6)
        return base_cmd, joint_delta

    def _predict_action_collision(
        self,
        base_cmd: np.ndarray,
        joint_delta: np.ndarray,
    ) -> bool:
        """在临时副本上执行动作并预测是否碰撞（不修改真实机器人状态）。"""
        original_pose = self.robot.base_pose.copy()
        original_joints = self.robot.joint_angles.copy()
        self.robot._fk_cache = None

        # 临时应用动作
        self.robot.apply_base_command(
            base_cmd[0], base_cmd[1], base_cmd[2],
            check_bounds=(self.env.bounds_low, self.env.bounds_high),
        )
        self.robot.apply_joint_command(joint_delta)

        boxes = self.robot.base_collision_boxes()
        arm_points = self.robot.arm_links_positions()
        collision = self.env.check_collision(boxes, arm_points)

        # 恢复原状态
        self.robot.base_pose = original_pose
        self.robot.joint_angles = original_joints
        self.robot._fk_cache = None
        return collision

    # ---------------------------------------------------------------------- #
    # 辅助
    # ---------------------------------------------------------------------- #
    def export_history(self, path: "str | Path") -> None:
        """将历史记录导出为 npz 文件。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        base_poses = np.stack([rec["base_pose"] for rec in self.history])
        ee_positions = np.stack([rec["ee_pos"] for rec in self.history])
        joint_angles = np.stack([rec["joint_angles_deg"] for rec in self.history])
        steps = np.array([rec["step"] for rec in self.history])
        states = np.array([rec["state"] for rec in self.history])

        np.savez(
            path,
            base_poses=base_poses,
            ee_positions=ee_positions,
            joint_angles=joint_angles,
            steps=steps,
            states=states,
            source_pos=self.env.source_pos,
        )
