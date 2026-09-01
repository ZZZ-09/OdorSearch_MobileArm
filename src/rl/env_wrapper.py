"""Gymnasium Env wrapper for training the odor search policy in OdorSim/GADEN.

The wrapper exposes:
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)

It reuses the existing ``OdorSearchSessionOdorSim`` and
``GadenBackedWarehouseEnv`` without modifying them.
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from src.odor_sim_adapter import GadenBackedWarehouseEnv
from src.rl.reward_shaper import RewardShaper
from src.simulation_odorsim import OdorSearchSessionOdorSim


class OdorSearchRLEnv(gym.Env):
    """RL training environment for mobile-base + arm odor source search.

    Args:
        scenario_path: GADEN scenario config directory.
        scene_id: GADEN scene id.
        env_config: warehouse env config dict or path.
        bridge: optional pre-connected GadenBridge.
        gaden_dt: GADEN timestep.
        max_steps: episode horizon.
        room_half: room half-size for normalization (m).
        ppm_history_len: number of past ppm values in observation.
        use_wind_obs: whether to include estimated wind in observation.
        seed: random seed.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(
        self,
        scenario_path: str,
        scene_id: str = "scene1",
        env_config: "str | dict[str, Any] | None" = None,
        bridge: Any = None,
        gaden_dt: float = 0.05,
        max_steps: int = 2000,
        room_half: float = 4.0,
        ppm_history_len: int = 4,
        use_wind_obs: bool = True,
        seed: int = 0,
    ):
        super().__init__()
        self.scenario_path = scenario_path
        self.scene_id = scene_id
        self.env_config = env_config
        self.bridge = bridge
        self.gaden_dt = gaden_dt
        self.max_steps = max_steps
        self.room_half = room_half
        self.ppm_history_len = ppm_history_len
        self.use_wind_obs = use_wind_obs
        self.seed = seed

        # Action space: [dx, dy, dyaw, dJ1..dJ6]
        self.action_space = gym.spaces.Box(
            low=np.array([-0.10, -0.10, -0.15, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0], dtype=np.float32),
            high=np.array([0.10, 0.10, 0.15, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0], dtype=np.float32),
            dtype=np.float32,
        )

        obs_dim = 3 + 2 + 6 + ppm_history_len + 4  # base pos, yaw sin/cos, joints, ee hist, corner ppm
        if use_wind_obs:
            obs_dim += 3
        obs_dim += 1  # normalized step
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self._session: OdorSearchSessionOdorSim | None = None
        self._rng = np.random.default_rng(seed)
        self._ppm_history: list[float] = []
        self._last_sensor_readings: dict[str, dict[str, float]] | None = None
        self._steps = 0
        self._reward_shaper = RewardShaper()

    # ------------------------------------------------------------------ #
    # Gymnasium API
    # ------------------------------------------------------------------ #
    def reset(
        self,
        *,
        seed: "int | None" = None,
        options: "dict[str, Any] | None" = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self.seed = seed
            self._rng = np.random.default_rng(seed)

        # Build a fresh GADEN-backed env/session each reset so source/wind can
        # be randomized per episode.
        env = GadenBackedWarehouseEnv(
            config=self.env_config,
            seed=self.seed,
            bridge=self.bridge,
            gaden_dt=self.gaden_dt,
        )
        self._session = OdorSearchSessionOdorSim(env=env, seed=self.seed)

        # Random start pose in the left-bottom quadrant (away from typical source).
        start_x = float(self._rng.uniform(-self.room_half + 0.5, -0.5))
        start_y = float(self._rng.uniform(-self.room_half + 0.5, -0.5))
        base_pose = np.array([start_x, start_y, 0.0, 0.0], dtype=float)

        obs = self._session.reset(base_pose=base_pose)
        self._ppm_history = [0.0] * self.ppm_history_len
        self._steps = 0
        self._reward_shaper.reset()
        return self._build_obs(obs), self._build_info(obs)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=float).clip(
            self.action_space.low, self.action_space.high
        )
        assert self._session is not None

        # Inject action into the session step manually.
        session = self._session
        env = session.env
        robot = session.robot

        # 1. Query sensors (GADEN)
        sensor_positions = robot.all_sensor_positions()
        concentrations = env.query_sensors(sensor_positions)
        sensor_readings = session.sensors.read(concentrations, dt=env.dt)
        vision_result = session.vision.detect_source()

        # 2. Use the FSM agent only if you want BC/teacher; for RL we bypass it.
        # Here we directly apply the policy action.
        base_cmd = action[:3]
        joint_delta = action[3:]

        safe_base_cmd, safe_joint_delta = session._ensure_action_safe(base_cmd, joint_delta)
        robot.apply_base_command(
            safe_base_cmd[0], safe_base_cmd[1], safe_base_cmd[2],
            check_bounds=(env.bounds_low, env.bounds_high),
        )
        robot.apply_joint_command(safe_joint_delta)

        # Collision check
        boxes = robot.base_collision_boxes()
        arm_points = robot.arm_links_positions()
        collision = env.check_collision(boxes, arm_points)

        env.step()
        session.step_count += 1
        self._steps += 1

        obs = session._get_observation()
        self._last_sensor_readings = sensor_readings
        info = {
            "base_cmd": base_cmd,
            "joint_delta": joint_delta,
            "collision": collision,
            "sensor_readings": sensor_readings,
            "vision": vision_result,
        }

        ee_ppm = sensor_readings.get("ee", {}).get("ppm", 0.0)
        self._ppm_history.append(float(ee_ppm))
        if len(self._ppm_history) > self.ppm_history_len:
            self._ppm_history.pop(0)

        terminated = self._check_success(session, sensor_readings)
        truncated = self._steps >= self.max_steps
        reward = self._compute_reward(session, sensor_readings, collision, terminated)

        return self._build_obs(obs, sensor_readings), float(reward), terminated, truncated, info

    def render(self) -> np.ndarray | None:
        return None

    def close(self) -> None:
        if self._session is not None:
            self._session = None

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _build_obs(
        self,
        obs: dict[str, Any],
        sensor_readings: dict[str, dict[str, float]] | None = None,
    ) -> np.ndarray:
        sensor_readings = sensor_readings or self._last_sensor_readings or {}
        base = obs["base_pose"][:3] / self.room_half
        yaw = np.array([np.sin(obs["base_pose"][3]), np.cos(obs["base_pose"][3])])
        joints = obs["joint_angles_deg"] / 180.0

        ee_hist = np.array(self._ppm_history, dtype=float)
        ee_hist = np.log1p(ee_hist) / 10.0  # compress

        # Corner sensor ppm (log-compress)
        corner = np.zeros(4, dtype=float)
        for i, name in enumerate(["front_left", "front_right", "rear_left", "rear_right"]):
            ppm = sensor_readings.get(name, {}).get("ppm", 0.0)
            corner[i] = np.log1p(ppm) / 10.0

        parts = [base, yaw, joints, ee_hist, corner]
        if self.use_wind_obs:
            wind = self._session.env.wind / (np.linalg.norm(self._session.env.wind) + 1e-6)
            parts.append(wind)
        parts.append(np.array([self._steps / self.max_steps], dtype=float))
        return np.concatenate(parts).astype(np.float32)

    def _build_info(self, obs: dict[str, Any]) -> dict[str, Any]:
        return {
            "source_pos": obs.get("source_pos"),
            "base_pose": obs.get("base_pose"),
            "ee_pos": obs.get("ee_pos"),
        }

    def _check_success(
        self,
        session: OdorSearchSessionOdorSim,
        sensor_readings: dict[str, dict[str, float]] | None = None,
    ) -> bool:
        ee_pos = session.robot.ee_sensor_position()
        dist = float(np.linalg.norm(ee_pos - session.env.source_pos))
        sensor_readings = sensor_readings or self._last_sensor_readings or {}
        ee_ppm = sensor_readings.get("ee", {}).get("ppm", 0.0)
        return dist < 0.55 and ee_ppm > 50.0

    def _compute_reward(
        self,
        session: OdorSearchSessionOdorSim,
        sensor_readings: dict,
        collision: bool,
        success: bool,
    ) -> float:
        return self._reward_shaper(session, sensor_readings, collision, success, self.max_steps)
