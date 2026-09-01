"""Reward shaping for the odor search RL task."""
from __future__ import annotations

from typing import Any

import numpy as np

from src.simulation_odorsim import OdorSearchSessionOdorSim


class RewardShaper:
    """Configurable reward function."""

    def __init__(
        self,
        close_coef: float = 5.0,
        detect_coef: float = 0.5,
        success_bonus: float = 100.0,
        success_bonus_decay: bool = True,
        time_penalty: float = -0.01,
        collision_penalty: float = -50.0,
        explore_coef: float = 0.02,
        source_threshold: float = 0.55,
        ppm_threshold: float = 50.0,
    ):
        self.close_coef = close_coef
        self.detect_coef = detect_coef
        self.success_bonus = success_bonus
        self.success_bonus_decay = success_bonus_decay
        self.time_penalty = time_penalty
        self.collision_penalty = collision_penalty
        self.explore_coef = explore_coef
        self.source_threshold = source_threshold
        self.ppm_threshold = ppm_threshold
        self._last_dist: float | None = None
        self._visited: set[tuple[int, int]] = set()

    def reset(self) -> None:
        self._last_dist = None
        self._visited = set()

    def __call__(
        self,
        session: OdorSearchSessionOdorSim,
        sensor_readings: dict[str, dict[str, float]],
        collision: bool,
        success: bool,
        max_steps: int,
    ) -> float:
        ee_pos = session.robot.ee_sensor_position()
        source = session.env.source_pos
        dist = float(np.linalg.norm(ee_pos - source))
        ee_ppm = sensor_readings.get("ee", {}).get("ppm", 0.0)

        reward = 0.0

        # 1. Distance improvement
        if self._last_dist is not None:
            reward += self.close_coef * (self._last_dist - dist)
        self._last_dist = dist

        # 2. Detection reward (shaped, not sparse)
        if ee_ppm > self.ppm_threshold:
            reward += self.detect_coef

        # 3. Success / collision / timeout
        if success:
            bonus = self.success_bonus
            if self.success_bonus_decay:
                bonus *= max(0.0, 1.0 - session.step_count / max_steps)
            reward += bonus
        if collision:
            reward += self.collision_penalty

        # 4. Time penalty
        reward += self.time_penalty

        # 5. Exploration reward for visiting new grid cells
        cell = (int(round(ee_pos[0] / 0.5)), int(round(ee_pos[1] / 0.5)))
        if cell not in self._visited:
            self._visited.add(cell)
            reward += self.explore_coef

        return float(reward)


# Global default instance used by env_wrapper.
_default_shaper = RewardShaper()


def default_reward(
    session: OdorSearchSessionOdorSim,
    sensor_readings: dict[str, dict[str, float]],
    collision: bool,
    success: bool,
    max_steps: int,
) -> float:
    return _default_shaper(session, sensor_readings, collision, success, max_steps)
