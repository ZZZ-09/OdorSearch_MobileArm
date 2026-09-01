"""OdorSim/GADEN adapter for OdorSearch_MobileArm.

This module lets the original ``OdorSearchAgent`` run against the GADEN gas
 dispersion backend shipped with OdorSim, while keeping the existing robot
 kinematics, collision model and visualization untouched.

Design choices (minimal intrusion):
* ``GadenBackedWarehouseEnv`` subclasses ``WarehouseEnv`` and only overrides
  concentration queries / time stepping.
* The robot motion model, search algorithm, and visualizer are reused verbatim.
* A dedicated ``OdorSearchSessionOdorSim`` (see ``src/simulation_odorsim.py``)
  accepts a pre-built env instance so the adapter can be injected.
* A small standalone GADEN scenario (``odorsim_scenarios/warehouse_20x16``)
  provides a 20×16 m empty room matching ``config/warehouse.yaml``.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from src.environment import WarehouseEnv
from src.search_algorithm import OdorSearchAgent


def _to_gaden(points: np.ndarray) -> np.ndarray:
    """Map warehouse-world points to the GADEN scenario frame.

    The custom ``warehouse_20x16`` scenario is aligned with the warehouse world
    frame (origin at room center, x∈[-10,10], y∈[-8,8], z∈[0,5]), so the map
    is the identity.  Modify here if a different scenario / alignment is used.
    """
    return np.asarray(points, dtype=float).reshape(-1, 3)


class GadenBackedWarehouseEnv(WarehouseEnv):
    """WarehouseEnv whose concentration field is provided by GADEN.

    Args:
        config: passed to ``WarehouseEnv``.
        seed: passed to ``WarehouseEnv``.
        bridge: a connected :class:`odor_sim.bridge.gaden_bridge.GadenBridge`.
        gaden_dt: GADEN simulation timestep (s).  Must match the ``deltaTime``
            field in the scenario's ``sim.yaml``.
    """

    def __init__(
        self,
        config: "dict[str, Any] | None" = None,
        seed: int = 0,
        bridge: Any = None,
        gaden_dt: float = 0.05,
    ):
        # Set bridge BEFORE super().__init__() because WarehouseEnv.__init__()
        # calls self.reset(), which we override to touch the bridge.
        self.bridge = bridge
        self.gaden_dt = float(gaden_dt)
        self._ppm_cache: dict[tuple[float, ...], float] = {}
        self._query_count = 0
        super().__init__(config, seed=seed)

    # ------------------------------------------------------------------ #
    # GADEN lifecycle hooks
    # ------------------------------------------------------------------ #
    def reset(self, seed: "int | None" = None) -> None:
        """Reset the warehouse env and (re-)initialise GADEN time/source pose."""
        super().reset(seed=seed)
        self._ppm_cache = {}
        if self.bridge is not None:
            self.bridge.reset_time()
            self._publish_source_poses()
            # Prime one GADEN step so that the first concentration query is not
            # all zeros.
            self.bridge.step(1)

    def step(self) -> None:
        """Advance the analytical turbulence clock and GADEN one lockstep."""
        super().step()
        self._ppm_cache = {}
        if self.bridge is not None:
            self._publish_source_poses()
            self.bridge.step(1)

    def _publish_source_poses(self) -> None:
        if self.bridge is None:
            return
        # The scenario has a single source; publish its warehouse-world pose.
        source_poses = np.array([self.source_pos], dtype=float)
        self.bridge.publish_source_poses(_to_gaden(source_poses))

    # ------------------------------------------------------------------ #
    # Concentration queries
    # ------------------------------------------------------------------ #
    def concentration_at(self, point: np.ndarray) -> float:
        """Return GADEN ppm at ``point`` (falls back to analytical if no bridge)."""
        if self.bridge is None:
            return super().concentration_at(point)

        key = tuple(np.round(np.asarray(point, dtype=float).reshape(3), 6))
        if key not in self._ppm_cache:
            ppm_dict = self.bridge.query_ppm(_to_gaden(np.array([key])))[0]
            self._ppm_cache[key] = self._extract_ppm(ppm_dict)
            self._query_count += 1
        return self._ppm_cache[key]

    def query_sensors(self, sensor_positions: dict[str, np.ndarray]) -> dict[str, float]:
        """Batch query GADEN ppm for all named sensor positions."""
        if self.bridge is None or not sensor_positions:
            return super().query_sensors(sensor_positions)

        names = list(sensor_positions.keys())
        points = np.array([sensor_positions[n] for n in names], dtype=float)
        ppm_dicts = self.bridge.query_ppm(_to_gaden(points))
        self._query_count += len(names)
        return {name: self._extract_ppm(d) for name, d in zip(names, ppm_dicts)}

    def _extract_ppm(self, ppm_dict: dict[str, float]) -> float:
        """Extract the ppm value for the configured gas, or fall back."""
        if not ppm_dict:
            return float(self.background)
        gas = getattr(self, "gas_type", None)
        if gas and gas in ppm_dict:
            return float(ppm_dict[gas])
        # If the configured gas is not present, sum all reported gases.
        return float(sum(ppm_dict.values()))

    # ------------------------------------------------------------------ #
    # Diagnostics
    # ------------------------------------------------------------------ #
    def get_query_stats(self) -> dict[str, Any]:
        return {
            "gaden_dt": self.gaden_dt,
            "query_count": self._query_count,
            "cache_size": len(self._ppm_cache),
        }


class GadenAwareSearchAgent(OdorSearchAgent):
    """OdorSearchAgent that estimates gradients with a single batched GADEN query.

    The original agent calls ``env.concentration_at()`` six times per gradient
    estimate.  When ``env`` is a ``GadenBackedWarehouseEnv`` each call is a ROS
    service round-trip and easily overloads the lockstep bridge.  This subclass
    overrides ``_estimate_gradient`` to query all seven required points in one
    ``env.query_sensors()`` call.
    """

    def _estimate_gradient(self, point: np.ndarray, delta: float = 0.08) -> np.ndarray:
        point = np.asarray(point, dtype=float)
        positions: dict[str, np.ndarray] = {"c": point}
        for i in range(3):
            p_plus = point.copy()
            p_minus = point.copy()
            p_plus[i] += delta
            p_minus[i] -= delta
            positions[f"p{i}_plus"] = p_plus
            positions[f"p{i}_minus"] = p_minus

        concentrations = self.env.query_sensors(positions)
        grad = np.zeros(3)
        for i in range(3):
            c_plus = concentrations[f"p{i}_plus"]
            c_minus = concentrations[f"p{i}_minus"]
            grad[i] = (c_plus - c_minus) / (2.0 * delta)
        return grad
