"""独立评估体系。

该模块独立于机器人运行时的决策逻辑，用于在仿真结束后对搜索结果进行评估：
- 当机器人声明找到气味源时，使用声明的源位置计算误差；
- 当机器人未声明找到时，使用最终末端传感器位置作为 fallback；
- 若实际源与声明/最终位置的距离小于阈值，则视为成功。
"""

from __future__ import annotations

from typing import Any

import numpy as np


class SourceSearchEvaluator:
    """气味源搜索任务离线评估器。

    Args:
        success_distance_threshold: 判定成功的距离阈值（m），默认 1.0 m。
    """

    def __init__(self, success_distance_threshold: float = 1.0):
        self.success_distance_threshold = float(success_distance_threshold)

    def evaluate(
        self,
        actual_source_pos: np.ndarray,
        declared_source_pos: "np.ndarray | None",
        final_ee_pos: np.ndarray,
        finished: bool = False,
    ) -> dict[str, Any]:
        """评估单次搜索结果。

        Args:
            actual_source_pos: 真实气味源位置。
            declared_source_pos: 机器人声明的源位置；若为 None 则使用 final_ee_pos。
            final_ee_pos: 最终末端传感器位置（fallback）。
            finished: 机器人是否进入了 FINISHED 状态。

        Returns:
            包含 error_distance、success、evaluated_position、finished 等字段的字典。
        """
        actual = np.asarray(actual_source_pos, dtype=float)
        evaluated = (
            np.asarray(declared_source_pos, dtype=float)
            if declared_source_pos is not None
            else np.asarray(final_ee_pos, dtype=float)
        )
        error = float(np.linalg.norm(evaluated - actual))
        success = error < self.success_distance_threshold

        return {
            "finished": bool(finished),
            "actual_source_pos": actual.copy(),
            "declared_source_pos": (
                np.asarray(declared_source_pos, dtype=float).copy()
                if declared_source_pos is not None
                else None
            ),
            "evaluated_position": evaluated.copy(),
            "error_distance_m": error,
            "success_distance_threshold_m": self.success_distance_threshold,
            "evaluated_success": success,
        }

    def evaluate_session(self, session: Any) -> dict[str, Any]:
        """直接对仿真会话对象进行评估。

        session 需具备以下属性/方法：
            - env.source_pos
            - agent.state / agent.declared_source_position
            - history（列表，最后一条包含 ee_pos）
        """
        from src.search_algorithm import SearchState

        actual = np.asarray(session.env.source_pos, dtype=float)
        finished = getattr(session.agent, "state", None) == SearchState.FINISHED
        declared = getattr(session.agent, "declared_source_position", None)
        final_ee = (
            session.history[-1]["ee_pos"]
            if session.history
            else actual + 1e6
        )
        return self.evaluate(actual, declared, final_ee, finished)
