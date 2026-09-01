"""气味传感器模型。

OdorSim 使用 GADEN 仿真的 ground-truth ppm 驱动 MOX/PID 传感器模型。
本项目采用简化版：
- 直接读取环境浓度场 ppm；
- 加入高斯白噪声模拟真实传感器波动；
- 提供阈值检测和电压输出（模拟 MOX 电阻型响应）。
"""

from __future__ import annotations

import math

import numpy as np


class OdorSensorArray:
    """机器人上的气味传感器阵列。

    包含：4 个固定在小车四角的传感器 + 1 个机械臂末端传感器。
    """

    def __init__(
        self,
        noise_std: float = 0.05,
        response_time: float = 0.3,
        seed: int = 0,
    ):
        self.noise_std = float(noise_std)
        self.response_time = float(response_time)
        self.rng = np.random.default_rng(seed)

        # 一阶滞后状态：保存上一时刻读数
        self._last_reading: dict[str, float] = {}

    def reset(self) -> None:
        self._last_reading = {}

    def read(
        self,
        concentrations: dict[str, float],
        dt: float,
    ) -> dict[str, dict[str, float]]:
        """读取当前浓度，返回每个传感器的电压、ppm、是否触发。

        返回格式：
            {
              "front_left": {"ppm": ..., "voltage": ..., "trigger": True/False},
              ...
            }
        """
        out = {}
        alpha = dt / (self.response_time + dt)
        for name, ppm_true in concentrations.items():
            # 传感器噪声：相对噪声 + 绝对噪声
            noise = self.rng.normal(0.0, self.noise_std * max(ppm_true, 1.0) + 0.1)
            ppm_raw = max(0.0, ppm_true + noise)

            # 一阶滞后
            prev = self._last_reading.get(name, ppm_raw)
            ppm_filtered = prev + alpha * (ppm_raw - prev)
            self._last_reading[name] = ppm_filtered

            voltage = self._ppm_to_voltage(ppm_filtered)
            out[name] = {
                "ppm": float(ppm_filtered),
                "voltage": float(voltage),
                "trigger": bool(ppm_filtered > 0.0),
            }
        return out

    @staticmethod
    def _ppm_to_voltage(ppm: float) -> float:
        """将 ppm 映射为电压（0~5 V）。

        使用简化 MOX 响应：浓度越高，传感器电阻越低，电压越高。
        """
        # 对数压缩，避免数值爆炸
        v = 5.0 * (1.0 - math.exp(-ppm / 50.0))
        return float(np.clip(v, 0.0, 5.0))
