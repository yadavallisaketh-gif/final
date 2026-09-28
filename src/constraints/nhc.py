"""Non-holonomic constraint: a road vehicle does not slide sideways.

Applied as *soft* pseudo-measurements v_lateral = 0 and v_vertical = 0, so a
real skid, a bump or a motorcycle's lean is penalised, not forbidden. The
lateral innovation also observes heading / gyro-bias errors (see ekf2d).
"""
from __future__ import annotations

from ..fusion.ekf2d import EKF2D

PROFILES = {"car": 0.15, "van": 0.2, "motorcycle": 1.0}


def apply_nhc(ekf: EKF2D, sigma: float, sigma_vertical: float | None = None) -> bool:
    return ekf.update_nhc(sigma, sigma_vertical)
