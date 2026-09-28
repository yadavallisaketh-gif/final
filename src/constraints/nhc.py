"""Non-holonomic constraint: a road vehicle does not slide sideways.

Applied as a *soft* pseudo-measurement v_lateral = 0 with std `sigma`, so a
real skid or a motorcycle's lean is penalised, not forbidden. Vertical velocity
is not part of the 2-D state; the preprocessing removes gravity instead.
"""
from __future__ import annotations

from ..fusion.ekf2d import EKF2D

PROFILES = {"car": 0.15, "van": 0.2, "motorcycle": 1.0}


def apply_nhc(ekf: EKF2D, sigma: float) -> bool:
    return ekf.update_nhc(sigma)
