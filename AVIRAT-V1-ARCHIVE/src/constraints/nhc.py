"""Non-holonomic constraint: a road vehicle's rear axle does not slide sideways.

Applied as a *soft* pseudo-measurement at the rear axle,
    v_l(phone) - (w - b_g) * r_x = 0,
so the phone's legitimate sideways motion in a turn (om * r_x, ~0.5 m/s at
0.3 rad/s and r_x = 1.8 m) is not mistaken for a violation. R grows in turns
and under lateral acceleration, and the update is skipped in aggressive
cornering (see ekf2d.update_nhc). Profiles give sigma for other vehicles.
"""
from __future__ import annotations

from ..fusion.ekf2d import EKF2D

PROFILES = {"car": 0.15, "van": 0.2, "motorcycle": 1.0}


def apply_nhc(ekf: EKF2D, sigma: float) -> bool:
    return ekf.update_nhc(sigma)
