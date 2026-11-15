"""Electric powertrain: accumulator power cap, motor torque/speed limits, regen."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

#: Speed floor [m/s] used in P/v so that the power-limited force stays finite at standstill.
#: Below the torque/power crossover the torque limit is active anyway, so the floor never binds.
V_FLOOR_MPS = 1e-3


@dataclass(frozen=True)
class Powertrain:
    """Single rear motor through a fixed final drive.

    Attributes:
        max_power_w: Accumulator output power cap [W] (battery side).
        motor_max_torque_nm: Motor peak torque [N·m].
        motor_max_rpm: Motor speed limit [rpm].
        gear_ratio: Final drive reduction [-].
        efficiency: Accumulator→wheel efficiency [-]; also applied on the torque path
            (spec 3.5: F = T·G·η/r).
        regen_max_power_w: Regen cap at the accumulator [W].
        regen_enabled: Whether regenerative braking recovers energy.
        differential: ``"lsd"`` (ideal, axle sum) or ``"open"`` (2 × weaker wheel).
    """

    max_power_w: float = 80_000.0
    motor_max_torque_nm: float = 230.0
    motor_max_rpm: float = 6500.0
    gear_ratio: float = 4.0
    efficiency: float = 0.85
    regen_max_power_w: float = 20_000.0
    regen_enabled: bool = True
    differential: Literal["lsd", "open"] = "lsd"

    def v_max_mps(self, wheel_radius_m: float) -> float:
        """Vehicle speed [m/s] at the motor speed limit."""
        return self.motor_max_rpm * 2.0 * math.pi / 60.0 / self.gear_ratio * wheel_radius_m

    def torque_force_n(self, wheel_radius_m: float) -> float:
        """Torque-limited tractive force at the wheels [N] (T·G·η/r)."""
        return self.motor_max_torque_nm * self.gear_ratio * self.efficiency / wheel_radius_m

    def drive_force_limit(self, v_mps: ArrayLike, wheel_radius_m: float) -> NDArray[np.float64]:
        """Max tractive force the powertrain can deliver at the wheels [N].

        ``min(T·G·η/r, P·η/v)`` below the rpm limit, 0 above it. At v → 0 the power branch
        diverges, so v is floored at :data:`V_FLOOR_MPS` (the torque branch is active there).
        """
        v = np.asarray(v_mps, dtype=float)
        f_power = self.max_power_w * self.efficiency / np.maximum(v, V_FLOOR_MPS)
        f = np.minimum(self.torque_force_n(wheel_radius_m), f_power)
        return np.where(v <= self.v_max_mps(wheel_radius_m) * (1.0 + 1e-9), f, 0.0)

    def regen_force_limit(self, v_mps: ArrayLike, wheel_radius_m: float) -> NDArray[np.float64]:
        """Max regenerative braking force at the wheels [N] (0 if regen disabled).

        Limited by motor torque (T·G/r) and the accumulator regen cap (P_regen/(η·v)).
        """
        v = np.asarray(v_mps, dtype=float)
        if not self.regen_enabled:
            return np.zeros_like(v)
        f_torque = self.motor_max_torque_nm * self.gear_ratio / wheel_radius_m
        f_power = self.regen_max_power_w / (self.efficiency * np.maximum(v, V_FLOOR_MPS))
        return np.minimum(f_torque, f_power)

    def motor_rpm(self, v_mps: ArrayLike, wheel_radius_m: float) -> NDArray[np.float64]:
        """Motor speed [rpm] at vehicle speed ``v_mps``."""
        v = np.asarray(v_mps, dtype=float)
        return v / wheel_radius_m * self.gear_ratio * 60.0 / (2.0 * math.pi)
