"""Vehicle model: mass/geometry, tyres, aero and an EV powertrain.

Sign conventions
----------------
* ``ClA`` is the *downforce* coefficient times reference area (positive = downforce).
* ``balance_front`` is the share of downforce acting on the front axle (0..1).
* ``a_x`` > 0 is acceleration, < 0 is braking.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

G = 9.81


@dataclass
class Tyre:
    mu_x: float = 1.55          # peak longitudinal friction coefficient at Fz0
    mu_y: float = 1.60          # peak lateral friction coefficient at Fz0
    load_sensitivity: float = 0.10  # fractional mu drop per +100% wheel load over Fz0
    Fz0: float = 700.0          # reference wheel load [N]

    def mu_scale(self, fz_wheel):
        """Load-sensitivity factor: mu(Fz) = mu0 * mu_scale(Fz)."""
        s = 1.0 - self.load_sensitivity * (np.asarray(fz_wheel) - self.Fz0) / self.Fz0
        return np.clip(s, 0.2, None)


@dataclass
class Aero:
    ClA: float = 3.5            # downforce coefficient * area [m^2]
    CdA: float = 1.3            # drag coefficient * area [m^2]
    balance_front: float = 0.45  # front share of downforce
    h_cp: float = 0.35          # height of drag centre of pressure [m]
    rho: float = 1.2            # air density [kg/m^3]


@dataclass
class Powertrain:
    max_power: float = 80_000.0  # accumulator power limit [W] (FS rules: 80 kW)
    max_torque: float = 140.0    # motor torque [Nm]
    max_rpm: float = 6500.0      # motor speed limit [rpm]
    gear_ratio: float = 4.0
    wheel_radius: float = 0.2    # [m]
    efficiency: float = 0.90     # accumulator -> wheel

    @property
    def v_max(self) -> float:
        """Road speed at the motor rpm limit [m/s]."""
        return self.max_rpm * 2 * np.pi / 60 * self.wheel_radius / self.gear_ratio

    def drive_force(self, v):
        """Maximum tractive force at the wheels [N] (torque & power limited)."""
        v = np.asarray(v, dtype=float)
        f_torque = self.max_torque * self.gear_ratio * self.efficiency / self.wheel_radius
        f_power = self.max_power * self.efficiency / np.maximum(v, 0.1)
        f = np.minimum(f_torque, f_power)
        return np.where(v >= self.v_max, 0.0, f)


@dataclass
class Vehicle:
    name: str = "vehicle"
    mass: float = 280.0               # incl. driver [kg]
    cg_height: float = 0.28           # [m]
    wheelbase: float = 1.53           # [m]
    weight_dist_front: float = 0.46   # static front axle load share
    rolling_resistance: float = 0.015
    tyre: Tyre = field(default_factory=Tyre)
    aero: Aero = field(default_factory=Aero)
    powertrain: Powertrain = field(default_factory=Powertrain)

    # ---- aero -------------------------------------------------------------
    def q(self, v):
        return 0.5 * self.aero.rho * np.asarray(v, dtype=float) ** 2

    def downforce(self, v):
        return self.q(v) * self.aero.ClA

    def drag(self, v):
        return self.q(v) * self.aero.CdA

    def resistance(self, v, fz_total):
        """Drag + rolling resistance [N]."""
        return self.drag(v) + self.rolling_resistance * fz_total

    # ---- loads ------------------------------------------------------------
    def axle_loads(self, v, a_x=0.0):
        """Front/rear axle vertical loads [N] including aero and longitudinal transfer."""
        m, L = self.mass, self.wheelbase
        df = self.downforce(v)
        fz_f = m * G * self.weight_dist_front + self.aero.balance_front * df
        fz_r = m * G * (1 - self.weight_dist_front) + (1 - self.aero.balance_front) * df
        # longitudinal load transfer from inertia and the drag pitching moment
        dfz = (m * np.asarray(a_x) * self.cg_height + self.drag(v) * self.aero.h_cp) / L
        fz_f = np.maximum(fz_f - dfz, 0.0)
        fz_r = np.maximum(fz_r + dfz, 0.0)
        return fz_f, fz_r

    def axle_grip(self, fz_axle):
        """(Fx_max, Fy_max) for one axle (two wheels) at axle load ``fz_axle``."""
        s = self.tyre.mu_scale(np.asarray(fz_axle) / 2.0)
        return self.tyre.mu_x * s * fz_axle, self.tyre.mu_y * s * fz_axle

    # ---- helpers ------------------------------------------------------------
    def with_changes(self, **changes) -> "Vehicle":
        """Copy with dotted-path overrides, e.g. ``with_changes(**{"aero.ClA": 4.0})``."""
        v = copy.deepcopy(self)
        for key, value in changes.items():
            obj = v
            *parents, attr = key.split(".")
            for p in parents:
                obj = getattr(obj, p)
            if not hasattr(obj, attr):
                raise AttributeError(f"unknown vehicle parameter: {key}")
            setattr(obj, attr, value)
        return v

    def scaled_aero(self, cl_scale=1.0, cd_scale=1.0) -> "Vehicle":
        return self.with_changes(**{"aero.ClA": self.aero.ClA * cl_scale,
                                    "aero.CdA": self.aero.CdA * cd_scale})

    def to_dict(self) -> dict:
        return asdict(self)


def load_vehicle(path) -> Vehicle:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    sub = {"tyre": Tyre, "aero": Aero, "powertrain": Powertrain}
    kwargs = {k: (sub[k](**v) if k in sub else v) for k, v in data.items()}
    return Vehicle(**kwargs)
