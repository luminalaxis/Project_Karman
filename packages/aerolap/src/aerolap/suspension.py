"""Suspension description of one axle (stiffnesses referred to the wheel centre).

Linear springs / ARB / third element, plus the optional nonlinear parts: a bump stop (extra rate
beyond a gap in bump travel), a bilinear damper with a rebound factor (transient solver), roll-centre
migration with heave, bump steer and a tabulated camber curve.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def _camber_curve(points, axle: str):
    if points is None:
        return None
    try:
        arr = np.asarray(points, dtype=float)
        ok = arr.ndim == 2 and arr.shape[1] == 2 and len(arr) >= 2
    except (TypeError, ValueError):
        ok = False
    if not ok:
        from .config import ConfigError

        raise ConfigError(path=f"suspension.{axle}.camber_curve", value=points,
                          condition="[[바퀴 이동 mm (범프 +), 캠버 변화 deg], ...] 2점 이상",
                          hint="예: [[-30, 0.6], [0, 0], [30, -0.9]]")
    arr = arr[np.argsort(arr[:, 0])]
    return arr[:, 0] / 1000.0, np.radians(arr[:, 1])


def series(k1: float, k2: float) -> float:
    """Two springs in series (inf-safe)."""
    if math.isinf(k1):
        return k2
    if math.isinf(k2):
        return k1
    if k1 <= 0.0 or k2 <= 0.0:
        return 0.0
    return k1 * k2 / (k1 + k2)


@dataclass(frozen=True)
class Axle:
    track: float  # m
    wheel_rate: float  # N/m per corner (spring * MR^2)
    heave_rate: float  # N/m, third element at the wheel (heave only)
    arb_rate: float  # N m/rad, anti-roll bar at the wheels
    roll_center: float  # m
    static_ride_height: float  # m
    static_camber: float  # rad
    camber_gain: float  # rad per m of bump
    tire_rate: float  # N/m per tyre
    anti_brake: float  # anti-dive (front) / anti-lift (rear), fraction
    anti_drive: float  # anti-squat (rear), fraction
    unsprung_mass: float  # kg, both corners
    motion_ratio: float = 1.0
    damper_rate: float | None = None  # N s/m at the damper (None -> damping ratio)
    damping_ratio: float = 0.65
    bump_stop_gap: float | None = None  # m of wheel travel in bump
    bump_stop_rate: float = 0.0  # N/m at the wheel
    damper_high_rate: float | None = None  # N s/m at the damper above the knee
    damper_knee: float = 0.05  # m/s at the damper
    rebound_factor: float = 1.0
    roll_center_gain: float = 0.0  # m of roll-centre height per m of heave (bump +)
    bump_steer: float = 0.0  # rad of toe-in per m of bump
    camber_curve: tuple | None = None  # (travel m, camber change rad) arrays

    @classmethod
    def from_config(cls, a: dict, track_mm: float, tire_rate: float, unsprung_kg: float,
                    is_front: bool) -> "Axle":
        mr = float(a["motion_ratio"])
        return cls(
            track=track_mm / 1000.0,
            wheel_rate=float(a["spring_rate_N_per_mm"]) * 1000.0 * mr * mr,
            heave_rate=float(a["heave_spring_N_per_mm"]) * 1000.0,
            arb_rate=float(a["arb_rate_Nm_per_deg"]) * 180.0 / math.pi,
            roll_center=float(a["roll_center_mm"]) / 1000.0,
            static_ride_height=float(a["static_ride_height_mm"]) / 1000.0,
            static_camber=math.radians(float(a["static_camber_deg"])),
            camber_gain=math.radians(float(a["camber_gain_deg_per_mm"])) * 1000.0,
            tire_rate=tire_rate,
            anti_brake=float(a["anti_dive"] if is_front else a["anti_lift"]),
            anti_drive=0.0 if is_front else float(a["anti_squat"]),
            unsprung_mass=unsprung_kg,
            motion_ratio=mr,
            damper_rate=None if a.get("damper_rate_Ns_per_mm") is None
            else float(a["damper_rate_Ns_per_mm"]) * 1000.0,
            damping_ratio=float(a.get("damping_ratio", 0.65)),
            bump_stop_gap=None if a.get("bump_stop_gap_mm") is None else float(a["bump_stop_gap_mm"]) / 1000.0,
            bump_stop_rate=float(a.get("bump_stop_rate_N_per_mm") or 0.0) * 1000.0,
            damper_high_rate=None if a.get("damper_high_speed_Ns_per_mm") is None
            else float(a["damper_high_speed_Ns_per_mm"]) * 1000.0,
            damper_knee=float(a.get("damper_knee_mm_s", 50.0)) / 1000.0,
            rebound_factor=float(a.get("damper_rebound_factor", 1.0)),
            roll_center_gain=float(a.get("roll_center_gain_mm_per_mm") or 0.0),
            bump_steer=math.radians(float(a.get("bump_steer_deg_per_mm") or 0.0)) * 1000.0,
            camber_curve=_camber_curve(a.get("camber_curve"), "front" if is_front else "rear"),
        )

    @property
    def has_bump_stop(self) -> bool:
        return self.bump_stop_gap is not None and self.bump_stop_rate > 0.0

    def corner_force(self, z):
        """Spring + bump stop force at the wheel (N) for wheel travel ``z`` (m, bump +)."""
        f = self.wheel_rate * z
        if self.has_bump_stop:
            f = f + self.bump_stop_rate * np.maximum(np.asarray(z) - self.bump_stop_gap, 0.0)
        return f

    def camber_change(self, z):
        """Camber change (rad) for wheel travel ``z`` (m): the table if given, else the linear gain."""
        if self.camber_curve is None:
            return self.camber_gain * np.asarray(z)
        tz, tc = self.camber_curve
        return np.interp(np.asarray(z), tz, tc)

    def damper_force(self, v, c_low: float) -> float:
        """Damper force at the wheel (N) for wheel speed ``v`` (m/s, bump +); ``c_low`` is the wheel
        damping below the knee. Bilinear above the knee, rebound scaled by ``rebound_factor``."""
        f = c_low * v
        if self.damper_high_rate is not None:
            mr = self.motion_ratio
            knee_w = self.damper_knee / mr
            if abs(v) > knee_w:
                c_high = self.damper_high_rate * mr * mr
                f = math.copysign(c_low * knee_w + c_high * (abs(v) - knee_w), v)
        return f * self.rebound_factor if v < 0.0 else f

    @property
    def heave_stiffness(self) -> float:
        """Axle heave stiffness of the springs (both corners + third element), N/m."""
        return 2.0 * self.wheel_rate + self.heave_rate

    @property
    def roll_stiffness_springs(self) -> float:
        """Suspension roll stiffness (springs + ARB), N m/rad."""
        return 0.5 * self.wheel_rate * self.track ** 2 + self.arb_rate

    @property
    def roll_stiffness(self) -> float:
        """Roll stiffness including tyre compliance in series, N m/rad."""
        return series(self.roll_stiffness_springs, 0.5 * self.tire_rate * self.track ** 2)

    def ride_rate(self) -> float:
        """Corner ride rate in heave (spring + half third element in series with the tyre)."""
        return series(self.wheel_rate + 0.5 * self.heave_rate, self.tire_rate)

    def ride_frequency(self, corner_sprung_mass: float) -> float:
        return math.sqrt(self.ride_rate() / corner_sprung_mass) / (2.0 * math.pi)

    def wheel_damping(self, corner_sprung_mass: float) -> float:
        """Damping per corner at the wheel, N s/m."""
        if self.damper_rate is not None:
            return self.damper_rate * self.motion_ratio ** 2
        k = self.wheel_rate + 0.5 * self.heave_rate
        return 2.0 * self.damping_ratio * math.sqrt(k * corner_sprung_mass)
