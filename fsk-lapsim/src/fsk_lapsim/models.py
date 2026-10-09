"""Quasi-steady vehicle models.

Both models answer three questions for given speed v [m/s] and lateral acceleration
ay [m/s²] (vectorised over numpy arrays):

* ``ay_max(v)``           — cornering limit at constant speed (DECISIONS D-08);
* ``ax_accel_max(v, ay)`` — largest longitudinal acceleration;
* ``ax_brake_min(v, ay)`` — most negative longitudinal acceleration.

Infeasible states (|ay| beyond the limit) return NaN.

**Reference state.** ``ax_ref(v) = min(0, ax_accel_max(v, 0))``: holding speed (ax = 0) or,
above the top speed, full power. ``ay_max(v)`` is the largest ay at which ``ax_ref`` is
still feasible, so for every |ay| ≤ ay_max the feasible accelerations form an interval
[ax_brake_min, ax_accel_max] that contains ``ax_ref``. On the limit the car can always hold its
speed, which keeps the lap integrator stable in long constant-radius corners.

**Feasibility** of a state (v, ax, ay) with required tyre force  F = m·ax + F_drag + F_roll:

* lateral: each axle can carry its share of m·ay;
* F ≥ 0 (driving): F ≤ min(powertrain limit, drive-axle force left on the friction ellipse);
* F < 0 (braking): −F ≤ brake force left on the friction ellipses.

Model A (:class:`PointMassModel`) — validation baseline: total grip μ0·(m·g + F_down) on one
friction ellipse, no load transfer, no load sensitivity, drive and brake on total grip.

Model B (:class:`AxleModel`) — main model: per-wheel loads with aero balance, longitudinal
(m·ax·h/L) and lateral (m·ay·h/t split by LLTD) load transfer, load-sensitive tyres, axle
lateral force demand from yaw equilibrium (Fy_f = m·ay·b/L, Fy_r = m·ay·a/L), rear-wheel drive
(LSD = axle sum, open diff = 2 × weaker wheel) and ideal or fixed-bias 4-wheel braking.
Within an axle the lateral force is shared in proportion to wheel capacity (equal
utilisation), which makes the axle-level friction ellipse exact.
"""

from __future__ import annotations

import abc
from typing import ClassVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .vehicle import Vehicle

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

#: Bisection iterations for the implicit model-B limits (bracket / 2^N resolution).
N_BISECT = 52
_TOL_N = 1e-9  # force tolerance [N] in feasibility comparisons


class VehicleModel(abc.ABC):
    """Common interface and shared force elements."""

    name: ClassVar[str] = "base"

    def __init__(self, vehicle: Vehicle) -> None:
        self.vehicle = vehicle

    # -- shared forces -------------------------------------------------------------------
    def downforce_n(self, v: ArrayLike) -> FloatArray:
        """Total downforce [N]."""
        return self.vehicle.aero.downforce_n(v)

    def drag_n(self, v: ArrayLike) -> FloatArray:
        """Aero drag [N]."""
        return self.vehicle.aero.drag_n(v)

    def rolling_n(self, v: ArrayLike) -> FloatArray:
        """Rolling resistance Crr·(m·g + F_down) [N]."""
        return self.vehicle.tire.rolling_resistance_coeff * (
            self.vehicle.weight_n + self.downforce_n(v)
        )

    def resistance_n(self, v: ArrayLike) -> FloatArray:
        """Drag + rolling resistance [N]."""
        return self.drag_n(v) + self.rolling_n(v)

    def drive_limit_n(self, v: ArrayLike) -> FloatArray:
        """Powertrain tractive force limit at the wheels [N]."""
        return self.vehicle.powertrain.drive_force_limit(v, self.vehicle.tire.radius_m)

    def coast_ax(self, v: ArrayLike) -> FloatArray:
        """Deceleration with zero tyre longitudinal force (−(drag + rolling)/m) [m/s²]."""
        return -self.resistance_n(v) / self.vehicle.mass_kg

    def ax_ref(self, v: ArrayLike) -> FloatArray:
        """Reference acceleration [m/s²]: 0, or the full-power value above top speed."""
        v = np.asarray(v, float)
        return np.minimum(0.0, self.ax_accel_max(v, np.zeros_like(v)))

    # -- model specific ------------------------------------------------------------------
    @abc.abstractmethod
    def ay_max(self, v: ArrayLike) -> FloatArray:
        """Cornering limit [m/s²] at constant speed (see module docstring)."""

    @abc.abstractmethod
    def ax_accel_max(self, v: ArrayLike, ay: ArrayLike, power_limited: bool = True) -> FloatArray:
        """Maximum longitudinal acceleration [m/s²] (NaN if |ay| > ay_max(v)).

        With ``power_limited=False`` the powertrain cap is ignored (tyre/grip limit only).
        Because the cap only bounds ax from above independently of ay,
        ``ax_accel_max = min(tyre-only value, power_ax(v))`` exactly.
        """

    def ax_accel_parts(self, v: ArrayLike, ay: ArrayLike) -> dict[str, FloatArray]:
        """Tyre-only acceleration limit of each constraint alone [m/s²].

        The feasible accelerations of every constraint form an interval starting at the
        reference state, so the tyre-only limit is the minimum over the parts. Each part is
        smooth, which lets the GGV interpolate them separately without kink errors.
        """
        return {"tyre": self.ax_accel_max(v, ay, power_limited=False)}

    def power_ax(self, v: ArrayLike) -> FloatArray:
        """Acceleration [m/s²] allowed by the powertrain alone: (F_pt − drag − rolling)/m."""
        return (self.drive_limit_n(v) - self.resistance_n(v)) / self.vehicle.mass_kg

    @abc.abstractmethod
    def ax_brake_min(self, v: ArrayLike, ay: ArrayLike) -> FloatArray:
        """Maximum deceleration as a negative ax [m/s²] (NaN if |ay| > ay_max(v))."""

    @abc.abstractmethod
    def wheel_loads(self, v: ArrayLike, ax: ArrayLike, ay: ArrayLike) -> FloatArray:
        """Vertical loads [N], shape (4, ...) ordered FL, FR, RL, RR (ay > 0 = left turn)."""

    def ay_max_by_limit(self, v: ArrayLike) -> dict[str, FloatArray]:
        """Cornering limit [m/s²] imposed by each constraint alone (at ``ax_ref``)."""
        return {"total": self.ay_max(v)}


class PointMassModel(VehicleModel):
    """Model A: point mass, constant μ0, friction ellipse on the total normal force."""

    name: ClassVar[str] = "point_mass"

    def _normal_n(self, v: ArrayLike) -> FloatArray:
        return self.vehicle.weight_n + self.downforce_n(v)

    def _ay_grip(self, v: ArrayLike) -> FloatArray:
        """Pure lateral grip limit μ0·N/m [m/s²]."""
        return self.vehicle.tire.mu0 * self._normal_n(v) / self.vehicle.mass_kg

    def _fx_tyre(self, v: FloatArray, ay: FloatArray) -> FloatArray:
        n = self._normal_n(v)
        u = np.abs(ay) / self._ay_grip(v)
        root = np.sqrt(np.clip(1.0 - u * u, 0.0, None))
        fx = self.vehicle.tire.mu_lon_lat_ratio * self.vehicle.tire.mu0 * n * root
        return np.where(u <= 1.0 + 1e-12, fx, np.nan)

    def _need_ref(self, v: FloatArray) -> FloatArray:
        """Tyre force at the reference state: min(drag + rolling, powertrain limit) [N]."""
        return np.minimum(self.resistance_n(v), self.drive_limit_n(v))

    def ay_max(self, v: ArrayLike) -> FloatArray:  # noqa: D102
        v = np.asarray(v, float)
        fx_max = self.vehicle.tire.mu_lon_lat_ratio * self.vehicle.tire.mu0 * self._normal_n(v)
        r = np.clip(self._need_ref(v) / fx_max, 0.0, 1.0)
        return self._ay_grip(v) * np.sqrt(1.0 - r * r)

    def ax_accel_max(self, v: ArrayLike, ay: ArrayLike, power_limited: bool = True) -> FloatArray:  # noqa: D102
        v, ay = np.broadcast_arrays(np.asarray(v, float), np.asarray(ay, float))
        f = self._fx_tyre(v, ay)
        if power_limited:
            f = np.minimum(f, self.drive_limit_n(v))
        out = (f - self.resistance_n(v)) / self.vehicle.mass_kg
        return np.where(np.abs(ay) <= self.ay_max(v) * (1 + 1e-12) + 1e-12, out, np.nan)

    def ax_brake_min(self, v: ArrayLike, ay: ArrayLike) -> FloatArray:  # noqa: D102
        v, ay = np.broadcast_arrays(np.asarray(v, float), np.asarray(ay, float))
        out = -(self._fx_tyre(v, ay) + self.resistance_n(v)) / self.vehicle.mass_kg
        return np.where(np.abs(ay) <= self.ay_max(v) * (1 + 1e-12) + 1e-12, out, np.nan)

    def wheel_loads(self, v: ArrayLike, ax: ArrayLike, ay: ArrayLike) -> FloatArray:  # noqa: D102
        v, ax, ay = np.broadcast_arrays(*(np.asarray(a, float) for a in (v, ax, ay)))
        n = self._normal_n(v) / 4.0
        return np.stack([n, n, n, n])


class AxleModel(VehicleModel):
    """Model B: axle-level quasi-steady model with load transfer (main model)."""

    name: ClassVar[str] = "axle"

    # -- loads ---------------------------------------------------------------------------
    def _axle_wheel_loads(
        self, v: FloatArray, ax: FloatArray, ay: FloatArray
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        """(front inner, front outer, rear inner, rear outer) vertical loads [N]."""
        veh = self.vehicle
        m, h, L = veh.mass_kg, veh.cg_height_m, veh.wheelbase_m
        fd = self.downforce_n(v)
        bal = veh.aero.balance_front
        dlong = m * ax * h / L
        fzf = np.maximum(veh.weight_n * veh.b_m / L + bal * fd - dlong, 0.0)
        fzr = np.maximum(veh.weight_n * veh.a_m / L + (1.0 - bal) * fd + dlong, 0.0)
        may_h = m * np.abs(ay) * h
        dff = veh.lltd_front * may_h / veh.track_front_m
        dfr = (1.0 - veh.lltd_front) * may_h / veh.track_rear_m
        fi = np.maximum(0.5 * fzf - dff, 0.0)
        ri = np.maximum(0.5 * fzr - dfr, 0.0)
        return fi, fzf - fi, ri, fzr - ri

    def wheel_loads(self, v: ArrayLike, ax: ArrayLike, ay: ArrayLike) -> FloatArray:  # noqa: D102
        v, ax, ay = np.broadcast_arrays(*(np.asarray(a, float) for a in (v, ax, ay)))
        fi, fo, ri, ro = self._axle_wheel_loads(v, ax, ay)
        left_turn = ay >= 0.0  # outer wheels are on the right
        return np.stack(
            [
                np.where(left_turn, fi, fo),
                np.where(left_turn, fo, fi),
                np.where(left_turn, ri, ro),
                np.where(left_turn, ro, ri),
            ]
        )

    # -- grip state ----------------------------------------------------------------------
    def _state(self, v: FloatArray, ax: FloatArray, ay: FloatArray) -> dict[str, FloatArray]:
        veh = self.vehicle
        tire = veh.tire
        fi, fo, ri, ro = self._axle_wheel_loads(v, ax, ay)
        cfi, cfo = tire.max_lateral_force(fi), tire.max_lateral_force(fo)
        cri, cro = tire.max_lateral_force(ri), tire.max_lateral_force(ro)
        cap_f, cap_r = cfi + cfo, cri + cro
        may = veh.mass_kg * np.abs(ay)
        req_f = may * veh.b_m / veh.wheelbase_m
        req_r = may * veh.a_m / veh.wheelbase_m
        with np.errstate(divide="ignore", invalid="ignore"):
            u_f = np.where(cap_f > 0, req_f / cap_f, np.where(req_f > 0, np.inf, 0.0))
            u_r = np.where(cap_r > 0, req_r / cap_r, np.where(req_r > 0, np.inf, 0.0))
        root_f = np.sqrt(np.clip(1.0 - u_f * u_f, 0.0, None))
        root_r = np.sqrt(np.clip(1.0 - u_r * u_r, 0.0, None))
        rxy = tire.mu_lon_lat_ratio
        rem_f = rxy * cap_f * root_f
        rem_r = rxy * cap_r * root_r
        if veh.powertrain.differential == "open":
            rem_drive = 2.0 * rxy * np.minimum(cri, cro) * root_r
        else:
            rem_drive = rem_r
        return {"u_f": u_f, "u_r": u_r, "rem_f": rem_f, "rem_r": rem_r, "rem_drive": rem_drive}

    def _brake_ok(self, st: dict[str, FloatArray], fb: FloatArray) -> BoolArray:
        veh = self.vehicle
        if veh.brake_distribution == "fixed":
            beta = veh.brake_front_bias
            return (fb * beta <= st["rem_f"] + _TOL_N) & (fb * (1.0 - beta) <= st["rem_r"] + _TOL_N)
        return fb <= st["rem_f"] + st["rem_r"] + _TOL_N

    def _feasible(
        self,
        v: FloatArray,
        ax: FloatArray,
        ay: FloatArray,
        res: FloatArray,
        f_pt: FloatArray,
        parts: str = "all",
    ) -> BoolArray:
        """Feasibility of (v, ax, ay); ``parts`` = all | front | rear | drive."""
        st = self._state(v, ax, ay)
        need = self.vehicle.mass_kg * ax + res
        if parts == "front":
            return st["u_f"] <= 1.0
        if parts == "rear":
            return st["u_r"] <= 1.0
        long_ok = np.where(
            need >= 0.0,
            need <= np.minimum(st["rem_drive"], f_pt) + _TOL_N,
            self._brake_ok(st, -need),
        )
        if parts == "drive":
            return long_ok
        return (st["u_f"] <= 1.0) & (st["u_r"] <= 1.0) & long_ok

    def _ay_bound(self, v: FloatArray) -> FloatArray:
        t = self.vehicle.tire
        n = self.vehicle.weight_n + self.downforce_n(v)
        mu_b = t.mu0 * (1.0 - t.load_sensitivity) if t.load_sensitivity < 0 else t.mu(n)
        return 1.05 * mu_b * n / self.vehicle.mass_kg + 1.0

    def _ay_limit(self, v: FloatArray, parts: str) -> FloatArray:
        res = self.resistance_n(v)
        f_pt = self.drive_limit_n(v)
        ax = self._ax_ref_straight(v, res, f_pt)
        lo = np.zeros_like(v)
        hi = self._ay_bound(v) * np.ones_like(v)
        for _ in range(N_BISECT):
            mid = 0.5 * (lo + hi)
            ok = self._feasible(v, ax, mid, res, f_pt, parts)
            lo = np.where(ok, mid, lo)
            hi = np.where(ok, hi, mid)
        return lo

    def _ax_ref_straight(self, v: FloatArray, res: FloatArray, f_pt: FloatArray) -> FloatArray:
        """min(0, straight-line max acceleration) (traction or power limited)."""
        zero = np.zeros_like(v)
        return np.minimum(0.0, self._accel_bisect(v, zero, res, f_pt, -res / self.vehicle.mass_kg))

    def _accel_bisect(
        self, v: FloatArray, ay: FloatArray, res: FloatArray, f_pt: FloatArray, lo: FloatArray
    ) -> FloatArray:
        """Largest feasible ax ≥ ``lo`` (``lo`` must be feasible), upper bound from ``f_pt``."""
        m = self.vehicle.mass_kg
        top = -res / m + f_pt / m
        top_ok = self._feasible(v, top, ay, res, f_pt)
        lo = lo.copy()
        hi = top.copy()
        for _ in range(N_BISECT):
            mid = 0.5 * (lo + hi)
            good = self._feasible(v, mid, ay, res, f_pt)
            lo = np.where(good, mid, lo)
            hi = np.where(good, hi, mid)
        return np.where(top_ok, np.maximum(top, lo), lo)

    def _grip_force_bound(self, v: FloatArray) -> FloatArray:
        """Upper bound on any tyre force [N] (replaces the powertrain cap in tyre-only mode)."""
        t = self.vehicle.tire
        n = self.vehicle.weight_n + self.downforce_n(v)
        mu_b = t.mu0 * (1.0 - t.load_sensitivity) if t.load_sensitivity < 0 else t.mu(n)
        return 1.05 * t.mu_lon_lat_ratio * mu_b * n + 1.0

    # -- limits --------------------------------------------------------------------------
    def ay_max(self, v: ArrayLike) -> FloatArray:  # noqa: D102
        return self._ay_limit(np.asarray(v, float), "all")

    def ay_max_by_limit(self, v: ArrayLike) -> dict[str, FloatArray]:
        """Cornering limit [m/s²] of each constraint alone: front axle, rear axle, drive."""
        v = np.asarray(v, float)
        out = {k: self._ay_limit(v, k) for k in ("front", "rear", "drive")}
        out["total"] = np.minimum(np.minimum(out["front"], out["rear"]), out["drive"])
        return out

    def ax_accel_max(self, v: ArrayLike, ay: ArrayLike, power_limited: bool = True) -> FloatArray:  # noqa: D102
        v, ay = np.broadcast_arrays(np.asarray(v, float), np.asarray(ay, float))
        res = self.resistance_n(v)
        f_pt = self.drive_limit_n(v)
        ref = self._ax_ref_straight(v, res, f_pt)
        ok0 = self._feasible(v, ref, ay, res, f_pt)
        cap = f_pt if power_limited else np.maximum(f_pt, self._grip_force_bound(v))
        return np.where(ok0, self._accel_bisect(v, ay, res, cap, ref), np.nan)

    def ax_accel_parts(self, v: ArrayLike, ay: ArrayLike) -> dict[str, FloatArray]:
        """Tyre-only acceleration limits: front lateral grip (front unloads) and traction."""
        v, ay = np.broadcast_arrays(np.asarray(v, float), np.asarray(ay, float))
        veh = self.vehicle
        m = veh.mass_kg
        res = self.resistance_n(v)
        f_pt = self.drive_limit_n(v)
        ref = self._ax_ref_straight(v, res, f_pt)
        ok0 = self._feasible(v, ref, ay, res, f_pt)
        bound = np.maximum(f_pt, self._grip_force_bound(v))
        # traction (rear ellipse, LSD/open) with the powertrain cap removed
        lo, hi = ref.copy(), -res / m + bound / m
        top_ok = self._feasible(v, hi, ay, res, bound, "drive")
        for _ in range(N_BISECT):
            mid = 0.5 * (lo + hi)
            good = self._feasible(v, mid, ay, res, bound, "drive")
            lo = np.where(good, mid, lo)
            hi = np.where(good, hi, mid)
        drive = np.where(top_ok, -res / m + bound / m, lo)
        # front lateral grip: bracket up to the acceleration that unloads the front axle
        fzf0 = veh.weight_n * veh.b_m / veh.wheelbase_m + veh.aero.balance_front * self.downforce_n(
            v
        )
        wheelie = (
            fzf0 * veh.wheelbase_m / (m * veh.cg_height_m)
            if veh.cg_height_m > 0
            else (np.full_like(v, 1e3))
        )
        lo, hi = ref.copy(), np.maximum(wheelie, ref)
        for _ in range(N_BISECT):
            mid = 0.5 * (lo + hi)
            good = self._feasible(v, mid, ay, res, bound, "front")
            lo = np.where(good, mid, lo)
            hi = np.where(good, hi, mid)
        return {"front": np.where(ok0, lo, np.nan), "drive": np.where(ok0, drive, np.nan)}

    def ax_brake_min(self, v: ArrayLike, ay: ArrayLike) -> FloatArray:  # noqa: D102
        v, ay = np.broadcast_arrays(np.asarray(v, float), np.asarray(ay, float))
        veh = self.vehicle
        m = veh.mass_kg
        res = self.resistance_n(v)
        f_pt = self.drive_limit_n(v)
        ref = self._ax_ref_straight(v, res, f_pt)
        ok0 = self._feasible(v, ref, ay, res, f_pt)
        n = veh.weight_n + self.downforce_n(v)
        t = veh.tire
        mu_b = t.mu0 * (1.0 - t.load_sensitivity) if t.load_sensitivity < 0 else t.mu(n)
        lo = -res / m - 1.05 * t.mu_lon_lat_ratio * mu_b * n / m - 1.0
        hi = ref.copy()
        for _ in range(N_BISECT):
            mid = 0.5 * (lo + hi)
            good = self._feasible(v, mid, ay, res, f_pt)
            hi = np.where(good, mid, hi)
            lo = np.where(good, lo, mid)
        return np.where(ok0, hi, np.nan)


MODELS: dict[str, type[VehicleModel]] = {
    "point_mass": PointMassModel,
    "axle": AxleModel,
    "A": PointMassModel,
    "B": AxleModel,
}


def make_model(vehicle: Vehicle, model: str = "axle") -> VehicleModel:
    """Instantiate a model by name: ``axle``/``B`` (default) or ``point_mass``/``A``."""
    try:
        return MODELS[model](vehicle)
    except KeyError:
        raise ValueError(f"unknown model {model!r}; use one of {sorted(MODELS)}") from None


def top_speed(model: VehicleModel) -> float:
    """Highest reachable steady straight-line speed [m/s] (drag/power or rpm limited)."""
    v_rpm = model.vehicle.v_max_mps
    if float(model.ax_accel_max(v_rpm, 0.0)) >= 0.0:
        return v_rpm
    lo, hi = 0.0, v_rpm
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if float(model.ax_accel_max(mid, 0.0)) >= 0.0:
            lo = mid
        else:
            hi = mid
    return lo
