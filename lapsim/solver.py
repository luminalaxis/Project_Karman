"""Quasi-steady-state lap solver.

1. Corner speed limit  v_max(s): highest speed at which both axles can carry their share of
   the lateral force (front: m v^2 k * b/L, rear: m v^2 k * a/L, rear also carrying the
   drag + rolling-resistance drive force on the friction ellipse).  Solved by vectorised
   bisection because downforce makes grip itself speed dependent.
2. Forward pass: accelerate with min(motor force, rear-axle grip left on the ellipse).
3. Backward pass: brake with the grip left on both axles (ideal brake balance).
4. v(s) = min(forward, backward); lap time = sum(ds / mean speed).

The marching passes use plain floats (they are sequential, numpy scalars would be slow);
the physics is identical to the vectorised ``Vehicle`` methods.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .track import Track
from .vehicle import G, Vehicle

PHASES = ("accel", "corner", "brake", "cruise")


@dataclass
class LapResult:
    lap_time: float
    s: np.ndarray          # [m]
    v: np.ndarray          # [m/s]
    v_corner: np.ndarray   # corner-limited speed [m/s]
    a_x: np.ndarray        # [m/s^2]
    a_y: np.ndarray        # [m/s^2]
    t: np.ndarray          # cumulative time at each s [s]
    phase: np.ndarray      # one of PHASES per point
    limiter: np.ndarray    # "front" / "rear" axle saturating first in the corner
    closed: bool

    def phase_time(self) -> dict:
        dt = np.diff(self.t)
        ph = self.phase[: len(dt)]
        return {p: float(dt[ph == p].sum()) for p in PHASES}

    def corner_limiter_time(self) -> dict:
        dt = np.diff(self.t)
        corner = self.phase[: len(dt)] == "corner"
        lim = self.limiter[: len(dt)]
        return {ax: float(dt[corner & (lim == ax)].sum()) for ax in ("front", "rear")}

    def summary(self) -> str:
        pt, lt = self.phase_time(), self.corner_limiter_time()
        lines = [
            f"lap time         : {self.lap_time:8.3f} s",
            f"distance         : {self.s[-1]:8.1f} m",
            f"avg / max / min v: {self.s[-1] / self.lap_time * 3.6:6.1f} / "
            f"{self.v.max() * 3.6:6.1f} / {self.v.min() * 3.6:6.1f} km/h",
            f"max lat accel    : {np.abs(self.a_y).max() / G:6.2f} g",
            "time share       : " + ", ".join(
                f"{p} {pt[p] / self.lap_time * 100:4.1f}%" for p in PHASES),
            f"corner limited by: front axle {lt['front']:.2f} s, rear axle {lt['rear']:.2f} s",
        ]
        return "\n".join(lines)


class _Params:
    """Flattened vehicle parameters for the scalar marching loops."""

    def __init__(self, veh: Vehicle):
        a, t, p = veh.aero, veh.tyre, veh.powertrain
        self.m, self.h, self.L = veh.mass, veh.cg_height, veh.wheelbase
        self.wf = veh.weight_dist_front
        self.crr = veh.rolling_resistance
        self.half_rho = 0.5 * a.rho
        self.cla, self.cda, self.bal, self.hcp = a.ClA, a.CdA, a.balance_front, a.h_cp
        self.mux, self.muy, self.ks, self.fz0 = t.mu_x, t.mu_y, t.load_sensitivity, t.Fz0
        self.v_top = p.v_max
        self.f_torque = p.max_torque * p.gear_ratio * p.efficiency / p.wheel_radius
        self.p_wheel = p.max_power * p.efficiency

    def loads(self, v, ax):
        q = self.half_rho * v * v
        df, drag = q * self.cla, q * self.cda
        dfz = (self.m * ax * self.h + drag * self.hcp) / self.L
        fz_f = max(self.m * G * self.wf + self.bal * df - dfz, 0.0)
        fz_r = max(self.m * G * (1 - self.wf) + (1 - self.bal) * df + dfz, 0.0)
        return fz_f, fz_r, drag

    def grip(self, fz):
        s = max(1.0 - self.ks * (fz / 2.0 - self.fz0) / self.fz0, 0.2)
        return self.mux * s * fz, self.muy * s * fz

    def drive(self, v):
        if v >= self.v_top:
            return 0.0
        return min(self.f_torque, self.p_wheel / max(v, 0.1))


def _ellipse(fx_max, fy_max, fy):
    if fy_max <= 0:
        return 0.0
    r = fy / fy_max
    return fx_max * math.sqrt(1.0 - r * r) if r < 1.0 else 0.0


# ---------------------------------------------------------------------------
def corner_speed(veh: Vehicle, kappa: np.ndarray, iters: int = 50):
    """Steady-state (a_x = 0) speed limit and the axle that saturates first."""
    k = np.abs(np.asarray(kappa, dtype=float))
    v_top = veh.powertrain.v_max * 0.999
    wf = veh.weight_dist_front

    def utilisation(v):
        fz_f, fz_r = veh.axle_loads(v, 0.0)
        _, fy_f = veh.axle_grip(fz_f)
        fx_r, fy_r = veh.axle_grip(fz_r)
        f_lat = veh.mass * v**2 * k
        u_f = f_lat * wf / np.maximum(fy_f, 1e-9)
        f_res = veh.resistance(v, fz_f + fz_r)
        u_r = np.sqrt((f_lat * (1 - wf) / np.maximum(fy_r, 1e-9)) ** 2
                      + (f_res / np.maximum(fx_r, 1e-9)) ** 2)
        return u_f, u_r

    lo = np.zeros_like(k)
    hi = np.full_like(k, v_top)
    u_f, u_r = utilisation(hi)
    free = np.maximum(u_f, u_r) <= 1.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        u_f, u_r = utilisation(mid)
        ok = np.maximum(u_f, u_r) <= 1.0
        lo = np.where(ok, mid, lo)
        hi = np.where(ok, hi, mid)
    v_lim = np.where(free, v_top, lo)
    u_f, u_r = utilisation(v_lim)
    limiter = np.where(u_f >= u_r, "front", "rear")
    return v_lim, limiter


def _forward(p: _Params, k, vmax, ds, v0):
    n = len(k)
    v = np.empty(n)
    v[0] = min(v0, vmax[0])
    rear_share = 1.0 - p.wf
    for i in range(n - 1):
        vi = v[i]
        f_lat_r = p.m * vi * vi * k[i] * rear_share
        ax = 0.0
        for _ in range(3):
            fz_f, fz_r, drag = p.loads(vi, ax)
            fx_max, fy_max = p.grip(fz_r)
            fx = min(p.drive(vi), _ellipse(fx_max, fy_max, f_lat_r))
            ax = (fx - drag - p.crr * (fz_f + fz_r)) / p.m
        v[i + 1] = min(math.sqrt(max(vi * vi + 2 * ax * ds, 0.0)), vmax[i + 1])
    return v


def _backward(p: _Params, k, vmax, ds, v_end):
    n = len(k)
    v = np.empty(n)
    v[-1] = min(v_end, vmax[-1])
    for i in range(n - 1, 0, -1):
        vi = v[i]
        f_lat = p.m * vi * vi * k[i]
        ax = 0.0
        for _ in range(3):
            fz_f, fz_r, drag = p.loads(vi, ax)
            fxf, fyf = p.grip(fz_f)
            fxr, fyr = p.grip(fz_r)
            f_brake = (_ellipse(fxf, fyf, f_lat * p.wf)
                       + _ellipse(fxr, fyr, f_lat * (1 - p.wf)))
            ax = -(f_brake + drag + p.crr * (fz_f + fz_r)) / p.m
        v[i - 1] = min(math.sqrt(vi * vi - 2 * ax * ds), vmax[i - 1])
    return v


def simulate(veh: Vehicle, track: Track, v_start: float | None = None) -> LapResult:
    """Run one lap.

    Closed tracks are a flying lap (track is tiled 3x, the middle lap is kept so the
    start speed is self-consistent).  Open tracks start at ``v_start`` (default 0 =
    standing start, e.g. autocross) and finish at the corner-limited speed.
    """
    n, ds = len(track.s), track.ds
    p = _Params(veh)

    if track.closed:
        k = np.abs(np.tile(track.kappa, 3))
        vmax, lim = corner_speed(veh, np.tile(track.kappa, 3))
        v0 = vmax.min()
        v = np.minimum(_forward(p, k, vmax, ds, v0), _backward(p, k, vmax, ds, vmax[-1]))
        sl = slice(n, 2 * n + 1)                       # middle lap + closing point
        v, vmax, lim, k = v[sl], vmax[sl], lim[sl], k[sl]
        s = np.arange(n + 1) * ds
        kappa_signed = np.append(track.kappa, track.kappa[0])
    else:
        k = np.abs(track.kappa)
        vmax, lim = corner_speed(veh, track.kappa)
        v0 = 0.0 if v_start is None else v_start
        v = np.minimum(_forward(p, k, vmax, ds, v0), _backward(p, k, vmax, ds, vmax[-1]))
        s = track.s.copy()
        kappa_signed = track.kappa

    v_seg = 0.5 * (v[1:] + v[:-1])
    dt = ds / np.maximum(v_seg, 1e-6)
    t = np.concatenate([[0.0], np.cumsum(dt)])
    a_x = np.append(np.diff(v**2) / (2 * ds), 0.0)
    a_y = v**2 * kappa_signed

    phase = np.full(len(v), "cruise", dtype=object)
    phase[a_x > 0.05] = "accel"
    phase[a_x < -0.05] = "brake"
    phase[(v >= 0.995 * vmax) & (vmax < 0.99 * p.v_top) & (k > 1e-3)] = "corner"

    return LapResult(float(t[-1]), s, v, vmax, a_x, a_y, t, phase.astype(str), lim, track.closed)
