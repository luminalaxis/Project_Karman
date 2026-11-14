"""Quasi-steady-state lap solver.

1. Corner speed limit  v_max(s): highest speed at which each axle can carry its lateral force
   (steady-state share of m*a_y plus the yaw-acceleration term Iz*r_dot/L) with load-sensitive
   tyres, lateral load transfer and ride-height-dependent aero; the rear also carries the
   drag + rolling-resistance drive force on its friction ellipse.  Vectorised bisection.
2. Forward pass: accelerate with min(motor force, rear-axle grip left on the ellipse), capped
   so the (unloaded) front axle still carries its lateral force.
3. Backward pass: brake with the grip left on both axles (ideal brake balance), capped so the
   (unloaded) rear axle still carries its lateral force.
4. v(s) = min(forward, backward); lap time = sum(ds / mean speed).

The physics lives in :mod:`lapsim.physics` and is shared by all steps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .physics import Model, ellipse
from .track import Track, smooth
from .vehicle import G, Vehicle

PHASES = ("accel", "corner", "brake", "cruise")
TRANSIENT_WINDOW = 3.0   # [m] smoothing of curvature before differentiating for yaw acceleration


@dataclass
class LapResult:
    lap_time: float
    s: np.ndarray          # [m]
    v: np.ndarray          # [m/s]
    v_corner: np.ndarray   # corner-limited speed [m/s]
    a_x: np.ndarray        # [m/s^2]
    a_y: np.ndarray        # [m/s^2] (signed, + = left)
    t: np.ndarray          # cumulative time at each s [s]
    phase: np.ndarray      # one of PHASES per point
    limiter: np.ndarray    # "front" / "rear" axle saturating first in the corner
    closed: bool
    kappa: np.ndarray      # signed path curvature [1/m]
    kappa_rate: np.ndarray  # d kappa / ds [1/m^2]
    x: np.ndarray          # path coordinates [m]
    y: np.ndarray

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


def kappa_rate(kappa: np.ndarray, ds: float, closed: bool, window: float = TRANSIENT_WINDOW):
    win = max(int(round(window / ds)), 1) | 1
    k = smooth(kappa, win, closed)
    if closed:
        return (np.roll(k, -1) - np.roll(k, 1)) / (2 * ds)
    return np.gradient(k, ds)


# ---------------------------------------------------------------------------
def utilisation(model: Model, v, kappa, krate):
    """(front, rear) axle friction utilisation at constant speed (a_x = 0)."""
    fzf, fzr, drag, _, _ = model.loads(v, 0.0)
    ay = v * v * np.abs(kappa)
    cf, cr = model.lateral_coeffs(kappa, krate)
    _, fyf = model.axle_grip(fzf, model.llt_f * ay)
    fxr, fyr = model.axle_grip(fzr, model.llt_r * ay)
    u_f = v * v * cf / np.maximum(fyf, 1e-9)
    f_res = drag + model.crr * (fzf + fzr)
    u_r = np.sqrt((v * v * cr / np.maximum(fyr, 1e-9)) ** 2 + (f_res / np.maximum(fxr, 1e-9)) ** 2)
    return u_f, u_r


def corner_speed(model: Model, kappa: np.ndarray, krate: np.ndarray, iters: int = 50):
    """Steady-state speed limit and the axle that saturates first."""
    kappa = np.asarray(kappa, dtype=float)
    v_top = model.v_top * 0.999
    lo = np.zeros_like(kappa)
    hi = np.full_like(kappa, v_top)
    u_f, u_r = utilisation(model, hi, kappa, krate)
    free = np.maximum(u_f, u_r) <= 1.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        u_f, u_r = utilisation(model, mid, kappa, krate)
        ok = np.maximum(u_f, u_r) <= 1.0
        lo = np.where(ok, mid, lo)
        hi = np.where(ok, hi, mid)
    v_lim = np.where(free, v_top, lo)
    u_f, u_r = utilisation(model, v_lim, kappa, krate)
    return v_lim, np.where(u_f >= u_r, "front", "rear")


def _solve_ax(g, tol=1e-4, max_eval=6):
    """Solve a_x = g(a_x) (acceleration <-> load transfer) by the secant method.

    Plain fixed-point iteration oscillates under hard braking (more transfer -> different
    grip -> different decel); the secant converges in 3-4 evaluations.
    """
    x0 = 0.0
    h0 = g(x0) - x0
    x1 = x0 + h0
    h1 = g(x1) - x1
    for _ in range(max_eval - 2):
        if abs(h1) < tol:
            break
        den = h1 - h0
        x2 = x1 - h1 * (x1 - x0) / den if abs(den) > 1e-12 else x1 + h1
        x0, h0 = x1, h1
        x1 = x2
        h1 = g(x1) - x1
    return x1 + h1 if abs(h1) >= tol else x1


def _cap_lateral(p: Model, v, ax, f_lat, dfz, axle, hf, hr, iters=10):
    """Limit a_x so that ``axle`` (0 front, 1 rear) can still carry its lateral force.

    Longitudinal load transfer unloads the front under acceleration and the rear under
    braking; a_x = 0 is always feasible below the corner limit, so bisect between 0 and ax.
    """
    def ok(a):
        loads = p.loads(v, a, hf, hr, 1)
        return p.axle_grip(loads[axle], dfz)[1] >= f_lat
    if f_lat <= 0 or ok(ax):
        return ax
    good, bad = 0.0, ax
    for _ in range(iters):
        mid = 0.5 * (good + bad)
        if ok(mid):
            good = mid
        else:
            bad = mid
    return good


def _forward(p: Model, kabs, cf, cr, vmax, ds, v0):
    n = len(kabs)
    v = np.empty(n)
    v[0] = min(v0, vmax[0])
    rh = [p.hf0, p.hr0]
    for i in range(n - 1):
        vi = v[i]
        v2 = vi * vi
        f_lat_r = v2 * cr[i]
        dfz_r = p.llt_r * v2 * kabs[i]
        f_drive = p.drive(vi)

        def g(ax):
            fzf, fzr, drag, rh[0], rh[1] = p.loads(vi, ax, rh[0], rh[1], 1)
            fx_max, fy_max = p.axle_grip(fzr, dfz_r)
            fx = min(f_drive, ellipse(fx_max, fy_max, f_lat_r))
            return (fx - drag - p.crr * (fzf + fzr)) / p.m

        ax = _solve_ax(g)
        hf, hr = rh
        if ax > 0:
            ax = _cap_lateral(p, vi, ax, v2 * cf[i], p.llt_f * v2 * kabs[i], 0, hf, hr)
        v[i + 1] = min(math.sqrt(max(v2 + 2 * ax * ds, 0.0)), vmax[i + 1])
    return v


def _backward(p: Model, kabs, cf, cr, vmax, ds, v_end):
    n = len(kabs)
    v = np.empty(n)
    v[-1] = min(v_end, vmax[-1])
    rh = [p.hf0, p.hr0]
    for i in range(n - 1, 0, -1):
        vi = v[i]
        v2 = vi * vi
        ay = v2 * kabs[i]
        fy_f, fy_r = v2 * cf[i], v2 * cr[i]

        def g(ax):
            fzf, fzr, drag, rh[0], rh[1] = p.loads(vi, ax, rh[0], rh[1], 1)
            fxf, fyf = p.axle_grip(fzf, p.llt_f * ay)
            fxr, fyr = p.axle_grip(fzr, p.llt_r * ay)
            f_brake = ellipse(fxf, fyf, fy_f) + ellipse(fxr, fyr, fy_r)
            return -(f_brake + drag + p.crr * (fzf + fzr)) / p.m

        ax = _solve_ax(g)
        hf, hr = rh
        ax = _cap_lateral(p, vi, ax, v2 * cr[i], p.llt_r * ay, 1, hf, hr)
        v[i - 1] = min(math.sqrt(v2 - 2 * ax * ds), vmax[i - 1])
    return v


def simulate(veh: Vehicle, track: Track, v_start: float | None = None) -> LapResult:
    """Run one lap.

    Closed tracks are a flying lap, solved from the slowest apex so the start speed is
    self-consistent.  Open tracks start at ``v_start`` (default 0 =
    standing start, e.g. autocross) and finish at the corner-limited speed.
    """
    n, ds = len(track.s), track.ds
    model = Model(veh)
    krate = kappa_rate(track.kappa, ds, track.closed)

    if track.closed:
        # Start the lap at the slowest corner apex: there v = v_max exactly, so a single lap
        # of forward and backward passes is already self-consistent (no lap tiling needed).
        vmax1, lim1 = corner_speed(model, track.kappa, krate)
        i0 = int(np.argmin(vmax1))
        idx = np.append(np.roll(np.arange(n), -i0), i0)       # n + 1 points, closed
        k_s, kr_s, vmax, lim = track.kappa[idx], krate[idx], vmax1[idx], lim1[idx]
        v0 = v_end = vmax1[i0]
    else:
        k_s, kr_s = track.kappa, krate
        vmax, lim = corner_speed(model, k_s, kr_s)
        v0, v_end = (0.0 if v_start is None else v_start), vmax[-1]
    cf, cr = model.lateral_coeffs(k_s, kr_s)
    kabs = np.abs(k_s)
    v = np.minimum(_forward(model, kabs, cf, cr, vmax, ds, v0),
                   _backward(model, kabs, cf, cr, vmax, ds, v_end))

    if track.closed:
        v = v[:n][np.roll(np.arange(n), i0)]                    # undo the rotation
        v = np.append(v, v[0])
        vmax, lim = np.append(vmax1, vmax1[0]), np.append(lim1, lim1[0])
        k_s, kr_s = np.append(track.kappa, track.kappa[0]), np.append(krate, krate[0])
        s = np.arange(n + 1) * ds
        x, y = np.append(track.x, track.x[0]), np.append(track.y, track.y[0])
    else:
        s, x, y = track.s.copy(), track.x, track.y

    v_seg = 0.5 * (v[1:] + v[:-1])
    dt = ds / np.maximum(v_seg, 1e-6)
    t = np.concatenate([[0.0], np.cumsum(dt)])
    a_x = np.append(np.diff(v**2) / (2 * ds), 0.0)
    a_y = v**2 * k_s

    phase = np.full(len(v), "cruise", dtype=object)
    phase[a_x > 0.05] = "accel"
    phase[a_x < -0.05] = "brake"
    phase[(v >= 0.995 * vmax) & (vmax < 0.99 * model.v_top) & (np.abs(k_s) > 1e-3)] = "corner"

    return LapResult(float(t[-1]), s, v, vmax, a_x, a_y, t, phase.astype(str), lim,
                     track.closed, k_s, kr_s, x, y)
