"""Quasi-steady-state (QSS) lap solver.

1. Discretise the track into nodes s_i = i·ds and curvature κ_i.
2. Curvature-limited speed at every node: v²·|κ| = ay_max(v) (root found on the GGV's
   ay_max(v), which depends on v through downforce), capped at the top speed. ay_max is the
   constant-speed limit, so on the limit ax_acc ≥ 0 ≥ ax_brk and both passes can hold the
   clamp in long constant-radius corners (DECISIONS D-08).
3. Forward pass (acceleration, Heun/RK2 in v² over distance using the GGV acceleration limit
   at the current ay = v²κ: remaining friction-ellipse capacity, powertrain, drag, rolling).
4. Backward pass (braking limit, same scheme from the end).
5. v = min(forward, backward).
6. Closed tracks (flying lap): repeat with start speed ← forward end speed and end speed ←
   backward start speed until the lap is periodic.

Integration uses v² linear in s inside a step, so dt = 2·ds/(v_i + v_{i+1}) is exact for
constant acceleration and a standing start (v_0 = 0) gives a finite first step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .energy import EnergyBreakdown, lap_energy
from .ggv import GGV, ScalarLookup, cached_ggv
from .models import make_model
from .track import Track
from .vehicle import Vehicle

FloatArray = NDArray[np.float64]
StartMode = Literal["auto", "standing", "flying"]

LIMIT_CORNER, LIMIT_ACCEL, LIMIT_BRAKE = 0, 1, 2


@dataclass(frozen=True)
class SolverOptions:
    """Numerical options.

    Attributes:
        ds_m: Distance step [m].
        model: ``"axle"`` (model B, default) or ``"point_mass"`` (model A).
        ggv_dv_mps: GGV speed grid step [m/s].
        ggv_n_theta: Number of lateral grid points of the GGV.
        max_periodic_iter: Max flying-lap iterations.
        periodic_tol_mps: Convergence tolerance of the lap start/end speed [m/s].
    """

    ds_m: float = 0.25
    model: str = "axle"
    ggv_dv_mps: float = 0.25
    ggv_n_theta: int = 73
    max_periodic_iter: int = 60
    periodic_tol_mps: float = 1e-7


@dataclass
class LapResult:
    """Simulation output. Node arrays have N+1 entries, segment arrays N.

    Attributes:
        track_name: Track label.
        model: Vehicle model name.
        closed: Whether the lap is periodic (flying) on a closed track.
        s: Distance [m] (nodes).
        kappa: Curvature [1/m] (nodes).
        v: Speed [m/s] (nodes).
        v_lim: Curvature/top-speed limit [m/s] (nodes).
        ax: Longitudinal acceleration [m/s²] (nodes; mean of adjacent segments).
        ay: Lateral acceleration v²κ [m/s²] (nodes, signed; > 0 = left).
        t: Elapsed time [s] (nodes).
        wheel_loads: Vertical loads [N], shape (4, N+1): FL, FR, RL, RR.
        limit: 0 = curvature limited, 1 = acceleration limited, 2 = braking limited (nodes).
        energy: Energy and powertrain breakdown (segments + totals).
        periodic_iterations: Flying-lap iterations used (0 for open laps).
    """

    track_name: str
    model: str
    closed: bool
    s: FloatArray
    kappa: FloatArray
    v: FloatArray
    v_lim: FloatArray
    ax: FloatArray
    ay: FloatArray
    t: FloatArray
    wheel_loads: FloatArray
    limit: NDArray[np.int64]
    energy: EnergyBreakdown
    periodic_iterations: int = 0
    extras: dict[str, FloatArray] = field(default_factory=dict)

    @property
    def lap_time_s(self) -> float:
        """Lap time [s]."""
        return float(self.t[-1])

    @property
    def length_m(self) -> float:
        """Distance covered [m]."""
        return float(self.s[-1])

    @property
    def mean_speed_mps(self) -> float:
        """Average speed [m/s]."""
        return self.length_m / self.lap_time_s

    def to_frame(self) -> pd.DataFrame:
        """Node channels as a table (segment quantities assigned to their start node)."""
        e = self.energy

        def seg(a: FloatArray) -> FloatArray:
            return np.append(a, np.nan)

        return pd.DataFrame(
            {
                "s_m": self.s,
                "t_s": self.t,
                "v_kph": self.v * 3.6,
                "v_lim_kph": self.v_lim * 3.6,
                "ax_g": self.ax / 9.81,
                "ay_g": self.ay / 9.81,
                "kappa_1pm": self.kappa,
                "fz_fl_n": self.wheel_loads[0],
                "fz_fr_n": self.wheel_loads[1],
                "fz_rl_n": self.wheel_loads[2],
                "fz_rr_n": self.wheel_loads[3],
                "limit": np.array(["corner", "accel", "brake"])[self.limit],
                "fx_tyre_n": seg(e.fx_tyre_n),
                "drag_n": seg(e.drag_n),
                "p_battery_kw": seg(e.p_battery_w / 1e3),
                "motor_rpm": seg(e.motor_rpm),
                "motor_torque_nm": seg(e.motor_torque_nm),
            }
        )

    def summary(self) -> dict[str, float]:
        """Scalar key figures."""
        e = self.energy
        return {
            "lap_time_s": self.lap_time_s,
            "length_m": self.length_m,
            "mean_speed_kph": 3.6 * self.mean_speed_mps,
            "max_speed_kph": 3.6 * float(self.v.max()),
            "min_speed_kph": 3.6 * float(self.v.min()),
            "max_ay_g": float(np.abs(self.ay).max()) / 9.81,
            "energy_kwh": e.battery_net_j / 3.6e6,
            "energy_no_regen_kwh": e.battery_drive_j / 3.6e6,
            "regen_kwh": e.regen_j / 3.6e6,
            "mean_power_kw": e.battery_net_j / self.lap_time_s / 1e3,
            "max_power_kw": float(e.p_battery_w.max()) / 1e3,
            "fraction_accel": float(np.mean(self.limit == LIMIT_ACCEL)),
            "fraction_brake": float(np.mean(self.limit == LIMIT_BRAKE)),
            "fraction_corner": float(np.mean(self.limit == LIMIT_CORNER)),
        }


# --------------------------------------------------------------------------------------------
# Integration passes (pure Python loops over floats: much faster than numpy scalars)
# --------------------------------------------------------------------------------------------
def _forward(
    v2lim: list[float],
    kap: list[float],
    ds: float,
    v2_start: float,
    acc: ScalarLookup,
    acc1: ScalarLookup | None = None,
    w: list[float] | None = None,
) -> list[float]:
    n = len(v2lim)
    out = [0.0] * n
    v2 = min(v2_start, v2lim[0])
    out[0] = v2
    sqrt = math.sqrt
    two_ds = 2.0 * ds
    for i in range(n - 1):
        if w is None or acc1 is None:
            a1 = acc(sqrt(v2), v2 * kap[i])
        else:
            wi = w[i]
            v = sqrt(v2)
            a1 = (1.0 - wi) * acc(v, v2 * kap[i]) + wi * acc1(v, v2 * kap[i])
        lim = v2lim[i + 1]
        p = v2 + two_ds * a1
        if p > lim:
            p = lim
        elif p < 0.0:
            p = 0.0
        if w is None or acc1 is None:
            a2 = acc(sqrt(p), p * kap[i + 1])
        else:
            wj = w[i + 1]
            vp = sqrt(p)
            a2 = (1.0 - wj) * acc(vp, p * kap[i + 1]) + wj * acc1(vp, p * kap[i + 1])
        v2 = v2 + ds * (a1 + a2)
        if v2 > lim:
            v2 = lim
        elif v2 < 0.0:
            v2 = 0.0
        out[i + 1] = v2
    return out


def _backward(
    v2lim: list[float],
    kap: list[float],
    ds: float,
    v2_end: float,
    brk: ScalarLookup,
    brk1: ScalarLookup | None = None,
    w: list[float] | None = None,
) -> list[float]:
    n = len(v2lim)
    out = [0.0] * n
    v2 = min(v2_end, v2lim[-1])
    out[-1] = v2
    sqrt = math.sqrt
    two_ds = 2.0 * ds
    for i in range(n - 1, 0, -1):
        if w is None or brk1 is None:
            d1 = -brk(sqrt(v2), v2 * kap[i])
        else:
            wi = w[i]
            v = sqrt(v2)
            d1 = -((1.0 - wi) * brk(v, v2 * kap[i]) + wi * brk1(v, v2 * kap[i]))
        lim = v2lim[i - 1]
        p = v2 + two_ds * d1
        if p > lim:
            p = lim
        elif p < 0.0:
            p = 0.0
        if w is None or brk1 is None:
            d2 = -brk(sqrt(p), p * kap[i - 1])
        else:
            wj = w[i - 1]
            vp = sqrt(p)
            d2 = -((1.0 - wj) * brk(vp, p * kap[i - 1]) + wj * brk1(vp, p * kap[i - 1]))
        v2 = v2 + ds * (d1 + d2)
        if v2 > lim:
            v2 = lim
        elif v2 < 0.0:
            v2 = 0.0
        out[i - 1] = v2
    return out


@dataclass
class SpeedProfile:
    """Raw solver output before post-processing."""

    v: FloatArray
    v_lim: FloatArray
    v_fwd: FloatArray
    v_bwd: FloatArray
    iterations: int


def solve_speed_profile(
    ggv: GGV,
    kappa: FloatArray,
    ds: float,
    periodic: bool,
    v_start: float | None = None,
    v_end: float | None = None,
    ggv_alt: GGV | None = None,
    weight: FloatArray | None = None,
    max_iter: int = 60,
    tol: float = 1e-7,
) -> SpeedProfile:
    """Solve the QSS speed profile on given node curvatures.

    Args:
        ggv: Envelope of the vehicle (state 0).
        kappa: Node curvatures [1/m] (N+1).
        ds: Node spacing [m].
        periodic: Flying lap (start speed = end speed).
        v_start: Start speed [m/s] for non-periodic laps (``None`` = limit speed, 0 = standing).
        v_end: Optional end-speed cap [m/s] for non-periodic laps.
        ggv_alt: Envelope of an alternative aero state (active aero), blended by ``weight``.
        weight: Per-node blend weight of ``ggv_alt`` (0 = state 0, 1 = alternative).
        max_iter: Max flying-lap iterations.
        tol: Flying-lap convergence tolerance on the start/end speed [m/s].
    """
    k = np.asarray(kappa, float)
    if ggv_alt is not None and weight is not None:
        w = np.clip(np.asarray(weight, float), 0.0, 1.0)
        vlim = _blended_curvature_speed(ggv, ggv_alt, k, w)
        acc1, brk1 = ggv_alt.scalar_lookups()
        wl: list[float] | None = w.tolist()
    else:
        vlim = ggv.curvature_speed(k)
        acc1 = brk1 = None
        wl = None
    acc, brk = ggv.scalar_lookups()
    v2lim = (vlim * vlim).tolist()
    kl = k.tolist()
    if not periodic:
        v2s = v2lim[0] if v_start is None else v_start**2
        v2e = v2lim[-1] if v_end is None else v_end**2
        f = _forward(v2lim, kl, ds, v2s, acc, acc1, wl)
        b = _backward(v2lim, kl, ds, v2e, brk, brk1, wl)
        vf, vb = np.sqrt(f), np.sqrt(b)
        return SpeedProfile(np.minimum(vf, vb), vlim, vf, vb, 0)
    v2s, v2e = v2lim[0], v2lim[-1]
    it = 0
    for it in range(1, max_iter + 1):  # noqa: B007 - iteration count is reported
        f = _forward(v2lim, kl, ds, v2s, acc, acc1, wl)
        b = _backward(v2lim, kl, ds, v2e, brk, brk1, wl)
        new_s, new_e = min(f[-1], v2lim[0]), min(b[0], v2lim[-1])
        done = (
            abs(math.sqrt(new_s) - math.sqrt(v2s)) < tol
            and abs(math.sqrt(new_e) - math.sqrt(v2e)) < tol
        )
        v2s, v2e = new_s, new_e
        if done:
            break
    vf, vb = np.sqrt(f), np.sqrt(b)
    return SpeedProfile(np.minimum(vf, vb), vlim, vf, vb, it)


def _blended_curvature_speed(g0: GGV, g1: GGV, k: FloatArray, w: FloatArray) -> FloatArray:
    v_top = (1.0 - w) * g0.v_top + w * g1.v_top
    lo = np.zeros_like(k)
    hi = v_top.copy()

    def margin(v: FloatArray) -> FloatArray:
        return (1.0 - w) * g0.ay_max_at(v) + w * g1.ay_max_at(v) - v * v * np.abs(k)

    free = margin(hi) >= 0.0
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        ok = margin(mid) >= 0.0
        lo = np.where(ok, mid, lo)
        hi = np.where(ok, hi, mid)
    return np.where(free, v_top, lo)


# --------------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------------
def simulate(
    vehicle: Vehicle,
    track: Track,
    options: SolverOptions | None = None,
    start: StartMode = "auto",
    v_start: float | None = None,
    v_end: float | None = None,
    alt_vehicle: Vehicle | None = None,
    alt_weight: FloatArray | None = None,
) -> LapResult:
    """Simulate one lap.

    Args:
        vehicle: Vehicle.
        track: Track (closed tracks default to a flying lap, open tracks to a standing start).
        options: Numerical options.
        start: ``"flying"`` (periodic, closed tracks only), ``"standing"`` (v0 = 0) or ``"auto"``.
        v_start: Explicit start speed for non-periodic laps [m/s] (overrides ``standing``).
        v_end: Optional speed cap at the end of a non-periodic lap [m/s].
        alt_vehicle: Alternative aero state (e.g. DRS open), same mass properties.
        alt_weight: Per-node blend weight towards ``alt_vehicle`` (0..1), length N+1.
    """
    opt = options or SolverOptions()
    s, kappa = track.discretize(opt.ds_m)
    ds = float(s[1] - s[0])
    ggv = cached_ggv(vehicle, opt.model, opt.ggv_dv_mps, opt.ggv_n_theta)
    ggv_alt = None
    if alt_vehicle is not None:
        if alt_weight is None or len(alt_weight) != len(s):
            raise ValueError("alt_weight must have one entry per node")
        ggv_alt = cached_ggv(alt_vehicle, opt.model, opt.ggv_dv_mps, opt.ggv_n_theta)
    if start == "auto":
        start = "flying" if track.closed else "standing"
    periodic = start == "flying"
    if periodic and not track.closed:
        raise ValueError("a flying lap needs a closed track")
    if not periodic and v_start is None and start == "standing":
        v_start = 0.0
    prof = solve_speed_profile(
        ggv,
        kappa,
        ds,
        periodic,
        v_start,
        v_end,
        ggv_alt,
        alt_weight,
        opt.max_periodic_iter,
        opt.periodic_tol_mps,
    )
    return build_result(
        vehicle, track.name, opt.model, s, kappa, prof, periodic, alt_vehicle, alt_weight
    )


def build_result(
    vehicle: Vehicle,
    track_name: str,
    model_name: str,
    s: FloatArray,
    kappa: FloatArray,
    prof: SpeedProfile,
    periodic: bool,
    alt_vehicle: Vehicle | None = None,
    alt_weight: FloatArray | None = None,
) -> LapResult:
    """Post-process a speed profile into a :class:`LapResult` (time, loads, energy)."""
    v = prof.v
    ds = np.diff(s)
    vsum = v[1:] + v[:-1]
    dt = 2.0 * ds / np.maximum(vsum, 1e-9)
    t = np.concatenate([[0.0], np.cumsum(dt)])
    ax_seg = (v[1:] ** 2 - v[:-1] ** 2) / (2.0 * ds)
    ax = np.empty_like(v)
    ax[1:-1] = 0.5 * (ax_seg[1:] + ax_seg[:-1])
    if periodic:
        ax[0] = ax[-1] = 0.5 * (ax_seg[0] + ax_seg[-1])
    else:
        ax[0], ax[-1] = ax_seg[0], ax_seg[-1]
    ay = v * v * kappa
    model = make_model(vehicle, model_name)
    loads = model.wheel_loads(v, ax, ay)
    if alt_vehicle is not None and alt_weight is not None:
        loads_alt = make_model(alt_vehicle, model_name).wheel_loads(v, ax, ay)
        loads = (1.0 - alt_weight) * loads + alt_weight * loads_alt
    tol = 1e-6
    limit = np.where(
        v >= prof.v_lim - tol,
        LIMIT_CORNER,
        np.where(prof.v_fwd <= prof.v_bwd, LIMIT_ACCEL, LIMIT_BRAKE),
    ).astype(np.int64)
    energy = lap_energy(vehicle, s, v, kappa, alt_vehicle, alt_weight, model_name)
    return LapResult(
        track_name=track_name,
        model=model_name,
        closed=periodic,
        s=s,
        kappa=kappa,
        v=v,
        v_lim=prof.v_lim,
        ax=ax,
        ay=ay,
        t=t,
        wheel_loads=loads,
        limit=limit,
        energy=energy,
        periodic_iterations=prof.iterations,
    )
