"""GGV performance envelope: precomputed ax limits on a (v, ay) grid with fast lookup.

Grid coordinates:

* speed: uniform from 0 to exactly the top speed (spacing ≤ dv);
* lateral: θ_j uniform on [0, π/2] with ay = sin(θ_j)·ay_max(v). Near the cornering limit the
  friction ellipse gives ax ∝ √(1 − (ay/ay_max)²) = cos θ, which is smooth in θ, so linear
  interpolation in θ stays accurate right up to the limit.

``ay_max(v)`` is interpolated linearly in v² (exact for the point-mass model, whose cornering
limit is linear in v²). ``ax`` tables are interpolated bilinearly in (v, θ).
At θ = π/2 (ay = ay_max) the car can hold its speed, so ax_acc ≥ 0 ≥ ax_brk on the limit
(see :mod:`fsk_lapsim.models`).

The acceleration limit is the minimum of independent constraints (each feasible set is an
interval starting at the reference state). The GGV stores one smooth table per **tyre**
constraint (model B: front lateral grip as the front unloads, rear traction) and evaluates the
powertrain cap ``power_ax(v)`` in closed form; the lookup returns the minimum. The kinks where
one constraint takes over from another are thus exact instead of being smeared by the
interpolation.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .models import VehicleModel, make_model, top_speed
from .vehicle import Vehicle

HALF_PI = 0.5 * math.pi
ScalarLookup = Callable[[float, float], float]


@dataclass(frozen=True)
class GGV:
    """Precomputed performance envelope of one vehicle/model.

    Attributes:
        v_grid: Speed grid [m/s], uniform from 0.
        theta_grid: Lateral grid angle θ [rad], uniform on [0, π/2].
        ay_max: Constant-speed cornering limit at each grid speed [m/s²].
        ax_acc: Tyre-limited max acceleration per constraint [m/s²], shape (n_parts, n_v, n_θ)
            (no powertrain cap).
        ax_brk: Max deceleration as negative ax [m/s²], shape (n_v, n_θ).
        v_top: Reachable top speed [m/s] (rpm- or drag-limited).
        model_name: Name of the vehicle model used.
    """

    v_grid: NDArray[np.float64]
    theta_grid: NDArray[np.float64]
    ay_max: NDArray[np.float64]
    ax_acc: NDArray[np.float64]
    ax_brk: NDArray[np.float64]
    v_top: float
    model_name: str
    power: tuple[float, float, float, float, float, float]
    """(F_torque [N], P·η [W], v_rpm [m/s], R0 [N], k_v2 [N·s²/m²], m [kg]) for power_ax."""

    @classmethod
    def build(cls, model: VehicleModel, dv_mps: float = 0.25, n_theta: int = 73) -> GGV:
        """Evaluate ``model`` on the grid (vectorised)."""
        v_top = top_speed(model)
        # end the grid exactly at the top speed: the drive force drops to 0 beyond the rpm
        # limit, and a cell straddling that jump would smear it into the interpolation
        n_v = max(math.ceil(v_top / dv_mps), 2)
        v_grid = np.linspace(0.0, v_top, n_v + 1)
        theta = np.linspace(0.0, HALF_PI, n_theta)
        ay_max = model.ay_max(v_grid)
        vv = np.repeat(v_grid[:, None], n_theta, axis=1)
        ay = np.sin(theta)[None, :] * ay_max[:, None]
        ay[:, -1] = ay_max  # exactly on the limit
        ax_acc = np.stack(list(model.ax_accel_parts(vv, ay).values()))
        ax_brk = model.ax_brake_min(vv, ay)
        if np.isnan(ax_acc).any() or np.isnan(ax_brk).any():  # pragma: no cover - guard
            raise RuntimeError("GGV grid contains infeasible points")
        veh = model.vehicle
        pt, aero, tire = veh.powertrain, veh.aero, veh.tire
        q = 0.5 * aero.air_density_kg_m3
        power = (
            pt.torque_force_n(tire.radius_m),
            pt.max_power_w * pt.efficiency,
            veh.v_max_mps,
            tire.rolling_resistance_coeff * veh.weight_n,
            q * aero.cda_m2 + tire.rolling_resistance_coeff * q * aero.cla_m2,
            veh.mass_kg,
        )
        return cls(v_grid, theta, ay_max, ax_acc, ax_brk, v_top, model.name, power)

    def power_ax_at(self, v: ArrayLike) -> NDArray[np.float64]:
        """Powertrain-limited acceleration (F_pt − resistance)/m [m/s²] (closed form)."""
        f_t, p_eta, v_rpm, r0, k2, m = self.power
        v = np.asarray(v, float)
        f = np.minimum(f_t, p_eta / np.maximum(v, 1e-3))
        f = np.where(v <= v_rpm * (1.0 + 1e-9), f, 0.0)
        return (f - r0 - k2 * v * v) / m

    # -- vectorised lookups --------------------------------------------------------------
    def ay_max_at(self, v: ArrayLike) -> NDArray[np.float64]:
        """Cornering limit [m/s²] interpolated linearly in v²."""
        v = np.asarray(v, float)
        return np.interp(v * v, self.v_grid**2, self.ay_max)

    def _coords(
        self, v: NDArray[np.float64], ay: NDArray[np.float64]
    ) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.int64], NDArray[np.float64]]:
        dv = self.v_grid[1] - self.v_grid[0]
        x = np.clip(v / dv, 0.0, len(self.v_grid) - 1.0)
        i = np.minimum(x.astype(np.int64), len(self.v_grid) - 2)
        fx = x - i
        u = np.clip(np.abs(ay) / self.ay_max_at(v), 0.0, 1.0)
        dth = self.theta_grid[1] - self.theta_grid[0]
        y = np.arcsin(u) / dth
        j = np.minimum(y.astype(np.int64), len(self.theta_grid) - 2)
        return i, fx, j, y - j

    def _bilinear(
        self, table: NDArray[np.float64], v: ArrayLike, ay: ArrayLike
    ) -> NDArray[np.float64]:
        v, ay = np.broadcast_arrays(np.asarray(v, float), np.asarray(ay, float))
        i, fx, j, fy = self._coords(v, ay)
        a0 = table[i, j] + fx * (table[i + 1, j] - table[i, j])
        a1 = table[i, j + 1] + fx * (table[i + 1, j + 1] - table[i, j + 1])
        return a0 + fy * (a1 - a0)

    def ax_acc_at(self, v: ArrayLike, ay: ArrayLike) -> NDArray[np.float64]:
        """Max acceleration [m/s²]: min(interpolated tyre limits, power limit)."""
        tyre = np.min([self._bilinear(t, v, ay) for t in self.ax_acc], axis=0)
        return np.minimum(tyre, self.power_ax_at(v))

    def ax_brk_at(self, v: ArrayLike, ay: ArrayLike) -> NDArray[np.float64]:
        """Interpolated max deceleration (negative) [m/s²]."""
        return self._bilinear(self.ax_brk, v, ay)

    def curvature_speed(self, kappa: ArrayLike, n_iter: int = 64) -> NDArray[np.float64]:
        """Speed [m/s] where v²·|κ| = ay_max(v), capped at the top speed.

        Vectorised bisection (ay_max(v)/v² decreases with v, so the root is unique).
        """
        k = np.abs(np.asarray(kappa, float))
        lo = np.zeros_like(k)
        hi = np.full_like(k, self.v_top)
        free = self.ay_max_at(hi) - hi * hi * k >= 0.0
        for _ in range(n_iter):
            mid = 0.5 * (lo + hi)
            ok = self.ay_max_at(mid) - mid * mid * k >= 0.0
            lo = np.where(ok, mid, lo)
            hi = np.where(ok, hi, mid)
        return np.where(free, self.v_top, lo)

    # -- scalar lookups for the integration loops ----------------------------------------
    def scalar_lookups(self) -> tuple[ScalarLookup, ScalarLookup]:
        """Pure-Python (acc, brk) lookups ``f(v, ay) -> ax`` for tight loops."""
        vg = self.v_grid
        inv_dv = 1.0 / (vg[1] - vg[0])
        nv = len(vg)
        v2 = (vg * vg).tolist()
        aym = self.ay_max.tolist()
        nth = len(self.theta_grid)
        inv_dth = 1.0 / (self.theta_grid[1] - self.theta_grid[0])
        acc_parts = [t.tolist() for t in self.ax_acc]
        brk = self.ax_brk.tolist()
        asin = math.asin
        f_t, p_eta, v_rpm, r0, k2, mass = self.power
        v_rpm_tol = v_rpm * (1.0 + 1e-9)

        def make(tables: list[list[list[float]]], with_power: bool) -> ScalarLookup:
            def look(v: float, ay: float) -> float:
                x = v * inv_dv
                i = int(x)
                if i > nv - 2:
                    i = nv - 2
                fx = x - i
                if fx > 1.0:
                    fx = 1.0
                vv = v * v
                w = (vv - v2[i]) / (v2[i + 1] - v2[i])
                if w > 1.0:
                    w = 1.0
                a_lim = aym[i] + w * (aym[i + 1] - aym[i])
                u = (ay if ay >= 0.0 else -ay) / a_lim
                th = HALF_PI if u >= 1.0 else asin(u)
                y = th * inv_dth
                j = int(y)
                if j > nth - 2:
                    j = nth - 2
                fy = y - j
                a = 1e30
                for table in tables:
                    row0 = table[i]
                    row1 = table[i + 1]
                    c0 = row0[j] + fx * (row1[j] - row0[j])
                    c1 = row0[j + 1] + fx * (row1[j + 1] - row0[j + 1])
                    c = c0 + fy * (c1 - c0)
                    if c < a:
                        a = c
                if with_power:
                    if v > v_rpm_tol:
                        f = 0.0
                    else:
                        f = p_eta / (v if v > 1e-3 else 1e-3)
                        if f > f_t:
                            f = f_t
                    ap = (f - r0_ - k2 * vv) / mass
                    if ap < a:
                        a = ap
                return a

            return look

        r0_ = r0
        return make(acc_parts, True), make([brk], False)


@lru_cache(maxsize=256)
def cached_ggv(
    vehicle: Vehicle, model: str = "axle", dv_mps: float = 0.25, n_theta: int = 73
) -> GGV:
    """GGV for a (hashable, frozen) vehicle, memoised — sweeps reuse identical vehicles."""
    return GGV.build(make_model(vehicle, model), dv_mps, n_theta)
