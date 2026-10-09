"""Aero sensitivity studies built on top of :func:`lapsim.solver.simulate`."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .solver import simulate
from .track import Track
from .vehicle import Vehicle


def lap_time(veh: Vehicle, track: Track, **kw) -> float:
    return simulate(veh, track, **kw).lap_time


def delta(veh: Vehicle, track: Track, d_cl: float = 0.0, d_cd: float = 0.0,
          balance: float | None = None, **kw):
    """Lap time change for fractional ClA/CdA changes (0.10 = +10 %)."""
    base = lap_time(veh, track, **kw)
    mod = veh.scaled_aero(1 + d_cl, 1 + d_cd)
    if balance is not None:
        mod = mod.with_changes(**{"aero.balance_front": balance})
    new = lap_time(mod, track, **kw)
    return base, new, new - base


def sweep(veh: Vehicle, track: Track, d_cl: np.ndarray, d_cd: np.ndarray, **kw) -> np.ndarray:
    """Lap time delta grid [s], rows = d_cd, columns = d_cl (fractions)."""
    base = lap_time(veh, track, **kw)
    out = np.empty((len(d_cd), len(d_cl)))
    for i, cd in enumerate(d_cd):
        for j, cl in enumerate(d_cl):
            out[i, j] = lap_time(veh.scaled_aero(1 + cl, 1 + cd), track, **kw) - base
    return out


def balance_sweep(veh: Vehicle, track: Track, balances: np.ndarray, **kw) -> np.ndarray:
    return np.array([lap_time(veh.with_changes(**{"aero.balance_front": float(b)}), track, **kw)
                     for b in balances])


@dataclass
class Coefficients:
    base: float
    dt_dcla: float     # [s per unit ClA]
    dt_dcda: float     # [s per unit CdA]
    dt_pct_cl: float   # [s per +1 % ClA]
    dt_pct_cd: float   # [s per +1 % CdA]

    @property
    def exchange_rate(self) -> float:
        """% downforce needed to pay back +1 % drag (lap-time neutral)."""
        return -self.dt_pct_cd / self.dt_pct_cl if self.dt_pct_cl else float("inf")

    @property
    def efficiency_equivalent(self) -> float:
        """Lap-time-neutral trade in absolute terms: units of ClA worth one unit of CdA."""
        return -self.dt_dcda / self.dt_dcla if self.dt_dcla else float("inf")

    def summary(self, veh: Vehicle) -> str:
        return "\n".join([
            f"baseline lap time    : {self.base:.3f} s  (ClA {veh.aero.ClA:.2f}, CdA {veh.aero.CdA:.2f}, "
            f"L/D {veh.aero.ClA / veh.aero.CdA:.2f})",
            f"+0.1 ClA (downforce) : {self.dt_dcla * 0.1 * 1e3:+8.1f} ms",
            f"+0.1 CdA (drag)      : {self.dt_dcda * 0.1 * 1e3:+8.1f} ms",
            f"+1 % ClA             : {self.dt_pct_cl * 1e3:+8.1f} ms",
            f"+1 % CdA             : {self.dt_pct_cd * 1e3:+8.1f} ms",
            f"exchange rate        : +1 % drag is paid back by +{self.exchange_rate:.2f} % downforce",
            f"lap-time neutral     : 1.0 CdA == {self.efficiency_equivalent:.2f} ClA "
            "(a device is worth it if its dCl/dCd exceeds this)",
        ])


def coefficients(veh: Vehicle, track: Track, step: float = 0.02, **kw) -> Coefficients:
    """Central-difference lap-time sensitivities to ClA and CdA."""
    base = lap_time(veh, track, **kw)
    t_cl = [lap_time(veh.scaled_aero(1 + s, 1), track, **kw) for s in (-step, step)]
    t_cd = [lap_time(veh.scaled_aero(1, 1 + s), track, **kw) for s in (-step, step)]
    d_pct_cl = (t_cl[1] - t_cl[0]) / (2 * step * 100)
    d_pct_cd = (t_cd[1] - t_cd[0]) / (2 * step * 100)
    return Coefficients(
        base=base,
        dt_dcla=d_pct_cl * 100 / veh.aero.ClA if veh.aero.ClA else 0.0,
        dt_dcda=d_pct_cd * 100 / veh.aero.CdA if veh.aero.CdA else 0.0,
        dt_pct_cl=d_pct_cl,
        dt_pct_cd=d_pct_cd,
    )
