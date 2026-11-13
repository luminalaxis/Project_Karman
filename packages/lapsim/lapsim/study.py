"""Aero sensitivity studies built on top of :func:`lapsim.solver.simulate`."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .solver import simulate
from .track import Track
from .vehicle import Vehicle


def lap_time(veh: Vehicle, track: Track, **kw) -> float:
    return simulate(veh, track, **kw).lap_time


def modify(veh: Vehicle, changes: dict | None = None, balance: float | None = None) -> Vehicle:
    """Apply fractional aero changes ``{device or None: (d_cl, d_cd)}`` (None = every device),
    then optionally re-trim to a target front ``balance``."""
    for dev, (d_cl, d_cd) in (changes or {}).items():
        veh = veh.scaled_aero(1 + d_cl, 1 + d_cd, None if dev is None else [dev])
    if balance is not None:
        veh = veh.with_balance(balance)
    return veh


def delta(veh: Vehicle, track: Track, changes: dict | None = None,
          balance: float | None = None, **kw):
    """Lap time before/after :func:`modify` (fractions: 0.10 = +10 %)."""
    base = lap_time(veh, track, **kw)
    new = lap_time(modify(veh, changes, balance), track, **kw)
    return base, new, new - base


def sweep(veh: Vehicle, track: Track, d_cl: np.ndarray, d_cd: np.ndarray,
          devices=None, **kw) -> np.ndarray:
    """Lap time delta grid [s], rows = d_cd, columns = d_cl (fractions of the chosen devices)."""
    base = lap_time(veh, track, **kw)
    out = np.empty((len(d_cd), len(d_cl)))
    for i, cd in enumerate(d_cd):
        for j, cl in enumerate(d_cl):
            out[i, j] = lap_time(veh.scaled_aero(1 + cl, 1 + cd, devices), track, **kw) - base
    return out


def balance_sweep(veh: Vehicle, track: Track, balances: np.ndarray, **kw) -> np.ndarray:
    """Lap times re-trimmed to each front balance (see :meth:`Vehicle.with_balance`); NaN if unreachable."""
    out = []
    for b in balances:
        try:
            out.append(lap_time(veh.with_balance(float(b)), track, **kw))
        except ValueError:
            out.append(np.nan)
    return np.array(out)


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


@dataclass
class DeviceCoefficient:
    name: str
    dt_dcla: float     # [s per unit ClA on this device]
    dt_dcda: float     # [s per unit CdA on this device]

    @property
    def break_even_ld(self) -> float:
        """A change to this device gains lap time if its dClA/dCdA exceeds this value."""
        return -self.dt_dcda / self.dt_dcla if self.dt_dcla < 0 else float("inf")


def device_coefficients(veh: Vehicle, track: Track, step: float = 0.05, **kw) -> list:
    """Per-device lap-time sensitivity to absolute ClA / CdA changes (balance moves with it)."""
    out = []
    for name, dev in veh.aero.devices.items():
        def t_at(attr, delta):
            return lap_time(veh.with_changes(**{f"aero.{name}.{attr}": getattr(dev, attr) + delta}),
                            track, **kw)
        dcl = (t_at("ClA", step) - t_at("ClA", -step)) / (2 * step)
        dcd = (t_at("CdA", step) - t_at("CdA", -step)) / (2 * step)
        out.append(DeviceCoefficient(name, dcl, dcd))
    return out


def device_table(coeffs: list) -> str:
    rows = [f"  {'device':<10}{'+0.1 ClA':>11}{'+0.1 CdA':>11}{'break-even L/D':>16}"]
    for c in coeffs:
        be = f"{c.break_even_ld:16.2f}" if np.isfinite(c.break_even_ld) else f"{'never':>16}"
        rows.append(f"  {c.name:<10}{c.dt_dcla * 0.1 * 1e3:+9.1f}ms{c.dt_dcda * 0.1 * 1e3:+9.1f}ms{be}")
    return "\n".join(rows)
