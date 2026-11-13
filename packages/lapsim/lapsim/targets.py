"""Per-device aero loads and lap-time-optimal performance targets.

* :func:`device_loads` turns a lap into per-device forces: downforce/drag at reference speeds,
  time-averaged over the lap, peak values and the energy each device's drag costs per lap.
* :func:`optimise_targets` searches the ClA of every tunable device (front / mid / rear ...)
  for the fastest lap.  Each device moves along its own polar ``CdA = CdA0 + k * ClA^2`` within
  ``[ClA_min, ClA_max]`` (see :class:`lapsim.vehicle.AeroDevice`), so extra downforce always
  costs drag, and the resulting balance shift is captured by the axle model.  An optional
  front-balance window keeps the answer drivable (QSS has no stability margin of its own), and
  ``mass_per_cla`` charges each device's extra structure to the car mass.  At the target the
  marginal value of every device (ms per +0.1 ClA / CdA, break-even L/D) is reported: these
  are the numbers a device designer works to.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from .solver import LapResult, simulate
from .study import device_coefficients
from .track import Track
from .vehicle import Vehicle

KMH = 1 / 3.6


# ---------------------------------------------------------------------------
@dataclass
class DeviceLoad:
    name: str
    ClA: float
    CdA: float
    front_share: float
    at_speed: dict          # {km/h: (downforce N, drag N)}
    mean_downforce: float   # time-averaged over the lap [N]
    mean_drag: float        # [N]
    peak_downforce: float   # [N]
    drag_energy: float      # drag work per lap [J]


def device_loads(veh: Vehicle, res: LapResult, speeds_kmh=(40, 60, 80, 100)) -> list:
    rho = veh.aero.rho
    q = 0.5 * rho * res.v**2
    q_seg = 0.5 * (q[1:] + q[:-1])
    dt, ds = np.diff(res.t), np.diff(res.s)
    out = []
    for name, d in veh.aero.devices.items():
        at = {s: (0.5 * rho * (s * KMH) ** 2 * d.ClA, 0.5 * rho * (s * KMH) ** 2 * d.CdA)
              for s in speeds_kmh}
        out.append(DeviceLoad(
            name=name, ClA=d.ClA, CdA=d.CdA, front_share=d.share(veh.wheelbase), at_speed=at,
            mean_downforce=float(np.sum(q_seg * dt) / res.lap_time * d.ClA),
            mean_drag=float(np.sum(q_seg * dt) / res.lap_time * d.CdA),
            peak_downforce=float(q.max() * d.ClA),
            drag_energy=float(np.sum(q_seg * ds) * d.CdA),
        ))
    return out


def loads_table(loads: list, veh: Vehicle) -> str:
    speeds = list(loads[0].at_speed)
    head = "".join(f"{f'@{s:g} km/h':>14}" for s in speeds)
    rows = [f"  {'device':<8}{'ClA':>6}{'CdA':>6}{'L/D':>6}{'F %':>7}{head}"
            f"{'lap mean':>14}{'peak DF':>9}{'drag E':>9}",
            f"  {'':<8}{'':>6}{'':>6}{'':>6}{'':>7}" + "".join(f"{'DF / drag N':>14}" for _ in speeds)
            + f"{'DF / drag N':>14}{'N':>9}{'Wh/lap':>9}"]
    tot = {k: 0.0 for k in ("cla", "cda", "mdf", "mdr", "pk", "e")}
    tot_at = {s: [0.0, 0.0] for s in speeds}
    for l in loads:
        ld = f"{l.ClA / l.CdA:6.2f}" if l.CdA else "     -"
        cells = "".join(f"{l.at_speed[s][0]:8.0f}/{l.at_speed[s][1]:<5.0f}" for s in speeds)
        rows.append(f"  {l.name:<8}{l.ClA:6.2f}{l.CdA:6.2f}{ld}{l.front_share * 100:7.1f}{cells}"
                    f"{l.mean_downforce:8.0f}/{l.mean_drag:<5.0f}{l.peak_downforce:9.0f}"
                    f"{l.drag_energy / 3600:9.1f}")
        for k, v in zip(tot, (l.ClA, l.CdA, l.mean_downforce, l.mean_drag, l.peak_downforce, l.drag_energy)):
            tot[k] += v
        for s in speeds:
            tot_at[s][0] += l.at_speed[s][0]
            tot_at[s][1] += l.at_speed[s][1]
    cells = "".join(f"{tot_at[s][0]:8.0f}/{tot_at[s][1]:<5.0f}" for s in speeds)
    rows.append(f"  {'total':<8}{tot['cla']:6.2f}{tot['cda']:6.2f}"
                f"{(tot['cla'] / tot['cda'] if tot['cda'] else 0):6.2f}"
                f"{veh.aero.balance_front * 100:7.1f}{cells}"
                f"{tot['mdf']:8.0f}/{tot['mdr']:<5.0f}{tot['pk']:9.0f}{tot['e'] / 3600:9.1f}")
    return "\n".join(rows)


# ---------------------------------------------------------------------------
@dataclass
class TargetResult:
    baseline: Vehicle
    target: Vehicle
    devices: list            # optimised device names
    base_time: float
    target_time: float
    evaluations: int
    balance_range: tuple | None
    marginal: dict = field(default_factory=dict)  # {device: DeviceCoefficient} at the target
    history: list = field(default_factory=list)   # (lap time, {device: ClA}) of accepted steps

    @property
    def gain(self) -> float:
        return self.target_time - self.base_time

    def summary(self) -> str:
        L = self.baseline.wheelbase
        rows = [f"  {'device':<8}{'ClA now':>9}{'target':>8}{'Δ':>7}{'CdA now':>9}{'target':>8}"
                f"{'L/D':>6}{'F %':>7}{'+0.1ClA':>9}{'+0.1CdA':>9}{'min L/D':>9}  note"]
        for name, d in self.target.aero.devices.items():
            b = self.baseline.aero.devices[name]
            ld = f"{d.ClA / d.CdA:6.2f}" if d.CdA else "     -"
            if name not in self.devices:
                note = "fixed"
            elif d.ClA_max is not None and d.ClA >= d.ClA_max - 0.02:
                note = "at ClA_max (package limit)"
            elif d.ClA_min is not None and d.ClA <= d.ClA_min + 0.02:
                note = "at ClA_min"
            elif self.balance_range and min(abs(self.target.aero.balance_front - b)
                                            for b in self.balance_range) < 0.003:
                note = "set by balance window"
            else:
                note = "optimum"
            m = self.marginal.get(name)
            mcols = (f"{m.dt_dcla * 100:+7.0f}ms{m.dt_dcda * 100:+7.0f}ms"
                     + (f"{m.break_even_ld:9.2f}" if np.isfinite(m.break_even_ld) else f"{'-':>9}")
                     if m else f"{'':>27}")
            rows.append(f"  {name:<8}{b.ClA:9.2f}{d.ClA:8.2f}{d.ClA - b.ClA:+7.2f}{b.CdA:9.2f}"
                        f"{d.CdA:8.2f}{ld}{d.share(L) * 100:7.1f}{mcols}  {note}")
        a0, a1 = self.baseline.aero, self.target.aero
        rows.append(f"  {'total':<8}{a0.ClA:9.2f}{a1.ClA:8.2f}{a1.ClA - a0.ClA:+7.2f}{a0.CdA:9.2f}"
                    f"{a1.CdA:8.2f}{a1.ClA / a1.CdA:6.2f}{a1.balance_front * 100:7.1f}")
        win = ("none" if self.balance_range is None
               else f"{self.balance_range[0] * 100:.1f}-{self.balance_range[1] * 100:.1f}% F")
        rows += ["",
                 f"  balance window : {win}  (baseline {a0.balance_front * 100:.1f}% -> "
                 f"target {a1.balance_front * 100:.1f}% F)",
                 f"  car mass       : {self.baseline.mass:.1f} kg -> {self.target.mass:.1f} kg",
                 f"  lap time       : {self.base_time:.3f} s -> {self.target_time:.3f} s "
                 f"({self.gain * 1e3:+.1f} ms, {self.evaluations} laps simulated)"]
        return "\n".join(rows)


def _freeze_polars(veh: Vehicle, names):
    """Pin CdA0/k_induced so the polar no longer moves with the operating point."""
    v = copy.deepcopy(veh)
    for n in names:
        d = v.aero.devices[n]
        d.CdA0, d.k_induced = d.polar()
        lo, hi = d.cla_bounds()
        d.ClA_min, d.ClA_max = lo, hi
    return v


def _apply(veh: Vehicle, names, x) -> Vehicle:
    """Move devices along their polars; device mass scales with ClA (``mass_per_cla``)."""
    v = copy.deepcopy(veh)
    for n, cla in zip(names, x):
        d = v.aero.devices[n]
        v.mass += (float(cla) - d.ClA) * d.mass_per_cla
        d.ClA = float(cla)
        d.CdA = d.cda_at(d.ClA)
    return v


def optimise_targets(veh: Vehicle, track: Track, devices=None, balance_range=None,
                     step: float = 0.2, tol: float = 0.01, max_evals: int = 600,
                     **kw) -> TargetResult:
    """Lap-time-optimal ClA (and polar CdA) for each tunable device.

    Compass (pattern) search: robust to the small kinks a QSS lap time has, needs no
    gradients and no scipy.  ``balance_range=(lo, hi)`` constrains the front balance.
    """
    names = list(devices) if devices else [n for n, d in veh.aero.devices.items() if d.tunable]
    unknown = set(names) - set(veh.aero.devices)
    if unknown:
        raise KeyError(f"unknown aero devices {sorted(unknown)}; have {list(veh.aero.devices)}")
    if not names:
        raise ValueError("no tunable aero devices")
    base = _freeze_polars(veh, names)
    bounds = np.array([base.aero.devices[n].cla_bounds() for n in names])
    cache = {}

    def lap(x):
        key = tuple(np.round(x, 6))
        if key not in cache:
            v = _apply(base, names, x)
            t = simulate(v, track, **kw).lap_time
            if balance_range is not None:
                b = v.aero.balance_front
                t += 100.0 * (max(balance_range[0] - b, 0.0) + max(b - balance_range[1], 0.0))
            cache[key] = t
        return cache[key]

    x = np.clip([base.aero.devices[n].ClA for n in names], bounds[:, 0], bounds[:, 1])
    base_time = simulate(veh, track, **kw).lap_time
    best = lap(x)
    history = [(best, dict(zip(names, x.tolist())))]
    while step >= tol and len(cache) < max_evals:
        improved = False
        for i in range(len(names)):
            for sign in (1.0, -1.0):
                trial = x.copy()
                trial[i] = np.clip(trial[i] + sign * step, *bounds[i])
                if trial[i] == x[i]:
                    continue
                t = lap(trial)
                if t < best - 1e-6:
                    x, best, improved = trial, t, True
                    history.append((best, dict(zip(names, x.tolist()))))
                    break
        if not improved:
            step /= 2
    target = _apply(base, names, x)
    marginal = {c.name: c for c in device_coefficients(target, track, **kw) if c.name in names}
    return TargetResult(veh, target, names, base_time, simulate(target, track, **kw).lap_time,
                        len(cache), balance_range, marginal, history)
