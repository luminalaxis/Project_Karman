"""Two-state active aero (DRS): high-downforce ↔ low-drag rear wing.

Open (low-drag) condition, evaluated on the current speed profile: |ay| < threshold, the car
is at full acceleration (forward pass limiting) and v ≥ the minimum opening speed. By default
that speed is the crossover where the open state accelerates harder than the closed state on
a straight: below it the car is traction limited and losing rear downforce costs more than the
drag saving (DECISIONS D-15). Each contiguous open zone gets a time-based
weight w(t) = min((t − t_open)/T, (t_brake − t)/T, 1) (linear coefficient blend over the
transition time T), so the flap is fully closed exactly where the zone ends — at the braking
point or where |ay| rises above the threshold. The close command therefore has to be issued T
seconds (≈ v·T metres) *before* that point. Zones whose fully-open time would be shorter than
``min_open_time_s`` are not opened. The lap is re-simulated with the blended aero and the zones
are re-detected; this repeats ``iterations`` times (braking points move as straights get
faster).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .aero import Aero, AeroComponent, components_from_config
from .config import DRSConfig, load_aero_components_config, load_drs_config, resolve_path
from .events import EventSetup, run_skidpad
from .solver import LIMIT_ACCEL, LapResult, SolverOptions, simulate
from .track import Track, straight_track
from .vehicle import Vehicle

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class DRSSettings:
    """Resolved DRS parameters (see ``drs.yaml``)."""

    rear: AeroComponent
    cl_reduction: float = 0.65
    cd_reduction: float = 0.55
    transition_time_s: float = 0.25
    ay_threshold_mps2: float = 4.9
    min_open_time_s: float = 0.3
    require_full_throttle: bool = True
    iterations: int = 4
    min_speed_mps: float | None = None

    @classmethod
    def from_config(cls, cfg: DRSConfig, base_dir: Path | None = None) -> DRSSettings:
        """Build from a validated config (reads the rear-wing component)."""
        comps = components_from_config(
            load_aero_components_config(resolve_path(cfg.components_file, base_dir))
        )
        rear = next((c for c in comps if c.name == cfg.rear_component), None)
        if rear is None:
            raise KeyError(f"no aero component named {cfg.rear_component!r}")
        vmin = None if cfg.min_speed_mps is None else cfg.min_speed_mps.value
        return cls(
            rear,
            cfg.cl_reduction_fraction.value,
            cfg.cd_reduction_fraction.value,
            cfg.transition_time_s.value,
            cfg.ay_threshold_mps2.value,
            cfg.min_open_time_s.value,
            cfg.require_full_throttle,
            cfg.iterations,
            vmin,
        )

    @classmethod
    def load(cls, path: str | Path | None = None) -> DRSSettings:
        """Load ``drs.yaml`` (default: bundled)."""
        base = resolve_path(path).parent if path is not None else None
        return cls.from_config(load_drs_config(path), base)


def open_state(vehicle: Vehicle, s: DRSSettings) -> Vehicle:
    """Low-drag vehicle: the rear wing loses ``cl_reduction``·ClA and ``cd_reduction``·CdA.

    Applied as a delta on top of the vehicle's aero so it works for any (lumped) aero set.
    """
    L = vehicle.wheelbase_m
    rw = s.rear
    d_cla = -s.cl_reduction * rw.cla_m2
    d_cda = -s.cd_reduction * rw.cda_m2
    d_front = d_cla * (1.0 - rw.x_cp_m / L) - d_cda * rw.z_cp_m / L
    a = vehicle.aero
    cla = a.cla_m2 + d_cla
    front = a.balance_front * a.cla_m2 + d_front
    if cla <= 0:
        raise ValueError("DRS reduction removes all downforce")
    aero = Aero(cla, a.cda_m2 + d_cda, front / cla, a.air_density_kg_m3)
    return dataclasses.replace(vehicle, aero=aero, name=vehicle.name + "+DRSopen")


def crossover_speed(
    closed_vehicle: Vehicle, open_vehicle: Vehicle, model: str = "axle", dv: float = 0.25
) -> float:
    """Lowest speed [m/s] from which the open state out-accelerates the closed one (ay = 0).

    Returns ``inf`` if opening never helps below the top speed.
    """
    from .models import make_model, top_speed

    mc, mo = make_model(closed_vehicle, model), make_model(open_vehicle, model)
    v = np.arange(0.0, top_speed(mc) + dv, dv)
    gain = mo.ax_accel_max(v, 0.0) - mc.ax_accel_max(v, 0.0)
    better = gain > 1e-9
    if not better.any():
        return float("inf")
    # first speed after which opening stays beneficial
    idx = len(v) - int(np.argmax(~better[::-1])) if not better[-1] else None
    if idx is None:
        last_bad = np.where(~better)[0]
        idx = int(last_bad[-1]) + 1 if last_bad.size else 0
    return float(v[min(idx, len(v) - 1)])


@dataclass
class DRSLap:
    """DRS lap result.

    Attributes:
        closed: Lap with the wing always closed (high downforce).
        drs: Lap with DRS.
        weight: Blend weight per node (0 closed … 1 open).
        zones: One row per open zone (positions, speeds, close-command lead).
        min_speed_mps: Opening speed threshold used [m/s].
    """

    closed: LapResult
    drs: LapResult
    weight: FloatArray
    zones: pd.DataFrame = field(default_factory=pd.DataFrame)
    min_speed_mps: float = 0.0

    @property
    def time_gain_s(self) -> float:
        """Lap time saved by DRS [s] (positive = faster)."""
        return self.closed.lap_time_s - self.drs.lap_time_s

    def summary(self) -> dict[str, float]:
        """Key DRS figures for this lap."""
        w = self.weight
        dt = np.diff(self.drs.t)
        wseg = 0.5 * (w[1:] + w[:-1])
        ds = np.diff(self.drs.s)
        z = self.zones
        return {
            "time_gain_s": self.time_gain_s,
            "zones_per_lap": float(len(z)),
            "switches_per_lap": float(2 * len(z)),
            "open_fraction_distance": float(np.sum(wseg * ds) / np.sum(ds)),
            "open_fraction_time": float(np.sum(wseg * dt) / np.sum(dt)),
            "max_close_lead_m": float(z["close_lead_m"].max()) if len(z) else 0.0,
            "max_speed_at_close_kph": float(z["v_brake_kph"].max()) if len(z) else 0.0,
            "energy_saving_kwh": (self.closed.energy.battery_net_j - self.drs.energy.battery_net_j)
            / 3.6e6,
            "min_open_speed_kph": 3.6 * self.min_speed_mps,
        }


def _zones(open_mask: NDArray[np.bool_], closed_track: bool) -> list[tuple[int, int]]:
    """Contiguous runs [start, end) of True; on loops a run may wrap (end < start)."""
    n = len(open_mask)
    if not open_mask.any():
        return []
    if open_mask.all():
        return [(0, n)]
    runs = []
    i = 0
    while i < n:
        if open_mask[i]:
            j = i
            while j < n and open_mask[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    if closed_track and len(runs) > 1 and runs[0][0] == 0 and runs[-1][1] == n:
        first = runs.pop(0)
        last = runs.pop()
        runs.append((last[0], first[1]))  # wraps over s = 0
    return runs


def drs_weight(res: LapResult, s: DRSSettings) -> tuple[FloatArray, pd.DataFrame]:
    """Blend weights from the open condition on ``res`` plus the zone table."""
    mask = np.abs(res.ay) < s.ay_threshold_mps2
    if s.require_full_throttle:
        mask &= res.limit == LIMIT_ACCEL
    if s.min_speed_mps is not None:
        mask &= res.v >= s.min_speed_mps
    n = len(res.s)
    t = res.t
    lap_t = float(t[-1])
    lap_s = float(res.s[-1])
    closed = res.closed
    if closed:
        mask[-1] = mask[0]  # node N is node 0
    w = np.zeros(n)
    rows = []
    T = s.transition_time_s
    for a, b in _zones(mask[:-1] if closed else mask, closed):
        # zone end: first node where the condition fails (flap must be closed there)
        if b > a:
            idx = np.arange(a, min(b, n - 1) + 1)
            tt = t[idx] - t[a]
        else:  # wraps
            idx = np.concatenate([np.arange(a, n - 1), np.arange(0, b + 1)])
            tt = np.where(idx >= a, t[idx] - t[a], t[idx] + lap_t - t[a])
        dur = float(tt[-1])
        full_open = dur - 2.0 * T
        if full_open < s.min_open_time_s:
            continue
        wz = np.clip(np.minimum(tt / T, (dur - tt) / T), 0.0, 1.0)
        w[idx] = np.maximum(w[idx], wz)
        end = idx[-1]
        k_cmd = idx[np.searchsorted(tt, dur - T, side="left").clip(0, len(idx) - 1)]
        s_brake = float(res.s[end])
        s_cmd = float(res.s[k_cmd])
        lead = s_brake - s_cmd if s_brake >= s_cmd else s_brake + lap_s - s_cmd
        rows.append(
            {
                "s_open_m": float(res.s[idx[0]]),
                "s_close_cmd_m": s_cmd,
                "s_brake_m": s_brake,
                "v_brake_kph": 3.6 * float(res.v[end]),
                "open_duration_s": dur,
                "close_lead_s": T,
                "close_lead_m": lead,
                "worst_case_lead_m": float(res.v[idx].max()) * T,
            }
        )
    if closed:
        w[-1] = w[0]
    return w, pd.DataFrame(rows)


def simulate_drs(
    closed_vehicle: Vehicle,
    track: Track,
    settings: DRSSettings,
    options: SolverOptions | None = None,
    start: str = "auto",
    v_end: float | None = None,
) -> DRSLap:
    """Simulate a lap with DRS (see module docstring)."""
    opt = options or SolverOptions()
    open_vehicle = open_state(closed_vehicle, settings)
    if settings.min_speed_mps is None:
        settings = dataclasses.replace(
            settings, min_speed_mps=crossover_speed(closed_vehicle, open_vehicle, opt.model)
        )
    base = simulate(closed_vehicle, track, opt, start=start, v_end=v_end)  # type: ignore[arg-type]
    res, w, zones = base, np.zeros(len(base.s)), pd.DataFrame()
    for _ in range(max(1, settings.iterations)):
        w_new, zones = drs_weight(res, settings)
        if np.array_equal(w_new > 0, w > 0) and res is not base:
            w = w_new
            break
        w = w_new
        res = simulate(
            closed_vehicle,
            track,
            opt,
            start=start,  # type: ignore[arg-type]
            v_end=v_end,
            alt_vehicle=open_vehicle,
            alt_weight=w,
        )
    # zones/weights consistent with the final profile
    res = simulate(
        closed_vehicle,
        track,
        opt,
        start=start,  # type: ignore[arg-type]
        v_end=v_end,
        alt_vehicle=open_vehicle,
        alt_weight=w,
    )
    return DRSLap(base, res, w, zones, float(settings.min_speed_mps or 0.0))


@dataclass
class DRSEventGains:
    """DRS effect per event."""

    table: pd.DataFrame
    laps: dict[str, DRSLap]


def drs_event_gains(
    vehicle: Vehicle,
    settings: DRSSettings,
    setup: EventSetup | None = None,
    options: SolverOptions | None = None,
) -> DRSEventGains:
    """Time gain, switch count and open fraction for every event."""
    setup = setup or EventSetup.load()
    opt = options or SolverOptions(ds_m=setup.ds_m)
    laps: dict[str, DRSLap] = {}
    laps["acceleration"] = simulate_drs(
        vehicle, straight_track(setup.acceleration_length_m), settings, opt, start="standing"
    )
    laps["autocross"] = simulate_drs(
        vehicle,
        setup.autocross_track,
        settings,
        opt,
        start=setup.autocross_start if setup.autocross_track.closed else "standing",
    )
    endu = simulate_drs(vehicle, setup.endurance_track, settings, opt, start="flying")
    laps["endurance_lap"] = endu
    rows = []
    for name, lap in laps.items():
        rows.append({"event": name, **lap.summary()})
    n = max(1, round(setup.endurance_distance_m / setup.endurance_track.length_m))
    e = endu.summary()
    rows.append(
        {
            "event": "endurance_total",
            "time_gain_s": e["time_gain_s"] * n,
            "zones_per_lap": e["zones_per_lap"],
            "switches_per_lap": e["switches_per_lap"],
            "open_fraction_distance": e["open_fraction_distance"],
            "open_fraction_time": e["open_fraction_time"],
            "max_close_lead_m": e["max_close_lead_m"],
            "max_speed_at_close_kph": e["max_speed_at_close_kph"],
            "energy_saving_kwh": e["energy_saving_kwh"] * n,
        }
    )
    sk = run_skidpad(vehicle, setup.skidpad_radius_m, opt.model)
    rows.append(
        {
            "event": "skidpad",
            "time_gain_s": 0.0,
            "zones_per_lap": 0.0,
            "switches_per_lap": 0.0,
            "open_fraction_distance": 0.0,
            "open_fraction_time": 0.0,
            "max_close_lead_m": 0.0,
            "max_speed_at_close_kph": 0.0,
            "energy_saving_kwh": 0.0,
            "note": f"|ay| {sk.ay_mps2:.1f} m/s² > threshold → closed",
        }
    )
    return DRSEventGains(pd.DataFrame(rows), laps)


def lead_distance_table(
    speeds_kph: Sequence[float], transition_times_s: Sequence[float]
) -> pd.DataFrame:
    """Close-command lead distance v·T [m] for firmware/fail-safe design."""
    rows = []
    for v in speeds_kph:
        rows.append({"speed_kph": v, **{f"T={t:.2f}s": v / 3.6 * t for t in transition_times_s}})
    return pd.DataFrame(rows)
