"""The four dynamic events: acceleration, skidpad, autocross and endurance.

* Acceleration — standing start over ``acceleration_length_m`` (75 m) on a straight.
* Skidpad — analytical steady-state cornering on the driven radius (default 9.125 m):
  the speed where v²/R equals the constant-speed limit ay_max(v); time = 2πR/v, averaged
  over left and right.
* Autocross — one lap, standing (default) or flying start.
* Endurance — flying laps; the first lap is a standing start that must end at the flying-lap
  speed. Laps = round(total distance / lap length). Driver change time is excluded.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .config import EventsConfig, load_events_config, resolve_path
from .models import make_model, top_speed
from .solver import LapResult, SolverOptions, simulate
from .track import Track, straight_track
from .trackgen import load_track
from .vehicle import Vehicle, with_param

EVENT_NAMES = ("acceleration", "skidpad", "autocross", "endurance")
EventName = Literal["acceleration", "skidpad", "autocross", "endurance"]


@dataclass
class EventSetup:
    """Resolved event definitions (tracks loaded).

    Attributes:
        acceleration_length_m: Acceleration distance [m].
        skidpad_radius_m: Driven radius on the skidpad [m].
        autocross_track: Autocross track.
        autocross_start: ``"standing"`` or ``"flying"``.
        endurance_track: Endurance lap.
        endurance_distance_m: Total endurance distance [m].
        ds_m: Solver distance step [m].
        endurance_max_power_w: Optional accumulator power cap for the endurance (energy
            management), ``None`` = the vehicle's full power.
    """

    acceleration_length_m: float
    skidpad_radius_m: float
    autocross_track: Track
    autocross_start: Literal["standing", "flying"]
    endurance_track: Track
    endurance_distance_m: float
    ds_m: float = 0.25
    endurance_max_power_w: float | None = None

    @classmethod
    def from_config(cls, cfg: EventsConfig, base_dir: Path | None = None) -> EventSetup:
        """Build from ``events.yaml`` (track paths relative to the YAML file)."""
        return cls(
            acceleration_length_m=cfg.acceleration_length_m.value,
            skidpad_radius_m=cfg.skidpad_radius_m.value,
            autocross_track=load_track(resolve_path(cfg.autocross_track, base_dir)),
            autocross_start=cfg.autocross_start,
            endurance_track=load_track(resolve_path(cfg.endurance_track, base_dir)),
            endurance_distance_m=cfg.endurance_total_distance_m.value,
            ds_m=cfg.ds_m.value,
            endurance_max_power_w=(
                None if cfg.endurance_max_power_w is None else cfg.endurance_max_power_w.value
            ),
        )

    @classmethod
    def load(cls, path: str | Path | None = None) -> EventSetup:
        """Load ``events.yaml`` (default: bundled)."""
        cfg = load_events_config(path)
        base = resolve_path(path).parent if path is not None else None
        return cls.from_config(cfg, base)


@dataclass
class SkidpadResult:
    """Steady-state skidpad result."""

    radius_m: float
    speed_mps: float
    ay_mps2: float
    time_left_s: float
    time_right_s: float

    @property
    def time_s(self) -> float:
        """Average of left and right lap times [s]."""
        return 0.5 * (self.time_left_s + self.time_right_s)


@dataclass
class EnduranceResult:
    """Endurance totals."""

    n_laps: int
    first_lap: LapResult
    flying_lap: LapResult

    @property
    def total_time_s(self) -> float:
        """Total time [s] (first lap + (n−1) flying laps)."""
        return self.first_lap.lap_time_s + (self.n_laps - 1) * self.flying_lap.lap_time_s

    def _total(self, attr: str) -> float:
        a = getattr(self.first_lap.energy, attr)
        b = getattr(self.flying_lap.energy, attr)
        return a + (self.n_laps - 1) * b

    @property
    def energy_kwh(self) -> float:
        """Net accumulator energy [kWh] (with regen if enabled)."""
        return self._total("battery_net_j") / 3.6e6

    @property
    def energy_no_regen_kwh(self) -> float:
        """Accumulator energy spent on driving [kWh] (regen not credited)."""
        return self._total("battery_drive_j") / 3.6e6

    @property
    def regen_kwh(self) -> float:
        """Recovered energy [kWh]."""
        return self._total("regen_j") / 3.6e6


@dataclass
class EventResults:
    """Results of the selected events (missing events are ``None``)."""

    vehicle_name: str
    acceleration: LapResult | None = None
    skidpad: SkidpadResult | None = None
    autocross: LapResult | None = None
    endurance: EnduranceResult | None = None
    warnings: list[str] = field(default_factory=list)

    def times(self) -> dict[str, float]:
        """Event times [s]."""
        out: dict[str, float] = {}
        if self.acceleration is not None:
            out["acceleration"] = self.acceleration.lap_time_s
        if self.skidpad is not None:
            out["skidpad"] = self.skidpad.time_s
        if self.autocross is not None:
            out["autocross"] = self.autocross.lap_time_s
        if self.endurance is not None:
            out["endurance"] = self.endurance.total_time_s
        return out

    def metrics(self) -> dict[str, float]:
        """Event times plus key secondary figures."""
        out = {f"{k}_time_s": v for k, v in self.times().items()}
        if self.acceleration is not None:
            out["acceleration_exit_speed_kph"] = 3.6 * float(self.acceleration.v[-1])
        if self.skidpad is not None:
            out["skidpad_ay_g"] = self.skidpad.ay_mps2 / 9.81
        if self.autocross is not None:
            out["autocross_mean_speed_kph"] = 3.6 * self.autocross.mean_speed_mps
        if self.endurance is not None:
            e = self.endurance
            out["endurance_flying_lap_s"] = e.flying_lap.lap_time_s
            out["endurance_laps"] = e.n_laps
            out["endurance_energy_kwh"] = e.energy_kwh
            out["endurance_energy_no_regen_kwh"] = e.energy_no_regen_kwh
            out["endurance_regen_kwh"] = e.regen_kwh
            out["endurance_mean_power_kw"] = e.energy_kwh * 3.6e6 / e.total_time_s / 1e3
        return out


def run_acceleration(
    vehicle: Vehicle, length_m: float = 75.0, options: SolverOptions | None = None
) -> LapResult:
    """Standing-start acceleration over ``length_m``."""
    return simulate(vehicle, straight_track(length_m, "acceleration"), options, start="standing")


def run_skidpad(vehicle: Vehicle, radius_m: float = 9.125, model: str = "axle") -> SkidpadResult:
    """Analytical steady-state skidpad (bisection on v²/R = ay_max(v))."""
    mdl = make_model(vehicle, model)
    lo, hi = 0.0, top_speed(mdl)
    if float(mdl.ay_max(hi)) >= hi * hi / radius_m:
        lo = hi
    else:
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if float(mdl.ay_max(mid)) >= mid * mid / radius_m:
                lo = mid
            else:
                hi = mid
    v = lo
    t = 2.0 * math.pi * radius_m / v
    # the model is left/right symmetric, so both directions give the same time
    return SkidpadResult(radius_m, v, v * v / radius_m, t, t)


def run_autocross(
    vehicle: Vehicle, track: Track, start: str = "standing", options: SolverOptions | None = None
) -> LapResult:
    """Single autocross lap."""
    mode = "flying" if (start == "flying" and track.closed) else "standing"
    return simulate(vehicle, track, options, start=mode)  # type: ignore[arg-type]


def run_endurance(
    vehicle: Vehicle,
    track: Track,
    total_distance_m: float = 22_000.0,
    options: SolverOptions | None = None,
    max_power_w: float | None = None,
) -> EnduranceResult:
    """Endurance: standing first lap + flying laps (closed track required).

    ``max_power_w`` caps the accumulator power for the whole event (energy management).
    """
    if not track.closed:
        raise ValueError("endurance needs a closed track")
    if max_power_w is not None and max_power_w < vehicle.powertrain.max_power_w:
        vehicle = with_param(vehicle, "powertrain.max_power_w", float(max_power_w))
    flying = simulate(vehicle, track, options, start="flying")
    first = simulate(vehicle, track, options, start="standing", v_end=float(flying.v[0]))
    n = max(1, round(total_distance_m / track.length_m))
    return EnduranceResult(n, first, flying)


def plausibility_warnings(res: EventResults) -> list[str]:
    """Flag results far outside typical FS-EV values (information only, no tuning)."""
    w: list[str] = []
    checks = [
        ("acceleration", 3.2, 5.0, "s"),
        ("skidpad", 4.4, 6.0, "s"),
    ]
    times = res.times()
    for name, lo, hi, unit in checks:
        if name in times and not lo <= times[name] <= hi:
            w.append(
                f"{name} {times[name]:.2f} {unit} outside typical FS EV range {lo}–{hi} {unit}"
            )
    if res.autocross is not None:
        kph = 3.6 * res.autocross.mean_speed_mps
        if not 40.0 <= kph <= 65.0:
            w.append(f"autocross mean speed {kph:.1f} km/h outside typical 40–65 km/h")
    return w


def run_events(
    vehicle: Vehicle,
    setup: EventSetup | None = None,
    options: SolverOptions | None = None,
    events: tuple[str, ...] = EVENT_NAMES,
) -> EventResults:
    """Run the selected events for one vehicle."""
    setup = setup or EventSetup.load()
    opt = options or SolverOptions(ds_m=setup.ds_m)
    unknown = set(events) - set(EVENT_NAMES)
    if unknown:
        raise ValueError(f"unknown event(s): {sorted(unknown)}")
    res = EventResults(vehicle.name)
    if "acceleration" in events:
        res.acceleration = run_acceleration(vehicle, setup.acceleration_length_m, opt)
    if "skidpad" in events:
        res.skidpad = run_skidpad(vehicle, setup.skidpad_radius_m, opt.model)
    if "autocross" in events:
        res.autocross = run_autocross(vehicle, setup.autocross_track, setup.autocross_start, opt)
    if "endurance" in events:
        res.endurance = run_endurance(
            vehicle,
            setup.endurance_track,
            setup.endurance_distance_m,
            opt,
            setup.endurance_max_power_w,
        )
    res.warnings = plausibility_warnings(res)
    return res
