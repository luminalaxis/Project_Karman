"""Aero balance that minimises lap time for fixed mechanical setup and total ClA/CdA.

Coarse scan over the balance range, then a bounded Brent refinement around the best scan
point. The quasi-steady optimum is where both axles saturate together; a real car needs a
stability margin (more rear balance) that this model does not know about.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from ..events import EventSetup, run_autocross, run_endurance
from ..models import AxleModel
from ..solver import SolverOptions
from ..vehicle import Vehicle, with_param


@dataclass
class BalanceOptResult:
    """Balance optimisation output.

    Attributes:
        objective: Objective name.
        best_balance: Optimal front aero balance [-].
        best_value: Objective at the optimum [s].
        base_balance: The vehicle's balance [-].
        base_value: Objective at the vehicle's balance [s].
        scan: Scan table (balance, objective).
        limits_at_best: Cornering limit of each constraint at 15/25 m/s at the optimum [g].
    """

    objective: str
    best_balance: float
    best_value: float
    base_balance: float
    base_value: float
    scan: pd.DataFrame
    limits_at_best: pd.DataFrame


def _objective(name: str, setup: EventSetup, opt: SolverOptions) -> Callable[[Vehicle], float]:
    if name == "autocross":
        return lambda v: (
            run_autocross(v, setup.autocross_track, setup.autocross_start, opt).lap_time_s
        )
    if name == "endurance_lap":
        return lambda v: (
            run_endurance(
                v, setup.endurance_track, setup.endurance_track.length_m, opt
            ).flying_lap.lap_time_s
        )
    if name == "combined":

        def f(v: Vehicle) -> float:
            ac = run_autocross(v, setup.autocross_track, setup.autocross_start, opt).lap_time_s
            en = run_endurance(
                v, setup.endurance_track, setup.endurance_track.length_m, opt
            ).flying_lap.lap_time_s
            return ac + en

        return f
    raise ValueError("objective must be autocross, endurance_lap or combined")


def optimise_balance(
    vehicle: Vehicle,
    setup: EventSetup | None = None,
    options: SolverOptions | None = None,
    objective: str = "autocross",
    bounds: tuple[float, float] = (0.30, 0.65),
    n_scan: int = 15,
) -> BalanceOptResult:
    """Find the front aero balance that minimises ``objective`` (see module docstring)."""
    setup = setup or EventSetup.load()
    opt = options or SolverOptions(ds_m=setup.ds_m)
    f = _objective(objective, setup, opt)
    cache: dict[float, float] = {}

    def g(b: float) -> float:
        key = round(float(b), 10)
        if key not in cache:
            cache[key] = f(with_param(vehicle, "aero.balance_front", key))
        return cache[key]

    grid = np.linspace(bounds[0], bounds[1], n_scan)
    vals = [g(b) for b in grid]
    k = int(np.argmin(vals))
    lo = grid[max(k - 1, 0)]
    hi = grid[min(k + 1, n_scan - 1)]
    res = minimize_scalar(g, bounds=(lo, hi), method="bounded", options={"xatol": 1e-4})
    best_b = float(res.x) if res.fun <= vals[k] else float(grid[k])
    best_v = g(best_b)
    base_v = g(vehicle.aero.balance_front)
    scan = pd.DataFrame(sorted(cache.items()), columns=["balance_front", objective + "_s"])
    lim = AxleModel(with_param(vehicle, "aero.balance_front", best_b)).ay_max_by_limit(
        np.array([15.0, 25.0])
    )
    limits = pd.DataFrame({k2: v / 9.81 for k2, v in lim.items()}, index=["15 m/s", "25 m/s"])
    return BalanceOptResult(
        objective, best_b, best_v, vehicle.aero.balance_front, base_v, scan, limits
    )
