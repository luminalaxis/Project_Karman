"""ClA × CdA grid sweep: autocross lap time and endurance energy maps."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..events import EventSetup, run_autocross, run_endurance
from ..solver import SolverOptions
from ..vehicle import Vehicle, with_params


@dataclass
class SweepResult:
    """Grid sweep output.

    Attributes:
        cla: ClA grid [m²] (n_cl).
        cda: CdA grid [m²] (n_cd).
        autocross_s: Autocross time [s], shape (n_cd, n_cl).
        endurance_s: Endurance total time [s], shape (n_cd, n_cl).
        endurance_kwh: Endurance net energy [kWh], shape (n_cd, n_cl).
        points: Marked designs (label, ClA, CdA, autocross s, endurance kWh).
    """

    cla: np.ndarray
    cda: np.ndarray
    autocross_s: np.ndarray
    endurance_s: np.ndarray
    endurance_kwh: np.ndarray
    points: pd.DataFrame = field(default_factory=pd.DataFrame)

    def to_frame(self) -> pd.DataFrame:
        """Long table of the grid."""
        cl, cd = np.meshgrid(self.cla, self.cda)
        return pd.DataFrame(
            {
                "ClA_m2": cl.ravel(),
                "CdA_m2": cd.ravel(),
                "autocross_s": self.autocross_s.ravel(),
                "endurance_s": self.endurance_s.ravel(),
                "endurance_kwh": self.endurance_kwh.ravel(),
            }
        )


def _evaluate(veh: Vehicle, setup: EventSetup, opt: SolverOptions) -> tuple[float, float, float]:
    ac = run_autocross(veh, setup.autocross_track, setup.autocross_start, opt).lap_time_s
    en = run_endurance(
        veh, setup.endurance_track, setup.endurance_distance_m, opt, setup.endurance_max_power_w
    )
    return ac, en.total_time_s, en.energy_kwh


def sweep_cla_cda(
    vehicle: Vehicle,
    cla_values: Sequence[float],
    cda_values: Sequence[float],
    setup: EventSetup | None = None,
    options: SolverOptions | None = None,
    points: Sequence[tuple[str, Vehicle]] = (),
) -> SweepResult:
    """Evaluate autocross time and endurance time/energy on a ClA × CdA grid.

    Balance, mass and CG stay at the vehicle's values. ``points`` (label, vehicle) are
    evaluated as they are (e.g. the current car and rear-wing candidates) for overlay.
    """
    setup = setup or EventSetup.load()
    opt = options or SolverOptions(ds_m=setup.ds_m)
    cla = np.asarray(cla_values, float)
    cda = np.asarray(cda_values, float)
    shape = (cda.size, cla.size)
    ac, et, ee = np.empty(shape), np.empty(shape), np.empty(shape)
    for i, cd in enumerate(cda):
        for j, cl in enumerate(cla):
            v = with_params(vehicle, {"aero.cla_m2": float(cl), "aero.cda_m2": float(cd)})
            ac[i, j], et[i, j], ee[i, j] = _evaluate(v, setup, opt)
    rows = []
    for label, veh in points:
        a, t, e = _evaluate(veh, setup, opt)
        rows.append(
            {
                "label": label,
                "ClA_m2": veh.aero.cla_m2,
                "CdA_m2": veh.aero.cda_m2,
                "balance_front": veh.aero.balance_front,
                "mass_kg": veh.mass_kg,
                "autocross_s": a,
                "endurance_s": t,
                "endurance_kwh": e,
            }
        )
    return SweepResult(cla, cda, ac, et, ee, pd.DataFrame(rows))
