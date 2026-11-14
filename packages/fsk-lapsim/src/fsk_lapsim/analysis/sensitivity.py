"""Central-difference sensitivities of event times (and endurance energy) to vehicle inputs.

d t / d p ≈ (t(p + h) − t(p − h)) / (2h)

Results are reported per SI unit and per a "display unit" (e.g. per 0.1 m² of ClA, per kg,
per cm of CG height, per 1 % of balance). The equivalence table expresses every parameter as
the change that costs the same time as +1 kg on a chosen event. The measurement-priority table
multiplies each sensitivity by the parameter's assumed uncertainty (σ) to rank which
assumption most needs a real measurement.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..events import EVENT_NAMES, EventResults, EventSetup, run_events
from ..parallel import pmap
from ..solver import SolverOptions
from ..vehicle import Vehicle, get_param, with_param


@dataclass(frozen=True)
class ParamSpec:
    """A perturbed parameter.

    Attributes:
        path: Dotted vehicle parameter path.
        step: Central-difference half step h (SI units).
        label: Display name.
        unit: SI unit.
        display_unit: Display scale, e.g. 0.1 for "per 0.1 m²".
        display_label: Text for the display unit, e.g. "0.1 m²".
    """

    path: str
    step: float
    label: str
    unit: str
    display_unit: float
    display_label: str


#: Parameters required by the spec (section 4).
CORE_PARAMS: tuple[ParamSpec, ...] = (
    ParamSpec("aero.cla_m2", 0.1, "ClA", "m²", 0.1, "0.1 m²"),
    ParamSpec("aero.cda_m2", 0.05, "CdA", "m²", 0.1, "0.1 m²"),
    ParamSpec("aero.balance_front", 0.01, "Aero balance (front)", "-", 0.01, "1 %p"),
    ParamSpec("mass_kg", 5.0, "Mass", "kg", 1.0, "1 kg"),
    ParamSpec("cg_height_m", 0.01, "CG height", "m", 0.01, "1 cm"),
    ParamSpec("tire.mu0", 0.02, "Tyre μ0", "-", 0.01, "0.01"),
    ParamSpec("powertrain.max_power_w", 5000.0, "Power cap", "W", 1000.0, "1 kW"),
)

#: Further assumed inputs used for the measurement-priority ranking.
EXTRA_PARAMS: tuple[ParamSpec, ...] = (
    ParamSpec("tire.load_sensitivity", 0.02, "Tyre load sensitivity k", "-", 0.01, "0.01"),
    ParamSpec("lltd_front", 0.02, "LLTD (front)", "-", 0.01, "1 %p"),
    ParamSpec("static_front_fraction", 0.01, "Static front weight", "-", 0.01, "1 %p"),
    ParamSpec(
        "tire.rolling_resistance_coeff", 0.002, "Rolling resistance Crr", "-", 0.001, "0.001"
    ),
    ParamSpec("powertrain.efficiency", 0.02, "Powertrain efficiency", "-", 0.01, "1 %p"),
    ParamSpec("powertrain.motor_max_torque_nm", 10.0, "Motor torque", "N·m", 10.0, "10 N·m"),
    ParamSpec("aero.air_density_kg_m3", 0.03, "Air density", "kg/m³", 0.01, "0.01 kg/m³"),
)

Runner = Callable[[Vehicle], EventResults]


@dataclass
class SensitivityResult:
    """Sensitivity output.

    Attributes:
        table: Long table: param, label, event, base value, dt/dp (SI), dt per display unit,
            plus endurance energy rows (event ``endurance_energy`` in kWh).
        base: Base event results.
    """

    table: pd.DataFrame
    base: EventResults

    def pivot(self, per_display_unit: bool = True) -> pd.DataFrame:
        """Matrix param × event of the sensitivities."""
        col = "per_display" if per_display_unit else "d_dp"
        return self.table.pivot(index="label", columns="event", values=col)


def _metric_values(res: EventResults, events: Sequence[str]) -> dict[str, float]:
    t = res.times()
    out = {ev: t[ev] for ev in events if ev in t}
    if res.endurance is not None:
        out["endurance_energy"] = res.endurance.energy_kwh
    return out


def _metrics_ctx(veh: Vehicle, ctx: dict) -> dict[str, float]:
    return _metric_values(
        run_events(veh, ctx["setup"], ctx["options"], ctx["events"]), ctx["events"]
    )


def sensitivity(
    vehicle: Vehicle,
    setup: EventSetup | None = None,
    options: SolverOptions | None = None,
    params: Sequence[ParamSpec] = CORE_PARAMS,
    events: tuple[str, ...] = EVENT_NAMES,
    runner: Runner | None = None,
    workers: int = 1,
) -> SensitivityResult:
    """Central-difference sensitivities for every parameter and event.

    ``workers > 1`` evaluates the perturbed vehicles in parallel processes (ignored when a
    custom ``runner`` is given).
    """
    setup = setup or EventSetup.load()
    run: Runner = runner or (lambda v: run_events(v, setup, options, events))
    base = run(vehicle)
    bounds: list[tuple[float, float, float]] = []
    variants: list[Vehicle] = []
    for spec in params:
        p0 = float(get_param(vehicle, spec.path))
        h = spec.step
        lo, hi = p0 - h, p0 + h
        if spec.path in (
            "aero.balance_front",
            "lltd_front",
            "static_front_fraction",
            "powertrain.efficiency",
        ):
            lo, hi = max(lo, 0.0), min(hi, 1.0)
        bounds.append((p0, lo, hi))
        variants += [with_param(vehicle, spec.path, hi), with_param(vehicle, spec.path, lo)]
    if runner is None and workers > 1:
        ctx = {"setup": setup, "options": options, "events": events}
        mets = pmap(_metrics_ctx, variants, ctx, workers)
    else:
        mets = [_metric_values(run(v), events) for v in variants]
    rows = []
    for k, spec in enumerate(params):
        p0, lo, hi = bounds[k]
        m_hi, m_lo = mets[2 * k], mets[2 * k + 1]
        for key in m_hi:
            d = (m_hi[key] - m_lo[key]) / (hi - lo)
            rows.append(
                {
                    "param": spec.path,
                    "label": spec.label,
                    "unit": spec.unit,
                    "base_value": p0,
                    "event": key,
                    "d_dp": d,
                    "per_display": d * spec.display_unit,
                    "display_unit": spec.display_label,
                    "plus_minus_asymmetry": (m_hi[key] - _metric_values(base, events)[key])
                    - (_metric_values(base, events)[key] - m_lo[key]),
                }
            )
    return SensitivityResult(pd.DataFrame(rows), base)


def equivalence_table(
    sens: SensitivityResult, event: str = "autocross", reference: str = "mass_kg"
) -> pd.DataFrame:
    """Parameter changes that cost the same ``event`` time as +1 unit of ``reference``.

    Example (autocross, mass): "+1 kg ≡ ΔClA −0.03 m² ≡ ΔCdA +0.08 m²".
    """
    t = sens.table[sens.table["event"] == event].set_index("param")
    if reference not in t.index:
        raise KeyError(f"{reference!r} not in the sensitivity table")
    d_ref = float(t.loc[reference, "d_dp"])
    rows = []
    for p, row in t.iterrows():
        d = float(row["d_dp"])
        eq = d_ref / d if abs(d) > 1e-12 else np.nan
        rows.append(
            {
                "param": p,
                "label": row["label"],
                "unit": row["unit"],
                "dt_dp_s_per_unit": d,
                "equivalent_change": eq,
            }
        )
    return pd.DataFrame(rows)


def measurement_priority(
    sens: SensitivityResult, sigma: dict[str, float], weights: dict[str, float] | None = None
) -> pd.DataFrame:
    """Rank parameters by |dt/dp|·σ (σ = assumed 1-sigma uncertainty of the input).

    ``weights`` combine events into one score (default: autocross 1, endurance 1/20 so that a
    lap and a full endurance are comparable, acceleration 1, skidpad 1).
    """
    w = weights or {"acceleration": 1.0, "skidpad": 1.0, "autocross": 1.0, "endurance": 0.05}
    t = sens.table
    rows = []
    for p, grp in t.groupby("param"):
        if p not in sigma:
            continue
        g = grp.set_index("event")
        effect = {ev: abs(float(g.loc[ev, "d_dp"])) * sigma[p] for ev in w if ev in g.index}
        score = sum(w[ev] * v for ev, v in effect.items())
        energy = (
            abs(float(g.loc["endurance_energy", "d_dp"])) * sigma[p]
            if ("endurance_energy" in g.index)
            else np.nan
        )
        rows.append(
            {
                "param": p,
                "label": grp["label"].iloc[0],
                "sigma": sigma[p],
                **{f"dt_{ev}_s": v for ev, v in effect.items()},
                "d_energy_kwh": energy,
                "score": score,
            }
        )
    df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    df.index = df.index + 1
    return df
