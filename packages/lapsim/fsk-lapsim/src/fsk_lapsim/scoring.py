"""Points conversion layer (extension E4).

The formulas live in ``scoring.yaml`` and are **placeholders** shaped like the usual
Formula Student "time ratio" scoring. Reference (best) times can be absolute or a ratio of
the baseline car's simulated time, so that the layer also works on synthetic tracks. The main
use is converting *time differences* between design variants into *points differences*:
the reference times are fixed by the baseline and every variant is scored against them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import default_config_dir, load_yaml
from .events import EventResults


@dataclass(frozen=True)
class EventFormula:
    """One event's scoring formula (see ``scoring.yaml``)."""

    name: str
    formula: str
    a: float
    b: float = 0.0
    exponent: float = 1.0
    tmax_factor: float = 1.25
    t_min_s: float | None = None
    t_min_ratio: float | None = None
    t_ref_ratio: float = 0.95
    e_ref_ratio: float = 0.85
    ef_min: float = 0.5
    ef_max: float = 1.0

    @classmethod
    def from_dict(cls, name: str, d: dict[str, Any]) -> EventFormula:
        """Parse one YAML entry (values may be ``{value: …}`` mappings)."""

        def num(x: Any) -> float:
            return float(x["value"] if isinstance(x, dict) else x)

        kw = {k: (num(v) if k != "formula" else str(v)) for k, v in d.items()}
        if kw.get("formula") not in ("time_ratio", "efficiency_factor"):
            raise ValueError(f"{name}: unknown formula {kw.get('formula')!r}")
        return cls(name=name, **kw)  # type: ignore[arg-type]

    def t_min(self, base_time_s: float) -> float:
        """Reference best time [s] (absolute or ratio of the baseline time)."""
        if self.t_min_s is not None:
            return self.t_min_s
        if self.t_min_ratio is not None:
            return self.t_min_ratio * base_time_s
        raise ValueError(f"{self.name}: give t_min_s or t_min_ratio")

    def time_score(self, t: float, t_min: float) -> float:
        """Points for time ``t`` [s] given the best time ``t_min``."""
        t_max = self.tmax_factor * t_min
        if t >= t_max:
            return self.b
        t = max(t, t_min)
        p = self.exponent
        return self.a * ((t_max / t) ** p - 1.0) / ((t_max / t_min) ** p - 1.0) + self.b

    def efficiency_score(self, t: float, e: float, t_ref: float, e_ref: float) -> float:
        """Points from the efficiency factor (T_ref/T)·(E_ref/E)."""
        ef = (t_ref / t) * (e_ref / e)
        return self.a * float(np.clip((ef - self.ef_min) / (self.ef_max - self.ef_min), 0, 1))


@dataclass
class ScoringModel:
    """All event formulas plus the placeholder flag."""

    formulas: dict[str, EventFormula]
    placeholder: bool = True
    refs: dict[str, float] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path | None = None) -> ScoringModel:
        """Load ``scoring.yaml`` (default: bundled placeholder)."""
        d = (
            load_yaml(path)
            if path is not None
            else load_yaml(default_config_dir() / "scoring.yaml")
        )
        f = {k: EventFormula.from_dict(k, v) for k, v in d["events"].items()}
        return cls(f, bool(d.get("placeholder", False)))

    def calibrate_values(
        self, times: dict[str, float], endurance_energy_kwh: float | None = None
    ) -> ScoringModel:
        """Fix reference times/energy from baseline event times (returns a new model)."""
        refs: dict[str, float] = {}
        for name, f in self.formulas.items():
            if f.formula == "time_ratio" and name in times:
                refs[name] = f.t_min(times[name])
            elif (
                f.formula == "efficiency_factor"
                and "endurance" in times
                and endurance_energy_kwh is not None
            ):
                refs["efficiency_t_ref"] = f.t_ref_ratio * times["endurance"]
                refs["efficiency_e_ref"] = f.e_ref_ratio * endurance_energy_kwh
        return ScoringModel(self.formulas, self.placeholder, refs)

    def calibrate(self, base: EventResults) -> ScoringModel:
        """Fix the reference times/energy from the baseline results (returns a new model)."""
        return self.calibrate_values(base.times(), _energy(base))

    def score_values(
        self, times: dict[str, float], endurance_energy_kwh: float | None = None
    ) -> dict[str, float]:
        """Points per event from event times [s] (and endurance energy [kWh])."""
        if not self.refs:
            raise RuntimeError("calibrate the model with the baseline results first")
        out: dict[str, float] = {}
        for name, f in self.formulas.items():
            if f.formula == "time_ratio" and name in times and name in self.refs:
                out[name] = f.time_score(times[name], self.refs[name])
            elif (
                f.formula == "efficiency_factor"
                and "endurance" in times
                and endurance_energy_kwh is not None
                and "efficiency_t_ref" in self.refs
            ):
                out[name] = f.efficiency_score(
                    times["endurance"],
                    endurance_energy_kwh,
                    self.refs["efficiency_t_ref"],
                    self.refs["efficiency_e_ref"],
                )
        out["total"] = sum(out.values())
        return out

    def score(self, res: EventResults) -> dict[str, float]:
        """Points per event (requires calibration first)."""
        return self.score_values(res.times(), _energy(res))

    def points_per_second(self, times: dict[str, float], h: float = 1e-3) -> dict[str, float]:
        """d(points)/d(time) per time-scored event [points/s] (negative)."""
        out = {}
        for name, f in self.formulas.items():
            if f.formula == "time_ratio" and name in times and name in self.refs:
                t = times[name]
                out[name] = (
                    f.time_score(t + h, self.refs[name]) - f.time_score(t - h, self.refs[name])
                ) / (2 * h)
        return out

    def table(self, res: EventResults) -> pd.DataFrame:
        """Per event: time, reference, points, points per 0.1 s."""
        times = res.times()
        sc = self.score(res)
        pps = self.points_per_second(times)
        rows = []
        for name in self.formulas:
            if name not in sc:
                continue
            rows.append(
                {
                    "event": name,
                    "time_s": times.get(name, np.nan),
                    "t_min_s": self.refs.get(name, np.nan),
                    "points": sc[name],
                    "points_per_0.1s": pps.get(name, np.nan) * 0.1,
                }
            )
        rows.append({"event": "total", "points": sc["total"]})
        return pd.DataFrame(rows)


def _energy(res: EventResults) -> float | None:
    return res.endurance.energy_kwh if res.endurance is not None else None


def delta_points(
    model: ScoringModel, base: EventResults, variants: dict[str, EventResults]
) -> pd.DataFrame:
    """Points difference of each variant against ``base`` (same reference times)."""
    m = model if model.refs else model.calibrate(base)
    b = m.score(base)
    rows = []
    for label, r in variants.items():
        s = m.score(r)
        rows.append({"variant": label, **{f"d_{k}": s[k] - b[k] for k in s}})
    return pd.DataFrame(rows)


def comparison_points(model: ScoringModel, table: pd.DataFrame, reference: str) -> pd.DataFrame:
    """Points of every row of a candidate comparison table, relative to ``reference``.

    Reference times are calibrated on the ``reference`` row so that all candidates are scored
    against the same best times.
    """

    def vals(row: pd.Series) -> tuple[dict[str, float], float | None]:
        times = {
            c[:-7]: float(row[c])
            for c in row.index
            if c.endswith("_time_s") and not c.startswith(("d_", "rank_"))
        }
        e = float(row["endurance_energy_kwh"]) if "endurance_energy_kwh" in row else None
        return times, e

    ref_row = table.loc[table["config_id"] == reference].iloc[0]
    m = model.calibrate_values(*vals(ref_row))
    base = m.score_values(*vals(ref_row))
    rows = []
    for _, r in table.iterrows():
        sc = m.score_values(*vals(r))
        rows.append({"config_id": r["config_id"], **{f"d_{k}_pts": sc[k] - base[k] for k in sc}})
    return pd.DataFrame(rows)


def sensitivity_points(
    model: ScoringModel, sens_table: pd.DataFrame, base: EventResults
) -> pd.DataFrame:
    """Points change per display unit of each sensitivity parameter (linearised)."""
    m = model if model.refs else model.calibrate(base)
    times0 = base.times()
    e0 = _energy(base)
    s0 = m.score_values(times0, e0)
    rows = []
    for (param, label), grp in sens_table.groupby(["param", "label"], sort=False):
        g = grp.set_index("event")["per_display"]
        times = {k: v + float(g.get(k, 0.0)) for k, v in times0.items()}
        e = None if e0 is None else e0 + float(g.get("endurance_energy", 0.0))
        s = m.score_values(times, e)
        rows.append(
            {
                "param": param,
                "label": label,
                "display_unit": grp["display_unit"].iloc[0],
                **{f"d_{k}_pts": s[k] - s0[k] for k in s},
            }
        )
    return pd.DataFrame(rows)
