"""Rear-wing candidate comparison: rank per event and time deltas to a reference."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from ..aero import AeroComponent, synthesize
from ..events import EVENT_NAMES, EventSetup, run_events
from ..parallel import pmap
from ..solver import SolverOptions
from ..vehicle import Vehicle
from ..wing import WingCandidate, WingConversion, candidate_vehicle


@dataclass
class Comparison:
    """Comparison output.

    Attributes:
        table: One row per candidate: aero figures, event times, deltas and ranks.
        reference: ``config_id`` used for the deltas.
        events: Events evaluated.
        dummy: True if any candidate row is DUMMY data.
    """

    table: pd.DataFrame
    reference: str
    events: tuple[str, ...]
    dummy: bool


def compare_candidates(
    vehicle: Vehicle,
    components: Sequence[AeroComponent],
    candidates: Sequence[WingCandidate],
    setup: EventSetup | None = None,
    options: SolverOptions | None = None,
    events: tuple[str, ...] = EVENT_NAMES,
    reference: str | None = None,
    component_name: str = "rear_wing",
    conv: WingConversion | None = None,
    retrim_front: bool = False,
    include_baseline: bool = True,
    workers: int = 1,
) -> Comparison:
    """Simulate every candidate and tabulate times, deltas to ``reference`` and ranks.

    Args:
        vehicle: Baseline vehicle that includes ``components``.
        components: Baseline aero package.
        candidates: Rear-wing candidates.
        setup: Event definitions (default bundled).
        options: Solver options.
        events: Events to run.
        reference: Candidate id for deltas (default: first candidate).
        component_name: Component replaced by the candidates.
        conv: 2D→3D conversion defaults.
        retrim_front: Rescale the front wing to keep the baseline aero balance.
        include_baseline: Add a row for the unmodified baseline package.
        workers: Parallel processes (results identical to the serial run).
    """
    if not candidates:
        raise ValueError("no candidates")
    setup = setup or EventSetup.load()
    ref = reference or candidates[0].config_id
    if ref not in {c.config_id for c in candidates}:
        raise ValueError(f"reference {ref!r} is not a candidate id")
    rows = []
    variants: list[tuple[str, Vehicle, dict]] = []
    if include_baseline:
        base_aero = synthesize(components, vehicle.wheelbase_m)
        variants.append(
            (
                "baseline",
                vehicle,
                {
                    "flap1_id": "-",
                    "flap2_id": "-",
                    "source": "baseline package",
                    "rw_ClA_m2": _rw(components, component_name).cla_m2,
                    "rw_CdA_m2": _rw(components, component_name).cda_m2,
                    "dummy": False,
                    "_bal": base_aero.balance_front,
                },
            )
        )
    for cand in candidates:
        veh, ca = candidate_vehicle(vehicle, components, cand, component_name, conv, retrim_front)
        variants.append(
            (
                cand.config_id,
                veh,
                {
                    "flap1_id": cand.flap1_id,
                    "flap2_id": cand.flap2_id,
                    **cand.params,
                    "source": ca.source,
                    "rw_ClA_m2": ca.component.cla_m2,
                    "rw_CdA_m2": ca.component.cda_m2,
                    "dummy": cand.is_dummy,
                },
            )
        )
    ctx = {"setup": setup, "options": options, "events": events}
    metrics = pmap(_metrics_ctx, [veh for _, veh, _ in variants], ctx, workers)
    for (cid, veh, info), met in zip(variants, metrics, strict=True):
        info = dict(info)
        info.pop("_bal", None)
        row = {
            "config_id": cid,
            **info,
            "ClA_m2": veh.aero.cla_m2,
            "CdA_m2": veh.aero.cda_m2,
            "balance_front": veh.aero.balance_front,
            "mass_kg": veh.mass_kg,
            "cg_height_m": veh.cg_height_m,
        }
        row.update(met)
        rows.append(row)
    df = pd.DataFrame(rows)
    ref_row = df.loc[df["config_id"] == ref].iloc[0]
    cand_mask = df["config_id"] != "baseline"
    for ev in events:
        col = f"{ev}_time_s"
        df[f"d_{ev}_s"] = df[col] - ref_row[col]
        ranks = df.loc[cand_mask, col].rank(method="min").astype(int)
        df[f"rank_{ev}"] = pd.NA
        df.loc[cand_mask, f"rank_{ev}"] = ranks
    if "endurance" in events:
        df["d_endurance_energy_kwh"] = df["endurance_energy_kwh"] - ref_row["endurance_energy_kwh"]
    return Comparison(df, ref, tuple(events), bool(df["dummy"].any()))


def _metrics_ctx(veh: Vehicle, ctx: dict) -> dict[str, float]:
    return run_events(veh, ctx["setup"], ctx["options"], ctx["events"]).metrics()


def _rw(components: Sequence[AeroComponent], name: str) -> AeroComponent:
    for c in components:
        if c.name == name:
            return c
    raise KeyError(f"no aero component named {name!r}")
