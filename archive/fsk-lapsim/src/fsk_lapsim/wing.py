"""Rear-wing candidates: 2D CFD → 3D estimate and conversion into vehicle variants.

**ROUGH ESTIMATE.** A 2D section result does not contain tip losses, endplate effects, the
interaction with the car's wake or the mounting. The finite-wing correction used here is

    Cl3D = k · Cl2D,     Cd3D = Cd2D + Cl3D² / (π · e · AR_eff),
    ClA = Cl3D · S,      CdA = Cd3D · S,      S = chord_ref · span,

with defaults AR_eff = (span/chord)·endplate_factor, k = AR_eff/(AR_eff + 2) (lifting-line
lift-slope ratio) and e = 0.9. Use it only to *rank* section variants of similar planform;
when 3D CFD (or wind-tunnel) values exist, put them in ``cla_3d_m2`` / ``cda_3d_m2`` — they take
priority over the conversion.

Candidate CSV columns (``#`` comment lines allowed):

* required: ``config_id, flap1_id, flap2_id, Cl_2d, Cd_2d, chord_ref_m, span_m``
* optional: ``Cm_2d`` (+ ``cm_ref_x_m``), ``notes``, ``mass_kg``, ``x_cp_m``, ``z_cp_m``,
  ``x_mass_m``, ``z_mass_m``, ``cla_3d_m2``, ``cda_3d_m2``, ``ar_eff``, ``k_lift``, ``oswald_e``
* any other column (flap angles, gaps, positions …) is kept as a free parameter.

Sign conventions: ``Cl_2d`` > 0 means downforce. ``Cm_2d`` is about the vehicle x position
``cm_ref_x_m`` and is positive when the downforce acts *behind* that point, so the centre of
pressure is ``x_cp = cm_ref_x_m + (Cm_2d/Cl_2d)·chord_ref``.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .aero import AeroComponent, replace_component, synthesize
from .vehicle import Vehicle, apply_component_change

ROUGH_NOTE = "2D→3D 변환은 대략적 추정 (rough 2D→3D estimate); 3D CFD 값이 있으면 직접 입력이 우선"

REQUIRED = ["config_id", "flap1_id", "flap2_id", "Cl_2d", "Cd_2d", "chord_ref_m", "span_m"]
KNOWN = set(REQUIRED) | {
    "Cm_2d",
    "cm_ref_x_m",
    "notes",
    "mass_kg",
    "x_cp_m",
    "z_cp_m",
    "x_mass_m",
    "z_mass_m",
    "cla_3d_m2",
    "cda_3d_m2",
    "ar_eff",
    "k_lift",
    "oswald_e",
}


@dataclass(frozen=True)
class WingConversion:
    """Default finite-wing correction parameters (ASSUMED, see ASSUMPTIONS.md).

    Attributes:
        endplate_factor: AR_eff = (span/chord) × factor when ``ar_eff`` is not given.
        oswald_e: Span efficiency e.
        k_lift: Cl3D/Cl2D; ``None`` → k_lift_scale · AR_eff/(AR_eff + 2).
        k_lift_scale: Correction multiplying the default k (uncertainty studies).
    """

    endplate_factor: float = 1.5
    oswald_e: float = 0.9
    k_lift: float | None = None
    k_lift_scale: float = 1.0


@dataclass(frozen=True)
class Wing3D:
    """Result of the 2D → 3D conversion (all per wing)."""

    cl_3d: float
    cd_3d: float
    cd_induced: float
    area_m2: float
    ar_eff: float
    k_lift: float
    oswald_e: float

    @property
    def cla_m2(self) -> float:
        """Downforce coefficient × area [m²]."""
        return self.cl_3d * self.area_m2

    @property
    def cda_m2(self) -> float:
        """Drag coefficient × area [m²]."""
        return self.cd_3d * self.area_m2


def wing_2d_to_3d(
    cl_2d: float,
    cd_2d: float,
    chord_m: float,
    span_m: float,
    ar_eff: float | None = None,
    k_lift: float | None = None,
    oswald_e: float = 0.9,
    endplate_factor: float = 1.5,
    k_lift_scale: float = 1.0,
) -> Wing3D:
    """Rough finite-wing estimate from 2D section coefficients (see module docstring).

    Args:
        cl_2d: 2D lift (downforce) coefficient on ``chord_m`` [-].
        cd_2d: 2D drag coefficient on ``chord_m`` [-].
        chord_m: Reference chord of the 2D coefficients [m].
        span_m: Wing span [m].
        ar_eff: Effective aspect ratio (default (span/chord)·endplate_factor).
        k_lift: Cl3D/Cl2D (default AR_eff/(AR_eff + 2)).
        oswald_e: Span efficiency for induced drag.
        endplate_factor: Used only when ``ar_eff`` is None.
        k_lift_scale: Multiplies the default k (ignored when ``k_lift`` is given).
    """
    if chord_m <= 0 or span_m <= 0:
        raise ValueError("chord and span must be positive")
    ar = ar_eff if ar_eff is not None else span_m / chord_m * endplate_factor
    k = k_lift if k_lift is not None else k_lift_scale * ar / (ar + 2.0)
    cl3 = k * cl_2d
    cdi = cl3 * cl3 / (math.pi * oswald_e * ar)
    return Wing3D(cl3, cd_2d + cdi, cdi, chord_m * span_m, ar, k, oswald_e)


@dataclass(frozen=True)
class WingCandidate:
    """One row of the candidate CSV."""

    config_id: str
    flap1_id: str
    flap2_id: str
    cl_2d: float
    cd_2d: float
    chord_ref_m: float
    span_m: float
    cm_2d: float | None = None
    notes: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    overrides: dict[str, float] = field(default_factory=dict)

    @property
    def is_dummy(self) -> bool:
        """True if the row is marked as DUMMY data."""
        return "DUMMY" in self.notes.upper()

    def label(self) -> str:
        """Short label ``config (flap1/flap2)``."""
        return f"{self.config_id} ({self.flap1_id}/{self.flap2_id})"


def _opt_float(v: Any) -> float | None:
    if v is None or (isinstance(v, float) and math.isnan(v)) or v == "":
        return None
    return float(v)


def read_wing_candidates(path: str | Path) -> list[WingCandidate]:
    """Read and validate a rear-wing candidate CSV."""
    df = pd.read_csv(path, comment="#", skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing column(s) {missing}")
    if df["config_id"].duplicated().any():
        raise ValueError(f"{path}: duplicate config_id")
    out: list[WingCandidate] = []
    for _, row in df.iterrows():
        params = {c: row[c] for c in df.columns if c not in KNOWN}
        overrides = {
            c: float(row[c])
            for c in KNOWN - set(REQUIRED) - {"Cm_2d", "notes"}
            if c in df.columns and _opt_float(row[c]) is not None
        }
        notes = "" if "notes" not in df.columns or pd.isna(row["notes"]) else str(row["notes"])
        cand = WingCandidate(
            config_id=str(row["config_id"]),
            flap1_id=str(row["flap1_id"]),
            flap2_id=str(row["flap2_id"]),
            cl_2d=float(row["Cl_2d"]),
            cd_2d=float(row["Cd_2d"]),
            chord_ref_m=float(row["chord_ref_m"]),
            span_m=float(row["span_m"]),
            cm_2d=_opt_float(row["Cm_2d"]) if "Cm_2d" in df.columns else None,
            notes=notes,
            params=params,
            overrides=overrides,
        )
        if cand.cl_2d <= 0 or cand.cd_2d <= 0:
            raise ValueError(f"{cand.config_id}: Cl_2d (downforce > 0) and Cd_2d must be > 0")
        out.append(cand)
    return out


@dataclass(frozen=True)
class CandidateAero:
    """Rear-wing component derived from a candidate, with provenance."""

    component: AeroComponent
    source: str  # "3D direct" or "2D→3D estimate"
    wing3d: Wing3D | None


def candidate_component(
    cand: WingCandidate, reference: AeroComponent, conv: WingConversion | None = None
) -> CandidateAero:
    """Build the rear-wing :class:`AeroComponent` for a candidate.

    Missing geometry/mass columns fall back to the ``reference`` component (the baseline rear
    wing). Direct 3D values (``cla_3d_m2``/``cda_3d_m2``) take priority over the conversion.
    """
    conv = conv or WingConversion()
    o = cand.overrides
    w3 = None
    if "cla_3d_m2" in o and "cda_3d_m2" in o:
        cla, cda, source = o["cla_3d_m2"], o["cda_3d_m2"], "3D direct"
    else:
        w3 = wing_2d_to_3d(
            cand.cl_2d,
            cand.cd_2d,
            cand.chord_ref_m,
            cand.span_m,
            ar_eff=o.get("ar_eff"),
            k_lift=o.get("k_lift", conv.k_lift),
            oswald_e=o.get("oswald_e", conv.oswald_e),
            endplate_factor=conv.endplate_factor,
        )
        cla, cda, source = w3.cla_m2, w3.cda_m2, "2D→3D estimate"
    x_cp = o.get("x_cp_m")
    if x_cp is None and cand.cm_2d is not None and "cm_ref_x_m" in o:
        x_cp = o["cm_ref_x_m"] + cand.cm_2d / cand.cl_2d * cand.chord_ref_m
    comp = AeroComponent(
        name=reference.name,
        cla_m2=cla,
        cda_m2=cda,
        x_cp_m=reference.x_cp_m if x_cp is None else x_cp,
        z_cp_m=o.get("z_cp_m", reference.z_cp_m),
        mass_kg=o.get("mass_kg", reference.mass_kg),
        x_mass_m=o.get("x_mass_m", reference.x_mass_m),
        z_mass_m=o.get("z_mass_m", reference.z_mass_m),
    )
    return CandidateAero(comp, source, w3)


def retrim_component(
    components: Sequence[AeroComponent],
    target_balance: float,
    wheelbase_m: float,
    name: str = "front_wing",
) -> list[AeroComponent]:
    """Scale ClA and CdA of component ``name`` so the package hits ``target_balance``.

    Mimics re-trimming the front wing angle after a rear-wing change (mass unchanged).
    """
    comps = list(components)
    idx = next((i for i, c in enumerate(comps) if c.name == name), None)
    if idx is None:
        raise KeyError(f"no aero component named {name!r}")
    others = [c for i, c in enumerate(comps) if i != idx]
    a_f = sum(c.front_cla_m2(wheelbase_m) for c in others)
    a_t = sum(c.cla_m2 for c in others)
    fw = comps[idx]
    k_f, k_t = fw.front_cla_m2(wheelbase_m), fw.cla_m2
    denom = k_f - target_balance * k_t
    if abs(denom) < 1e-12:
        raise ValueError("component cannot change the balance")
    scale = (target_balance * a_t - a_f) / denom
    if scale <= 0:
        raise ValueError(f"retrim needs a non-positive scale ({scale:.3f})")
    comps[idx] = fw.scaled(scale, scale)
    return comps


def candidate_vehicle(
    vehicle: Vehicle,
    components: Sequence[AeroComponent],
    cand: WingCandidate,
    component_name: str = "rear_wing",
    conv: WingConversion | None = None,
    retrim_front: bool = False,
) -> tuple[Vehicle, CandidateAero]:
    """Vehicle with the candidate rear wing fitted (mass, CG and aero updated).

    ``vehicle`` must include ``components`` (the baseline package). With ``retrim_front`` the
    front wing is rescaled to keep the baseline aero balance.
    """
    ref = next((c for c in components if c.name == component_name), None)
    if ref is None:
        raise KeyError(f"no aero component named {component_name!r}")
    ca = candidate_component(cand, ref, conv)
    new = replace_component(components, ca.component)
    if retrim_front:
        base_bal = synthesize(components, vehicle.wheelbase_m).balance_front
        new = retrim_component(new, base_bal, vehicle.wheelbase_m)
    veh = apply_component_change(vehicle, components, new)
    return dataclasses.replace(veh, name=f"{vehicle.name}+{cand.config_id}"), ca


def candidate_table(
    cands: Sequence[WingCandidate], reference: AeroComponent, conv: WingConversion | None = None
) -> pd.DataFrame:
    """Per-candidate aero summary (2D inputs, 3D estimate, L/D)."""
    rows = []
    for c in cands:
        ca = candidate_component(c, reference, conv)
        w = ca.wing3d
        rows.append(
            {
                "config_id": c.config_id,
                "flap1_id": c.flap1_id,
                "flap2_id": c.flap2_id,
                **c.params,
                "Cl_2d": c.cl_2d,
                "Cd_2d": c.cd_2d,
                "L/D_2d": c.cl_2d / c.cd_2d,
                "source": ca.source,
                "ClA_m2": ca.component.cla_m2,
                "CdA_m2": ca.component.cda_m2,
                "L/D_3d": ca.component.cla_m2 / ca.component.cda_m2,
                "AR_eff": np.nan if w is None else w.ar_eff,
                "k_lift": np.nan if w is None else w.k_lift,
                "mass_kg": ca.component.mass_kg,
                "dummy": c.is_dummy,
            }
        )
    return pd.DataFrame(rows)
