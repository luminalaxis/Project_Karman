"""Aero-only optimisation of the realistic-target package (scripts/aero_targets.py "realistic").

Free (aero only): angle-of-attack scale of front wing / rear wing / undertray along each part's polar,
and the aero balance (aero.cop_shift). Everything else (mass, battery, power, suspension, tyres) is
the fsk2026.yaml car exactly as in aero_targets.py.

Polar of a part (scale s = 1 is its target point ClA0 / CdA0):
    ClA(s) = ClA0 * s                       up to stall (s_stall), then flat and falling 10 % by s = 1.6
    CdA(s) = CdA0 * (p + (1 - p) * s^2)     p = profile/parasitic share, the rest grows with lift^2 (induced)
Nominal p: wings 0.35, undertray 0.60, stall 1.30. The optimum is re-checked on a pessimistic polar
(p 0.20 / 0.40, stall 1.15) so the gain does not rely on the polar assumption.

    python scripts/aero_optimize.py            # from packages/aerolap, ~15-30 min
"""
import copy
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from aero_targets import BALANCE, OUT, PACKAGES, TRACK, vehicle  # noqa: E402
from aerolap.analysis.objectives import Objective  # noqa: E402
from aerolap.analysis.optimize import OptimizationSpec, Variable, run_optimization  # noqa: E402
from aerolap.config import load_vehicle_config as load_vehicle  # noqa: E402

NOMINAL = {"wing": (0.35, 1.30), "under": (0.60, 1.30)}
PESSIMISTIC = {"wing": (0.20, 1.15), "under": (0.40, 1.15)}
S = np.round(np.arange(0.4, 1.61, 0.05), 2)
UNDERSTEER = [1.0, 1.3]  # same window as aerolap aero-split
COP_SHIFT = (-0.10, 0.10)  # balance 0.35 .. 0.55


def polar(cla0, cda0, p, s_stall):
    cla = np.where(S <= s_stall, S, s_stall * (1 - 0.10 * (S - s_stall) / (1.6 - s_stall))) * cla0
    cda = cda0 * (p + (1 - p) * S ** 2)
    return np.column_stack([S, cla, cda]).tolist()


def car_cfg(polars, scales=None, cop_shift=0.0, cfd=None):
    """``cfd``: polar.json of scripts/cfd_polar.py - its CFD points replace the assumed polar of those parts."""
    cfg = load_vehicle(vehicle("realistic", PACKAGES["realistic"]))
    for part, (cla0, cda0) in PACKAGES["realistic"].items():
        if part == "body":
            continue
        p, st = polars["under" if part == "under" else "wing"]
        c = cfg["aero"]["components"][part]
        c["points"] = cfd["parts"][part]["points"] if cfd and part in cfd["parts"] else polar(cla0, cda0, p, st)
        c["scale"] = 1.0 if scales is None else float(scales[part])
    cfg["aero"]["cop_shift"] = float(cop_shift)
    return cfg


def totals(cfg):
    from aerolap.aero.maps import parse_components
    cla = cda = 0.0
    for c in parse_components(cfg["aero"]["components"]):
        f, r, d = c.forces()
        cla, cda = cla + f + r, cda + d
    return cla, cda


def evaluate(cfg):
    """Fine envelope: endurance (battery limited), one flying lap, skidpad, understeer window."""
    out = {}
    for typ in ("endurance_time", "lap_time", "skidpad_time"):
        spec = {"type": typ, "track": str(ROOT / TRACK), "level": "L4"}
        ev = Objective(spec, ROOT, {"understeer_index": UNDERSTEER})(copy.deepcopy(cfg))
        out[typ] = ev.value
        out["feasible"] = out.get("feasible", True) and ev.feasible
        out["us"] = (ev.metrics.get("understeer_index_min"), ev.metrics.get("understeer_index_max"))
    return out


def optimise(objective, max_evals):
    base = car_cfg(NOMINAL)
    names = ["front", "rear", "under"]
    spec = OptimizationSpec(
        vehicle_cfg=base,
        objective_spec={"type": objective, "track": str(ROOT / TRACK), "level": "L4"},
        variables=[Variable(f"aero.components.{n}.scale", 0.4, 1.6) for n in names]
        + [Variable("aero.cop_shift", *COP_SHIFT)],
        constraints={"understeer_index": UNDERSTEER}, base_dir=ROOT, max_evals=max_evals)
    res = run_optimization(spec, lambda n, x, ev, err: print(
        f"  [{objective}] {n:3d} " + " ".join(f"{k.split('.')[-2] if 'scale' in k else 'cop'}={v:.3f}"
                                              for k, v in x.items())
        + (f" -> {ev.value:.3f}{'' if ev.feasible else ' X'}" if ev else f" error {err}"), flush=True))
    b = res.best
    return {n: b[f"aero.components.{n}.scale"] for n in names}, b["aero.cop_shift"]


def row(label, cfg):
    cla, cda = totals(cfg)
    e = evaluate(cfg)
    bal = BALANCE + cfg["aero"]["cop_shift"]
    print(f"{label:34s} ClA {cla:5.3f} CdA {cda:5.3f} L/D {cla / cda:4.2f} bal {bal:4.2f} | "
          f"endurance {e['endurance_time']:7.1f}  lap {e['lap_time']:6.3f}  skid {e['skidpad_time']:5.3f}  "
          f"US {e['us'][0]:.2f}..{e['us'][1]:.2f}{'' if e['feasible'] else '  INFEASIBLE'}", flush=True)
    return e


def grid():
    """Robust check: uniform scale x balance on both polars (basis of the recommendation)."""
    for s in (1.0, 1.05, 1.1, 1.15, 1.2):
        for bal in (0.45, 0.50, 0.55, 0.60):
            for name, pol in (("nom", NOMINAL), ("pes", PESSIMISTIC)):
                row(f"s={s:.2f} bal={bal:.2f} {name}", car_cfg(pol, {p: s for p in ("front", "rear", "under")},
                                                             bal - BALANCE))


def grid_cfd(path):
    """CFD polars for front / rear (independent angles), assumed nominal polar for the undertray (scale 1.10)."""
    import json
    cfd = json.loads(Path(path).read_text(encoding="utf-8"))
    k = cfd["scale_per_deg"]
    print(f"CFD polar: {path}  (scale = 1 + {k} x d_aoa)")
    for df in (-2, 0, 1, 2, 3, 4):
        for dr in (-2, 0, 1, 2, 3, 4):
            for bal in (0.45, 0.50, 0.55, 0.60):
                row(f"front {df:+d} rear {dr:+d} bal={bal:.2f}",
                    car_cfg(NOMINAL, {"front": 1 + k * df, "rear": 1 + k * dr, "under": 1.10}, bal - BALANCE, cfd))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    if "--cfd" in sys.argv:  # python scripts/aero_optimize.py --cfd out/cfd_polar/polar.json   (~15 min)
        grid_cfd(sys.argv[sys.argv.index("--cfd") + 1])
        sys.exit()
    if "--grid" in sys.argv:  # python scripts/aero_optimize.py --grid   (~3 min)
        grid()
        sys.exit()
    found = {}
    for obj, n in (("endurance_time", 120), ("lap_time", 120)):
        print(f"\n=== optimise {obj} ===", flush=True)
        found[obj] = optimise(obj, n)
        print("  best:", found[obj], flush=True)

    print("\n=== results (fine envelope, L4) ===")
    for pol_name, pol in (("nominal polar", NOMINAL), ("pessimistic polar", PESSIMISTIC)):
        print(f"-- {pol_name}")
        row("realistic (current target)", car_cfg(pol))
        for obj, (sc, cs) in found.items():
            cfg = car_cfg(pol, sc, cs)
            row(f"opt {obj}: " + "/".join(f"{sc[k]:.2f}" for k in ("front", "rear", "under")), cfg)
            if pol is NOMINAL:
                for part in ("front", "rear", "under"):
                    c = next(c for c in __import__("aerolap.aero.maps", fromlist=["x"]).parse_components(
                        {part: cfg["aero"]["components"][part]}))
                    f, r, d = c.forces()
                    print(f"     {part:6s} scale {sc[part]:.2f}: ClA {f + r:.3f}  CdA {d:.3f}")
