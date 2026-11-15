"""CFD-based part polars for the MEF7 study (replaces the assumed polar of scripts/aero_optimize.py).

1. The front / rear 3-element wings of data/vehicles/fsk2026.yaml get the AERO_AUTOMATION results
   (runs/summary.csv, several summaries per wing are merged) as section_data.
2. AeroLap's own wing model turns the 2-D sections into 3-D ClA / CdA (finite span, end plates, induced
   drag, ground clearance from the ride height) for main-element angles design + d_aoa.
3. The ratios to the design angle scale the MEF7 part budget: ClA_part(d) = ClA0 * ClA(d) / ClA(0), same for
   CdA. The part polar is written as aerolap `points` with scale = 1 + d_aoa / 10 (1.10 = +1 deg).

    python scripts/cfd_polar.py --front C:/aero/front/runs/summary.csv \
        --rear C:/aero/rear/runs/summary.csv C:/aero/rear_h/runs/summary.csv [--out out/cfd_polar]

Writes polar.json (points per part, used by aero_optimize.py --cfd) and polar.txt / polar.png.
"""
import argparse
import copy
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aerolap.aero.devices import geometry_from_cfg  # noqa: E402
from aerolap.aero.maps import parse_components  # noqa: E402

D_AOA = np.arange(-6.0, 10.01, 1.0)  # main-element angle change from the design angle [deg]
SCALE_PER_DEG = 0.1  # polar scale = 1 + 0.1 * d_aoa
SPEED_KMH = 70.0  # speed of the CFD runs (Re 2.3e5-2.7e5)


def merged_summary(paths, out):
    """One csv with the rows of all summaries (same wing geometry at different clearances)."""
    rows, cols = [], []
    for p in paths:
        with open(p, encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                rows.append(r)
                cols += [k for k in r if k not in cols]
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, cols)
        w.writeheader()
        w.writerows(rows)
    return out, sum(r.get("state") == "done" for r in rows)


def wing_polar(car, name, summary, rh_offsets=(0.0,)):
    """3-D ClA, CdA of one wing for design + D_AOA (static attitude, ride height + offsets [mm])."""
    cfg = copy.deepcopy(car)
    comp = copy.deepcopy(cfg["aero"]["components"][name])
    comp["section_data"] = str(summary)
    comp["section_speed_kmh"] = SPEED_KMH  # CFD speed (single-speed data: query it there)
    geo = geometry_from_cfg(cfg)
    a0 = float(comp["aoa_deg"])
    out = {}
    for dz in rh_offsets:
        cla, cda, oor = [], [], []
        for d in D_AOA:
            c = dict(comp, aoa_deg=a0 + d)
            dev = parse_components({name: c}, ROOT / "data" / "vehicles")[0]
            f, r, dd = (float(np.asarray(x)) for x in dev.forces_at(geo, geo.rh_f0 + dz, geo.rh_r0 + dz, 0.0, 0.0))
            cla.append(f + r)
            cda.append(dd)
        out[dz] = (np.array(cla), np.array(cda))
    return a0, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--front", nargs="+", required=True)
    ap.add_argument("--rear", nargs="+", required=True)
    ap.add_argument("--vehicle", default=str(ROOT / "data/vehicles/fsk2026.yaml"), help="wing geometry")
    ap.add_argument("--out", default=str(ROOT / "out/cfd_polar"))
    a = ap.parse_args()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    car = yaml.safe_load(Path(a.vehicle).read_text(encoding="utf-8"))
    car.pop("base", None)
    from aerolap.config import apply_defaults
    car = apply_defaults(car)

    sys.path.insert(0, str(ROOT / "scripts"))
    from aero_targets import PACKAGES
    budget = PACKAGES["realistic"]  # MEF7 part budget at its design angle (scale 1)

    lines, result = [], {"scale_per_deg": SCALE_PER_DEG, "d_aoa": D_AOA.tolist(), "parts": {}}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for name, paths, part, offsets in (("front", a.front, "front", (-15.0, -10.0, -5.0, 0.0, 5.0)),
                                       ("rear", a.rear, "rear", (0.0,))):
        summ, n_done = merged_summary(paths, out / f"{name}_summary.csv")
        a0, pol = wing_polar(car, name, summ, offsets)
        cla, cda = pol[0.0]
        i0 = int(np.argmin(np.abs(D_AOA)))
        cla0, cda0 = budget[part]
        pts = [[1 + SCALE_PER_DEG * d, cla0 * cl / cla[i0], cda0 * cd / cda[i0]] for d, cl, cd in zip(D_AOA, cla, cda)]
        result["parts"][part] = {"points": pts, "design_aoa": a0, "cfd_cases": n_done,
                                 "device_cla": cla.tolist(), "device_cda": cda.tolist(),
                                 "ride_height": {str(k): v[0].tolist() for k, v in pol.items()}}
        k = np.argmax(cla)
        lines.append(f"{name}: design main aoa {a0:.2f} deg, {n_done} CFD cases; device ClA {cla[i0]:.3f} CdA {cda[i0]:.3f} "
                     f"at design; max ClA {cla[k]:.3f} at {a0 + D_AOA[k]:+.1f} deg (d {D_AOA[k]:+.0f})")
        lines.append("  d_aoa  scale   ClA/ClA0  CdA/CdA0  dCdA/dClA (incremental L/D)")
        for j, d in enumerate(D_AOA):
            inc = ((cla[j] - cla[j - 1]) / (cda[j] - cda[j - 1])) if j and cda[j] != cda[j - 1] else float("nan")
            lines.append(f"  {d:+5.0f}  {1 + SCALE_PER_DEG * d:5.2f}   {cla[j] / cla[i0]:7.3f}   {cda[j] / cda[i0]:7.3f}"
                         f"   {inc:8.2f}")
        if len(offsets) > 1:
            lines.append("  ride height (front axle, both axles moved) -> device ClA at design aoa:")
            lines += [f"    {dz:+5.0f} mm: ClA {pol[dz][0][i0]:.3f} ({pol[dz][0][i0] / cla[i0] - 1:+.1%})" for dz in offsets]
        lines.append("")
        axes[0].plot(a0 + D_AOA, cla / cla[i0], "-o", ms=4, label=name)
        axes[1].plot(a0 + D_AOA, cda / cda[i0], "-o", ms=4, label=name)
        axes[2].plot(cda / cda[i0], cla / cla[i0], "-o", ms=4, label=name)
    for ax, t in zip(axes, ("ClA / design", "CdA / design", "polar (ClA vs CdA, / design)")):
        ax.set_title(t, loc="left")
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_xlabel("main element aoa [deg]")
    axes[1].set_xlabel("main element aoa [deg]")
    fig.tight_layout()
    fig.savefig(out / "polar.png", dpi=130)
    (out / "polar.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    (out / "polar.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print("written:", out / "polar.json")


if __name__ == "__main__":
    main()
