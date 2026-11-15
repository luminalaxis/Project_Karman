"""Aero target study: no aero vs 2026 vs realistic / challenge targets (parts budget ClA / CdA).

Writes one vehicle YAML per package (fsk2026.yaml with the aero block replaced by the parts budget)
and runs `aerolap events` (L4) and `aerolap energy` (battery-limited 22 km endurance) on each.

    python scripts/aero_targets.py            # from packages/aerolap

Other car / track (all three aero_* scripts read these):
    AERO_VEHICLE=data/vehicles/mip_228mv_fd4.yaml AERO_TRACK=data/tracks/fsk2026_v6_ol.csv     AERO_OUT=aero_targets_v6ol python scripts/aero_targets.py
"""
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / os.environ.get("AERO_OUT", "aero_targets")
TRACK = os.environ.get("AERO_TRACK", "data/tracks/fsk2026_endurance_smoothed.csv")
BASE_VEHICLE = os.environ.get("AERO_VEHICLE", "data/vehicles/fsk2026.yaml")  # non-aero values come from here
RHO = 1.225

BALANCE = 0.45  # ASSUMED front axle share of the total downforce, same for every package (set by wing
#                 placement / flap trim); keeps the comparison about ClA / CdA, not about balance
MASS = {"front": 3.5, "rear": 4.5, "under": 0.6}  # ASSUMED device masses [kg]

# parts budget (ClA, CdA) from the target table; "body" = chassis + tyres + driver
PACKAGES = {
    "no_aero":   {"body": (0.00, 0.193)},
    "2026":      {"front": (0.47, 0.105), "rear": (0.53, 0.175), "body": (0.00, 0.193)},
    "realistic": {"front": (0.81, 0.209), "rear": (0.52, 0.139), "under": (0.33, 0.108), "body": (-0.15, 0.193)},
    # realistic with drag cut to CdA 0.58 (L/D 2.6): break-even vs 2026 in battery-limited endurance ~0.66 (undertray 0.6 kg)
    "realistic_v2": {"front": (0.81, 0.175), "rear": (0.52, 0.125), "under": (0.33, 0.087), "body": (-0.15, 0.193)},
    "challenge":{"front": (0.81, 0.198), "rear": (0.52, 0.122), "under": (0.33, 0.081), "body": (-0.05, 0.170)},
}
BASE_MASS = 310.0 - MASS["front"] - MASS["rear"]  # 2026 car (310 kg) minus its wings


def vehicle(name, parts):
    car = yaml.safe_load((ROOT / BASE_VEHICLE).read_text(encoding="utf-8"))
    car["name"] = f"FSK aero target: {name}"
    mf = (car.get("tires") or {}).get("mf_file")
    if mf:  # the package YAML is written to OUT: keep the .tir path valid
        car["tires"]["mf_file"] = str(((ROOT / BASE_VEHICLE).parent / mf).resolve())
    car["mass"]["total_kg"] = BASE_MASS + sum(MASS.get(p, 0.0) for p in parts)
    car["aero"] = {**{k: car["aero"][k] for k in ("drag_height_mm", "floor_width_mm", "bottoming_clearance_mm")},
                   "type": "constant", "cla": 0.0, "cda": 0.0, "cop_front": 0.5,
                   "components": {p: {"cla": cla, "cda": cda, "cop_front": BALANCE}
                                  for p, (cla, cda) in parts.items()}}
    path = OUT / f"{name}.yaml"
    path.write_text(yaml.safe_dump(car, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def cli(*args):
    code = "from aerolap.cli import main; import sys; sys.argv=['aerolap']+sys.argv[1:]; main()"
    r = subprocess.run([sys.executable, "-c", code, *args], cwd=ROOT, capture_output=True, text=True,
                       encoding="utf-8", errors="replace",
                       env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONIOENCODING": "utf-8"})
    return r.stdout + r.stderr


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    print("package     ClA    CdA   L/D   mass |  " + "  ".join(f"{v:>3d}km/h L/D[N]" for v in (40, 60, 80, 100)))
    for name, parts in PACKAGES.items():
        cla = sum(c for c, _ in parts.values())
        cda = sum(d for _, d in parts.values())
        q = [0.5 * RHO * (v / 3.6) ** 2 for v in (40, 60, 80, 100)]
        print(f"{name:10s} {cla:5.2f} {cda:6.3f} {cla / cda:5.2f} {BASE_MASS + sum(MASS.get(p, 0) for p in parts):5.1f} | "
              + "  ".join(f"{qq * cla:5.0f}/{qq * cda:4.0f}    " for qq in q))
    for name, parts in PACKAGES.items():
        car = vehicle(name, parts)
        ev = cli("events", str(car), "--level", "L4", "--autocross", TRACK, "--endurance", TRACK,
                 "--out", str(OUT / name))
        t = {k: float(re.search(rf"^{k}\s+([\d.]+)", ev, re.M).group(1))
             for k in ("acceleration", "skidpad", "autocross", "endurance")}
        lap = float(re.search(r"flat out:\s+lap\s+([\d.]+)", ev).group(1))
        cap = re.search(r"power cap ([\d.]+) kW", ev)
        cap = f"{cap.group(1)} kW" if cap else "none, fits flat out"
        print(f"{name:13s} accel {t['acceleration']:.3f}  skid {t['skidpad']:.3f}  autoX {t['autocross']:.3f}  "
              f"lap(flat out) {lap:.3f}  endurance {t['endurance']:.1f} (cap {cap})")
