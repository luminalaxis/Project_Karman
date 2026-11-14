"""Report figures: lap times, endurance speed trace, downforce / drag vs speed, balance x AoA grid.

Runs the recommended package (realistic polar, scale 1.10, balance 0.55) through `aerolap events`, then
plots it against no aero / 2026 / realistic target from scripts/aero_targets.py outputs.

    python scripts/aero_targets.py && python scripts/aero_report_plots.py     # from packages/aerolap
"""
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from aero_optimize import BALANCE, NOMINAL, car_cfg, totals  # noqa: E402
from aero_targets import OUT, RHO, TRACK, cli  # noqa: E402

FIG = OUT / "figures"
PKG = {  # name -> (label, colour: categorical slots 1-4 in fixed order)
    "no_aero": ("No-aero", "#2a78d6"),
    "2026": ("MEF6", "#eb6834"),
    "recommended": ("MEF7 (final)", "#eda100"),
}
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e4e3df", "grid.linewidth": 0.8, "axes.edgecolor": "#8a8984",
                     "text.color": "#0b0b0b", "axes.labelcolor": "#52514e", "xtick.color": "#52514e",
                     "ytick.color": "#52514e", "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                     "savefig.dpi": 160})


def save(fig, name):
    """Render in memory, then write with retries: OneDrive sync briefly locks files it is uploading."""
    import io
    import time
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    for i in range(10):
        try:
            (FIG / name).write_bytes(buf.getvalue())
            return
        except OSError:
            time.sleep(0.5 * (i + 1))
    raise OSError(f"could not write {FIG / name} (locked by OneDrive?)")


def run_recommended():
    cfg = car_cfg(NOMINAL, {p: 1.10 for p in ("front", "rear", "under")}, 0.55 - BALANCE)
    path = OUT / "recommended.yaml"
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    cli("events", str(path), "--level", "L4", "--autocross", TRACK, "--endurance", TRACK,
        "--out", str(OUT / "recommended"))
    return totals(cfg)


def times(name):
    txt = (OUT / name / "events.txt").read_text(encoding="utf-8")
    return {k: float(re.search(rf"^{k}\s+([\d.]+)", txt, re.M).group(1))
            for k in ("acceleration", "skidpad", "autocross", "endurance")}


def fig_lap_times(t):
    events = [("autocross", "오토크로스 [s]"), ("skidpad", "스키드패드 [s]"), ("endurance", "내구 22 km [s]")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    names = list(PKG)
    for ax, (ev, title) in zip(axes, events):
        vals = np.array([t[n][ev] for n in names])
        span = vals.max() - vals.min()
        ax.barh(range(len(names)), vals, color=[PKG[n][1] for n in names], height=0.6)
        ax.set_xlim(vals.min() - 0.6 * span, vals.max() + 0.5 * span)
        ax.set_yticks(range(len(names)), [PKG[n][0] for n in names])
        ax.invert_yaxis()
        ax.set_title(title, loc="left", fontsize=11)
        ax.grid(axis="y", visible=False)
        for i, v in enumerate(vals):
            d = v - t["2026"][ev]
            ax.text(v + 0.02 * span, i, f"{v:.3f}" + ("" if names[i] == "2026" else f" ({d:+.2f})"),
                    va="center", fontsize=9, color="#52514e")
    fig.suptitle("종목별 시간 (괄호: MEF6 대비, 축은 0에서 시작하지 않음)", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    save(fig, "lap_times.png")


def fig_speed_trace():
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ref = np.genfromtxt(OUT / "2026" / "trace_endurance.csv", delimiter=",", names=True)
    for n, (label, col) in PKG.items():
        d = np.genfromtxt(OUT / n / "trace_endurance.csv", delimiter=",", names=True)
        a1.plot(d["s_m"], d["v_kmh"], color=col, lw=1.6, label=label)
        if n != "2026":
            a2.plot(d["s_m"], np.interp(d["s_m"], d["s_m"], d["v_kmh"]) - np.interp(d["s_m"], ref["s_m"], ref["v_kmh"]),
                    color=col, lw=1.4, label=label)
    a1.set_ylabel("속도 [km/h]")
    a1.set_title("내구 1랩 속도 (배터리 한계 출력 제한 포함)", loc="left", fontsize=12)
    a1.legend(ncol=4, frameon=False, loc="upper right")
    a2.axhline(0, color="#8a8984", lw=1)
    a2.set_ylabel("MEF6 대비 [km/h]")
    a2.set_xlabel("거리 [m]")
    a2.legend(ncol=3, frameon=False, loc="lower right")
    fig.tight_layout()
    save(fig, "speed_trace_endurance.png")


def fig_forces(coef):
    v = np.linspace(0, 110, 111)
    q = 0.5 * RHO * (v / 3.6) ** 2
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, k, title in ((axes[0], 0, "다운포스 [N]"), (axes[1], 1, "드래그 [N]")):
        for n, (label, col) in PKG.items():
            if n == "no_aero" and k == 0:
                continue
            f = q * coef[n][k]
            ax.plot(v, f, color=col, lw=2, label=f"{label} ({coef[n][k]:.2f})")
            ax.plot(80, np.interp(80, v, f), "o", ms=8, color=col, mec="#fcfcfb", mew=2)
        ax.set_title(title + "  · 점: 80 km/h", loc="left", fontsize=12)
        ax.set_xlabel("속도 [km/h]")
        ax.legend(frameon=False, title="ClA" if k == 0 else "CdA")
    fig.tight_layout()
    save(fig, "downforce_drag_vs_speed.png")


def fig_grid():
    rows = re.findall(r"s=([\d.]+) bal=([\d.]+) nom\s.*?endurance\s+([\d.]+)",
                      (OUT / "grid_log.txt").read_text(encoding="utf-8"))
    data = {}
    for s, b, e in rows:
        data.setdefault(float(s), []).append((float(b), float(e)))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    blues = ["#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#104281"]  # sequential: one hue, light -> dark
    for col, (s, pts) in zip(blues, sorted(data.items())):
        b, e = np.array(sorted(pts)).T
        ax.plot(b, e, "-o", color=col, lw=2, ms=6, label=f"받음각 배율 {s:.2f}")
    ax.axvline(0.55, color="#8a8984", ls="--", lw=1)
    ax.text(0.553, ax.get_ylim()[1], "추천 상한 0.55\n(넘으면 중립·오버스티어 쪽)", va="top", fontsize=8.5, color="#52514e")
    ax.set_xlabel("공력 밸런스 (앞축 비율)")
    ax.set_ylabel("내구 22 km [s]")
    ax.set_title("밸런스 × 받음각 그리드 (기본 폴라)", loc="left", fontsize=12)
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    save(fig, "balance_aoa_grid.png")


def native_figures():
    """AeroLap's own outputs for MEF6 and MEF7 (final): lap.png / track_map.png (run) and the aero exchange
    rates for battery-limited endurance (aero_exchange.png)."""
    for name in ("2026", "recommended"):
        car, d = str(OUT / f"{name}.yaml"), FIG / "aerolap" / PKG[name][0].split()[0]
        cli("run", car, TRACK, "--level", "L4", "--out", str(d / "lap"))
        cli("aero-exchange", car, TRACK, "--level", "L4", "--objective", "endurance_time", "--out", str(d / "exchange"))


def optimumlap_check(coef, olveh="data/vehicles/MIP_228MV_FD4_Diagnostic.OLVeh", area=0.635):
    """What OptimumLap itself would show: L0 model (point mass, no balance / battery), only ClA / CdA changed.
    Enter cl = ClA / area, cd = CdA / area in OptimumLap to reproduce."""
    lines = [f"OptimumLap-equivalent (L0) one lap, {olveh}, track {TRACK}",
             f"{'package':14s} {'ClA':>6s} {'CdA':>6s} {'OL cl':>6s} {'OL cd':>6s} {'lap [s]':>8s}"]
    for name, (cla, cda) in coef.items():
        if name not in PKG:
            continue
        txt = cli("run", olveh, TRACK, "--level", "L0", "--set", f"aero.cl={cla / area}",
                  "--set", f"aero.cd={cda / area}", "--out", str(OUT / "optimumlap_L0" / name))
        lap = float(re.search(r"lap_time_s\s+([\d.]+)", txt).group(1))
        lines.append(f"{PKG[name][0]:14s} {cla:6.3f} {cda:6.3f} {cla / area:6.3f} {cda / area:6.3f} {lap:8.3f}")
    (OUT / "optimumlap_L0.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    rec = run_recommended()
    native_figures()
    coef = {"no_aero": (0.0, 0.193), "2026": (1.00, 0.473), "realistic": (1.51, 0.649), "recommended": rec}
    t = {n: times(n) for n in PKG}
    for n in PKG:
        print(f"{PKG[n][0]:8s} " + "  ".join(f"{k} {v:.3f}" for k, v in t[n].items()))
    fig_lap_times(t)
    fig_speed_trace()
    fig_forces(coef)
    fig_grid()
    optimumlap_check(coef)
    print("figures:", *sorted(str(p.relative_to(FIG)) for p in FIG.rglob("*.png")))
