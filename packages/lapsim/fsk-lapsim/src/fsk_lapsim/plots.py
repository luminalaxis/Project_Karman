"""Matplotlib figures (English labels to avoid CJK font issues).

Colour roles follow one rule each: categorical slots in fixed order for identity, a single
blue ramp (light → dark) for magnitude, blue ↔ red with a grey midpoint for signed effects.
"""

from __future__ import annotations

import base64
import io
import math
from collections.abc import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from .active_aero import DRSLap
from .analysis.balance_opt import BalanceOptResult
from .analysis.compare import Comparison
from .analysis.sensitivity import SensitivityResult
from .analysis.sweep import SweepResult
from .energy import operating_point_histogram
from .ggv import GGV
from .solver import LapResult
from .track import Track
from .vehicle import Vehicle

#: Categorical slots (fixed order, never cycled).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
ORANGE_RAMP = ["#fde3d6", "#f7b99c", "#f18f63", "#eb6834", "#c8501f", "#9a3d17", "#6c2b10"]
POS, NEG, MID = "#e34948", "#2a78d6", "#c9c7c1"  # slower (+), faster (−), neutral
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

SEQ_BLUE = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)
SEQ_ORANGE = LinearSegmentedColormap.from_list("seq_orange", ORANGE_RAMP)

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 110,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.edgecolor": "#b9b8b2",
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.6,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.axisbelow": True,
    }
)


def fig_to_png_bytes(fig: Figure) -> bytes:
    """Render a figure to PNG bytes and close it."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def fig_to_base64(fig: Figure) -> str:
    """Render a figure to a base64 PNG (for embedding in HTML)."""
    return base64.b64encode(fig_to_png_bytes(fig)).decode("ascii")


def save(fig: Figure, path: str) -> None:
    """Save a figure as PNG and close it."""
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------------------------
def speed_trace(laps: Sequence[tuple[str, LapResult]], title: str = "Speed trace") -> Figure:
    """Speed (top) and accelerations (bottom) along the lap; one colour per lap."""
    fig, (a1, a2, a3) = plt.subplots(
        3, 1, figsize=(10, 6.4), sharex=True, gridspec_kw={"height_ratios": [2.2, 1, 1]}
    )
    for k, (label, lap) in enumerate(laps[: len(SERIES)]):
        c = SERIES[k]
        a1.plot(lap.s, lap.v * 3.6, color=c, label=label)
        a2.plot(lap.s, lap.ax / 9.81, color=c, lw=1.1)
        a3.plot(lap.s, lap.ay / 9.81, color=c, lw=1.1)
    first = laps[0][1]
    lim = np.minimum(first.v_lim, first.v.max() * 1.25) * 3.6
    a1.plot(first.s, lim, color=INK2, lw=0.8, ls=":", label="curvature limit (" + laps[0][0] + ")")
    a1.set_ylabel("speed [km/h]")
    a2.set_ylabel("ax [g]")
    a3.set_ylabel("ay [g]")
    a3.set_xlabel("distance [m]")
    a1.set_title(title, loc="left")
    a1.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=min(len(laps) + 1, 4))
    fig.tight_layout()
    return fig


def track_map(track: Track, lap: LapResult, title: str | None = None) -> Figure:
    """Driven line coloured by speed (single-hue sequential ramp)."""
    x, y = track.xy(lap.s)
    pts = np.column_stack([x, y]).reshape(-1, 1, 2)
    seg = np.concatenate([pts[:-1], pts[1:]], axis=1)
    vk = 0.5 * (lap.v[1:] + lap.v[:-1]) * 3.6
    fig, ax = plt.subplots(figsize=(7.5, 6))
    lc = LineCollection(seg, cmap=SEQ_BLUE, linewidths=3.0)
    lc.set_array(vk)
    ax.add_collection(lc)
    ax.plot(
        x[0],
        y[0],
        marker="o",
        ms=8,
        color=SERIES[1],
        mec="white",
        mew=1.5,
        ls="none",
        label="start",
    )
    ax.annotate("start", (x[0], y[0]), xytext=(6, 6), textcoords="offset points", color=INK2)
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.grid(False)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    cb = fig.colorbar(lc, ax=ax, shrink=0.8)
    cb.set_label("speed [km/h]")
    ax.set_title(title or f"{track.name}: {lap.lap_time_s:.2f} s", loc="left")
    return fig


def ggv_plot(ggv: GGV, speeds_mps: Sequence[float] = (8.0, 16.0, 24.0, 32.0)) -> Figure:
    """g-g envelopes at several speeds (ax vs ay in g)."""
    fig, ax = plt.subplots(figsize=(6.4, 6))
    th = np.linspace(0.0, math.pi / 2, 120)
    for k, v in enumerate([s for s in speeds_mps if s <= ggv.v_top][:4]):
        aym = float(ggv.ay_max_at(v))
        ay = np.sin(th) * aym
        vv = np.full_like(ay, v)
        acc = ggv.ax_acc_at(vv, ay) / 9.81
        brk = ggv.ax_brk_at(vv, ay) / 9.81
        yy = np.concatenate([acc, brk[::-1]])
        xx = np.concatenate([ay, ay[::-1]]) / 9.81
        ax.plot(
            np.concatenate([xx, -xx[::-1]]),
            np.concatenate([yy, yy[::-1]]),
            color=SERIES[k],
            label=f"{v * 3.6:.0f} km/h",
        )
    ax.axhline(0, color="#b9b8b2", lw=0.8)
    ax.axvline(0, color="#b9b8b2", lw=0.8)
    ax.set_xlabel("lateral acceleration ay [g]")
    ax.set_ylabel("longitudinal acceleration ax [g]")
    ax.set_aspect("equal")
    ax.set_title(f"GGV envelope ({ggv.model_name} model)", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4)
    return fig


def sensitivity_bars(sens: SensitivityResult, event: str = "autocross") -> Figure:
    """Signed sensitivity per display unit (red = slower, blue = faster)."""
    t = sens.table[sens.table["event"] == event].copy()
    t = t.reindex(t["per_display"].abs().sort_values().index)
    labels = [f"{r.label} (+{r.display_unit})" for r in t.itertuples()]
    vals = t["per_display"].to_numpy()
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(t) + 1.2))
    colors = [POS if v > 0 else NEG if v < 0 else MID for v in vals]
    ax.barh(labels, vals, color=colors, height=0.6)
    span = max(abs(vals).max(), 1e-9)
    for i, v in enumerate(vals):
        ax.text(
            v + (0.02 * span if v >= 0 else -0.02 * span),
            i,
            f"{v:+.3f} s",
            va="center",
            ha="left" if v >= 0 else "right",
            color=INK,
            fontsize=8,
        )
    ax.axvline(0, color="#8d8c86", lw=0.8)
    ax.set_xlim(-1.35 * span, 1.35 * span)
    ax.set_xlabel(f"Δ {event} time [s] per display unit  (red = slower, blue = faster)")
    ax.grid(axis="y", visible=False)
    ax.set_title(f"Sensitivity — {event}", loc="left")
    fig.tight_layout()
    return fig


def sweep_contours(sw: SweepResult) -> Figure:
    """ClA–CdA maps: autocross time (blue) and endurance energy (orange), marked designs."""
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    cl, cd = np.meshgrid(sw.cla, sw.cda)
    panels = [
        (
            axs[0],
            sw.autocross_s,
            SEQ_BLUE.reversed(),
            "autocross time [s]",
            "autocross_s",
            SERIES[1],
        ),
        (
            axs[1],
            sw.endurance_kwh,
            SEQ_ORANGE,
            "endurance energy [kWh]",
            "endurance_kwh",
            SERIES[0],
        ),
    ]
    for ax, z, cmap, lab, key, mark in panels:
        cf = ax.contourf(cl, cd, z, levels=12, cmap=cmap)
        cs = ax.contour(cl, cd, z, levels=6, colors="white", linewidths=0.8)
        ax.clabel(cs, fmt="%.2f" if key == "autocross_s" else "%.1f", fontsize=7, colors=INK)
        cb = fig.colorbar(cf, ax=ax)
        cb.set_label(lab)
        ax.set_xlabel("ClA [m²]")
        ax.grid(False)
        if len(sw.points):
            for k, r in enumerate(sw.points.itertuples()):
                c = INK if k == 0 else mark
                ax.plot(
                    r.ClA_m2,
                    r.CdA_m2,
                    marker="o" if k == 0 else "s",
                    ms=8 if k == 0 else 6,
                    color=c,
                    mec="white",
                    mew=1.2,
                    ls="none",
                )
                if k == 0 or len(sw.points) <= 6:
                    ax.annotate(
                        str(r.label),
                        (r.ClA_m2, r.CdA_m2),
                        xytext=(5, 4),
                        textcoords="offset points",
                        fontsize=7,
                        color=INK,
                    )
    axs[0].set_ylabel("CdA [m²]")
    axs[0].set_title("Autocross lap time", loc="left")
    axs[1].set_title("Endurance energy (accumulator)", loc="left")
    fig.tight_layout()
    return fig


def motor_map(lap: LapResult, vehicle: Vehicle) -> Figure:
    """Time-weighted motor operating points with torque and power limit curves."""
    pt = vehicle.powertrain
    hist, re_, te = operating_point_histogram(
        lap.energy, 40, 48, pt.motor_max_rpm * 1.02, pt.motor_max_torque_nm * 1.05
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    h = np.where(hist > 0, hist, np.nan)
    m = ax.pcolormesh(re_, te, h.T, cmap=SEQ_BLUE, shading="flat")
    cb = fig.colorbar(m, ax=ax)
    cb.set_label("time in cell [s]")
    rpm = np.linspace(1.0, pt.motor_max_rpm, 300)
    omega = rpm * 2 * math.pi / 60
    tlim = np.minimum(pt.motor_max_torque_nm, pt.max_power_w / omega)
    ax.plot(rpm, tlim, color=SERIES[1], lw=1.6, label="drive limit (torque / power)")
    rlim = (
        np.minimum(pt.motor_max_torque_nm, pt.regen_max_power_w / (pt.efficiency * omega))
        if pt.regen_enabled
        else np.zeros_like(rpm)
    )
    ax.plot(rpm, -rlim, color=SERIES[2], lw=1.6, label="regen limit")
    ax.axhline(0, color="#8d8c86", lw=0.8)
    ax.set_xlabel("motor speed [rpm]")
    ax.set_ylabel("motor torque [N·m]")
    ax.legend(loc="lower left")
    ax.grid(False)
    ax.set_title(f"Motor operating points — {lap.track_name}", loc="left")
    return fig


def compare_deltas(cmp: Comparison) -> Figure:
    """Δ time to the reference per event (one small panel per event: scales differ)."""
    t = cmp.table
    evs = list(cmp.events)
    fig, axs = plt.subplots(
        1, len(evs), figsize=(3.2 * len(evs), 0.38 * len(t) + 1.4), sharey=True, squeeze=False
    )
    for ax, ev in zip(axs[0], evs, strict=True):
        vals = t[f"d_{ev}_s"].to_numpy()
        colors = [POS if v > 1e-9 else NEG if v < -1e-9 else MID for v in vals]
        ax.barh(t["config_id"], vals, color=colors, height=0.6)
        ax.axvline(0, color="#8d8c86", lw=0.8)
        ax.set_title(ev, loc="left")
        ax.set_xlabel("Δ time [s]")
        ax.grid(axis="y", visible=False)
        span = max(np.abs(vals).max(), 1e-6)
        ax.set_xlim(-1.3 * span, 1.3 * span)
        ax.xaxis.set_major_locator(MaxNLocator(3, symmetric=True))
    axs[0][0].invert_yaxis()
    fig.suptitle(
        f"Rear-wing candidates vs {cmp.reference} (red = slower)"
        + ("  [DUMMY DATA]" if cmp.dummy else ""),
        x=0.01,
        ha="left",
        fontsize=10,
    )
    fig.tight_layout()
    return fig


def balance_scan(res: BalanceOptResult) -> Figure:
    """Objective versus front aero balance."""
    fig, ax = plt.subplots(figsize=(7, 3.8))
    sc = res.scan
    ax.plot(sc.iloc[:, 0] * 100, sc.iloc[:, 1], color=SERIES[0], marker="o", ms=3)
    ax.axvline(res.best_balance * 100, color=SERIES[1], lw=1.0, ls="--")
    ax.annotate(
        f"optimum {res.best_balance * 100:.1f} %",
        (res.best_balance * 100, res.best_value),
        xytext=(6, 12),
        textcoords="offset points",
        color=INK,
    )
    ax.axvline(res.base_balance * 100, color=INK2, lw=0.8, ls=":")
    ax.annotate(
        f"current {res.base_balance * 100:.1f} %",
        (res.base_balance * 100, res.base_value),
        xytext=(6, -14),
        textcoords="offset points",
        color=INK2,
    )
    ax.set_xlabel("front aero balance [%]")
    ax.set_ylabel(f"{res.objective} [s]")
    ax.set_title("Aero balance scan", loc="left")
    fig.tight_layout()
    return fig


def drs_trace(lap: DRSLap) -> Figure:
    """Speed with and without DRS, and the flap state (0 closed … 1 open)."""
    fig, (a1, a2) = plt.subplots(
        2, 1, figsize=(10, 4.6), sharex=True, gridspec_kw={"height_ratios": [2.3, 1]}
    )
    a1.plot(lap.closed.s, lap.closed.v * 3.6, color=SERIES[0], label="closed (high downforce)")
    a1.plot(lap.drs.s, lap.drs.v * 3.6, color=SERIES[1], label="DRS")
    a1.set_ylabel("speed [km/h]")
    a1.legend(loc="lower right")
    a1.set_title(f"DRS — {lap.drs.track_name}: gain {lap.time_gain_s:.3f} s", loc="left")
    a2.fill_between(lap.drs.s, 0, lap.weight, color=SERIES[1], alpha=0.35, lw=0)
    a2.plot(lap.drs.s, lap.weight, color=SERIES[1], lw=1.0)
    a2.set_ylim(-0.05, 1.05)
    a2.set_ylabel("flap open")
    a2.set_xlabel("distance [m]")
    fig.tight_layout()
    return fig


def histogram(
    values: Sequence[float], xlabel: str, title: str, markers: Sequence[tuple[str, float]] = ()
) -> Figure:
    """Distribution of a scalar (Monte Carlo)."""
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.hist(values, bins=30, color=SERIES[0], edgecolor="white", linewidth=1.0)
    for k, (lab, x) in enumerate(markers):
        ax.axvline(x, color=SERIES[(k + 1) % len(SERIES)], lw=1.2, ls="--")
        ax.annotate(
            lab,
            (x, ax.get_ylim()[1] * 0.9),
            xytext=(4, 0),
            textcoords="offset points",
            color=INK,
            fontsize=8,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("samples")
    ax.grid(axis="x", visible=False)
    ax.set_title(title, loc="left")
    fig.tight_layout()
    return fig


def measured_vs_sim(
    s_meas: np.ndarray, v_meas: np.ndarray, sim: LapResult, title: str = "GPS vs simulation"
) -> Figure:
    """Measured (GPS) speed trace against the simulated one."""
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(s_meas, v_meas * 3.6, color=SERIES[0], label="measured (GPS)")
    ax.plot(sim.s, sim.v * 3.6, color=SERIES[1], label="simulated")
    ax.set_xlabel("distance [m]")
    ax.set_ylabel("speed [km/h]")
    ax.legend(loc="lower right")
    ax.set_title(title, loc="left")
    fig.tight_layout()
    return fig


def rank_probability(stats: object, metric: str) -> Figure:
    """P(fastest) and P(same rank as nominal) per candidate (Monte Carlo)."""
    df: pd.DataFrame = stats  # type: ignore[assignment]
    fig, axs = plt.subplots(1, 2, figsize=(10, 0.35 * len(df) + 1.6), sharey=True)
    labels = [f"{r.variant} (#{r.nominal_rank})" for r in df.itertuples()]
    for ax, col, title in (
        (axs[0], "p_fastest", "P(fastest)"),
        (axs[1], "p_same_rank", "P(same rank as nominal)"),
    ):
        ax.barh(labels, df[col], color=SERIES[0], height=0.6)
        for i, v in enumerate(df[col]):
            ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=8, color=INK)
        ax.set_xlim(0, 1.1)
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
    axs[0].invert_yaxis()
    fig.suptitle(f"Monte Carlo ranking robustness — {metric}", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    return fig


def thermal_trace(heat: object) -> Figure:
    """Lumped component temperatures over the endurance, with limits."""
    from .thermal import HeatResult

    h: HeatResult = heat  # type: ignore[assignment]
    fig, (a1, a2) = plt.subplots(
        2, 1, figsize=(10, 5.2), sharex=True, gridspec_kw={"height_ratios": [1.6, 1]}
    )
    tmin = h.trace["t_s"] / 60.0
    for k, name in enumerate(("motor", "inverter", "battery")):
        c = SERIES[k]
        a1.plot(tmin, h.trace[f"{name}_c"], color=c, label=name)
        lim = float(h.table.set_index("component").loc[name, "limit_c"])
        a1.axhline(lim, color=c, lw=0.9, ls="--")
        # time-weighted 30 s mean from the cumulative heat (segments have unequal duration)
        t_end = h.trace["t_s"].to_numpy()
        p_w = h.trace[f"{name}_w"].to_numpy()
        dt = np.diff(np.concatenate([[0.0], t_end]))
        e_cum = np.concatenate([[0.0], np.cumsum(p_w * dt)])
        t_cum = np.concatenate([[0.0], t_end])
        win = np.minimum(30.0, t_end)
        e_back = np.interp(t_end - win, t_cum, e_cum)
        a2.plot(tmin, (e_cum[1:] - e_back) / np.maximum(win, 1e-9) / 1e3, color=c, lw=1.2)
    a1.set_ylabel("temperature [°C]")
    a1.legend(loc="upper left", ncol=3)
    a1.set_title("Endurance heat (lumped model, dashed = limits)", loc="left")
    a2.set_ylabel("loss [kW] (30 s mean)")
    a2.set_xlabel("time [min]")
    fig.tight_layout()
    return fig


def motor_efficiency_map(loss_model: object, lap: LapResult, vehicle: Vehicle) -> Figure:
    """Motor efficiency contours (loss model) with the lap's operating points."""
    from .thermal import LossModel

    lm: LossModel = loss_model  # type: ignore[assignment]
    pt = vehicle.powertrain
    rpm = np.linspace(100, pt.motor_max_rpm, 120)
    tq = np.linspace(2, pt.motor_max_torque_nm, 100)
    rr, tt = np.meshgrid(rpm, tq)
    eff = lm.motor_efficiency(tt, rr)
    omega = rr * 2 * math.pi / 60
    eff = np.where(tt <= np.minimum(pt.motor_max_torque_nm, pt.max_power_w / omega), eff, np.nan)
    fig, ax = plt.subplots(figsize=(8, 5))
    cf = ax.contourf(
        rr, tt, eff * 100, levels=[60, 70, 80, 85, 88, 90, 92, 93, 94, 95, 96], cmap=SEQ_BLUE
    )
    cb = fig.colorbar(cf, ax=ax)
    cb.set_label("motor efficiency [%]")
    e = lap.energy
    drive = e.motor_torque_nm > 0
    ax.scatter(
        e.motor_rpm[drive],
        e.motor_torque_nm[drive],
        s=6,
        color=SERIES[1],
        alpha=0.5,
        lw=0,
        label="operating points (drive)",
    )
    ax.set_xlabel("motor speed [rpm]")
    ax.set_ylabel("motor torque [N·m]")
    ax.legend(loc="upper right")
    ax.grid(False)
    ax.set_title("Motor efficiency map (assumed loss model)", loc="left")
    return fig
