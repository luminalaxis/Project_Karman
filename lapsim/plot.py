"""Matplotlib reports (``pip install matplotlib``)."""

from __future__ import annotations

import numpy as np

from .channels import channels, delta_time

COLORS = ["#2a6fdb", "#e8590c", "#2b8a3e", "#ae3ec9", "#495057"]


def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover
        raise SystemExit("plotting needs matplotlib: pip install matplotlib") from e
    return plt


def plot_lap(runs: dict, path, track=None):
    """Lap report.  ``runs`` = {label: (vehicle, LapResult)}; the first run is the reference.

    Panels: speed trace (+ corner limit), running time delta, longitudinal / lateral g,
    g-g diagram, track map coloured by speed (with track limits and every run's path),
    ride heights and aero balance, axle friction utilisation.
    """
    plt = _plt()
    labels = list(runs)
    chs = {k: channels(*runs[k]) for k in labels}
    ref_res = runs[labels[0]][1]

    fig = plt.figure(figsize=(16, 13))
    gs = fig.add_gridspec(4, 3, height_ratios=[1.3, 0.8, 1, 1])
    ax_v = fig.add_subplot(gs[0, :2])
    ax_map = fig.add_subplot(gs[0:2, 2])
    ax_dt = fig.add_subplot(gs[1, :2], sharex=ax_v)
    ax_g = fig.add_subplot(gs[2, :2], sharex=ax_v)
    ax_gg = fig.add_subplot(gs[2, 2])
    ax_rh = fig.add_subplot(gs[3, :2], sharex=ax_v)
    ax_u = fig.add_subplot(gs[3, 2])

    for i, k in enumerate(labels):
        veh, res = runs[k]
        ch, c = chs[k], COLORS[i % len(COLORS)]
        ax_v.plot(res.s, ch["speed_kmh"], color=c, lw=1.4, label=f"{k}  {res.lap_time:.3f} s")
        if i == 0:
            lim = np.minimum(ch["corner_limit_kmh"], ch["speed_kmh"].max() * 1.15)
            ax_v.plot(res.s, lim, ":", color="0.4", lw=0.8, label="corner limit")
        else:
            s, dt = delta_time(ref_res, res)
            ax_dt.plot(s, dt, color=c, lw=1.4, label=f"{k} − {labels[0]}")
        ax_g.plot(res.s, ch["ax_g"], color=c, lw=1.0)
        ax_g.plot(res.s, ch["ay_g"], color=c, lw=1.0, ls="--")
        ax_gg.plot(ch["ay_g"], ch["ax_g"], ".", ms=1.5, color=c)
        ax_rh.plot(res.s, ch["ride_height_front_mm"], color=c, lw=1.2)
        ax_rh.plot(res.s, ch["ride_height_rear_mm"], color=c, lw=1.2, ls="--")
        ax_u.hist([ch["util_front"], ch["util_rear"]], bins=30, range=(0, 1.05), histtype="step",
                  color=[c, c], linestyle=["-", "--"])
        ax_map.plot(res.x, res.y, color=c, lw=0.8 if i else 0.0)

    res0, ch0 = ref_res, chs[labels[0]]
    if track is not None and track.edges() is not None:
        (xl, yl), (xr, yr) = track.edges()
        ax_map.plot(xl, yl, color="0.6", lw=0.7)
        ax_map.plot(xr, yr, color="0.6", lw=0.7)
    sc = ax_map.scatter(res0.x, res0.y, c=ch0["speed_kmh"], s=4, cmap="viridis", zorder=3)
    ax_map.plot(res0.x[0], res0.y[0], "r>", ms=8, zorder=4)
    ax_map.set_aspect("equal")
    ax_map.set_title(f"track map ({labels[0]} speed)")
    fig.colorbar(sc, ax=ax_map, label="km/h", shrink=0.7)

    ax_v.set(ylabel="speed [km/h]", title="speed trace")
    ax_v.legend(loc="lower right", fontsize=8)
    ax_dt.axhline(0, color="0.5", lw=0.6)
    ax_dt.set(ylabel="Δt [s]", title="running time delta (+ = slower than reference)")
    if len(labels) > 1:
        ax_dt.legend(fontsize=8)
    ax_g.set(ylabel="g", title="longitudinal (solid) / lateral (dashed) acceleration")
    ax_gg.set(xlabel="a_y [g]", ylabel="a_x [g]", title="g-g diagram")
    ax_gg.set_aspect("equal")
    ax_rh.set(xlabel="distance [m]", ylabel="mm", title="ride height front (solid) / rear (dashed)")
    ax_u.set(xlabel="axle friction utilisation", ylabel="points",
             title="front (solid) / rear (dashed) utilisation")
    for a in (ax_v, ax_dt, ax_g, ax_gg, ax_rh, ax_u):
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_sweep(d_cl, d_cd, grid, path):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 5.5))
    lim = np.abs(grid).max() * 1e3 or 1.0
    im = ax.pcolormesh(np.asarray(d_cl) * 100, np.asarray(d_cd) * 100, grid * 1e3,
                       cmap="RdBu_r", vmin=-lim, vmax=lim, shading="auto")
    cs = ax.contour(np.asarray(d_cl) * 100, np.asarray(d_cd) * 100, grid * 1e3,
                    colors="k", linewidths=0.6)
    ax.clabel(cs, fmt="%.0f ms", fontsize=7)
    ax.set(xlabel="ΔClA [%]", ylabel="ΔCdA [%]", title="Δ lap time [ms] (blue = faster)")
    fig.colorbar(im, ax=ax, label="ms")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_balance(balances, times, path):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.asarray(balances) * 100, times, "o-")
    ax.set(xlabel="aero balance front [%]", ylabel="lap time [s]", title="aero balance sweep")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
