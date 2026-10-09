"""Optional matplotlib plots (``pip install matplotlib``)."""

from __future__ import annotations

import numpy as np


def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover
        raise SystemExit("plotting needs matplotlib: pip install matplotlib") from e
    return plt


def plot_lap(results: dict, track, path):
    """Speed trace (one line per labelled LapResult) and the track map coloured by speed."""
    plt = _plt()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [2, 1]})
    for label, r in results.items():
        ax1.plot(r.s, r.v * 3.6, label=f"{label} ({r.lap_time:.3f} s)", lw=1.4)
    first = next(iter(results.values()))
    ax1.plot(first.s, np.minimum(first.v_corner, first.v.max() * 1.2) * 3.6, "k:", lw=0.8,
             label="corner limit")
    ax1.set(xlabel="distance [m]", ylabel="speed [km/h]", title="speed trace")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="lower right", fontsize=8)
    n = len(track.x)
    sc = ax2.scatter(track.x, track.y, c=first.v[:n] * 3.6, s=4, cmap="viridis")
    ax2.plot(track.x[0], track.y[0], "r>", ms=8)
    ax2.set_aspect("equal")
    ax2.set_title(track.name)
    fig.colorbar(sc, ax=ax2, label="km/h")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
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
