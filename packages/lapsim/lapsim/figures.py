"""Interactive plotly figures for the GUI.  ``runs`` = {label: (vehicle, LapResult, channels)}."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .channels import delta_time

COLORS = ["#2a6fdb", "#e8590c", "#2b8a3e", "#ae3ec9", "#495057"]


def _layout(fig, title, height=380, **kw):
    fig.update_layout(title=title, height=height, margin=dict(l=50, r=20, t=50, b=80),
                      legend=dict(orientation="h", y=-0.3, yanchor="top"), hovermode="x unified", **kw)
    return fig


def speed(runs: dict):
    fig = go.Figure()
    for i, (k, (_, res, ch)) in enumerate(runs.items()):
        fig.add_scatter(x=res.s, y=ch["speed_kmh"], name=f"{k} ({res.lap_time:.3f} s)",
                        line=dict(color=COLORS[i % 5], width=2))
        if i == 0:
            lim = np.minimum(ch["corner_limit_kmh"], ch["speed_kmh"].max() * 1.15)
            fig.add_scatter(x=res.s, y=lim, name="corner limit",
                            line=dict(color="gray", width=1, dash="dot"))
    fig.update_xaxes(title="distance [m]")
    fig.update_yaxes(title="speed [km/h]")
    return _layout(fig, "Speed trace")


def time_delta(runs: dict):
    labels = list(runs)
    ref = runs[labels[0]][1]
    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color="gray", width=1))
    for i, k in enumerate(labels[1:], start=1):
        s, dt = delta_time(ref, runs[k][1])
        fig.add_scatter(x=s, y=dt, name=f"{k} − {labels[0]}", line=dict(color=COLORS[i % 5], width=2))
    fig.update_xaxes(title=f"distance on {labels[0]} [m]")
    fig.update_yaxes(title="Δt [s] (+ = slower)")
    return _layout(fig, "Running time delta", height=300)


def track_map(runs: dict, limits=None, channel: str = "speed_kmh", unit: str = "km/h"):
    fig = go.Figure()
    if limits is not None and limits.edges() is not None:
        (xl, yl), (xr, yr) = limits.edges()
        for x, y in ((xl, yl), (xr, yr)):
            fig.add_scatter(x=np.append(x, x[0]), y=np.append(y, y[0]), mode="lines",
                            line=dict(color="#adb5bd", width=1), showlegend=False, hoverinfo="skip")
    if limits is not None and limits.slalom_cones is not None:
        fig.add_scatter(x=limits.slalom_cones[:, 0], y=limits.slalom_cones[:, 1], mode="markers",
                        name="slalom cones", marker=dict(size=9, color="#2b8a3e", symbol="triangle-up"))
    labels = list(runs)
    for i, k in enumerate(labels[1:], start=1):
        res = runs[k][1]
        fig.add_scatter(x=res.x, y=res.y, mode="lines", name=k,
                        line=dict(color=COLORS[i % 5], width=1.5))
    _, res0, ch0 = runs[labels[0]]
    fig.add_scatter(x=res0.x, y=res0.y, mode="markers", name=labels[0],
                    marker=dict(size=4, color=ch0[channel], colorscale="Viridis",
                                colorbar=dict(title=unit)),
                    text=[f"s={s:.0f} m" for s in res0.s],
                    hovertemplate="%{text}<br>%{marker.color:.1f} " + unit + "<extra></extra>")
    fig.add_scatter(x=[res0.x[0]], y=[res0.y[0]], mode="markers", name="start",
                    marker=dict(symbol="triangle-right", size=12, color="red"))
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig = _layout(fig, f"Track map — {channel}", height=520)
    fig.update_layout(hovermode="closest")
    return fig


def gg(runs: dict):
    fig = go.Figure()
    for i, (k, (_, _, ch)) in enumerate(runs.items()):
        fig.add_scatter(x=ch["ay_g"], y=ch["ax_g"], mode="markers", name=k,
                        marker=dict(size=3, color=COLORS[i % 5], opacity=0.6))
    fig.update_xaxes(title="a_y [g]")
    fig.update_yaxes(title="a_x [g]", scaleanchor="x", scaleratio=1)
    fig = _layout(fig, "g-g diagram", height=420)
    fig.update_layout(hovermode="closest")
    return fig


def lines(runs: dict, keys, title: str, ytitle: str, height=320):
    """Several channels vs distance.  One run: a colour per channel.  Several runs: a colour
    per run and a dash style per channel."""
    dashes = ["solid", "dash", "dot", "dashdot"]
    many = len(runs) > 1
    fig = go.Figure()
    for i, (k, (_, res, ch)) in enumerate(runs.items()):
        for j, key in enumerate(keys):
            line = (dict(color=COLORS[i % 5], dash=dashes[j % 4], width=1.5) if many
                    else dict(color=COLORS[j % 5], width=1.5))
            fig.add_scatter(x=res.s, y=ch[key], name=f"{k}: {key}" if many else key, line=line)
    fig.update_xaxes(title="distance [m]")
    fig.update_yaxes(title=ytitle)
    return _layout(fig, title, height=height)


def aero_devices(veh, res, ch):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("downforce [N]", "drag [N]"))
    for j, name in enumerate(veh.aero.devices):
        c = COLORS[j % 5]
        fig.add_scatter(x=res.s, y=ch[f"downforce_{name}_N"], name=name, stackgroup="df",
                        line=dict(color=c, width=0.5), row=1, col=1)
        fig.add_scatter(x=res.s, y=ch[f"drag_{name}_N"], name=name, stackgroup="dr", showlegend=False,
                        line=dict(color=c, width=0.5), row=1, col=2)
    fig.update_xaxes(title="distance [m]")
    return _layout(fig, "Aero load per device (stacked)", height=360)


def utilisation_hist(runs: dict):
    fig = go.Figure()
    for i, (k, (_, _, ch)) in enumerate(runs.items()):
        for key, dash in (("util_front", "solid"), ("util_rear", "dash")):
            hist, edges = np.histogram(np.clip(ch[key], 0, 1.05), bins=42, range=(0, 1.05))
            fig.add_scatter(x=0.5 * (edges[1:] + edges[:-1]), y=hist, name=f"{k}: {key}",
                            line=dict(color=COLORS[i % 5], dash=dash, shape="hvh"))
    fig.update_xaxes(title="friction utilisation (1 = at the limit)")
    fig.update_yaxes(title="points")
    return _layout(fig, "Axle friction utilisation", height=320)


def sweep(d_cl, d_cd, grid):
    z = grid * 1e3
    lim = float(np.abs(z).max()) or 1.0
    fig = go.Figure(go.Contour(x=np.asarray(d_cl) * 100, y=np.asarray(d_cd) * 100, z=z,
                               colorscale="RdBu_r", zmin=-lim, zmax=lim,
                               contours=dict(showlabels=True), colorbar=dict(title="ms")))
    fig.update_xaxes(title="ΔClA [%]")
    fig.update_yaxes(title="ΔCdA [%]")
    fig = _layout(fig, "Δ lap time [ms] (blue = faster)", height=480)
    fig.update_layout(hovermode="closest")
    return fig


def balance(bals, times):
    fig = go.Figure(go.Scatter(x=np.asarray(bals) * 100, y=times, mode="lines+markers"))
    fig.update_xaxes(title="aero balance front [%]")
    fig.update_yaxes(title="lap time [s]")
    return _layout(fig, "Aero balance sweep", height=360)
