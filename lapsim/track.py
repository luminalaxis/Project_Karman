"""Track loading: JSON in several common layouts -> uniformly sampled curvature profile.

Supported JSON layouts (auto-detected):

1. Centerline points      ``{"x": [...], "y": [...]}`` or ``{"centerline": [[x, y], ...]}``
                          or a cone2track export ``{"centerline": {"closed":, "points": [{x, y}]}}``
2. Cones                  ``{"left": [[x, y], ...], "right": [[x, y], ...]}``
                          (``blue``/``yellow`` or ``inner``/``outer`` also accepted), or
                          ``{"cones": [{"side": "left", "lat":, "lng":}, ...]}``
3. Segments               ``{"segments": [{"type": "straight", "length": 50},
                                          {"type": "corner", "radius": 9, "angle": 180,
                                           "direction": "left"}, ...]}``
4. Distance / curvature   ``{"s": [...], "curvature": [...]}``

Optional keys: ``"closed"`` (bool, default True for point/cone layouts, False for
segments and curvature), ``"name"``, driving direction ``"direction"`` /
``"meta": {"run": "clockwise"}`` or ``"reverse": true``.  Points may be ``[x, y]`` pairs or ``{"x":, "y":}`` dicts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MIN_RADIUS = 2.5  # [m] curvature clip; FS hairpins have >= 4.5 m outside radius


@dataclass
class Track:
    name: str
    s: np.ndarray       # distance along the path [m], uniform spacing, s[0] = 0
    kappa: np.ndarray   # signed curvature [1/m] (+ = left turn)
    x: np.ndarray
    y: np.ndarray
    closed: bool

    @property
    def ds(self) -> float:
        return float(self.s[1] - self.s[0])

    @property
    def length(self) -> float:
        """Driven length; for a closed track this includes the segment back to the start."""
        return float(self.s[-1] + (self.ds if self.closed else 0.0))


# ---------------------------------------------------------------------------
def _points(raw) -> np.ndarray:
    pts = [(p["x"], p["y"]) if isinstance(p, dict) else (p[0], p[1]) for p in raw]
    return np.asarray(pts, dtype=float)


def _side_cones(data):
    """Split a ``"cones": [{"side": "left", "x"/"y" or "lat"/"lng"}, ...]`` list."""
    cones = data["cones"]
    if all("x" in c for c in cones):
        xy = np.array([[c["x"], c["y"]] for c in cones], dtype=float)
    else:                                   # local equirectangular projection
        proj = data.get("projection", {})
        lat = np.array([c["lat"] for c in cones], dtype=float)
        lng = np.array([c["lng"] for c in cones], dtype=float)
        lat0, lng0 = proj.get("lat0", lat.mean()), proj.get("lng0", lng.mean())
        mlat = proj.get("mlat", 110_540.0)
        mlng = proj.get("mlng", 111_320.0 * np.cos(np.radians(lat0)))
        xy = np.column_stack([(lng - lng0) * mlng, (lat - lat0) * mlat])
    side = np.array([str(c.get("side", "")).lower() for c in cones])
    return xy[side == "left"], xy[side == "right"]


def _orient(pts, data, closed):
    """Honour the driving direction of a closed loop.

    ``"direction"`` / ``"meta": {"run": ...}`` = "clockwise" | "counterclockwise", or an
    explicit ``"reverse": true``.  Coordinates are x east, y north.
    """
    if data.get("reverse") is True:
        return pts[::-1]
    run = str(data.get("direction") or data.get("meta", {}).get("run", "")).lower()
    if not closed or run not in ("clockwise", "cw", "counterclockwise", "anticlockwise", "ccw"):
        return pts
    x, y = pts[:, 0], pts[:, 1]
    ccw = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) > 0
    want_ccw = run not in ("clockwise", "cw")
    return pts if ccw == want_ccw else pts[::-1]


def _cones_to_centerline(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Midpoints between each cone of the denser side and its nearest cone on the other side."""
    a, b = (left, right) if len(left) >= len(right) else (right, left)
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    return 0.5 * (a + b[np.argmin(d, axis=1)])


def _segments_to_curvature(segments, ds):
    k_list = []
    for seg in segments:
        kind = seg.get("type", "straight").lower()
        if kind == "straight":
            length, k = float(seg["length"]), 0.0
        elif kind in ("corner", "turn", "arc"):
            r = float(seg["radius"])
            length = r * np.radians(float(seg["angle"]))
            sign = -1.0 if str(seg.get("direction", "left")).lower() in ("right", "r", "cw") else 1.0
            k = sign / r
        else:
            raise ValueError(f"unknown segment type: {kind!r}")
        n = max(int(round(length / ds)), 1)
        k_list.append(np.full(n, k))
    kappa = np.concatenate(k_list)
    return np.arange(len(kappa)) * ds, kappa


def _integrate_xy(s, kappa):
    ds = s[1] - s[0] if len(s) > 1 else 0.0
    psi = np.concatenate([[0.0], np.cumsum(kappa[:-1] * ds)])
    x = np.concatenate([[0.0], np.cumsum(np.cos(psi[:-1]) * ds)])
    y = np.concatenate([[0.0], np.cumsum(np.sin(psi[:-1]) * ds)])
    return x, y


def _smooth(v, window, closed):
    if window <= 1:
        return v
    kernel = np.ones(window) / window
    pad = window // 2
    vp = np.pad(v, pad, mode="wrap" if closed else "edge")
    return np.convolve(vp, kernel, mode="same")[pad:pad + len(v)]


def _resample_xy(pts, ds, closed):
    if closed and np.linalg.norm(pts[0] - pts[-1]) > 1e-6:
        pts = np.vstack([pts, pts[:1]])
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    keep = np.concatenate([[True], seg > 1e-9])          # drop duplicate points
    pts = pts[keep]
    s_raw = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    total = s_raw[-1]
    n = int(round(total / ds))
    ds = total / n                                        # exact spacing that closes the loop
    s = np.arange(n + (0 if closed else 1)) * ds
    return s, np.interp(s, s_raw, pts[:, 0]), np.interp(s, s_raw, pts[:, 1])


def _curvature_xy(x, y, ds, closed):
    if closed:
        dx = (np.roll(x, -1) - np.roll(x, 1)) / (2 * ds)
        dy = (np.roll(y, -1) - np.roll(y, 1)) / (2 * ds)
        ddx = (np.roll(x, -1) - 2 * x + np.roll(x, 1)) / ds**2
        ddy = (np.roll(y, -1) - 2 * y + np.roll(y, 1)) / ds**2
    else:
        dx, dy = np.gradient(x, ds), np.gradient(y, ds)
        ddx, ddy = np.gradient(dx, ds), np.gradient(dy, ds)
    return (dx * ddy - dy * ddx) / np.maximum(dx**2 + dy**2, 1e-12) ** 1.5


# ---------------------------------------------------------------------------
def track_from_dict(data: dict, ds: float = 0.5, smooth_m: float = 4.0, name: str = "track") -> Track:
    name = data.get("name", name)
    keys = set(data)
    pairs = [("left", "right"), ("blue", "yellow"), ("inner", "outer")]

    if "segments" in keys or ("s" in keys and ("curvature" in keys or "kappa" in keys)):
        closed = bool(data.get("closed", False))
        if "segments" in keys:
            s, kappa = _segments_to_curvature(data["segments"], ds)
        else:
            s_raw = np.asarray(data["s"], dtype=float)
            k_raw = np.asarray(data.get("curvature", data.get("kappa")), dtype=float)
            s = np.arange(0.0, s_raw[-1] - s_raw[0] + 1e-9, ds)
            kappa = np.interp(s, s_raw - s_raw[0], k_raw)
        kappa = np.clip(kappa, -1 / MIN_RADIUS, 1 / MIN_RADIUS)
        x, y = _integrate_xy(s, kappa)
        return Track(name, s, kappa, x, y, closed)

    closed = data.get("closed")
    if "x" in keys and "y" in keys:
        pts = np.column_stack([np.asarray(data["x"], float), np.asarray(data["y"], float)])
    elif "centerline" in keys and isinstance(data["centerline"], dict):   # cone2track export
        cl = data["centerline"]
        pts = _points(cl["points"])
        closed = cl.get("closed", closed)
    elif "centerline" in keys:
        pts = _points(data["centerline"])
    elif "cones" in keys:                                               # raw surveyed cones
        pts = _cones_to_centerline(*_side_cones(data))
    else:
        for l, r in pairs:
            if l in keys and r in keys:
                pts = _cones_to_centerline(_points(data[l]), _points(data[r]))
                break
        else:
            raise ValueError(
                "unrecognised track JSON; expected one of x/y, centerline, left/right cones, "
                f"segments or s/curvature. Found keys: {sorted(keys)}")

    closed = True if closed is None else bool(closed)
    pts = _orient(pts, data, closed)
    s, x, y = _resample_xy(pts, ds, closed)
    ds = s[1] - s[0]
    win = max(int(round(smooth_m / ds)), 1) | 1          # odd window
    x, y = _smooth(x, win, closed), _smooth(y, win, closed)
    kappa = _smooth(_curvature_xy(x, y, ds, closed), win, closed)
    kappa = np.clip(kappa, -1 / MIN_RADIUS, 1 / MIN_RADIUS)
    return Track(name, s, kappa, x, y, closed)


def load_track(path, ds: float = 0.5, smooth_m: float = 4.0) -> Track:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):                    # bare list of points
        data = {"centerline": data}
    return track_from_dict(data, ds=ds, smooth_m=smooth_m, name=path.stem)
