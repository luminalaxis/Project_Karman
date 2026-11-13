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

Track width (for the racing line): per-point ``widthL``/``widthR`` (cone2track), ``width_left`` /
``width_right`` lists for the x/y layout, distances to the cones, or a scalar ``"width"``.

Slalom cones (driven through in an S): cone2track cones with ``"side": "center"`` or
``"slalom": [[x, y], ...]``.

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
    width_left: np.ndarray | None = None   # usable width left of the path [m]
    width_right: np.ndarray | None = None  # usable width right of the path [m]
    slalom_cones: np.ndarray | None = None  # (k, 2) cones to weave through (cone2track "center")

    @property
    def ds(self) -> float:
        return float(self.s[1] - self.s[0])

    @property
    def length(self) -> float:
        """Driven length; for a closed track this includes the segment back to the start."""
        return float(self.s[-1] + (self.ds if self.closed else 0.0))

    def normals(self):
        """Left-pointing unit normals of the path."""
        if self.closed:
            tx, ty = np.roll(self.x, -1) - np.roll(self.x, 1), np.roll(self.y, -1) - np.roll(self.y, 1)
        else:
            tx, ty = np.gradient(self.x), np.gradient(self.y)
        n = np.hypot(tx, ty)
        return -ty / n, tx / n

    def edges(self):
        """((x_left, y_left), (x_right, y_right)) track limits, or None without width data."""
        if self.width_left is None or self.width_right is None:
            return None
        nx, ny = self.normals()
        return ((self.x + nx * self.width_left, self.y + ny * self.width_left),
                (self.x - nx * self.width_right, self.y - ny * self.width_right))


# ---------------------------------------------------------------------------
def _points(raw) -> np.ndarray:
    pts = [(p["x"], p["y"]) if isinstance(p, dict) else (p[0], p[1]) for p in raw]
    return np.asarray(pts, dtype=float)


def _side_cones(data):
    """Split a ``"cones": [{"side": "left", "x"/"y" or "lat"/"lng"}, ...]`` list into
    (left, right, center); "center" cones are slalom cones the car weaves through."""
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
    return xy[side == "left"], xy[side == "right"], xy[side == "center"]


def _needs_reverse(pts, data, closed) -> bool:
    """Whether the point order must be reversed to honour the driving direction.

    ``"direction"`` / ``"meta": {"run": ...}`` = "clockwise" | "counterclockwise", or an
    explicit ``"reverse": true``.  Coordinates are x east, y north.
    """
    if data.get("reverse") is True:
        return True
    run = str(data.get("direction") or data.get("meta", {}).get("run", "")).lower()
    if not closed or run not in ("clockwise", "cw", "counterclockwise", "anticlockwise", "ccw"):
        return False
    x, y = pts[:, 0], pts[:, 1]
    ccw = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) > 0
    want_ccw = run not in ("clockwise", "cw")
    return ccw != want_ccw


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


def smooth(v, window, closed):
    if window <= 1:
        return v
    kernel = np.ones(window) / window
    pad = window // 2
    vp = np.pad(v, pad, mode="wrap" if closed else "edge")
    return np.convolve(vp, kernel, mode="same")[pad:pad + len(v)]


def _resample(cols, ds, closed):
    """Resample columns [x, y, ...] uniformly in arc length (x, y define the arc length)."""
    if closed and np.linalg.norm(cols[0, :2] - cols[-1, :2]) > 1e-6:
        cols = np.vstack([cols, cols[:1]])
    seg = np.linalg.norm(np.diff(cols[:, :2], axis=0), axis=1)
    cols = cols[np.concatenate([[True], seg > 1e-9])]     # drop duplicate points
    s_raw = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(cols[:, :2], axis=0), axis=1))])
    total = s_raw[-1]
    n = int(round(total / ds))
    ds = total / n                                        # exact spacing that closes the loop
    s = np.arange(n + (0 if closed else 1)) * ds
    return s, [np.interp(s, s_raw, cols[:, j]) for j in range(cols.shape[1])]


def _point_widths(raw):
    """(left, right) widths from cone2track-style point dicts, else None."""
    if raw and isinstance(raw[0], dict) and "widthL" in raw[0] and "widthR" in raw[0]:
        return (np.array([p["widthL"] for p in raw], float), np.array([p["widthR"] for p in raw], float))
    return None


def _cone_widths(x, y, left, right):
    """Distance from each path point to the nearest left / right cone."""
    p = np.column_stack([x, y])
    wl = np.min(np.linalg.norm(p[:, None, :] - left[None, :, :], axis=2), axis=1)
    wr = np.min(np.linalg.norm(p[:, None, :] - right[None, :, :], axis=2), axis=1)
    return wl, wr


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
def track_from_dict(data: dict, ds: float = 0.5, smooth_m: float = 4.0, name: str = "track",
                    kappa_smooth_m: float | None = None) -> Track:
    """``smooth_m``: moving-average window for the path points and, unless ``kappa_smooth_m``
    is given, for the curvature as well."""
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
        w = np.full(len(s), float(data["width"]) / 2) if "width" in keys else None
        return Track(name, s, kappa, x, y, closed, w, w)

    closed = data.get("closed")
    widths, cones = None, None
    if "x" in keys and "y" in keys:
        pts = np.column_stack([np.asarray(data["x"], float), np.asarray(data["y"], float)])
        if "width_left" in keys and "width_right" in keys:
            widths = (np.asarray(data["width_left"], float), np.asarray(data["width_right"], float))
    elif "centerline" in keys and isinstance(data["centerline"], dict):   # cone2track export
        cl = data["centerline"]
        pts = _points(cl["points"])
        widths = _point_widths(cl["points"])
        closed = cl.get("closed", closed)
    elif "centerline" in keys:
        pts = _points(data["centerline"])
        widths = _point_widths(data["centerline"])
    elif "cones" in keys:                                               # raw surveyed cones
        cones = _side_cones(data)[:2]
        pts = _cones_to_centerline(*cones)
    else:
        for l, r in pairs:
            if l in keys and r in keys:
                cones = (_points(data[l]), _points(data[r]))
                pts = _cones_to_centerline(*cones)
                break
        else:
            raise ValueError(
                "unrecognised track JSON; expected one of x/y, centerline, left/right cones, "
                f"segments or s/curvature. Found keys: {sorted(keys)}")

    closed = True if closed is None else bool(closed)
    cols = pts if widths is None else np.column_stack([pts, widths[0], widths[1]])
    if _needs_reverse(pts, data, closed):
        cols = cols[::-1]
        if widths is not None:                       # left/right swap with the direction
            cols = cols[:, [0, 1, 3, 2]]
        if cones is not None:
            cones = (cones[1], cones[0])
    s, res = _resample(cols, ds, closed)
    x, y = res[0], res[1]
    ds = s[1] - s[0]
    win = max(int(round(smooth_m / ds)), 1) | 1          # odd window
    kwin = win if kappa_smooth_m is None else max(int(round(kappa_smooth_m / ds)), 1) | 1
    x, y = smooth(x, win, closed), smooth(y, win, closed)
    kappa = smooth(_curvature_xy(x, y, ds, closed), kwin, closed)
    kappa = np.clip(kappa, -1 / MIN_RADIUS, 1 / MIN_RADIUS)
    if widths is not None:
        wl, wr = res[2], res[3]
    elif cones is not None:
        wl, wr = _cone_widths(x, y, *cones)
    elif "width" in keys:
        wl = wr = np.full(len(s), float(data["width"]) / 2)
    else:
        wl = wr = None
    return Track(name, s, kappa, x, y, closed, wl, wr, _slalom_cones(data))


def _slalom_cones(data):
    """Slalom cones: cone2track ``"cones"`` with side "center", or ``"slalom": [[x, y], ...]``."""
    if "slalom" in data:
        pts = _points(data["slalom"])
    elif isinstance(data.get("cones"), list) and data["cones"] and isinstance(data["cones"][0], dict):
        pts = _side_cones(data)[2]
    else:
        return None
    return pts if len(pts) else None


def load_track(path, ds: float = 0.5, smooth_m: float = 4.0) -> Track:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):                    # bare list of points
        data = {"centerline": data}
    return track_from_dict(data, ds=ds, smooth_m=smooth_m, name=path.stem)
