"""Track representation: curvature κ(s) along distance s, plus (x, y) for plotting.

Two concrete kinds:

* :class:`SegmentTrack` — straights, constant-radius arcs and sinusoidal slaloms with exact
  analytical curvature (used by the generator, the templates, acceleration and skidpad).
* :class:`SampledTrack` — curvature sampled on a grid (CSV/GPS import via smoothing spline).

Sign convention: κ > 0 turns left (counter-clockwise), heading θ is measured from +x.
"""

from __future__ import annotations

import abc
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import interpolate, optimize, special

Pose = tuple[float, float, float]
"""(x [m], y [m], heading θ [rad])."""

#: Largest slalom heading amplitude accepted [rad] (86°); H0 is monotonic below it.
_THETA_AMP_MAX = 1.5


# --------------------------------------------------------------------------------------------
# Segments
# --------------------------------------------------------------------------------------------
class Segment(abc.ABC):
    """A piece of track with analytical curvature in local arc length ``0 ≤ s ≤ length``."""

    length_m: float

    @abc.abstractmethod
    def curvature(self, s: NDArray[np.float64]) -> NDArray[np.float64]:
        """Curvature κ(s) [1/m] at local distances ``s``."""

    @abc.abstractmethod
    def heading(self, s: NDArray[np.float64]) -> NDArray[np.float64]:
        """Heading change θ(s) − θ(0) [rad] at local distances ``s``."""

    @abc.abstractmethod
    def local_xy(self, s: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Position in the segment frame (start at origin, heading +x) at local distances ``s``."""

    @abc.abstractmethod
    def mirrored(self) -> Segment:
        """Left/right mirror image."""

    def end_pose(self, start: Pose) -> Pose:
        """Exact pose at the end of the segment for a given start pose."""
        s_end = np.array([self.length_m])
        lx, ly = self.local_xy(s_end)
        x0, y0, th0 = start
        c, s = math.cos(th0), math.sin(th0)
        dth = float(self.heading(s_end)[0])
        return (x0 + c * lx[0] - s * ly[0], y0 + s * lx[0] + c * ly[0], th0 + dth)


@dataclass(frozen=True)
class Straight(Segment):
    """Straight of given length [m]."""

    length_m: float

    def curvature(self, s: NDArray[np.float64]) -> NDArray[np.float64]:  # noqa: D102
        return np.zeros_like(np.asarray(s, dtype=float))

    def heading(self, s: NDArray[np.float64]) -> NDArray[np.float64]:  # noqa: D102
        return np.zeros_like(np.asarray(s, dtype=float))

    def local_xy(self, s: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:  # noqa: D102
        s = np.asarray(s, dtype=float)
        return s.copy(), np.zeros_like(s)

    def mirrored(self) -> Straight:  # noqa: D102
        return self


@dataclass(frozen=True)
class Arc(Segment):
    """Constant-radius turn.

    Attributes:
        radius_m: Radius of the driven line [m].
        angle_rad: Swept angle (> 0) [rad].
        direction: +1 = left (counter-clockwise), −1 = right.
    """

    radius_m: float
    angle_rad: float
    direction: int = 1
    length_m: float = field(init=False)

    def __post_init__(self) -> None:
        if self.radius_m <= 0 or self.angle_rad <= 0 or self.direction not in (1, -1):
            raise ValueError(f"invalid arc R={self.radius_m}, angle={self.angle_rad}")
        object.__setattr__(self, "length_m", self.radius_m * self.angle_rad)

    def curvature(self, s: NDArray[np.float64]) -> NDArray[np.float64]:  # noqa: D102
        return np.full_like(np.asarray(s, dtype=float), self.direction / self.radius_m)

    def heading(self, s: NDArray[np.float64]) -> NDArray[np.float64]:  # noqa: D102
        return self.direction * np.asarray(s, dtype=float) / self.radius_m

    def local_xy(self, s: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:  # noqa: D102
        phi = np.asarray(s, dtype=float) / self.radius_m
        return self.radius_m * np.sin(phi), self.direction * self.radius_m * (1.0 - np.cos(phi))

    def mirrored(self) -> Arc:  # noqa: D102
        return Arc(self.radius_m, self.angle_rad, -self.direction)


@dataclass(frozen=True)
class Slalom(Segment):
    """Sinusoidal weave through ``n_cones`` cones placed every ``spacing_m``.

    Curvature κ(s) = d·κ_max·cos(2πs/λ) with λ = 2·spacing, so the heading
    θ(s) = d·θ_a·sin(2πs/λ) (θ_a = κ_max·λ/2π) is zero at every cone gap and the segment
    starts and ends straight. The lateral peak-to-peak excursion of the driven line is
    ``weave_m``; θ_a follows exactly from  weave = (λ/2)·H0(θ_a)  (H0: Struve function).

    Attributes:
        n_cones: Number of cones (= number of half wavelengths).
        spacing_m: Cone spacing [m].
        weave_m: Peak-to-peak lateral excursion of the driven line [m].
        direction: +1 = first swerve to the left, −1 = to the right.
    """

    n_cones: int
    spacing_m: float
    weave_m: float = 2.0
    direction: int = 1
    length_m: float = field(init=False)
    theta_amp_rad: float = field(init=False)

    def __post_init__(self) -> None:
        if self.n_cones < 1 or self.spacing_m <= 0 or self.weave_m <= 0:
            raise ValueError(f"invalid slalom n={self.n_cones}, spacing={self.spacing_m}")
        half = self.spacing_m
        if self.weave_m >= half * special.struve(0, _THETA_AMP_MAX):
            raise ValueError("slalom weave too large for the cone spacing")
        th = optimize.brentq(
            lambda a: half * special.struve(0, a) - self.weave_m, 1e-12, _THETA_AMP_MAX
        )
        object.__setattr__(self, "length_m", self.n_cones * self.spacing_m)
        object.__setattr__(self, "theta_amp_rad", float(th))

    @property
    def wavelength_m(self) -> float:
        """Wavelength λ = 2 × spacing [m]."""
        return 2.0 * self.spacing_m

    @property
    def kappa_max(self) -> float:
        """Peak curvature [1/m]."""
        return self.theta_amp_rad * 2.0 * math.pi / self.wavelength_m

    def curvature(self, s: NDArray[np.float64]) -> NDArray[np.float64]:  # noqa: D102
        k = 2.0 * math.pi / self.wavelength_m
        return self.direction * self.kappa_max * np.cos(k * np.asarray(s, dtype=float))

    def heading(self, s: NDArray[np.float64]) -> NDArray[np.float64]:  # noqa: D102
        k = 2.0 * math.pi / self.wavelength_m
        return self.direction * self.theta_amp_rad * np.sin(k * np.asarray(s, dtype=float))

    def local_xy(self, s: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Position by fine trapezoidal integration (exact closed form at whole half-waves)."""
        s = np.atleast_1d(np.asarray(s, dtype=float))
        n = max(int(self.length_m / 0.01), 200)
        sf = np.linspace(0.0, self.length_m, n + 1)
        th = self.heading(sf)
        dsf = np.diff(sf)
        xf = np.concatenate([[0.0], np.cumsum(0.5 * (np.cos(th[1:]) + np.cos(th[:-1])) * dsf)])
        yf = np.concatenate([[0.0], np.cumsum(0.5 * (np.sin(th[1:]) + np.sin(th[:-1])) * dsf)])
        # replace the endpoint by the exact Bessel/Struve result to keep closures exact
        half = self.spacing_m
        xf[-1] = self.n_cones * half * special.j0(self.theta_amp_rad)
        yf[-1] = self.direction * self.weave_m * (self.n_cones % 2)
        return np.interp(s, sf, xf), np.interp(s, sf, yf)

    def mirrored(self) -> Slalom:  # noqa: D102
        return Slalom(self.n_cones, self.spacing_m, self.weave_m, -self.direction)


# --------------------------------------------------------------------------------------------
# Tracks
# --------------------------------------------------------------------------------------------
class Track(abc.ABC):
    """Common track interface."""

    name: str
    closed: bool

    @property
    @abc.abstractmethod
    def length_m(self) -> float:
        """Track length [m]."""

    @abc.abstractmethod
    def curvature(self, s: ArrayLike) -> NDArray[np.float64]:
        """Curvature [1/m] at distances ``s`` (wrapped for closed tracks)."""

    @abc.abstractmethod
    def xy(self, s: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Driven-line position [m] at distances ``s``."""

    @abc.abstractmethod
    def mirrored(self) -> Track:
        """Left/right mirror image (κ → −κ, y → −y)."""

    def discretize(self, ds_m: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Uniform nodes ``s_i = i·L/N`` (N = ceil(L/ds)) and curvature at the nodes.

        For closed tracks the last node coincides with the first and gets the same κ.
        """
        n = max(math.ceil(self.length_m / ds_m - 1e-9), 2)
        s = np.linspace(0.0, self.length_m, n + 1)
        k = self.curvature(s)
        if self.closed:
            k[-1] = k[0]
        return s, k

    def summary(self, ds_m: float = 0.5) -> dict[str, float]:
        """Key geometry numbers for reports."""
        s, k = self.discretize(ds_m)
        ak = np.abs(k)
        straight = ak < 1.0 / 200.0
        longest = 0.0
        run = 0.0
        for flag in straight:
            run = run + ds_m if flag else 0.0
            longest = max(longest, run)
        turning = float(np.sum(np.abs(k[:-1]) * np.diff(s)))
        return {
            "length_m": self.length_m,
            "min_radius_m": float(1.0 / ak.max()) if ak.max() > 0 else float("inf"),
            "longest_straight_m": longest,
            "straight_fraction": float(straight.mean()),
            "total_turning_deg": math.degrees(turning),
        }


class SegmentTrack(Track):
    """Track assembled from analytical segments."""

    def __init__(
        self,
        segments: Sequence[Segment],
        closed: bool = False,
        name: str = "track",
        start_pose: Pose = (0.0, 0.0, 0.0),
    ) -> None:
        if not segments:
            raise ValueError("track needs at least one segment")
        self.segments: tuple[Segment, ...] = tuple(segments)
        self.closed = closed
        self.name = name
        self.start_pose = start_pose
        lengths = np.array([seg.length_m for seg in self.segments])
        self._starts = np.concatenate([[0.0], np.cumsum(lengths)])
        poses = [start_pose]
        for seg in self.segments:
            poses.append(seg.end_pose(poses[-1]))
        self._poses = poses

    @property
    def length_m(self) -> float:  # noqa: D102
        return float(self._starts[-1])

    @property
    def end_pose(self) -> Pose:
        """Pose at the end of the last segment."""
        return self._poses[-1]

    @property
    def segment_poses(self) -> list[Pose]:
        """Start pose of every segment, followed by the end pose (len = n_segments + 1)."""
        return list(self._poses)

    def closure_error(self) -> tuple[float, float]:
        """(position error [m], heading error [rad, wrapped]) between end and start poses."""
        x0, y0, t0 = self.start_pose
        x1, y1, t1 = self.end_pose
        dth = (t1 - t0 + math.pi) % (2.0 * math.pi) - math.pi
        return math.hypot(x1 - x0, y1 - y0), abs(dth)

    def _locate(self, s: ArrayLike) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        s = np.asarray(s, dtype=float)
        if self.closed:
            s = np.mod(s, self.length_m)
        s = np.clip(s, 0.0, self.length_m)
        idx = np.searchsorted(self._starts, s, side="right") - 1
        idx = np.clip(idx, 0, len(self.segments) - 1)
        return idx, s - self._starts[idx]

    def curvature(self, s: ArrayLike) -> NDArray[np.float64]:  # noqa: D102
        idx, sl = self._locate(s)
        out = np.empty_like(sl)
        for i, seg in enumerate(self.segments):
            m = idx == i
            if m.any():
                out[m] = seg.curvature(sl[m])
        return out

    def xy(self, s: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:  # noqa: D102
        idx, sl = self._locate(s)
        x = np.empty_like(sl)
        y = np.empty_like(sl)
        for i, seg in enumerate(self.segments):
            m = idx == i
            if not m.any():
                continue
            lx, ly = seg.local_xy(sl[m])
            x0, y0, th0 = self._poses[i]
            c, sn = math.cos(th0), math.sin(th0)
            x[m] = x0 + c * lx - sn * ly
            y[m] = y0 + sn * lx + c * ly
        return x, y

    def mirrored(self) -> SegmentTrack:  # noqa: D102
        x0, y0, t0 = self.start_pose
        segs = [seg.mirrored() for seg in self.segments]
        return SegmentTrack(segs, self.closed, self.name + "_mirror", (x0, -y0, -t0))


class SampledTrack(Track):
    """Track defined by curvature (and position) sampled on a distance grid.

    Curvature is linearly interpolated; closed tracks wrap around.
    """

    def __init__(
        self,
        s_m: ArrayLike,
        kappa: ArrayLike,
        x_m: ArrayLike | None = None,
        y_m: ArrayLike | None = None,
        closed: bool = False,
        name: str = "track",
    ) -> None:
        s = np.asarray(s_m, dtype=float)
        k = np.asarray(kappa, dtype=float)
        bad = s.ndim != 1 or s.shape != k.shape or s.size < 3
        if bad or s[0] != 0.0 or np.any(np.diff(s) <= 0):
            raise ValueError("s must be increasing from 0 with the same shape as kappa")
        self._s, self._k = s, k
        if x_m is None or y_m is None:
            x_m, y_m = _integrate_xy(s, k)
        self._x, self._y = np.asarray(x_m, dtype=float), np.asarray(y_m, dtype=float)
        self.closed = closed
        self.name = name

    @property
    def length_m(self) -> float:  # noqa: D102
        return float(self._s[-1])

    def _wrap(self, s: ArrayLike) -> NDArray[np.float64]:
        s = np.asarray(s, dtype=float)
        if self.closed:
            s = np.where(s == self.length_m, s, np.mod(s, self.length_m))
        return s

    def curvature(self, s: ArrayLike) -> NDArray[np.float64]:  # noqa: D102
        return np.interp(self._wrap(s), self._s, self._k)

    def xy(self, s: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:  # noqa: D102
        sw = self._wrap(s)
        return np.interp(sw, self._s, self._x), np.interp(sw, self._s, self._y)

    def mirrored(self) -> SampledTrack:  # noqa: D102
        name = self.name + "_mirror"
        return SampledTrack(self._s, -self._k, self._x, -self._y, self.closed, name)


def _integrate_xy(
    s: NDArray[np.float64], k: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    ds = np.diff(s)
    th = np.concatenate([[0.0], np.cumsum(0.5 * (k[1:] + k[:-1]) * ds)])
    thm = 0.5 * (th[1:] + th[:-1])
    x = np.concatenate([[0.0], np.cumsum(np.cos(thm) * ds)])
    y = np.concatenate([[0.0], np.cumsum(np.sin(thm) * ds)])
    return x, y


# --------------------------------------------------------------------------------------------
# Import from (x, y) points
# --------------------------------------------------------------------------------------------
def track_from_xy(
    x: ArrayLike,
    y: ArrayLike,
    closed: bool = True,
    smoothing_sigma_m: float = 0.25,
    ds_m: float = 0.25,
    name: str = "imported",
    reverse: bool = False,
) -> SampledTrack:
    """Fit a smoothing spline through (x, y) points and compute curvature by arc length.

    Args:
        x: Point x coordinates [m].
        y: Point y coordinates [m].
        closed: Treat as a loop (periodic spline).
        smoothing_sigma_m: Expected position noise σ [m]; the spline smoothing factor is
            ``n·σ²`` (scipy ``splprep`` convention). 0 interpolates exactly. Larger values
            straighten the line and raise corner speeds — keep it near the survey accuracy.
        ds_m: Output sampling interval [m].
        name: Track name.
        reverse: Reverse the driving direction.
    """
    px = np.asarray(x, dtype=float)
    py = np.asarray(y, dtype=float)
    if px.shape != py.shape or px.size < 5:
        raise ValueError("need at least 5 (x, y) points of equal length")
    if reverse:
        px, py = px[::-1], py[::-1]
    # remove consecutive duplicates (and the closing duplicate for loops)
    keep = np.concatenate([[True], np.hypot(np.diff(px), np.diff(py)) > 1e-9])
    px, py = px[keep], py[keep]
    if closed and math.hypot(px[-1] - px[0], py[-1] - py[0]) < 1e-6:
        px, py = px[:-1], py[:-1]
    n = px.size
    smooth = n * smoothing_sigma_m**2
    if closed:
        px = np.append(px, px[0])
        py = np.append(py, py[0])
    k_deg = 3 if n > 3 else n - 1
    tck, _ = interpolate.splprep([px, py], s=smooth, per=1 if closed else 0, k=k_deg)
    u_dense = np.linspace(0.0, 1.0, 20 * n + 1)
    dx, dy = interpolate.splev(u_dense, tck, der=1)
    speed = np.hypot(dx, dy)
    arc = np.concatenate([[0.0], np.cumsum(0.5 * (speed[1:] + speed[:-1]) * np.diff(u_dense))])
    length = float(arc[-1])
    m = max(math.ceil(length / ds_m), 4)
    s = np.linspace(0.0, length, m + 1)
    u = np.interp(s, arc, u_dense)
    xs, ys = interpolate.splev(u, tck)
    d1x, d1y = interpolate.splev(u, tck, der=1)
    d2x, d2y = interpolate.splev(u, tck, der=2)
    kappa = (d1x * d2y - d1y * d2x) / np.power(d1x**2 + d1y**2, 1.5)
    if closed:
        kappa[-1] = kappa[0]
    return SampledTrack(s, kappa, xs, ys, closed=closed, name=name)


def read_xy_csv(path: str | Path) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Read ``x``/``y`` columns from a CSV (header row; ``#`` comments, extra columns ok)."""
    import pandas as pd

    df = pd.read_csv(path, comment="#")
    cols = {c.strip().lower(): c for c in df.columns}
    if "x" not in cols or "y" not in cols:
        raise ValueError(f"{path}: CSV needs 'x' and 'y' columns (got {list(df.columns)})")
    return df[cols["x"]].to_numpy(float), df[cols["y"]].to_numpy(float)


# --------------------------------------------------------------------------------------------
# Standard event geometries
# --------------------------------------------------------------------------------------------
def straight_track(length_m: float, name: str = "straight") -> SegmentTrack:
    """Open straight (acceleration event)."""
    return SegmentTrack([Straight(length_m)], closed=False, name=name)


def circle_track(radius_m: float, direction: int = 1, name: str = "circle") -> SegmentTrack:
    """Closed full circle of the given radius (skidpad / validation)."""
    return SegmentTrack([Arc(radius_m, 2.0 * math.pi, direction)], closed=True, name=name)


def with_name(track: SegmentTrack, name: str) -> SegmentTrack:
    """Copy of a segment track with a new name."""
    return SegmentTrack(track.segments, track.closed, name, track.start_pose)


__all__ = [
    "Arc",
    "Pose",
    "SampledTrack",
    "Segment",
    "SegmentTrack",
    "Slalom",
    "Straight",
    "Track",
    "circle_track",
    "read_xy_csv",
    "straight_track",
    "track_from_xy",
    "with_name",
]
