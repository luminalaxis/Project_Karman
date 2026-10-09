"""GPS (NMEA 0183, e.g. u-blox NEO-6M) import → track + measured speed trace (extension E2).

Supported sentences (any talker ID: GP, GN, GL …): RMC (time, status, position, speed over
ground, date), GGA (fix quality, satellites, HDOP, altitude) and VTG (speed). Sentences with a
bad checksum or a void RMC status are skipped.

Pipeline: parse → local metric coordinates (equirectangular around the first fix) → lap
split at closest approaches to the start point (interpolated between samples) → track of one
lap → simulate → compare speed traces on a common distance axis.

Two track reconstructions (DECISIONS D-19):

* ``"course"`` (default): distance s = ∫ v dt from the Doppler speed, heading ψ from the
  Doppler course over ground, curvature κ = dψ/ds from a speed-weighted smoothing spline of
  ψ(s) (wrapped periodically, heading closure error removed). At 1–5 Hz this is far better
  conditioned than positions: a 5 m hairpin gets only a few position fixes, and a position
  spline either cuts it (too fast) or oscillates through the noise (too slow).
* ``"position"``: smoothing spline through the fixes (:func:`fsk_lapsim.track.track_from_xy`).
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import interpolate

from .solver import LapResult
from .track import SampledTrack, Track, track_from_xy

ArrayLike = Sequence[float] | NDArray[np.float64]

EARTH_RADIUS_M = 6_371_008.8
KNOT_MPS = 0.514444


def nmea_checksum(body: str) -> str:
    """XOR checksum of the characters between '$' and '*' (two hex digits)."""
    c = 0
    for ch in body:
        c ^= ord(ch)
    return f"{c:02X}"


def _valid(sentence: str) -> str | None:
    s = sentence.strip()
    if not s.startswith("$") or "*" not in s:
        return None
    body, _, cs = s[1:].partition("*")
    if nmea_checksum(body) != cs[:2].upper():
        return None
    return body


def _latlon(value: str, hemi: str, deg_digits: int) -> float:
    if not value:
        return math.nan
    deg = float(value[:deg_digits])
    minutes = float(value[deg_digits:])
    out = deg + minutes / 60.0
    return -out if hemi in ("S", "W") else out


def _tod(hhmmss: str) -> float:
    if not hhmmss:
        return math.nan
    return int(hhmmss[0:2]) * 3600 + int(hhmmss[2:4]) * 60 + float(hhmmss[4:])


def parse_nmea(lines: Iterable[str]) -> pd.DataFrame:
    """Parse NMEA sentences into one row per epoch (keyed by UTC time of day).

    Columns: ``t_s`` (seconds from the first valid epoch, midnight roll-over handled),
    ``lat_deg``, ``lon_deg``, ``speed_mps``, ``course_deg``, ``fix``, ``sats``, ``hdop``,
    ``alt_m``. Only epochs with a valid RMC position are kept.
    """
    epochs: dict[float, dict[str, float]] = {}
    order: list[float] = []
    last_rmc: float | None = None
    for line in lines:
        body = _valid(line)
        if body is None:
            continue
        f = body.split(",")
        kind = f[0][2:]
        if kind == "RMC" and len(f) >= 10:
            if f[2] != "A":
                continue
            tod = _tod(f[1])
            rec = epochs.setdefault(tod, {})
            if tod not in order:
                order.append(tod)
            rec.update(
                lat_deg=_latlon(f[3], f[4], 2),
                lon_deg=_latlon(f[5], f[6], 3),
                speed_mps=float(f[7]) * KNOT_MPS if f[7] else math.nan,
                course_deg=float(f[8]) if f[8] else math.nan,
            )
            last_rmc = tod
        elif kind == "GGA" and len(f) >= 10:
            tod = _tod(f[1])
            rec = epochs.setdefault(tod, {})
            rec.update(
                fix=float(f[6] or 0),
                sats=float(f[7] or 0),
                hdop=float(f[8]) if f[8] else math.nan,
                alt_m=float(f[9]) if f[9] else math.nan,
            )
        elif kind == "VTG" and len(f) >= 8 and last_rmc is not None:
            rec = epochs[last_rmc]
            if f[7] and math.isnan(rec.get("speed_mps", math.nan)):
                rec["speed_mps"] = float(f[7]) / 3.6
    rows = [dict(tod=t, **epochs[t]) for t in order if "lat_deg" in epochs[t]]
    if not rows:
        raise ValueError("no valid RMC fixes in the NMEA data")
    df = pd.DataFrame(rows)
    tod = df["tod"].to_numpy()
    rollover = np.concatenate([[0.0], np.cumsum(np.diff(tod) < -43_200) * 86_400.0])
    df.insert(0, "t_s", tod + rollover - tod[0])
    return df.drop(columns="tod").reset_index(drop=True)


def read_nmea(path: str | Path) -> pd.DataFrame:
    """Read an NMEA log file (see :func:`parse_nmea`)."""
    with open(path, encoding="ascii", errors="replace") as fh:
        return parse_nmea(fh)


def to_local_xy(
    lat_deg: ArrayLike, lon_deg: ArrayLike, lat0: float | None = None, lon0: float | None = None
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Equirectangular projection to metres around (lat0, lon0) (default: first point)."""
    lat = np.asarray(lat_deg, float)
    lon = np.asarray(lon_deg, float)
    la0 = float(lat[0]) if lat0 is None else lat0
    lo0 = float(lon[0]) if lon0 is None else lon0
    x = EARTH_RADIUS_M * math.cos(math.radians(la0)) * np.radians(lon - lo0)
    y = EARTH_RADIUS_M * np.radians(lat - la0)
    return x, y


@dataclass
class GpsLap:
    """One lap cut out of a GPS log."""

    index: int
    t_start_s: float
    t_end_s: float
    x_m: NDArray[np.float64]
    y_m: NDArray[np.float64]
    t_s: NDArray[np.float64]
    speed_mps: NDArray[np.float64]
    course_deg: NDArray[np.float64] | None = None

    @property
    def lap_time_s(self) -> float:
        """Lap time from the interpolated line crossings [s]."""
        return self.t_end_s - self.t_start_s

    @property
    def distance_m(self) -> NDArray[np.float64]:
        """Cumulative chord distance of the samples [m] (inflated by position noise)."""
        return np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(self.x_m), np.diff(self.y_m)))])

    @property
    def odometer_m(self) -> NDArray[np.float64]:
        """Distance from the Doppler speed, ∫ v dt [m] (falls back to chord distance)."""
        v = self.speed_mps
        if np.isnan(v).any():
            return self.distance_m
        return np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(self.t_s))])


def _closest_time(
    t: NDArray[np.float64],
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    k: int,
    x0: float,
    y0: float,
) -> float:
    """Time of the closest approach to (x0, y0) on the segments around sample k."""
    best_d, best_t = math.hypot(x[k] - x0, y[k] - y0), float(t[k])
    for a, b in ((k - 1, k), (k, k + 1)):
        if a < 0 or b >= len(t):
            continue
        dx, dy = x[b] - x[a], y[b] - y[a]
        seg2 = dx * dx + dy * dy
        if seg2 == 0:
            continue
        u = min(max(((x0 - x[a]) * dx + (y0 - y[a]) * dy) / seg2, 0.0), 1.0)
        d = math.hypot(x[a] + u * dx - x0, y[a] + u * dy - y0)
        if d < best_d:
            best_d, best_t = d, float(t[a] + u * (t[b] - t[a]))
    return best_t


def split_laps(
    df: pd.DataFrame,
    start_xy: tuple[float, float] | None = None,
    gate_radius_m: float = 15.0,
    min_lap_m: float = 200.0,
) -> list[GpsLap]:
    """Cut the log into laps at closest approaches to the start point.

    ``start_xy`` defaults to the first fix. A crossing is a local minimum of the distance
    to the start that is within ``gate_radius_m`` and at least ``min_lap_m`` of travel after
    the previous crossing.
    """
    x, y = to_local_xy(df["lat_deg"], df["lon_deg"])
    t = df["t_s"].to_numpy(float)
    v = df["speed_mps"].to_numpy(float)
    cog = df["course_deg"].to_numpy(float) if "course_deg" in df else None
    x0, y0 = start_xy if start_xy is not None else (float(x[0]), float(y[0]))
    d = np.hypot(x - x0, y - y0)
    travel = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    # the first fix is a crossing when it *is* the start point
    cross: list[int] = [0] if start_xy is None else []
    for k in range(1, len(d) - 1):
        is_min = d[k] <= gate_radius_m and d[k] <= d[k - 1] and d[k] < d[k + 1]
        if is_min and (not cross or travel[k] - travel[cross[-1]] >= min_lap_m):
            cross.append(k)
    laps = []
    for i, (a, b) in enumerate(itertools.pairwise(cross)):
        ta = _closest_time(t, x, y, a, x0, y0)
        tb = _closest_time(t, x, y, b, x0, y0)
        sl = slice(a, b + 1)
        laps.append(GpsLap(i, ta, tb, x[sl], y[sl], t[sl], v[sl], None if cog is None else cog[sl]))
    return laps


def track_from_lap(
    lap: GpsLap,
    method: str = "course",
    smoothing_sigma_m: float = 1.0,
    course_sigma_deg: float = 0.5,
    ds_m: float = 0.25,
    name: str = "gps_lap",
) -> SampledTrack:
    """Closed track of the driven line from one GPS lap.

    Args:
        lap: Lap cut by :func:`split_laps`.
        method: ``"course"`` (Doppler course + speed, default) or ``"position"``.
        smoothing_sigma_m: Position noise σ for the position spline.
        course_sigma_deg: Course noise σ at speed (NEO-6M datasheet: 0.5°); below 5 m/s the
            course is down-weighted in proportion to speed.
        ds_m: Output spacing [m].
        name: Track name.
    """
    if method == "position":
        return track_from_xy(
            lap.x_m[:-1],
            lap.y_m[:-1],
            closed=True,
            smoothing_sigma_m=smoothing_sigma_m,
            ds_m=ds_m,
            name=name,
        )
    if method != "course":
        raise ValueError("method must be 'course' or 'position'")
    if lap.course_deg is None or np.isnan(lap.course_deg).any():
        raise ValueError("course over ground missing; use method='position'")
    t, v = lap.t_s, np.maximum(lap.speed_mps, 0.0)
    s = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(t))])
    # math heading (CCW from east), unwrapped
    psi = np.unwrap(np.radians(90.0 - lap.course_deg))
    L = float(s[-1])
    turn = psi[-1] - psi[0]
    n_turns = round(turn / (2 * math.pi))
    closure = turn - n_turns * 2 * math.pi
    psi = psi - closure * s / L  # remove heading drift so the lap closes in heading
    sig = math.radians(course_sigma_deg) / np.clip(v / 5.0, 0.05, 1.0)
    # periodic padding: one lap before and after
    ss = np.concatenate([s[:-1] - L, s[:-1], s[:-1] + L])
    shift = n_turns * 2 * math.pi
    pp = np.concatenate([psi[:-1] - shift, psi[:-1], psi[:-1] + shift])
    ww = np.tile(1.0 / sig[:-1], 3)
    keep = np.concatenate([[True], np.diff(ss) > 1e-6])
    spl = interpolate.UnivariateSpline(ss[keep], pp[keep], w=ww[keep], k=3, s=float(keep.sum()))
    m = max(math.ceil(L / ds_m), 4)
    grid = np.linspace(0.0, L, m + 1)
    kappa = spl.derivative()(grid)
    kappa[-1] = kappa[0]
    th = spl(grid)
    dsg = np.diff(grid)
    thm = 0.5 * (th[1:] + th[:-1])
    x = lap.x_m[0] + np.concatenate([[0.0], np.cumsum(np.cos(thm) * dsg)])
    y = lap.y_m[0] + np.concatenate([[0.0], np.cumsum(np.sin(thm) * dsg)])
    frac = grid / L
    x -= frac * (x[-1] - x[0])  # spread the small position closure error
    y -= frac * (y[-1] - y[0])
    return SampledTrack(grid, kappa, x, y, closed=True, name=name)


@dataclass
class TraceComparison:
    """Measured vs simulated speed on a common distance axis."""

    s_m: NDArray[np.float64]
    v_measured_mps: NDArray[np.float64]
    v_sim_mps: NDArray[np.float64]
    lap_time_measured_s: float
    lap_time_sim_s: float

    @property
    def rmse_mps(self) -> float:
        """RMS speed difference [m/s]."""
        return float(np.sqrt(np.mean((self.v_sim_mps - self.v_measured_mps) ** 2)))

    @property
    def mean_diff_mps(self) -> float:
        """Mean (sim − measured) speed [m/s]; positive = model faster than the driver."""
        return float(np.mean(self.v_sim_mps - self.v_measured_mps))

    def summary(self) -> dict[str, float]:
        """Scalar comparison figures."""
        return {
            "lap_time_measured_s": self.lap_time_measured_s,
            "lap_time_sim_s": self.lap_time_sim_s,
            "lap_time_ratio": self.lap_time_measured_s / self.lap_time_sim_s,
            "speed_rmse_kph": 3.6 * self.rmse_mps,
            "mean_speed_diff_kph": 3.6 * self.mean_diff_mps,
            "max_speed_measured_kph": 3.6 * float(np.nanmax(self.v_measured_mps)),
            "max_speed_sim_kph": 3.6 * float(self.v_sim_mps.max()),
        }


def compare_trace(lap: GpsLap, track: Track, sim: LapResult) -> TraceComparison:
    """Put measured and simulated speed on one distance axis.

    The measured samples are placed by their Doppler odometer (∫ v dt, the same axis as the
    course reconstruction) scaled to the track length, and the simulated speed is
    interpolated there.
    """
    dist = lap.odometer_m
    s = dist / dist[-1] * track.length_m
    v_sim = np.interp(s, sim.s, sim.v)
    return TraceComparison(s, lap.speed_mps, v_sim, lap.lap_time_s, sim.lap_time_s)


# --------------------------------------------------------------------------------------------
# Synthetic NMEA (DUMMY sample data and tests)
# --------------------------------------------------------------------------------------------
def _fmt_lat(lat: float) -> tuple[str, str]:
    a = abs(lat)
    d = int(a)
    return f"{d:02d}{(a - d) * 60:07.4f}", "N" if lat >= 0 else "S"


def _fmt_lon(lon: float) -> tuple[str, str]:
    a = abs(lon)
    d = int(a)
    return f"{d:03d}{(a - d) * 60:07.4f}", "E" if lon >= 0 else "W"


def _sentence(body: str) -> str:
    return f"${body}*{nmea_checksum(body)}"


def synth_nmea(
    track: Track,
    lap: LapResult,
    laps: int = 3,
    rate_hz: float = 5.0,
    pos_sigma_m: float = 0.5,
    speed_sigma_mps: float = 0.1,
    speed_scale: float = 1.0,
    course_sigma_deg: float = 0.5,
    lat0: float = 35.2919,
    lon0: float = 126.5741,
    t0_s: float = 3 * 3600.0,
    seed: int = 0,
) -> list[str]:
    """Generate NMEA RMC/GGA/VTG sentences for ``laps`` repeats of a simulated lap.

    ``speed_scale`` < 1 imitates a driver below the limit (time stretched accordingly).
    Positions get white noise of ``pos_sigma_m``, speeds of ``speed_sigma_mps`` and the
    (Doppler) course over ground of ``course_sigma_deg`` around the true heading.
    """
    rng = np.random.default_rng(seed)
    t_lap = lap.t / speed_scale
    T = float(t_lap[-1])
    total = laps * T
    times = np.arange(0.0, total, 1.0 / rate_hz)
    tl = np.mod(times, T)
    s = np.interp(tl, t_lap, lap.s)
    v = np.interp(tl, t_lap, lap.v) * speed_scale
    x, y = track.xy(s)
    xa, ya = track.xy(np.minimum(s + 0.05, track.length_m))
    xb, yb = track.xy(np.maximum(s - 0.05, 0.0))
    hd = np.degrees(np.arctan2(xa - xb, ya - yb)) + rng.normal(0, course_sigma_deg, s.size)
    hd = hd % 360.0  # course: clockwise from north
    x = x + rng.normal(0, pos_sigma_m, x.size)
    y = y + rng.normal(0, pos_sigma_m, y.size)
    v = np.maximum(v + rng.normal(0, speed_sigma_mps, v.size), 0.0)
    lat = lat0 + np.degrees(y / EARTH_RADIUS_M)
    lon = lon0 + np.degrees(x / (EARTH_RADIUS_M * math.cos(math.radians(lat0))))
    out = []
    for k in range(times.size):
        tod = (t0_s + times[k]) % 86_400.0
        hh, rem = divmod(tod, 3600.0)
        mm, ss = divmod(rem, 60.0)
        stamp = f"{int(hh):02d}{int(mm):02d}{ss:05.2f}"
        la, ns = _fmt_lat(float(lat[k]))
        lo, ew = _fmt_lon(float(lon[k]))
        kn = v[k] / KNOT_MPS
        out.append(
            _sentence(f"GPRMC,{stamp},A,{la},{ns},{lo},{ew},{kn:.3f},{hd[k]:.2f},091026,,,A")
        )
        out.append(_sentence(f"GPVTG,{hd[k]:.2f},T,,M,{kn:.3f},N,{v[k] * 3.6:.3f},K,A"))
        out.append(_sentence(f"GPGGA,{stamp},{la},{ns},{lo},{ew},1,08,0.9,25.0,M,20.0,M,,"))
    return out
