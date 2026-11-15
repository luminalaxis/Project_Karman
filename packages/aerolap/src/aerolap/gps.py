"""GPS logger (NMEA) -> laps, driven line and measured vs simulated speed (ported from fsk-lapsim E2).

Only ``$--RMC`` sentences are used (time, position, speed over ground); bad checksums and void fixes are
skipped. Laps are split where the car passes closest to a start gate (default: the first fix).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

KNOT = 0.514444
R_EARTH = 6_371_000.0


def _checksum_ok(line: str) -> bool:
    if "*" not in line:
        return True  # some loggers omit it
    body, cs = line[1:].split("*", 1)
    x = 0
    for c in body:
        x ^= ord(c)
    return cs[:2].upper() == f"{x:02X}"


def _deg(v: str, hemi: str, n: int) -> float:
    d = float(v[:n]) + float(v[n:]) / 60.0
    return -d if hemi in "SW" else d


def parse_nmea(lines) -> np.ndarray:
    """Rows of (t [s from the first fix], lat [deg], lon [deg], speed [m/s])."""
    rows = []
    for line in lines:
        line = line.strip()
        if not line.startswith("$") or line[3:6] != "RMC" or not _checksum_ok(line):
            continue
        f = line.split("*")[0].split(",")
        if len(f) < 8 or f[2] != "A" or not f[3] or not f[5]:
            continue
        hh, mm, ss = float(f[1][:2]), float(f[1][2:4]), float(f[1][4:])
        rows.append((hh * 3600 + mm * 60 + ss, _deg(f[3], f[4], 2), _deg(f[5], f[6], 3),
                     float(f[7] or "nan") * KNOT))
    if not rows:
        raise ValueError("no valid RMC fixes in the NMEA data")
    a = np.array(rows)
    a[:, 0] = np.concatenate([[0.0], np.cumsum(np.where(np.diff(a[:, 0]) < -43_200, 86_400.0, 0.0))]) + a[:, 0]
    a[:, 0] -= a[0, 0]  # midnight roll-over handled above
    return a


def local_xy(lat, lon, lat0: float, lon0: float):
    """Equirectangular projection around (lat0, lon0) [m]; fine for a few-km circuit."""
    x = np.radians(np.asarray(lon) - lon0) * R_EARTH * math.cos(math.radians(lat0))
    y = np.radians(np.asarray(lat) - lat0) * R_EARTH
    return x, y


@dataclass
class GpsLap:
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    v: np.ndarray

    @property
    def time_s(self) -> float:
        return float(self.t[-1] - self.t[0])

    @property
    def s(self) -> np.ndarray:
        return np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(self.x), np.diff(self.y)))])


def split_laps(fixes: np.ndarray, gate: tuple[float, float] | None = None, radius_m: float = 15.0,
               min_lap_m: float = 200.0) -> list[GpsLap]:
    lat0, lon0 = (fixes[0, 1], fixes[0, 2]) if gate is None else gate
    x, y = local_xy(fixes[:, 1], fixes[:, 2], lat0, lon0)
    d = np.hypot(x, y)
    near = d < radius_m
    cuts, i = [], 0
    while i < len(d):  # one cut per pass through the gate circle, at the closest point
        if near[i]:
            j = i
            while j < len(d) and near[j]:
                j += 1
            cuts.append(i + int(np.argmin(d[i:j])))
            i = j
        else:
            i += 1
    laps = []
    for a, b in zip(cuts, cuts[1:]):
        lap = GpsLap(fixes[a:b + 1, 0], x[a:b + 1], y[a:b + 1], fixes[a:b + 1, 3])
        if lap.s[-1] >= min_lap_m:
            laps.append(lap)
    return laps


def write_track_csv(lap: GpsLap, path: str | Path, name: str, cutoff_m: float = 10.0) -> None:
    """Driven line as an aerolap x-y track (smoothed by Track.from_file with ``cutoff_m``)."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"# name: {name}\n# closed: true\n# cutoff_m: {cutoff_m:g}\n# source: GPS lap {lap.time_s:.2f} s\n")
        fh.write("x_m,y_m\n" + "".join(f"{a:.3f},{b:.3f}\n" for a, b in zip(lap.x[:-1], lap.y[:-1])))


def compare_speed(lap: GpsLap, sim_s: np.ndarray, sim_v: np.ndarray, n: int = 500):
    """Measured vs simulated speed on a common distance fraction (lengths differ slightly).

    Returns (fraction, v_gps, v_sim, rms, mean sim - gps) in m/s."""
    f = np.linspace(0.0, 1.0, n)
    vg = np.interp(f, lap.s / lap.s[-1], lap.v)
    vs = np.interp(f, sim_s / sim_s[-1], sim_v)
    return f, vg, vs, float(np.sqrt(np.mean((vs - vg) ** 2))), float(np.mean(vs - vg))


def _nmea(t: float, lat: float, lon: float, v: float) -> str:
    hh, mm, ss = int(t // 3600) % 24, int(t % 3600 // 60), t % 60
    la, lo = abs(lat), abs(lon)
    body = (f"GPRMC,{hh:02d}{mm:02d}{ss:05.2f},A,{int(la):02d}{(la % 1) * 60:07.4f},{'N' if lat >= 0 else 'S'},"
            f"{int(lo):03d}{(lo % 1) * 60:07.4f},{'E' if lon >= 0 else 'W'},{v / KNOT:.2f},0.0,010126,,")
    x = 0
    for c in body:
        x ^= ord(c)
    return f"${body}*{x:02X}"

