"""Slalom cones: the car must pass them alternately left and right (an S).

Cones are projected onto the track path and grouped into slaloms by their spacing.  For each
cone the path offset alpha (along the left normal) must satisfy

    pass left :  alpha >= cone_offset + d          d = car_width/2 + CONE_RADIUS + margin
    pass right:  alpha <= cone_offset - d

with sides alternating inside a group.  The first side of every group is a free choice;
:func:`choose_sides` picks it by simulating the lap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .track import Track, track_from_dict

CONE_RADIUS = 0.15   # [m] FS cone base ~ 0.23-0.29 m
GROUP_GAP = 15.0     # [m] cones closer than this along the track form one slalom


@dataclass
class Gate:
    index: int        # track point nearest to the cone (unwrapped for groups across the seam)
    offset: float     # cone offset from the path along the left normal [m]
    cone: int = -1    # row in track.slalom_cones


def find_slaloms(track: Track, gap: float = GROUP_GAP) -> list[list[Gate]]:
    if track.slalom_cones is None or len(track.slalom_cones) == 0:
        return []
    P = np.column_stack([track.x, track.y])
    nx, ny = track.normals()
    gates = []
    for k, c in enumerate(track.slalom_cones):
        i = int(np.argmin(np.hypot(P[:, 0] - c[0], P[:, 1] - c[1])))
        gates.append(Gate(i, float((c[0] - P[i, 0]) * nx[i] + (c[1] - P[i, 1]) * ny[i]), k))
    gates.sort(key=lambda g: g.index)
    gap_pts = gap / track.ds
    groups = [[gates[0]]]
    for g in gates[1:]:
        if g.index - groups[-1][-1].index > gap_pts:
            groups.append([g])
        else:
            groups[-1].append(g)
    n = len(track.s)
    if track.closed and len(groups) > 1 and groups[0][0].index + n - groups[-1][-1].index <= gap_pts:
        groups[-1] += [Gate(g.index + n, g.offset, g.cone) for g in groups.pop(0)]   # across the seam
    return groups


def pass_distance(car_width: float, margin: float) -> float:
    return car_width / 2 + CONE_RADIUS + margin


def cone_sides(groups, first_sides) -> list[list[int]]:
    """+1 = pass the cone on its left, -1 = on its right; alternating within a group."""
    return [[first * (-1) ** k for k in range(len(g))] for g, first in zip(groups, first_sides)]


def slalom_offsets(track: Track, groups, sides, d: float) -> np.ndarray:
    """Smooth path offsets that weave through every group and return to the path in between."""
    n = len(track.s)
    alpha = np.zeros(n)
    for g, sd in zip(groups, sides):
        idx = np.array([x.index for x in g], float)
        val = np.array([x.offset + s * d for x, s in zip(g, sd)])
        lead = max(np.mean(np.diff(idx)) if len(idx) > 1 else 0.0, 6.0 / track.ds)
        knots = np.concatenate([[idx[0] - lead], idx, [idx[-1] + lead]])
        vals = np.concatenate([[0.0], val, [0.0]])
        span = np.arange(int(np.ceil(knots[0])), int(np.floor(knots[-1])) + 1)
        j = np.clip(np.searchsorted(knots, span, side="right") - 1, 0, len(knots) - 2)
        u = (span - knots[j]) / (knots[j + 1] - knots[j])
        w = 0.5 * (1 - np.cos(np.pi * u))                  # zero slope at every knot
        a = vals[j] + (vals[j + 1] - vals[j]) * w
        if track.closed:
            alpha[span % n] = a
        else:
            ok = (span >= 0) & (span < n)
            alpha[span[ok]] = a[ok]
    return alpha


def slalom_path(track: Track, groups, sides, d: float, car_width: float = 1.45,
                margin: float = 0.25) -> Track:
    """Fallback S-path for tracks without width data: cosine weave through the cones."""
    alpha = slalom_offsets(track, groups, sides, d)
    if track.width_left is not None:
        half = car_width / 2 + margin
        alpha = np.clip(alpha, -(track.width_right - half), track.width_left - half)
    nx, ny = track.normals()
    path = track_from_dict({"x": (track.x + alpha * nx).tolist(), "y": (track.y + alpha * ny).tolist(),
                            "closed": track.closed}, ds=track.ds, smooth_m=1.0,
                           name=f"{track.name} (slalom)")
    path.slalom_cones = track.slalom_cones
    return path


def choose_sides(veh, build_path, groups):
    """First side per slalom, picked greedily by lap time; ``build_path(sides)`` -> Track."""
    from .solver import simulate
    first = [1] * len(groups)

    def lap(fs):
        return simulate(veh, build_path(cone_sides(groups, fs))).lap_time

    best = lap(first)
    for k in range(len(groups)):
        trial = first.copy()
        trial[k] = -trial[k]
        t = lap(trial)
        if t < best:
            first, best = trial, t
    return cone_sides(groups, first)


def clearance(path: Track, cones: np.ndarray) -> np.ndarray:
    """Signed lateral distance of each cone from the driven path (+ = cone on the left)."""
    P = np.column_stack([path.x, path.y])
    nx, ny = path.normals()
    out = []
    for c in cones:
        i = int(np.argmin(np.hypot(P[:, 0] - c[0], P[:, 1] - c[1])))
        out.append((c[0] - P[i, 0]) * nx[i] + (c[1] - P[i, 1]) * ny[i])
    return np.array(out)
