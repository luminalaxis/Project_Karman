"""Rule-based synthetic track generator, Dubins loop closure, rule checks and YAML loading.

The layout ranges follow the usual Formula Student guidelines (straights ≤ 80 m, constant
turns ≤ 50 m diameter, hairpins ≥ 9 m outside diameter, slalom cones 7.5–12 m apart). They are
ASSUMED placeholders until replaced by the FSK rulebook (see ASSUMPTIONS.md).

The generated geometry is the *driven line* (the solver has no racing-line optimiser), so the
hairpin radius range describes a driven line that uses the track width, not the cone radius.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .config import load_yaml, resolve_path
from .track import (
    Arc,
    Pose,
    SampledTrack,
    Segment,
    SegmentTrack,
    Slalom,
    Straight,
    Track,
    read_xy_csv,
    track_from_xy,
)

TWO_PI = 2.0 * math.pi


@dataclass(frozen=True)
class LayoutRules:
    """Geometric limits for generated layouts (ASSUMED, replace with FSK rules).

    Attributes:
        max_straight_m: Longest straight [m].
        min_straight_m: Shortest straight between features [m].
        turn_radius_m: Range of constant-turn driven-line radii [m] (≤ 25 m ⇔ ≤ 50 m diameter).
        hairpin_radius_m: Range of hairpin driven-line radii [m].
        slalom_spacing_m: Range of slalom cone spacings [m].
        slalom_cones: Range (inclusive) of cones per slalom.
        slalom_weave_m: Peak-to-peak lateral excursion of the line through a slalom [m].
        min_gap_m: Minimum distance between non-adjacent parts of the driven line [m].
        gap_exclusion_m: Arc-length window treated as "adjacent" for the gap check [m].
        closure_radius_m: Radii tried for the Dubins closing turns of a loop [m].
    """

    max_straight_m: float = 80.0
    min_straight_m: float = 10.0
    turn_radius_m: tuple[float, float] = (7.0, 25.0)
    hairpin_radius_m: tuple[float, float] = (4.5, 7.0)
    slalom_spacing_m: tuple[float, float] = (7.5, 12.0)
    slalom_cones: tuple[int, int] = (4, 7)
    slalom_weave_m: float = 2.0
    min_gap_m: float = 8.0
    gap_exclusion_m: float = 25.0
    closure_radius_m: tuple[float, ...] = (10.0, 12.0, 15.0, 18.0, 22.0, 25.0)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> LayoutRules:
        """Build from a (partial) mapping; lists become tuples."""
        if not data:
            return cls()
        names = {f.name for f in fields(cls)}
        unknown = set(data) - names
        if unknown:
            raise ValueError(f"unknown layout rule(s): {sorted(unknown)}")
        conv = {k: tuple(v) if isinstance(v, list) else v for k, v in data.items()}
        return replace(cls(), **conv)


# --------------------------------------------------------------------------------------------
# Dubins paths (CSC family) for loop closure
# --------------------------------------------------------------------------------------------
def _mod2pi(a: float) -> float:
    return a % TWO_PI


def dubins_csc(start: Pose, goal: Pose, radius_m: float) -> list[tuple[str, float, float, float]]:
    """All feasible CSC Dubins paths from ``start`` to ``goal``.

    Returns a list of ``(word, t, p, q)`` with arc angles t, q [rad] and straight length
    p [m], for the words LSL, RSR, LSR, RSL, sorted by total length.
    """
    x0, y0, th0 = start
    x1, y1, th1 = goal
    dx, dy = x1 - x0, y1 - y0
    d = math.hypot(dx, dy) / radius_m
    theta = math.atan2(dy, dx)
    a = _mod2pi(th0 - theta)
    b = _mod2pi(th1 - theta)
    sa, sb, ca, cb = math.sin(a), math.sin(b), math.cos(a), math.cos(b)
    cab = math.cos(a - b)
    out: list[tuple[str, float, float, float]] = []
    p2 = 2 + d * d - 2 * cab + 2 * d * (sa - sb)
    if p2 >= 0:
        tmp = math.atan2(cb - ca, d + sa - sb)
        out.append(("LSL", _mod2pi(-a + tmp), math.sqrt(p2), _mod2pi(b - tmp)))
    p2 = 2 + d * d - 2 * cab + 2 * d * (sb - sa)
    if p2 >= 0:
        tmp = math.atan2(ca - cb, d - sa + sb)
        out.append(("RSR", _mod2pi(a - tmp), math.sqrt(p2), _mod2pi(-b + tmp)))
    p2 = -2 + d * d + 2 * cab + 2 * d * (sa + sb)
    if p2 >= 0:
        p = math.sqrt(p2)
        tmp = math.atan2(-ca - cb, d + sa + sb) - math.atan2(-2.0, p)
        out.append(("LSR", _mod2pi(-a + tmp), p, _mod2pi(-b + tmp)))
    p2 = d * d - 2 + 2 * cab - 2 * d * (sa + sb)
    if p2 >= 0:
        p = math.sqrt(p2)
        tmp = math.atan2(ca + cb, d - sa - sb) - math.atan2(2.0, p)
        out.append(("RSL", _mod2pi(a - tmp), p, _mod2pi(b - tmp)))
    res = [(w, t, p * radius_m, q) for (w, t, p, q) in out]
    res.sort(key=lambda r: r[1] * radius_m + r[2] + r[3] * radius_m)
    return res


def dubins_segments(word: str, t: float, p: float, q: float, radius_m: float) -> list[Segment]:
    """Convert a Dubins ``(word, t, p, q)`` into track segments (zero-length parts dropped)."""
    dirs = {"L": 1, "R": -1}
    segs: list[Segment] = []
    if t > 1e-9:
        segs.append(Arc(radius_m, t, dirs[word[0]]))
    if p > 1e-9:
        segs.append(Straight(p))
    if q > 1e-9:
        segs.append(Arc(radius_m, q, dirs[word[2]]))
    return segs


# --------------------------------------------------------------------------------------------
# Geometry checks
# --------------------------------------------------------------------------------------------
def sample_points(
    track: Track, step_m: float = 1.0
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Points every ``step_m`` along the driven line: returns (s [n], xy [n×2])."""
    n = max(math.ceil(track.length_m / step_m), 2)
    s = np.linspace(0.0, track.length_m, n + 1)
    if track.closed:
        s = s[:-1]
    x, y = track.xy(s)
    return s, np.column_stack([x, y])


def min_gap(track: Track, exclusion_m: float, step_m: float = 1.0) -> float:
    """Smallest distance between parts of the line more than ``exclusion_m`` apart along s."""
    s, p = sample_points(track, step_m)
    ds = np.abs(s[:, None] - s[None, :])
    if track.closed:
        ds = np.minimum(ds, track.length_m - ds)
    d = np.hypot(p[:, None, 0] - p[None, :, 0], p[:, None, 1] - p[None, :, 1])
    far = ds > exclusion_m
    return float(d[far].min()) if far.any() else float("inf")


def longest_straight(track: Track, ds_m: float = 0.25, kappa_tol: float = 1e-6) -> float:
    """Longest run of zero curvature [m] (wrapping around for closed tracks)."""
    s, k = track.discretize(ds_m)
    flat = np.abs(k[:-1]) < kappa_tol
    if track.closed and flat.all():
        return track.length_m
    seq = np.concatenate([flat, flat]) if track.closed else flat
    best = run = 0
    for f in seq:
        run = run + 1 if f else 0
        best = max(best, run)
    return min(best * (s[1] - s[0]), track.length_m)


def check_layout(track: Track, rules: LayoutRules) -> dict[str, Any]:
    """Check a track against the layout rules; returns measured values and a list of issues."""
    issues: list[str] = []
    ls = longest_straight(track)
    if ls > rules.max_straight_m + 1e-6:
        issues.append(f"straight {ls:.1f} m > {rules.max_straight_m} m")
    _, k = track.discretize(0.25)
    kmax = float(np.abs(k).max())
    rmin = 1.0 / kmax if kmax > 0 else math.inf
    if rmin < rules.hairpin_radius_m[0] - 1e-6:
        issues.append(f"min radius {rmin:.2f} m < {rules.hairpin_radius_m[0]} m")
    if isinstance(track, SegmentTrack):
        for seg in track.segments:
            if isinstance(seg, Arc) and seg.radius_m > rules.turn_radius_m[1] + 1e-6:
                issues.append(
                    f"constant turn radius {seg.radius_m:.1f} m > {rules.turn_radius_m[1]}"
                )
            if isinstance(seg, Slalom) and not (
                rules.slalom_spacing_m[0] - 1e-9
                <= seg.spacing_m
                <= rules.slalom_spacing_m[1] + 1e-9
            ):
                issues.append(f"slalom spacing {seg.spacing_m:.2f} m outside rule range")
    gap = min_gap(track, rules.gap_exclusion_m)
    if gap < rules.min_gap_m - 1e-6:
        issues.append(f"line comes within {gap:.2f} m of itself (< {rules.min_gap_m} m)")
    closure = None
    if isinstance(track, SegmentTrack) and track.closed:
        closure = track.closure_error()
        if closure[0] > 1e-6 or closure[1] > 1e-6:
            issues.append(f"loop not closed: {closure[0]:.3g} m, {closure[1]:.3g} rad")
    return {
        "length_m": track.length_m,
        "longest_straight_m": ls,
        "min_radius_m": rmin,
        "min_gap_m": gap,
        "closure_error": closure,
        "issues": issues,
    }


# --------------------------------------------------------------------------------------------
# Random generator
# --------------------------------------------------------------------------------------------
class _Builder:
    """Incrementally places segments while enforcing the self-gap rule."""

    def __init__(self, rules: LayoutRules, start_pose: Pose = (0.0, 0.0, 0.0)) -> None:
        self.rules = rules
        self.segments: list[Segment] = []
        self.pose = start_pose
        self.start_pose = start_pose
        self.length = 0.0
        self.pts = np.zeros((0, 2))
        self.pts_s = np.zeros(0)

    def _points(
        self, segs: Sequence[Segment], pose: Pose, s0: float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], Pose]:
        pts, ss = [], []
        for seg in segs:
            n = max(math.ceil(seg.length_m), 1)
            sl = np.linspace(0.0, seg.length_m, n + 1)[1:]
            lx, ly = seg.local_xy(sl)
            c, sn = math.cos(pose[2]), math.sin(pose[2])
            pts.append(np.column_stack([pose[0] + c * lx - sn * ly, pose[1] + sn * lx + c * ly]))
            ss.append(s0 + sl)
            s0 += seg.length_m
            pose = seg.end_pose(pose)
        return np.vstack(pts), np.concatenate(ss), pose

    def fits(self, segs: Sequence[Segment], loop_length: float | None = None) -> bool:
        """True if ``segs`` appended to the current line keep the minimum gap."""
        if not segs:
            return True
        p, s, _ = self._points(segs, self.pose, self.length)
        allp = np.vstack([self.pts, p])
        alls = np.concatenate([self.pts_s, s])
        ds = np.abs(s[:, None] - alls[None, :])
        if loop_length is not None:
            ds = np.minimum(ds, loop_length - ds)
        far = ds > self.rules.gap_exclusion_m
        if not far.any():
            return True
        d = np.hypot(p[:, None, 0] - allp[None, :, 0], p[:, None, 1] - allp[None, :, 1])
        return bool(d[far].min() >= self.rules.min_gap_m)

    def add(self, segs: Sequence[Segment]) -> None:
        """Append segments (no checks)."""
        if not segs:
            return
        p, s, pose = self._points(segs, self.pose, self.length)
        self.pts = np.vstack([self.pts, p])
        self.pts_s = np.concatenate([self.pts_s, s])
        self.segments += list(segs)
        self.pose = pose
        self.length += sum(seg.length_m for seg in segs)


def _feature(kind: str, rng: np.random.Generator, rules: LayoutRules, d: int) -> list[Segment]:
    u = rng.uniform
    if kind == "straight":
        return [Straight(u(rules.min_straight_m, rules.max_straight_m))]
    if kind == "turn":
        return [Arc(u(*rules.turn_radius_m), math.radians(u(40, 150)), d)]
    if kind == "hairpin":
        return [Arc(u(*rules.hairpin_radius_m), math.radians(u(165, 195)), d)]
    if kind == "slalom":
        n = int(rng.integers(rules.slalom_cones[0], rules.slalom_cones[1] + 1))
        return [Slalom(n, u(*rules.slalom_spacing_m), rules.slalom_weave_m, d)]
    if kind == "chicane":
        r, a = u(7.0, 14.0), math.radians(u(25, 50))
        return [Arc(r, a, d), Straight(u(3.0, 8.0)), Arc(r, a, -d)]
    if kind == "decreasing":
        lo, hi = rules.turn_radius_m
        r1 = u(max(lo, 0.6 * hi), hi)
        r2 = u(lo, max(lo, 0.5 * hi))
        return [Arc(r1, math.radians(u(40, 70)), d), Arc(r2, math.radians(u(40, 80)), d)]
    raise ValueError(kind)


_FEATURES = ("turn", "hairpin", "slalom", "chicane", "decreasing")
_FEATURE_P = np.array([0.36, 0.2, 0.12, 0.17, 0.15])


def _homing_direction(
    rng: np.random.Generator, pose: Pose, home: tuple[float, float], strength: float
) -> int:
    """Pick a turn direction, biased towards ``home`` with weight ``strength`` ∈ [0, 1]."""
    x, y, th = pose
    bearing = math.atan2(home[1] - y, home[0] - x)
    err = (bearing - th + math.pi) % TWO_PI - math.pi
    p_left = 0.5 + 0.5 * strength * math.copysign(min(abs(err) / (math.pi / 2), 1.0), err)
    return 1 if rng.uniform() < p_left else -1


def generate_layout(
    kind: str = "autocross",
    seed: int = 0,
    target_length_m: float = 900.0,
    rules: LayoutRules | None = None,
    max_attempts: int = 400,
    name: str | None = None,
) -> SegmentTrack:
    """Generate a rule-conforming layout deterministically from ``seed``.

    Args:
        kind: ``"autocross"`` (open course) or ``"endurance"`` (closed loop, Dubins-closed).
        seed: RNG seed (same seed → same track).
        target_length_m: Desired length [m]; accepted within −10 %/+15 %.
        rules: Layout limits (defaults: :class:`LayoutRules`).
        max_attempts: Restarts before giving up.
        name: Track name (default ``f"{kind}_seed{seed}"``).
    """
    if kind not in ("autocross", "endurance"):
        raise ValueError("kind must be 'autocross' or 'endurance'")
    rules = rules or LayoutRules()
    rng = np.random.default_rng(seed)
    name = name or f"{kind}_seed{seed}"
    closed = kind == "endurance"
    arena = 0.11 * target_length_m  # soft radius to keep the layout compact
    for _ in range(max_attempts):
        track = _attempt(rng, rules, target_length_m, closed, arena, name)
        if track is not None and not check_layout(track, rules)["issues"]:
            return track
    raise RuntimeError(f"no valid {kind} layout for seed {seed} in {max_attempts} attempts")


def _attempt(
    rng: np.random.Generator,
    rules: LayoutRules,
    target: float,
    closed: bool,
    arena: float,
    name: str,
) -> SegmentTrack | None:
    b = _Builder(rules)
    b.add([Straight(rng.uniform(30.0, 60.0))])  # start straight
    last_was_straight = True
    while b.length < 1.15 * target:
        frac = b.length / target
        dist = math.hypot(b.pose[0], b.pose[1])
        if closed:
            strength = min(1.0, max(0.0, (frac - 0.35) / 0.45)) if frac > 0.35 else 0.0
            strength = max(strength, min(1.0, dist / (2.0 * arena)) * 0.6)
        else:
            strength = min(1.0, max(0.0, (dist - arena) / arena))
        placed = False
        for _try in range(25):
            if not last_was_straight and rng.uniform() < 0.7:
                kind = "straight"
            else:
                kind = str(rng.choice(_FEATURES, p=_FEATURE_P))
            d = _homing_direction(rng, b.pose, (0.0, 0.0), strength)
            segs = _feature(kind, rng, rules, d)
            if b.fits(segs):
                b.add(segs)
                last_was_straight = kind == "straight"
                placed = True
                break
        if not placed:
            return None
        if not closed and b.length >= 0.9 * target:
            if not last_was_straight:
                tail = [Straight(rng.uniform(15.0, 40.0))]
                if b.fits(tail):
                    b.add(tail)
            return SegmentTrack(b.segments, closed=False, name=name)
        if closed and b.length >= 0.55 * target:
            closing = _try_close(b, rules, target)
            if closing is not None:
                b.add(closing)
                return SegmentTrack(b.segments, closed=True, name=name)
    return None


def _try_close(b: _Builder, rules: LayoutRules, target: float) -> list[Segment] | None:
    for radius in rules.closure_radius_m:
        for word, t, p, q in dubins_csc(b.pose, b.start_pose, radius):
            if p > rules.max_straight_m or t > math.radians(200) or q > math.radians(200):
                continue
            segs = dubins_segments(word, t, p, q, radius)
            total = b.length + sum(sg.length_m for sg in segs)
            if not 0.9 * target <= total <= 1.15 * target:
                continue
            if b.fits(segs, loop_length=total):
                return segs
    return None


# --------------------------------------------------------------------------------------------
# YAML track files
# --------------------------------------------------------------------------------------------
def segment_from_dict(d: dict[str, Any]) -> Segment:
    """Build a segment from ``{type: straight|arc|slalom, ...}``."""
    t = d.get("type")
    direction = {"left": 1, "right": -1, 1: 1, -1: -1}
    if t == "straight":
        return Straight(float(d["length_m"]))
    if t == "arc":
        return Arc(
            float(d["radius_m"]),
            math.radians(float(d["angle_deg"])),
            direction[d.get("direction", "left")],
        )
    if t == "slalom":
        return Slalom(
            int(d["n_cones"]),
            float(d["spacing_m"]),
            float(d.get("weave_m", 2.0)),
            direction[d.get("direction", "left")],
        )
    raise ValueError(f"unknown segment type {t!r} in {d}")


def close_with_dubins(
    segments: Sequence[Segment], radius_m: float, name: str, max_straight_m: float = math.inf
) -> SegmentTrack:
    """Close an open segment list into a loop with the shortest valid CSC Dubins path."""
    open_track = SegmentTrack(segments, closed=False, name=name)
    for word, t, p, q in dubins_csc(open_track.end_pose, open_track.start_pose, radius_m):
        if p <= max_straight_m:
            return SegmentTrack(
                list(segments) + dubins_segments(word, t, p, q, radius_m), closed=True, name=name
            )
    raise ValueError("no Dubins closure within the straight-length limit")


def close_by_straights(
    segments: Sequence[Segment], idx_a: int, idx_b: int, name: str, max_straight_m: float = math.inf
) -> SegmentTrack:
    """Close a loop exactly by re-solving the lengths of two non-parallel straights.

    The segment list must already turn by a multiple of 360°. Changing a straight's length
    translates everything after it along that straight's heading, so the end-position error
    is cancelled by a 2×2 linear solve.
    """
    segs = list(segments)
    for i in (idx_a, idx_b):
        if not isinstance(segs[i], Straight):
            raise ValueError(f"segment {i} is not a straight")
    open_track = SegmentTrack(segs, closed=False, name=name)
    x0, y0, t0 = open_track.start_pose
    x1, y1, t1 = open_track.end_pose
    dth = (t1 - t0 + math.pi) % TWO_PI - math.pi
    if abs(dth) > 1e-9:
        raise ValueError(
            f"segments must turn by a multiple of 360° (off by {math.degrees(dth):.3f}°)"
        )
    poses = open_track.segment_poses
    ua = (math.cos(poses[idx_a][2]), math.sin(poses[idx_a][2]))
    ub = (math.cos(poses[idx_b][2]), math.sin(poses[idx_b][2]))
    mat = np.array([[ua[0], ub[0]], [ua[1], ub[1]]])
    if abs(np.linalg.det(mat)) < 1e-6:
        raise ValueError("the two adjustable straights are parallel")
    da, db = np.linalg.solve(mat, np.array([x0 - x1, y0 - y1]))
    la, lb = segs[idx_a].length_m + da, segs[idx_b].length_m + db
    if min(la, lb) <= 0.0 or max(la, lb) > max_straight_m:
        raise ValueError(f"closure needs straights of {la:.1f} m and {lb:.1f} m (out of range)")
    segs[idx_a], segs[idx_b] = Straight(float(la)), Straight(float(lb))
    return SegmentTrack(segs, closed=True, name=name)


def track_from_dict(data: dict[str, Any], base_dir: Path | None = None) -> Track:
    """Build a track from a parsed track YAML mapping (see README for the format)."""
    kind = data.get("type", "segments")
    name = str(data.get("name", "track"))
    if kind == "generated":
        g = dict(data.get("generator", {}))
        return generate_layout(
            kind=g.get("kind", "autocross"),
            seed=int(g.get("seed", 0)),
            target_length_m=float(g.get("target_length_m", 900.0)),
            rules=LayoutRules.from_dict(g.get("rules")),
            name=name,
        )
    if kind == "segments":
        segs = [segment_from_dict(s) for s in data["segments"]]
        closure = data.get("closure", "none")
        max_straight = float(data.get("max_straight_m", math.inf))
        if closure == "dubins":
            return close_with_dubins(
                segs, float(data.get("closure_radius_m", 12.0)), name, max_straight
            )
        if closure == "adjust_straights":
            ia, ib = (int(i) for i in data["adjust"])
            return close_by_straights(segs, ia, ib, name, max_straight)
        if closure != "none":
            raise ValueError(f"unknown closure {closure!r}")
        return SegmentTrack(segs, closed=bool(data.get("closed", False)), name=name)
    if kind == "csv":
        path = resolve_path(data["path"], base_dir)
        x, y = read_xy_csv(path)
        return track_from_xy(
            x,
            y,
            closed=bool(data.get("closed", True)),
            smoothing_sigma_m=float(data.get("smoothing_sigma_m", 0.25)),
            ds_m=float(data.get("ds_m", 0.25)),
            name=name,
            reverse=bool(data.get("reverse", False)),
        )
    raise ValueError(f"unknown track type {kind!r}")


_TRACK_CACHE: dict[tuple[str, float], Track] = {}


def load_track(path: str | Path) -> Track:
    """Load a track YAML file (relative paths: cwd, then bundled ``configs/``). Cached."""
    p = resolve_path(path)
    key = (str(p), p.stat().st_mtime)
    if key not in _TRACK_CACHE:
        _TRACK_CACHE[key] = track_from_dict(load_yaml(p), p.parent)
    return _TRACK_CACHE[key]


TrackFactory = Callable[[], Track]

__all__ = [
    "LayoutRules",
    "SampledTrack",
    "check_layout",
    "close_by_straights",
    "close_with_dubins",
    "dubins_csc",
    "dubins_segments",
    "generate_layout",
    "load_track",
    "longest_straight",
    "min_gap",
    "sample_points",
    "segment_from_dict",
    "track_from_dict",
]
