import math

import numpy as np
import pytest

from fsk_lapsim.track import (
    Arc,
    SampledTrack,
    SegmentTrack,
    Slalom,
    Straight,
    circle_track,
    read_xy_csv,
    straight_track,
    track_from_xy,
    with_name,
)
from fsk_lapsim.trackgen import (
    LayoutRules,
    check_layout,
    close_by_straights,
    dubins_csc,
    dubins_segments,
    generate_layout,
    load_track,
    longest_straight,
    min_gap,
    segment_from_dict,
    track_from_dict,
)

BUNDLED = [
    "tracks/autocross_generated.yaml",
    "tracks/endurance_generated.yaml",
    "tracks/template_autocross_A.yaml",
    "tracks/template_endurance_B.yaml",
    "tracks/template_endurance_C.yaml",
]


def _numeric_heading(track, s):
    x, y = track.xy(s)
    return np.unwrap(np.arctan2(np.gradient(y, s), np.gradient(x, s)))


@pytest.mark.parametrize(
    "seg",
    [
        Straight(12.0),
        Arc(8.0, math.radians(120), 1),
        Arc(5.0, math.pi, -1),
        Slalom(5, 10.0, 2.0, 1),
        Slalom(4, 7.5, 1.5, -1),
    ],
)
def test_segment_geometry_is_consistent(seg):
    tr = SegmentTrack([Straight(5.0), seg, Straight(5.0)])
    s = np.linspace(0.0, tr.length_m, 4001)
    th = _numeric_heading(tr, s)
    k_num = np.gradient(th, s)
    inner = (s > 6.0) & (s < tr.length_m - 6.0)
    assert np.allclose(k_num[inner], tr.curvature(s)[inner], atol=2e-3)
    # heading integral of curvature equals pose heading change (fine grid: κ jumps at joins)
    sf = np.linspace(0.0, tr.length_m, 200001)
    total = np.trapezoid(tr.curvature(sf), sf)
    assert total == pytest.approx(tr.end_pose[2], abs=1e-3)


def test_slalom_weave_and_exact_endpoint():
    sl = Slalom(5, 10.0, 2.0, 1)
    s = np.linspace(0.0, sl.length_m, 20001)
    _, y = sl.local_xy(s)
    assert y.max() - y.min() == pytest.approx(2.0, rel=1e-3)
    assert sl.heading(np.array([sl.length_m]))[0] == pytest.approx(0.0, abs=1e-12)
    # analytic endpoint agrees with fine numerical integration
    th = sl.heading(s)
    x_num = np.trapezoid(np.cos(th), s)
    assert sl.local_xy(np.array([sl.length_m]))[0][0] == pytest.approx(x_num, abs=1e-6)
    assert sl.kappa_max == pytest.approx(0.0998, rel=1e-2)  # R_min ~ 10 m for 10 m spacing
    with pytest.raises(ValueError):
        Slalom(4, 2.0, 5.0)
    with pytest.raises(ValueError):
        Arc(-1.0, 1.0)


def test_circle_closes_and_mirror():
    c = circle_track(9.125)
    pos, ang = c.closure_error()
    assert pos < 1e-9 and ang < 1e-9
    assert c.length_m == pytest.approx(2 * math.pi * 9.125)
    m = c.mirrored()
    s = np.linspace(0, c.length_m, 50)
    assert np.allclose(m.curvature(s), -c.curvature(s))
    assert np.allclose(m.xy(s)[1], -np.asarray(c.xy(s)[1]))
    assert with_name(c, "x").name == "x"


def test_discretize_closed_and_open():
    c = circle_track(10.0)
    s, k = c.discretize(0.25)
    assert s[0] == 0 and s[-1] == pytest.approx(c.length_m)
    assert np.diff(s).max() <= 0.25 + 1e-12
    assert k[-1] == k[0]
    st = straight_track(75.0)
    s, k = st.discretize(0.25)
    assert len(s) == 301 and not k.any()


def test_dubins_paths_reach_goal():
    rng = np.random.default_rng(3)
    for _ in range(200):
        start = (rng.uniform(-60, 60), rng.uniform(-60, 60), rng.uniform(-np.pi, np.pi))
        goal = (rng.uniform(-60, 60), rng.uniform(-60, 60), rng.uniform(-np.pi, np.pi))
        paths = dubins_csc(start, goal, 9.0)
        assert paths, "at least one CSC word always exists for distant poses"
        for word, t, p, q in paths:
            segs = dubins_segments(word, t, p, q, 9.0)
            if not segs:
                continue
            tr = SegmentTrack(segs, start_pose=start)
            e = tr.end_pose
            assert math.hypot(e[0] - goal[0], e[1] - goal[1]) < 1e-8
            assert abs((e[2] - goal[2] + math.pi) % (2 * math.pi) - math.pi) < 1e-8


@pytest.mark.parametrize(
    "kind,seed", [("autocross", 0), ("autocross", 5), ("endurance", 1), ("endurance", 4)]
)
def test_generator_respects_rules_and_is_deterministic(kind, seed):
    rules = LayoutRules()
    target = 900.0
    a = generate_layout(kind, seed, target)
    b = generate_layout(kind, seed, target)
    assert a.segments == b.segments
    report = check_layout(a, rules)
    assert report["issues"] == []
    assert 0.85 * target <= a.length_m <= 1.15 * target
    assert a.closed == (kind == "endurance")
    if a.closed:
        assert report["closure_error"][0] < 1e-6


def test_generator_different_seeds_differ_and_bad_kind():
    assert generate_layout("autocross", 1).segments != generate_layout("autocross", 2).segments
    with pytest.raises(ValueError):
        generate_layout("sprint", 1)


def test_generator_gives_up_with_impossible_rules():
    rules = LayoutRules(min_gap_m=1e4, gap_exclusion_m=1.0)
    with pytest.raises(RuntimeError):
        generate_layout("autocross", 0, 900.0, rules=rules, max_attempts=3)


@pytest.mark.parametrize("path", BUNDLED)
def test_bundled_tracks_respect_rules(path):
    tr = load_track(path)
    assert check_layout(tr, LayoutRules())["issues"] == []
    assert tr.length_m > 400


def test_fsk2026_reference_track_import():
    tr = load_track("tracks/fsk2026_endurance.yaml")
    assert tr.closed
    assert 835 < tr.length_m < 855  # cone2track export says 850.8 m before smoothing
    s, k = tr.discretize(0.25)
    turning = np.trapezoid(k, s)
    assert turning == pytest.approx(-2 * math.pi, abs=0.05)  # clockwise run
    summary = tr.summary()
    assert summary["min_radius_m"] < 4.0  # real FSK hairpin centreline is tighter than FS rules


def test_csv_import_of_noisy_circle_and_smoothing(tmp_path):
    rng = np.random.default_rng(0)
    th = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    r = 20.0
    x = r * np.cos(th) + rng.normal(0, 0.05, th.size)
    y = r * np.sin(th) + rng.normal(0, 0.05, th.size)
    p = tmp_path / "c.csv"
    p.write_text(
        "# noisy circle\nx,y,extra\n" + "\n".join(f"{a},{b},0" for a, b in zip(x, y, strict=True))
    )
    xr, yr = read_xy_csv(p)
    smooth = track_from_xy(xr, yr, closed=True, smoothing_sigma_m=0.05)
    rough = track_from_xy(xr, yr, closed=True, smoothing_sigma_m=0.0)
    s = np.linspace(0, smooth.length_m, 500)
    assert smooth.length_m == pytest.approx(2 * np.pi * r, rel=2e-3)
    assert np.median(smooth.curvature(s)) == pytest.approx(1 / r, rel=0.02)
    assert np.std(smooth.curvature(s)) < np.std(
        rough.curvature(np.linspace(0, rough.length_m, 500))
    )
    rev = track_from_xy(xr, yr, closed=True, smoothing_sigma_m=0.05, reverse=True)
    assert np.median(rev.curvature(s)) == pytest.approx(-1 / r, rel=0.02)
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError):
        read_xy_csv(bad)
    with pytest.raises(ValueError):
        track_from_xy([0, 1], [0, 1])


def test_open_csv_import_and_sampled_track_api():
    xs = np.linspace(0, 100, 101)
    ys = 0.0 * xs
    tr = track_from_xy(xs, ys, closed=False, smoothing_sigma_m=0.0)
    assert tr.length_m == pytest.approx(100.0, rel=1e-6)
    assert np.abs(tr.curvature([10.0, 50.0])).max() < 1e-6
    st = SampledTrack([0, 1, 2, 3], [0, 0.1, 0.1, 0])
    assert st.length_m == 3.0 and st.mirrored().curvature([1.0])[0] == pytest.approx(-0.1)
    with pytest.raises(ValueError):
        SampledTrack([0, 2, 1], [0, 0, 0])


def test_close_by_straights_errors():
    sq = [
        Straight(30),
        Arc(5, math.pi / 2),
        Straight(20),
        Arc(5, math.pi / 2),
        Straight(30),
        Arc(5, math.pi / 2),
        Straight(20),
        Arc(5, math.pi / 2),
    ]
    tr = close_by_straights(sq, 0, 2, "sq")
    assert tr.closure_error()[0] < 1e-9
    with pytest.raises(ValueError):
        close_by_straights(sq, 1, 2, "x")  # not a straight
    with pytest.raises(ValueError):
        close_by_straights(sq, 0, 4, "x")  # parallel
    with pytest.raises(ValueError):
        close_by_straights(sq[:-1], 0, 2, "x")  # turns only 270°
    with pytest.raises(ValueError):
        close_by_straights(sq, 0, 2, "x", max_straight_m=10.0)


def test_track_dict_errors_and_rules_parsing():
    with pytest.raises(ValueError):
        track_from_dict({"type": "magic"})
    with pytest.raises(ValueError):
        track_from_dict(
            {
                "type": "segments",
                "segments": [{"type": "straight", "length_m": 5}],
                "closure": "glue",
            }
        )
    with pytest.raises(ValueError):
        segment_from_dict({"type": "spiral"})
    with pytest.raises(ValueError):
        LayoutRules.from_dict({"bogus": 1})
    r = LayoutRules.from_dict({"turn_radius_m": [6, 20]})
    assert r.turn_radius_m == (6, 20)
    assert LayoutRules.from_dict(None) == LayoutRules()
    tr = track_from_dict(
        {
            "type": "segments",
            "closure": "dubins",
            "closure_radius_m": 10,
            "segments": [
                {"type": "straight", "length_m": 40},
                {"type": "arc", "radius_m": 10, "angle_deg": 180},
            ],
        }
    )
    assert tr.closed and tr.closure_error()[0] < 1e-8


def test_longest_straight_wraps_on_loops_and_gap():
    sq = [
        Straight(30),
        Arc(5, math.pi / 2),
        Straight(20),
        Arc(5, math.pi / 2),
        Straight(30),
        Arc(5, math.pi / 2),
        Straight(20),
        Arc(5, math.pi / 2),
        Straight(10),
    ]
    loop = close_by_straights(sq, 0, 2, "sq")  # E/W straights: 20 (+10 across s=0) and 30
    assert longest_straight(loop) == pytest.approx(30.0, abs=0.3)
    hairpin = SegmentTrack([Straight(60), Arc(5, math.pi), Straight(60)])
    assert min_gap(hairpin, exclusion_m=25.0) == pytest.approx(10.0, abs=1e-6)
    segs = [Straight(30), Arc(5, math.pi), Straight(10), Arc(5, math.pi), Straight(25)]
    assert longest_straight(SegmentTrack(segs, closed=True)) == pytest.approx(55.0, abs=0.3)
    assert longest_straight(SegmentTrack(segs, closed=False)) == pytest.approx(30.0, abs=0.3)
    assert longest_straight(straight_track(50)) == pytest.approx(50)
    assert check_layout(straight_track(120.0), LayoutRules())["issues"]
