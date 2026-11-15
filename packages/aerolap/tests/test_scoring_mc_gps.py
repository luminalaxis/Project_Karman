"""Points conversion, Monte Carlo statistics and GPS laps (ported from fsk-lapsim)."""

import math

import numpy as np

from aerolap.analysis.montecarlo import MonteCarloResult
from aerolap.analysis.scoring import points, references
from aerolap.cli import main
from aerolap.gps import R_EARTH, _nmea, parse_nmea, split_laps


def test_points():
    refs = references({"autocross": 60.0, "endurance": 1500.0}, 6.0)
    assert math.isclose(points({"autocross": 57.0}, None, refs)["autocross"], 100.0)  # T_min -> a + b
    assert points({"autocross": 80.0}, None, refs)["autocross"] == 4.5  # beyond T_max -> b
    assert points({"autocross": 59.0}, None, refs)["autocross"] > points({"autocross": 60.0}, None, refs)["autocross"]


def test_monte_carlo_statistics():
    r = MonteCarloResult(["a", "b"], np.array([1.0, 2.0]),
                         np.array([[1.0, 2.0], [2.5, 2.0], [1.0, 3.0], [1.2, 1.1]]))
    txt = r.report("t", "s")
    assert "P(whole nominal order kept) = 50.0%" in txt and "P(flip)  50.0%" in txt


def _circle_log(path, laps=3.2, r=50.0, v=10.0, lat0=37.6, lon0=127.1):
    lines = []
    for k in range(int(laps * 2 * math.pi * r / v * 10)):
        t = k / 10
        th = v * t / r - math.pi / 2
        x, y = r * math.cos(th), r * math.sin(th) + r
        lines.append(_nmea(43_000 + t, lat0 + math.degrees(y / R_EARTH),
                           lon0 + math.degrees(x / (R_EARTH * math.cos(math.radians(lat0)))), v))
    lines.append(lines[5][:-2] + "00")  # bad checksum: skipped
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def test_gps_laps_and_cli(tmp_path):
    log = _circle_log(tmp_path / "log.nmea")
    laps = split_laps(parse_nmea(log.read_text(encoding="utf-8").splitlines()))
    assert len(laps) == 3
    assert all(abs(lap.time_s - 2 * math.pi * 5.0) < 0.15 for lap in laps)
    trace = tmp_path / "trace.csv"
    s = np.linspace(0, 2 * math.pi * 50, 50)
    np.savetxt(trace, np.c_[s, s / 11.0, np.full_like(s, 11.0)], delimiter=",", header="s_m,t_s,v_mps", comments="")
    out = tmp_path / "out"
    assert main(["gps", str(log), "--trace", str(trace), "--track-out", str(tmp_path / "t.csv"), "--out", str(out)]) == 0
    txt = (out / "gps.txt").read_text(encoding="utf-8")
    assert "3 laps" in txt and "mean (sim - GPS) +3.60 km/h" in txt
    assert (out / "gps_vs_sim.png").exists() and (tmp_path / "t.csv").exists()
