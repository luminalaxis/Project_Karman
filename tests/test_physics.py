import json
import math

import numpy as np
import pytest

from lapsim import Vehicle, simulate
from lapsim.study import coefficients
from lapsim.track import load_track, track_from_dict
from lapsim.vehicle import G

ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent


def circle(radius, n=400):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return track_from_dict({"x": (radius * np.cos(th)).tolist(),
                            "y": (radius * np.sin(th)).tolist(), "closed": True}, smooth_m=1.0)


def no_aero_car(**kw):
    v = Vehicle().with_changes(**{"aero.ClA": 0.0, "aero.CdA": 0.0, "rolling_resistance": 0.0,
                                  "tyre.load_sensitivity": 0.0})
    return v.with_changes(**kw)


@pytest.fixture(scope="module")
def autocross():
    return load_track(ROOT / "tracks" / "sample_autocross.json")


def test_circle_curvature():
    trk = circle(20.0)
    assert np.allclose(trk.kappa, 1 / 20.0, rtol=0.01)
    assert trk.length == pytest.approx(2 * np.pi * 20, rel=1e-3)


def test_circle_speed_matches_mu_g_r():
    r = 20.0
    car = no_aero_car()
    res = simulate(car, circle(r))
    expected = math.sqrt(car.tyre.mu_y * G * r)
    assert res.v.mean() == pytest.approx(expected, rel=0.01)
    assert res.lap_time == pytest.approx(2 * np.pi * r / expected, rel=0.01)


def test_drag_limited_top_speed():
    car = Vehicle().with_changes(**{"powertrain.max_rpm": 1e5, "rolling_resistance": 0.0})
    straight = track_from_dict({"segments": [{"type": "straight", "length": 3000}]})
    res = simulate(car, straight)
    p = car.powertrain
    v_eq = (p.max_power * p.efficiency / (0.5 * car.aero.rho * car.aero.CdA)) ** (1 / 3)
    assert res.v[-1] == pytest.approx(v_eq, rel=0.02)


def test_rpm_limited_top_speed():
    car = Vehicle()
    straight = track_from_dict({"segments": [{"type": "straight", "length": 1000}]})
    assert simulate(car, straight).v.max() <= car.powertrain.v_max + 1e-6


def test_downforce_helps_drag_hurts(autocross):
    car = Vehicle()
    base = simulate(car, autocross).lap_time
    assert simulate(car.scaled_aero(1.1, 1.0), autocross).lap_time < base
    assert simulate(car.scaled_aero(1.0, 1.1), autocross).lap_time > base
    c = coefficients(car, autocross)
    assert c.dt_pct_cl < 0 < c.dt_pct_cd


def test_balance_has_interior_optimum(autocross):
    car = Vehicle()
    times = [simulate(car.with_changes(**{"aero.balance_front": b}), autocross).lap_time
             for b in (0.1, 0.48, 0.9)]
    assert times[1] < times[0] and times[1] < times[2]


def test_discretisation_converges():
    path = ROOT / "tracks" / "sample_autocross.json"
    t1 = simulate(Vehicle(), load_track(path, ds=0.5)).lap_time
    t2 = simulate(Vehicle(), load_track(path, ds=0.25)).lap_time
    assert t1 == pytest.approx(t2, rel=0.005)


@pytest.mark.parametrize("layout", ["xy", "centerline", "cones", "curvature"])
def test_track_layouts(layout, tmp_path):
    r, n = 25.0, 300
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.column_stack([r * np.cos(th), r * np.sin(th)])
    data = {
        "xy": {"x": pts[:, 0].tolist(), "y": pts[:, 1].tolist()},
        "centerline": {"centerline": pts.tolist()},
        "cones": {"left": (pts * (r - 1.5) / r).tolist(),
                  "right": [{"x": x, "y": y} for x, y in pts * (r + 1.5) / r]},
        "curvature": {"s": [0, 2 * np.pi * r], "curvature": [1 / r, 1 / r]},
    }[layout]
    f = tmp_path / "t.json"
    f.write_text(json.dumps(data))
    trk = load_track(f)
    assert np.median(np.abs(trk.kappa)) == pytest.approx(1 / r, rel=0.02)


def test_unknown_layout_raises():
    with pytest.raises(ValueError, match="unrecognised"):
        track_from_dict({"foo": []})
