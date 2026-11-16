"""Nonlinear suspension: bump stops, floor contact, bilinear dampers, roll-centre migration,
bump steer and camber curves."""

import numpy as np
import pytest

from aerolap.config import with_overrides
from aerolap.features import features_for_level
from aerolap.solvers.ggv import SolverOptions
from aerolap.solvers.transient import TransientOptions, TransientSolver
from aerolap.track import Track
from aerolap.vehicle import FO, FI, VehicleModel

V = 45.0  # m/s, lots of downforce


def _state(cfg, ay=0.0, ax=0.0, v=V):
    car = VehicleModel(cfg)
    return car, car.solve_state(v, ax, ay)


def test_far_bump_stop_changes_nothing(ev_cfg):
    _, a = _state(ev_cfg, ay=12.0)
    _, b = _state(with_overrides(ev_cfg, {"suspension.front.bump_stop_gap_mm": 500.0,
                                          "suspension.front.bump_stop_rate_N_per_mm": 200.0}), ay=12.0)
    assert np.allclose(a.fz, b.fz, atol=1e-3)
    assert float(a.rh_f) == pytest.approx(float(b.rh_f), abs=1e-7)
    assert float(a.roll) == pytest.approx(float(b.roll), abs=1e-7)


def test_bump_stop_holds_the_ride_height_up(ev_cfg):
    car, free = _state(ev_cfg)
    travel = car.front.static_ride_height - float(free.rh_f)
    stop = {"suspension.front.bump_stop_gap_mm": 0.3 * travel * 1000.0,
            "suspension.front.bump_stop_rate_N_per_mm": 300.0}
    _, held = _state(with_overrides(ev_cfg, stop))
    assert float(held.rh_f) > float(free.rh_f) + 0.3 * travel * 0.5
    # in roll the outer wheel hits the stop: less roll
    _, r_free = _state(ev_cfg, ay=14.0)
    _, r_held = _state(with_overrides(ev_cfg, stop), ay=14.0)
    assert float(r_held.roll) < float(r_free.roll)


def test_floor_contact_unloads_the_tyres_and_drags(ev_cfg):
    low = {"suspension.front.static_ride_height_mm": 8.0, "suspension.rear.static_ride_height_mm": 10.0}
    car0, s0 = _state(with_overrides(ev_cfg, low))
    car1, s1 = _state(with_overrides(ev_cfg, {**low, "aero.plank_rate_N_per_mm": 2000.0}))
    assert float(s1.plank) > 100.0
    assert float(s1.fz.sum()) == pytest.approx(float(s0.fz.sum()) - float(s1.plank), rel=0.02)
    assert float(car1.coast_ax(s1)) < float(car0.coast_ax(s0))


def test_bilinear_damper_and_rebound(ev_cfg):
    cfg = with_overrides(ev_cfg, {"suspension.front.damper_high_speed_Ns_per_mm": 0.5,
                                  "suspension.front.damper_knee_mm_s": 40.0,
                                  "suspension.front.damper_rebound_factor": 1.5})
    ax = VehicleModel(cfg).front
    mr = ax.motion_ratio
    knee_w = 0.04 / mr
    c_low = 1000.0
    v = 3.0 * knee_w
    assert ax.damper_force(0.5 * knee_w, c_low) == pytest.approx(c_low * 0.5 * knee_w)
    assert ax.damper_force(v, c_low) == pytest.approx(c_low * knee_w + 500.0 * mr * mr * (v - knee_w))
    assert ax.damper_force(-v, c_low) == pytest.approx(-1.5 * ax.damper_force(v, c_low))


def test_camber_curve_matches_the_linear_gain(ev_cfg):
    gain = float(ev_cfg["suspension"]["front"]["camber_gain_deg_per_mm"]) or -0.02
    lin = with_overrides(ev_cfg, {"suspension.front.camber_gain_deg_per_mm": gain})
    tab = with_overrides(ev_cfg, {"suspension.front.camber_curve": [[-60.0, -60.0 * gain], [60.0, 60.0 * gain]]})
    _, a = _state(lin, ay=12.0)
    _, b = _state(tab, ay=12.0)
    assert np.allclose(a.camber, b.camber, atol=1e-9)


def test_roll_centre_migration_changes_the_load_transfer(ev_cfg):
    _, a = _state(ev_cfg, ay=14.0)
    _, b = _state(with_overrides(ev_cfg, {"suspension.front.roll_center_gain_mm_per_mm": -2.0,
                                          "suspension.rear.roll_center_gain_mm_per_mm": -2.0}), ay=14.0)
    # the car sits lower at speed -> roll centres drop -> more roll
    assert float(b.roll) > float(a.roll)


def test_rear_bump_steer_changes_sideslip(ev_cfg):
    _, a = _state(ev_cfg, ay=12.0, v=20.0)
    _, b = _state(with_overrides(ev_cfg, {"suspension.rear.bump_steer_deg_per_mm": 0.01}), ay=12.0, v=20.0)
    assert float(b.yaw) != pytest.approx(float(a.yaw), abs=1e-5)
    assert float(b.fz[FO]) == pytest.approx(float(a.fz[FO]), rel=0.01)
    assert float(b.fz[FI]) == pytest.approx(float(a.fz[FI]), rel=0.01)


def test_transient_runs_with_the_nonlinear_suspension(ev_cfg):
    cfg = with_overrides(ev_cfg, {"suspension.front.bump_stop_gap_mm": 8.0,
                                  "suspension.front.bump_stop_rate_N_per_mm": 150.0,
                                  "suspension.front.damper_high_speed_Ns_per_mm": 0.6,
                                  "suspension.rear.damper_rebound_factor": 1.4,
                                  "suspension.rear.bump_steer_deg_per_mm": 0.005,
                                  "aero.plank_rate_N_per_mm": 1000.0})
    car = VehicleModel(cfg, features_for_level("L4"))
    track = Track.from_segments([("straight", 40.0, 0.0, ""), ("corner", 40.0, 20.0, "left"),
                                 ("straight", 40.0, 0.0, "")], name="t", closed=False)
    res = TransientSolver(TransientOptions(iterations=1, qss=SolverOptions.coarse())).solve(car, track, "standing")
    assert np.isfinite(res.lap_time) and res.lap_time > 0
