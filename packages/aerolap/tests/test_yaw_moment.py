"""Yaw moments: LSD torque bias, torque vectoring, aero side force."""

import pytest

from aerolap.config import ConfigError, with_overrides
from aerolap.features import features_for_level
from aerolap.solvers.ggv import SolverOptions
from aerolap.solvers.transient import TransientOptions, TransientSolver
from aerolap.track import Track
from aerolap.vehicle import VehicleModel


def _exit(cfg):  # corner exit: lateral + traction
    return VehicleModel(cfg).solve_state(15.0, 3.0, 12.0)


def test_lsd_locking_one_is_an_open_diff(ev_cfg):
    open_ = _exit(with_overrides(ev_cfg, {"powertrain.differential": "open"}))
    tbr1 = _exit(with_overrides(ev_cfg, {"powertrain.lsd_locking": 1.0}))
    assert float(tbr1.u_rear) == pytest.approx(float(open_.u_rear), rel=1e-6)
    assert float(tbr1.understeer_index) == pytest.approx(float(open_.understeer_index), rel=1e-6)


def test_lsd_bias_adds_understeer_and_torque_vectoring_removes_it(ev_cfg):
    base = _exit(ev_cfg)
    lsd = _exit(with_overrides(ev_cfg, {"powertrain.lsd_locking": 3.0}))
    tv = _exit(with_overrides(ev_cfg, {"powertrain.ev.n_motors": 2, "powertrain.torque_vectoring": True}))
    assert float(lsd.understeer_index) > float(base.understeer_index)
    assert float(tv.understeer_index) < float(base.understeer_index)


def test_aero_side_force_behind_the_cg_stabilises(ev_cfg):
    car = VehicleModel(with_overrides(ev_cfg, {"aero.side_force_area_m2_per_deg": 0.05,
                                               "aero.side_force_x_mm": 1500.0}))
    fs, mz = car.aero_side(30.0, 0.05)
    assert fs > 0 and mz > 0  # outward force behind the CG yaws the nose into the turn (weathercock)
    ahead = VehicleModel(with_overrides(ev_cfg, {"aero.side_force_area_m2_per_deg": 0.05,
                                                 "aero.side_force_x_mm": 0.0}))
    assert ahead.aero_side(30.0, 0.05)[1] < 0


def test_torque_vectoring_needs_two_motors(ev_cfg):
    with pytest.raises(ConfigError):
        with_overrides(ev_cfg, {"powertrain.torque_vectoring": True})


def test_transient_with_yaw_moments_finishes(ev_cfg):
    cfg = with_overrides(ev_cfg, {"powertrain.ev.n_motors": 2, "powertrain.torque_vectoring": True,
                                  "aero.side_force_area_m2_per_deg": 0.03, "aero.side_force_x_mm": 1300.0})
    car = VehicleModel(cfg, features_for_level("L4"))
    track = Track.from_segments([("straight", 40.0, 0.0, ""), ("corner", 40.0, 20.0, "left"),
                                 ("straight", 40.0, 0.0, "")], name="t", closed=False)
    res = TransientSolver(TransientOptions(iterations=1, qss=SolverOptions.coarse())).solve(car, track, "standing")
    assert res.lap_time > 0
