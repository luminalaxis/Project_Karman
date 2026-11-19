"""Tyre temperature window / thermal model, transient wheel slip, MF conicity / ply steer."""

import numpy as np
import pytest

from aerolap.config import load_vehicle_config, with_overrides
from aerolap.features import features_for_level
from aerolap.solvers.ggv import SolverOptions
from aerolap.solvers.qss import QSSSolver
from aerolap.solvers.transient import TransientOptions, TransientSolver
from aerolap.track import Track
from aerolap.vehicle import VehicleModel

from .conftest import DATA

CORNER = Track.from_segments([("straight", 40.0, 0.0, ""), ("corner", 60.0, 15.0, "left"),
                              ("straight", 40.0, 0.0, "")], name="t", closed=False)
COARSE = SolverOptions.coarse()


def _transient(cfg, track=CORNER, **kw):
    car = VehicleModel(cfg, features_for_level("L4"))
    return TransientSolver(TransientOptions(iterations=1, qss=COARSE, **kw)).solve(car, track, "standing")


def test_cold_tyres_are_slower(ev_cfg):
    warm = QSSSolver(COARSE).solve(VehicleModel(ev_cfg), CORNER, "standing")
    cold = QSSSolver(COARSE).solve(VehicleModel(with_overrides(ev_cfg, {"tires.temperature_C": 30.0})),
                                   CORNER, "standing")
    assert cold.lap_time > warm.lap_time
    assert VehicleModel(ev_cfg).grip_temp_factor(80.0) == pytest.approx(1.0)


def test_tyres_heat_up_and_cold_start_costs_time(ev_cfg):
    th = {"start_C": 25.0, "heat_capacity_J_per_K": 600.0}
    cold = _transient(with_overrides(ev_cfg, {"tires.thermal": th}))
    hot = _transient(with_overrides(ev_cfg, {"tires.thermal": {**th, "start_C": 80.0}}))
    t = cold.channels["tyre_temp_front_C"]
    assert t[-1] > t[0] + 1.0
    assert cold.lap_time > hot.lap_time


def test_wheel_dynamics_keep_the_steady_lap_and_show_slip(ev_cfg):
    plain = _transient(ev_cfg)
    wheels = _transient(ev_cfg, wheel_dynamics=True)
    assert wheels.lap_time == pytest.approx(plain.lap_time, rel=0.03)
    k = wheels.channels["slip_ratio_rear"]
    drive = wheels.channels["drive_force_N"] > 500.0
    assert np.nanmean(k[drive]) > 0.0  # driven wheels turn faster than the road under traction


def test_conicity_makes_the_car_pull(ev_cfg):
    straight = Track.from_segments([("straight", 120.0, 0.0, "")], name="s", closed=False)
    mf = load_vehicle_config(DATA / "vehicles" / "fsk2026_mf.yaml", warn=False)
    plain = _transient(mf, straight)
    conic = with_overrides(mf, {"tires.mf": {"PVY1": 0.03}})
    off = _transient(conic, straight)  # off by default: the .tir offsets do not change the run
    assert np.allclose(off.channels["steer_deg"], plain.channels["steer_deg"])
    pull = _transient(with_overrides(conic, {"tires.mf_offsets": True}), straight)
    assert abs(np.mean(pull.channels["steer_deg"][20:])) > abs(np.mean(plain.channels["steer_deg"][20:])) + 0.002


def test_fixed_tyre_temperature_also_limits_transient_cornering(ev_cfg):
    """A fixed cold tyre (grip factor f) corners like a tyre with mu x f - laterally too, not only in QSS."""
    skidpad = Track.circle(9.125, name="skidpad")
    cold = with_overrides(ev_cfg, {"tires.temperature_C": 30.0})
    f = VehicleModel(cold).tyre_grip
    low_mu = with_overrides(ev_cfg, {"tires.mu_x": ev_cfg["tires"]["mu_x"] * f, "tires.mu_y": ev_cfg["tires"]["mu_y"] * f})
    a = TransientSolver(TransientOptions(iterations=1, qss=COARSE)).solve(VehicleModel(cold, features_for_level("L4")),
                                                                          skidpad, "flying")
    b = TransientSolver(TransientOptions(iterations=1, qss=COARSE)).solve(VehicleModel(low_mu, features_for_level("L4")),
                                                                          skidpad, "flying")
    assert f < 1.0
    assert np.allclose(a.channels["steer_deg"], b.channels["steer_deg"], atol=1e-6)
