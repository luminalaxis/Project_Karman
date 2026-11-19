"""Aero: device wake interaction, Reynolds scaling, signed roll / yaw map, unsteady lag / hysteresis."""

import numpy as np
import pytest

from aerolap.config import load_vehicle_config, with_overrides
from aerolap.features import features_for_level
from aerolap.solvers.ggv import SolverOptions
from aerolap.solvers.transient import TransientOptions, TransientSolver
from aerolap.track import Track
from aerolap.vehicle import VehicleModel

from .conftest import DATA

CAR = DATA / "vehicles" / "fsk2026.yaml"


def _static(cfg, speed=None):
    car = VehicleModel(cfg)
    g = car.aero.geometry
    return car.aero.coefficients(car.default_mode, g.rh_f0, g.rh_r0, 0.0, 0.0, speed)


def test_rear_wing_in_the_front_wing_wake_loses_downforce():
    cfg = load_vehicle_config(CAR, warn=False)
    a = _static(cfg)
    b = _static(with_overrides(cfg, {"aero.components.rear.wake_of": "front", "aero.components.rear.wake_factor": 0.2}))
    assert float(b[1]) < float(a[1])  # rear axle downforce
    assert float(b[0]) == pytest.approx(float(a[0]), rel=0.05)


def test_wake_chains_cycles_and_switched_off_upstream_devices():
    from aerolap.config import ConfigError

    cfg = load_vehicle_config(CAR, warn=False)
    # side (listed before rear) sits in the rear's wake, the rear in the front's: evaluated front, rear, side
    chain = _static(with_overrides(cfg, {"aero.components.side.wake_of": "rear", "aero.components.rear.wake_of": "front"}))
    assert np.isfinite(float(chain[0] + chain[1]))
    with pytest.raises(ConfigError, match="wake_of"):
        VehicleModel(with_overrides(cfg, {"aero.components.front.wake_of": "rear", "aero.components.rear.wake_of": "front"}))
    # front wing switched off: the rear wing simply has no wake to sit in
    off = {"aero.components.front.enabled": False}
    a = _static(with_overrides(cfg, off))
    b = _static(with_overrides(cfg, {**off, "aero.components.rear.wake_of": "front"}))
    assert float(b[1]) == pytest.approx(float(a[1]))


def test_reynolds_scaling(ev_cfg):
    cfg = with_overrides(ev_cfg, {"aero.reynolds": {"cl_exponent": 0.05, "cd_exponent": -0.1, "ref_speed_kmh": 60}})
    lo = _static(cfg, 60 / 3.6)
    hi = _static(cfg, 120 / 3.6)
    assert float(hi[0] + hi[1]) == pytest.approx(float(lo[0] + lo[1]) * 2 ** 0.05, rel=1e-6)
    assert float(hi[2]) == pytest.approx(float(lo[2]) * 2 ** -0.1, rel=1e-6)
    assert float(_static(ev_cfg, 120 / 3.6)[2]) == pytest.approx(float(_static(ev_cfg)[2]))


def test_symmetric_map_ignores_the_sign():
    cfg = load_vehicle_config(CAR, warn=False)
    car = VehicleModel(cfg)
    g = car.aero.geometry
    pos = car.aero.coefficients(0, g.rh_f0, g.rh_r0, 1.0, 3.0)
    neg = car.aero.coefficients(0, g.rh_f0, g.rh_r0, -1.0, -3.0)
    assert np.allclose(pos[:3], neg[:3])


def test_unsteady_lag_and_hysteresis_run(ev_cfg):
    cfg = with_overrides(ev_cfg, {"aero.unsteady_tau_s": 0.1, "aero.hysteresis_mm": 3.0})
    track = Track.from_segments([("straight", 60.0, 0.0, ""), ("corner", 40.0, 20.0, "left"),
                                 ("straight", 40.0, 0.0, "")], name="t", closed=False)
    car = VehicleModel(cfg, features_for_level("L4"))
    res = TransientSolver(TransientOptions(iterations=1, qss=SolverOptions.coarse())).solve(car, track, "standing")
    base = TransientSolver(TransientOptions(iterations=1, qss=SolverOptions.coarse())).solve(
        VehicleModel(ev_cfg, features_for_level("L4")), track, "standing")
    assert res.lap_time == pytest.approx(base.lap_time, rel=0.03)
    assert not np.allclose(res.channels["cla"], base.channels["cla"])
