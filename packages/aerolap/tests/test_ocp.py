"""Optimal control: minimum-time line and the MPC driver (skipped without casadi)."""

import numpy as np
import pytest

pytest.importorskip("casadi")

from aerolap.features import features_for_level  # noqa: E402
from aerolap.ocp import min_time_line  # noqa: E402
from aerolap.solvers.ggv import SolverOptions  # noqa: E402
from aerolap.solvers.transient import TransientOptions, TransientSolver  # noqa: E402
from aerolap.track import Track  # noqa: E402
from aerolap.vehicle import VehicleModel  # noqa: E402

from .conftest import DATA  # noqa: E402

COARSE = SolverOptions.coarse()


def test_min_time_line_is_not_slower_than_minimum_curvature(ev_cfg):
    car = VehicleModel(ev_cfg)
    trk = Track.from_file(DATA / "tracks" / "fsae_autocross_example.csv").with_width(4.0)
    res = min_time_line(car, trk, opts=COARSE)
    curv, ocp = res.candidates
    assert np.isnan(ocp.weight)
    assert ocp.lap_time <= curv.lap_time + 1e-6
    assert np.all(ocp.alpha <= res.hi + 1e-6) and np.all(ocp.alpha >= res.lo - 1e-6)


def test_mpc_driver_completes_a_corner(ev_cfg):
    car = VehicleModel(ev_cfg, features_for_level("L4"))
    trk = Track.from_segments([("straight", 30.0, 0.0, ""), ("corner", 40.0, 20.0, "left"),
                               ("straight", 30.0, 0.0, "")], name="t", closed=False)
    res = TransientSolver(TransientOptions(iterations=0, driver="mpc", qss=COARSE)).solve(car, trk, "standing")
    assert res.lap_time < 1.3 * res.info["qss_lap_time_s"]
    assert np.max(np.abs(res.channels["path_error_m"])) < 0.6
