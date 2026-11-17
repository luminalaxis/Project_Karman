"""Banked road and vertical curvature (crests / dips)."""

import numpy as np
import pytest

from aerolap.features import features_for_level
from aerolap.solvers.ggv import SolverOptions
from aerolap.solvers.qss import QSSSolver
from aerolap.solvers.transient import TransientOptions, TransientSolver
from aerolap.track import Track
from aerolap.vehicle import VehicleModel

COARSE = SolverOptions.coarse()


def _corner(bank_deg=0.0, direction="left"):
    trk = Track.from_segments([("straight", 30.0, 0.0, ""), ("corner", 80.0, 25.0, direction),
                               ("straight", 30.0, 0.0, "")], name="b", closed=False, ds=1.0)
    trk.bank = np.where(np.abs(trk.kappa) > 0, np.radians(bank_deg), 0.0)
    return trk


def _apex(res):
    return float(res.v[np.argmin(np.abs(res.s - 70.0))])


def test_zero_bank_changes_nothing(ev_cfg):
    car = VehicleModel(ev_cfg)
    a = QSSSolver(COARSE).solve(car, _corner(0.0), "flying")
    trk = _corner(0.0)
    trk.bank = None
    b = QSSSolver(COARSE).solve(car, trk, "flying")
    assert a.lap_time == pytest.approx(b.lap_time, rel=1e-9)


def test_banking_into_the_turn_is_faster_and_against_it_slower(ev_cfg):
    car = VehicleModel(ev_cfg)
    flat = QSSSolver(COARSE).solve(car, _corner(0.0), "flying")
    into = QSSSolver(COARSE).solve(car, _corner(8.0), "flying")
    against = QSSSolver(COARSE).solve(car, _corner(8.0, "right"), "flying")
    assert _apex(into) > _apex(flat) * 1.03
    assert _apex(against) < _apex(flat)
    assert "gravity_scale" in into.channels
    # steady state on a bank: v^2 / R = (mu g' + g sin) ... the speed gain is near the frictionless estimate
    assert into.channels["gravity_scale"].max() > 1.0


def _undulating(height):
    s = np.linspace(0.0, 140.0, 141)
    trk = _corner(0.0)
    trk.z = np.interp(trk.s, s, height * np.cos(np.pi * (s - 70.0) / 70.0))  # + crest / - dip at the apex
    trk.use_grade = False
    trk.use_vertical_curvature = True
    return trk


def test_dip_loads_the_tyres_and_a_crest_unloads_them(ev_cfg):
    car = VehicleModel(ev_cfg)
    flat = QSSSolver(COARSE).solve(car, _corner(0.0), "flying")
    dip = QSSSolver(COARSE).solve(car, _undulating(-0.5), "flying")
    crest = QSSSolver(COARSE).solve(car, _undulating(0.5), "flying")
    assert _apex(dip) > _apex(flat) > _apex(crest)


def test_transient_on_a_banked_corner(ev_cfg):
    car = VehicleModel(ev_cfg, features_for_level("L4"))
    res = TransientSolver(TransientOptions(iterations=1, qss=COARSE)).solve(car, _corner(8.0), "standing")
    flat = TransientSolver(TransientOptions(iterations=1, qss=COARSE)).solve(car, _corner(0.0), "standing")
    assert res.lap_time < flat.lap_time
