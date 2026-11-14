"""E5: parallel paths give results identical to the serial ones."""

import numpy as np
import pytest

from fsk_lapsim.aero import components_from_config
from fsk_lapsim.analysis.compare import compare_candidates
from fsk_lapsim.analysis.sensitivity import CORE_PARAMS, sensitivity
from fsk_lapsim.analysis.sweep import sweep_cla_cda
from fsk_lapsim.config import default_data_dir, load_aero_components_config
from fsk_lapsim.events import EventSetup
from fsk_lapsim.models import AxleModel
from fsk_lapsim.parallel import default_workers, pmap
from fsk_lapsim.wing import read_wing_candidates


@pytest.fixture(scope="module")
def setup():
    return EventSetup.load()


def _square(x, ctx):
    return x * x + ctx["offset"]


def test_pmap_serial_and_parallel():
    items = list(range(7))
    assert pmap(_square, items, {"offset": 1}, workers=1) == [x * x + 1 for x in items]
    assert pmap(_square, items, {"offset": 1}, workers=2) == [x * x + 1 for x in items]
    assert pmap(_square, [3], {"offset": 0}, workers=4) == [9]
    assert 1 <= default_workers() <= 8


def test_sweep_parallel_identical(baseline, setup):
    a = sweep_cla_cda(baseline, [3.0, 4.0], [1.2, 1.6], setup, points=[("c", baseline)])
    b = sweep_cla_cda(baseline, [3.0, 4.0], [1.2, 1.6], setup, points=[("c", baseline)], workers=2)
    assert np.array_equal(a.autocross_s, b.autocross_s)
    assert np.array_equal(a.endurance_kwh, b.endurance_kwh)
    assert a.points.equals(b.points)


def test_sensitivity_and_compare_parallel_identical(baseline, setup):
    ev = ("skidpad", "autocross")
    a = sensitivity(baseline, setup, params=CORE_PARAMS[:2], events=ev)
    b = sensitivity(baseline, setup, params=CORE_PARAMS[:2], events=ev, workers=2)
    assert np.array_equal(a.table["d_dp"], b.table["d_dp"])
    comps = components_from_config(load_aero_components_config())
    cands = read_wing_candidates(default_data_dir() / "rear_wing_candidates_DUMMY.csv")[:2]
    ca = compare_candidates(baseline, comps, cands, setup, events=ev)
    cb = compare_candidates(baseline, comps, cands, setup, events=ev, workers=2)
    assert ca.table["autocross_time_s"].equals(cb.table["autocross_time_s"])


def test_reference_memoisation_matches_direct(baseline):
    m = AxleModel(baseline)
    v = np.array([0.0, 5.0, 12.0, 30.0, 38.0])
    grid = np.repeat(v[:, None], 7, axis=1)
    res, fpt = m.resistance_n(grid), m.drive_limit_n(grid)
    memo = m._ax_ref_straight(grid, res, fpt)
    direct = m._ax_ref_straight(v, m.resistance_n(v), m.drive_limit_n(v))
    assert np.allclose(memo, direct[:, None], atol=1e-12)
