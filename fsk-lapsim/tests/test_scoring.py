"""E4: points conversion layer (placeholder formulas)."""

import pandas as pd
import pytest

from fsk_lapsim.analysis.sensitivity import CORE_PARAMS, sensitivity
from fsk_lapsim.events import EventSetup, run_events
from fsk_lapsim.scoring import (
    EventFormula,
    ScoringModel,
    comparison_points,
    delta_points,
    sensitivity_points,
)
from fsk_lapsim.vehicle import with_param


@pytest.fixture(scope="module")
def setup():
    return EventSetup.load()


@pytest.fixture(scope="module")
def base(baseline, setup):
    return run_events(baseline, setup)


def test_time_ratio_formula_endpoints():
    f = EventFormula(
        "autocross", "time_ratio", a=95.5, b=4.5, exponent=1, tmax_factor=1.25, t_min_s=50.0
    )
    assert f.t_min(999.0) == 50.0
    assert f.time_score(50.0, 50.0) == pytest.approx(100.0)
    assert f.time_score(40.0, 50.0) == pytest.approx(100.0)  # cannot beat the best time
    assert f.time_score(62.5, 50.0) == pytest.approx(4.5)
    assert f.time_score(80.0, 50.0) == pytest.approx(4.5)
    sq = EventFormula(
        "skidpad", "time_ratio", a=71.5, b=3.5, exponent=2, tmax_factor=1.25, t_min_ratio=0.95
    )
    assert sq.t_min(5.0) == pytest.approx(4.75)
    mid = sq.time_score(5.0, 4.75)
    assert 3.5 < mid < 75.0
    with pytest.raises(ValueError):
        EventFormula("x", "time_ratio", a=1.0).t_min(1.0)
    with pytest.raises(ValueError):
        EventFormula.from_dict("x", {"formula": "lottery", "a": 1})


def test_efficiency_score_clipped():
    f = EventFormula("efficiency", "efficiency_factor", a=100.0, ef_min=0.5, ef_max=1.0)
    assert f.efficiency_score(100.0, 10.0, 100.0, 10.0) == pytest.approx(100.0)
    assert f.efficiency_score(100.0, 10.0, 75.0, 10.0) == pytest.approx(50.0)
    assert f.efficiency_score(100.0, 30.0, 50.0, 10.0) == 0.0


def test_model_table_and_points_per_second(base):
    sm = ScoringModel.load()
    assert sm.placeholder
    with pytest.raises(RuntimeError):
        sm.score(base)
    cal = sm.calibrate(base)
    t = cal.table(base).set_index("event")
    assert set(t.index) == {
        "acceleration",
        "skidpad",
        "autocross",
        "endurance",
        "efficiency",
        "total",
    }
    assert t.loc["total", "points"] == pytest.approx(t["points"].drop("total").sum())
    assert (t["points_per_0.1s"].dropna() < 0).all()
    assert t.loc["autocross", "t_min_s"] == pytest.approx(0.95 * base.times()["autocross"])


def test_delta_and_sensitivity_points(baseline, setup, base):
    sm = ScoringModel.load()
    better = run_events(with_param(baseline, "tire.mu0", 1.5), setup)
    d = delta_points(sm, base, {"mu+": better}).iloc[0]
    assert d["d_autocross"] > 0 and d["d_skidpad"] > 0
    sens = sensitivity(baseline, setup, params=CORE_PARAMS[:2])
    sp = sensitivity_points(sm, sens.table, base).set_index("param")
    assert sp.loc["aero.cla_m2", "d_autocross_pts"] > 0
    assert sp.loc["aero.cda_m2", "d_endurance_pts"] < 0
    assert sp.loc["aero.cda_m2", "d_efficiency_pts"] < 0


def test_comparison_points_reference_is_zero():
    sm = ScoringModel.load()
    table = pd.DataFrame(
        {
            "config_id": ["A", "B"],
            "autocross_time_s": [55.0, 54.9],
            "skidpad_time_s": [5.1, 5.1],
            "d_autocross_s": [0.0, -0.1],
            "rank_autocross": [2, 1],
        }
    )
    cp = comparison_points(sm, table, "A").set_index("config_id")
    assert cp.loc["A", "d_total_pts"] == 0.0
    assert cp.loc["B", "d_autocross_pts"] > 0 and cp.loc["B", "d_skidpad_pts"] == 0.0
