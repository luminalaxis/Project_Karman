"""E1: Monte Carlo uncertainty propagation."""

import numpy as np
import pytest

from fsk_lapsim.aero import components_from_config
from fsk_lapsim.analysis.compare import compare_candidates
from fsk_lapsim.analysis.montecarlo import MC_OPTIONS, run_montecarlo
from fsk_lapsim.analysis.uncertainty import Dist, Uncertainty
from fsk_lapsim.config import default_data_dir, load_aero_components_config
from fsk_lapsim.events import EventSetup
from fsk_lapsim.vehicle import apply_component_change
from fsk_lapsim.wing import read_wing_candidates


@pytest.fixture(scope="module")
def comps():
    return components_from_config(load_aero_components_config())


@pytest.fixture(scope="module")
def cands():
    return read_wing_candidates(default_data_dir() / "rear_wing_candidates_DUMMY.csv")


@pytest.fixture(scope="module")
def setup():
    return EventSetup.load()


@pytest.fixture(scope="module")
def mc(baseline, setup, comps, cands):
    return run_montecarlo(baseline, 6, 1, setup, candidates=cands[:3], components=comps)


def test_mc_shapes_and_stats(mc):
    assert mc.n == 6 and len(mc.inputs) == 6
    assert set(mc.times["variant"]) == {"base", "RW01", "RW02", "RW03"}
    assert mc.matrix("autocross").shape == (6, 4)
    d = mc.distribution().set_index("metric")
    assert d.loc["autocross", "p05_s"] <= d.loc["autocross", "p50_s"] <= d.loc["autocross", "p95_s"]
    assert d.loc["autocross", "std_s"] > 0.1  # μ0 and k uncertainty dominate
    rs = mc.rank_stats("autocross")
    assert rs["p_fastest"].sum() == pytest.approx(1.0)
    assert rs["p_same_rank"].between(0, 1).all()
    pf = mc.pairwise_flip("endurance_lap")
    assert len(pf) == 3 and pf["p_flip"].between(0, 1).all()
    assert 0.0 <= mc.p_ranking_unchanged() <= 1.0


def test_mc_is_reproducible(baseline, setup, comps, cands, mc):
    again = run_montecarlo(baseline, 6, 1, setup, candidates=cands[:3], components=comps)
    assert np.allclose(again.matrix("autocross"), mc.matrix("autocross"))


def test_mc_nominal_matches_compare(baseline, setup, comps, cands, mc):
    cmp = compare_candidates(
        baseline, comps, cands[:3], setup, MC_OPTIONS, events=("autocross",), include_baseline=True
    )
    t = cmp.table.set_index("config_id")
    assert mc.nominal.loc["base", "autocross"] == pytest.approx(
        t.loc["baseline", "autocross_time_s"]
    )
    assert mc.nominal.loc["RW02", "autocross"] == pytest.approx(t.loc["RW02", "autocross_time_s"])


def test_zero_uncertainty_keeps_nominal_ranking(baseline, setup, comps, cands):
    unc = Uncertainty(parameters={}, wing_conversion={})
    res = run_montecarlo(baseline, 3, 0, setup, unc, cands[:3], comps)
    assert res.p_ranking_unchanged("autocross") == 1.0
    assert (res.rank_stats("autocross")["p_same_rank"] == 1.0).all()
    assert (res.pairwise_flip("autocross")["p_flip"] == 0.0).all()
    assert np.allclose(res.matrix("autocross")["base"], res.nominal.loc["base", "autocross"])


def test_mc_parallel_matches_serial(baseline, setup):
    unc = Uncertainty(parameters={"mass_kg": Dist("normal", sigma=5.0)})
    a = run_montecarlo(baseline, 3, 7, setup, unc, workers=1)
    b = run_montecarlo(baseline, 3, 7, setup, unc, workers=2)
    assert np.allclose(a.matrix("autocross"), b.matrix("autocross"))
    assert a.candidates() == [] and a.rank_stats().empty and a.p_ranking_unchanged() == 1.0


def test_mc_errors(baseline, setup, cands, comps):
    with pytest.raises(ValueError):
        run_montecarlo(baseline, 1, 0, setup)
    with pytest.raises(ValueError):
        run_montecarlo(baseline, 3, 0, setup, candidates=cands[:1])
    with pytest.raises(KeyError):
        run_montecarlo(
            baseline, 3, 0, setup, candidates=cands[:1], components=comps, component_name="canard"
        )


def test_component_change_is_a_delta_on_vehicle_aero(baseline, comps):
    from fsk_lapsim.vehicle import with_param

    bumped = with_param(baseline, "aero.cla_m2", 4.0)  # e.g. a Monte Carlo sample
    same = apply_component_change(bumped, comps, comps)
    assert same.aero.cla_m2 == pytest.approx(4.0)
    assert same.aero.balance_front == pytest.approx(bumped.aero.balance_front)


def test_report_mc_section_renders(mc):
    from types import SimpleNamespace

    from fsk_lapsim.report import _mc_section

    html = _mc_section(SimpleNamespace(extras={"mc": mc}, comparison=SimpleNamespace(dummy=True)))
    assert "몬테카를로" in html and "DUMMY" in html and "data:image/png;base64," in html


def test_mc_retrim_keeps_balance_spread_small(baseline, setup, comps, cands):
    unc = Uncertainty(parameters={}, wing_conversion={})
    res = run_montecarlo(baseline, 2, 0, setup, unc, cands[:2], comps, retrim_front=True)
    plain = run_montecarlo(baseline, 2, 0, setup, unc, cands[:2], comps)
    assert res.nominal.loc["RW01", "autocross"] != plain.nominal.loc["RW01", "autocross"]
