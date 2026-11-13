"""P7: sensitivity, equivalence, measurement priority, sweep and balance optimisation."""

import numpy as np
import pytest

from fsk_lapsim.analysis.balance_opt import optimise_balance
from fsk_lapsim.analysis.sensitivity import (
    CORE_PARAMS,
    EXTRA_PARAMS,
    ParamSpec,
    equivalence_table,
    measurement_priority,
    sensitivity,
)
from fsk_lapsim.analysis.sweep import sweep_cla_cda
from fsk_lapsim.analysis.uncertainty import Dist, Uncertainty
from fsk_lapsim.events import EventSetup, run_events
from fsk_lapsim.vehicle import with_param


@pytest.fixture(scope="module")
def setup():
    return EventSetup.load()


@pytest.fixture(scope="module")
def sens(baseline, setup):
    return sensitivity(
        baseline, setup, events=("acceleration", "skidpad", "autocross", "endurance")
    )


def test_core_params_cover_spec():
    paths = {p.path for p in CORE_PARAMS}
    assert paths == {
        "aero.cla_m2",
        "aero.cda_m2",
        "aero.balance_front",
        "mass_kg",
        "cg_height_m",
        "tire.mu0",
        "powertrain.max_power_w",
    }
    assert not paths & {p.path for p in EXTRA_PARAMS}


def test_sensitivity_signs(sens):
    p = sens.pivot(per_display_unit=False)
    assert p.loc["ClA", "autocross"] < 0 and p.loc["ClA", "endurance"] < 0
    assert p.loc["CdA", "autocross"] > 0 and p.loc["CdA", "endurance_energy"] > 0
    assert p.loc["Mass", "autocross"] > 0 and p.loc["Mass", "acceleration"] > 0
    assert p.loc["Tyre μ0", "skidpad"] < 0
    assert p.loc["Power cap", "acceleration"] < 0
    assert p.loc["Power cap", "skidpad"] == pytest.approx(0.0, abs=1e-9)
    assert p.loc["CG height", "skidpad"] > 0  # more lateral load transfer → less grip
    assert set(sens.table["event"]) == {
        "acceleration",
        "skidpad",
        "autocross",
        "endurance",
        "endurance_energy",
    }


def test_sensitivity_matches_manual_difference(baseline, setup, sens):
    up = run_events(with_param(baseline, "mass_kg", 325.0), setup, events=("autocross",))
    dn = run_events(with_param(baseline, "mass_kg", 315.0), setup, events=("autocross",))
    manual = (up.times()["autocross"] - dn.times()["autocross"]) / 10.0
    t = sens.table
    row = t[(t.param == "mass_kg") & (t.event == "autocross")].iloc[0]
    assert row["d_dp"] == pytest.approx(manual, rel=1e-9)
    assert row["per_display"] == pytest.approx(manual)  # display unit 1 kg


def test_equivalence_table(sens):
    eq = equivalence_table(sens, "autocross", "mass_kg").set_index("param")
    assert eq.loc["mass_kg", "equivalent_change"] == pytest.approx(1.0)
    assert eq.loc["aero.cla_m2", "equivalent_change"] < 0  # losing downforce ≡ adding mass
    assert eq.loc["aero.cda_m2", "equivalent_change"] > 0
    with pytest.raises(KeyError):
        equivalence_table(sens, "autocross", "wheelbase_m")


def test_measurement_priority_ranks_by_sigma_times_sensitivity(sens):
    unc = Uncertainty.load()
    pr = measurement_priority(sens, unc.sigmas())
    assert list(pr["score"]) == sorted(pr["score"], reverse=True)
    assert pr.index[0] == 1
    assert "powertrain.max_power_w" not in set(pr["param"])  # rule value, no σ given
    pr0 = measurement_priority(sens, {"mass_kg": 0.0, "tire.mu0": 0.1})
    assert pr0.iloc[0]["param"] == "tire.mu0" and pr0.iloc[1]["score"] == 0.0


def test_custom_runner_and_clipped_fraction(baseline):
    calls = []

    def runner(v):
        calls.append(v.aero.balance_front)
        return run_events(v, events=("skidpad",))

    spec = ParamSpec("aero.balance_front", 0.6, "Balance", "-", 0.01, "1 %")
    res = sensitivity(baseline, params=[spec], events=("skidpad",), runner=runner)
    assert min(calls) == 0.0 and max(calls) == 1.0  # clipped to [0, 1]
    assert len(res.table) == 1


def test_uncertainty_dist_and_sampling(baseline):
    rng = np.random.default_rng(0)
    d = Dist("uniform", low=-1.0, high=1.0, clip=(9.5, 10.6))
    x = d.sample(rng, 10.0, 2000)
    assert x.min() >= 9.5 and x.max() <= 10.6
    assert d.std == pytest.approx(2 / np.sqrt(12))
    assert Dist("uniform", low=1.2, high=1.9).sample(rng, 99.0, 50, absolute=True).max() < 1.9
    with pytest.raises(ValueError):
        Dist.from_dict({"dist": "cauchy"})
    unc = Uncertainty.load()
    vs = unc.sample_vehicles(baseline, 5, rng)
    assert len({v.mass_kg for v in vs}) == 5
    assert all(0.30 <= v.aero.balance_front <= 0.65 for v in vs)


def test_sweep_monotone_and_points(baseline, setup):
    sw = sweep_cla_cda(baseline, [2.5, 4.5], [1.0, 1.8], setup, points=[("current", baseline)])
    ac = sw.autocross_s
    assert ac[0, 1] < ac[0, 0] and ac[1, 1] < ac[1, 0]  # more ClA → faster
    assert ac[1, 0] > ac[0, 0]  # more CdA → slower
    assert sw.endurance_kwh[1, 0] > sw.endurance_kwh[0, 0]  # more CdA → more energy
    df = sw.to_frame()
    assert len(df) == 4 and set(df.columns) >= {"ClA_m2", "CdA_m2", "autocross_s"}
    assert sw.points.iloc[0]["label"] == "current"
    assert sw.points.iloc[0]["ClA_m2"] == baseline.aero.cla_m2


def test_balance_optimisation(baseline, setup):
    res = optimise_balance(baseline, setup, n_scan=8)
    assert 0.30 <= res.best_balance <= 0.65
    assert res.best_value <= res.base_value + 1e-9
    assert res.best_value <= res.scan.iloc[:, 1].min() + 1e-9
    assert set(res.limits_at_best.columns) == {"front", "rear", "drive", "total"}
    res2 = optimise_balance(
        baseline, setup, objective="endurance_lap", n_scan=5, bounds=(0.35, 0.55)
    )
    assert 0.35 <= res2.best_balance <= 0.55
    res3 = optimise_balance(baseline, setup, objective="combined", n_scan=4, bounds=(0.40, 0.50))
    assert res3.objective == "combined"
    with pytest.raises(ValueError):
        optimise_balance(baseline, setup, objective="skidpad")
