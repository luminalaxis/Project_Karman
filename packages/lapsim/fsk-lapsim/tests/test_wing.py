"""P5: aero components, 2D→3D conversion, rear-wing candidates and comparison."""

import math

import pytest

from fsk_lapsim.aero import components_from_config, replace_component, synthesize
from fsk_lapsim.analysis.compare import compare_candidates
from fsk_lapsim.config import default_data_dir, load_aero_components_config
from fsk_lapsim.events import EventSetup
from fsk_lapsim.wing import (
    ROUGH_NOTE,
    WingCandidate,
    WingConversion,
    candidate_component,
    candidate_table,
    candidate_vehicle,
    read_wing_candidates,
    retrim_component,
    wing_2d_to_3d,
)

DUMMY = default_data_dir() / "rear_wing_candidates_DUMMY.csv"


@pytest.fixture(scope="module")
def comps():
    return components_from_config(load_aero_components_config())


@pytest.fixture(scope="module")
def rear(comps):
    return next(c for c in comps if c.name == "rear_wing")


@pytest.fixture(scope="module")
def setup():
    return EventSetup.load()


def test_2d_to_3d_formula():
    w = wing_2d_to_3d(3.0, 0.1, chord_m=0.5, span_m=1.0, ar_eff=3.0, k_lift=0.6, oswald_e=0.8)
    assert w.cl_3d == pytest.approx(1.8)
    assert w.cd_induced == pytest.approx(1.8**2 / (math.pi * 0.8 * 3.0))
    assert w.cd_3d == pytest.approx(0.1 + w.cd_induced)
    assert w.cla_m2 == pytest.approx(1.8 * 0.5)
    assert w.cda_m2 == pytest.approx(w.cd_3d * 0.5)
    d = wing_2d_to_3d(3.0, 0.1, chord_m=0.5, span_m=1.0)  # defaults
    assert d.ar_eff == pytest.approx(2.0 * 1.5)
    assert d.k_lift == pytest.approx(3.0 / 5.0)
    assert wing_2d_to_3d(3.2, 0.1, 0.5, 1.0).cla_m2 > d.cla_m2
    assert "대략적" in ROUGH_NOTE
    with pytest.raises(ValueError):
        wing_2d_to_3d(3.0, 0.1, 0.0, 1.0)


def test_dummy_csv_is_marked_and_parsed():
    cands = read_wing_candidates(DUMMY)
    assert len(cands) == 8
    assert all(c.is_dummy for c in cands)
    assert "DUMMY" in DUMMY.name
    assert {"flap1_aoa_deg", "flap2_aoa_deg", "flap2_gap_mm"} <= set(cands[0].params)
    assert cands[0].label() == "RW01 (F1A/F2A)"
    # 2 flap1 x 2 flap2 combinations present
    assert {(c.flap1_id, c.flap2_id) for c in cands} == {
        ("F1A", "F2A"),
        ("F1A", "F2B"),
        ("F1B", "F2A"),
        ("F1B", "F2B"),
    }


def test_csv_validation(tmp_path):
    p = tmp_path / "c.csv"
    p.write_text("config_id,flap1_id,flap2_id,Cl_2d,Cd_2d,chord_ref_m\nA,a,b,2,0.1,0.5\n")
    with pytest.raises(ValueError, match="missing"):
        read_wing_candidates(p)
    p.write_text(
        "config_id,flap1_id,flap2_id,Cl_2d,Cd_2d,chord_ref_m,span_m\n"
        "A,a,b,2,0.1,0.5,1\nA,a,c,2,0.1,0.5,1\n"
    )
    with pytest.raises(ValueError, match="duplicate"):
        read_wing_candidates(p)
    p.write_text("config_id,flap1_id,flap2_id,Cl_2d,Cd_2d,chord_ref_m,span_m\nA,a,b,-2,0.1,0.5,1\n")
    with pytest.raises(ValueError, match="downforce"):
        read_wing_candidates(p)


def test_direct_3d_values_take_priority(tmp_path, rear):
    p = tmp_path / "c.csv"
    p.write_text(
        "config_id,flap1_id,flap2_id,Cl_2d,Cd_2d,chord_ref_m,span_m,cla_3d_m2,"
        "cda_3d_m2,mass_kg,notes\nX,a,b,2.9,0.12,0.7,1.1,1.42,0.47,6.0,CFD 3D\n"
        "Y,a,b,2.9,0.12,0.7,1.1,,,,\n"
    )
    x, y = read_wing_candidates(p)
    cx = candidate_component(x, rear)
    assert cx.source == "3D direct" and cx.wing3d is None
    assert (cx.component.cla_m2, cx.component.cda_m2, cx.component.mass_kg) == (1.42, 0.47, 6.0)
    assert not x.is_dummy
    cy = candidate_component(y, rear)
    assert cy.source == "2D→3D estimate"
    assert cy.component.mass_kg == rear.mass_kg  # falls back to the baseline component
    assert cy.component.x_cp_m == rear.x_cp_m


def test_centre_of_pressure_from_cm(rear):
    c = WingCandidate("C", "a", "b", 3.0, 0.1, 0.7, 1.1, cm_2d=0.9, overrides={"cm_ref_x_m": 1.7})
    comp = candidate_component(c, rear).component
    assert comp.x_cp_m == pytest.approx(1.7 + 0.9 / 3.0 * 0.7)


def test_candidate_vehicle_mass_and_aero(baseline, comps):
    cands = read_wing_candidates(DUMMY)
    veh, ca = candidate_vehicle(baseline, comps, cands[3])  # RW04, 5.2 kg vs 5.0 baseline
    assert veh.mass_kg == pytest.approx(baseline.mass_kg + 0.2)
    rw = next(c for c in comps if c.name == "rear_wing")
    assert veh.aero.cla_m2 == pytest.approx(baseline.aero.cla_m2 - rw.cla_m2 + ca.component.cla_m2)
    assert veh.name.endswith("RW04")
    trimmed, _ = candidate_vehicle(baseline, comps, cands[3], retrim_front=True)
    assert trimmed.aero.balance_front == pytest.approx(baseline.aero.balance_front, abs=1e-9)
    with pytest.raises(KeyError):
        candidate_vehicle(baseline, comps, cands[0], component_name="nose")


def test_retrim_component_errors(comps, baseline):
    with pytest.raises(KeyError):
        retrim_component(comps, 0.45, baseline.wheelbase_m, name="canard")
    with pytest.raises(ValueError):
        retrim_component(comps, -0.1, baseline.wheelbase_m)  # would need negative scale
    trimmed = retrim_component(comps, 0.47, baseline.wheelbase_m)
    assert synthesize(trimmed, baseline.wheelbase_m).balance_front == pytest.approx(0.47)
    with pytest.raises(KeyError):
        replace_component(comps, comps[0], name="nope")


def test_candidate_table(rear):
    df = candidate_table(read_wing_candidates(DUMMY), rear, WingConversion(endplate_factor=2.0))
    assert len(df) == 8 and df["dummy"].all()
    assert (df["L/D_3d"] < df["L/D_2d"]).all()  # induced drag lowers L/D
    assert df["ClA_m2"].is_monotonic_increasing is False  # ordering follows the CSV


def test_compare_candidates_ranks_and_deltas(baseline, comps, setup):
    cands = read_wing_candidates(DUMMY)[:4]
    cmp = compare_candidates(baseline, comps, cands, setup, events=("skidpad", "autocross"))
    t = cmp.table
    assert cmp.reference == "RW01" and cmp.dummy
    assert list(t["config_id"]) == ["baseline", "RW01", "RW02", "RW03", "RW04"]
    ref = t[t.config_id == "RW01"].iloc[0]
    assert ref["d_autocross_s"] == 0.0 and ref["d_skidpad_s"] == 0.0
    cand = t[t.config_id != "baseline"]
    assert sorted(cand["rank_autocross"].astype(int)) == [1, 2, 3, 4]
    best = cand.sort_values("autocross_time_s").iloc[0]
    assert int(best["rank_autocross"]) == 1
    cmp2 = compare_candidates(
        baseline,
        comps,
        cands,
        setup,
        events=("skidpad",),
        reference="RW03",
        retrim_front=True,
        include_baseline=False,
    )
    assert "baseline" not in set(cmp2.table["config_id"])
    assert cmp2.table["balance_front"].round(9).nunique() == 1
    with pytest.raises(ValueError):
        compare_candidates(baseline, comps, cands, setup, reference="XX")
    with pytest.raises(ValueError):
        compare_candidates(baseline, comps, [], setup)
