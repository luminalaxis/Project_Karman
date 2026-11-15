"""P8: two-state active aero (DRS)."""

import dataclasses

import numpy as np
import pytest

from fsk_lapsim.active_aero import (
    DRSSettings,
    _zones,
    crossover_speed,
    drs_event_gains,
    drs_weight,
    lead_distance_table,
    open_state,
    simulate_drs,
)
from fsk_lapsim.events import EventSetup
from fsk_lapsim.solver import simulate
from fsk_lapsim.trackgen import load_track
from fsk_lapsim.vehicle import with_param


@pytest.fixture(scope="module")
def drs():
    return DRSSettings.load()


@pytest.fixture(scope="module")
def loop():
    return load_track("tracks/endurance_generated.yaml")


def test_settings_from_bundled_yaml(drs):
    assert drs.rear.name == "rear_wing"
    assert (drs.cl_reduction, drs.cd_reduction, drs.transition_time_s) == (0.65, 0.55, 0.25)
    assert drs.min_speed_mps is None and drs.require_full_throttle


def test_open_state_coefficients(baseline, drs):
    o = open_state(baseline, drs)
    assert o.aero.cla_m2 == pytest.approx(3.5 - 0.65 * 1.30)
    assert o.aero.cda_m2 == pytest.approx(1.4 - 0.55 * 0.45)
    assert o.aero.balance_front > baseline.aero.balance_front  # rear downforce removed
    assert o.mass_kg == baseline.mass_kg
    with pytest.raises(ValueError):
        open_state(baseline, dataclasses.replace(drs, cl_reduction=10.0))


def test_zone_detection_with_wrap():
    m = np.array([1, 1, 0, 0, 1, 1, 0, 1, 1], bool)
    assert _zones(m, closed_track=False) == [(0, 2), (4, 6), (7, 9)]
    assert _zones(m, closed_track=True) == [(4, 6), (7, 2)]
    assert _zones(np.zeros(4, bool), True) == []
    assert _zones(np.ones(4, bool), True) == [(0, 4)]


def test_crossover_speed(baseline, drs):
    v = crossover_speed(baseline, open_state(baseline, drs))
    assert 10.0 < v < baseline.v_max_mps
    # without drag reduction opening still cuts rolling resistance (Crr × downforce) ...
    no_drag_gain = open_state(baseline, dataclasses.replace(drs, cd_reduction=0.0))
    assert crossover_speed(baseline, no_drag_gain) > v
    # ... so only with Crr = 0 as well does opening never pay off
    no_rr = with_param(baseline, "tire.rolling_resistance_coeff", 0.0)
    no_gain = open_state(no_rr, dataclasses.replace(drs, cd_reduction=0.0))
    assert crossover_speed(no_rr, no_gain) == float("inf")


def test_drs_weight_profile_and_lead(baseline, drs, loop):
    s = dataclasses.replace(drs, min_speed_mps=crossover_speed(baseline, open_state(baseline, drs)))
    res = simulate(baseline, loop)
    w, zones = drs_weight(res, s)
    assert w.min() >= 0 and w.max() == pytest.approx(1.0)
    assert w[0] == w[-1]
    assert len(zones) >= 1
    for _, z in zones.iterrows():
        k = int(np.argmin(np.abs(res.s - z["s_brake_m"])))
        assert w[k] == pytest.approx(0.0, abs=1e-9)  # fully closed where braking starts
        v_mps = z["v_brake_kph"] / 3.6
        assert z["close_lead_m"] == pytest.approx(v_mps * drs.transition_time_s, rel=0.1)
        assert z["worst_case_lead_m"] >= z["close_lead_m"] - 0.3
    # nothing opens below the threshold speed or while braking/cornering
    assert np.all(w[res.v < s.min_speed_mps - 1.0] == 0.0)


def test_simulate_drs_gains_and_energy(baseline, drs, loop):
    lap = simulate_drs(baseline, loop, drs)
    assert lap.time_gain_s > 0
    sm = lap.summary()
    assert sm["switches_per_lap"] == 2 * sm["zones_per_lap"] > 0
    assert 0 < sm["open_fraction_time"] < sm["open_fraction_distance"] < 1
    assert sm["energy_saving_kwh"] > 0
    assert lap.drs.v[0] == pytest.approx(lap.drs.v[-1], abs=1e-6)  # still periodic


def test_drs_never_opens_when_disabled_by_settings(baseline, drs, loop):
    for s in (
        dataclasses.replace(drs, ay_threshold_mps2=0.0),
        dataclasses.replace(drs, min_open_time_s=60.0),
        dataclasses.replace(drs, min_speed_mps=1e3),
    ):
        lap = simulate_drs(baseline, loop, s)
        assert lap.time_gain_s == pytest.approx(0.0, abs=1e-12)
        assert lap.weight.max() == 0.0 and len(lap.zones) == 0
        assert lap.summary()["max_close_lead_m"] == 0.0


def test_opening_everywhere_at_low_speed_costs_time(baseline, drs, loop):
    """Opening while traction limited loses rear downforce: the auto speed limit exists for that."""
    eager = simulate_drs(baseline, loop, dataclasses.replace(drs, min_speed_mps=0.0))
    auto = simulate_drs(baseline, loop, drs)
    assert auto.time_gain_s > eager.time_gain_s


def test_event_gains_table(baseline, drs):
    g = drs_event_gains(baseline, drs, EventSetup.load())
    t = g.table.set_index("event")
    assert set(t.index) == {
        "acceleration",
        "autocross",
        "endurance_lap",
        "endurance_total",
        "skidpad",
    }
    assert t.loc["skidpad", "time_gain_s"] == 0.0
    assert t.loc["endurance_total", "time_gain_s"] > t.loc["endurance_lap", "time_gain_s"]
    assert t.loc["acceleration", "time_gain_s"] > 0


def test_lead_distance_table():
    df = lead_distance_table([36.0, 108.0], [0.1, 0.3])
    assert df.loc[0, "T=0.10s"] == pytest.approx(1.0)
    assert df.loc[1, "T=0.30s"] == pytest.approx(9.0)
