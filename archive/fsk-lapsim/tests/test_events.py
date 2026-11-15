"""P6: the four events."""

import math

import pytest

from fsk_lapsim.events import (
    EventResults,
    EventSetup,
    plausibility_warnings,
    run_acceleration,
    run_autocross,
    run_endurance,
    run_events,
    run_skidpad,
)
from fsk_lapsim.models import make_model
from fsk_lapsim.solver import simulate
from fsk_lapsim.track import circle_track, straight_track
from fsk_lapsim.trackgen import load_track
from fsk_lapsim.vehicle import with_param


@pytest.fixture(scope="module")
def setup():
    return EventSetup.load()


def test_event_setup_from_bundled_config(setup):
    assert setup.acceleration_length_m == 75.0
    assert setup.skidpad_radius_m == 9.125
    assert setup.endurance_track.closed and not setup.autocross_track.closed
    assert setup.endurance_distance_m == 22000.0
    assert setup.endurance_max_power_w is None


def test_acceleration_is_standing_start(baseline):
    res = run_acceleration(baseline, 75.0)
    assert res.v[0] == 0.0 and res.length_m == pytest.approx(75.0)
    assert res.lap_time_s == pytest.approx(simulate(baseline, straight_track(75.0)).lap_time_s)


def test_skidpad_is_steady_state_and_matches_circle_sim(baseline):
    sk = run_skidpad(baseline, 9.125)
    mdl = make_model(baseline)
    assert sk.ay_mps2 == pytest.approx(float(mdl.ay_max(sk.speed_mps)), rel=1e-9)
    assert sk.time_s == pytest.approx(2 * math.pi * 9.125 / sk.speed_mps)
    assert sk.time_left_s == sk.time_right_s
    lap_l = simulate(baseline, circle_track(9.125, 1)).lap_time_s
    lap_r = simulate(baseline, circle_track(9.125, -1)).lap_time_s
    assert lap_l == pytest.approx(sk.time_s, rel=1e-4)
    assert lap_r == pytest.approx(lap_l, rel=1e-12)
    # point-mass skidpad with no drag equals the closed-form downforce solution
    veh = with_param(with_param(baseline, "aero.cda_m2", 0.0), "tire.rolling_resistance_coeff", 0.0)
    sk_a = run_skidpad(veh, 9.125, model="point_mass")
    m, g, mu, rho, cla = 320.0, 9.81, 1.45, 1.2, 3.5
    v = math.sqrt(mu * m * g / (m / 9.125 - 0.5 * rho * mu * cla))
    assert sk_a.speed_mps == pytest.approx(v, rel=1e-9)


def test_skidpad_huge_radius_hits_top_speed(baseline):
    sk = run_skidpad(baseline, 5000.0)
    assert sk.speed_mps == pytest.approx(baseline.v_max_mps)


def test_autocross_start_modes(baseline, setup):
    tr = setup.autocross_track
    st = run_autocross(baseline, tr, "standing")
    assert st.v[0] == 0.0
    loop = load_track("tracks/endurance_generated.yaml")
    fl = run_autocross(baseline, loop, "flying")
    assert fl.closed and fl.lap_time_s < run_autocross(baseline, loop, "standing").lap_time_s
    # flying start requested on an open course falls back to standing
    assert run_autocross(baseline, tr, "flying").v[0] == 0.0


def test_endurance_totals(baseline, setup):
    e = run_endurance(baseline, setup.endurance_track, 22000.0)
    assert e.n_laps == round(22000.0 / setup.endurance_track.length_m)
    assert e.total_time_s == pytest.approx(
        e.first_lap.lap_time_s + (e.n_laps - 1) * e.flying_lap.lap_time_s
    )
    assert e.first_lap.lap_time_s > e.flying_lap.lap_time_s
    assert e.first_lap.v[-1] == pytest.approx(e.flying_lap.v[0], abs=1e-6)
    assert e.energy_kwh == pytest.approx(e.energy_no_regen_kwh - e.regen_kwh)
    capped = run_endurance(baseline, setup.endurance_track, 22000.0, max_power_w=40e3)
    assert capped.total_time_s > e.total_time_s and capped.energy_kwh < e.energy_kwh
    with pytest.raises(ValueError):
        run_endurance(baseline, straight_track(100.0))


def test_run_events_all(baseline, setup):
    res = run_events(baseline, setup)
    times = res.times()
    assert set(times) == {"acceleration", "skidpad", "autocross", "endurance"}
    m = res.metrics()
    assert m["endurance_laps"] >= 1 and m["skidpad_ay_g"] > 1.0
    assert isinstance(res.warnings, list)
    sub = run_events(baseline, setup, events=("skidpad",))
    assert set(sub.times()) == {"skidpad"} and sub.autocross is None
    with pytest.raises(ValueError):
        run_events(baseline, setup, events=("drag_race",))


def test_plausibility_warnings_fire_for_silly_car(baseline, setup):
    slow = with_param(with_param(baseline, "powertrain.max_power_w", 8e3), "tire.mu0", 0.6)
    res = run_events(slow, setup, events=("acceleration", "skidpad", "autocross"))
    w = plausibility_warnings(res)
    assert len(w) == 3
    assert plausibility_warnings(EventResults("empty")) == []
