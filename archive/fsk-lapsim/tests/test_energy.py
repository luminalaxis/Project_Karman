"""Powertrain/energy checks incl. spec validation test 8 (energy balance)."""

import numpy as np
import pytest

from fsk_lapsim.energy import operating_point_histogram, rms
from fsk_lapsim.solver import SolverOptions, simulate
from fsk_lapsim.track import straight_track
from fsk_lapsim.trackgen import load_track
from fsk_lapsim.vehicle import with_param, with_params

TRACKS = [
    "tracks/autocross_generated.yaml",
    "tracks/endurance_generated.yaml",
    "tracks/fsk2026_endurance.yaml",
]


def _time_domain_losses(veh, res):
    """Drag and rolling work integrated over *time* from node values (independent quadrature)."""
    v = res.v
    q = 0.5 * veh.aero.air_density_kg_m3 * v * v
    drag_p = q * veh.aero.cda_m2 * v
    roll_p = veh.tire.rolling_resistance_coeff * (veh.weight_n + q * veh.aero.cla_m2) * v
    return np.trapezoid(drag_p, res.t), np.trapezoid(roll_p, res.t)


# ------------------------------------------------------------------ test 8: energy balance
@pytest.mark.parametrize(
    "track,start",
    [
        (TRACKS[0], "standing"),
        (TRACKS[1], "standing"),
        (TRACKS[1], "flying"),
        (TRACKS[2], "flying"),
    ],
)
def test_8_energy_balance(baseline, track, start):
    tr = load_track(track)
    res = simulate(baseline, tr, start=start)
    e = res.energy
    # (a) bookkeeping identity with the solver's own segment quantities (exact)
    lhs = e.work_drive_j - e.work_brake_j
    rhs = e.delta_ke_j + e.work_drag_j + e.work_rolling_j
    assert lhs == pytest.approx(rhs, rel=1e-9)
    # (b) independent check: losses integrated over time from node speeds (trapezoid in t).
    #     Tolerance 0.5 % of the drive work (quadrature differs, O(ds²)); observed ~1e-5.
    drag_t, roll_t = _time_domain_losses(baseline, res)
    ke = 0.5 * baseline.mass_kg * (res.v[-1] ** 2 - res.v[0] ** 2)
    assert abs((e.work_drive_j - e.work_brake_j) - (ke + drag_t + roll_t)) < 5e-3 * e.work_drive_j
    if start == "flying":
        assert e.delta_ke_j == pytest.approx(0.0, abs=1e-6)
    # accumulator side: drive energy = tyre drive work / η, regen bounded by braking work
    eta = baseline.powertrain.efficiency
    assert e.battery_drive_j == pytest.approx(e.work_drive_j / eta, rel=1e-12)
    assert 0.0 <= e.regen_j <= eta * e.work_brake_j + 1e-9
    assert e.friction_brake_j >= -1e-6


def test_power_and_regen_limits(baseline):
    res = simulate(baseline, load_track(TRACKS[1]))
    e = res.energy
    pmax = baseline.powertrain.max_power_w
    assert e.p_battery_w.max() <= pmax * 1.01  # discretisation: F at node speed, P at mean speed
    regen_power = baseline.powertrain.efficiency * e.regen_force_n * e.v_mean_mps
    assert regen_power.max() <= baseline.powertrain.regen_max_power_w * (1 + 1e-9)
    assert e.regen_j > 0
    s = res.summary()
    assert s["energy_kwh"] == pytest.approx(s["energy_no_regen_kwh"] - s["regen_kwh"])
    assert 0 < s["mean_power_kw"] < s["max_power_kw"] <= 81.0


def test_regen_disabled_and_fixed_bias(baseline):
    tr = load_track(TRACKS[1])
    no_regen = simulate(with_param(baseline, "powertrain.regen_enabled", False), tr)
    assert no_regen.energy.regen_j == 0.0
    with_regen = simulate(baseline, tr)
    assert with_regen.lap_time_s == pytest.approx(no_regen.lap_time_s)
    assert with_regen.energy.battery_net_j < no_regen.energy.battery_net_j
    fixed = with_params(baseline, {"brake_distribution": "fixed", "brake_front_bias": 0.7})
    rf = simulate(fixed, tr)
    assert rf.energy.regen_j > 0


def test_motor_operating_points(baseline):
    res = simulate(baseline, load_track(TRACKS[0]))
    e = res.energy
    pt = baseline.powertrain
    assert e.motor_rpm.max() <= pt.motor_max_rpm * (1 + 1e-6)
    assert e.motor_torque_nm.max() <= pt.motor_max_torque_nm * (1 + 1e-3)
    assert e.motor_torque_nm.min() >= -pt.motor_max_torque_nm * (1 + 1e-6)
    hist, rpm_edges, tq_edges = operating_point_histogram(e)
    assert hist.sum() == pytest.approx(res.lap_time_s)
    assert rpm_edges[0] == 0 and tq_edges[0] < 0 < tq_edges[-1]
    assert rms(e.motor_torque_nm, e.dt_s) > 0


def test_standing_start_power_and_torque_limits(baseline):
    res = simulate(baseline, straight_track(75.0))
    e = res.energy
    # standing start: traction limited (finite force at v = 0, no P/v blow-up)
    assert np.isfinite(e.fx_tyre_n).all()
    assert e.fx_tyre_n[0] < baseline.powertrain.torque_force_n(baseline.tire.radius_m)
    # later the accumulator cap is reached
    assert e.p_battery_w.max() == pytest.approx(baseline.powertrain.max_power_w, rel=0.01)
    assert res.energy.regen_j == 0.0


def test_model_a_energy(baseline):
    res = simulate(baseline, load_track(TRACKS[0]), SolverOptions(model="point_mass"))
    e = res.energy
    assert e.work_drive_j - e.work_brake_j == pytest.approx(
        e.delta_ke_j + e.work_drag_j + e.work_rolling_j, rel=1e-9
    )
