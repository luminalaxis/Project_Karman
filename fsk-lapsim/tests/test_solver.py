"""Validation tests 1–6 of the spec (section 6) plus solver API checks."""

import dataclasses
import itertools
import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from fsk_lapsim.ggv import GGV, cached_ggv
from fsk_lapsim.models import AxleModel, PointMassModel, make_model, top_speed
from fsk_lapsim.solver import SolverOptions, simulate, solve_speed_profile
from fsk_lapsim.track import circle_track, straight_track
from fsk_lapsim.trackgen import load_track
from fsk_lapsim.vehicle import Vehicle, with_param, with_params

AUTOX = "tracks/autocross_generated.yaml"
ENDU = "tracks/endurance_generated.yaml"


def _no_resistance(v: Vehicle) -> Vehicle:
    return with_params(v, {"aero.cda_m2": 0.0, "tire.rolling_resistance_coeff": 0.0})


# ------------------------------------------------------------------ test 1: circle analytic
@pytest.mark.parametrize("radius", [6.0, 9.125, 15.0, 40.0])
def test_1_circle_matches_downforce_analytic(baseline, radius):
    veh = _no_resistance(baseline)
    m, g, mu, rho, cla = (
        veh.mass_kg,
        veh.g_mps2,
        veh.tire.mu0,
        veh.aero.air_density_kg_m3,
        veh.aero.cla_m2,
    )
    v_exact = math.sqrt(mu * m * g / (m / radius - 0.5 * rho * mu * cla))
    res = simulate(veh, circle_track(radius), SolverOptions(model="point_mass"))
    t_exact = 2 * math.pi * radius / v_exact
    assert res.lap_time_s == pytest.approx(t_exact, rel=1e-3)
    assert res.lap_time_s == pytest.approx(t_exact, rel=1e-6)  # actually far tighter
    assert np.allclose(res.v, v_exact, rtol=1e-6)


def test_1b_axle_model_reduces_to_point_mass(baseline):
    """No load transfer (h=0), no load sensitivity, balance = weight split → B equals A."""
    veh = _no_resistance(baseline)
    veh = with_params(
        veh,
        {
            "cg_height_m": 0.0,
            "tire.load_sensitivity": 0.0,
            "aero.balance_front": veh.static_front_fraction,
        },
    )
    v = np.linspace(0, 35, 15)
    assert np.allclose(AxleModel(veh).ay_max(v), PointMassModel(veh).ay_max(v), rtol=1e-9)
    t_a = simulate(veh, circle_track(12.0), SolverOptions(model="point_mass")).lap_time_s
    t_b = simulate(veh, circle_track(12.0), SolverOptions(model="axle")).lap_time_s
    assert t_b == pytest.approx(t_a, rel=1e-6)


# ------------------------------------------------------------------ test 2: top speed
def test_2_top_speed_power_drag_analytic(baseline):
    veh = with_param(baseline, "powertrain.motor_max_rpm", 1e5)  # remove rpm limit
    pt, a, t = veh.powertrain, veh.aero, veh.tire

    def balance(v):
        resist = 0.5 * a.air_density_kg_m3 * a.cda_m2 * v**2 + t.rolling_resistance_coeff * (
            veh.weight_n + 0.5 * a.air_density_kg_m3 * a.cla_m2 * v**2
        )
        return pt.max_power_w * pt.efficiency - resist * v

    v_exact = brentq(balance, 5.0, 100.0, xtol=1e-12)
    for name in ("point_mass", "axle"):
        assert top_speed(make_model(veh, name)) == pytest.approx(v_exact, rel=1e-6)
    # the lap solver approaches it on a very long straight
    res = simulate(veh, straight_track(3000.0), SolverOptions(ds_m=0.5))
    assert res.v.max() == pytest.approx(v_exact, rel=5e-3)
    assert res.v.max() <= v_exact * (1 + 1e-9)


def test_2b_top_speed_rpm_limited(baseline):
    v_rpm = baseline.v_max_mps
    assert v_rpm == pytest.approx(38.80, abs=0.01)
    assert top_speed(make_model(baseline)) == pytest.approx(v_rpm)
    res = simulate(baseline, straight_track(1500.0))
    assert res.v.max() == pytest.approx(v_rpm, rel=1e-9)


# ------------------------------------------------------------------ test 3: acceleration ODE
@pytest.mark.parametrize("model", ["point_mass", "axle"])
def test_3_acceleration_matches_solve_ivp(baseline, model):
    mdl = make_model(baseline, model)
    length = 75.0

    def rhs(_t, y):
        return [y[1], float(mdl.ax_accel_max(max(y[1], 0.0), 0.0))]

    def finish(_t, y):
        return y[0] - length

    finish.terminal = True
    sol = solve_ivp(rhs, (0, 20), [0.0, 0.0], events=finish, rtol=1e-9, atol=1e-9, max_step=0.01)
    t_ref = float(sol.t_events[0][0])
    res = simulate(baseline, straight_track(length), SolverOptions(model=model))
    assert res.lap_time_s == pytest.approx(t_ref, rel=5e-3)
    assert res.lap_time_s == pytest.approx(t_ref, rel=1e-3)  # observed accuracy is better


# ------------------------------------------------------------------ test 4: grid convergence
@pytest.mark.parametrize("track", [AUTOX, ENDU, "tracks/fsk2026_endurance.yaml"])
@pytest.mark.parametrize("model", ["point_mass", "axle"])
def test_4_grid_convergence(baseline, track, model):
    tr = load_track(track)
    t1 = simulate(baseline, tr, SolverOptions(ds_m=0.25, model=model)).lap_time_s
    t2 = simulate(baseline, tr, SolverOptions(ds_m=0.125, model=model)).lap_time_s
    assert abs(t1 - t2) / t2 < 1e-3


# ------------------------------------------------------------------ test 5: monotonicity
def _laptimes(veh, path, values):
    tr = load_track(path)
    return [simulate(with_param(veh, path_, x), tr).lap_time_s for path_, x in values]


@pytest.mark.parametrize("track", [AUTOX, ENDU])
def test_5_monotonicity(baseline, track):
    tol = 1e-9  # relative numerical noise allowance
    p = _laptimes(
        baseline, track, [("powertrain.max_power_w", x) for x in (40e3, 60e3, 80e3, 100e3)]
    )
    assert all(b <= a * (1 + tol) for a, b in itertools.pairwise(p))
    assert p[-1] < p[0]
    mu = _laptimes(baseline, track, [("tire.mu0", x) for x in (1.2, 1.35, 1.45, 1.6)])
    assert all(b <= a * (1 + tol) for a, b in itertools.pairwise(mu))
    assert mu[-1] < mu[0]
    cd = _laptimes(baseline, track, [("aero.cda_m2", x) for x in (0.8, 1.1, 1.4, 1.8)])
    assert all(b >= a * (1 - tol) for a, b in itertools.pairwise(cd))
    assert cd[-1] > cd[0]
    cl = _laptimes(baseline, track, [("aero.cla_m2", x) for x in (0.0, 1.5, 3.5, 5.0)])
    assert all(b <= a * (1 + tol) for a, b in itertools.pairwise(cl))
    assert cl[-1] < cl[0]


def test_5b_monotonicity_point_mass(baseline):
    tr = load_track(AUTOX)
    opts = SolverOptions(model="point_mass")
    times = [
        simulate(with_param(baseline, "aero.cla_m2", x), tr, opts).lap_time_s
        for x in (0.0, 2.0, 4.0)
    ]
    assert times[0] > times[1] > times[2]


# ------------------------------------------------------------------ test 6: symmetry
@pytest.mark.parametrize("track", [AUTOX, ENDU, "tracks/template_autocross_A.yaml"])
def test_6_mirror_symmetry(baseline, track):
    tr = load_track(track)
    t = simulate(baseline, tr).lap_time_s
    tm = simulate(baseline, tr.mirrored()).lap_time_s
    assert tm == pytest.approx(t, rel=1e-12)


def test_6b_mirror_symmetry_imported_track(baseline):
    tr = load_track("tracks/fsk2026_endurance.yaml")
    assert simulate(baseline, tr.mirrored()).lap_time_s == pytest.approx(
        simulate(baseline, tr).lap_time_s, rel=1e-12
    )


# ------------------------------------------------------------------ API behaviour
def test_flying_lap_is_periodic(baseline):
    res = simulate(baseline, load_track(ENDU))
    assert res.closed and res.periodic_iterations >= 1
    assert res.v[0] == pytest.approx(res.v[-1], abs=1e-6)


def test_standing_and_flying_options(baseline):
    tr = load_track(ENDU)
    flying = simulate(baseline, tr, start="flying")
    standing = simulate(baseline, tr, start="standing")
    assert standing.v[0] == 0.0 and standing.lap_time_s > flying.lap_time_s
    capped = simulate(baseline, tr, start="standing", v_end=5.0)
    assert capped.v[-1] == pytest.approx(5.0) and capped.lap_time_s >= standing.lap_time_s
    with pytest.raises(ValueError):
        simulate(baseline, straight_track(50), start="flying")
    with pytest.raises(ValueError):
        simulate(baseline, tr, alt_vehicle=baseline, alt_weight=np.zeros(3))


@pytest.mark.parametrize("model", ["point_mass", "axle"])
def test_circle_with_drag_holds_constant_speed_limit(baseline, model):
    """With drag the car circles at the constant-speed limit: it can hold, but not exceed it."""
    res = simulate(baseline, circle_track(15.0), SolverOptions(model=model))
    assert res.v.max() - res.v.min() < 1e-9
    mdl = make_model(baseline, model)
    v = float(res.v.mean())
    assert v * v / 15.0 == pytest.approx(float(mdl.ay_max(v)), rel=1e-4)
    assert float(mdl.ax_accel_max(v, v * v / 15.0)) >= -1e-9
    assert np.isnan(mdl.ax_accel_max(v * 1.01, (v * 1.01) ** 2 / 15.0))
    # holding speed needs drive force, so the limit is below the coasting (Fx = 0) limit
    if model == "point_mass":
        coast_ay = baseline.tire.mu0 * (baseline.weight_n + float(mdl.downforce_n(v))) / 320.0
        assert v * v / 15.0 < coast_ay


def test_summary_and_limits(baseline):
    res = simulate(baseline, load_track(AUTOX))
    s = res.summary()
    assert s["lap_time_s"] == res.lap_time_s
    assert 0 < s["fraction_corner"] < 1
    assert s["fraction_accel"] + s["fraction_brake"] + s["fraction_corner"] == pytest.approx(1)
    assert s["max_speed_kph"] <= baseline.v_max_mps * 3.6 + 1e-6
    assert res.wheel_loads.shape == (4, len(res.s))
    assert np.all(res.wheel_loads >= 0)


def test_ggv_cache_and_speed_profile_api(baseline):
    g1 = cached_ggv(baseline)
    g2 = cached_ggv(dataclasses.replace(baseline))
    assert g1 is g2
    k = np.zeros(11)
    prof = solve_speed_profile(g1, k, 1.0, periodic=False, v_start=0.0)
    assert prof.v[0] == 0 and np.all(np.diff(prof.v) > 0)


def test_unknown_model_raises(baseline):
    with pytest.raises(ValueError):
        make_model(baseline, "bicycle")


def test_ggv_interpolation_is_exact_on_grid(baseline):
    g = GGV.build(make_model(baseline, "point_mass"))
    acc, brk = g.scalar_lookups()
    i, j = 40, 10
    v = g.v_grid[i]
    ay = math.sin(g.theta_grid[j]) * g.ay_max[i]
    assert acc(v, ay) == pytest.approx(g.ax_acc[i, j], abs=1e-9)
    assert brk(v, -ay) == pytest.approx(g.ax_brk[i, j], abs=1e-9)
    assert float(g.ax_acc_at(v, ay)) == pytest.approx(g.ax_acc[i, j], abs=1e-9)
