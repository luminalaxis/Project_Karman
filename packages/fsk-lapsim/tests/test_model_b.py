"""Model B checks, incl. spec validation tests 7 (balance extremes) and 9 (GGV vs direct)."""

import numpy as np
import pytest

from fsk_lapsim.ggv import GGV
from fsk_lapsim.models import AxleModel, PointMassModel, make_model
from fsk_lapsim.powertrain import Powertrain
from fsk_lapsim.solver import SolverOptions, simulate
from fsk_lapsim.tire import Tire, ellipse_remaining
from fsk_lapsim.trackgen import load_track
from fsk_lapsim.vehicle import with_param, with_params


# ------------------------------------------------------------------ test 7: balance extremes
@pytest.mark.parametrize("v", [20.0, 30.0])
def test_7_extreme_balance_moves_limit_to_starved_axle(baseline, v):
    rear_heavy = AxleModel(with_param(baseline, "aero.balance_front", 0.0)).ay_max_by_limit(v)
    front_heavy = AxleModel(with_param(baseline, "aero.balance_front", 1.0)).ay_max_by_limit(v)
    # all downforce on the rear: the front axle saturates first (understeer limit)
    assert rear_heavy["front"] < rear_heavy["rear"]
    assert rear_heavy["total"] == pytest.approx(rear_heavy["front"])
    # all downforce on the front: the rear axle (lateral or traction) saturates first
    assert front_heavy["rear"] < front_heavy["front"]
    assert min(front_heavy["rear"], front_heavy["drive"]) == pytest.approx(front_heavy["total"])


def test_7b_balance_effect_grows_with_speed(baseline):
    """At walking pace aero balance hardly matters; at speed the starved axle stops gaining."""
    m0 = AxleModel(with_param(baseline, "aero.balance_front", 0.0))
    lim_lo, lim_hi = m0.ay_max_by_limit(3.0), m0.ay_max_by_limit(30.0)
    assert lim_hi["front"] == pytest.approx(lim_lo["front"], rel=0.02)
    assert lim_hi["rear"] > 1.3 * lim_lo["rear"]


def test_7c_lap_time_has_interior_balance_optimum(baseline):
    tr = load_track("tracks/autocross_generated.yaml")
    times = {
        b: simulate(with_param(baseline, "aero.balance_front", b), tr).lap_time_s
        for b in (0.0, 0.45, 1.0)
    }
    assert times[0.45] < times[0.0] and times[0.45] < times[1.0]


# ------------------------------------------------------------------ test 9: GGV vs direct
@pytest.mark.parametrize("model", ["point_mass", "axle"])
def test_9_ggv_interpolation_matches_direct(baseline, model):
    mdl = make_model(baseline, model)
    g = GGV.build(mdl)
    rng = np.random.default_rng(42)
    v = rng.uniform(0.0, g.v_top, 3000)
    ay = np.sqrt(rng.uniform(0, 1, v.size)) * mdl.ay_max(v) * np.sign(rng.uniform(-1, 1, v.size))
    acc_err = np.abs(g.ax_acc_at(v, ay) - mdl.ax_accel_max(v, ay))
    brk_err = np.abs(g.ax_brk_at(v, ay) - mdl.ax_brake_min(v, ay))
    # tolerance: 0.3 % of 1 g at worst, 0.05 % g for 99 % of the envelope
    assert np.percentile(acc_err, 99) < 0.005
    assert acc_err.max() < 0.03
    assert brk_err.max() < 0.03
    assert np.abs(g.ay_max_at(v) - mdl.ay_max(v)).max() < 1e-3
    acc, brk = g.scalar_lookups()
    sc = np.array([acc(a, b) for a, b in zip(v[:300], ay[:300], strict=True)])
    assert np.allclose(sc, g.ax_acc_at(v[:300], ay[:300]), atol=1e-12)
    sb = np.array([brk(a, b) for a, b in zip(v[:300], ay[:300], strict=True)])
    assert np.allclose(sb, g.ax_brk_at(v[:300], ay[:300]), atol=1e-12)


def test_9b_lap_time_insensitive_to_ggv_resolution(baseline):
    tr = load_track("tracks/autocross_generated.yaml")
    coarse = simulate(baseline, tr, SolverOptions(ggv_n_theta=37, ggv_dv_mps=0.5)).lap_time_s
    fine = simulate(baseline, tr, SolverOptions(ggv_n_theta=145, ggv_dv_mps=0.125)).lap_time_s
    assert coarse == pytest.approx(fine, rel=2e-5)


def test_ggv_limit_row_holds_speed(baseline):
    g = GGV.build(make_model(baseline))
    lim = g.ax_acc_at(g.v_grid, g.ay_max)
    assert np.all(lim >= -1e-9)  # on the limit the car can hold its speed
    assert np.all(g.ax_brk[:, -1] <= 1e-9)
    assert np.all(np.diff(g.ax_acc, axis=2) <= 1e-9)  # each tyre limit shrinks with ay
    parts = make_model(baseline).ax_accel_parts(12.0, 10.0)
    assert set(parts) == {"front", "drive"}
    assert min(parts.values()) == pytest.approx(
        float(make_model(baseline).ax_accel_max(12.0, 10.0, power_limited=False)), abs=1e-9
    )


# ------------------------------------------------------------------ loads and transfers
def test_wheel_loads_equilibrium_and_transfer(baseline):
    mdl = AxleModel(baseline)
    v, ax, ay = 20.0, 3.0, 12.0
    fl, fr, rl, rr = mdl.wheel_loads(v, ax, ay)
    total = baseline.weight_n + float(mdl.downforce_n(v))
    assert fl + fr + rl + rr == pytest.approx(total)
    assert fr > fl and rr > rl  # left turn → right wheels outer
    m, h, L = baseline.mass_kg, baseline.cg_height_m, baseline.wheelbase_m
    f0 = mdl.wheel_loads(v, 0.0, ay)
    assert (rl + rr) - (f0[2] + f0[3]) == pytest.approx(m * ax * h / L)
    lat_f = (fr - fl) / 2 * baseline.track_front_m
    lat_r = (rr - rl) / 2 * baseline.track_rear_m
    assert lat_f / (lat_f + lat_r) == pytest.approx(baseline.lltd_front)
    assert lat_f + lat_r == pytest.approx(m * ay * h)
    mirrored = mdl.wheel_loads(v, ax, -ay)
    assert np.allclose(mirrored, [fr, fl, rr, rl])


def test_wheel_lift_is_clamped(baseline):
    mdl = AxleModel(with_param(baseline, "cg_height_m", 0.9))
    loads = mdl.wheel_loads(0.0, 0.0, 14.0)
    assert np.all(loads >= 0) and loads[0] == 0.0
    assert loads.sum() == pytest.approx(baseline.weight_n)


def test_point_mass_loads_and_nan_outside_envelope(baseline):
    pm = PointMassModel(baseline)
    assert np.allclose(
        pm.wheel_loads(10.0, 1.0, 5.0), (baseline.weight_n + float(pm.downforce_n(10.0))) / 4
    )
    lim = float(pm.ay_max(15.0))
    assert np.isnan(pm.ax_accel_max(15.0, 1.05 * lim))
    assert np.isnan(pm.ax_brake_min(15.0, 1.05 * lim))
    ax = AxleModel(baseline)
    lim = float(ax.ay_max(15.0))
    assert np.isnan(ax.ax_accel_max(15.0, 1.05 * lim))
    assert np.isnan(ax.ax_brake_min(15.0, 1.05 * lim))
    assert pm.ay_max_by_limit(10.0)["total"] == pytest.approx(pm.ay_max(10.0))


def test_ax_ref_negative_above_top_speed(baseline):
    veh = with_param(baseline, "powertrain.motor_max_rpm", 1e5)
    for mdl in (AxleModel(veh), PointMassModel(veh)):
        assert float(mdl.ax_ref(10.0)) == 0.0
        assert float(mdl.ax_ref(60.0)) < 0.0
        assert float(mdl.ay_max(60.0)) > 0.0


# ------------------------------------------------------------------ drivetrain / brakes
def test_open_diff_only_matters_in_corners(baseline):
    lsd = AxleModel(baseline)
    opn = AxleModel(with_param(baseline, "powertrain.differential", "open"))
    assert float(opn.ax_accel_max(8.0, 0.0)) == pytest.approx(float(lsd.ax_accel_max(8.0, 0.0)))
    assert float(opn.ax_accel_max(8.0, 8.0)) < float(lsd.ax_accel_max(8.0, 8.0))
    tr = load_track("tracks/autocross_generated.yaml")
    assert (
        simulate(with_param(baseline, "powertrain.differential", "open"), tr).lap_time_s
        > simulate(baseline, tr).lap_time_s
    )


def test_fixed_brake_bias_never_beats_ideal(baseline):
    ideal = AxleModel(baseline)
    v = np.linspace(2, 35, 12)
    for bias in (0.5, 0.6, 0.7, 0.8):
        fixed = AxleModel(
            with_params(baseline, {"brake_distribution": "fixed", "brake_front_bias": bias})
        )
        for ay_frac in (0.0, 0.5):
            ay = ay_frac * ideal.ay_max(v)
            assert np.all(fixed.ax_brake_min(v, ay) >= ideal.ax_brake_min(v, ay) - 1e-9)


def test_load_sensitivity_costs_grip(baseline):
    no_sens = AxleModel(with_param(baseline, "tire.load_sensitivity", 0.0))
    assert float(no_sens.ay_max(15.0)) > float(AxleModel(baseline).ay_max(15.0))
    pos = AxleModel(with_param(baseline, "tire.load_sensitivity", 0.05))
    assert float(pos.ay_max(15.0)) > float(no_sens.ay_max(15.0))
    assert float(pos.ax_brake_min(15.0, 3.0)) < float(no_sens.ax_brake_min(15.0, 3.0))


# ------------------------------------------------------------------ tyre and powertrain units
def test_tire_peak_clamp_and_ellipse():
    t = Tire()
    assert t.peak_load_n == pytest.approx(700 * 1.15 / 0.3)
    fz = np.linspace(-100, 6000, 500)
    f = t.max_lateral_force(fz)
    assert np.all(np.diff(f) >= -1e-9) and f[0] == 0.0
    assert t.max_longitudinal_force(1000.0) == pytest.approx(t.max_lateral_force(1000.0))
    assert Tire(load_sensitivity=0.1).peak_load_n == float("inf")
    assert t.mu(700.0) == pytest.approx(1.45)
    rem = ellipse_remaining(
        [100.0, 100.0, 0.0, 0.0], [100.0, 100.0, 0.0, 0.0], [60.0, 120.0, 0.0, 5.0]
    )
    assert rem[0] == pytest.approx(80.0) and np.isnan(rem[1]) and rem[2] == 0.0
    assert np.isnan(rem[3])


def test_powertrain_limits():
    pt = Powertrain()
    r = 0.228
    assert np.isfinite(pt.drive_force_limit(0.0, r))
    assert float(pt.drive_force_limit(0.0, r)) == pytest.approx(230 * 4 * 0.85 / r)
    assert float(pt.drive_force_limit(30.0, r)) == pytest.approx(80e3 * 0.85 / 30.0)
    assert float(pt.drive_force_limit(pt.v_max_mps(r) + 0.1, r)) == 0.0
    assert float(pt.motor_rpm(pt.v_max_mps(r), r)) == pytest.approx(6500.0)
    assert float(pt.regen_force_limit(30.0, r)) == pytest.approx(20e3 / (0.85 * 30.0))
    assert float(pt.regen_force_limit(1.0, r)) == pytest.approx(230 * 4 / r)
    assert float(Powertrain(regen_enabled=False).regen_force_limit(10.0, r)) == 0.0
