"""Lateral load transfer, yaw inertia, ride-height aero, racing line, channels."""

import csv
import math

import numpy as np
import pytest

from lapsim import Vehicle, load_track, load_vehicle, simulate
from lapsim.channels import channels, delta_time, export_csv
from lapsim.line import best_line, box_qp, min_curvature_line
from lapsim.physics import Model
from lapsim.track import track_from_dict
from lapsim.vehicle import G, Suspension

ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent


def ring(radius=20.0, width=None, n=600):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    d = {"x": (radius * np.cos(th)).tolist(), "y": (radius * np.sin(th)).tolist(), "closed": True}
    if width:
        d["width"] = width
    return track_from_dict(d, smooth_m=1.0)


def bare(**kw):
    return Vehicle().scaled_aero(0, 0).with_changes(rolling_resistance=0.0, **kw)


@pytest.fixture(scope="module")
def car():
    return load_vehicle(ROOT / "cars" / "fsk_ev_baseline.json")


@pytest.fixture(scope="module")
def fsk():
    return load_track(ROOT / "tracks" / "fsk_2026_endurance.json")


# --- lateral load transfer ----------------------------------------------------
def test_load_transfer_costs_grip_only_with_load_sensitivity():
    trk = ring()
    susp = Suspension(lltd_front=0.5)
    with_lt = simulate(bare(suspension=susp), trk).v.mean()
    without = simulate(bare(), trk).v.mean()
    assert with_lt < without                                   # load sensitivity 0.1
    flat = {"tyre.load_sensitivity": 0.0}
    assert simulate(bare(suspension=susp).with_changes(**flat), trk).v.mean() == pytest.approx(
        simulate(bare().with_changes(**flat), trk).v.mean(), rel=1e-6)


def test_lltd_moves_the_limiting_axle():
    trk = ring()
    front_heavy = simulate(bare(suspension=Suspension(lltd_front=0.8)), trk)
    rear_heavy = simulate(bare(suspension=Suspension(lltd_front=0.2)), trk)
    assert set(front_heavy.limiter) == {"front"}
    assert set(rear_heavy.limiter) == {"rear"}


def test_roll_stiffness_defines_lltd():
    s = Suspension(roll_stiffness_front=300, roll_stiffness_rear=100)
    assert s.lltd == pytest.approx(0.75)


# --- yaw inertia ----------------------------------------------------------------
def test_yaw_inertia_does_not_change_steady_cornering():
    trk = ring()
    assert simulate(bare(yaw_inertia=150.0), trk).lap_time == pytest.approx(
        simulate(bare(), trk).lap_time, rel=1e-4)


def test_yaw_inertia_loads_front_on_entry(autocross=None):
    m = Model(Vehicle(yaw_inertia=100.0))
    cf_entry, cr_entry = m.lateral_coeffs(0.1, +0.02)      # curvature growing = turn-in
    cf_ss, cr_ss = m.lateral_coeffs(0.1, 0.0)
    assert cf_entry > cf_ss and cr_entry < cr_ss


# --- ride-height aero -------------------------------------------------------------
def test_ride_height_drops_and_ground_effect_grows(car):
    m = Model(car)
    _, _, _, hf_slow, hr_slow = m.loads(10.0, 0.0)
    _, _, _, hf_fast, hr_fast = m.loads(30.0, 0.0)
    assert hf_fast < hf_slow and hr_fast < hr_slow
    mid = {n: (f, d) for n, f, d in m.device_forces(30.0, hf_fast, hr_fast)}["mid"]
    q = 0.5 * car.aero.rho * 30.0**2
    assert mid[0] > q * car.aero.devices["mid"].ClA            # lower car -> more underbody load


def test_bump_stop(car):
    m = Model(car)
    _, _, _, hf, _ = m.loads(35.0, -3.0 * G)
    assert hf >= car.suspension.min_ride_height - 1e-9


# --- racing line ----------------------------------------------------------------------
def test_box_qp_matches_bruteforce():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(6, 3))
    Q, c = A.T @ A, rng.normal(size=3)
    lo, hi = -0.3 * np.ones(3), 0.4 * np.ones(3)
    x = box_qp(Q, c, lo, hi)
    grid = np.linspace(-0.3, 0.4, 36)
    best = min(0.5 * np.array(p) @ Q @ np.array(p) + c @ np.array(p)
               for p in np.array(np.meshgrid(grid, grid, grid)).reshape(3, -1).T)
    assert 0.5 * x @ Q @ x + c @ x <= best + 1e-6


def test_racing_line_on_a_ring_uses_the_outside():
    trk = ring(radius=20.0, width=6.0)
    line = min_curvature_line(trk, car_width=1.0, margin=0.0, smooth_m=1.0)
    r = np.hypot(line.x, line.y)
    assert np.median(r) == pytest.approx(22.5, abs=0.1)        # outer limit R + w/2 - half car
    car = bare()
    assert simulate(car, line).v.mean() > simulate(car, trk).v.mean()


def test_racing_lines_faster_on_fsk(car, fsk):
    line, times = best_line(car, fsk)
    centre = simulate(car, fsk).lap_time
    assert all(t < centre for t in times.values())
    assert simulate(car, line).lap_time == pytest.approx(min(times.values()))
    (xl, yl), (xr, yr) = fsk.edges()
    assert fsk.width_left.min() > 0 and fsk.width_right.min() > 0


# --- channels ---------------------------------------------------------------------------
def test_channels_consistent(car, fsk, tmp_path):
    res = simulate(car, fsk)
    ch = channels(car, res)
    wheels = ch["Fz_FL_N"] + ch["Fz_FR_N"] + ch["Fz_RL_N"] + ch["Fz_RR_N"]
    assert np.allclose(wheels, ch["Fz_front_N"] + ch["Fz_rear_N"])
    assert np.mean(ch["util_front"] <= 1.02) > 0.99 and np.mean(ch["util_rear"] <= 1.02) > 0.99
    assert ch["power_battery_kW"].max() <= car.powertrain.max_power / 1e3 + 1e-6
    assert ch["energy_battery_Wh"][-1] > 0
    f = tmp_path / "lap.csv"
    export_csv(ch, f)
    rows = list(csv.reader(open(f, encoding="utf-8")))
    assert rows[0] == list(ch) and len(rows) == len(res.s) + 1
    s, dt = delta_time(res, res)
    assert np.allclose(dt, 0)
