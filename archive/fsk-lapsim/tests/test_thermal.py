"""E3: loss models and lumped endurance heat."""

import math

import numpy as np
import pytest

from fsk_lapsim.events import EventSetup, run_endurance
from fsk_lapsim.thermal import LossModel, ThermalNode, endurance_heat, segment_losses


@pytest.fixture(scope="module")
def endu(baseline):
    st = EventSetup.load()
    return run_endurance(baseline, st.endurance_track, 3 * st.endurance_track.length_m)


def test_bundled_yaml_matches_defaults():
    assert LossModel.load() == LossModel()


def test_loss_formulas():
    lm = LossModel()
    w = 3000 * 2 * math.pi / 60
    assert float(lm.motor_loss_w(np.array([100.0]), np.array([3000.0]))[0]) == pytest.approx(
        0.066 * 100**2 + 0.156 * w**1.5 + 6.5e-4 * w**2
    )
    i = 100.0 / 0.77
    assert float(lm.inverter_loss_w(np.array([-100.0]))[0]) == pytest.approx(
        50 + 3 * i + 0.006 * i * i
    )
    assert float(lm.battery_loss_w(np.array([40_000.0]))[0]) == pytest.approx(0.1 * 100**2)
    eff = lm.motor_efficiency(np.array([230.0, 100.0, 0.0]), np.array([3300.0, 4000.0, 1000.0]))
    assert 0.93 < eff[0] < 0.96 and 0.94 < eff[1] < 0.97 and eff[2] == 0.0


def test_endurance_heat_bookkeeping(endu):
    lm = LossModel()
    h = endurance_heat(endu, lm)
    tab = h.table.set_index("component")
    assert set(tab.index) == {"motor", "inverter", "battery"}
    seg = segment_losses(endu.flying_lap.energy, lm)
    first = segment_losses(endu.first_lap.energy, lm)
    q_motor = np.sum(first["motor_w"] * first["dt_s"]) + (endu.n_laps - 1) * np.sum(
        seg["motor_w"] * seg["dt_s"]
    )
    assert tab.loc["motor", "heat_kj"] == pytest.approx(q_motor / 1e3)
    total_t = endu.total_time_s
    assert h.trace["t_s"].iloc[-1] == pytest.approx(total_t)
    assert tab.loc["motor", "mean_kw"] == pytest.approx(q_motor / total_t / 1e3)
    r = tab.loc["inverter"]
    assert r["required_r_k_per_w"] == pytest.approx((r["limit_c"] - 30.0) / (r["mean_kw"] * 1e3))
    assert (tab["max_temp_c"] >= 30.0).all()
    assert 0.85 < h.implied_motor_inverter_efficiency < 0.99


def test_lumped_model_approaches_steady_state(endu):
    # τ = 0.2 ms ≪ segment duration (~20 ms): temperature follows each segment's power
    fast = LossModel(motor=ThermalNode(0.01, 0.02, 140.0))
    h = endurance_heat(endu, fast)
    tab = h.table.set_index("component")
    p_last = h.trace["motor_w"].iloc[-1]
    assert h.trace["motor_c"].iloc[-1] == pytest.approx(30.0 + p_last * 0.02, rel=1e-6)
    slow = LossModel(motor=ThermalNode(1e9, 0.02, 140.0))
    assert endurance_heat(endu, slow).table.set_index("component").loc[
        "motor", "end_temp_c"
    ] == pytest.approx(30.0, abs=0.01)
    assert tab.loc["motor", "time_constant_s"] == pytest.approx(2e-4)
