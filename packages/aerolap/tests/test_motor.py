import math

import numpy as np
import pytest

from aerolap.config import with_overrides
from aerolap.motors import Motor
from aerolap.solvers.ggv import SolverOptions
from aerolap.solvers.qss import QSSSolver
from aerolap.track import Track
from aerolap.vehicle import VehicleModel


def _emrax(variant="MV", v=630.0, **kw):
    return Motor.from_config({"motor": "emrax_228", "motor_variant": variant, "motor_cooling": "LC",
                              "dc_voltage_V": v, **kw})


def test_emrax_228_matches_datasheet_peak_points():
    mv = _emrax("MV", 630.0)
    p = lambda m, rpm: float(m.max_torque(rpm)) * rpm * 2 * math.pi / 60 / 1000  # noqa: E731
    assert float(mv.max_torque(1000)) == pytest.approx(220.0)
    assert p(mv, 5500) == pytest.approx(124.0, rel=0.02)  # "124 kW at 5500 rpm"
    hv = _emrax("HV", 830.0)
    assert p(hv, 4500) == pytest.approx(104.0, rel=0.02)  # "104 kW at 4500 rpm" (voltage limited)
    assert float(mv.max_torque(6600)) == 0.0  # limiting speed 6500 rpm
    assert mv.rotor_inertia == pytest.approx(0.0398)


def test_lower_bus_voltage_moves_base_speed_down():
    full, low = _emrax("MV", 630.0), _emrax("MV", 450.0)
    assert low.base_rpm == pytest.approx(8.68 * 450.0)
    assert float(low.max_torque(5000)) < float(full.max_torque(5000))
    assert float(low.max_torque(3000)) == float(full.max_torque(3000))


def test_controller_current_limits_torque():
    m = _emrax("MV", 630.0, controller_current_A=250.0)
    assert m.torque_limit == pytest.approx(0.64 * 250.0)


def test_efficiency_map_shape():
    m = _emrax("MV", 630.0)
    assert float(m.efficiency(3000, 110)) == pytest.approx(0.96, abs=0.005)  # datasheet peak 96 %
    assert float(m.efficiency(1000, 20)) < 0.91  # 86-90 % region
    assert float(m.efficiency(500, 20)) < 0.87
    assert float(m.efficiency(2500, 100)) > float(m.efficiency(2500, 30))


def test_battery_power_limit_and_rotor_inertia(ev_cfg):
    car = VehicleModel(ev_cfg)
    pt = car.powertrain
    v = np.array([25.0, 30.0])
    f, _ = pt.max_force(v)
    assert np.allclose(pt.battery_power(v, f), 80e3, rtol=1e-3)  # 80 kW accumulator limit
    assert pt.m_rot == pytest.approx(0.0398 * 3.6 ** 2 / 0.203 ** 2, rel=1e-6)
    # rotor inertia costs acceleration when the motor (not the tyres) is the limit
    track = Track.straight(75.0)
    no_inertia = VehicleModel(with_overrides(ev_cfg, {"powertrain.ev.rotor_inertia_kgm2": 0.0}))
    t_with = QSSSolver(SolverOptions.coarse()).solve(car, track, start="standing").lap_time
    t_without = QSSSolver(SolverOptions.coarse()).solve(no_inertia, track, start="standing").lap_time
    assert t_with > t_without


def test_lap_energy_uses_motor_efficiency(ev_cfg):
    car = VehicleModel(ev_cfg)
    res = QSSSolver(SolverOptions.coarse()).solve(car, Track.straight(75.0), start="standing")
    ch = res.channels
    assert {"battery_power_kW", "motor_rpm", "motor_torque_Nm", "motor_efficiency"} <= set(ch)
    assert np.nanmax(ch["battery_power_kW"]) <= 80.0 * 1.01
    mech = np.nansum(ch["drive_power_kW"][:-1] * np.diff(res.t)) * 1000.0
    assert res.energy_J > mech  # losses on top of the energy at the wheels


def test_motor_peak_and_mean_power(ev_cfg):
    from aerolap.solvers.qss import QSSSolver
    from aerolap.track import Track
    from aerolap.vehicle import VehicleModel

    car = VehicleModel(ev_cfg)
    res = QSSSolver().solve(car, Track.straight(200.0), start="standing")
    ch = res.channels
    p = ch["motor_torque_Nm"] * ch["motor_rpm"] * 2 * np.pi / 60 * car.powertrain.n_motors / 1000
    st = res.summary()
    assert st["motor_peak_kW"] == pytest.approx(p.max())
    dt = np.diff(res.t)
    assert st["motor_mean_kW"] == pytest.approx((0.5 * (p[1:] + p[:-1]) * dt).sum() / dt.sum())
    assert st["motor_mean_kW"] <= st["motor_mean_drive_kW"] <= st["motor_peak_kW"]
    assert st["motor_elec_peak_kW"] > st["motor_peak_kW"]  # motor losses


def test_field_weakening_and_winding_temperature_cost_efficiency():
    base = Motor.from_config({"motor": "emrax_228", "dc_voltage_V": 550})
    fw = Motor.from_config({"motor": "emrax_228", "dc_voltage_V": 550, "field_weakening_loss": True})
    hot = Motor.from_config({"motor": "emrax_228", "dc_voltage_V": 550, "winding_temp_C": 100})
    assert float(fw.efficiency(6000, 30)) < float(base.efficiency(6000, 30))
    assert float(fw.efficiency(2000, 100)) == pytest.approx(float(base.efficiency(2000, 100)))  # below base speed
    assert float(hot.efficiency(3000, 110)) < float(base.efficiency(3000, 110))
    # one loss model: efficiency = P / (P + losses)
    p = 6000 * 30 * 2 * math.pi / 60
    assert float(fw.efficiency(6000, 30)) == pytest.approx(p / (p + float(fw.losses(6000, 30))))


def test_measured_efficiency_map(tmp_path):
    f = tmp_path / "map.csv"
    f.write_text("rpm,torque_Nm,efficiency\n0,0,80\n6500,0,85\n0,220,85\n6500,220,93\n3000,110,96\n")
    m = Motor.from_config({"motor": "emrax_228", "efficiency_map": str(f)})
    assert float(m.efficiency(3000, 110)) == pytest.approx(0.96)
    p = 3000 * 110 * 2 * math.pi / 60
    assert float(m.losses(3000, 110)) == pytest.approx(p * (1 / 0.96 - 1))
    assert np.shape(m.efficiency(np.array([1000.0, 2000.0]), 50.0)) == (2,)


def test_no_load_loss_heats_a_turning_motor_without_torque():
    from aerolap.config import ConfigError

    m = _emrax()
    assert float(m.losses(3000, 0)) == pytest.approx(float(m.no_load_loss(3000)))  # turning, no torque
    assert float(m.no_load_loss(3000)) > 0.0
    meas = _emrax(no_load_loss_W=[[6000, 1500], [0, 0], [3000, 450]])  # measured spin-down points, any order
    assert float(meas.losses(4500, 0)) == pytest.approx(975.0) and float(meas.losses(8000, 0)) == 1500.0
    cu = 3 * (110 / meas.kt) ** 2 * meas.r_phase
    assert float(meas.losses(3000, 110)) == pytest.approx(cu + 450.0)  # part of every drive loss too
    with pytest.raises(ConfigError):
        _emrax(no_load_loss_W=[[1000, -5]])


def test_energy_strategy_counts_the_coasting_motor_heat(ev_cfg):
    from aerolap.analysis.energy_strategy import LapDP

    from .conftest import DATA

    dp = LapDP(VehicleModel(ev_cfg), Track.from_file(DATA / "tracks" / "fsk2026_endurance_smoothed.csv"),
               SolverOptions.coarse())
    vm = np.array([15.0])
    pt = dp.car.powertrain
    coast = float(np.ravel(dp._motor_heat(vm, np.array([0.0])))[0])
    assert coast == pytest.approx(float(np.ravel(pt.motor.no_load_loss(pt.motor_rpm(vm)))[0]) * pt.n_motors)
    assert coast > 0.0
