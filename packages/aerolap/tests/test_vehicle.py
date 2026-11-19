import numpy as np
import pytest

from aerolap.config import with_overrides
from aerolap.features import features_for_level
from aerolap.solvers.ggv import SolverOptions, compute_ggv
from aerolap.vehicle import FI, FO, RI, RO, G, VehicleModel


def test_load_sum_and_pitch_balance(simple_cfg):
    cfg = with_overrides(simple_cfg, {"aero.drag_height_mm": 500.0})
    car = VehicleModel(cfg)
    v, ax, ay = 25.0, -8.0, 6.0
    st = car.solve_state(v, ax, ay)
    fz = st.fz
    assert fz.sum() == pytest.approx(car.m * G + st.df_f + st.df_r, rel=1e-9)
    rear = fz[RO] + fz[RI]
    expected_rear = car.W_r + st.df_r + (car.m * ax * car.h + st.drag * car.h_drag) / car.L
    assert rear == pytest.approx(expected_rear, rel=1e-9)


def test_lateral_load_transfer_total(simple_cfg):
    car = VehicleModel(simple_cfg)
    ay = 12.0
    st = car.solve_state(5.0, 0.0, ay)
    tf, tr = car.front.track, car.rear.track
    moment = (st.fz[FO] - st.fz[FI]) * tf / 2 + (st.fz[RO] - st.fz[RI]) * tr / 2
    # m ay h plus the gravity moment of the rolled sprung mass
    expected = car.m * ay * car.h + car.m_s * G * car.d_roll * st.roll
    assert moment == pytest.approx(expected, rel=1e-6)


def test_roll_stiffness_distribution_sets_elastic_transfer(simple_raw):
    simple_raw["suspension"]["front"]["roll_center_mm"] = 0.0
    simple_raw["suspension"]["rear"]["roll_center_mm"] = 0.0
    from aerolap.config import vehicle_config_from_dict
    car = VehicleModel(vehicle_config_from_dict(simple_raw))
    st = car.solve_state(5.0, 0.0, 10.0)
    dz_f = (st.fz[FO] - st.fz[FI]) / 2 * car.front.track
    dz_r = (st.fz[RO] - st.fz[RI]) / 2 * car.rear.track
    assert dz_f / (dz_f + dz_r) == pytest.approx(car.K_f / (car.K_f + car.K_r), rel=1e-6)


def test_downforce_compresses_springs_and_tyres(simple_cfg):
    car = VehicleModel(simple_cfg)
    st = car.solve_state(30.0, 0.0, 0.0)
    fr = car.front
    expected = fr.static_ride_height - st.df_f / fr.heave_stiffness - st.df_f / (2 * fr.tire_rate)
    assert st.rh_f == pytest.approx(expected, rel=1e-9)
    assert st.roll == pytest.approx(0.0)


def test_wheel_lift_is_clamped(simple_cfg):
    car = VehicleModel(simple_cfg)
    st = car.solve_state(5.0, 0.0, 60.0)
    assert st.wheel_lift
    assert st.fz.min() >= 0.0


def test_point_mass_level_has_no_load_transfer(simple_cfg):
    car = VehicleModel(simple_cfg, features_for_level("L0"))
    st = car.solve_state(20.0, 5.0, 10.0)
    assert st.fz[FO] == pytest.approx(st.fz[FI])
    assert st.fz[RO] == pytest.approx(st.fz[RI])
    q = 0.5 * car.rho * 400.0
    assert st.fz[FO] + st.fz[FI] == pytest.approx(car.W_f + q * 3.0 * car.wd_front)


def test_braking_pitch_lowers_the_nose(ev_cfg):
    car = VehicleModel(ev_cfg)
    st0 = car.solve_state(25.0, 0.0, 0.0)
    st1 = car.solve_state(25.0, -18.0, 0.0)
    assert st1.rh_f < st0.rh_f
    assert st1.rh_r > st0.rh_r
    assert st1.pitch > 0.0


def test_roll_and_yaw_feed_the_aero_map(ev_cfg):
    l3 = VehicleModel(ev_cfg, features_for_level("L3")).solve_state(22.0, 0.0, 15.0)
    l4 = VehicleModel(ev_cfg, features_for_level("L4")).solve_state(22.0, 0.0, 15.0)
    assert l4.roll > 0.005 and abs(l4.yaw) > 0.01
    assert l4.cla < l3.cla  # the example map loses downforce in roll / yaw


def test_fixed_point_converges_on_whole_envelope(ev_cfg):
    car = VehicleModel(ev_cfg)
    opts = SolverOptions.coarse()
    g = compute_ggv(car, car.default_mode, opts)
    V, F = np.meshgrid(g.v, g.fracs, indexing="ij")
    for table in (g.ax_acc, g.ax_brk):
        st = car.solve_state(V, table, F * g.ay_max[:, None])
        assert st.converged.all()



def test_removed_keys_are_migrated_with_a_warning(simple_raw):
    from aerolap.config import vehicle_config_from_dict

    del simple_raw["powertrain"]["efficiency"]
    simple_raw["powertrain"]["ev"]["drivetrain_efficiency"] = 0.95
    simple_raw["battery"] = {"max_power_kW": 50.0}
    cfg = vehicle_config_from_dict(simple_raw)
    assert VehicleModel(cfg).powertrain.efficiency == 0.9  # no motor model: the ev key was never used
    assert cfg["powertrain"]["ev"]["power_limit_kW"] == 50.0  # min(60 rule limit, 50 pack)
    assert "max_power_kW" not in cfg["battery"]
    assert len(cfg["_warnings"]) == 2


@pytest.mark.parametrize("ev_extra,eff,expect", [
    ({"drivetrain_efficiency": 0.95}, 0.9, 0.95),  # motor model: the ev key was the one applied
    ({}, 0.9, 0.97),  # old GUI-saved default 0.9 was ignored by the motor model
    ({}, None, 0.97),
])
def test_motor_model_keeps_its_old_drivetrain_efficiency(ev_cfg, ev_extra, eff, expect):
    import copy

    from aerolap.config import vehicle_config_from_dict

    raw = {k: copy.deepcopy(v) for k, v in ev_cfg.items() if not k.startswith("_")}
    raw["powertrain"]["efficiency"] = eff
    raw["powertrain"]["ev"].update(ev_extra)
    cfg = vehicle_config_from_dict(raw, ev_cfg["_base_dir"])
    assert VehicleModel(cfg).powertrain.efficiency == pytest.approx(expect)


def test_anti_keys_belong_to_one_axle(simple_raw, tmp_path):
    import yaml

    from aerolap.config import load_vehicle_config

    simple_raw["suspension"]["front"]["anti_squat"] = 0.3
    p = tmp_path / "car.yaml"
    p.write_text(yaml.safe_dump(simple_raw))
    cfg = load_vehicle_config(p, warn=False)
    assert any("suspension.front.anti_squat" in w for w in cfg["_warnings"])


def test_inactive_keys_flag_keys_another_model_overrides():
    from aerolap.config import inactive_keys

    msgs = inactive_keys({"tires": {"mf_file": "x.tir", "peak_slip_deg": 7.0},
                          "aero": {"cla": 1.0, "cl": 2.0, "components": {"fl": {"type": "floor", "span_m": 1.0}}}})
    assert any("peak_slip_deg" in m for m in msgs)
    assert any("aero.cl " in m for m in msgs)
    assert any("span_m" in m and "floor" in m for m in msgs)
