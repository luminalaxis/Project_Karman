import dataclasses

import pytest
import yaml
from pydantic import ValidationError

from fsk_lapsim.aero import components_from_config, synthesize
from fsk_lapsim.config import (
    Param,
    assumed_parameters,
    load_aero_components_config,
    load_events_config,
    load_vehicle_config,
    resolve_path,
)
from fsk_lapsim.vehicle import (
    Vehicle,
    apply_component_change,
    get_param,
    load_vehicle,
    vehicle_from_config,
    with_param,
    with_params,
)


def test_param_scalar_shorthand_is_not_assumed():
    p = Param.model_validate(3.0)
    assert p.value == 3.0 and not p.assumed


def test_param_rejects_bool():
    with pytest.raises(ValidationError):
        Param.model_validate(True)


def test_vehicle_yaml_matches_spec_assumptions(baseline: Vehicle):
    assert baseline.mass_kg == 320.0
    assert baseline.wheelbase_m == 1.55
    assert (baseline.track_front_m, baseline.track_rear_m) == (1.22, 1.20)
    assert baseline.cg_height_m == 0.28
    assert baseline.static_front_fraction == 0.47
    assert baseline.lltd_front == 0.52
    assert baseline.tire.mu0 == 1.45 and baseline.tire.fz0_n == 700.0
    assert baseline.tire.load_sensitivity == -0.15
    assert baseline.powertrain.max_power_w == 80_000.0
    assert baseline.aero.cla_m2 == 3.5 and baseline.aero.cda_m2 == 1.4
    assert baseline.aero.balance_front == 0.45


def test_all_spec_placeholders_flagged_assumed():
    cfg = load_vehicle_config()
    assumed = dict(assumed_parameters(cfg))
    # every numeric vehicle input is an assumption from the spec's [ ] values
    all_params = dict(cfg.iter_params())
    assert set(assumed) == set(all_params)
    assert "aero.cla_m2" in assumed and assumed["aero.cla_m2"].unit == "m^2"


def test_components_reproduce_lumped_aero(baseline: Vehicle):
    comps = components_from_config(load_aero_components_config())
    aero = synthesize(comps, baseline.wheelbase_m, 1.2)
    assert aero.cla_m2 == pytest.approx(3.5)
    assert aero.cda_m2 == pytest.approx(1.4)
    assert aero.balance_front == pytest.approx(0.45, abs=1e-6)


def test_component_change_updates_mass_and_cg(baseline: Vehicle):
    comps = components_from_config(load_aero_components_config())
    same = apply_component_change(baseline, comps, comps)
    assert same.mass_kg == pytest.approx(baseline.mass_kg)
    assert same.cg_height_m == pytest.approx(baseline.cg_height_m)
    assert same.static_front_fraction == pytest.approx(baseline.static_front_fraction)
    # heavier, higher rear wing: mass up, CG up, weight moves rearwards
    heavy = [dataclasses.replace(c, mass_kg=c.mass_kg + 2.0) if c.name == "rear_wing" else c
             for c in comps]
    v2 = apply_component_change(baseline, comps, heavy)
    assert v2.mass_kg == pytest.approx(322.0)
    assert v2.cg_height_m == pytest.approx((320 * 0.28 + 2.0 * 0.95) / 322.0)
    assert v2.static_front_fraction < baseline.static_front_fraction


def test_components_file_in_vehicle_yaml(tmp_path):
    src = resolve_path("vehicle.yaml")
    data = yaml.safe_load(src.read_text(encoding="utf-8"))
    data["aero"]["components_file"] = str(resolve_path("aero_components.yaml"))
    data["aero"]["cla_m2"] = 1.0  # overridden by the components
    p = tmp_path / "veh.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    v = load_vehicle(p)
    assert v.aero.cla_m2 == pytest.approx(3.5)


def test_with_param_and_aliases(baseline: Vehicle):
    v = with_param(baseline, "cla", 4.0)
    assert v.aero.cla_m2 == 4.0 and baseline.aero.cla_m2 == 3.5
    v = with_params(baseline, {"tire.mu0": 1.6, "mass_kg": 300.0})
    assert get_param(v, "mu") == 1.6 and get_param(v, "mass") == 300.0
    with pytest.raises(AttributeError):
        with_param(baseline, "aero.nope", 1.0)
    with pytest.raises(AttributeError):
        with_param(baseline, "mass_kg.x", 1.0)


def test_events_config():
    ev = load_events_config()
    assert ev.acceleration_length_m.value == 75.0
    assert ev.skidpad_radius_m.value == 9.125


def test_resolve_path_missing():
    with pytest.raises(FileNotFoundError):
        resolve_path("does/not/exist.yaml")


def test_vehicle_from_config_roundtrip():
    v = vehicle_from_config(load_vehicle_config())
    assert v.a_m == pytest.approx(1.55 * 0.53)
    assert v.b_m == pytest.approx(1.55 * 0.47)
    assert v.v_max_mps == pytest.approx(6500 * 2 * 3.141592653589793 / 60 / 4 * 0.228)
