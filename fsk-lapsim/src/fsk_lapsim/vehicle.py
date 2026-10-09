"""Runtime vehicle description (plain floats, hashable) and helpers to modify it."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .aero import Aero, AeroComponent, components_from_config, synthesize
from .config import (
    VehicleConfig,
    load_aero_components_config,
    load_vehicle_config,
    resolve_path,
)
from .constants import G
from .powertrain import Powertrain
from .tire import Tire


@dataclass(frozen=True)
class Vehicle:
    """Complete vehicle parameter set (SI units).

    Attributes:
        name: Label.
        mass_kg: Total mass including driver [kg].
        cg_height_m: Centre of gravity height above ground [m].
        static_front_fraction: Static front axle load / total weight [-].
        wheelbase_m: Wheelbase L [m].
        track_front_m: Front track width [m].
        track_rear_m: Rear track width [m].
        lltd_front: Front share of the total lateral load transfer [-].
        tire: Tyre model.
        aero: Lumped aero coefficients.
        powertrain: Powertrain.
        brake_distribution: ``"ideal"`` or ``"fixed"``.
        brake_front_bias: Front brake force fraction when ``brake_distribution == "fixed"``.
        g_mps2: Gravitational acceleration [m/s²].
    """

    name: str = "vehicle"
    mass_kg: float = 320.0
    cg_height_m: float = 0.28
    static_front_fraction: float = 0.47
    wheelbase_m: float = 1.55
    track_front_m: float = 1.22
    track_rear_m: float = 1.20
    lltd_front: float = 0.52
    tire: Tire = field(default_factory=Tire)
    aero: Aero = field(default_factory=Aero)
    powertrain: Powertrain = field(default_factory=Powertrain)
    brake_distribution: Literal["ideal", "fixed"] = "ideal"
    brake_front_bias: float = 0.65
    g_mps2: float = G

    @property
    def a_m(self) -> float:
        """Distance CG → front axle [m]."""
        return self.wheelbase_m * (1.0 - self.static_front_fraction)

    @property
    def b_m(self) -> float:
        """Distance CG → rear axle [m]."""
        return self.wheelbase_m * self.static_front_fraction

    @property
    def weight_n(self) -> float:
        """Weight m·g [N]."""
        return self.mass_kg * self.g_mps2

    @property
    def v_max_mps(self) -> float:
        """Speed at the motor rpm limit [m/s]."""
        return self.powertrain.v_max_mps(self.tire.radius_m)


_ALIASES: dict[str, str] = {
    "cla": "aero.cla_m2",
    "cda": "aero.cda_m2",
    "balance": "aero.balance_front",
    "mass": "mass_kg",
    "cg_height": "cg_height_m",
    "mu": "tire.mu0",
    "power": "powertrain.max_power_w",
    "rho": "aero.air_density_kg_m3",
}


def get_param(vehicle: Vehicle, path: str) -> Any:
    """Read a (dotted) parameter, e.g. ``aero.cla_m2``; short aliases such as ``cla`` work."""
    obj: Any = vehicle
    for part in _ALIASES.get(path, path).split("."):
        obj = getattr(obj, part)
    return obj


def with_param(vehicle: Vehicle, path: str, value: Any) -> Vehicle:
    """Return a copy of ``vehicle`` with the dotted parameter ``path`` set to ``value``."""
    parts = _ALIASES.get(path, path).split(".")

    def _set(obj: Any, keys: list[str]) -> Any:
        if not dataclasses.is_dataclass(obj):
            raise AttributeError(f"cannot set {path!r}: {type(obj).__name__} is not a dataclass")
        names = {f.name for f in dataclasses.fields(obj)}
        if keys[0] not in names:
            raise AttributeError(f"{type(obj).__name__} has no field {keys[0]!r} (in {path!r})")
        if len(keys) == 1:
            return dataclasses.replace(obj, **{keys[0]: value})
        return dataclasses.replace(obj, **{keys[0]: _set(getattr(obj, keys[0]), keys[1:])})

    return _set(vehicle, parts)


def with_params(vehicle: Vehicle, updates: dict[str, Any]) -> Vehicle:
    """Apply several :func:`with_param` updates."""
    for k, v in updates.items():
        vehicle = with_param(vehicle, k, v)
    return vehicle


def apply_component_change(
    vehicle: Vehicle,
    old_components: Sequence[AeroComponent],
    new_components: Sequence[AeroComponent],
) -> Vehicle:
    """Swap the aero package and update mass, CG height and weight distribution.

    ``vehicle`` is assumed to *include* ``old_components`` (mass and aero). The change is
    applied as a **delta** on top of the vehicle's current aero: ClA, CdA and the front-axle
    downforce coefficient change by (new − old) package values. When the vehicle aero equals
    the synthesis of ``old_components`` this is identical to re-synthesising, and it keeps
    perturbed or lumped-config aero values consistent (e.g. in Monte Carlo samples). Mass
    properties change by the component masses at their x/z positions.
    """
    L = vehicle.wheelbase_m
    m0 = vehicle.mass_kg
    x0 = vehicle.a_m  # CG rearwards from the front axle
    z0 = vehicle.cg_height_m
    dm = sum(c.mass_kg for c in new_components) - sum(c.mass_kg for c in old_components)
    dmx = sum(c.mass_kg * c.x_mass_m for c in new_components) - sum(
        c.mass_kg * c.x_mass_m for c in old_components
    )
    dmz = sum(c.mass_kg * c.z_mass_m for c in new_components) - sum(
        c.mass_kg * c.z_mass_m for c in old_components
    )
    m1 = m0 + dm
    if m1 <= 0.0:
        raise ValueError("component change gives non-positive vehicle mass")
    x1 = (m0 * x0 + dmx) / m1
    z1 = (m0 * z0 + dmz) / m1
    rho = vehicle.aero.air_density_kg_m3
    a_old, a_new = synthesize(old_components, L, rho), synthesize(new_components, L, rho)
    a = vehicle.aero
    cla = a.cla_m2 + a_new.cla_m2 - a_old.cla_m2
    front = a.balance_front * a.cla_m2 + (
        a_new.balance_front * a_new.cla_m2 - a_old.balance_front * a_old.cla_m2
    )
    if cla <= 0.0:
        raise ValueError("component change gives non-positive total ClA")
    aero = Aero(cla, a.cda_m2 + a_new.cda_m2 - a_old.cda_m2, front / cla, rho)
    return dataclasses.replace(
        vehicle,
        mass_kg=m1,
        cg_height_m=z1,
        static_front_fraction=1.0 - x1 / L,
        aero=aero,
    )


def vehicle_from_config(cfg: VehicleConfig, base_dir: Path | None = None) -> Vehicle:
    """Build a :class:`Vehicle` from a validated config.

    If ``aero.components_file`` is set, the aero coefficients are synthesised from the
    components (their masses are taken as already included in ``total_mass_kg``).
    """
    t = cfg.tire
    p = cfg.powertrain
    aero = Aero(
        cla_m2=cfg.aero.cla_m2.value,
        cda_m2=cfg.aero.cda_m2.value,
        balance_front=cfg.aero.balance_front.value,
        air_density_kg_m3=cfg.aero.air_density_kg_m3.value,
    )
    veh = Vehicle(
        name=cfg.name,
        mass_kg=cfg.mass.total_mass_kg.value,
        cg_height_m=cfg.mass.cg_height_m.value,
        static_front_fraction=cfg.mass.static_front_weight_fraction.value,
        wheelbase_m=cfg.geometry.wheelbase_m.value,
        track_front_m=cfg.geometry.track_front_m.value,
        track_rear_m=cfg.geometry.track_rear_m.value,
        lltd_front=cfg.geometry.lltd_front.value,
        tire=Tire(
            radius_m=t.radius_m.value,
            mu0=t.mu0.value,
            fz0_n=t.fz0_n.value,
            load_sensitivity=t.load_sensitivity.value,
            mu_lon_lat_ratio=t.mu_lon_lat_ratio.value,
            rolling_resistance_coeff=t.rolling_resistance_coeff.value,
        ),
        aero=aero,
        powertrain=Powertrain(
            max_power_w=p.max_power_w.value,
            motor_max_torque_nm=p.motor_max_torque_nm.value,
            motor_max_rpm=p.motor_max_rpm.value,
            gear_ratio=p.gear_ratio.value,
            efficiency=p.efficiency.value,
            regen_max_power_w=p.regen_max_power_w.value,
            regen_enabled=p.regen_enabled,
            differential=p.differential,
        ),
        brake_distribution=cfg.brakes.distribution,
        brake_front_bias=cfg.brakes.front_bias.value,
    )
    if cfg.aero.components_file:
        comp_path = resolve_path(cfg.aero.components_file, base_dir)
        comps = components_from_config(load_aero_components_config(comp_path))
        veh = dataclasses.replace(
            veh, aero=synthesize(comps, veh.wheelbase_m, aero.air_density_kg_m3)
        )
    return veh


def load_vehicle(path: str | Path | None = None) -> Vehicle:
    """Load ``vehicle.yaml`` (default: bundled example) into a :class:`Vehicle`."""
    cfg = load_vehicle_config(path)
    base = resolve_path(path).parent if path is not None else None
    return vehicle_from_config(cfg, base)
