"""YAML configuration schema (pydantic) and loaders.

Every numeric input is a :class:`Param`: a value plus an ``assumed`` flag, a unit and a note.
A plain number in YAML is accepted and treated as a *measured / given* value
(``assumed: false``). Values that are placeholders must be written as a mapping with
``assumed: true`` so that they show up in assumption listings and in the report.
"""

from __future__ import annotations

from collections.abc import Iterator
from importlib import resources
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Param(BaseModel):
    """A scalar input with provenance metadata."""

    model_config = ConfigDict(extra="forbid")

    value: float
    unit: str = ""
    assumed: bool = False
    note: str = ""

    @model_validator(mode="before")
    @classmethod
    def _from_scalar(cls, data: Any) -> Any:
        """Allow ``mass_kg: 320`` as shorthand for ``{value: 320, assumed: false}``."""
        if isinstance(data, bool):
            raise ValueError("boolean is not a valid numeric parameter")
        if isinstance(data, int | float):
            return {"value": float(data)}
        return data


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")

    def iter_params(self, prefix: str = "") -> Iterator[tuple[str, Param]]:
        """Yield ``(dotted_name, Param)`` for every parameter in this section (recursive)."""
        for name in type(self).model_fields:
            val = getattr(self, name)
            key = f"{prefix}{name}"
            if isinstance(val, Param):
                yield key, val
            elif isinstance(val, _Section):
                yield from val.iter_params(prefix=f"{key}.")


class MassConfig(_Section):
    """Total mass and centre of gravity."""

    total_mass_kg: Param
    cg_height_m: Param
    static_front_weight_fraction: Param


class GeometryConfig(_Section):
    """Wheelbase, track widths and lateral load transfer distribution."""

    wheelbase_m: Param
    track_front_m: Param
    track_rear_m: Param
    lltd_front: Param


class TireConfig(_Section):
    """Effective tyre model parameters (per wheel)."""

    radius_m: Param
    mu0: Param
    fz0_n: Param
    load_sensitivity: Param
    mu_lon_lat_ratio: Param = Field(default_factory=lambda: Param(value=1.0))
    rolling_resistance_coeff: Param


class PowertrainConfig(_Section):
    """Electric powertrain: accumulator power cap, motor, final drive, regen."""

    max_power_w: Param
    motor_max_torque_nm: Param
    motor_max_rpm: Param
    gear_ratio: Param
    efficiency: Param
    regen_max_power_w: Param
    regen_enabled: bool = True
    differential: Literal["lsd", "open"] = "lsd"


class BrakeConfig(_Section):
    """Brake force distribution."""

    distribution: Literal["ideal", "fixed"] = "ideal"
    front_bias: Param = Field(default_factory=lambda: Param(value=0.65, assumed=True))


class AeroConfig(_Section):
    """Lumped aero coefficients (or a component file that overrides them)."""

    cla_m2: Param
    cda_m2: Param
    balance_front: Param
    air_density_kg_m3: Param
    components_file: str | None = None


class VehicleConfig(_Section):
    """Top-level vehicle file (``vehicle.yaml``)."""

    name: str = "vehicle"
    description: str = ""
    mass: MassConfig
    geometry: GeometryConfig
    tire: TireConfig
    powertrain: PowertrainConfig
    brakes: BrakeConfig = Field(default_factory=BrakeConfig)
    aero: AeroConfig


class AeroComponentConfig(_Section):
    """One aero component; x positions are measured rearwards from the front axle."""

    name: str
    cla_m2: Param
    cda_m2: Param
    x_cp_m: Param
    z_cp_m: Param = Field(default_factory=lambda: Param(value=0.0))
    mass_kg: Param = Field(default_factory=lambda: Param(value=0.0))
    x_mass_m: Param | None = None
    z_mass_m: Param = Field(default_factory=lambda: Param(value=0.0))


class AeroComponentsConfig(_Section):
    """``aero_components.yaml``: the aero package as a list of components."""

    description: str = ""
    baseline_included_in_vehicle_mass: bool = True
    components: list[AeroComponentConfig]

    def iter_params(self, prefix: str = "") -> Iterator[tuple[str, Param]]:
        """Yield parameters of every component, keyed ``components.<name>.<field>``."""
        for comp in self.components:
            yield from comp.iter_params(prefix=f"{prefix}components.{comp.name}.")


class EventsConfig(_Section):
    """``events.yaml``: event definitions."""

    acceleration_length_m: Param
    skidpad_radius_m: Param
    autocross_track: str
    autocross_start: Literal["standing", "flying"] = "standing"
    endurance_track: str
    endurance_total_distance_m: Param
    ds_m: Param = Field(default_factory=lambda: Param(value=0.25))


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping at top level")
    return data


def default_config_dir() -> Path:
    """Directory of the configuration files bundled with the package."""
    return Path(str(resources.files("fsk_lapsim") / "configs"))


def default_data_dir() -> Path:
    """Directory of the data files bundled with the package (DUMMY CSVs, tracks, GPS)."""
    return Path(str(resources.files("fsk_lapsim") / "data"))


def resolve_path(path: str | Path, base: Path | None = None) -> Path:
    """Resolve ``path`` relative to ``base`` (if relative), then the bundled config/data dirs.

    Lookup order for a relative path: ``base / path``, ``cwd / path``, bundled ``configs/``,
    bundled ``data/``. Raises ``FileNotFoundError`` if nothing exists.
    """
    p = Path(path)
    if p.is_absolute():
        if p.exists():
            return p
        raise FileNotFoundError(p)
    candidates = []
    if base is not None:
        candidates.append(base / p)
    candidates += [Path.cwd() / p, default_config_dir() / p, default_data_dir() / p]
    for c in candidates:
        if c.exists():
            return c.resolve()
    raise FileNotFoundError(f"{path} (searched: {', '.join(str(c) for c in candidates)})")


def load_vehicle_config(path: str | Path | None = None) -> VehicleConfig:
    """Load and validate ``vehicle.yaml`` (default: the bundled example)."""
    p = resolve_path(path) if path is not None else default_config_dir() / "vehicle.yaml"
    return VehicleConfig.model_validate(_read_yaml(p))


def load_aero_components_config(path: str | Path | None = None) -> AeroComponentsConfig:
    """Load and validate ``aero_components.yaml`` (default: the bundled example)."""
    p = resolve_path(path) if path is not None else default_config_dir() / "aero_components.yaml"
    return AeroComponentsConfig.model_validate(_read_yaml(p))


def load_events_config(path: str | Path | None = None) -> EventsConfig:
    """Load and validate ``events.yaml`` (default: the bundled example)."""
    p = resolve_path(path) if path is not None else default_config_dir() / "events.yaml"
    return EventsConfig.model_validate(_read_yaml(p))


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load an arbitrary YAML mapping, resolving relative paths like :func:`resolve_path`."""
    return _read_yaml(resolve_path(path))


def assumed_parameters(*sections: _Section) -> list[tuple[str, Param]]:
    """Return every parameter flagged ``assumed: true`` in the given config sections."""
    out: list[tuple[str, Param]] = []
    for sec in sections:
        out += [(k, p) for k, p in sec.iter_params() if p.assumed]
    return out
