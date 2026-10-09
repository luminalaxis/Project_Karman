"""Aerodynamic model: lumped coefficients and component synthesis.

Lumped model:  F_down = ½ρ·ClA·v²,  F_drag = ½ρ·CdA·v²,  front share of downforce = balance.

``balance_front`` is defined as the fraction of total downforce carried by the front axle
*including* the pitching moment of drag about the ground (the usual CFD/wind-tunnel
convention). The vehicle model therefore applies no separate drag pitching moment.

Component synthesis (each component i: ClA_i, CdA_i, x_cp,i, z_cp,i; x from the front axle,
positive rearwards, z above ground):

    ClA   = Σ ClA_i
    CdA   = Σ CdA_i
    ClA_f = Σ ClA_i·(1 − x_cp,i/L) − Σ CdA_i·z_cp,i/L
    balance = ClA_f / ClA
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .config import AeroComponentsConfig


@dataclass(frozen=True)
class Aero:
    """Lumped aero coefficients.

    Attributes:
        cla_m2: Downforce coefficient × reference area [m²] (positive = downforce).
        cda_m2: Drag coefficient × reference area [m²].
        balance_front: Fraction of downforce on the front axle [-].
        air_density_kg_m3: Air density ρ [kg/m³].
    """

    cla_m2: float = 3.5
    cda_m2: float = 1.4
    balance_front: float = 0.45
    air_density_kg_m3: float = 1.20

    def dynamic_pressure(self, v_mps: ArrayLike) -> NDArray[np.float64]:
        """Dynamic pressure q = ½ρv² [Pa]."""
        v = np.asarray(v_mps, dtype=float)
        return 0.5 * self.air_density_kg_m3 * v * v

    def downforce_n(self, v_mps: ArrayLike) -> NDArray[np.float64]:
        """Total downforce [N]."""
        return self.dynamic_pressure(v_mps) * self.cla_m2

    def drag_n(self, v_mps: ArrayLike) -> NDArray[np.float64]:
        """Aerodynamic drag [N]."""
        return self.dynamic_pressure(v_mps) * self.cda_m2


@dataclass(frozen=True)
class AeroComponent:
    """One aero component with its mass properties.

    Attributes:
        name: Identifier (e.g. ``rear_wing``).
        cla_m2: Downforce coefficient × area [m²].
        cda_m2: Drag coefficient × area [m²].
        x_cp_m: Downforce centre of pressure, rearwards from the front axle [m].
        z_cp_m: Height of the drag force above ground [m] (0 = not modelled).
        mass_kg: Component mass [kg].
        x_mass_m: Component centre of mass, rearwards from the front axle [m].
        z_mass_m: Component centre of mass height above ground [m].
    """

    name: str
    cla_m2: float
    cda_m2: float
    x_cp_m: float
    z_cp_m: float = 0.0
    mass_kg: float = 0.0
    x_mass_m: float = 0.0
    z_mass_m: float = 0.0

    def front_cla_m2(self, wheelbase_m: float) -> float:
        """Contribution of this component to the front-axle downforce coefficient [m²]."""
        lift_part = self.cla_m2 * (1.0 - self.x_cp_m / wheelbase_m)
        return lift_part - self.cda_m2 * self.z_cp_m / wheelbase_m

    def scaled(self, cl_factor: float = 1.0, cd_factor: float = 1.0) -> AeroComponent:
        """Copy with ClA and CdA scaled (e.g. an opened DRS flap)."""
        return AeroComponent(
            self.name,
            self.cla_m2 * cl_factor,
            self.cda_m2 * cd_factor,
            self.x_cp_m,
            self.z_cp_m,
            self.mass_kg,
            self.x_mass_m,
            self.z_mass_m,
        )


def synthesize(
    components: Iterable[AeroComponent], wheelbase_m: float, air_density_kg_m3: float = 1.2
) -> Aero:
    """Combine components into lumped coefficients (see module docstring)."""
    comps = list(components)
    cla = sum(c.cla_m2 for c in comps)
    cda = sum(c.cda_m2 for c in comps)
    cla_f = sum(c.front_cla_m2(wheelbase_m) for c in comps)
    balance = 0.5 if cla <= 0.0 else cla_f / cla
    return Aero(cla_m2=cla, cda_m2=cda, balance_front=balance, air_density_kg_m3=air_density_kg_m3)


def components_from_config(cfg: AeroComponentsConfig) -> list[AeroComponent]:
    """Build :class:`AeroComponent` objects from a validated YAML config."""
    out: list[AeroComponent] = []
    for c in cfg.components:
        x_mass = c.x_mass_m.value if c.x_mass_m is not None else c.x_cp_m.value
        out.append(
            AeroComponent(
                name=c.name,
                cla_m2=c.cla_m2.value,
                cda_m2=c.cda_m2.value,
                x_cp_m=c.x_cp_m.value,
                z_cp_m=c.z_cp_m.value,
                mass_kg=c.mass_kg.value,
                x_mass_m=x_mass,
                z_mass_m=c.z_mass_m.value,
            )
        )
    return out


def replace_component(
    components: Sequence[AeroComponent], new: AeroComponent, name: str | None = None
) -> list[AeroComponent]:
    """Copy of ``components`` with the one called ``name`` (default ``new.name``) swapped."""
    key = name or new.name
    if not any(c.name == key for c in components):
        raise KeyError(f"no aero component named {key!r}")
    return [new if c.name == key else c for c in components]
