"""Uncertainty definitions (``uncertainty.yaml``) shared by Monte Carlo and priority ranking."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..config import default_config_dir, load_yaml
from ..vehicle import Vehicle, get_param, with_params


@dataclass(frozen=True)
class Dist:
    """A distribution around a nominal value.

    ``normal``: N(nominal, sigma). ``uniform``: nominal + U(low, high) for vehicle parameters,
    or U(low, high) absolute for the wing-conversion factors. Samples are clipped to ``clip``.
    """

    kind: str
    sigma: float = 0.0
    low: float = 0.0
    high: float = 0.0
    clip: tuple[float, float] | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Dist:
        """Parse one YAML entry."""
        kind = d.get("dist", "normal")
        if kind not in ("normal", "uniform"):
            raise ValueError(f"unknown distribution {kind!r}")
        clip = tuple(d["clip"]) if "clip" in d else None
        return cls(
            kind,
            float(d.get("sigma", 0.0)),
            float(d.get("low", 0.0)),
            float(d.get("high", 0.0)),
            clip,
        )  # type: ignore[arg-type]

    @property
    def std(self) -> float:
        """Equivalent standard deviation."""
        if self.kind == "normal":
            return self.sigma
        return (self.high - self.low) / math.sqrt(12.0)

    def sample(
        self, rng: np.random.Generator, nominal: float, n: int, absolute: bool = False
    ) -> np.ndarray:
        """Draw ``n`` samples."""
        if self.kind == "normal":
            x = rng.normal(nominal, self.sigma, n)
        else:
            base = 0.0 if absolute else nominal
            x = base + rng.uniform(self.low, self.high, n)
        if self.clip is not None:
            x = np.clip(x, *self.clip)
        return x


@dataclass
class Uncertainty:
    """Parsed ``uncertainty.yaml``."""

    parameters: dict[str, Dist]
    wing_conversion: dict[str, Dist] = field(default_factory=dict)
    cl_2d_rel_sigma: float = 0.0
    cd_2d_rel_sigma: float = 0.0

    @classmethod
    def load(cls, path: str | Path | None = None) -> Uncertainty:
        """Load the YAML file (default: bundled)."""
        data = (
            load_yaml(path)
            if path is not None
            else load_yaml(default_config_dir() / "uncertainty.yaml")
        )
        params = {k: Dist.from_dict(v) for k, v in (data.get("parameters") or {}).items()}
        wing = {k: Dist.from_dict(v) for k, v in (data.get("wing_conversion") or {}).items()}
        noise = data.get("candidate_noise") or {}
        return cls(
            params,
            wing,
            float(noise.get("cl_2d_rel_sigma", 0.0)),
            float(noise.get("cd_2d_rel_sigma", 0.0)),
        )

    def sigmas(self) -> dict[str, float]:
        """Equivalent 1σ of every vehicle parameter."""
        return {k: d.std for k, d in self.parameters.items()}

    def sample_vehicles(self, vehicle: Vehicle, n: int, rng: np.random.Generator) -> list[Vehicle]:
        """``n`` perturbed copies of ``vehicle`` (independent parameters)."""
        cols = {
            k: d.sample(rng, float(get_param(vehicle, k)), n) for k, d in self.parameters.items()
        }
        return [with_params(vehicle, {k: float(v[i]) for k, v in cols.items()}) for i in range(n)]
