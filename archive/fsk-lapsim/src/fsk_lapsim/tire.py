"""Load-sensitive effective tyre model with a friction ellipse.

Per wheel:  μ(Fz) = μ0 · (1 + k · (Fz − Fz0) / Fz0),   F_max(Fz) = μ(Fz) · Fz.

With k < 0 the curve F_max(Fz) is a downward parabola with its peak at
Fz_peak = Fz0 · (1 − k) / (−2k) (2683 N for the default k = −0.15, Fz0 = 700 N).
Beyond the peak the parabola would *lose* force with more load, which is unphysical, so the
force is held constant at F_max(Fz_peak) (see DECISIONS.md D-07). Negative loads (wheel lift)
give zero force.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class Tire:
    """Effective tyre parameters (identical for all four wheels).

    Attributes:
        radius_m: Loaded rolling radius [m].
        mu0: Lateral friction coefficient at the reference load ``fz0_n`` [-].
        fz0_n: Reference vertical load [N].
        load_sensitivity: k in μ(Fz) = μ0·(1 + k·(Fz − Fz0)/Fz0) [-] (negative = degressive).
        mu_lon_lat_ratio: μx / μy, the friction-ellipse aspect ratio [-].
        rolling_resistance_coeff: Rolling resistance coefficient Crr [-].
    """

    radius_m: float = 0.228
    mu0: float = 1.45
    fz0_n: float = 700.0
    load_sensitivity: float = -0.15
    mu_lon_lat_ratio: float = 1.0
    rolling_resistance_coeff: float = 0.015

    @property
    def peak_load_n(self) -> float:
        """Load [N] at which F_max(Fz) peaks (``inf`` if k >= 0)."""
        k = self.load_sensitivity
        if k >= 0.0:
            return float("inf")
        return self.fz0_n * (1.0 - k) / (-2.0 * k)

    def mu(self, fz_n: ArrayLike) -> NDArray[np.float64]:
        """Lateral friction coefficient μ(Fz) [-] (raw formula, no clamping)."""
        fz = np.asarray(fz_n, dtype=float)
        return self.mu0 * (1.0 + self.load_sensitivity * (fz - self.fz0_n) / self.fz0_n)

    def max_lateral_force(self, fz_n: ArrayLike) -> NDArray[np.float64]:
        """Peak lateral force F_y,max = μ(Fz)·Fz [N], clamped to be non-decreasing in Fz."""
        fz = np.clip(np.asarray(fz_n, dtype=float), 0.0, self.peak_load_n)
        return self.mu(fz) * fz

    def max_longitudinal_force(self, fz_n: ArrayLike) -> NDArray[np.float64]:
        """Peak longitudinal force F_x,max = (μx/μy)·F_y,max [N]."""
        return self.mu_lon_lat_ratio * self.max_lateral_force(fz_n)


def ellipse_remaining(fx_max: ArrayLike, fy_max: ArrayLike, fy: ArrayLike) -> NDArray[np.float64]:
    """Longitudinal force still available on a friction ellipse [N].

    Solves (Fx/Fx_max)² + (Fy/Fy_max)² = 1 for Fx. Returns NaN where |Fy| > Fy_max
    (state not feasible) and 0 where Fy_max is 0 and Fy is 0.
    """
    fx_max = np.asarray(fx_max, dtype=float)
    fy_max = np.asarray(fy_max, dtype=float)
    fy = np.abs(np.asarray(fy, dtype=float))
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.where(fy_max > 0.0, fy / fy_max, np.where(fy > 0.0, np.inf, 0.0))
    rem = fx_max * np.sqrt(np.clip(1.0 - u * u, 0.0, None))
    return np.where(u <= 1.0, rem, np.nan)
