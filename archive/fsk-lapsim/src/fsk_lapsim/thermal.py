"""Motor / inverter / accumulator losses and a lumped thermal estimate (extension E3).

Loss models (per segment of the lap, ω = motor speed [rad/s], T = motor torque [N·m]):

    motor     P = k_cu·T² + k_fe·ω^n + k_w·ω²
    inverter  P = P0 + a·I + b·I²,     I = |T| / k_t   (phase current)
    battery   P = R_pack · (P_batt / V_pack)²

Each component is a single thermal mass C with a resistance R to the coolant/ambient:

    C·dθ/dt = P_loss − (θ − θ_amb)/R

integrated over the endurance (first lap + flying laps). The result gives total heat, mean
and peak heat rates, end temperature and the thermal resistance that would keep the
component at its limit in steady state, R_req = (θ_limit − θ_amb)/P_mean.

These are post-processing estimates: the lap simulation itself keeps the constant overall
efficiency of the spec (3.5). The implied efficiency of the loss model is reported for
comparison.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .config import default_config_dir, load_yaml
from .energy import EnergyBreakdown
from .events import EnduranceResult

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ThermalNode:
    """Lumped thermal mass."""

    heat_capacity_j_per_k: float
    thermal_resistance_k_per_w: float
    temperature_limit_c: float


@dataclass(frozen=True)
class LossModel:
    """Loss and thermal parameters (see ``thermal.yaml``)."""

    k_cu: float = 0.066
    k_fe: float = 0.156
    fe_exp: float = 1.5
    k_w: float = 6.5e-4
    inv_p0: float = 50.0
    inv_a: float = 3.0
    inv_b: float = 0.006
    k_t: float = 0.77
    v_pack: float = 400.0
    r_pack: float = 0.10
    motor: ThermalNode = ThermalNode(6000.0, 0.02, 140.0)
    inverter: ThermalNode = ThermalNode(1500.0, 0.03, 90.0)
    accumulator: ThermalNode = ThermalNode(45000.0, 0.05, 60.0)
    ambient_c: float = 30.0

    @classmethod
    def load(cls, path: str | Path | None = None) -> LossModel:
        """Load ``thermal.yaml`` (default: bundled)."""
        d = (
            load_yaml(path)
            if path is not None
            else load_yaml(default_config_dir() / "thermal.yaml")
        )

        def v(sec: dict, key: str) -> float:
            x = sec[key]
            return float(x["value"] if isinstance(x, dict) else x)

        def node(sec: dict) -> ThermalNode:
            return ThermalNode(
                v(sec, "heat_capacity_j_per_k"),
                v(sec, "thermal_resistance_k_per_w"),
                v(sec, "temperature_limit_c"),
            )

        m, i, a = d["motor"], d["inverter"], d["accumulator"]
        return cls(
            k_cu=v(m, "copper_loss_coeff_w_per_nm2"),
            k_fe=v(m, "iron_loss_coeff"),
            fe_exp=v(m, "iron_loss_exponent"),
            k_w=v(m, "windage_coeff_w_per_rad2s2"),
            inv_p0=v(i, "idle_loss_w"),
            inv_a=v(i, "linear_loss_w_per_a"),
            inv_b=v(i, "resistive_loss_ohm"),
            k_t=v(i, "torque_constant_nm_per_a"),
            v_pack=v(a, "pack_voltage_v"),
            r_pack=v(a, "pack_resistance_ohm"),
            motor=node(m),
            inverter=node(i),
            accumulator=node(a),
            ambient_c=float(
                d["ambient_c"]["value"] if isinstance(d["ambient_c"], dict) else d["ambient_c"]
            ),
        )

    def motor_loss_w(self, torque_nm: FloatArray, rpm: FloatArray) -> FloatArray:
        """Motor loss [W] at the operating points."""
        w = np.asarray(rpm, float) * 2.0 * math.pi / 60.0
        t = np.abs(np.asarray(torque_nm, float))
        return self.k_cu * t * t + self.k_fe * np.power(w, self.fe_exp) + self.k_w * w * w

    def inverter_loss_w(self, torque_nm: FloatArray) -> FloatArray:
        """Inverter loss [W] from the phase current |T|/k_t."""
        i = np.abs(np.asarray(torque_nm, float)) / self.k_t
        return self.inv_p0 + self.inv_a * i + self.inv_b * i * i

    def battery_loss_w(self, p_batt_w: FloatArray) -> FloatArray:
        """Accumulator I²R loss [W]."""
        i = np.asarray(p_batt_w, float) / self.v_pack
        return self.r_pack * i * i

    def motor_efficiency(self, torque_nm: FloatArray, rpm: FloatArray) -> FloatArray:
        """Motoring efficiency P_mech/(P_mech + P_loss) of the motor alone."""
        w = np.asarray(rpm, float) * 2.0 * math.pi / 60.0
        p = np.abs(np.asarray(torque_nm, float)) * w
        loss = self.motor_loss_w(torque_nm, rpm)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(p > 0, p / (p + loss), 0.0)


def segment_losses(energy: EnergyBreakdown, lm: LossModel) -> pd.DataFrame:
    """Per-segment losses [W] and durations of one lap."""
    return pd.DataFrame(
        {
            "dt_s": energy.dt_s,
            "motor_w": lm.motor_loss_w(energy.motor_torque_nm, energy.motor_rpm),
            "inverter_w": lm.inverter_loss_w(energy.motor_torque_nm),
            "battery_w": lm.battery_loss_w(energy.p_battery_w),
            "p_mech_w": np.abs(energy.motor_torque_nm)
            * energy.motor_rpm
            * 2
            * math.pi
            / 60
            * np.sign(energy.motor_torque_nm),
        }
    )


@dataclass
class HeatResult:
    """Endurance heat estimate.

    Attributes:
        table: One row per component: total heat, mean/peak power, end temperature, limit,
            required thermal resistance.
        trace: Time series of losses and temperatures over the endurance.
        implied_motor_inverter_efficiency: Mechanical energy / (mechanical + motor + inverter
            losses) while driving, to compare with the constant efficiency of the lap model.
    """

    table: pd.DataFrame
    trace: pd.DataFrame
    implied_motor_inverter_efficiency: float


def endurance_heat(endu: EnduranceResult, lm: LossModel | None = None) -> HeatResult:
    """Losses and lumped temperatures over the whole endurance (first + flying laps)."""
    lm = lm or LossModel.load()
    first = segment_losses(endu.first_lap.energy, lm)
    fly = segment_losses(endu.flying_lap.energy, lm)
    seq = pd.concat([first] + [fly] * (endu.n_laps - 1), ignore_index=True)
    t = np.concatenate([[0.0], np.cumsum(seq["dt_s"].to_numpy())])
    rows = []
    trace = {"t_s": t[1:]}
    for name, node in (("motor", lm.motor), ("inverter", lm.inverter), ("battery", lm.accumulator)):
        p = seq[f"{name}_w"].to_numpy()
        dt = seq["dt_s"].to_numpy()
        theta = np.empty_like(p)
        th = lm.ambient_c
        for k in range(p.size):  # exact exponential step for a constant power in the step
            tau = node.heat_capacity_j_per_k * node.thermal_resistance_k_per_w
            th_inf = lm.ambient_c + p[k] * node.thermal_resistance_k_per_w
            th = th_inf + (th - th_inf) * math.exp(-dt[k] / tau)
            theta[k] = th
        q = float(np.sum(p * dt))
        p_mean = q / t[-1]
        rows.append(
            {
                "component": name,
                "heat_kj": q / 1e3,
                "mean_kw": p_mean / 1e3,
                "peak_kw": float(p.max()) / 1e3,
                "end_temp_c": float(theta[-1]),
                "max_temp_c": float(theta.max()),
                "limit_c": node.temperature_limit_c,
                "steady_temp_c": lm.ambient_c + p_mean * node.thermal_resistance_k_per_w,
                "required_r_k_per_w": (node.temperature_limit_c - lm.ambient_c) / p_mean
                if p_mean > 0
                else math.inf,
                "assumed_r_k_per_w": node.thermal_resistance_k_per_w,
                "time_constant_s": node.heat_capacity_j_per_k * node.thermal_resistance_k_per_w,
            }
        )
        trace[f"{name}_w"] = p
        trace[f"{name}_c"] = theta
    drive = seq["p_mech_w"].to_numpy() > 0
    dt = seq["dt_s"].to_numpy()
    e_mech = float(np.sum(seq["p_mech_w"].to_numpy()[drive] * dt[drive]))
    e_loss = float(np.sum((seq["motor_w"] + seq["inverter_w"]).to_numpy()[drive] * dt[drive]))
    eff = e_mech / (e_mech + e_loss) if e_mech > 0 else math.nan
    return HeatResult(pd.DataFrame(rows), pd.DataFrame(trace), eff)
