"""Energy, power and motor operating points from a solved speed profile.

Per segment (between nodes i and i+1, v² linear in s):

* mean square speed  v̄² = (v_i² + v_{i+1}²)/2  → exact segment averages of drag and
  downforce (both ∝ v²);
* required net tyre force  Fx = m·ax + F_drag + F_roll  (ax = Δ(v²)/(2·ds));
* Fx > 0: drive.  Accumulator energy = Fx·ds/η, motor torque T = Fx·r/(G·η);
* Fx < 0: braking.  The rear-axle share (ideal: proportional to axle load; fixed: 1 − bias)
  can be recovered by the motor up to the torque limit and the regen power cap; recovered
  accumulator energy = η·F_regen·ds. The rest is friction-brake heat.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .models import make_model
from .vehicle import Vehicle

FloatArray = NDArray[np.float64]


@dataclass
class EnergyBreakdown:
    """Segment arrays (length N) and lap totals.

    Attributes:
        dt_s: Segment duration [s].
        v_mean_mps: RMS segment speed √(v̄²) [m/s].
        ax_mps2: Segment acceleration [m/s²].
        fx_tyre_n: Net tyre longitudinal force required [N] (+ drive, − brake).
        drag_n: Aero drag [N].
        rolling_n: Rolling resistance [N].
        regen_force_n: Regenerative braking force at the wheels [N].
        p_battery_w: Accumulator power [W] (+ discharge, − regen).
        motor_rpm: Motor speed [rpm].
        motor_torque_nm: Motor torque [N·m] (+ drive, − regen).
        battery_drive_j: Accumulator energy spent on driving [J].
        regen_j: Accumulator energy recovered [J].
        work_drive_j: Positive tyre work [J].
        work_brake_j: Braking tyre work (positive number) [J].
        work_drag_j: Drag work [J].
        work_rolling_j: Rolling resistance work [J].
        delta_ke_j: Kinetic energy change over the lap [J].
    """

    dt_s: FloatArray
    v_mean_mps: FloatArray
    ax_mps2: FloatArray
    fx_tyre_n: FloatArray
    drag_n: FloatArray
    rolling_n: FloatArray
    regen_force_n: FloatArray
    p_battery_w: FloatArray
    motor_rpm: FloatArray
    motor_torque_nm: FloatArray
    battery_drive_j: float
    regen_j: float
    work_drive_j: float
    work_brake_j: float
    work_drag_j: float
    work_rolling_j: float
    delta_ke_j: float

    @property
    def battery_net_j(self) -> float:
        """Net accumulator energy [J] (drive − regen)."""
        return self.battery_drive_j - self.regen_j

    @property
    def friction_brake_j(self) -> float:
        """Energy dissipated in the friction brakes [J]."""
        return self.work_brake_j - float(np.sum(self.regen_force_n * self.v_mean_mps * self.dt_s))


def lap_energy(
    vehicle: Vehicle,
    s: FloatArray,
    v: FloatArray,
    kappa: FloatArray,
    alt_vehicle: Vehicle | None = None,
    alt_weight: FloatArray | None = None,
    model_name: str = "axle",
) -> EnergyBreakdown:
    """Compute the energy breakdown of a speed profile (see module docstring)."""
    m = vehicle.mass_kg
    pt = vehicle.powertrain
    r = vehicle.tire.radius_m
    ds = np.diff(s)
    v0, v1 = v[:-1], v[1:]
    v2m = 0.5 * (v0 * v0 + v1 * v1)
    vm = np.sqrt(v2m)
    dt = 2.0 * ds / np.maximum(v0 + v1, 1e-9)
    ax = (v1 * v1 - v0 * v0) / (2.0 * ds)
    ay = v2m * 0.5 * (kappa[:-1] + kappa[1:])
    rho = vehicle.aero.air_density_kg_m3
    cla = np.full_like(ds, vehicle.aero.cla_m2)
    cda = np.full_like(ds, vehicle.aero.cda_m2)
    if alt_vehicle is not None and alt_weight is not None:
        wseg = 0.5 * (alt_weight[:-1] + alt_weight[1:])
        cla = (1.0 - wseg) * cla + wseg * alt_vehicle.aero.cla_m2
        cda = (1.0 - wseg) * cda + wseg * alt_vehicle.aero.cda_m2
    drag = 0.5 * rho * cda * v2m
    down = 0.5 * rho * cla * v2m
    roll = vehicle.tire.rolling_resistance_coeff * (vehicle.weight_n + down)
    fx = m * ax + drag + roll
    drive = np.maximum(fx, 0.0)
    brake = np.maximum(-fx, 0.0)

    if vehicle.brake_distribution == "fixed":
        rear_share = np.full_like(ds, 1.0 - vehicle.brake_front_bias)
    else:
        loads = make_model(vehicle, model_name).wheel_loads(vm, ax, ay)
        front, rear = loads[0] + loads[1], loads[2] + loads[3]
        rear_share = rear / np.maximum(front + rear, 1e-9)
    regen = np.minimum(brake * rear_share, pt.regen_force_limit(vm, r))
    eta = pt.efficiency
    e_drive = drive * ds / eta
    e_regen = eta * regen * ds
    p_batt = (e_drive - e_regen) / np.maximum(dt, 1e-12)
    rpm = pt.motor_rpm(vm, r)
    torque = np.where(fx > 0.0, drive * r / (pt.gear_ratio * eta), -regen * r / pt.gear_ratio)
    return EnergyBreakdown(
        dt_s=dt,
        v_mean_mps=vm,
        ax_mps2=ax,
        fx_tyre_n=fx,
        drag_n=drag,
        rolling_n=roll,
        regen_force_n=regen,
        p_battery_w=p_batt,
        motor_rpm=rpm,
        motor_torque_nm=torque,
        battery_drive_j=float(e_drive.sum()),
        regen_j=float(e_regen.sum()),
        work_drive_j=float(np.sum(drive * ds)),
        work_brake_j=float(np.sum(brake * ds)),
        work_drag_j=float(np.sum(drag * ds)),
        work_rolling_j=float(np.sum(roll * ds)),
        delta_ke_j=0.5 * m * (float(v[-1]) ** 2 - float(v[0]) ** 2),
    )


def operating_point_histogram(
    energy: EnergyBreakdown,
    rpm_bins: int = 26,
    torque_bins: int = 24,
    rpm_max: float | None = None,
    torque_max: float | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Time-weighted histogram of motor operating points.

    Returns ``(hist_s, rpm_edges, torque_edges)``; ``hist_s[i, j]`` is the time [s] spent in
    rpm bin i and torque bin j (negative torque = regen).
    """
    rmax = rpm_max or max(float(energy.motor_rpm.max()), 1.0)
    tmax = torque_max or max(float(np.abs(energy.motor_torque_nm).max()), 1.0)
    rpm_edges = np.linspace(0.0, rmax * 1.0001, rpm_bins + 1)
    tq_edges = np.linspace(-tmax * 1.0001, tmax * 1.0001, torque_bins + 1)
    hist, _, _ = np.histogram2d(
        energy.motor_rpm, energy.motor_torque_nm, bins=[rpm_edges, tq_edges], weights=energy.dt_s
    )
    return hist, rpm_edges, tq_edges


def rms(values: FloatArray, dt: FloatArray) -> float:
    """Time-weighted RMS of a segment quantity."""
    return math.sqrt(float(np.sum(values * values * dt) / np.sum(dt)))
