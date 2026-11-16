"""Powertrain: maximum tractive force at the contact patches, battery power and rotating inertia.

EV options (``powertrain.ev``):

* ``motor: emrax_228`` + ``motor_variant`` / ``motor_cooling`` / ``dc_voltage_V`` /
  ``controller_current_A`` - torque-speed envelope, efficiency and rotor inertia from the motor
  database (:mod:`aerolap.motors`), or
* ``motor_torque_curve: [[rpm, Nm], ...]`` + ``max_rpm`` - a plain torque curve.

``power_limit_kW`` is the accumulator (battery-side) limit, e.g. 80 kW in Formula Student; with a cell
pack the effective limit is ``min(power_limit_kW, deliverable pack power)``. ``powertrain.efficiency`` is
the drivetrain (gears, chain) efficiency: the wheel force is ``T G eta / r``; with a motor model the battery
power is ``T omega / (eta_motor(n, T) eta_inverter)`` on top (default 0.97), without one ``P_wheel / eta``
(default 0.9, OptimumLap's single "drive efficiency").

Regeneration (``battery.regen_power_kW`` > 0, EV): the motor brakes the driven axle with as much of the
braking demand as its torque, the battery regen power and (in :meth:`VehicleModel.grip_usage`) the grip
of that axle allow; the friction brakes add the rest by ``brakes.bias_front``. Regen therefore moves the
brake balance towards the driven axle.

Battery (``battery:``): with a ``cell`` section the pack model of :mod:`aerolap.battery` sets the
deliverable power and the bus voltage (motor base speed) at ``battery.soc``; otherwise
``capacity_kWh`` x ``usable_fraction`` is the energy available for a run
(endurance check in :mod:`aerolap.analysis.energy`); ``regen_power_kW`` > 0 recovers part of the
braking on the driven axle (see below), limited by the motor torque
and that power, through drivetrain, motor and inverter efficiency.
"""

from __future__ import annotations

import math

import numpy as np

from .battery import BatteryPack
from .motors import Motor


def _curve(points, name: str):
    try:
        arr = np.asarray(points, dtype=float)
    except (TypeError, ValueError):
        arr = np.zeros((0, 2))
    if arr.ndim != 2 or arr.shape[1] != 2 or len(arr) < 2:
        from .config import ConfigError

        raise ConfigError(path=name, value=points, condition="[rpm, 토크 Nm] 쌍이 2개 이상인 목록")
    order = np.argsort(arr[:, 0])
    return arr[order, 0], arr[order, 1]


class Powertrain:
    def __init__(self, cfg: dict, tire_radius: float, battery: dict | None = None):
        self.type = cfg["type"]
        eff = cfg.get("efficiency")
        motor_model = cfg["type"] == "ev" and bool((cfg.get("ev") or {}).get("motor"))
        self.efficiency = float(eff) if eff is not None else (0.97 if motor_model else 0.9)
        self.drive = cfg["drive"]
        self.differential = cfg["differential"]
        frac = cfg.get("awd_front_fraction")
        self.awd_front_fraction = None if frac is None else float(frac)
        lock = cfg.get("lsd_locking")
        self.lsd_locking = None if lock is None else float(lock)
        self.torque_vectoring = bool(cfg.get("torque_vectoring", False))
        self.radius = tire_radius
        self.shift_time = 0.0
        self.motor: Motor | None = None
        self.n_motors = 1
        self.inverter_efficiency = 1.0
        self.power_limit = None
        self.m_rot = 0.0  # kg, rotating driveline inertia reflected to the contact patches
        b = battery or {}
        cap = b.get("capacity_kWh")
        self.battery_capacity = None if cap is None else float(cap) * 3.6e6  # J
        self.usable_fraction = float(b.get("usable_fraction", 0.95))
        self.pack_power = None
        self.regen_power = float(b.get("regen_power_kW") or 0.0) * 1000.0
        self.pack = BatteryPack.from_config(b)  # cell-based pack model (None: fixed limits only)
        self.pack_voltage = None  # terminal voltage at full power (pack model)
        if self.type == "ice":
            c = cfg["ice"]
            self.rpm, self.torque = _curve(c["torque_curve"], "powertrain.ice.torque_curve")
            self.rev_limit = float(c.get("rev_limit_rpm", self.rpm[-1]))
            primary = float(c.get("primary_ratio", 1.0))
            final = float(c["final_drive"])
            self.ratios = np.array([primary * float(g) * final for g in c["gear_ratios"]])
            self.shift_time = float(c.get("shift_time_s", 0.0))
            top = self.ratios.min()
            self.v_max = self.rev_limit * 2.0 * math.pi / 60.0 / top * tire_radius
            return
        c = cfg["ev"]
        g = float(c["reduction_ratio"])
        self.ratios = np.array([g])
        self.n_motors = int(c.get("n_motors", 1))
        pl = c.get("power_limit_kW")
        self.power_limit = None if pl is None else float(pl) * 1000.0
        if self.pack is not None:  # deliverable power at this SOC (voltage sag, current limits)
            self.pack_power = self.pack.max_power()
        if self.pack_power is not None:
            self.power_limit = self.pack_power if self.power_limit is None else min(self.power_limit,
                                                                                  self.pack_power)
        if self.pack is not None:
            self.pack_voltage = float(self.pack.terminal_voltage(self.power_limit))
        if c.get("motor"):
            if self.pack_voltage is not None:  # bus voltage at full power sets the motor base speed
                c = {**c, "dc_voltage_V": self.pack_voltage}
            self.motor = Motor.from_config(c)
            self.inverter_efficiency = float(c.get("inverter_efficiency", 0.97))
            self.rev_limit = self.motor.max_rpm
            inertia = self.motor.rotor_inertia
        else:
            self.rpm, self.torque = _curve(c["motor_torque_curve"], "powertrain.ev.motor_torque_curve")
            self.rev_limit = float(c.get("max_rpm", self.rpm[-1]))
            inertia = float(c.get("rotor_inertia_kgm2") or 0.0)
        self.m_rot = self.n_motors * inertia * g * g / tire_radius ** 2
        self.v_max = self.rev_limit * 2.0 * math.pi / 60.0 / g * tire_radius

    # ------------------------------------------------------------------ helpers
    def motor_rpm(self, v):
        return np.maximum(np.asarray(v, dtype=float), 0.0) / self.radius * self.ratios[0] * 60.0 / (2.0 * math.pi)

    def _motor_torque_for_force(self, f_wheel):
        return np.asarray(f_wheel, dtype=float) * self.radius / (self.ratios[0] * self.efficiency * self.n_motors)

    def max_force(self, v):
        """Maximum tractive force (N) and the gear index (0-based) that delivers it."""
        v = np.asarray(v, dtype=float)
        if self.motor is not None:
            rpm = self.motor_rpm(v)
            t = self.motor.max_torque(rpm)
            if self.power_limit is not None:
                omega = np.maximum(rpm * 2.0 * math.pi / 60.0, 1e-6)
                eta = np.full(np.shape(t), 0.95)
                for _ in range(3):  # battery limit: T omega n / (eta_m eta_inv) <= P_lim
                    t_p = self.power_limit * eta * self.inverter_efficiency / (omega * self.n_motors)
                    eta = self.motor.efficiency(rpm, np.minimum(t, t_p))
                t = np.minimum(t, t_p)
            f = t * self.n_motors * self.ratios[0] * self.efficiency / self.radius
            return f, np.zeros(np.shape(f), dtype=int)
        omega_wheel = np.maximum(v, 0.0) / self.radius
        rpm = omega_wheel[..., None] * self.ratios * 60.0 / (2.0 * math.pi)  # (..., n_gears)
        torque = np.interp(np.maximum(rpm, self.rpm[0]), self.rpm, self.torque)
        force = np.where(rpm <= self.rev_limit, torque * self.ratios * self.efficiency / self.radius, 0.0)
        gear = np.argmax(force, axis=-1)
        f = np.take_along_axis(force, gear[..., None], axis=-1)[..., 0]
        if self.type == "ev" and self.power_limit is not None:
            f = np.minimum(f, self.power_limit * self.efficiency / np.maximum(v, 0.1))
        return f, gear

    def motor_point(self, v, f_motor):
        """(rpm, torque per motor, motor efficiency) for a force ``f_motor`` at the wheels."""
        rpm = self.motor_rpm(v)
        t = np.maximum(self._motor_torque_for_force(f_motor), 0.0)
        eta = self.motor.efficiency(rpm, t) if self.motor is not None else np.full(np.shape(t), 1.0)
        return rpm, t, eta

    def battery_power(self, v, f_motor):
        """Power drawn from the battery / tank (W) to deliver ``f_motor`` (>= 0) at the wheels."""
        f = np.maximum(np.asarray(f_motor, dtype=float), 0.0)
        p_wheel = f * np.maximum(np.asarray(v, dtype=float), 0.0)
        if self.motor is None:
            return p_wheel / self.efficiency
        _, _, eta = self.motor_point(v, f)
        return p_wheel / (self.efficiency * eta * self.inverter_efficiency)

    @property
    def usable_energy(self) -> float | None:
        """Energy available for a run (J), or None without a battery capacity."""
        if self.pack is not None:
            return self.pack.usable_energy
        if self.battery_capacity is None:
            return None
        return self.battery_capacity * self.usable_fraction

    def _regen_eta(self, v, f):
        """(battery-side efficiency, per-motor torque) of a regen force ``f`` at the wheels."""
        if self.motor is None:
            return np.full(np.shape(f), self.efficiency), np.zeros(np.shape(f))
        rpm = self.motor_rpm(v)
        t = f * self.radius / (self.ratios[0] * self.n_motors)
        return self.motor.efficiency(rpm, t) * self.inverter_efficiency * self.efficiency, t

    @property
    def has_regen(self) -> bool:
        return self.type == "ev" and self.regen_power > 0.0

    def regen_force(self, v, f_brake, cap=None):
        """Regenerative braking force at the wheels (N >= 0) for a braking demand ``f_brake`` (N): all of
        it, limited by the motor torque, the battery regen power and ``cap`` (what the driven axle's
        tyres can take)."""
        f_brake = np.maximum(np.asarray(f_brake, dtype=float), 0.0)
        if not self.has_regen:
            return np.zeros(np.shape(f_brake))
        v = np.maximum(np.asarray(v, dtype=float), 0.0)
        f = f_brake if cap is None else np.minimum(f_brake, np.maximum(cap, 0.0))
        if self.motor is not None:
            t_max = self.motor.max_torque(self.motor_rpm(v)) * self.n_motors
            f = np.minimum(f, t_max * self.ratios[0] / self.radius)
        eta, _ = self._regen_eta(v, f)
        p_cap = self.regen_power / np.maximum(v * eta, 1e-6)  # battery power limit as a wheel force
        return np.minimum(f, p_cap)

    def regen_battery_power(self, v, f_brake, f_regen=None):
        """Power returned to the battery (W >= 0) while braking with ``f_brake`` (N) at the tyres, or
        for a known regen force ``f_regen``."""
        v = np.maximum(np.asarray(v, dtype=float), 0.0)
        f = self.regen_force(v, f_brake) if f_regen is None else np.maximum(np.asarray(f_regen, dtype=float), 0.0)
        eta, _ = self._regen_eta(v, f)
        return np.minimum(f * v * eta, self.regen_power)

    def gear_lookup(self, v_max: float, step: float = 0.05):
        """Pre-computed best-gear table for fast scalar lookups in the speed integration."""
        grid = np.arange(0.0, v_max + step, step)
        _, gear = self.max_force(grid)
        return step, gear.tolist()
