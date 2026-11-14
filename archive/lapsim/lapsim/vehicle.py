"""Vehicle model: mass/geometry, tyres, aero and an EV powertrain.

Sign conventions
----------------
* ``ClA`` is the *downforce* coefficient times reference area (positive = downforce).
* Aero is a set of named devices (e.g. front / mid / rear); totals, the front balance and the
  drag height are derived from them.
* ``balance_front`` is the share of downforce acting on the front axle (0..1).
* ``a_x`` > 0 is acceleration, < 0 is braking.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

G = 9.81


@dataclass
class Tyre:
    mu_x: float = 1.55          # peak longitudinal friction coefficient at Fz0
    mu_y: float = 1.60          # peak lateral friction coefficient at Fz0
    load_sensitivity: float = 0.10  # fractional mu drop per +100% wheel load over Fz0
    Fz0: float = 700.0          # reference wheel load [N]


@dataclass
class AeroDevice:
    """One aero package (front wing, undertray/diffuser, rear wing, body ...).

    Position the downforce with either ``x_cp`` (centre of pressure behind the front
    axle [m]; negative = ahead of it, > wheelbase = behind the rear axle) or directly
    with ``front_share`` (fraction of this device's downforce on the front axle).
    """
    ClA: float = 0.0
    CdA: float = 0.0
    x_cp: float | None = None
    h_cp: float = 0.35          # height of this device's drag force [m]
    front_share: float | None = None
    # design space for target optimisation (lapsim.targets); all optional
    ClA_min: float | None = None   # default 0
    ClA_max: float | None = None   # default 2 x current ClA
    CdA0: float | None = None      # drag at zero downforce [m^2]
    k_induced: float | None = None  # CdA = CdA0 + k_induced * ClA^2
    induced_share: float = 0.5     # used to derive CdA0/k_induced when they are not given
    mass_per_cla: float = 0.0      # device mass change per unit ClA change [kg]
    # ride-height aero map (optional): relative change of ClA / CdA as a polynomial in the
    # front/rear ride-height change dhf, dhr [mm] from ``rh_ref`` (default: static ride heights)
    #   factor = 1 + c[0]*dhf + c[1]*dhr + c[2]*dhf^2 + c[3]*dhr^2 + c[4]*dhf*dhr
    rh_cl: list | None = None
    rh_cd: list | None = None
    rh_ref: list | None = None     # [front, rear] ride height of the map reference [mm]

    def share(self, wheelbase: float) -> float:
        if self.front_share is not None:
            return self.front_share
        if self.x_cp is None:
            raise ValueError("aero device needs x_cp or front_share")
        return 1.0 - self.x_cp / wheelbase

    # ---- device polar -------------------------------------------------------
    @property
    def tunable(self) -> bool:
        """Devices that make downforce (or declare a ClA_max) are varied by the optimiser."""
        return self.ClA > 0 or (self.ClA_max or 0) > 0

    def polar(self) -> tuple[float, float]:
        """(CdA0, k): CdA(ClA) = CdA0 + k*ClA^2, anchored on the current operating point."""
        if self.CdA0 is not None and self.k_induced is not None:
            return self.CdA0, self.k_induced
        if self.ClA <= 0:
            return self.CdA, (self.k_induced or 0.0)
        if self.k_induced is not None:
            return max(self.CdA - self.k_induced * self.ClA**2, 0.0), self.k_induced
        if self.CdA0 is not None:
            return self.CdA0, max(self.CdA - self.CdA0, 0.0) / self.ClA**2
        induced = self.induced_share * self.CdA
        return self.CdA - induced, induced / self.ClA**2

    def cda_at(self, cla: float) -> float:
        cd0, k = self.polar()
        return cd0 + k * cla**2

    def cla_bounds(self) -> tuple[float, float]:
        lo = 0.0 if self.ClA_min is None else self.ClA_min
        hi = 2.0 * self.ClA if self.ClA_max is None else self.ClA_max
        return lo, max(hi, lo)


@dataclass
class Aero:
    """Sum of the aero devices.  ``ClA``, ``CdA``, ``balance_front`` and ``h_cp`` are derived."""
    devices: dict = field(default_factory=lambda: {
        "total": AeroDevice(ClA=3.5, CdA=1.3, front_share=0.45, h_cp=0.35)})
    rho: float = 1.2            # air density [kg/m^3]
    wheelbase: float = 1.53     # kept in sync by Vehicle, used to turn x_cp into a share

    @property
    def ClA(self) -> float:
        return sum(d.ClA for d in self.devices.values())

    @property
    def CdA(self) -> float:
        return sum(d.CdA for d in self.devices.values())

    @property
    def balance_front(self) -> float:
        cla = self.ClA
        if cla == 0:
            return 0.5
        return sum(d.ClA * d.share(self.wheelbase) for d in self.devices.values()) / cla

    @property
    def h_cp(self) -> float:
        """Drag-weighted height of the total drag force."""
        cda = self.CdA
        return sum(d.CdA * d.h_cp for d in self.devices.values()) / cda if cda else 0.0

    def table(self) -> str:
        rows = [f"  {'device':<10}{'ClA':>7}{'CdA':>7}{'L/D':>7}{'front %':>9}"]
        for name, d in self.devices.items():
            ld = f"{d.ClA / d.CdA:7.2f}" if d.CdA else "      -"
            rows.append(f"  {name:<10}{d.ClA:7.2f}{d.CdA:7.2f}{ld}{d.share(self.wheelbase) * 100:9.1f}")
        rows.append(f"  {'total':<10}{self.ClA:7.2f}{self.CdA:7.2f}"
                    f"{(self.ClA / self.CdA if self.CdA else 0):7.2f}{self.balance_front * 100:9.1f}")
        return "\n".join(rows)

    @classmethod
    def from_dict(cls, data: dict) -> "Aero":
        data = dict(data)
        rho = data.pop("rho", 1.2)
        if "devices" in data:
            devices = {k: AeroDevice(**v) for k, v in data["devices"].items()}
        else:                                       # legacy single-number format
            devices = {"total": AeroDevice(ClA=data.get("ClA", 0.0), CdA=data.get("CdA", 0.0),
                                           front_share=data.get("balance_front", 0.45),
                                           h_cp=data.get("h_cp", 0.35))}
        return cls(devices=devices, rho=rho)


@dataclass
class Powertrain:
    max_power: float = 80_000.0  # accumulator power limit [W] (FS rules: 80 kW)
    max_torque: float = 140.0    # motor torque [Nm]
    max_rpm: float = 6500.0      # motor speed limit [rpm]
    gear_ratio: float = 4.0
    wheel_radius: float = 0.2    # [m]
    efficiency: float = 0.90     # accumulator -> wheel

    @property
    def v_max(self) -> float:
        """Road speed at the motor rpm limit [m/s]."""
        return self.max_rpm * 2 * np.pi / 60 * self.wheel_radius / self.gear_ratio


@dataclass
class Suspension:
    """Lateral load transfer and ride height.

    * Lateral load transfer: per-wheel  dFz_axle = LLTD_axle * m * a_y * h / track_axle, where the
      front share LLTD comes from ``roll_stiffness_front/rear`` [Nm/deg] or directly ``lltd_front``.
    * Ride height: axle heave rate [N/mm] (wheel rate x 2, at the axle) lowers the static ride
      height under aero load and longitudinal transfer; 0 = rigid (ride height stays static).
    """
    track_front: float = 1.22          # [m]
    track_rear: float = 1.18           # [m]
    lltd_front: float | None = None    # front share of lateral load transfer (0..1)
    roll_stiffness_front: float | None = None   # [Nm/deg]
    roll_stiffness_rear: float | None = None    # [Nm/deg]
    ride_height_front: float = 30.0    # static [mm]
    ride_height_rear: float = 35.0     # static [mm]
    heave_rate_front: float = 0.0      # [N/mm] at the axle, 0 = rigid
    heave_rate_rear: float = 0.0       # [N/mm]
    min_ride_height: float = 8.0       # bump stop / plank [mm]

    @property
    def lltd(self) -> float:
        if self.roll_stiffness_front is not None and self.roll_stiffness_rear is not None:
            return self.roll_stiffness_front / (self.roll_stiffness_front + self.roll_stiffness_rear)
        return 0.5 if self.lltd_front is None else self.lltd_front


@dataclass
class Vehicle:
    name: str = "vehicle"
    mass: float = 280.0               # incl. driver [kg]
    cg_height: float = 0.28           # [m]
    wheelbase: float = 1.53           # [m]
    weight_dist_front: float = 0.46   # static front axle load share
    rolling_resistance: float = 0.015
    tyre: Tyre = field(default_factory=Tyre)
    aero: Aero = field(default_factory=Aero)
    powertrain: Powertrain = field(default_factory=Powertrain)
    suspension: Suspension | None = None   # None = no lateral load transfer, rigid ride height
    yaw_inertia: float = 0.0               # Iz [kg m^2]; 0 = no yaw-acceleration demand

    def __post_init__(self):
        self.aero.wheelbase = self.wheelbase

    # ---- helpers ------------------------------------------------------------
    def with_changes(self, **changes) -> "Vehicle":
        """Copy with dotted-path overrides.

        e.g. ``with_changes(**{"mass": 270, "aero.rear.ClA": 1.6, "tyre.mu_y": 1.7})``;
        aero devices are addressed by name (``aero.<device>.<field>``).
        """
        v = copy.deepcopy(self)
        for key, value in changes.items():
            obj = v
            *parents, attr = key.split(".")
            for p in parents:
                if isinstance(obj, Aero) and p in obj.devices:
                    obj = obj.devices[p]
                else:
                    obj = getattr(obj, p)
            if isinstance(obj, Aero) and attr in ("ClA", "CdA", "balance_front", "h_cp"):
                raise AttributeError(f"{key} is derived from the devices; set aero.<device>.{attr}, "
                                     "use scaled_aero() or with_balance()")
            if not hasattr(obj, attr):
                raise AttributeError(f"unknown vehicle parameter: {key}")
            setattr(obj, attr, value)
        v.aero.wheelbase = v.wheelbase
        return v

    def scaled_aero(self, cl_scale=1.0, cd_scale=1.0, devices=None) -> "Vehicle":
        """Scale ClA/CdA of the given devices (default: all)."""
        v = copy.deepcopy(self)
        names = list(v.aero.devices) if devices is None else list(devices)
        for n in names:
            if n not in v.aero.devices:
                raise KeyError(f"unknown aero device {n!r}; have {list(v.aero.devices)}")
            d = v.aero.devices[n]
            d.ClA *= cl_scale
            d.CdA *= cd_scale
        return v

    def with_balance(self, balance: float, front: str | None = None, rear: str | None = None) -> "Vehicle":
        """Re-trim to a target front aero balance at constant total ClA.

        Scales the front-most and rear-most devices (wing angle change): their CdA scales with
        their ClA, so a balance change also changes drag.  A single-device car just moves the
        downforce share.
        """
        v = copy.deepcopy(self)
        devs, L = v.aero.devices, v.wheelbase
        if len(devs) == 1:
            next(iter(devs.values())).front_share = balance
            return v
        lifting = {k: d for k, d in devs.items() if d.ClA != 0}
        front = front or max(lifting, key=lambda k: lifting[k].share(L))
        rear = rear or min(lifting, key=lambda k: lifting[k].share(L))
        f, r = devs[front], devs[rear]
        sf, sr = f.share(L), r.share(L)
        if abs(sf - sr) < 1e-9:
            raise ValueError("front and rear devices have the same position; cannot re-balance")
        total = v.aero.ClA
        fixed = total - f.ClA - r.ClA
        fixed_mom = sum(d.ClA * d.share(L) for k, d in devs.items() if k not in (front, rear))
        # solve  F + R = total - fixed ;  F*sf + R*sr = balance*total - fixed_mom
        F = (balance * total - fixed_mom - sr * (total - fixed)) / (sf - sr)
        R = total - fixed - F
        if F < 0 or R < 0:
            raise ValueError(f"balance {balance:.3f} not reachable with {front}/{rear} at ClA {total:.2f}")
        for d, new in ((f, F), (r, R)):
            k = new / d.ClA
            d.ClA, d.CdA = new, d.CdA * k
        return v

    def to_dict(self) -> dict:
        return asdict(self)


def load_vehicle(path) -> Vehicle:
    return vehicle_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def vehicle_from_dict(data: dict) -> Vehicle:
    build = {"tyre": lambda d: Tyre(**d), "aero": Aero.from_dict,
             "powertrain": lambda d: Powertrain(**d),
             "suspension": lambda d: None if d is None else Suspension(**d)}
    kwargs = {k: build[k](v) if k in build else v for k, v in data.items()}
    return Vehicle(**kwargs)


def vehicle_to_dict(veh: Vehicle) -> dict:
    d = veh.to_dict()
    d["aero"].pop("wheelbase", None)
    if d.get("suspension") is None:
        d.pop("suspension", None)
    else:
        d["suspension"] = {k: v for k, v in d["suspension"].items() if v is not None}
    for name, dev in d["aero"]["devices"].items():
        d["aero"]["devices"][name] = {k: v for k, v in dev.items() if v is not None}
    return d


def save_vehicle(veh: Vehicle, path) -> None:
    Path(path).write_text(json.dumps(vehicle_to_dict(veh), indent=2, ensure_ascii=False) + "\n",
                          encoding="utf-8")
