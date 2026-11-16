"""Four-wheel vehicle model with suspension-coupled aerodynamics.

All methods are vectorised: ``v``, ``ax``, ``ay`` may be scalars or arrays of any
(broadcast-compatible) shape. ``ay`` is the magnitude of the lateral acceleration;
tyres are labelled by their position relative to the turn (outer / inner).

Physics are written as explicit *state -> forces/loads* functions
(:meth:`VehicleModel.evaluate`). The quasi-steady-state equilibrium
(:meth:`VehicleModel.solve_state`) is one consumer of these functions; a transient
solver can call :meth:`VehicleModel.evaluate` with dynamic ride heights / roll / yaw.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .aero.maps import AeroModel, disabled_mass
from .aero.strategy import StrategyConfig
from .environment import density_from_config
from .features import ModelFeatures
from .powertrain import Powertrain
from .suspension import Axle
from .tire import tire_from_config

G = 9.80665
FO, FI, RO, RI = 0, 1, 2, 3  # front-outer, front-inner, rear-outer, rear-inner
LIMIT_NAMES = ("front axle", "rear axle", "power")


@dataclass
class VehicleState:
    """Quasi-static vehicle state (all arrays share one shape; per-tyre arrays have a leading 4)."""

    v: np.ndarray
    ax: np.ndarray
    ay: np.ndarray
    mode: int
    rh_f: np.ndarray  # m, axle ride heights
    rh_r: np.ndarray
    roll: np.ndarray  # rad, body roll towards the outside of the turn
    yaw: np.ndarray  # rad, body sideslip angle (aero yaw)
    pitch: np.ndarray  # rad, nose-down positive, relative to static
    cla_f: np.ndarray
    cla_r: np.ndarray
    cda: np.ndarray
    df_f: np.ndarray  # N
    df_r: np.ndarray
    drag: np.ndarray
    fz: np.ndarray  # (4, ...) N
    camber: np.ndarray  # (4, ...) rad, wheel frame (negative = top towards chassis)
    fx_cap: np.ndarray  # (4, ...)
    fy_cap: np.ndarray  # (4, ...)
    clearance: np.ndarray  # m, minimum floor clearance (front/rear axle, roll included)
    alpha_f: np.ndarray  # rad
    alpha_r: np.ndarray
    steer: np.ndarray  # rad, road-wheel steer angle
    uy_f: np.ndarray  # lateral utilisation of each axle
    uy_r: np.ndarray
    u_front: np.ndarray  # combined utilisation of each axle (<= 1 feasible)
    u_rear: np.ndarray
    u_power: np.ndarray
    utilization: np.ndarray
    limit: np.ndarray  # index into LIMIT_NAMES
    understeer_index: np.ndarray  # uy_f / uy_r (> 1: front saturates first)
    fx_drive: np.ndarray  # N, tractive force demanded (>= 0)
    fx_brake: np.ndarray  # N, braking force demanded (>= 0)
    aero_out_of_range: np.ndarray
    bottoming: np.ndarray
    wheel_lift: np.ndarray
    converged: np.ndarray = field(default=None)  # type: ignore[assignment]
    plank: np.ndarray = field(default=None)  # type: ignore[assignment]  # N, floor contact force (both axles)
    fx_regen: np.ndarray = field(default=None)  # type: ignore[assignment]  # N, regenerative part of fx_brake

    @property
    def cla(self):
        return self.cla_f + self.cla_r

    @property
    def cop_front(self):
        total = self.cla
        return np.where(np.abs(total) > 1e-9, self.cla_f / np.where(total == 0, 1.0, total), np.nan)

    @property
    def feasible(self):
        return self.utilization <= 1.0


class VehicleModel:
    def __init__(self, cfg: dict, features: ModelFeatures | None = None):
        self.cfg = cfg
        self.features = features or ModelFeatures()
        self.name = str(cfg.get("name", "vehicle"))
        ms = cfg["mass"]
        geo = cfg["geometry"]
        # mass.total_kg is the car with every aero device; a device that is switched off takes its mass_kg along
        self.m = float(ms["total_kg"]) - disabled_mass((cfg.get("aero") or {}).get("components"))
        self.wd_front = float(ms["weight_dist_front"])
        self.L = float(geo["wheelbase_mm"]) / 1000.0
        self.a = self.L * (1.0 - self.wd_front)  # CG -> front axle
        self.b = self.L * self.wd_front  # CG -> rear axle
        self.h = float(ms["cg_height_mm"]) / 1000.0

        self.tire = tire_from_config(cfg["tires"], cfg.get("_base_dir", "."))
        gw = cfg["tires"].get("grip_window") or {}
        self.grip_window = (float(gw.get("optimum_C", 80.0)), float(gw.get("width_C", 50.0)),
                            float(gw.get("min_factor", 0.6)))
        t_tyre = cfg["tires"].get("temperature_C")
        # grip factor of each tyre (FO, FI, RO, RI order or scalar): the tyre temperature window
        self.tyre_grip = 1.0 if t_tyre is None else self.grip_temp_factor(float(t_tyre))
        r = self.tire.radius
        sus = cfg["suspension"]
        m_uf = float(ms["unsprung_front_kg"])
        m_ur = float(ms["unsprung_rear_kg"])
        self.front = Axle.from_config(sus["front"], float(geo["track_front_mm"]),
                                      self.tire.vertical_rate, m_uf, is_front=True)
        self.rear = Axle.from_config(sus["rear"], float(geo["track_rear_mm"]),
                                     self.tire.vertical_rate, m_ur, is_front=False)
        self.m_uf, self.m_ur = m_uf, m_ur
        self.m_s = self.m - m_uf - m_ur
        if self.m_s <= 0:
            from .config import ConfigError

            raise ConfigError(path="mass.unsprung_front_kg", value=m_uf,
                              paths=["mass.unsprung_front_kg", "mass.unsprung_rear_kg", "mass.total_kg"],
                              condition=f"앞+뒤 비현가 질량 ({m_uf:g} + {m_ur:g} kg) < 전체 질량 ({self.m:g} kg)")
        self.h_s = (self.m * self.h - (m_uf + m_ur) * r) / self.m_s
        self.a_s = (self.m * self.a - m_ur * self.L) / self.m_s  # sprung CG -> front axle
        self.b_s = self.L - self.a_s
        h_ra = self.front.roll_center + (self.rear.roll_center - self.front.roll_center) * self.a_s / self.L
        self.d_roll = self.h_s - h_ra  # sprung CG height above the roll axis
        self.K_f = self.front.roll_stiffness
        self.K_r = self.rear.roll_stiffness
        k_eff = self.K_f + self.K_r - self.m_s * G * self.d_roll
        if k_eff <= 0:
            from .config import ConfigError

            raise ConfigError(path="suspension.front.spring_rate_N_per_mm", has_value=False,
                              paths=["suspension.front.spring_rate_N_per_mm", "suspension.rear.spring_rate_N_per_mm",
                                     "suspension.front.arb_rate_Nm_per_deg", "suspension.rear.arb_rate_Nm_per_deg",
                                     "mass.cg_height_mm"],
                              condition=f"앞+뒤 롤 강성 ({(self.K_f + self.K_r) / 1000 * 3.14159 / 180:.2f} kNm/deg) > "
                                        f"현가질량 x g x 롤 팔 ({self.m_s * G * self.d_roll / 1000 * 3.14159 / 180:.2f} kNm/deg)",
                              message="롤 강성이 너무 낮아 차가 롤로 넘어가는 상태(불안정)입니다",
                              hint="스프링·ARB를 키우거나 무게중심을 낮추세요")
        self._k_roll_eff = k_eff

        self.gs = 1.0  # normal-gravity scale (bank / vertical curvature), scalar or per-state array

        # Transient-only parameters: sprung-mass inertias and damping.
        t_avg = 0.5 * (self.front.track + self.rear.track)
        ip, ir = ms.get("pitch_inertia_kgm2"), ms.get("roll_inertia_kgm2")
        self.I_pitch = float(ip) if ip is not None else self.m_s * (0.32 * self.L) ** 2
        self.I_roll = float(ir) if ir is not None else self.m_s * (0.28 * t_avg) ** 2
        iz = ms.get("yaw_inertia_kgm2")
        self.I_yaw = float(iz) if iz is not None else self.m * (0.4 * self.L) ** 2
        mf_corner = 0.5 * self.m_s * self.b_s / self.L
        mr_corner = 0.5 * self.m_s * self.a_s / self.L
        cf = self.front.wheel_damping(mf_corner)
        cr = self.rear.wheel_damping(mr_corner)
        self.C_heave_f, self.C_heave_r = 2.0 * cf, 2.0 * cr  # N s/m per axle
        self.C_roll_f = 0.5 * cf * self.front.track ** 2  # N m s/rad
        self.C_roll_r = 0.5 * cr * self.rear.track ** 2

        aero = cfg["aero"]
        from .aero.devices import geometry_from_cfg

        self.aero = AeroModel(aero, cfg.get("_base_dir", "."), geometry_from_cfg(cfg))
        self.h_drag = float(aero["drag_height_mm"]) / 1000.0
        self.floor_half_width = 0.5 * float(aero["floor_width_mm"]) / 1000.0
        self.bottoming_clearance = float(aero["bottoming_clearance_mm"]) / 1000.0
        self.cop_lateral = float(aero["cop_lateral_shift_mm_per_deg"]) / 1000.0 * 180.0 / math.pi  # m/rad
        self.cy_area = float(aero.get("side_force_area_m2_per_deg") or 0.0) * 180.0 / math.pi  # m^2 per rad
        xs = aero.get("side_force_x_mm")
        self.side_lever = 0.0 if xs is None else self.a - float(xs) / 1000.0  # m ahead of the CG
        kp = aero.get("plank_rate_N_per_mm")
        self.plank_rate = None if kp is None else float(kp) * 1000.0  # N/m per axle
        self.plank_mu = float(aero.get("plank_friction", 0.3))
        # bump stops / floor contact: the quasi-static attitude is solved iteratively
        self.nonlinear_ride = self.front.has_bump_stop or self.rear.has_bump_stop or self.plank_rate is not None
        self.strategy = StrategyConfig.from_config(aero["strategy"], self.aero)

        self.rho = density_from_config(cfg["environment"])
        self.powertrain = Powertrain(cfg["powertrain"], r, cfg.get("battery"))
        bias = cfg["brakes"].get("bias_front")
        self.brake_bias = None if bias is None else float(bias)

    def grip_temp_factor(self, temp_c):
        """Friction factor of the tyre temperature window, ``max(min, 1 - ((T - opt) / width)^2)``."""
        opt, width, lo = self.grip_window
        return np.maximum(lo, 1.0 - ((np.asarray(temp_c, dtype=float) - opt) / width) ** 2)

    @property
    def W_f(self):
        """Static front axle load (N), scaled by the normal-gravity factor ``gs``."""
        return self.m * G * self.gs * self.b / self.L

    @property
    def W_r(self):
        return self.m * G * self.gs * self.a / self.L

    # ------------------------------------------------------------------ info
    @property
    def mode_names(self) -> list[str]:
        return self.aero.mode_names

    @property
    def default_mode(self) -> int:
        return self.aero.default_mode

    def active_strategy(self) -> StrategyConfig:
        if self.features.active_aero:
            return self.strategy
        return StrategyConfig.fixed(self.default_mode)

    def ride_frequencies(self) -> tuple[float, float]:
        mf = 0.5 * (self.m_s * self.b_s / self.L)
        mr = 0.5 * (self.m_s * self.a_s / self.L)
        return self.front.ride_frequency(mf), self.rear.ride_frequency(mr)

    def static_aero(self, mode: int | None = None):
        mode = self.default_mode if mode is None else mode
        return self.aero.coefficients(mode, self.front.static_ride_height * 1000.0,
                                      self.rear.static_ride_height * 1000.0, 0.0, 0.0)

    # ------------------------------------------------------------------ physics
    def aero_coefficients(self, mode: int, rh_f, rh_r, roll, yaw, speed=None):
        """ClA_f, ClA_r, CdA (m^2) for a body attitude (and speed, for the Reynolds effect), honouring
        the feature switches."""
        f = self.features
        if not f.heave_pitch_aero:
            rh_f, rh_r = self.front.static_ride_height, self.rear.static_ride_height
        if not f.roll_yaw_aero:
            roll, yaw = 0.0, 0.0
        clamp = self.bottoming_clearance
        cla_f, cla_r, cda, oor = self.aero.coefficients(
            mode, np.maximum(rh_f, clamp) * 1000.0, np.maximum(rh_r, clamp) * 1000.0,
            np.degrees(roll), np.degrees(yaw), speed)
        if not f.aero_balance:
            total = cla_f + cla_r
            cla_f, cla_r = total * self.wd_front, total * (1.0 - self.wd_front)
        return cla_f, cla_r, cda, oor

    # ---- building blocks shared by the quasi-static and the transient solver -------------
    def roll_axis(self, heave_f=0.0, heave_r=0.0):
        """Roll-centre heights (front, rear) and the sprung CG height above the roll axis, with the
        roll centres moving ``roll_center_gain`` per m of heave."""
        fr, rr = self.front, self.rear
        if fr.roll_center_gain == 0.0 and rr.roll_center_gain == 0.0:
            return fr.roll_center, rr.roll_center, self.d_roll
        rc_f = fr.roll_center + fr.roll_center_gain * np.asarray(heave_f)
        rc_r = rr.roll_center + rr.roll_center_gain * np.asarray(heave_r)
        return rc_f, rc_r, self.h_s - (rc_f + (rc_r - rc_f) * self.a_s / self.L)

    def load_components(self, ax, ay, df_f, df_r, drag, roll, heave_f=0.0, heave_r=0.0):
        """Rigid-body load transfer for the given accelerations and aero forces.

        ``heave_*`` (spring compression, m) only moves the roll centres (``roll_center_gain``).

        Returns

        * ``d_long`` - total longitudinal transfer to the rear axle (N),
        * ``el_f``, ``el_r`` - axle load change (vs static) the *springs* carry once the body has
          settled: downforce plus the elastic part of the pitch transfer (anti geometry excluded),
        * ``m_roll`` - roll moment on the sprung mass about the roll axis (gravity term excluded),
        * ``lat_geo_f``, ``lat_geo_r`` - per-tyre lateral transfer that bypasses the springs
          (roll-centre height and unsprung masses).
        """
        f = self.features
        fr, rr = self.front, self.rear
        L, r = self.L, self.tire.radius
        zero = np.zeros(np.broadcast(ax, ay, df_f, drag).shape)
        if not f.load_transfer:
            return {"d_long": zero, "el_f": df_f + zero, "el_r": df_r + zero, "m_roll": zero,
                    "lat_geo_f": zero, "lat_geo_r": zero, "d_roll": zero + self.d_roll}
        h_d = self.h_drag if f.drag_height else 0.0
        d_long = (self.m * ax * self.h + drag * h_d) / L
        sprung_long = self.m_s * ax * self.h_s / L
        anti_f = np.where(ax < 0, fr.anti_brake, fr.anti_drive)
        anti_r = np.where(ax < 0, rr.anti_brake, rr.anti_drive)
        el_f = df_f - sprung_long * (1.0 - anti_f) - drag * h_d / L
        el_r = df_r + sprung_long * (1.0 - anti_r) + drag * h_d / L
        y_cp = self.cop_lateral * np.abs(roll)
        rc_f, rc_r, d_roll = self.roll_axis(heave_f, heave_r)
        m_roll = self.m_s * ay * d_roll + (df_f + df_r) * y_cp
        lat_geo_f = (self.m_s * ay * (self.b_s / L) * rc_f + self.m_uf * ay * r) / fr.track
        lat_geo_r = (self.m_s * ay * (self.a_s / L) * rc_r + self.m_ur * ay * r) / rr.track
        return {"d_long": d_long, "el_f": el_f, "el_r": el_r, "m_roll": m_roll,
                "lat_geo_f": lat_geo_f, "lat_geo_r": lat_geo_r, "d_roll": d_roll + zero}

    def tyre_loads(self, df_f, df_r, comp, spring_f, spring_r, roll_moment_f, roll_moment_r):
        """Four tyre loads from the forces the suspension actually transmits.

        ``spring_*`` is the axle spring (+ damper) force change vs static and
        ``roll_moment_*`` the roll moment carried by each axle's springs / ARB (+ dampers).
        In the quasi-static case ``spring_* = el_*`` and ``roll_moment_* = K_* * roll``.
        """
        axle_f = self.W_f + df_f - comp["d_long"] + (spring_f - comp["el_f"])
        axle_r = self.W_r + df_r + comp["d_long"] + (spring_r - comp["el_r"])
        dz_f = roll_moment_f / self.front.track + comp["lat_geo_f"]
        dz_r = roll_moment_r / self.rear.track + comp["lat_geo_r"]
        fz = np.stack(np.broadcast_arrays(0.5 * axle_f + dz_f, 0.5 * axle_f - dz_f,
                                          0.5 * axle_r + dz_r, 0.5 * axle_r - dz_r))
        wheel_lift = (fz < 0.0).any(axis=0)
        return np.maximum(fz, 0.0), wheel_lift, axle_f, axle_r

    def attitude(self, heave_f, heave_r, axle_f, axle_r, roll, phi_f=None, phi_r=None):
        """Ride heights, corner cambers and roll steer from spring compression (heave), tyre load and roll.

        ``phi_*`` is the suspension (not tyre) roll of each axle; by default the linear split of
        ``roll``. Returns ``(rh_f, rh_r, camber, inclination, (steer_f, steer_r))``; the roll steer of
        each axle is positive into the turn.
        """
        fr, rr = self.front, self.rear
        heave_f, heave_r, axle_f, axle_r, roll = np.broadcast_arrays(
            *(np.asarray(x, dtype=float) for x in (heave_f, heave_r, axle_f, axle_r, roll)))
        rh_f = fr.static_ride_height - heave_f - (axle_f - self.W_f) / (2.0 * fr.tire_rate)
        rh_r = rr.static_ride_height - heave_r - (axle_r - self.W_r) / (2.0 * rr.tire_rate)
        if phi_f is None:
            phi_f = self.K_f * roll / fr.roll_stiffness_springs
        if phi_r is None:
            phi_r = self.K_r * roll / rr.roll_stiffness_springs
        z = np.stack(np.broadcast_arrays(heave_f + 0.5 * fr.track * phi_f, heave_f - 0.5 * fr.track * phi_f,
                                         heave_r + 0.5 * rr.track * phi_r, heave_r - 0.5 * rr.track * phi_r))
        camber, inclination = self.corner_camber(z, roll)
        steer = (0.5 * fr.bump_steer * (z[FO] - z[FI]), 0.5 * rr.bump_steer * (z[RO] - z[RI]))
        return rh_f, rh_r, camber, inclination, steer

    def corner_camber(self, travel, roll):
        """Camber (wheel frame) and lean towards the turn centre of (FO, FI, RO, RI).

        ``travel`` is the bump travel of each corner (compression +, leading axis 4) and
        ``roll`` the body roll towards the outside of the turn.
        """
        fr, rr = self.front, self.rear
        z = np.asarray(travel, dtype=float)
        cam0 = np.array([fr.static_camber, fr.static_camber, rr.static_camber, rr.static_camber])
        side = np.array([1.0, -1.0, 1.0, -1.0])  # body roll tilts outer wheels positive
        shp = (4,) + (1,) * (z.ndim - 1)
        if fr.camber_curve is None and rr.camber_curve is None:
            gain = np.array([fr.camber_gain, fr.camber_gain, rr.camber_gain, rr.camber_gain])
            change = gain.reshape(shp) * z
        else:
            change = np.concatenate([fr.camber_change(z[0:2]), rr.camber_change(z[2:4])])
        camber = cam0.reshape(shp) + change + side.reshape(shp) * np.asarray(roll)
        inclination = -side.reshape(shp) * camber  # lean towards the turn centre
        return camber, inclination

    def suspension_response(self, ax, ay, df_f, df_r, drag, roll, rh_f=None, rh_r=None):
        """Tyre loads and the quasi-static body attitude for given forces.

        ``roll`` is only used for the aero lateral CoP shift, ``rh_*`` (current ride heights) for
        the roll-centre migration. Returns a dict with the four tyre loads, the new ride heights,
        roll angle, corner cambers, roll steer and floor contact force.
        """
        fr, rr = self.front, self.rear
        hf0 = 0.0 if rh_f is None else fr.static_ride_height - np.asarray(rh_f)
        hr0 = 0.0 if rh_r is None else rr.static_ride_height - np.asarray(rh_r)
        comp = self.load_components(ax, ay, df_f, df_r, drag, roll, hf0, hr0)
        k_eff = self.K_f + self.K_r - self.m_s * G * self.gs * comp["d_roll"]
        roll_new = comp["m_roll"] / k_eff
        heave_f = comp["el_f"] / fr.heave_stiffness
        heave_r = comp["el_r"] / rr.heave_stiffness
        phi_f = phi_r = None
        m_f, m_r = self.K_f * roll_new, self.K_r * roll_new
        plank_f = plank_r = 0.0
        if self.nonlinear_ride:
            heave_f, heave_r, roll_new, phi_f, phi_r, m_f, m_r, plank_f, plank_r = self._ride_nonlinear(
                comp, df_f, df_r, heave_f, heave_r, roll_new)
        fz, wheel_lift, axle_f, axle_r = self.tyre_loads(
            df_f, df_r, comp, comp["el_f"] - plank_f, comp["el_r"] - plank_r, m_f, m_r)
        rh_f, rh_r, camber, inclination, steer = self.attitude(heave_f, heave_r, axle_f, axle_r, roll_new,
                                                               phi_f, phi_r)
        return {
            "fz": fz, "rh_f": rh_f, "rh_r": rh_r, "roll": roll_new, "camber": camber,
            "inclination": inclination, "wheel_lift": wheel_lift, "steer": steer,
            "plank": plank_f + plank_r + 0.0 * roll_new,
        }

    def plank_force(self, rh, roll):
        """Floor contact force (N) of one axle: ``plank_rate`` x how far the lowest floor edge is
        below the road."""
        if self.plank_rate is None:
            return 0.0 * np.asarray(rh)
        edge = np.asarray(rh) - self.floor_half_width * np.sin(np.abs(roll))
        return self.plank_rate * np.maximum(-edge, 0.0)

    def _ride_nonlinear(self, comp, df_f, df_r, heave_f, heave_r, roll):
        """Quasi-static attitude with bump stops and floor contact (Newton, vectorised).

        Unknowns per state: heave of each axle, body roll and the suspension roll of each axle (the
        tyres carry the rest of the body roll in series). Equations: axle spring forces + floor
        contact = elastic axle loads, total roll moment of the springs = applied moment + gravity
        term, and each axle's spring roll moment = its tyre roll moment.
        """
        fr, rr = self.front, self.rear
        kt_f, kt_r = 0.5 * fr.tire_rate * fr.track ** 2, 0.5 * rr.tire_rate * rr.track ** 2
        el_f, el_r, m_roll = comp["el_f"], comp["el_r"], comp["m_roll"]
        grav = self.m_s * G * self.gs * comp["d_roll"]
        defl_f = (df_f - comp["d_long"]) / (2.0 * fr.tire_rate)  # tyre deflection change (plank excluded)
        defl_r = (df_r + comp["d_long"]) / (2.0 * rr.tire_rate)

        def axle(ax_, h, p):
            fo, fi = ax_.corner_force(h + 0.5 * ax_.track * p), ax_.corner_force(h - 0.5 * ax_.track * p)
            return fo + fi + ax_.heave_rate * h, 0.5 * ax_.track * (fo - fi) + ax_.arb_rate * p

        def resid(x):
            hf, hr, rl, pf, pr = x
            sf, mf = axle(fr, hf, pf)
            sr, mr = axle(rr, hr, pr)
            pk_f = self.plank_force(fr.static_ride_height - hf - defl_f, rl)
            pk_r = self.plank_force(rr.static_ride_height - hr - defl_r, rl)
            return np.stack(np.broadcast_arrays(
                sf + pk_f - el_f, sr + pk_r - el_r, mf + mr - m_roll - grav * rl,
                mf - kt_f * (rl - pf), mr - kt_r * (rl - pr))), (mf, mr, pk_f, pk_r)

        roll = np.asarray(roll, dtype=float)
        x = np.stack(np.broadcast_arrays(heave_f, heave_r, roll, self.K_f * roll / fr.roll_stiffness_springs,
                                         self.K_r * roll / rr.roll_stiffness_springs)).astype(float)
        eps = 1e-6
        for _ in range(25):
            r0, _ = resid(x)
            jac = np.empty((5,) + x.shape)
            for j in range(5):
                xp = x.copy()
                xp[j] = xp[j] + eps
                jac[:, j] = (resid(xp)[0] - r0) / eps
            jm = np.moveaxis(jac, (0, 1), (-2, -1))
            dx = np.linalg.solve(jm, -np.moveaxis(r0, 0, -1)[..., None])[..., 0]
            dx = np.moveaxis(dx, -1, 0)
            dx = np.clip(dx, -0.02, 0.02)  # m / rad: keep a step across a contact kink bounded
            x = x + dx
            if np.max(np.abs(dx)) < 1e-9:
                break
        _, (mf, mr, pk_f, pk_r) = resid(x)
        hf, hr, rl, pf, pr = x
        return hf, hr, rl, pf, pr, mf, mr, pk_f, pk_r

    def aero_side(self, v, yaw):
        """Aero side force (N, towards the outside of the turn) and its yaw moment about the CG (N m,
        into the turn +) at aero yaw ``yaw`` (rad, turn frame)."""
        if self.cy_area == 0.0:
            return 0.0, 0.0
        fs = 0.5 * self.rho * np.asarray(v) ** 2 * self.cy_area * np.asarray(yaw)
        return fs, -fs * self.side_lever

    def _drive_split(self, f_rear, fx_o, fx_i, track):
        """Driven-axle (left / right) force split: (axle capacity denominator, yaw moment into the turn).

        ``fx_o`` / ``fx_i`` are the longitudinal capacities left by the lateral load of the outer / inner
        tyre. Torque vectoring: each side in proportion to its capacity (equal use). LSD with torque bias
        ratio ``TBR``: the slower inner wheel takes up to TBR / (1 + TBR) of the torque, the outer at
        most TBR x the inner (inner spinning limits the axle to (1 + TBR) x its capacity).
        """
        pt = self.powertrain
        if pt.torque_vectoring:
            den = fx_o + fx_i
            share_o = _safe_ratio(fx_o, den, default=0.5)
            return den, (2.0 * share_o - 1.0) * f_rear * 0.5 * track
        tbr = pt.lsd_locking
        f_in = np.minimum(tbr / (1.0 + tbr) * f_rear, fx_i)
        f_out = f_rear - f_in
        den = np.minimum(fx_i * (1.0 + tbr), fx_o + fx_i)
        return den, (f_out - f_in) * 0.5 * track

    def grip_usage(self, v, ax, ay, fz, camber, inclination, drag, fy_f=None, fy_r=None, roll_steer=None,
                   fy_ext=0.0, mz_ext=0.0):
        """Tyre capacities, slip angles and the utilisation of each axle and of the powertrain.

        The lateral demand per axle is the steady-state yaw-moment split of ``m ay`` (+ ``fy_ext``, a
        force pushing the car out of the turn, and the yaw moment ``mz_ext`` into the turn plus the
        one of the driven axle's left / right split) unless the actual axle lateral forces
        ``fy_f`` / ``fy_r`` are given (transient lateral dynamics; then the drive yaw moment is only
        returned as ``mz_drive``). ``roll_steer`` = (front, rear) steer of the axles from the
        suspension (into the turn +).
        """
        if self.features.camber:
            fx_cap, fy_cap = self.tire.capacity(fz, inclination, camber)
        else:
            fx_cap, fy_cap = self.tire.capacity(fz)
        if not (np.isscalar(self.tyre_grip) and self.tyre_grip == 1.0):  # tyre temperature window
            s = np.asarray(self.tyre_grip, dtype=float)
            if s.ndim:
                s = s.reshape((4,) + (1,) * (np.ndim(fx_cap) - 1))
            fx_cap, fy_cap = fx_cap * s, fy_cap * s

        # Lateral demand from steady-state yaw-moment balance (or the given axle forces).
        pt = self.powertrain
        given = fy_f is not None
        total_y = self.m * ay + fy_ext
        if fy_f is None:
            fy_f = (self.b * total_y - mz_ext) / self.L
        if fy_r is None:
            fy_r = (self.a * total_y + mz_ext) / self.L
        fy_f, fy_r = np.abs(fy_f), np.abs(fy_r)
        uy_f = _safe_ratio(fy_f, fy_cap[FO] + fy_cap[FI])
        uy_r = _safe_ratio(fy_r, fy_cap[RO] + fy_cap[RI])
        split = pt.lsd_locking is not None or pt.torque_vectoring
        mz_drive = 0.0
        if split:  # yaw moment of the driven axle's left / right split -> one correction of the lateral split
            n0 = self.tire.combined_exponent
            f_drive0 = np.maximum(self.m * ax + drag + self.tire.rolling_resistance * fz.sum(axis=0), 0.0)
            f_rear0 = f_drive0 if pt.drive == "rwd" else f_drive0 * (1.0 - (pt.awd_front_fraction or 0.5))
            lat0 = np.clip(1.0 - np.minimum(uy_r, 1.0) ** n0, 0.0, 1.0) ** (1.0 / n0)
            _, mz_drive = self._drive_split(f_rear0, fx_cap[RO] * lat0, fx_cap[RI] * lat0, self.rear.track)
            if not given:
                fy_f = np.abs((self.b * total_y - mz_ext - mz_drive) / self.L)
                fy_r = np.abs((self.a * total_y + mz_ext + mz_drive) / self.L)
                uy_f = _safe_ratio(fy_f, fy_cap[FO] + fy_cap[FI])
                uy_r = _safe_ratio(fy_r, fy_cap[RO] + fy_cap[RI])

        # Sideslip / steer from the tyre slip curve.
        kappa = ay / np.maximum(v, 1.0) ** 2
        alpha_f = self.tire.axle_slip_angle(uy_f, fz[FO], fz[FI])
        alpha_r = self.tire.axle_slip_angle(uy_r, fz[RO], fz[RI])
        beta = self.b * kappa - alpha_r
        steer = self.L * kappa + alpha_f - alpha_r
        if roll_steer is not None:  # rear roll steer turns the car, front roll steer saves hand-wheel steer
            beta = beta + roll_steer[1]
            steer = steer + roll_steer[1] - roll_steer[0]

        # Longitudinal demand and its distribution.
        rr_force = self.tire.rolling_resistance * fz.sum(axis=0)
        fx_req = self.m * ax + drag + rr_force
        f_drive = np.maximum(fx_req, 0.0)
        f_brake = np.maximum(-fx_req, 0.0)
        n_ell = self.tire.combined_exponent  # friction "ellipse" |ux|^n + |uy|^n <= 1 (n = 2: ellipse)
        if n_ell == 2.0:
            lat_f = np.sqrt(np.clip(1.0 - uy_f * uy_f, 0.0, 1.0))
            lat_r = np.sqrt(np.clip(1.0 - uy_r * uy_r, 0.0, 1.0))
        else:
            lat_f = np.clip(1.0 - np.minimum(uy_f, 1.0) ** n_ell, 0.0, 1.0) ** (1.0 / n_ell)
            lat_r = np.clip(1.0 - np.minimum(uy_r, 1.0) ** n_ell, 0.0, 1.0) ** (1.0 / n_ell)
        min_f = np.minimum(fx_cap[FO], fx_cap[FI])
        min_r = np.minimum(fx_cap[RO], fx_cap[RI])
        sum_f = fx_cap[FO] + fx_cap[FI]
        sum_r = fx_cap[RO] + fx_cap[RI]
        den_drive_f = 2.0 * min_f if pt.differential == "open" else sum_f
        den_drive_r = 2.0 * min_r if pt.differential == "open" else sum_r
        if split:  # LSD with a torque bias / torque vectoring on the rear axle (per-axle lateral share)
            den_drive_r, _ = self._drive_split(0.0 * f_drive, fx_cap[RO], fx_cap[RI], self.rear.track)
        if pt.drive == "rwd":
            fd_f, fd_r = 0.0 * f_drive, f_drive
        elif pt.awd_front_fraction is not None:
            fd_f, fd_r = pt.awd_front_fraction * f_drive, (1.0 - pt.awd_front_fraction) * f_drive
        else:
            cf, cr = den_drive_f * lat_f, den_drive_r * lat_r
            share = _safe_ratio(cf, cf + cr, default=0.5)
            fd_f, fd_r = share * f_drive, (1.0 - share) * f_drive
        # Brakes: equal pressure left/right -> equal force per wheel on an axle.
        cf, cr = 2.0 * min_f * lat_f, 2.0 * min_r * lat_r
        if self.brake_bias is None:
            share = _safe_ratio(cf, cf + cr, default=self.wd_front)
        else:
            share = self.brake_bias
        f_regen = 0.0 * f_brake
        if pt.has_regen and pt.drive == "rwd":
            # the motor brakes the rear axle first (up to what its tyres take); the friction brakes add
            # the rest by the bias - with an ideal (adjustable) bias they only top up to the ideal split
            f_regen = pt.regen_force(v, f_brake, cr)
            fric = f_brake - f_regen
            if self.brake_bias is None:
                fb_r = np.maximum((1.0 - share) * f_brake, f_regen)
                fb_f = f_brake - fb_r
            else:
                fb_f, fb_r = share * fric, (1.0 - share) * fric + f_regen
        else:
            if pt.has_regen:  # awd: regen on both axles in the brake split, no balance change
                f_regen = pt.regen_force(v, f_brake, cf + cr)
            fb_f, fb_r = share * f_brake, (1.0 - share) * f_brake
        ux_f = _safe_ratio(fd_f, den_drive_f) + _safe_ratio(fb_f, 2.0 * min_f)
        ux_r = _safe_ratio(fd_r, den_drive_r) + _safe_ratio(fb_r, 2.0 * min_r)
        if n_ell == 2.0:
            u_front = np.sqrt(ux_f * ux_f + uy_f * uy_f)
            u_rear = np.sqrt(ux_r * ux_r + uy_r * uy_r)
        else:
            u_front = (ux_f ** n_ell + uy_f ** n_ell) ** (1.0 / n_ell)
            u_rear = (ux_r ** n_ell + uy_r ** n_ell) ** (1.0 / n_ell)
        f_power, _ = pt.max_force(v)
        # the motor also accelerates the rotating driveline (rotor inertia reflected to the wheels)
        u_power = _safe_ratio(f_drive + pt.m_rot * np.maximum(ax, 0.0) * (f_drive > 0.0), f_power)
        stack = np.stack(np.broadcast_arrays(u_front, u_rear, u_power))
        with np.errstate(divide="ignore", invalid="ignore"):
            usi = np.where(uy_r > 1e-9, uy_f / np.where(uy_r > 1e-9, uy_r, 1.0), np.nan)
        return {
            "fx_cap": fx_cap, "fy_cap": fy_cap, "uy_f": uy_f, "uy_r": uy_r, "alpha_f": alpha_f,
            "alpha_r": alpha_r, "beta": beta, "steer": steer, "f_drive": f_drive, "f_brake": f_brake,
            "u_front": u_front, "u_rear": u_rear, "u_power": u_power, "limit": np.argmax(stack, axis=0),
            "ux_f": ux_f, "ux_r": ux_r, "f_regen": f_regen, "fx_f": fd_f - fb_f, "fx_r": fd_r - fb_r,
            "cx_f": sum_f * lat_f, "cx_r": sum_r * lat_r, "mz_drive": mz_drive + 0.0 * f_drive,
            "utilization": stack.max(axis=0), "understeer_index": usi,
        }

    def make_state(self, v, ax, ay, mode, rh_f, rh_r, roll, yaw, cla_f, cla_r, cda, oor, df_f, df_r, drag,
                   fz, camber, wheel_lift, g) -> VehicleState:
        rh_f, rh_r = np.broadcast_arrays(rh_f, rh_r)
        clearance = np.minimum(rh_f, rh_r) - self.floor_half_width * np.sin(np.abs(roll))
        pitch = ((rh_r - self.rear.static_ride_height) - (rh_f - self.front.static_ride_height)) / self.L
        bshape = g["utilization"].shape
        return VehicleState(
            v=_b(v, bshape), ax=_b(ax, bshape), ay=_b(ay, bshape), mode=mode,
            rh_f=_b(rh_f, bshape), rh_r=_b(rh_r, bshape), roll=_b(roll, bshape), yaw=_b(yaw, bshape),
            pitch=_b(pitch, bshape), cla_f=_b(cla_f, bshape), cla_r=_b(cla_r, bshape), cda=_b(cda, bshape),
            df_f=_b(df_f, bshape), df_r=_b(df_r, bshape), drag=_b(drag, bshape), fz=fz,
            camber=np.broadcast_to(camber, fz.shape), fx_cap=g["fx_cap"], fy_cap=g["fy_cap"],
            clearance=_b(clearance, bshape), alpha_f=_b(g["alpha_f"], bshape),
            alpha_r=_b(g["alpha_r"], bshape), steer=_b(g["steer"], bshape), uy_f=_b(g["uy_f"], bshape),
            uy_r=_b(g["uy_r"], bshape), u_front=_b(g["u_front"], bshape), u_rear=_b(g["u_rear"], bshape),
            u_power=_b(g["u_power"], bshape), utilization=g["utilization"], limit=g["limit"],
            understeer_index=_b(g["understeer_index"], bshape), fx_drive=_b(g["f_drive"], bshape),
            fx_brake=_b(g["f_brake"], bshape), aero_out_of_range=_b(oor, bshape),
            bottoming=_b(clearance < self.bottoming_clearance, bshape), wheel_lift=_b(wheel_lift, bshape),
        )

    def evaluate(self, v, ax, ay, mode: int, rh_f, rh_r, roll, yaw):
        """One explicit evaluation of every subsystem for a given attitude.

        Returns ``(state, new_attitude)`` where ``new_attitude`` is the attitude the
        quasi-static spring balance would settle at for these forces.
        """
        q = 0.5 * self.rho * v * v
        cla_f, cla_r, cda, oor = self.aero_coefficients(mode, rh_f, rh_r, roll, yaw, v)
        df_f, df_r, drag = q * cla_f, q * cla_r, q * cda
        sr = self.suspension_response(ax, ay, df_f, df_r, drag, roll, rh_f, rh_r)
        rs = sr["steer"] if (self.front.bump_steer or self.rear.bump_steer) else None
        fs, mzs = self.aero_side(v, yaw)
        g = self.grip_usage(v, ax, ay, sr["fz"], sr["camber"], sr["inclination"], drag + self.plank_mu * sr["plank"],
                            roll_steer=rs, fy_ext=fs, mz_ext=mzs)
        state = self.make_state(v, ax, ay, mode, rh_f, rh_r, roll, yaw, cla_f, cla_r, cda, oor, df_f, df_r,
                                drag, sr["fz"], sr["camber"], sr["wheel_lift"], g)
        state.plank = _b(sr["plank"], state.utilization.shape)
        state.fx_regen = _b(g["f_regen"], state.utilization.shape)
        beta = g["beta"]
        new_attitude = (sr["rh_f"], sr["rh_r"], sr["roll"], beta if self.features.roll_yaw_aero else 0.0 * beta)
        return state, new_attitude

    def solve_state(self, v, ax, ay, mode: int | None = None, init: VehicleState | None = None,
                    max_iter: int = 40, tol: float = 1e-5, relax: float = 1.0, g_scale=None) -> VehicleState:
        """Quasi-static equilibrium: aero <-> loads <-> ride heights / roll / yaw fixed point.

        ``g_scale`` (scalar or per state) scales the normal gravity: banked road, crests and dips."""
        if g_scale is not None:
            old, self.gs = self.gs, np.asarray(g_scale, dtype=float)
            try:
                return self.solve_state(v, ax, ay, mode, init, max_iter, tol, relax)
            finally:
                self.gs = old
        mode = self.default_mode if mode is None else mode
        v, ax, ay = (np.array(x, dtype=float) for x in np.broadcast_arrays(
            np.asarray(v, dtype=float), np.asarray(ax, dtype=float), np.abs(np.asarray(ay, dtype=float))))
        shape = v.shape
        if init is not None and init.rh_f.shape == shape:
            rh_f, rh_r, roll, yaw = init.rh_f.copy(), init.rh_r.copy(), init.roll.copy(), init.yaw.copy()
        else:
            rh_f = np.full(shape, self.front.static_ride_height)
            rh_r = np.full(shape, self.rear.static_ride_height)
            roll = np.zeros(shape)
            yaw = np.zeros(shape)
        converged = np.zeros(shape, dtype=bool)
        state = None
        for _ in range(max_iter):
            state, (nf, nr, nroll, nyaw) = self.evaluate(v, ax, ay, mode, rh_f, rh_r, roll, yaw)
            dmax = np.maximum.reduce([np.abs(nf - rh_f), np.abs(nr - rh_r),
                                      np.abs(nroll - roll), np.abs(nyaw - yaw)])
            converged = dmax < tol
            rh_f = rh_f + relax * (nf - rh_f)
            rh_r = rh_r + relax * (nr - rh_r)
            roll = roll + relax * (nroll - roll)
            yaw = yaw + relax * (nyaw - yaw)
            if converged.all():
                break
        state, _ = self.evaluate(v, ax, ay, mode, rh_f, rh_r, roll, yaw)
        state.converged = converged
        return state

    def coast_ax(self, state: VehicleState) -> np.ndarray:
        """Acceleration with zero tyre longitudinal force (drag + rolling resistance only)."""
        rr_force = self.tire.rolling_resistance * state.fz.sum(axis=0)
        plank = 0.0 if state.plank is None else self.plank_mu * state.plank
        return -(state.drag + rr_force + plank) / self.m


def _safe_ratio(num, den, default: float = 0.0):
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(den > 1e-9, num / np.where(den > 1e-9, den, 1.0),
                       np.where(num > 1e-9, np.inf, default))
    return out


def _b(x, shape):
    return np.broadcast_to(np.asarray(x), shape)
