"""Vehicle physics shared by every part of the solver.

All functions are written with plain arithmetic (no ``np.maximum``/``np.where``) so the same
code runs on Python floats in the sequential acceleration/braking passes (fast) and on numpy
arrays in the vectorised corner-speed solve and the post-processed channels.

Per point the model resolves, at speed ``v`` and longitudinal acceleration ``a_x``:

* aero  : every device's downforce/drag, scaled by its ride-height map at the current ride heights
* loads : static + aero + longitudinal transfer (inertia and drag pitching moment) per axle,
          ride heights from the axle heave rates (fixed-point iteration when aero maps exist)
* grip  : per wheel, load-sensitive mu, with lateral load transfer split by the LLTD
"""

from __future__ import annotations

from .vehicle import G, Vehicle


def pos(x):
    """max(x, 0) for floats and arrays."""
    return (x + abs(x)) * 0.5


def vmin(a, b):
    """min(a, b) for floats and arrays."""
    return 0.5 * (a + b - abs(a - b))


def _poly(c, dhf, dhr):
    return 1.0 + c[0] * dhf + c[1] * dhr + c[2] * dhf * dhf + c[3] * dhr * dhr + c[4] * dhf * dhr


class Model:
    def __init__(self, veh: Vehicle):
        self.veh = veh
        a, t, p, s = veh.aero, veh.tyre, veh.powertrain, veh.suspension
        self.m, self.h, self.L = veh.mass, veh.cg_height, veh.wheelbase
        self.wf = veh.weight_dist_front
        self.crr = veh.rolling_resistance
        self.half_rho = 0.5 * a.rho
        self.mux, self.muy, self.ks, self.fz0 = t.mu_x, t.mu_y, t.load_sensitivity, t.Fz0
        self.v_top = p.v_max
        self.f_torque = p.max_torque * p.gear_ratio * p.efficiency / p.wheel_radius
        self.p_wheel = p.max_power * p.efficiency
        self.eff = p.efficiency
        self.Iz = veh.yaw_inertia
        self.fzf0 = self.m * G * self.wf
        self.fzr0 = self.m * G * (1 - self.wf)

        if s is None:
            self.llt_f = self.llt_r = 0.0
            self.hf0 = self.hr0 = 0.0
            self.kf = self.kr = 0.0
            self.h_min = 0.0
        else:
            lltd = s.lltd
            self.llt_f = lltd * self.m * self.h / s.track_front        # per-wheel dFz per unit a_y
            self.llt_r = (1 - lltd) * self.m * self.h / s.track_rear
            self.hf0, self.hr0 = s.ride_height_front, s.ride_height_rear
            self.kf, self.kr = s.heave_rate_front, s.heave_rate_rear
            self.h_min = s.min_ride_height

        self.devices = []
        for name, d in a.devices.items():
            ref = d.rh_ref or [self.hf0, self.hr0]
            self.devices.append((name, d.ClA, d.CdA, d.share(self.L), d.h_cp,
                                 _pad(d.rh_cl), _pad(d.rh_cd), ref[0], ref[1]))
        # devices without a ride-height map collapse into constants; only mapped ones loop
        fixed = [dv for dv in self.devices if not (dv[5] or dv[6])]
        self.mapped = [dv for dv in self.devices if dv[5] or dv[6]]
        self.c_cl = sum(dv[1] for dv in fixed)
        self.c_clf = sum(dv[1] * dv[3] for dv in fixed)
        self.c_cd = sum(dv[2] for dv in fixed)
        self.c_cdh = sum(dv[2] * dv[4] for dv in fixed)
        self.ride_height_moves = bool(self.mapped) and (self.kf > 0 or self.kr > 0)
        self.rh_iters = 3 if self.ride_height_moves else 1

    # ---- aero -----------------------------------------------------------------
    def device_forces(self, v, hf, hr):
        """[(name, downforce, drag)] per device at ride heights hf, hr [mm]."""
        q = self.half_rho * v * v
        out = []
        for name, cla, cda, _share, _h, rcl, rcd, rf, rr in self.devices:
            fcl = pos(_poly(rcl, hf - rf, hr - rr)) if rcl else 1.0
            fcd = pos(_poly(rcd, hf - rf, hr - rr)) if rcd else 1.0
            out.append((name, q * cla * fcl, q * cda * fcd))
        return out

    def aero(self, v, hf, hr):
        """(downforce, front downforce, drag, drag pitching moment)."""
        q = self.half_rho * v * v
        df, dff, drag, mom = q * self.c_cl, q * self.c_clf, q * self.c_cd, q * self.c_cdh
        for _name, cla, cda, share, hcp, rcl, rcd, rf, rr in self.mapped:
            fi = q * cla * (pos(_poly(rcl, hf - rf, hr - rr)) if rcl else 1.0)
            di = q * cda * (pos(_poly(rcd, hf - rf, hr - rr)) if rcd else 1.0)
            df += fi
            dff += fi * share
            drag += di
            mom += di * hcp
        return df, dff, drag, mom

    # ---- loads ------------------------------------------------------------------
    def loads(self, v, ax, hf=None, hr=None, iters=None):
        """(Fz_front, Fz_rear, drag, ride height front, ride height rear).

        Ride height and aero load are solved by fixed-point iteration; the marching passes
        warm-start it with the previous point's ride heights and use a single iteration.
        """
        hf = self.hf0 if hf is None else hf
        hr = self.hr0 if hr is None else hr
        for _ in range(self.rh_iters if iters is None else iters):
            df, dff, drag, mom = self.aero(v, hf, hr)
            dlon = (self.m * ax * self.h + mom) / self.L
            fzf = pos(self.fzf0 + dff - dlon)
            fzr = pos(self.fzr0 + (df - dff) + dlon)
            if self.ride_height_moves:
                if self.kf > 0:     # bump stop: the ride height cannot go below h_min
                    hf = self.h_min + pos(self.hf0 - (fzf - self.fzf0) / self.kf - self.h_min)
                if self.kr > 0:
                    hr = self.h_min + pos(self.hr0 - (fzr - self.fzr0) / self.kr - self.h_min)
        return fzf, fzr, drag, hf, hr

    def wheel_split(self, fz_axle, dfz):
        """(outer, inner) wheel loads; transfer is capped at inner-wheel lift."""
        half = 0.5 * fz_axle
        d = vmin(dfz, half)
        return half + d, half - d

    def mu_scale(self, fz_wheel):
        s = 1.0 - self.ks * (fz_wheel - self.fz0) / self.fz0
        return 0.2 + pos(s - 0.2)

    def axle_grip(self, fz_axle, dfz):
        """(Fx_max, Fy_max) of an axle with per-wheel lateral transfer ``dfz`` (>= 0)."""
        fo, fi = self.wheel_split(fz_axle, dfz)
        cap = self.mu_scale(fo) * fo + self.mu_scale(fi) * fi
        return self.mux * cap, self.muy * cap

    # ---- powertrain ----------------------------------------------------------------
    def drive(self, v):
        """Tractive force limit [N] (scalar)."""
        if v >= self.v_top:
            return 0.0
        return min(self.f_torque, self.p_wheel / max(v, 0.1))

    # ---- yaw ------------------------------------------------------------------------
    def lateral_coeffs(self, kappa, kappa_rate):
        """Per-axle lateral force per v^2 [kg/m]:  Fy_axle = v^2 * c_axle.

        Steady state splits m*a_y by the static weight distribution; the yaw acceleration
        r_dot ~ v^2 * dkappa/ds adds +/- Iz*r_dot/L (front on corner entry, rear on exit).
        """
        yaw = self.Iz * kappa_rate / self.L
        cf = abs(self.m * kappa * self.wf + yaw)
        cr = abs(self.m * kappa * (1 - self.wf) - yaw)
        return cf, cr


def ellipse(fx_max, fy_max, fy):
    """Longitudinal force left on the friction ellipse (scalar)."""
    if fy_max <= 0:
        return 0.0
    r = fy / fy_max
    return fx_max * (1.0 - r * r) ** 0.5 if r < 1.0 else 0.0


def _pad(c):
    if not c:
        return None
    c = list(c) + [0.0] * (5 - len(c))
    return c[:5]
