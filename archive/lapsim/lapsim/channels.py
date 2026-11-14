"""Logged-data style channels for a solved lap, and CSV export.

Everything is recomputed from v(s), a_x(s) and the path with the same physics the solver
used, so the channels are consistent with the lap time.
"""

from __future__ import annotations

import csv

import numpy as np

from .physics import Model
from .solver import LapResult
from .vehicle import G, Vehicle


def channels(veh: Vehicle, res: LapResult) -> dict:
    m = Model(veh)
    v = res.v
    # a_x as the solver used it: accelerating segments are solved at their start point,
    # braking segments (backward pass) at their end point
    seg = np.diff(v**2) / (2 * np.diff(res.s))
    fwd, bwd = np.append(seg, seg[-1]), np.concatenate([[seg[0]], seg])
    ax = np.where(bwd < -0.05, bwd, fwd)
    ay = res.a_y
    ay_abs = np.abs(ay)
    fzf, fzr, drag, hf, hr = m.loads(v, ax)
    hf = np.broadcast_to(hf, v.shape).astype(float)
    hr = np.broadcast_to(hr, v.shape).astype(float)

    # wheel loads: outer wheels are on the right in a left turn (a_y > 0)
    dff, dfr = m.llt_f * ay_abs, m.llt_r * ay_abs
    fo_f, fi_f = m.wheel_split(fzf, dff)
    fo_r, fi_r = m.wheel_split(fzr, dfr)
    left_turn = ay > 0
    fl = np.where(left_turn, fi_f, fo_f)
    fr = np.where(left_turn, fo_f, fi_f)
    rl = np.where(left_turn, fi_r, fo_r)
    rr = np.where(left_turn, fo_r, fi_r)

    # axle utilisation: lateral (incl. yaw term) + longitudinal (drive on the rear, braking
    # split in proportion to each axle's longitudinal capacity = ideal brake balance)
    cf, cr = m.lateral_coeffs(res.kappa, res.kappa_rate)
    fxf_max, fyf_max = m.axle_grip(fzf, dff)
    fxr_max, fyr_max = m.axle_grip(fzr, dfr)
    f_long = m.m * ax + drag + m.crr * (fzf + fzr)          # tyre force the car needs
    brake = np.clip(-f_long, 0, None)
    # braking is shared like the solver does it: by what each axle has left on its ellipse
    left_f = fxf_max * np.sqrt(np.clip(1 - (v**2 * cf / np.maximum(fyf_max, 1e-9)) ** 2, 0, None))
    left_r = fxr_max * np.sqrt(np.clip(1 - (v**2 * cr / np.maximum(fyr_max, 1e-9)) ** 2, 0, None))
    share_f = np.where(left_f + left_r > 1e-9, left_f / np.maximum(left_f + left_r, 1e-9), 0.5)
    fx_f = brake * share_f
    fx_r = np.clip(f_long, 0, None) + brake * (1 - share_f)
    util_f = np.sqrt((v**2 * cf / np.maximum(fyf_max, 1e-9)) ** 2 + (fx_f / np.maximum(fxf_max, 1e-9)) ** 2)
    util_r = np.sqrt((v**2 * cr / np.maximum(fyr_max, 1e-9)) ** 2 + (fx_r / np.maximum(fxr_max, 1e-9)) ** 2)

    p_wheel = np.clip(f_long, 0, None) * v
    p_batt = p_wheel / m.eff
    dt = np.diff(res.t)
    e_batt = np.concatenate([[0.0], np.cumsum(0.5 * (p_batt[1:] + p_batt[:-1]) * dt)])

    ch = {
        "s_m": res.s, "t_s": res.t, "x_m": res.x, "y_m": res.y,
        "speed_kmh": v * 3.6, "corner_limit_kmh": res.v_corner * 3.6,
        "ax_g": ax / G, "ay_g": ay / G, "curvature_1pm": res.kappa,
        "phase": res.phase, "corner_limiter": res.limiter,
        "Fz_FL_N": fl, "Fz_FR_N": fr, "Fz_RL_N": rl, "Fz_RR_N": rr,
        "Fz_front_N": fzf, "Fz_rear_N": fzr,
        "ride_height_front_mm": hf, "ride_height_rear_mm": hr,
    }
    total_df = np.zeros_like(v)
    total_df_front = np.zeros_like(v)
    for (name, f_down, f_drag), dev in zip(m.device_forces(v, hf, hr), m.devices):
        f_down = np.broadcast_to(f_down, v.shape).astype(float)
        ch[f"downforce_{name}_N"] = f_down
        ch[f"drag_{name}_N"] = np.broadcast_to(f_drag, v.shape).astype(float)
        total_df += f_down
        total_df_front += f_down * dev[3]
    ch["downforce_total_N"] = total_df
    ch["drag_total_N"] = np.broadcast_to(drag, v.shape).astype(float)
    ch["aero_balance_front_pct"] = np.where(total_df > 1e-6, total_df_front / np.maximum(total_df, 1e-9) * 100,
                                            np.nan)
    ch["util_front"] = util_f
    ch["util_rear"] = util_r
    ch["power_wheel_kW"] = p_wheel / 1e3
    ch["power_battery_kW"] = p_batt / 1e3
    ch["energy_battery_Wh"] = e_batt / 3600
    return ch


def export_csv(ch: dict, path) -> None:
    keys = list(ch)
    n = len(ch["s_m"])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for i in range(n):
            w.writerow([_fmt(ch[k][i]) for k in keys])


def _fmt(x):
    if isinstance(x, (str, np.str_)):
        return str(x)
    return f"{float(x):.6g}"


def delta_time(ref: LapResult, other: LapResult) -> tuple[np.ndarray, np.ndarray]:
    """Running time difference other - ref [s] over ref's distance.

    Distances are compared as a fraction of the lap so runs on different paths (centerline vs
    racing line) line up corner by corner.
    """
    frac_ref = ref.s / ref.s[-1]
    t_other = np.interp(frac_ref, other.s / other.s[-1], other.t)
    return ref.s, t_other - ref.t
