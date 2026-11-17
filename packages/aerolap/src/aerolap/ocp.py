"""Optimal control with CasADi / IPOPT (optional dependency: ``pip install aerolap[ocp]``).

* :func:`min_time_line` - minimum-time racing line: lateral offset ``a_i`` of every reference point and
  the speed ``v_i`` there are optimised together (direct collocation in space). The tyres / powertrain
  enter through the car's g-g-V envelope (smooth B-spline of the QSS tables): at every interval the
  longitudinal acceleration ``(v_{i+1}^2 - v_i^2) / 2 l_i`` lies between the braking and traction limit
  at the interval's speed and lateral acceleration ``v^2 kappa``, and ``|v^2 kappa| <= ay_max(v)``.
  Objective: lap time ``sum 2 l_i / (v_i + v_{i+1})``. The geometry (segment lengths ``l_i``, discrete
  curvature ``kappa_i``) follows the moved points exactly, so the optimiser trades a longer line for
  higher corner speed by itself. Warm start: the minimum-curvature line and its QSS speeds.
* :class:`MPCDriver` - receding-horizon driver for the transient solver (``--driver mpc``): over a
  short horizon it chooses acceleration and steering of a bicycle model that maximise progress, inside
  the friction circle of the current tyre loads and below the QSS speed plan ahead (which carries the
  braking points beyond the horizon). Trail braking falls out of the combined-grip constraint.

Ignored by the OCP line: active aero modes (the closed / default mode envelope is used) and the bank /
vertical curvature (the grade is included).
"""

from __future__ import annotations

import math

import numpy as np

G = 9.80665


def _casadi():
    try:
        import casadi
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("최적 제어(OCP / MPC)에는 casadi가 필요합니다: pip install \"aerolap[ocp]\" "
                          "(또는 pip install casadi)") from exc
    return casadi


def _envelope(ggv):
    """Smooth g-g-V for the NLP, all B-splines in speed only (fast and smooth for IPOPT): traction and
    braking margin above / below the coast deceleration at ay = 0, ``ay_max(v)``, and per speed the
    exponents ``n`` of ``(ax / ax0)^n + (ay / ay_max)^n <= 1`` fitted conservatively to the QSS envelope
    (the largest ``n`` that stays inside it, so the NLP never sees more grip than the QSS check)."""
    ca = _casadi()
    vg = np.asarray(ggv.v, dtype=float)
    f = np.minimum(np.asarray(ggv.fracs, dtype=float), 0.999)
    c = np.asarray(ggv.ax_coast, dtype=float)
    ns = np.arange(1.0, 6.001, 0.05)
    model = (1.0 - f[None, :] ** ns[:, None]) ** (1.0 / ns[:, None])  # (n, frac), grows with n
    out = []
    for tab in (np.asarray(ggv.ax_acc), np.asarray(ggv.ax_brk)):
        m0 = np.abs(tab[:, 0] - c[:, 0])
        rel = np.abs(tab - c) / np.maximum(m0[:, None], 1e-6)
        n_v = np.empty(len(vg))
        for k in range(len(vg)):
            inside = np.all(model <= rel[k][None, :] + 0.02, axis=1)
            n_v[k] = ns[np.flatnonzero(inside)[-1]] if inside.any() else ns[0]
        out.append((np.maximum(m0, 0.05), n_v))
    (acc0, n_acc), (brk0, n_brk) = out
    sp = lambda name, y: ca.interpolant(name, "bspline", [vg], np.asarray(y, dtype=float))  # noqa: E731
    return (sp("acc0", acc0), sp("brk0", np.maximum(brk0, 0.5)), sp("coast", c[:, 0]),
            sp("aym", np.maximum(np.asarray(ggv.ay_max, dtype=float), 0.5)), sp("nacc", n_acc), sp("nbrk", n_brk),
            float(vg[0]), float(vg[-1]))


def min_time_line(vehicle, track, car_width: float | None = None, clearance: float = 0.2, step: float = 2.0,
                  opts=None, start: str | None = None, max_iter: int = 1500, progress=None):
    """Minimum-time line inside the track width (see the module docstring); returns a
    :class:`aerolap.racing_line.LineResult` whose candidates are the minimum-curvature warm start and
    the OCP solution (the faster one by QSS lap time is chosen)."""
    ca = _casadi()
    from .racing_line import Candidate, LineResult, line_track, optimal_offsets, reference_line
    from .solvers.qss import QSSSolver

    if car_width is None:
        car_width = max(vehicle.front.track, vehicle.rear.track) + 0.2
    ref = reference_line(track, step)
    margin = 0.5 * car_width + clearance
    hi, lo = ref.wl - margin, -(ref.wr - margin)
    narrow = hi < lo
    mid = 0.5 * (hi + lo)
    hi = np.where(narrow, np.clip(mid, -ref.wr, ref.wl), hi)
    lo = np.where(narrow, hi, lo)
    start = start or ("flying" if track.closed else "standing")
    solver = QSSSolver(opts)
    ds = solver.opts.ds
    reference_lap = solver.solve(vehicle, track, start=start)

    # warm start: minimum-curvature line and its QSS speed profile
    a0, it0 = optimal_offsets(ref, lo, hi, 0.0)
    trk0 = line_track(ref, a0, f"{track.name} (racing line)", ds=min(ds, 0.5))
    lap0 = solver.solve(vehicle, trk0, start=start)
    if progress:
        progress(1, 2, "minimum curvature", lap0.lap_time)
    n = ref.n
    frac = np.arange(n) / (n if ref.closed else n - 1)
    v_init = np.interp(frac * lap0.s[-1], lap0.s, lap0.v)

    ggv = solver.ggv(vehicle, vehicle.active_strategy().closed_mode)
    f_acc, f_brk, f_coast, f_aym, f_nacc, f_nbrk, v_lo, v_hi = _envelope(ggv)
    grade = None
    if track.z is not None and track.use_grade and ref.z is not None and float(np.ptp(ref.z)) > 1e-6:
        dz = np.diff(np.append(ref.z, ref.z[0]) if ref.closed else ref.z)
        grade = G * dz / np.hypot(dz, ref.h)

    opti = ca.Opti()
    a = opti.variable(n)
    v = opti.variable(n)
    x = ca.DM(ref.x) + a * ca.DM(ref.nx)
    y = ca.DM(ref.y) + a * ca.DM(ref.ny)
    if ref.closed:
        xm, xp = ca.vertcat(x[n - 1], x[0:n - 1]), ca.vertcat(x[1:n], x[0])
        ym, yp = ca.vertcat(y[n - 1], y[0:n - 1]), ca.vertcat(y[1:n], y[0])
        xc, yc = x, y
    else:
        xm, xp, ym, yp, xc, yc = x[0:n - 2], x[2:n], y[0:n - 2], y[2:n], x[1:n - 1], y[1:n - 1]
    da, db = 0.5 * (xp - xm), 0.5 * (yp - ym)
    dc, dd = xp - 2.0 * xc + xm, yp - 2.0 * yc + ym
    kap = (da * dd - db * dc) / (da * da + db * db + 1e-9) ** 1.5
    if not ref.closed:  # ends take the neighbour's curvature
        kap = ca.vertcat(kap[0], kap, kap[kap.shape[0] - 1])
    m = n if ref.closed else n - 1  # intervals
    nxt = list(range(1, n)) + ([0] if ref.closed else [])
    x1, y1, v1, k1 = (ca.vertcat(*[e[j] for j in nxt]) for e in (x, y, v, kap))
    x0_, y0_, v0_, k0_ = x[0:m], y[0:m], v[0:m], kap[0:m]
    seg = ca.sqrt((x1 - x0_) ** 2 + (y1 - y0_) ** 2 + 1e-9)
    vm = 0.5 * (v0_ + v1)
    ax = (v1 * v1 - v0_ * v0_) / (2.0 * seg)
    if grade is not None:
        ax = ax + ca.DM(grade[:m])  # the envelope is in tyre force: uphill needs g sin(theta) more
    vm_c = ca.fmin(ca.fmax(vm, v_lo), v_hi)
    # lap time + a small smoothness term (the discrete curvature of free offsets is non-convex: without
    # it IPOPT can wander into zig-zag lines)
    d2 = a[2:n] - 2.0 * a[1:n - 1] + a[0:n - 2]
    opti.minimize(ca.sum1(2.0 * seg / (v0_ + v1)) + 1e-3 * ca.sumsqr(d2) / ref.h ** 2)
    # tyre force above / below coasting, split into traction (ap) and braking (an), each inside the
    # fitted "ellipse" with the lateral fraction of the cornering limit
    vr = vm_c.T
    aym_v = ca.fmax(f_aym(vr).T, 0.5)
    ap = opti.variable(m)
    an = opti.variable(m)
    opti.subject_to(ax - f_coast(vr).T == ap - an)
    opti.subject_to(opti.bounded(0.0, ap, ca.inf))
    opti.subject_to(opti.bounded(0.0, an, ca.inf))
    na, nb = f_nacc(vr).T, f_nbrk(vr).T
    ux_a = ca.sqrt(ap * ap + 1e-8) / ca.fmax(f_acc(vr).T, 0.05)  # sqrt(.^2): no NaN from a slightly
    ux_b = ca.sqrt(an * an + 1e-8) / ca.fmax(f_brk(vr).T, 0.5)  # negative iterate under a real power
    # the curvature at both ends of the interval (an average would let alternating +/- curvature cancel)
    for k_end in (k0_, k1):
        uy = ca.sqrt((vm * vm * k_end) ** 2 + 1e-4) / aym_v
        opti.subject_to(ux_a ** na + uy ** na <= 1.0)
        opti.subject_to(ux_b ** nb + uy ** nb <= 1.0)
    opti.subject_to(opti.bounded(-0.35, kap, 0.35))  # >= ~3 m radius
    opti.subject_to(opti.bounded(ca.DM(lo), a, ca.DM(hi)))
    opti.subject_to(opti.bounded(0.3, v, min(v_hi, vehicle.powertrain.v_max)))
    if not ref.closed and start == "standing":
        opti.subject_to(v[0] == 0.3)
    opti.set_initial(a, a0)
    v_ws = np.clip(v_init, 0.5, None)
    opti.set_initial(v, v_ws)
    v_next = np.append(v_ws[1:], v_ws[0]) if ref.closed else v_ws[1:]
    a_ws = (v_next ** 2 - v_ws[:m] ** 2) / (2.0 * ref.h)
    opti.set_initial(ap, np.maximum(a_ws, 0.0) + 0.1)
    opti.set_initial(an, np.maximum(-a_ws, 0.0) + 0.1)
    opti.solver("ipopt", {"print_time": False}, {"print_level": 0, "max_iter": int(max_iter), "tol": 1e-6,
                                                 "acceptable_tol": 1e-4, "sb": "yes",
                                                 "mu_strategy": "adaptive"})
    try:
        sol = opti.solve()
        a_opt = np.asarray(sol.value(a), dtype=float).ravel()
        status = "optimal"
    except RuntimeError:  # not converged: use the last iterate if it is better
        a_opt = np.asarray(opti.debug.value(a), dtype=float).ravel()
        status = "not converged (last iterate)"
    a_opt = np.clip(a_opt, lo, hi)
    trk1 = line_track(ref, a_opt, f"{track.name} (racing line)", ds=min(ds, 0.5))
    lap1 = solver.solve(vehicle, trk1, start=start)
    if progress:
        progress(2, 2, f"min-time OCP ({status})", lap1.lap_time)
    cands = [Candidate(0.0, lap0.lap_time, trk0.length, float(np.max(np.abs(trk0.kappa))), it0, a0.copy(), trk0),
             Candidate(math.nan, lap1.lap_time, trk1.length, float(np.max(np.abs(trk1.kappa))), 0, a_opt, trk1)]
    best = min(cands, key=lambda c: c.lap_time)
    lap = lap1 if best is cands[1] else lap0
    return LineResult(ref, lo, hi, best, cands, reference_lap, lap, float(car_width), float(clearance),
                      int(narrow.sum()))


class MPCDriver:
    """Receding-horizon driver: acceleration and steering of a bicycle model over ``horizon`` seconds.

    States (Frenet frame of the line): lateral error ``e_y``, heading error ``e_psi``, speed ``v``,
    lateral velocity ``vy``, yaw rate ``r``, distance ``s``. Axle lateral forces saturate smoothly at the
    current capacities (``tanh``); the total force stays inside the friction circle of the current tyre
    loads; the acceleration is capped by the powertrain at the current speed; the speed stays below the
    QSS speed plan along the predicted path. Objective: distance at the end of the horizon, small path
    errors and smooth steering.
    """

    def __init__(self, car, s_ref, v_ref, k_of_s, horizon: float = 0.8, n: int = 16, path_limit: float = 0.5):
        ca = _casadi()
        self.ca = ca
        self.car = car
        self.n = n
        self.h = horizon / n
        self.path_limit = path_limit
        s_ref = np.asarray(s_ref, dtype=float)
        self.s_max = float(s_ref[-1])
        grid = np.linspace(0.0, self.s_max + 200.0, 600)
        self.f_v = ca.interpolant("vref", "linear", [grid], np.interp(grid, s_ref, v_ref, right=v_ref[-1]))
        self.f_k = ca.interpolant("kref", "linear", [grid], np.interp(grid, s_ref, k_of_s, right=k_of_s[-1]))
        self._build()
        self._guess = None

    def _build(self):
        ca, car, n, h = self.ca, self.car, self.n, self.h
        opti = ca.Opti()
        X = opti.variable(6, n + 1)  # e_y, e_psi, v, vy, r, s
        U = opti.variable(2, n)  # ax, delta
        p = opti.parameter(6)  # initial state
        cap = opti.parameter(5)  # Fy cap front, rear, total grip [N], a_max (power), alpha peak
        opti.subject_to(X[:, 0] == p)
        cost = 0
        for k in range(n):
            ey, ep, v, vy, r, s = (X[i, k] for i in range(6))
            ax, d = U[0, k], U[1, k]
            vx = ca.fmax(v, 1.0)
            af = d - (vy + car.a * r) / vx
            ar = -(vy - car.b * r) / vx
            fyf = cap[0] * ca.tanh(af / cap[4])
            fyr = cap[1] * ca.tanh(ar / cap[4])
            kp = self.f_k(s)
            sdot = (vx * ca.cos(ep) - vy * ca.sin(ep)) / (1.0 - kp * ey)
            xdot = ca.vertcat(vx * ca.sin(ep) + vy * ca.cos(ep), r - kp * sdot, ax + vy * r,
                              (fyf + fyr) / car.m - vx * r, (car.a * fyf - car.b * fyr) / car.I_yaw, sdot)
            opti.subject_to(X[:, k + 1] == X[:, k] + h * xdot)
            # friction circle of the whole car and the powertrain limit
            opti.subject_to((car.m * ax) ** 2 + (fyf + fyr) ** 2 <= cap[2] ** 2)
            opti.subject_to(ax <= cap[3])
            opti.subject_to(X[2, k + 1] <= self.f_v(X[5, k + 1]) + 0.3)
            cost += 30.0 * ey ** 2 + 5.0 * ep ** 2
            if k > 0:
                cost += 50.0 * (U[1, k] - U[1, k - 1]) ** 2
        opti.subject_to(opti.bounded(-0.5, U[1, :], 0.5))
        opti.subject_to(opti.bounded(-self.path_limit, X[0, :], self.path_limit))
        # progress (scaled to order 1) against path error and steering activity
        opti.minimize(-(X[5, n] - X[5, 0]) / (h * n) + 0.05 * cost)
        opti.solver("ipopt", {"print_time": False}, {"print_level": 0, "max_iter": 100, "sb": "yes", "tol": 1e-6, "hessian_approximation": "limited-memory",
                                                     "acceptable_iter": 0})
        self.opti, self.X, self.U, self.p, self.cap = opti, X, U, p, cap

    def command(self, state, caps) -> tuple[float, float]:
        """(ax, steer) for ``state`` = (e_y, e_psi, v, vy, r, s) and ``caps`` = (Fy cap front, rear,
        total grip, max acceleration, peak slip angle). Falls back to the last plan if IPOPT fails."""
        o = self.opti
        o.set_value(self.p, state)
        o.set_value(self.cap, caps)
        if self._guess is not None:
            o.set_initial(self.X, self._guess[0])
            o.set_initial(self.U, self._guess[1])
        else:
            o.set_initial(self.X, np.tile(np.asarray(state, dtype=float)[:, None], (1, self.n + 1)))
        try:
            sol = o.solve()
            xv, uv = sol.value(self.X), sol.value(self.U)
        except RuntimeError:
            xv, uv = o.debug.value(self.X), o.debug.value(self.U)
        # shift the plan by one step as the next warm start
        self._guess = (np.hstack([xv[:, 1:], xv[:, -1:]]), np.hstack([uv[:, 1:], uv[:, -1:]]))
        return float(uv[0, 0]), float(uv[1, 0])
