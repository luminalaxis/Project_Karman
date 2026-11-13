"""Racing lines inside the track limits: minimum curvature and a curvature/length compromise.

The path is the centerline shifted along its normals, p_i = c_i + alpha_i * n_i, with
-(w_right - half) <= alpha_i <= (w_left - half), where ``half`` is half the car width plus a
safety margin.  The cost is the integral of curvature squared, sum l_i * kappa_i^2, with kappa_i the exact
three-point (Menger) curvature of the shifted points.  It is minimised by Gauss-Newton: each
step linearises kappa(alpha) and solves the resulting box-constrained quadratic programme with a
small primal active-set method (numpy only), inside a trust region.  Minimum curvature is the standard fast stand-in for the
minimum-time line: it maximises corner radii, so it is close to optimal on tight FS layouts.
"""

from __future__ import annotations

import numpy as np

from .track import Track, track_from_dict


def _menger(px, py, closed):
    """Signed curvature of the circle through each point and its neighbours (exact for
    non-uniform spacing); open ends get 0."""
    ax_, ay_ = np.roll(px, 1), np.roll(py, 1)
    bx_, by_ = np.roll(px, -1), np.roll(py, -1)
    cross = (px - ax_) * (by_ - ay_) - (py - ay_) * (bx_ - ax_)
    a = np.hypot(px - ax_, py - ay_)
    b = np.hypot(bx_ - px, by_ - py)
    c = np.hypot(bx_ - ax_, by_ - ay_)
    k = 2.0 * cross / np.maximum(a * b * c, 1e-12)
    ell = 0.5 * (a + b)
    if not closed:
        k[[0, -1]] = 0.0
        ell[[0, -1]] = 0.0
    return k, ell


def _compromise(cx, cy, nx, ny, lo, hi, closed):
    """min sum |p_{i-1} - 2 p_i + p_{i+1}|^2 on the *reference* grid: one QP.

    Because the reference spacing is kept fixed, this cost also rewards shorter paths, so the
    result sits between the minimum-curvature and the shortest path, which is often closer to
    the minimum-time line on tight layouts.
    """
    n = len(cx)
    D = np.zeros((n, n))
    for i in (range(n) if closed else range(1, n - 1)):
        D[i, (i - 1) % n] += 1.0
        D[i, i] -= 2.0
        D[i, (i + 1) % n] += 1.0
    if not closed:
        D = D[1:-1]
    Ax, Ay = D * nx, D * ny                              # D @ diag(n)
    Q = Ax.T @ Ax + Ay.T @ Ay
    c = Ax.T @ (D @ cx) + Ay.T @ (D @ cy)
    return box_qp(Q, c, lo, hi)


def _gauss_newton(cx, cy, nx, ny, lo, hi, closed, iterations, trust=1.0, h=1e-4):
    """min sum l_i * kappa_i(alpha)^2 over the box, by Gauss-Newton on the exact curvature.

    kappa_i depends only on alpha_{i-1..i+1}, so the Jacobian is tridiagonal (cyclic) and is
    built from three perturbations (every third offset at a time).
    """
    n = len(cx)
    alpha = np.clip(np.zeros(n), lo, hi)                # start feasible (slalom cones!)

    def curv(a):
        return _menger(cx + a * nx, cy + a * ny, closed)

    def cost(a):
        k, ell = curv(a)
        return float(np.sum(ell * k * k))

    f = cost(alpha)
    rows = np.arange(n)
    # colour the offsets so perturbed ones are >= 3 apart (also across the closing seam)
    colour = rows % 3
    tail = n % 3
    if tail:
        colour[n - tail:] = 3 + np.arange(tail)
    for _ in range(iterations):
        k0, ell = curv(alpha)
        J = np.zeros((n, n))
        for col in range(colour.max() + 1):
            pert = colour == col
            kp, _ = curv(alpha + h * pert)
            dk = (kp - k0) / h
            for off in (-1, 0, 1):
                j = (rows + off) % n
                sel = pert[j]
                if not closed:
                    sel &= (rows + off >= 0) & (rows + off < n)
                J[rows[sel], j[sel]] = dk[sel]
        W = ell[:, None]
        Q = J.T @ (W * J)
        c = J.T @ (ell * (k0 - J @ alpha))
        lo_t, hi_t = np.maximum(lo, alpha - trust), np.minimum(hi, alpha + trust)
        new = box_qp(Q, c, lo_t, hi_t, x0=alpha)
        f_new = cost(new)
        if f_new < f:
            step = np.max(np.abs(new - alpha))
            alpha, f = new, f_new
            if step < 1e-3:
                break
        else:
            trust *= 0.5
            if trust < 1e-3:
                break
    return alpha


def box_qp(Q: np.ndarray, c: np.ndarray, lo: np.ndarray, hi: np.ndarray, max_iter: int = 500,
           x0: np.ndarray | None = None):
    """min 0.5 x'Qx + c'x  s.t. lo <= x <= hi  (Q symmetric positive semi-definite)."""
    n = len(c)
    Q = Q + 1e-9 * np.trace(Q) / n * np.eye(n)
    x = np.clip(np.zeros(n) if x0 is None else x0, lo, hi)
    for _ in range(max_iter):
        g = Q @ x + c
        at_lo = (x <= lo + 1e-12) & (g > 0)
        at_hi = (x >= hi - 1e-12) & (g < 0)
        free = ~(at_lo | at_hi)
        target = x.copy()
        if free.any():
            rhs = -(c[free] + Q[np.ix_(free, ~free)] @ x[~free])
            target[free] = np.linalg.solve(Q[np.ix_(free, free)], rhs)
        d = target - x
        if np.max(np.abs(d)) < 1e-10:
            break
        with np.errstate(divide="ignore", invalid="ignore"):
            t_hi = np.where(d > 0, (hi - x) / d, np.inf)
            t_lo = np.where(d < 0, (lo - x) / d, np.inf)
        step = min(1.0, float(np.min(t_hi)), float(np.min(t_lo)))
        x = np.clip(x + step * d, lo, hi)
    return x


def _hermite(tk, ak, n: int, closed: bool) -> np.ndarray:
    """C1 cubic Hermite (Catmull-Rom style slopes) through knots (tk, ak), evaluated at 0..n-1."""
    tk, ak = np.asarray(tk, float), np.asarray(ak, float)
    if closed:
        tk = np.concatenate([[tk[-1] - n], tk, [tk[0] + n]])
        ak = np.concatenate([[ak[-1]], ak, [ak[0]]])
    m = np.empty_like(ak)
    m[1:-1] = (ak[2:] - ak[:-2]) / (tk[2:] - tk[:-2])
    m[0] = (ak[1] - ak[0]) / (tk[1] - tk[0])
    m[-1] = (ak[-1] - ak[-2]) / (tk[-1] - tk[-2])
    t = np.arange(n, dtype=float)
    j = np.clip(np.searchsorted(tk, t, side="right") - 1, 0, len(tk) - 2)
    h = tk[j + 1] - tk[j]
    u = (t - tk[j]) / h
    h00, h10 = 2 * u**3 - 3 * u**2 + 1, u**3 - 2 * u**2 + u
    h01, h11 = -2 * u**3 + 3 * u**2, u**3 - u**2
    return h00 * ak[j] + h10 * h * m[j] + h01 * ak[j + 1] + h11 * h * m[j + 1]


def _cubic_spline(tk, ak, n: int, closed: bool) -> np.ndarray:
    """C2 cubic spline through knots (tk, ak) evaluated at 0..n-1 (periodic when closed,
    natural otherwise) - continuous curvature, so no spikes at the knots."""
    tk, ak = np.asarray(tk, float), np.asarray(ak, float)
    m = len(tk)
    if closed:
        h = np.diff(np.append(tk, tk[0] + n))            # h[j] = t[j+1] - t[j], wrapping
        A = np.zeros((m, m))
        r = np.zeros(m)
        for j in range(m):
            hp, hn = h[j - 1], h[j]
            A[j, (j - 1) % m] = hp
            A[j, j] = 2 * (hp + hn)
            A[j, (j + 1) % m] = hn
            r[j] = 6 * ((ak[(j + 1) % m] - ak[j]) / hn - (ak[j] - ak[j - 1]) / hp)
        M = np.linalg.solve(A, r)
        tk_e, ak_e, M_e = np.append(tk, tk[0] + n), np.append(ak, ak[0]), np.append(M, M[0])
        t = np.arange(n, dtype=float)
        t = np.where(t < tk[0], t + n, t)
    else:
        h = np.diff(tk)
        A = np.zeros((m, m))
        r = np.zeros(m)
        A[0, 0] = A[-1, -1] = 1.0
        for j in range(1, m - 1):
            A[j, j - 1], A[j, j], A[j, j + 1] = h[j - 1], 2 * (h[j - 1] + h[j]), h[j]
            r[j] = 6 * ((ak[j + 1] - ak[j]) / h[j] - (ak[j] - ak[j - 1]) / h[j - 1])
        M = np.linalg.solve(A, r)
        tk_e, ak_e, M_e = tk, ak, M
        t = np.arange(n, dtype=float)
    j = np.clip(np.searchsorted(tk_e, t, side="right") - 1, 0, len(tk_e) - 2)
    hj = tk_e[j + 1] - tk_e[j]
    a, b = (tk_e[j + 1] - t) / hj, (t - tk_e[j]) / hj
    return (a * ak_e[j] + b * ak_e[j + 1]
            + ((a**3 - a) * M_e[j] + (b**3 - b) * M_e[j + 1]) * hj**2 / 6)


def racing_line(track: Track, method: str = "mincurv", car_width: float = 1.45,
                margin: float = 0.25, step: float = 2.0, default_width: float | None = None,
                ds: float | None = None, smooth_m: float = 4.0, iterations: int = 15,
                slalom=None, free: np.ndarray | None = None,
                fixed: np.ndarray | None = None) -> Track:
    """Racing line inside the track limits as a new :class:`Track`.

    ``method``: "mincurv" (exact minimum curvature) or "compromise" (curvature/length blend).
    ``slalom``: (groups, sides, d) from :mod:`lapsim.slalom` - a knot is placed at every
    slalom cone and the line must pass it on the given side with clearance ``d``.
    ``free``: boolean mask over track points; elsewhere the line is held on the path.
    ``fixed``: offsets over track points (NaN = free) the line must take, e.g. a slalom S.
    The returned track carries ``offset``: its lateral offset from ``track`` per point.
    """
    n = len(track.s)
    if track.width_left is None or track.width_right is None:
        if default_width is None:
            raise ValueError("track has no width information; pass default_width (e.g. --track-width 4)")
        wl = wr = np.full(n, default_width / 2)
    else:
        wl, wr = track.width_left, track.width_right

    k = max(int(round(step / track.ds)), 1)
    idx = np.arange(0, n, k)
    if not track.closed and idx[-1] != n - 1:
        idx = np.append(idx, n - 1)
    cone_req, cone_off, d_req = {}, {}, 0.0               # knot position -> (lo, hi) requirement
    if slalom is not None:
        groups, sides, d = slalom
        d_req = d
        for g, sd in zip(groups, sides):
            for gate, side in zip(g, sd):
                i = gate.index % n
                j = int(np.argmin(np.abs(idx - i) if not track.closed
                                  else np.minimum(np.abs(idx - i), n - np.abs(idx - i))))
                idx[j] = i                                # move the nearest knot onto the cone
                # the line crosses a slalom at an angle ~ atan(2d / spacing): lateral clearance
                # must exceed d / cos(angle) for the true distance to reach d
                spacing = _gate_spacing(g, gate) * track.ds
                d_lat = d * np.sqrt(1 + (2 * d / max(spacing, 1e-6)) ** 2)
                cone_req[i] = ((gate.offset + d_lat, np.inf) if side > 0
                               else (-np.inf, gate.offset - d_lat))
                cone_off[i] = gate.offset
        idx = np.unique(idx)
    cx, cy = track.x[idx], track.y[idx]
    nx, ny = (a[idx] for a in track.normals())           # left-pointing unit normal

    half = car_width / 2 + margin
    hi = wl[idx] - half
    lo = -(wr[idx] - half)
    mid = 0.5 * (lo + hi)
    lo, hi = np.minimum(lo, mid), np.maximum(hi, mid)    # too narrow -> stay on the middle
    if free is not None:
        hold = ~free[idx]
        lo[hold] = hi[hold] = 0.0
    if fixed is not None:
        f = fixed[idx]
        pin = np.isfinite(f)
        lo[pin] = hi[pin] = f[pin]
    for j, i in enumerate(idx):
        if i in cone_req:
            rl, rh = cone_req[i]
            lo[j], hi[j] = max(lo[j], rl), min(hi[j], rh)
            if lo[j] > hi[j]:                            # not enough room: honour the cone side
                lo[j] = hi[j] = rl if np.isfinite(rl) else rh
    if not track.closed:                                 # start / finish on the centerline
        lo[[0, -1]] = hi[[0, -1]] = 0.0

    if method not in ("mincurv", "compromise"):
        raise ValueError(f"unknown racing line method {method!r}")
    fnx, fny = track.normals()
    cone_knots = [j for j, i in enumerate(idx) if i in cone_req and lo[j] < hi[j]]
    for _ in range(10):
        if method == "mincurv":
            alpha = _gauss_newton(cx, cy, nx, ny, lo, hi, track.closed, iterations)
        else:
            alpha = _compromise(cx, cy, nx, ny, lo, hi, track.closed)
        # C2 spline through the knot positions: continuous curvature without the moving-average
        # smoothing that would cut hairpins past the limits and shave the slalom
        px = _cubic_spline(idx, cx + alpha * nx, n, track.closed)
        py = _cubic_spline(idx, cy + alpha * ny, n, track.closed)
        if not cone_knots:
            break
        # the line crosses the slalom at an angle, so the true (Euclidean) distance to a cone is
        # shorter than the lateral offset: raise the requirement of short cones and re-solve
        short = False
        for j in cone_knots:
            i = idx[j]
            cxy = np.array([cx[j] + cone_off[i] * nx[j], cy[j] + cone_off[i] * ny[j]])
            dist = np.min(np.hypot(px - cxy[0], py - cxy[1]))
            if dist < d_req - 0.01:
                short = True
                extra = (d_req - dist) * 1.5
                if np.isfinite(cone_req[i][0]):
                    lo[j] = min(lo[j] + extra, max(hi[j], lo[j]))
                else:
                    hi[j] = max(hi[j] - extra, min(lo[j], hi[j]))
        if not short:
            break
    # positions: only light smoothing (keeps the slalom geometry); curvature: smoothed like the
    # track path so lines and centerline are compared on equal terms
    line = track_from_dict({"x": px.tolist(), "y": py.tolist(), "closed": track.closed},
                           ds=ds or track.ds, smooth_m=1.0, kappa_smooth_m=smooth_m,
                           name=f"{track.name} ({method} line)")
    line.slalom_cones = track.slalom_cones
    line.offset = (px - track.x) * fnx + (py - track.y) * fny
    return line


def _gate_spacing(group, gate) -> float:
    idx = [g.index for g in group]
    if len(idx) < 2:
        return np.inf
    k = idx.index(gate.index)
    gaps = [idx[k] - idx[k - 1]] if k > 0 else []
    gaps += [idx[k + 1] - idx[k]] if k + 1 < len(idx) else []
    return float(min(gaps))


def min_curvature_line(track: Track, car_width: float = 1.45, margin: float = 0.25, **kw) -> Track:
    return racing_line(track, "mincurv", car_width, margin, **kw)


def _slalom_window(track: Track, groups) -> np.ndarray:
    n = len(track.s)
    free = np.zeros(n, bool)
    for g in groups:
        idx = [x.index for x in g]
        lead = int(max(np.mean(np.diff(idx)) if len(idx) > 1 else 0, 8.0 / track.ds))
        span = np.arange(idx[0] - lead, idx[-1] + lead + 1)
        if track.closed:
            free[span % n] = True
        else:
            free[span[(span >= 0) & (span < n)]] = True
    return free


def driven_path(veh, track: Track, mode: str = "center", car_width: float = 1.45, margin: float = 0.25,
                default_width: float | None = None, slalom: bool = True, smooth_m: float = 4.0):
    """The path the car drives, and an info dict.

    mode: "center" (track path, rebuilt like the lines; slalom sections re-routed through the cones),
    "compromise" / "mincurv" (racing lines) or "best" (whichever line is faster).
    Slalom cones (``track.slalom_cones``) are passed alternately left / right; the first side
    of each slalom is chosen by lap time.
    """
    from .slalom import (choose_sides, clearance, find_slaloms, pass_distance, slalom_path)
    from .solver import simulate

    info = {"line_times": {}, "slalom_sides": None, "cone_clearance": None}
    groups = find_slaloms(track) if slalom else []
    kw = dict(default_width=default_width, smooth_m=smooth_m)
    d = pass_distance(car_width, margin)
    has_width = track.width_left is not None or default_width is not None

    def centre_path(sides):
        # rebuilt through the same spline knots as the racing lines (offset 0 outside slaloms),
        # so centre and racing lines are compared on equal geometric terms
        if not has_width:
            return slalom_path(track, groups, sides, d, car_width, margin) if groups else track
        if not groups:
            return racing_line(track, "compromise", car_width, margin,
                               free=np.zeros(len(track.s), bool), **kw)
        return racing_line(track, "compromise", car_width, margin, slalom=(groups, sides, d),
                           free=_slalom_window(track, groups), **kw)

    sides = None
    if groups:
        sides = choose_sides(veh, centre_path, groups)
        info["slalom_sides"] = sides
    sl, fixed = None, None
    centre = centre_path(sides)
    if groups and has_width:
        # racing lines keep the slalom S of the centre path and optimise everything else:
        # the geometric line costs are poor stand-ins for lap time inside a slalom
        sl = (groups, sides, d)
        fixed = np.where(_slalom_window(track, groups), centre.offset, np.nan)

    if mode == "center":
        path = centre
    elif mode in ("mincurv", "compromise"):
        path = racing_line(track, mode, car_width, margin, slalom=sl, fixed=fixed, **kw)
    elif mode == "best":
        lines = {m: racing_line(track, m, car_width, margin, slalom=sl, fixed=fixed, **kw)
                 for m in ("mincurv", "compromise")}
        info["line_times"] = {m: simulate(veh, ln).lap_time for m, ln in lines.items()}
        path = lines[min(info["line_times"], key=info["line_times"].get)]
    else:
        raise ValueError(f"unknown path mode {mode!r}")
    if groups:
        info["cone_clearance"] = clearance(path, track.slalom_cones)
        info["pass_distance"] = d
    return path, info


def best_line(veh, track: Track, car_width: float = 1.45, margin: float = 0.25, **kw):
    """Fastest racing line (and lap time per method), slalom cones included."""
    path, info = driven_path(veh, track, "best", car_width, margin, **kw)
    return path, info["line_times"]
