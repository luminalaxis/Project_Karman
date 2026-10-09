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
    alpha = np.zeros(n)

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


def racing_line(track: Track, method: str = "mincurv", car_width: float = 1.45,
                margin: float = 0.25, step: float = 2.0, default_width: float | None = None,
                ds: float | None = None, smooth_m: float = 4.0, iterations: int = 15) -> Track:
    """Racing line inside the track limits as a new :class:`Track`.

    ``method``: "mincurv" (exact minimum curvature) or "compromise" (curvature/length blend).
    """
    if track.width_left is None or track.width_right is None:
        if default_width is None:
            raise ValueError("track has no width information; pass default_width (e.g. --track-width 4)")
        wl = wr = np.full(len(track.s), default_width / 2)
    else:
        wl, wr = track.width_left, track.width_right

    k = max(int(round(step / track.ds)), 1)
    idx = np.arange(0, len(track.s), k)
    if not track.closed and idx[-1] != len(track.s) - 1:
        idx = np.append(idx, len(track.s) - 1)
    cx, cy = track.x[idx], track.y[idx]
    nx, ny = (a[idx] for a in track.normals())           # left-pointing unit normal

    half = car_width / 2 + margin
    hi = wl[idx] - half
    lo = -(wr[idx] - half)
    mid = 0.5 * (lo + hi)
    lo, hi = np.minimum(lo, mid), np.maximum(hi, mid)    # too narrow -> stay on the middle
    if not track.closed:                                 # start / finish on the centerline
        lo[[0, -1]] = hi[[0, -1]] = 0.0

    if method == "mincurv":
        alpha = _gauss_newton(cx, cy, nx, ny, lo, hi, track.closed, iterations)
    elif method == "compromise":
        alpha = _compromise(cx, cy, nx, ny, lo, hi, track.closed)
    else:
        raise ValueError(f"unknown racing line method {method!r}")
    px, py = cx + alpha * nx, cy + alpha * ny
    return track_from_dict({"x": px.tolist(), "y": py.tolist(), "closed": track.closed},
                           ds=ds or track.ds, smooth_m=max(smooth_m, 2 * step),  # hide polygon kinks
                           name=f"{track.name} ({method} line)")


def min_curvature_line(track: Track, car_width: float = 1.45, margin: float = 0.25, **kw) -> Track:
    return racing_line(track, "mincurv", car_width, margin, **kw)


def best_line(veh, track: Track, car_width: float = 1.45, margin: float = 0.25, **kw):
    """Build every line type, simulate them with ``veh`` and return (fastest track, {method: lap time})."""
    from .solver import simulate
    lines = {m: racing_line(track, m, car_width, margin, **kw) for m in ("mincurv", "compromise")}
    times = {m: simulate(veh, ln).lap_time for m, ln in lines.items()}
    best = min(times, key=times.get)
    return lines[best], times
