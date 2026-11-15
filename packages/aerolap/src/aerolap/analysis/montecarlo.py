"""Monte Carlo ranking confidence for design candidates (ported from fsk-lapsim E1).

Each sample perturbs every candidate vehicle with

* ``shared`` relative errors (one draw per sample, the same for all candidates: assumptions common to
  every variant, e.g. tyre grip, mass, CG height);
* ``own`` relative errors (an independent draw per candidate: e.g. the CFD error of that candidate's
  ClA / CdA),

``value * (1 + sigma * N(0, 1))``, and evaluates the objective. Because the shared part uses common random
numbers, the result answers "does B stay faster than A if our CFD is off by X %", not just "how wide is
the lap-time spread".
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from ..config import get_path, set_path
from .energy import copy_cfg
from .objectives import Objective

DEFAULT_OWN = {"aero.cla_scale": 0.05, "aero.cda_scale": 0.05}  # ASSUMED CFD error (1 sigma) per candidate

_OBJ: dict = {}


def _init(spec: dict, base_dir: str) -> None:
    _OBJ["f"] = Objective(spec, base_dir)


def _eval(cfg: dict) -> float:
    return _OBJ["f"](cfg).value


def perturbed(cfg: dict, sig: dict[str, float], z: np.ndarray) -> dict:
    out = copy_cfg(cfg)
    for (path, s), zz in zip(sig.items(), z):
        set_path(out, path, float(get_path(out, path)) * (1.0 + s * zz))
    return out


@dataclass
class MonteCarloResult:
    names: list[str]
    nominal: np.ndarray  # (n_cand,)
    samples: np.ndarray  # (n_samples, n_cand)

    def report(self, label: str, unit: str) -> str:
        order = np.argsort(self.nominal)
        rank = np.argsort(np.argsort(self.samples, axis=1), axis=1)  # 0 = fastest
        nom_rank = np.argsort(order)
        lines = [f"objective: {label}   samples: {len(self.samples)}", "",
                 f"{'candidate':28s} {'nominal':>10s} {'mean':>10s} {'std':>8s} {'P(fastest)':>11s} {'P(same rank)':>13s}"]
        for i in order:
            lines.append(f"{self.names[i][:28]:28s} {self.nominal[i]:10.3f} {self.samples[:, i].mean():10.3f} "
                         f"{self.samples[:, i].std():8.3f} {np.mean(rank[:, i] == 0):11.1%} "
                         f"{np.mean(rank[:, i] == nom_rank[i]):13.1%}")
        lines += ["", f"P(whole nominal order kept) = {np.mean(np.all(rank == nom_rank, axis=1)):.1%}", "",
                  "pairwise (nominal faster first): P(order flips), mean difference [" + unit + "]"]
        for a_i, a in enumerate(order):
            for b in order[a_i + 1:]:
                d = self.samples[:, b] - self.samples[:, a]
                lines.append(f"  {self.names[a][:24]:24s} < {self.names[b][:24]:24s}  "
                             f"P(flip) {np.mean(d < 0):6.1%}   diff {d.mean():+8.3f} +- {d.std():.3f}")
        return "\n".join(lines)


def monte_carlo(cfgs: dict[str, dict], spec: dict, base_dir: str = ".", n: int = 100,
                shared: dict[str, float] | None = None, own: dict[str, float] | None = None,
                seed: int = 1, workers: int | None = None) -> MonteCarloResult:
    shared, own = dict(shared or {}), dict(DEFAULT_OWN if own is None else own)
    names = list(cfgs)
    rng = np.random.default_rng(seed)
    zs = rng.standard_normal((n, len(shared)))
    zo = rng.standard_normal((n, len(names), len(own)))
    jobs = [cfgs[k] for k in names]  # nominal first
    for i in range(n):
        for j, k in enumerate(names):
            jobs.append(perturbed(perturbed(cfgs[k], shared, zs[i]), own, zo[i, j]))
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(spec, str(base_dir))) as ex:
        vals = np.array(list(ex.map(_eval, jobs, chunksize=4)))
    return MonteCarloResult(names, vals[:len(names)], vals[len(names):].reshape(n, len(names)))

