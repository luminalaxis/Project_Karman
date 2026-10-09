"""Monte Carlo uncertainty propagation (extension E1).

Each sample draws

* the uncertain vehicle inputs (``uncertainty.yaml: parameters``) — shared by all candidates;
* the 2D→3D conversion factors (endplate factor, Oswald e, k scale) — shared by all candidates,
  because the same conversion is applied to every 2D result;
* an independent CFD error per candidate (relative σ on Cl_2d and Cd_2d).

and evaluates the autocross lap and the endurance flying lap of the base car and of every
rear-wing candidate. Results: lap-time distributions and, per candidate, the probability of
being fastest, of keeping its nominal rank, and pairwise probabilities that the nominal order
of two candidates flips.

A coarser GGV grid (0.5 m/s, 37 θ) is used by default: the lap-time effect is ~1e−5 relative,
far below the candidate differences, and it makes the GGV build ~6× faster.
"""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..aero import AeroComponent, replace_component, synthesize
from ..events import EventSetup, run_autocross, run_endurance
from ..solver import SolverOptions
from ..vehicle import Vehicle, apply_component_change, get_param
from ..wing import WingCandidate, WingConversion, candidate_component, retrim_component
from .uncertainty import Uncertainty

METRICS = ("autocross", "endurance_lap")
MC_OPTIONS = SolverOptions(ggv_dv_mps=0.5, ggv_n_theta=37)

_STATE: dict[str, object] = {}


def _init_worker(setup: EventSetup, opt: SolverOptions) -> None:
    _STATE["setup"] = setup
    _STATE["opt"] = opt


def _metrics(veh: Vehicle) -> dict[str, float]:
    setup: EventSetup = _STATE["setup"]  # type: ignore[assignment]
    opt: SolverOptions = _STATE["opt"]  # type: ignore[assignment]
    ac = run_autocross(veh, setup.autocross_track, setup.autocross_start, opt).lap_time_s
    en = run_endurance(veh, setup.endurance_track, setup.endurance_track.length_m, opt)
    return {"autocross": ac, "endurance_lap": en.flying_lap.lap_time_s}


def _eval_variants(variants: list[tuple[str, Vehicle]]) -> dict[str, dict[str, float]]:
    return {label: _metrics(v) for label, v in variants}


@dataclass
class MCResult:
    """Monte Carlo output.

    Attributes:
        inputs: Sampled inputs, one row per sample.
        times: Long table: sample, variant (``base`` or candidate id), metric, time [s].
        nominal: Nominal (unperturbed) times: variant × metric.
        n: Number of samples.
    """

    inputs: pd.DataFrame
    times: pd.DataFrame
    nominal: pd.DataFrame
    n: int

    def matrix(self, metric: str) -> pd.DataFrame:
        """Sample × variant table of one metric."""
        t = self.times[self.times["metric"] == metric]
        return t.pivot(index="sample", columns="variant", values="time_s")

    def distribution(self) -> pd.DataFrame:
        """Percentiles of the base-car lap times."""
        rows = []
        for metric in METRICS:
            x = self.matrix(metric)["base"].to_numpy()
            rows.append(
                {
                    "metric": metric,
                    "nominal_s": float(self.nominal.loc["base", metric]),
                    "mean_s": x.mean(),
                    "std_s": x.std(ddof=1),
                    "p05_s": np.percentile(x, 5),
                    "p50_s": np.percentile(x, 50),
                    "p95_s": np.percentile(x, 95),
                }
            )
        return pd.DataFrame(rows)

    def candidates(self) -> list[str]:
        """Candidate ids (excluding the base car)."""
        return [c for c in self.nominal.index if c != "base"]

    def rank_stats(self, metric: str = "autocross") -> pd.DataFrame:
        """Per candidate: nominal rank, P(fastest), mean rank, P(same rank as nominal)."""
        cands = self.candidates()
        if not cands:
            return pd.DataFrame()
        m = self.matrix(metric)[cands]
        ranks = m.rank(axis=1, method="min")
        nom = self.nominal.loc[cands, metric].rank(method="min")
        rows = []
        for c in cands:
            rows.append(
                {
                    "variant": c,
                    "nominal_rank": int(nom[c]),
                    "nominal_s": float(self.nominal.loc[c, metric]),
                    "p_fastest": float((ranks[c] == 1).mean()),
                    "mean_rank": float(ranks[c].mean()),
                    "p_same_rank": float((ranks[c] == nom[c]).mean()),
                    "delta_to_nominal_best_p50_s": float(np.median(m[c] - m[nom.idxmin()])),
                }
            )
        return pd.DataFrame(rows).sort_values("nominal_rank").reset_index(drop=True)

    def pairwise_flip(self, metric: str = "autocross") -> pd.DataFrame:
        """For every candidate pair: nominal gap and probability that their order flips."""
        cands = self.candidates()
        m = self.matrix(metric)
        rows = []
        for a, b in itertools.combinations(cands, 2):
            na, nb = self.nominal.loc[a, metric], self.nominal.loc[b, metric]
            faster, slower = (a, b) if na <= nb else (b, a)
            flip = float((m[faster] > m[slower]).mean())
            rows.append(
                {
                    "faster_nominal": faster,
                    "slower_nominal": slower,
                    "nominal_gap_s": abs(nb - na),
                    "p_flip": flip,
                }
            )
        return pd.DataFrame(rows).sort_values("nominal_gap_s").reset_index(drop=True)

    def p_ranking_unchanged(self, metric: str = "autocross") -> float:
        """Probability that the complete candidate ranking equals the nominal one."""
        cands = self.candidates()
        if len(cands) < 2:
            return 1.0
        order_nom = list(self.nominal.loc[cands, metric].sort_values().index)
        m = self.matrix(metric)[cands]
        same = [list(row.sort_values().index) == order_nom for _, row in m.iterrows()]
        return float(np.mean(same))


def _candidate_vehicle(
    base: Vehicle,
    comps: Sequence[AeroComponent],
    cand: WingCandidate,
    ref: AeroComponent,
    conv: WingConversion,
    cl_f: float,
    cd_f: float,
    retrim: bool = False,
) -> Vehicle:
    ca = candidate_component(cand, ref, conv)
    comp = dataclasses.replace(
        ca.component, cla_m2=ca.component.cla_m2 * cl_f, cda_m2=ca.component.cda_m2 * cd_f
    )
    new = replace_component(comps, comp)
    if retrim:
        L = base.wheelbase_m
        new = retrim_component(new, synthesize(comps, L).balance_front, L)
    return apply_component_change(base, comps, new)


def run_montecarlo(
    vehicle: Vehicle,
    n: int = 100,
    seed: int = 0,
    setup: EventSetup | None = None,
    uncertainty: Uncertainty | None = None,
    candidates: Sequence[WingCandidate] = (),
    components: Sequence[AeroComponent] = (),
    component_name: str = "rear_wing",
    options: SolverOptions = MC_OPTIONS,
    workers: int = 1,
    retrim_front: bool = False,
) -> MCResult:
    """Run the Monte Carlo study (see module docstring).

    ``retrim_front`` rescales the front wing for every candidate to keep the package balance
    (isolates the wing's efficiency from its balance shift).
    """
    if n < 2:
        raise ValueError("need at least 2 samples")
    setup = setup or EventSetup.load()
    unc = uncertainty or Uncertainty.load()
    rng = np.random.default_rng(seed)
    if candidates and not components:
        raise ValueError("candidates need the baseline aero components")
    ref = next((c for c in components if c.name == component_name), None)
    if candidates and ref is None:
        raise KeyError(f"no aero component named {component_name!r}")

    bases = unc.sample_vehicles(vehicle, n, rng)
    wc = unc.wing_conversion
    defaults = WingConversion()

    def draw(key: str, default: float) -> np.ndarray:
        return wc[key].sample(rng, default, n, absolute=True) if key in wc else np.full(n, default)

    ef, oe, ks = draw("endplate_factor", 1.5), draw("oswald_e", 0.9), draw("k_lift_scale", 1.0)
    k_cl = rng.normal(1.0, unc.cl_2d_rel_sigma, (n, len(candidates)))
    k_cd = rng.normal(1.0, unc.cd_2d_rel_sigma, (n, len(candidates)))

    inputs = pd.DataFrame({k: [float(get_param(b, k)) for b in bases] for k in unc.parameters})
    inputs["endplate_factor"], inputs["oswald_e"], inputs["k_lift_scale"] = ef, oe, ks

    jobs: list[list[tuple[str, Vehicle]]] = []
    for i in range(n):
        conv = dataclasses.replace(
            defaults, endplate_factor=float(ef[i]), oswald_e=float(oe[i]), k_lift_scale=float(ks[i])
        )
        var = [("base", bases[i])]
        for j, c in enumerate(candidates):
            var.append(
                (
                    c.config_id,
                    _candidate_vehicle(
                        bases[i],
                        components,
                        c,
                        ref,
                        conv,  # type: ignore[arg-type]
                        float(k_cl[i, j]),
                        float(k_cd[i, j]),
                        retrim_front,
                    ),
                )
            )
        jobs.append(var)
    nominal_job = [("base", vehicle)] + [
        (
            c.config_id,
            _candidate_vehicle(
                vehicle,
                components,
                c,
                ref,
                defaults,
                1.0,
                1.0,
                retrim_front,  # type: ignore[arg-type]
            ),
        )
        for c in candidates
    ]

    if workers > 1:
        with ProcessPoolExecutor(
            workers, initializer=_init_worker, initargs=(setup, options)
        ) as ex:
            results = list(ex.map(_eval_variants, [nominal_job, *jobs], chunksize=2))
    else:
        _init_worker(setup, options)
        results = [_eval_variants(j) for j in [nominal_job, *jobs]]
    nominal = pd.DataFrame(results[0]).T
    rows = [
        {"sample": i, "variant": label, "metric": metric, "time_s": t}
        for i, res in enumerate(results[1:])
        for label, mets in res.items()
        for metric, t in mets.items()
    ]
    return MCResult(inputs, pd.DataFrame(rows), nominal, n)
