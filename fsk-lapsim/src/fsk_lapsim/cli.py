"""Command line interface (typer): ``fsk-lapsim --help``."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer

from . import plots
from .config import default_config_dir, default_data_dir, resolve_path
from .events import EVENT_NAMES, EventSetup, run_events
from .solver import SolverOptions, simulate
from .vehicle import Vehicle, load_vehicle

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="FSK lap time simulator and aero sensitivity tool (results are only as "
    "good as the ASSUMED inputs — see `fsk-lapsim assumptions`).",
)

VehicleOpt = Annotated[str | None, typer.Option("--vehicle", "-v", help="vehicle.yaml")]
EventsOpt = Annotated[str | None, typer.Option("--events", "-e", help="events.yaml")]
ModelOpt = Annotated[str, typer.Option("--model", "-m", help="axle (B) | point_mass (A)")]
DsOpt = Annotated[float | None, typer.Option("--ds", help="distance step [m]")]
OutOpt = Annotated[Path | None, typer.Option("--out", "-o", help="output file (CSV/PNG/HTML)")]
PlotOpt = Annotated[Path | None, typer.Option("--plot", "-p", help="save a PNG figure")]

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)


def _ctx(
    vehicle: str | None, events: str | None, model: str, ds: float | None
) -> tuple[Vehicle, EventSetup, SolverOptions]:
    veh = load_vehicle(vehicle)
    setup = EventSetup.load(events)
    opt = SolverOptions(ds_m=ds or setup.ds_m, model=model)
    return veh, setup, opt


def parse_range(text: str) -> list[float]:
    """``"a:b:step"`` (inclusive) or ``"x,y,z"`` → list of floats."""
    if ":" in text:
        a, b, st = (float(x) for x in text.split(":"))
        if st <= 0 or b < a:
            raise typer.BadParameter(f"bad range {text!r}")
        n = round((b - a) / st)
        return [round(a + i * st, 10) for i in range(n + 1)]
    return [float(x) for x in text.split(",") if x.strip()]


def _save_df(df: pd.DataFrame, out: Path | None) -> None:
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        typer.echo(f"saved {out}")


def _save_fig(fig: object, path: Path | None) -> None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        plots.save(fig, str(path))  # type: ignore[arg-type]
        typer.echo(f"saved {path}")


@app.command()
def run(
    vehicle: VehicleOpt = None,
    events: EventsOpt = None,
    model: ModelOpt = "axle",
    ds: DsOpt = None,
    event: Annotated[
        list[str] | None, typer.Option("--event", help="limit to these events")
    ] = None,
    track: Annotated[
        str | None, typer.Option("--track", "-t", help="simulate one lap of this track YAML")
    ] = None,
    start: Annotated[str, typer.Option(help="auto | standing | flying (with --track)")] = "auto",
    channels: Annotated[Path | None, typer.Option(help="CSV of lap channels")] = None,
    plot_dir: Annotated[Path | None, typer.Option(help="directory for PNG figures")] = None,
) -> None:
    """Run the events (or one lap of --track) and print the results (ASSUMED inputs)."""
    veh, setup, opt = _ctx(vehicle, events, model, ds)
    if track:
        from .trackgen import load_track

        tr = load_track(track)
        lap = simulate(veh, tr, opt, start=start)  # type: ignore[arg-type]
        typer.echo(pd.Series(lap.summary()).round(4).to_string())
        if channels:
            _save_df(lap.to_frame(), channels)
        if plot_dir:
            _save_fig(plots.track_map(tr, lap), plot_dir / f"{tr.name}_map.png")
            _save_fig(plots.speed_trace([(tr.name, lap)]), plot_dir / f"{tr.name}_speed.png")
        return
    evs = tuple(event) if event else EVENT_NAMES
    res = run_events(veh, setup, opt, evs)
    typer.echo("ASSUMED-input results (가정값 기반)")
    typer.echo(pd.Series(res.metrics()).round(4).to_string())
    for w in res.warnings:
        typer.echo(f"WARNING: {w}")
    if channels and res.autocross is not None:
        _save_df(res.autocross.to_frame(), channels)
    if plot_dir:
        from .ggv import cached_ggv

        if res.autocross is not None:
            _save_fig(
                plots.track_map(setup.autocross_track, res.autocross),
                plot_dir / "autocross_map.png",
            )
            _save_fig(
                plots.speed_trace([("autocross", res.autocross)]), plot_dir / "autocross_speed.png"
            )
        if res.endurance is not None:
            _save_fig(
                plots.motor_map(res.endurance.flying_lap, veh),
                plot_dir / "motor_operating_points.png",
            )
        _save_fig(plots.ggv_plot(cached_ggv(veh, model)), plot_dir / "ggv.png")


@app.command()
def sensitivity(
    vehicle: VehicleOpt = None,
    events: EventsOpt = None,
    model: ModelOpt = "axle",
    ds: DsOpt = None,
    extra: Annotated[bool, typer.Option(help="also the extra assumed parameters")] = False,
    event: Annotated[str, typer.Option(help="event for plot/equivalence table")] = "autocross",
    out: OutOpt = None,
    plot: PlotOpt = None,
) -> None:
    """Central-difference sensitivities, equivalence table and measurement priority."""
    from .analysis.sensitivity import (
        CORE_PARAMS,
        EXTRA_PARAMS,
        equivalence_table,
        measurement_priority,
    )
    from .analysis.sensitivity import sensitivity as sens_fn
    from .analysis.uncertainty import Uncertainty

    veh, setup, opt = _ctx(vehicle, events, model, ds)
    res = sens_fn(veh, setup, opt, CORE_PARAMS + (EXTRA_PARAMS if extra else ()))
    typer.echo("Δt per display unit [s] (endurance_energy in kWh):")
    piv = res.pivot()
    piv.insert(0, "per", res.table.drop_duplicates("label").set_index("label")["display_unit"])
    typer.echo(piv.round(4).to_string())
    typer.echo(f"\nEquivalence on {event} (change costing the same as +1 kg):")
    typer.echo(equivalence_table(res, event).round(5).to_string(index=False))
    typer.echo("\nMeasurement priority (|dt/dp| × σ):")
    typer.echo(measurement_priority(res, Uncertainty.load().sigmas()).round(4).to_string())
    _save_df(res.table, out)
    if plot:
        _save_fig(plots.sensitivity_bars(res, event), plot)


@app.command()
def sweep(
    vehicle: VehicleOpt = None,
    events: EventsOpt = None,
    model: ModelOpt = "axle",
    ds: DsOpt = None,
    cla: Annotated[str, typer.Option(help="ClA grid a:b:step or list")] = "2.5:4.5:0.5",
    cda: Annotated[str, typer.Option(help="CdA grid a:b:step or list")] = "1.0:1.8:0.2",
    candidates: Annotated[str | None, typer.Option(help="rear-wing CSV to mark")] = None,
    no_candidates: Annotated[bool, typer.Option(help="do not mark candidates")] = False,
    out: OutOpt = None,
    plot: PlotOpt = None,
) -> None:
    """ClA × CdA grid → autocross time and endurance energy maps."""
    from .aero import components_from_config
    from .analysis.sweep import sweep_cla_cda
    from .config import load_aero_components_config
    from .wing import candidate_vehicle, read_wing_candidates

    veh, setup, opt = _ctx(vehicle, events, model, ds)
    points: list[tuple[str, Vehicle]] = [("current", veh)]
    if not no_candidates:
        comps = components_from_config(load_aero_components_config())
        path = (
            resolve_path(candidates)
            if candidates
            else (default_data_dir() / "rear_wing_candidates_DUMMY.csv")
        )
        points += [
            (c.config_id, candidate_vehicle(veh, comps, c)[0]) for c in read_wing_candidates(path)
        ]
    res = sweep_cla_cda(veh, parse_range(cla), parse_range(cda), setup, opt, points)
    df = res.to_frame()
    typer.echo("autocross time [s] (rows CdA, cols ClA):")
    typer.echo(
        df.pivot(index="CdA_m2", columns="ClA_m2", values="autocross_s").round(3).to_string()
    )
    typer.echo("\nendurance energy [kWh]:")
    typer.echo(
        df.pivot(index="CdA_m2", columns="ClA_m2", values="endurance_kwh").round(3).to_string()
    )
    typer.echo("\nmarked designs:")
    typer.echo(res.points.round(4).to_string(index=False))
    _save_df(df, out)
    if plot:
        _save_fig(plots.sweep_contours(res), plot)


@app.command()
def compare(
    vehicle: VehicleOpt = None,
    events: EventsOpt = None,
    model: ModelOpt = "axle",
    ds: DsOpt = None,
    candidates: Annotated[str | None, typer.Option(help="rear-wing CSV (default DUMMY)")] = None,
    reference: Annotated[str | None, typer.Option(help="reference config_id")] = None,
    retrim_front: Annotated[bool, typer.Option(help="re-trim front wing to keep balance")] = False,
    event: Annotated[list[str] | None, typer.Option("--event")] = None,
    out: OutOpt = None,
    plot: PlotOpt = None,
) -> None:
    """Rank rear-wing candidates per event (2D results via the rough 2D→3D conversion)."""
    from .aero import components_from_config
    from .analysis.compare import compare_candidates
    from .config import load_aero_components_config
    from .wing import ROUGH_NOTE, read_wing_candidates

    veh, setup, opt = _ctx(vehicle, events, model, ds)
    comps = components_from_config(load_aero_components_config())
    path = (
        resolve_path(candidates)
        if candidates
        else (default_data_dir() / "rear_wing_candidates_DUMMY.csv")
    )
    cmp = compare_candidates(
        veh,
        comps,
        read_wing_candidates(path),
        setup,
        opt,
        tuple(event) if event else EVENT_NAMES,
        reference,
        retrim_front=retrim_front,
    )
    if cmp.dummy:
        typer.echo("*** DUMMY candidate data — ranking is NOT meaningful ***")
    typer.echo(f"NOTE: {ROUGH_NOTE}")
    keep = ["config_id", "source", "rw_ClA_m2", "rw_CdA_m2", "balance_front", "mass_kg"]
    keep += [c for c in cmp.table.columns if c.startswith(("d_", "rank_"))]
    typer.echo(cmp.table[keep].round(4).to_string(index=False))
    _save_df(cmp.table, out)
    if plot:
        _save_fig(plots.compare_deltas(cmp), plot)


@app.command("balance-opt")
def balance_opt(
    vehicle: VehicleOpt = None,
    events: EventsOpt = None,
    model: ModelOpt = "axle",
    ds: DsOpt = None,
    objective: Annotated[
        str, typer.Option(help="autocross | endurance_lap | combined")
    ] = "autocross",
    bounds: Annotated[str, typer.Option("--range", help="lo:hi front balance")] = "0.30:0.65",
    plot: PlotOpt = None,
) -> None:
    """Front aero balance that minimises lap time (QSS optimum: no stability margin)."""
    from .analysis.balance_opt import optimise_balance

    veh, setup, opt = _ctx(vehicle, events, model, ds)
    lo, hi = (float(x) for x in bounds.split(":"))
    res = optimise_balance(veh, setup, opt, objective, (lo, hi))
    typer.echo(
        f"optimum balance {res.best_balance * 100:.2f} % → {res.best_value:.4f} s "
        f"(current {res.base_balance * 100:.1f} % → {res.base_value:.4f} s)"
    )
    typer.echo("cornering limits at the optimum [g]:")
    typer.echo(res.limits_at_best.round(3).to_string())
    if plot:
        _save_fig(plots.balance_scan(res), plot)


@app.command("active-aero")
def active_aero(
    vehicle: VehicleOpt = None,
    events: EventsOpt = None,
    model: ModelOpt = "axle",
    ds: DsOpt = None,
    drs: Annotated[str | None, typer.Option(help="drs.yaml")] = None,
    min_speed: Annotated[float | None, typer.Option(help="fixed opening speed [m/s]")] = None,
    plot: PlotOpt = None,
) -> None:
    """DRS evaluation: gains, switches, open fraction and close-command lead."""
    import dataclasses

    from .active_aero import DRSSettings, drs_event_gains, lead_distance_table

    veh, setup, opt = _ctx(vehicle, events, model, ds)
    st = DRSSettings.load(drs)
    if min_speed is not None:
        st = dataclasses.replace(st, min_speed_mps=min_speed)
    g = drs_event_gains(veh, st, setup, opt)
    typer.echo(g.table.round(3).to_string(index=False))
    typer.echo("\nendurance-lap open zones:")
    typer.echo(g.laps["endurance_lap"].zones.round(2).to_string(index=False))
    typer.echo("\nclose-command lead distance [m] = speed × transition time:")
    typer.echo(
        lead_distance_table([60, 80, 100, 120, 140], [0.1, 0.25, 0.5])
        .round(2)
        .to_string(index=False)
    )
    if plot:
        _save_fig(plots.drs_trace(g.laps["endurance_lap"]), plot)


@app.command()
def report(
    out: Annotated[Path, typer.Option("--out", "-o")] = Path("out/report.html"),
    vehicle: VehicleOpt = None,
    events: EventsOpt = None,
    model: ModelOpt = "axle",
    candidates: Annotated[str | None, typer.Option(help="rear-wing CSV")] = None,
    quick: Annotated[bool, typer.Option(help="coarser sweep, core sensitivities only")] = False,
    mc: Annotated[int, typer.Option(help="Monte Carlo samples (0 = skip)")] = 0,
) -> None:
    """Build the single-file HTML report (Korean)."""
    from .report import ReportOptions, write_report

    opts = ReportOptions(
        vehicle=vehicle,
        events=events,
        model=model,
        candidates=candidates,
        quick=quick,
        montecarlo_samples=mc,
    )
    path = write_report(out, opts, lambda m: typer.echo(f"  … {m}"))
    typer.echo(f"saved {path}")


@app.command()
def montecarlo(
    vehicle: VehicleOpt = None,
    events: EventsOpt = None,
    n: Annotated[int, typer.Option("--n", help="number of samples")] = 100,
    seed: int = 0,
    workers: Annotated[int, typer.Option(help="parallel processes")] = 4,
    candidates: Annotated[str | None, typer.Option(help="rear-wing CSV (default DUMMY)")] = None,
    no_candidates: Annotated[bool, typer.Option(help="only the base car")] = False,
    uncertainty: Annotated[str | None, typer.Option(help="uncertainty.yaml")] = None,
    retrim_front: Annotated[
        bool, typer.Option(help="keep balance by re-trimming front wing")
    ] = False,
    out: OutOpt = None,
    plot: PlotOpt = None,
) -> None:
    """Monte Carlo: lap-time distribution and rear-wing ranking flip probabilities (E1)."""
    from .aero import components_from_config
    from .analysis.montecarlo import run_montecarlo
    from .analysis.uncertainty import Uncertainty
    from .config import load_aero_components_config
    from .wing import read_wing_candidates

    veh = load_vehicle(vehicle)
    setup = EventSetup.load(events)
    comps = components_from_config(load_aero_components_config())
    cands = []
    if not no_candidates:
        path = (
            resolve_path(candidates)
            if candidates
            else (default_data_dir() / "rear_wing_candidates_DUMMY.csv")
        )
        cands = read_wing_candidates(path)
        if any(c.is_dummy for c in cands):
            typer.echo("*** DUMMY candidate data — probabilities are NOT meaningful ***")
    res = run_montecarlo(
        veh,
        n,
        seed,
        setup,
        Uncertainty.load(uncertainty),
        cands,
        comps,
        workers=workers,
        retrim_front=retrim_front,
    )
    typer.echo("base-car lap time distribution:")
    typer.echo(res.distribution().round(3).to_string(index=False))
    for metric in ("autocross", "endurance_lap"):
        if cands:
            typer.echo(
                f"\n{metric}: ranking robustness "
                f"(P(full ranking unchanged) = {res.p_ranking_unchanged(metric):.2f})"
            )
            typer.echo(res.rank_stats(metric).round(3).to_string(index=False))
            typer.echo("closest pairs:")
            typer.echo(res.pairwise_flip(metric).head(8).round(3).to_string(index=False))
    _save_df(res.times, out)
    if plot:
        if cands:
            _save_fig(plots.rank_probability(res.rank_stats("autocross"), "autocross"), plot)
        else:
            x = res.matrix("autocross")["base"]
            _save_fig(
                plots.histogram(
                    x,
                    "autocross time [s]",
                    "Monte Carlo autocross",
                    [("nominal", float(res.nominal.loc["base", "autocross"]))],
                ),
                plot,
            )


@app.command()
def gps(
    log: Annotated[
        str | None, typer.Argument(help="NMEA log (default: bundled DUMMY sample)")
    ] = None,
    vehicle: VehicleOpt = None,
    model: ModelOpt = "axle",
    lap: Annotated[int, typer.Option(help="lap index to build the track from")] = 0,
    method: Annotated[str, typer.Option(help="course | position")] = "course",
    course_sigma: Annotated[float, typer.Option(help="course noise σ [deg]")] = 0.5,
    gate: Annotated[float, typer.Option(help="start gate radius [m]")] = 15.0,
    export_track: Annotated[Path | None, typer.Option(help="write x,y CSV of the track")] = None,
    plot: PlotOpt = None,
) -> None:
    """GPS (NMEA, NEO-6M) log → laps, track, simulated vs measured speed (E2)."""
    from .gps import compare_trace, read_nmea, split_laps, track_from_lap

    path = resolve_path(log) if log else default_data_dir() / "gps" / "sample_DUMMY.nmea"
    if "DUMMY" in path.name:
        typer.echo("*** DUMMY GPS log (synthetic) ***")
    df = read_nmea(path)
    dt = float(np.median(np.diff(df["t_s"])))
    typer.echo(f"{len(df)} fixes, {1 / dt:.1f} Hz, {df['t_s'].iloc[-1]:.1f} s")
    if dt > 0.25:
        typer.echo(
            "WARNING: logging below 4 Hz under-samples hairpins (lap time too fast); "
            "set the receiver to 5–10 Hz (NEO-6M default is 1 Hz)"
        )
    laps = split_laps(df, gate_radius_m=gate)
    if not laps:
        raise typer.BadParameter("no complete lap found (check --gate)")
    typer.echo("laps: " + ", ".join(f"#{lp.index} {lp.lap_time_s:.3f} s" for lp in laps))
    sel = laps[min(lap, len(laps) - 1)]
    tr = track_from_lap(sel, method=method, course_sigma_deg=course_sigma, name=path.stem)
    veh = load_vehicle(vehicle)
    sim = simulate(veh, tr, SolverOptions(model=model))
    cmp = compare_trace(sel, tr, sim)
    typer.echo(pd.Series({**tr.summary(), **cmp.summary()}).round(3).to_string())
    if export_track:
        s = np.arange(0.0, tr.length_m, 0.5)
        x, y = tr.xy(s)
        _save_df(pd.DataFrame({"x": x, "y": y}), export_track)
    if plot:
        _save_fig(
            plots.measured_vs_sim(
                cmp.s_m, cmp.v_measured_mps, sim, f"GPS lap #{sel.index} vs simulation"
            ),
            plot,
        )


@app.command()
def assumptions(
    vehicle: VehicleOpt = None,
    events: EventsOpt = None,
) -> None:
    """List every parameter flagged ``assumed: true``."""
    from .report import ReportOptions, collect_assumptions

    df = collect_assumptions(ReportOptions(vehicle=vehicle, events=events))
    typer.echo(df.to_string(index=False))
    typer.echo(f"\n{len(df)} assumed parameters")


@app.command("init-configs")
def init_configs(dest: Annotated[Path, typer.Argument(help="target directory")]) -> None:
    """Copy the bundled example configs and data for editing."""
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copytree(default_config_dir(), dest / "configs", dirs_exist_ok=True)
    shutil.copytree(default_data_dir(), dest / "data", dirs_exist_ok=True)
    typer.echo(f"copied configs/ and data/ to {dest}")


@app.command()
def track(
    path: Annotated[str | None, typer.Option("--track", "-t", help="track YAML")] = None,
    generate: Annotated[str | None, typer.Option(help="autocross | endurance")] = None,
    seed: int = 0,
    length: Annotated[float, typer.Option(help="target length [m]")] = 900.0,
    plot: PlotOpt = None,
    export_csv: Annotated[Path | None, typer.Option(help="write x,y CSV")] = None,
) -> None:
    """Inspect, generate and export tracks (rule check included)."""
    from .trackgen import LayoutRules, check_layout, generate_layout, load_track

    if generate:
        tr = generate_layout(generate, seed, length)
    elif path:
        tr = load_track(path)
    else:
        raise typer.BadParameter("give --track or --generate")
    rep = check_layout(tr, LayoutRules())
    typer.echo(pd.Series(tr.summary()).round(3).to_string())
    typer.echo(f"rule issues: {rep['issues'] or 'none'}")
    if export_csv:
        s = np.arange(0.0, tr.length_m, 0.5)
        x, y = tr.xy(s)
        _save_df(pd.DataFrame({"x": x, "y": y}), export_csv)
    if plot:
        lap = simulate(load_vehicle(), tr)
        _save_fig(plots.track_map(tr, lap), plot)


@app.command()
def wing2d3d(
    cl: Annotated[float, typer.Option(help="2D Cl (downforce > 0)")],
    cd: Annotated[float, typer.Option(help="2D Cd")],
    chord: Annotated[float, typer.Option(help="reference chord [m]")],
    span: Annotated[float, typer.Option(help="span [m]")],
    ar_eff: Annotated[float | None, typer.Option(help="effective aspect ratio")] = None,
    k_lift: Annotated[float | None, typer.Option(help="Cl3D/Cl2D")] = None,
    e: Annotated[float, typer.Option(help="Oswald efficiency")] = 0.9,
) -> None:
    """Rough 2D → 3D wing estimate (ClA, CdA)."""
    from .wing import ROUGH_NOTE, wing_2d_to_3d

    w = wing_2d_to_3d(cl, cd, chord, span, ar_eff, k_lift, e)
    typer.echo(
        f"AR_eff {w.ar_eff:.3f}, k {w.k_lift:.3f}, Cl3D {w.cl_3d:.3f}, "
        f"Cd3D {w.cd_3d:.4f} (induced {w.cd_induced:.4f})"
    )
    typer.echo(f"ClA {w.cla_m2:.4f} m², CdA {w.cda_m2:.4f} m², L/D {w.cla_m2 / w.cda_m2:.2f}")
    typer.echo(ROUGH_NOTE)


def main() -> None:  # pragma: no cover - console entry
    """Console entry point."""
    app()


if __name__ == "__main__":  # pragma: no cover
    app()
