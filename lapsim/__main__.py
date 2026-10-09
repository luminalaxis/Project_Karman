"""Command line interface.

    python -m lapsim run      --track tracks/sample_autocross.json
    python -m lapsim run      --line mincurv --plot lap.png --export lap.csv
    python -m lapsim delta    --dcl 10 --dcd 5                     # every aero device
    python -m lapsim delta    --dev rear:cl=10,cd=5 --dev front:cl=-5
    python -m lapsim coeffs                                        # total + per device
    python -m lapsim forces   --speeds 40,60,80,100                # per-device downforce/drag [N]
    python -m lapsim targets  --bal-range 45:49 --save cars/target.json
    python -m lapsim sweep    --cl=-20:20:5 --cd=-20:20:5   (use "=" for negative starts)
    python -m lapsim balance  --range 30:60:2.5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import study, targets
from .channels import channels, export_csv
from .line import best_line, racing_line
from .solver import simulate
from .track import load_track
from .vehicle import load_vehicle, save_vehicle

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CAR = ROOT / "cars" / "fsk_ev_baseline.json"
DEFAULT_TRACK = ROOT / "tracks" / "fsk_2026_endurance.json"


def _range(text: str) -> np.ndarray:
    """'a:b:step' (inclusive) -> array."""
    a, b, step = (float(x) for x in text.split(":"))
    return np.round(np.arange(a, b + step / 2, step), 6)


def _set_overrides(veh, items):
    for item in items or []:
        key, value = item.split("=", 1)
        veh = veh.with_changes(**{key: float(value)})
    return veh


def _device_changes(items):
    """['rear:cl=10,cd=5', ...] -> {'rear': (0.10, 0.05)} (percent -> fraction)."""
    out = {}
    for item in items or []:
        name, _, spec = item.partition(":")
        vals = {"cl": 0.0, "cd": 0.0}
        for part in filter(None, spec.split(",")):
            k, v = part.split("=")
            if k.strip().lower() not in vals:
                raise SystemExit(f"--dev {item!r}: use cl=<%>,cd=<%>")
            vals[k.strip().lower()] = float(v) / 100
        out[name] = (vals["cl"], vals["cd"])
    return out


def _describe(changes, bal):
    parts = [f"{'all devices' if d is None else d}: ClA {cl * 100:+.1f}%, CdA {cd * 100:+.1f}%"
             for d, (cl, cd) in changes.items()]
    if bal is not None:
        parts.append(f"re-trim to {bal * 100:.1f}% F")
    return "; ".join(parts) or "none"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="lapsim", description="QSS lap time simulator (FSK aero studies)")
    ap.add_argument("command", choices=["run", "delta", "coeffs", "sweep", "balance",
                                                   "forces", "targets"])
    ap.add_argument("--car", default=DEFAULT_CAR)
    ap.add_argument("--track", default=DEFAULT_TRACK)
    ap.add_argument("--ds", type=float, default=0.5, help="track discretisation [m]")
    ap.add_argument("--smooth", type=float, default=4.0, help="centerline smoothing window [m]")
    ap.add_argument("--flying", action="store_true", help="open track: start at corner-limited speed")
    ap.add_argument("--set", action="append", metavar="KEY=VALUE",
                    help="override a car parameter, e.g. --set aero.rear.ClA=1.6 --set mass=270")
    ap.add_argument("--dcl", type=float, default=0.0, help="delta: ClA change of every device [%%]")
    ap.add_argument("--dcd", type=float, default=0.0, help="delta: CdA change of every device [%%]")
    ap.add_argument("--dev", action="append", metavar="NAME:cl=X,cd=Y",
                    help="delta: change one aero device [%%], repeatable, e.g. --dev rear:cl=10,cd=5")
    ap.add_argument("--device", action="append", metavar="NAME",
                    help="sweep/targets: only vary these devices (repeatable; default all)")
    ap.add_argument("--speeds", default="40,60,80,100", help="forces/targets: reference speeds [km/h]")
    ap.add_argument("--bal-range", metavar="LO:HI", help="targets: allowed front balance window [%%]")
    ap.add_argument("--save", metavar="JSON", help="targets: write the target car to this file")
    ap.add_argument("--bal", type=float, default=None, help="delta: new front aero balance [%%]")
    ap.add_argument("--cl", default="-20:20:5", help="sweep: ClA range a:b:step [%%]")
    ap.add_argument("--cd", default="-20:20:5", help="sweep: CdA range a:b:step [%%]")
    ap.add_argument("--range", default="30:60:2.5", help="balance: front balance range [%%]")
    ap.add_argument("--plot", metavar="PNG", help="save a plot")
    ap.add_argument("--export", metavar="CSV", help="run/delta: write all lap channels to CSV")
    ap.add_argument("--line", choices=["center", "mincurv", "compromise", "best"], default="center",
                    help="driven path: centerline, minimum-curvature line, curvature/length "
                         "compromise line, or best = whichever of the two lines is faster")
    ap.add_argument("--car-width", type=float, default=1.45, help="racing line: car width [m]")
    ap.add_argument("--line-margin", type=float, default=0.25, help="racing line: margin to limits [m]")
    ap.add_argument("--track-width", type=float, default=None,
                    help="racing line: track width [m] when the track file has none")
    args = ap.parse_args(argv)

    veh = _set_overrides(load_vehicle(args.car), args.set)
    track = load_track(args.track, ds=args.ds, smooth_m=args.smooth)
    limits = track
    if args.line == "best":
        track, times = best_line(veh, track, args.car_width, args.line_margin,
                                 default_width=args.track_width, smooth_m=args.smooth)
        print("racing lines: " + ", ".join(f"{m} {t:.3f} s" for m, t in times.items()))
    elif args.line != "center":
        track = racing_line(track, args.line, args.car_width, args.line_margin,
                            default_width=args.track_width, smooth_m=args.smooth)
    kw = {"v_start": 1e3} if args.flying else {}
    print(f"car   : {veh.name}  (m {veh.mass:.0f} kg)")
    print(veh.aero.table())
    print(f"track : {track.name}  ({track.length:.1f} m, {'closed' if track.closed else 'open'})\n")

    if args.command == "run":
        res = simulate(veh, track, **kw)
        print(res.summary())
        if args.plot:
            from .plot import plot_lap
            plot_lap({"baseline": (veh, res)}, args.plot, limits)
        if args.export:
            export_csv(channels(veh, res), args.export)
            print(f"\nchannels written to {args.export}")

    elif args.command == "delta":
        bal = None if args.bal is None else args.bal / 100
        changes = {None: (args.dcl / 100, args.dcd / 100)} if (args.dcl or args.dcd) else {}
        changes.update(_device_changes(args.dev))
        mod = study.modify(veh, changes, bal)
        base, new = simulate(veh, track, **kw), simulate(mod, track, **kw)
        d = new.lap_time - base.lap_time
        print(f"change    : {_describe(changes, bal)}")
        print(f"aero      : ClA {veh.aero.ClA:.2f} -> {mod.aero.ClA:.2f}, CdA {veh.aero.CdA:.2f} -> "
              f"{mod.aero.CdA:.2f}, balance {veh.aero.balance_front * 100:.1f} -> "
              f"{mod.aero.balance_front * 100:.1f}% F")
        print(f"baseline  : {base.lap_time:.3f} s")
        print(f"modified  : {new.lap_time:.3f} s")
        print(f"delta     : {d * 1e3:+.1f} ms  ({'faster' if d < 0 else 'slower'})")
        if args.plot:
            from .plot import plot_lap
            plot_lap({"baseline": (veh, base), "modified": (mod, new)}, args.plot, limits)
        if args.export:
            export_csv(channels(mod, new), args.export)
            print(f"\nmodified-car channels written to {args.export}")

    elif args.command == "coeffs":
        print(study.coefficients(veh, track, **kw).summary(veh))
        print("\nper device (absolute change on that device only; balance shifts with it):")
        print(study.device_table(study.device_coefficients(veh, track, **kw)))

    elif args.command == "sweep":
        d_cl, d_cd = _range(args.cl) / 100, _range(args.cd) / 100
        grid = study.sweep(veh, track, d_cl, d_cd, args.device, **kw)
        print(f"Δ lap time [ms]   devices: {', '.join(args.device or ['all'])}   "
              "rows: ΔCdA %, columns: ΔClA %")
        print("CdA\\ClA " + "".join(f"{c * 100:+8.1f}" for c in d_cl))
        for cd, row in zip(d_cd, grid):
            print(f"{cd * 100:+7.1f} " + "".join(f"{x * 1e3:+8.1f}" for x in row))
        if args.plot:
            from .plot import plot_sweep
            plot_sweep(d_cl, d_cd, grid, args.plot)

    elif args.command == "balance":
        bals = _range(args.range) / 100
        times = study.balance_sweep(veh, track, bals, **kw)
        best = int(np.nanargmin(times))
        print(f"re-trim front/rear devices at constant ClA {veh.aero.ClA:.2f} "
              f"(current {veh.aero.balance_front * 100:.1f}% F)")
        for b, t in zip(bals, times):
            if np.isnan(t):
                print(f"balance {b * 100:5.1f}% F : not reachable")
                continue
            mark = "  <- best" if b == bals[best] else ""
            print(f"balance {b * 100:5.1f}% F : {t:.3f} s ({(t - times[best]) * 1e3:+.1f} ms){mark}")
        if args.plot:
            from .plot import plot_balance
            plot_balance(bals, times, args.plot)

    elif args.command == "forces":
        res = simulate(veh, track, **kw)
        speeds = [float(s) for s in args.speeds.split(",")]
        print(f"lap time {res.lap_time:.3f} s — per-device aero loads "
              "(DF = downforce, drag E = drag work per lap)")
        print(targets.loads_table(targets.device_loads(veh, res, speeds), veh))

    elif args.command == "targets":
        bal = None
        if args.bal_range:
            lo, hi = (float(x) / 100 for x in args.bal_range.split(":"))
            bal = (lo, hi)
        print("optimising device ClA along each device polar (CdA = CdA0 + k*ClA^2) ...")
        tr = targets.optimise_targets(veh, track, args.device, bal, **kw)
        print(tr.summary())
        speeds = [float(s) for s in args.speeds.split(",")]
        print("\ntarget loads:")
        print(targets.loads_table(targets.device_loads(tr.target, simulate(tr.target, track, **kw),
                                                       speeds), tr.target))
        if args.save:
            tr.target.name = f"{veh.name} (aero targets)"
            save_vehicle(tr.target, args.save)
            print(f"\ntarget car saved to {args.save}")

    if args.plot:
        print(f"\nplot saved to {args.plot}")


if __name__ == "__main__":
    main()
