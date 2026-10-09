"""Command line interface.

    python -m lapsim run      --track tracks/sample_autocross.json
    python -m lapsim delta    --dcl 10 --dcd 5
    python -m lapsim coeffs
    python -m lapsim sweep    --cl=-20:20:5 --cd=-20:20:5   (use "=" for negative starts)
    python -m lapsim balance  --range 30:60:2.5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import study
from .solver import simulate
from .track import load_track
from .vehicle import load_vehicle

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CAR = ROOT / "cars" / "fsk_ev_baseline.json"
DEFAULT_TRACK = ROOT / "tracks" / "sample_autocross.json"


def _range(text: str) -> np.ndarray:
    """'a:b:step' (inclusive) -> array."""
    a, b, step = (float(x) for x in text.split(":"))
    return np.round(np.arange(a, b + step / 2, step), 6)


def _set_overrides(veh, items):
    for item in items or []:
        key, value = item.split("=", 1)
        veh = veh.with_changes(**{key: float(value)})
    return veh


def main(argv=None):
    ap = argparse.ArgumentParser(prog="lapsim", description="QSS lap time simulator (FSK aero studies)")
    ap.add_argument("command", choices=["run", "delta", "coeffs", "sweep", "balance"])
    ap.add_argument("--car", default=DEFAULT_CAR)
    ap.add_argument("--track", default=DEFAULT_TRACK)
    ap.add_argument("--ds", type=float, default=0.5, help="track discretisation [m]")
    ap.add_argument("--smooth", type=float, default=4.0, help="centerline smoothing window [m]")
    ap.add_argument("--flying", action="store_true", help="open track: start at corner-limited speed")
    ap.add_argument("--set", action="append", metavar="KEY=VALUE",
                    help="override a car parameter, e.g. --set aero.ClA=4.2 --set mass=270")
    ap.add_argument("--dcl", type=float, default=0.0, help="delta: ClA change [%%]")
    ap.add_argument("--dcd", type=float, default=0.0, help="delta: CdA change [%%]")
    ap.add_argument("--bal", type=float, default=None, help="delta: new front aero balance [%%]")
    ap.add_argument("--cl", default="-20:20:5", help="sweep: ClA range a:b:step [%%]")
    ap.add_argument("--cd", default="-20:20:5", help="sweep: CdA range a:b:step [%%]")
    ap.add_argument("--range", default="30:60:2.5", help="balance: front balance range [%%]")
    ap.add_argument("--plot", metavar="PNG", help="save a plot")
    args = ap.parse_args(argv)

    veh = _set_overrides(load_vehicle(args.car), args.set)
    track = load_track(args.track, ds=args.ds, smooth_m=args.smooth)
    kw = {"v_start": 1e3} if args.flying else {}
    print(f"car   : {veh.name}  (m {veh.mass:.0f} kg, ClA {veh.aero.ClA:.2f}, CdA {veh.aero.CdA:.2f}, "
          f"balance {veh.aero.balance_front * 100:.1f}% F)")
    print(f"track : {track.name}  ({track.length:.1f} m, {'closed' if track.closed else 'open'})\n")

    if args.command == "run":
        res = simulate(veh, track, **kw)
        print(res.summary())
        if args.plot:
            from .plot import plot_lap
            plot_lap({"baseline": res}, track, args.plot)

    elif args.command == "delta":
        bal = None if args.bal is None else args.bal / 100
        base, new, d = study.delta(veh, track, args.dcl / 100, args.dcd / 100, bal, **kw)
        print(f"change    : ClA {args.dcl:+.1f}%, CdA {args.dcd:+.1f}%"
              + ("" if bal is None else f", balance {args.bal:.1f}% F"))
        print(f"baseline  : {base:.3f} s")
        print(f"modified  : {new:.3f} s")
        print(f"delta     : {d * 1e3:+.1f} ms  ({'faster' if d < 0 else 'slower'})")
        if args.plot:
            from .plot import plot_lap
            mod = veh.scaled_aero(1 + args.dcl / 100, 1 + args.dcd / 100)
            if bal is not None:
                mod = mod.with_changes(**{"aero.balance_front": bal})
            plot_lap({"baseline": simulate(veh, track, **kw), "modified": simulate(mod, track, **kw)},
                     track, args.plot)

    elif args.command == "coeffs":
        print(study.coefficients(veh, track, **kw).summary(veh))

    elif args.command == "sweep":
        d_cl, d_cd = _range(args.cl) / 100, _range(args.cd) / 100
        grid = study.sweep(veh, track, d_cl, d_cd, **kw)
        print("Δ lap time [ms]   rows: ΔCdA %, columns: ΔClA %")
        print("CdA\\ClA " + "".join(f"{c * 100:+8.1f}" for c in d_cl))
        for cd, row in zip(d_cd, grid):
            print(f"{cd * 100:+7.1f} " + "".join(f"{x * 1e3:+8.1f}" for x in row))
        if args.plot:
            from .plot import plot_sweep
            plot_sweep(d_cl, d_cd, grid, args.plot)

    elif args.command == "balance":
        bals = _range(args.range) / 100
        times = study.balance_sweep(veh, track, bals, **kw)
        best = int(np.argmin(times))
        for b, t in zip(bals, times):
            mark = "  <- best" if b == bals[best] else ""
            print(f"balance {b * 100:5.1f}% F : {t:.3f} s ({(t - times[best]) * 1e3:+.1f} ms){mark}")
        if args.plot:
            from .plot import plot_balance
            plot_balance(bals, times, args.plot)

    if args.plot:
        print(f"\nplot saved to {args.plot}")


if __name__ == "__main__":
    main()
