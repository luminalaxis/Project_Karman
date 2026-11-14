"""Read an OptimumLap result (.OLRes) and compare its channels with an AeroLap trace.

    python scripts/olres_compare.py RESULT.OLRes out/ol_check/flying/trace.csv [--plot out.png]
"""
import re
import struct
import sys
from pathlib import Path

import numpy as np


def read_olres(path):
    """Channels {name: array} (SI units) and scalar results {name: value} of an OptimumLap .OLRes."""
    b = Path(path).read_bytes()
    ch, sc = {}, {}
    # channel header: <len>name <len>name <len>Label <len>Unit, then 01 32 01 <int32 n> <n doubles>
    for m in re.finditer(rb"([a-zA-Z][A-Za-z0-9]{2,})[\x00-\x7f]\1[\x00-\x7f]([A-Z][ -~]*?)[\x00-\x7f]([A-Za-z]+)\x01([\x31-\x36])\x01", b):
        kind, end = m.group(4), m.end()
        name = m.group(1).decode()
        if kind == b"2":  # array channel
            n = struct.unpack_from("<i", b, end)[0]
            if 0 < n < 10**6 and end + 4 + 8 * n <= len(b):
                ch[name] = np.frombuffer(b, "<f8", n, end + 4).copy()
        else:  # scalar
            sc[name] = struct.unpack_from("<d", b, end)[0]
    return ch, sc


if __name__ == "__main__":
    ch, sc = read_olres(sys.argv[1])
    print("OptimumLap lap time", sc.get("lapTime"), "s; channels:", ", ".join(f"{k}[{len(v)}]" for k, v in ch.items()))
    d = np.genfromtxt(sys.argv[2], delimiter=",", names=True)
    s_ol, v_ol = ch["elapsedDistance"], ch["speed"] * 3.6
    v_al = np.interp(s_ol, d["s_m"], d["v_kmh"])
    dt_ol = np.diff(ch["elapsedTime"])
    print(f"v max OL {v_ol.max():.1f}  AL {d['v_kmh'].max():.1f} km/h | v min OL {v_ol.min():.1f}  AL {d['v_kmh'].min():.1f}")
    print(f"mean |dv| {np.mean(np.abs(v_al - v_ol)):.2f} km/h, AL faster on {np.mean(v_al > v_ol):.0%} of the distance")
    for key in ("downForce", "dragForce", "power", "lateralAcceleration", "longitudinalAcceleration"):
        if key in ch:
            print(f"{key:26s} OL min {ch[key].min():9.2f} max {ch[key].max():9.2f}")
    if "--plot" in sys.argv:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (a, b2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
        a.plot(s_ol, v_ol, label=f"OptimumLap ({sc.get('lapTime', 0):.2f} s)")
        a.plot(d["s_m"], d["v_kmh"], label=f"AeroLap ({d['t_s'][-1]:.2f} s)")
        a.set_ylabel("speed [km/h]")
        a.legend()
        b2.plot(s_ol, v_al - v_ol)
        b2.axhline(0, color="k", lw=0.8)
        b2.set_ylabel("AeroLap - OptimumLap [km/h]")
        b2.set_xlabel("distance [m]")
        fig.tight_layout()
        fig.savefig(sys.argv[sys.argv.index("--plot") + 1], dpi=130)
