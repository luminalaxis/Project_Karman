"""Event times -> competition points (ported from fsk-lapsim E4).

The formulas are **placeholders** shaped like the usual Formula Student "time ratio" scoring; replace
them with the FSK rule book (``--scoring FILE.yaml``, same keys as ``PLACEHOLDER``). Reference best
times are a ratio of the BASE car's time, so the useful output is the *points difference* between
design variants scored against the same references.

time_ratio:        T_max = tmax_factor * T_min
                   points = a * ((T_max/T)^p - 1) / ((T_max/T_min)^p - 1) + b   (b if T >= T_max)
efficiency_factor: EF = (T_ref/T) * (E_ref/E), points = a * clip((EF - ef_min) / (ef_max - ef_min), 0, 1)
"""

from __future__ import annotations

PLACEHOLDER = {
    "acceleration": {"formula": "time_ratio", "a": 71.5, "b": 3.5, "exponent": 1, "tmax_factor": 1.5, "t_min_ratio": 0.90},
    "skidpad": {"formula": "time_ratio", "a": 71.5, "b": 3.5, "exponent": 2, "tmax_factor": 1.25, "t_min_ratio": 0.95},
    "autocross": {"formula": "time_ratio", "a": 95.5, "b": 4.5, "exponent": 1, "tmax_factor": 1.25, "t_min_ratio": 0.95},
    "endurance": {"formula": "time_ratio", "a": 300.0, "b": 25.0, "exponent": 1, "tmax_factor": 1.333, "t_min_ratio": 0.95},
    "efficiency": {"formula": "efficiency_factor", "a": 100.0, "t_ref_ratio": 0.95, "e_ref_ratio": 0.85,
                   "ef_min": 0.5, "ef_max": 1.0},
}


def references(base_times: dict[str, float], base_energy_kWh: float | None, formulas: dict = PLACEHOLDER) -> dict:
    """Reference best times / efficiency references from the base car (``t_min_s`` overrides the ratio)."""
    refs = {}
    for name, f in formulas.items():
        if f["formula"] == "time_ratio" and name in base_times:
            refs[name] = f.get("t_min_s") or f["t_min_ratio"] * base_times[name]
        elif f["formula"] == "efficiency_factor" and "endurance" in base_times and base_energy_kWh:
            refs[name] = (f["t_ref_ratio"] * base_times["endurance"], f["e_ref_ratio"] * base_energy_kWh)
    return refs


def points(times: dict[str, float], energy_kWh: float | None, refs: dict, formulas: dict = PLACEHOLDER) -> dict:
    """Points per event plus ``total``."""
    out = {}
    for name, f in formulas.items():
        if name not in refs:
            continue
        if f["formula"] == "time_ratio" and name in times:
            t_min = refs[name]
            t_max = f["tmax_factor"] * t_min
            t, p = max(times[name], t_min), f["exponent"]
            out[name] = f["b"] if t >= t_max else f["a"] * ((t_max / t) ** p - 1) / ((t_max / t_min) ** p - 1) + f["b"]
        elif f["formula"] == "efficiency_factor" and energy_kWh:
            t_ref, e_ref = refs[name]
            ef = (t_ref / times["endurance"]) * (e_ref / energy_kWh)
            out[name] = f["a"] * min(max((ef - f["ef_min"]) / (f["ef_max"] - f["ef_min"]), 0.0), 1.0)
    out["total"] = sum(out.values())
    return out


def points_table(rows: list[tuple[str, dict[str, float], float | None]], formulas: dict = PLACEHOLDER,
                 placeholder: bool = True) -> str:
    """rows = [(name, event times, endurance energy kWh)]; the first row is the base (references)."""
    refs = references(rows[0][1], rows[0][2], formulas)
    scored = [(n, points(t, e, refs, formulas)) for n, t, e in rows]
    keys = list(scored[0][1])
    lines = [f"{'car':28s}" + "".join(f"{k:>13s}" for k in keys)]
    for n, sc in scored:
        lines.append(f"{n[:28]:28s}" + "".join(f"{sc[k]:13.2f}" for k in keys))
    lines += ["", f"points vs {rows[0][0]}:"]
    for n, sc in scored[1:]:
        lines.append(f"{n[:28]:28s}" + "".join(f"{sc[k] - scored[0][1][k]:+13.2f}" for k in keys))
    if placeholder:
        lines += ["", "NOTE: placeholder formulas (FS-style time ratio, references = ratio of the base car's "
                      "times). Use --scoring with the FSK rule-book values before quoting absolute points."]
    return "\n".join(lines)

