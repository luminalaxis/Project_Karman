"""Lap result analysis: what limits the lap, where the powertrain tops out, whether the energy lasts -
for one run or several side by side (vehicles x tracks x parameter variants).

``analyze(car, res)`` -> :class:`LapReport` (metrics + plain-language findings);
``compare_table(reports)`` -> one table, a column per run and the change against the first run.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from ..solvers.base import LapResult
from ..vehicle import VehicleModel

# (key, label, format) in table order
ROWS = (
    ("lap_time_s", "lap time [s]", "{:.2f}"),
    ("v_avg_kmh", "v avg [km/h]", "{:.1f}"),
    ("v_max_kmh", "v max [km/h]", "{:.1f}"),
    ("ay_max_g", "ay max [g]", "{:.2f}"),
    ("ax_min_g", "braking max [g]", "{:.2f}"),
    ("lim_braking", "limit: braking [%]", "{:.1f}"),
    ("lim_front", "limit: front axle grip [%]", "{:.1f}"),
    ("lim_rear", "limit: rear axle grip [%]", "{:.1f}"),
    ("lim_power", "limit: power [%]", "{:.1f}"),
    ("understeer_front_lim", "understeer idx (front-limited)", "{:.2f}"),
    ("motor_peak_kW", "motor peak [kW]", "{:.1f}"),
    ("drive_ceiling_kW", "drive ceiling at wheels [kW]", "{:.1f}"),
    ("power_limit_kW", "power limit setting [kW]", "{:.1f}"),
    ("battery_kWh_lap", "battery per lap [kWh]", "{:.3f}"),
    ("endurance_need_kWh", "endurance need [kWh]", "{:.2f}"),
    ("usable_kWh", "usable energy [kWh]", "{:.2f}"),
    ("energy_margin_pct", "energy margin [%]", "{:+.0f}"),
)


@dataclass
class LapReport:
    label: str
    metrics: dict[str, float]
    limits: dict[str, tuple[float, float]]  # limiter -> (share of distance %, mean speed km/h)
    findings: list[str] = field(default_factory=list)

    def text(self) -> str:
        lines = [f"[{self.label}]", "limiting factor (share of distance, mean speed):"]
        for name, (share, v) in sorted(self.limits.items(), key=lambda kv: -kv[1][0]):
            lines.append(f"  {name:22s} {share:5.1f} %   {v:5.1f} km/h")
        lines.append("")
        lines += [f"{label:32s} {_fmt(fmt, self.metrics.get(key))}" for key, label, fmt in ROWS]
        if self.findings:
            lines += ["", "findings:"] + [f"  - {f}" for f in self.findings]
        return "\n".join(lines)


def _fmt(fmt: str, x) -> str:
    return "-" if x is None or not np.isfinite(x) else fmt.format(x)


def analyze(car: VehicleModel, res: LapResult, label: str = "", endurance_km: float = 22.0) -> LapReport:
    m = dict(res.summary())
    ch = res.channels
    ds = np.diff(res.s)
    lim = np.asarray(ch["limit"][:-1])
    v = res.v[:-1] * 3.6
    limits = {}
    for name in sorted(set(lim)):
        sel = lim == name
        limits[str(name)] = (float(ds[sel].sum() / ds.sum() * 100.0), float(np.average(v[sel], weights=ds[sel])))
    share = {k: s for k, (s, _) in limits.items()}
    m["lim_braking"] = sum(s for k, s in share.items() if k.startswith("braking"))
    m["lim_front"] = share.get("front axle", 0.0)
    m["lim_rear"] = share.get("rear axle", 0.0)
    m["lim_power"] = share.get("power", 0.0)
    sel = lim == "front axle"
    us = ch.get("understeer_index")
    m["understeer_front_lim"] = float(np.nanmean(us[:-1][sel])) if us is not None and sel.any() else float("nan")

    pt = car.powertrain
    vs = np.linspace(5.0, max(float(res.v.max()), 6.0), 60)
    m["drive_ceiling_kW"] = float(np.max(np.asarray(pt.max_force(vs)[0]) * vs) / 1000.0)
    m["power_limit_kW"] = pt.power_limit / 1000.0 if pt.power_limit is not None else float("nan")

    laps = max(1, round(endurance_km * 1000.0 / float(res.s[-1])))
    per_lap = m.get("battery_mean_kW", float("nan")) * res.lap_time / 3600.0
    usable = pt.usable_energy if pt.type == "ev" else None
    m["battery_kWh_lap"] = per_lap
    m["endurance_need_kWh"] = per_lap * laps
    m["usable_kWh"] = usable / 3.6e6 if usable else float("nan")
    m["energy_margin_pct"] = (m["usable_kWh"] / m["endurance_need_kWh"] - 1.0) * 100.0 if usable else float("nan")
    m["endurance_laps"] = laps

    rep = LapReport(label or f"{res.vehicle_name} | {res.track_name}", m, limits)
    rep.findings = _findings(rep, pt, endurance_km)
    return rep


def _findings(rep: LapReport, pt, endurance_km: float) -> list[str]:
    m, out = rep.metrics, []
    grip = m["lim_front"] + m["lim_rear"]
    if m["lim_braking"] + grip > 60.0:
        out.append(f"그립이 랩을 정함 (브레이킹 {m['lim_braking']:.0f} % + 코너/탈출 {grip:.0f} %): "
                   "타이어·다운포스·무게중심이 파워보다 효과 큼")
    bf = rep.limits.get("braking front axle", (0.0, 0.0))[0]
    br = rep.limits.get("braking rear axle", (0.0, 0.0))[0]
    front, rear = m["lim_front"] + bf, m["lim_rear"] + br
    for name, a, b, other, hint in (("앞", m["lim_front"], bf, rear, "앞 다운포스·앞 그립(캠버, 롤 강성 배분)"),
                                    ("뒤", m["lim_rear"], br, front, "탈출 트랙션(LSD, 뒤 다운포스)·감속 시 뒤 하중")):
        if a + b > 1.5 * other and a + b > 20.0:
            out.append(f"{name} 차축이 주 한계 {a + b:.0f} % (코너/가속 {a:.0f} + 브레이킹 {b:.0f}) vs 반대 "
                       f"{other:.0f} %: {hint}")
    plim = m["power_limit_kW"]
    if np.isfinite(plim) and m["drive_ceiling_kW"] < 0.85 * plim * pt.efficiency:
        out.append(f"구동 출력 상한 {m['drive_ceiling_kW']:.0f} kW < 설정 {plim:.0f} kW: 모터 토크 곡선 / 팩 전압 "
                   "(약계자)이나 배터리 전류가 먼저 막음 - power_limit_kW 올려도 효과 없음")
    if m["lim_power"] > 25.0:
        out.append(f"출력 제한 구간 {m['lim_power']:.0f} %: 기어비·출력 상한을 볼 만함")
    if np.isfinite(m["energy_margin_pct"]):
        word = "부족" if m["energy_margin_pct"] < 0 else "여유"
        out.append(f"엔듀런스 {endurance_km:g} km ({m['endurance_laps']} 랩) 이 페이스로 {m['endurance_need_kWh']:.2f} kWh "
                   f"/ 사용 가능 {m['usable_kWh']:.2f} kWh -> {word} {abs(m['energy_margin_pct']):.0f} % "
                   "(전 랩 최대 페이스 기준; 'aerolap energy'로 정밀 계산)")
    if pt.type == "ev" and not pt.has_regen and m["lim_braking"] > 25.0:
        out.append(f"회생제동 꺼짐인데 브레이킹 구간 {m['lim_braking']:.0f} %: battery.regen_power_kW 확인")
    return out


def compare_table(reports: list[LapReport]) -> str:
    """Runs side by side; from the second run on each cell also shows the change against the first."""
    labels = [r.label for r in reports]
    names = [f"#{i + 1}" for i in range(len(reports))]
    w = 16
    head = f"{'':32s}" + "".join(f"{n:>{w}s}" for n in names)
    lines = [f"{n} = {lab}" for n, lab in zip(names, labels)] + ["", head]
    base = reports[0].metrics
    for key, label, fmt in ROWS:
        cells = []
        for k, r in enumerate(reports):
            x = r.metrics.get(key)
            cell = _fmt(fmt, x)
            if k and x is not None and np.isfinite(x) and np.isfinite(base.get(key, np.nan)):
                d = x - base[key]
                cell += f" ({'+' if d >= 0 else '-'}{_fmt(fmt.replace('+', ''), abs(d))})"
            cells.append(f"{cell:>{w}s}")
        lines.append(f"{label:32s}" + "".join(cells))
    for n, r in zip(names, reports):
        if r.findings:
            lines += ["", f"{n} findings:"] + [f"  - {f}" for f in r.findings]
    return "\n".join(lines)


# Records: <excel_data_analysis>/시뮬레이션_기록/<kind>/<run time>/ holds each run's files (기록.xlsx = one sheet,
# trace csv, png). The comparison workbook 시뮬레이션_결과.xlsx keeps only "비교" (every lap run ranked per track;
# the run time links to that run's 기록.xlsx) and "최고기록" (best lap per track, read by the aero design sheet with
# INDEX/MATCH - keep its column order stable; new columns go at the end).
EXCEL_LOG = Path(__file__).resolve().parents[3] / "excel_data_analysis" / "시뮬레이션_결과.xlsx"
RUN_BOOK = "기록.xlsx"
HEAD = ["실행 시간", "차량", "트랙", "모델"]
INPUTS = (("mass_kg", "질량 [kg]"), ("cla_avg", "ClA 평균 [m²]"), ("cda_avg", "CdA 평균 [m²]"),
          ("cop_front_avg", "공력 밸런스 전륜 평균 [-]"))
LAP_COL = len(HEAD) + len(INPUTS)  # 0-based column of the lap time in a run row
COLUMNS = (HEAD + [lab for _, lab in INPUTS] + [lab for _, lab, _ in ROWS]
           + ["한계 요인 (거리 %)", "findings", "기록 파일"])


def records_root() -> Path:
    return EXCEL_LOG.parent / "시뮬레이션_기록"


def run_dir(kind: str) -> Path:
    """New folder <records>/<kind>/<run time> for one run."""
    p = records_root() / kind / f"{datetime.now():%Y-%m-%d_%H-%M-%S}"
    while p.exists():  # two runs in the same second
        p = p.with_name(p.name + "_")
    p.mkdir(parents=True)
    return p


def write_run_book(folder: Path, rows: list[list], trace: Path | None = None) -> Path | None:
    """folder/기록.xlsx with one sheet: ``rows`` (the report), then the trace table (default: the folder's
    trace csv). Returns the path, or None if openpyxl is missing."""
    try:
        import openpyxl
    except ImportError:
        print("excel record skipped: pip install openpyxl")
        return None
    import csv

    folder = Path(folder)
    if trace is None:
        found = sorted(folder.glob("trace*.csv"), key=lambda f: (f.name != "trace.csv", f.name))
        trace = found[0] if found else None
    wb = openpyxl.Workbook(write_only=True)
    ws = wb.create_sheet("기록")
    ws.column_dimensions["A"].width = 34
    for r in rows:
        ws.append(r)
    if trace is not None and trace.exists():
        ws.append([])
        ws.append([f"주행 기록 ({trace.name}, 거리 간격마다 한 행)"])
        with open(trace, newline="", encoding="utf-8") as fh:
            for i, r in enumerate(csv.reader(fh)):
                ws.append(r if i == 0 else [_cell(x) for x in r])
    path = folder / RUN_BOOK
    wb.save(path)
    return path


def text_rows(title: str, text: str) -> list[list]:
    """A text report as sheet rows (one line per row)."""
    return [[title], [f"실행 시간 {datetime.now():%Y-%m-%d %H:%M:%S}"], []] + [[ln] for ln in text.splitlines()]


def _cell(x: str):
    try:
        return float(x)
    except ValueError:
        return x


def run_inputs(car: VehicleModel, res: LapResult) -> dict[str, float]:
    """Setup values worth comparing between runs; aero as the distance-weighted lap mean."""
    ds = np.diff(res.s)
    out = {"mass_kg": car.m}
    for key, ch in (("cla_avg", "cla"), ("cda_avg", "cda"), ("cop_front_avg", "cop_front")):
        x = res.channels.get(ch)
        out[key] = float(np.average(np.asarray(x[:-1], float), weights=ds)) if x is not None else float("nan")
    return out


def log_to_excel(rep: LapReport, res: LapResult, level: str = "", path: Path | None = None,
                 inputs: dict[str, float] | None = None, folder: Path | None = None) -> Path | None:
    """Write this run's 기록.xlsx into ``folder`` (with the folder's trace csv) and add the run to the
    comparison workbook. Returns the comparison workbook path, or None if it could not be written."""
    path = Path(path or EXCEL_LOG)
    try:
        import openpyxl
    except ImportError:
        print("excel log skipped: pip install openpyxl")
        return None

    inputs = inputs or {}
    when = f"{datetime.now():%Y-%m-%d %H:%M:%S}"
    book = None
    if folder is not None:
        rows = [[f"실행 시간 {when}"], ["차량", res.vehicle_name], ["트랙", res.track_name],
                ["모델", level or res.level], [], ["항목", "값"]]
        rows += [[label, _num(inputs.get(key))] for key, label in INPUTS]
        rows += [[label, _num(rep.metrics.get(key))] for key, label, _ in ROWS]
        rows += [[], ["한계 요인", "거리 비율 [%]", "평균 속도 [km/h]"]]
        rows += [[k, round(s, 2), round(v, 2)] for k, (s, v) in sorted(rep.limits.items(), key=lambda kv: -kv[1][0])]
        rows += [[], ["요약 값"]] + [[k, _num(v)] for k, v in res.summary().items()]
        if rep.findings:
            rows += [[], ["findings"]] + [[f] for f in rep.findings]
        book = write_run_book(folder, rows)

    wb = openpyxl.load_workbook(path) if path.exists() else openpyxl.Workbook()
    n = len(COLUMNS)
    runs = ([(list(r[2:]) + [None] * n)[:n] for r in wb["비교"].iter_rows(min_row=2, values_only=True)]
            if "비교" in wb.sheetnames else [])
    limits = ", ".join(f"{k} {s:.0f}" for k, (s, _) in sorted(rep.limits.items(), key=lambda kv: -kv[1][0]))
    rel = Path(os.path.relpath(book, path.parent)).as_posix() if book else None
    runs.append([when, res.vehicle_name, res.track_name, level or res.level]
                + [_num(inputs.get(k)) for k, _ in INPUTS] + [_num(rep.metrics.get(k)) for k, _, _ in ROWS]
                + [limits, " / ".join(rep.findings), rel])
    for name in list(wb.sheetnames):  # the workbook holds only the comparison
        del wb[name]
    _write_sheets(wb, runs)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        wb.save(path)
    except PermissionError:  # open in Excel
        print(f"excel log skipped: close {path.name} in Excel and run again")
        return None
    return path


def _write_sheets(wb, runs: list[list]) -> None:
    """Build 비교 and 최고기록 (the run time links to the run's own 기록.xlsx)."""
    from openpyxl.styles import Font, PatternFill

    runs = [r for r in runs if r[LAP_COL] is not None]
    runs.sort(key=lambda r: (r[2], r[LAP_COL]))
    best = {}
    for r in runs:
        best.setdefault(r[2], r)
    cmp_ = wb.create_sheet("비교", 0)
    cmp_.append(["순위", "최고 대비 [s]"] + COLUMNS)
    green = PatternFill("solid", fgColor="C6EFCE")
    rank, last = 0, None
    for r in runs:
        rank = 1 if r[2] != last else rank + 1
        last = r[2]
        cmp_.append([rank, round(r[LAP_COL] - best[r[2]][LAP_COL], 4)] + r)
        if rank == 1:
            for c in cmp_[cmp_.max_row]:
                c.fill = green
        _link(cmp_.cell(cmp_.max_row, 3), r[-1])
    top = wb.create_sheet("최고기록", 1)
    top.append(["트랙"] + COLUMNS)  # column A = track for MATCH
    for track, r in best.items():
        top.append([track] + r)
        _link(top.cell(top.max_row, 2), r[-1])
    for ws in (cmp_, top):
        for c in ws[1]:
            c.font = Font(bold=True)
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        for col, w in (("A", 12), ("B", 20), ("C", 20), ("D", 30), ("E", 40)):
            ws.column_dimensions[col].width = w
    wb.active = 0


def _link(cell, target) -> None:
    """Run time cell -> hyperlink to that run's 기록.xlsx (relative to the comparison workbook)."""
    from openpyxl.styles import Font

    if target:
        cell.hyperlink = str(target)
        cell.font = Font(color="0563C1", underline="single")


def _num(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 4)
