"""Lap analysis report and side-by-side comparison."""

from pathlib import Path

import pytest

from aerolap.analysis.lap_report import analyze, compare_table
from aerolap.config import load_vehicle_config, with_overrides
from aerolap.solvers.ggv import SolverOptions
from aerolap.solvers.qss import QSSSolver
from aerolap.track import Track
from aerolap.vehicle import VehicleModel

ROOT = Path(__file__).resolve().parents[1]
CAR = ROOT / "data" / "vehicles" / "fsk2026.yaml"
TRACK = ROOT / "data" / "tracks" / "fsk2026_endurance_smoothed.csv"


def _report(cfg, label):
    car = VehicleModel(cfg)
    res = QSSSolver(SolverOptions.coarse()).solve(car, Track.from_file(TRACK), start="flying")
    return analyze(car, res, label)


def test_report_and_comparison():
    cfg = load_vehicle_config(CAR)
    a, b = _report(cfg, "base"), _report(with_overrides(cfg, {"tires.mu_y": 1.7}), "grip")
    m = a.metrics
    assert sum(s for s, _ in a.limits.values()) == pytest.approx(100.0, abs=0.1)
    assert m["lim_braking"] + m["lim_front"] + m["lim_rear"] + m["lim_power"] == pytest.approx(100.0, abs=0.1)
    assert 0 < m["drive_ceiling_kW"] < m["power_limit_kW"]  # 70S pack: the motor tops out below the setting
    assert m["endurance_need_kWh"] == pytest.approx(m["battery_kWh_lap"] * m["endurance_laps"])
    assert any("구동 출력 상한" in f for f in a.findings)
    assert b.metrics["lap_time_s"] < m["lap_time_s"]
    table = compare_table([a, b])
    assert "#1 = base" in table and "#2 = grip" in table and "(-" in table.split("lap time")[1].splitlines()[0]


def test_run_folder_record_and_comparison_workbook(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    from aerolap.analysis.lap_report import LAP_COL, RUN_BOOK, log_to_excel, records_root, run_dir, run_inputs

    car = VehicleModel(load_vehicle_config(CAR))
    res = QSSSolver().solve(car, Track.from_file(ROOT / "data" / "tracks" / "fsae_accel.csv"), start="standing")
    rep = analyze(car, res)
    p = tmp_path / "r.xlsx"
    d1, d2 = run_dir("run"), run_dir("run")
    assert d1 != d2 and d1.parent == records_root() / "run"  # <records>/<kind>/<run time>
    res.to_csv(d1 / "trace.csv")
    log_to_excel(rep, res, path=p, inputs=run_inputs(car, res), folder=d1)
    rep.metrics["lap_time_s"] -= 1.0  # second, faster run
    log_to_excel(rep, res, path=p, folder=d2)

    book = openpyxl.load_workbook(d1 / RUN_BOOK)
    assert book.sheetnames == ["기록"]  # one sheet per run folder: report, then the trace
    col_a = [r[0] for r in book["기록"].iter_rows(values_only=True)]
    assert "s_m" in col_a and len(col_a) > len(res.s)

    wb = openpyxl.load_workbook(p)
    assert wb.sheetnames == ["비교", "최고기록"]  # run data lives in the run folders
    cmp_ = wb["비교"]
    assert cmp_.max_row == 3 and cmp_.cell(3, 7).value == round(car.m, 4)  # slower run = rank 2
    assert (p.parent / cmp_.cell(3, 3).hyperlink.target).resolve() == (d1 / RUN_BOOK).resolve()
    best = wb["최고기록"]
    assert best.max_row == 2 and best.cell(2, LAP_COL + 2).value == round(res.lap_time - 1.0, 4)
    assert [cmp_.cell(r, 1).value for r in (2, 3)] == [1, 2] and cmp_.cell(3, 2).value == 1.0
