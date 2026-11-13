"""P9: CLI commands, plots and the HTML report."""

import pandas as pd
import pytest
import typer
from typer.testing import CliRunner

from fsk_lapsim.cli import app, parse_range
from fsk_lapsim.report import ReportOptions, build_report_data, collect_assumptions, render_html

runner = CliRunner()


def _ok(args):
    res = runner.invoke(app, args)
    assert res.exit_code == 0, res.output + repr(res.exception)
    return res.output


def test_parse_range():
    assert parse_range("1:2:0.5") == [1.0, 1.5, 2.0]
    assert parse_range("1,3") == [1.0, 3.0]
    with pytest.raises(typer.BadParameter):
        parse_range("2:1:0.5")


def test_run_events_and_track(tmp_path):
    out = _ok(["run", "--event", "skidpad", "--event", "acceleration"])
    assert "skidpad_time_s" in out and "가정값" in out
    ch = tmp_path / "lap.csv"
    out = _ok(
        [
            "run",
            "--track",
            "tracks/template_autocross_A.yaml",
            "--channels",
            str(ch),
            "--plot-dir",
            str(tmp_path),
        ]
    )
    assert "lap_time_s" in out
    df = pd.read_csv(ch)
    assert {"s_m", "v_kph", "fz_fl_n", "limit"} <= set(df.columns)
    assert (tmp_path / "template_autocross_A_map.png").exists()


def test_run_all_with_plots(tmp_path):
    _ok(["run", "--plot-dir", str(tmp_path), "--channels", str(tmp_path / "ax.csv")])
    for f in ("autocross_map.png", "autocross_speed.png", "motor_operating_points.png", "ggv.png"):
        assert (tmp_path / f).stat().st_size > 1000


def test_sensitivity_command(tmp_path):
    out = _ok(
        [
            "sensitivity",
            "--out",
            str(tmp_path / "s.csv"),
            "--plot",
            str(tmp_path / "s.png"),
            "--event",
            "autocross",
        ]
    )
    assert "Measurement priority" in out and "Equivalence" in out
    assert (tmp_path / "s.png").exists()
    assert len(pd.read_csv(tmp_path / "s.csv")) == 7 * 5


def test_sweep_compare_balance_drs_commands(tmp_path):
    out = _ok(
        [
            "sweep",
            "--cla",
            "3,4",
            "--cda",
            "1.2,1.6",
            "--out",
            str(tmp_path / "w.csv"),
            "--plot",
            str(tmp_path / "w.png"),
        ]
    )
    assert "marked designs" in out and "RW01" in out
    _ok(["sweep", "--cla", "3,4", "--cda", "1.4", "--no-candidates"])
    out = _ok(
        [
            "compare",
            "--event",
            "skidpad",
            "--event",
            "autocross",
            "--reference",
            "RW02",
            "--retrim-front",
            "--out",
            str(tmp_path / "c.csv"),
            "--plot",
            str(tmp_path / "c.png"),
        ]
    )
    assert "DUMMY" in out and "rank_autocross" in out
    out = _ok(["balance-opt", "--range", "0.40:0.50", "--plot", str(tmp_path / "b.png")])
    assert "optimum balance" in out
    out = _ok(["active-aero", "--plot", str(tmp_path / "d.png")])
    assert "close-command lead" in out
    _ok(["active-aero", "--min-speed", "25"])


def test_misc_commands(tmp_path):
    out = _ok(["assumptions"])
    assert "aero.cla_m2" in out and "assumed parameters" in out
    _ok(["init-configs", str(tmp_path / "cfg")])
    assert (tmp_path / "cfg" / "configs" / "vehicle.yaml").exists()
    assert (tmp_path / "cfg" / "data" / "rear_wing_candidates_DUMMY.csv").exists()
    out = _ok(
        [
            "track",
            "--generate",
            "endurance",
            "--seed",
            "3",
            "--length",
            "900",
            "--export-csv",
            str(tmp_path / "t.csv"),
            "--plot",
            str(tmp_path / "t.png"),
        ]
    )
    assert "rule issues: none" in out
    assert len(pd.read_csv(tmp_path / "t.csv")) > 100
    _ok(["track", "--track", "tracks/fsk2026_endurance.yaml"])
    assert runner.invoke(app, ["track"]).exit_code != 0
    out = _ok(["wing2d3d", "--cl", "2.9", "--cd", "0.12", "--chord", "0.7", "--span", "1.1"])
    assert "ClA" in out and "대략적" in out


def test_vehicle_option_with_edited_config(tmp_path):
    import yaml

    from fsk_lapsim.config import default_config_dir

    data = yaml.safe_load((default_config_dir() / "vehicle.yaml").read_text(encoding="utf-8"))
    data["mass"]["total_mass_kg"] = 300  # measured value → plain number
    p = tmp_path / "veh.yaml"
    p.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    out = _ok(["run", "--vehicle", str(p), "--event", "acceleration"])
    assert "acceleration_time_s" in out
    df = collect_assumptions(ReportOptions(vehicle=str(p)))
    assert "mass.total_mass_kg" not in set(df["parameter"])


@pytest.mark.slow
def test_report_quick(tmp_path):
    data = build_report_data(ReportOptions(quick=True, montecarlo_samples=2, workers=2))
    html = render_html(data)
    assert html.startswith("<!doctype html>")
    for text in (
        "가정값 기반",
        "DUMMY",
        "먼저 실측해야 할 파라미터",
        "능동 공력",
        "모델 한계",
        "data:image/png;base64,",
        "가정 목록",
        "발열",
        "자리표시자 점수",
        "몬테카를로",
    ):
        assert text in html, text
    for sec in (
        "summary",
        "vehicle",
        "ggv",
        "sens",
        "priority",
        "sweep",
        "wing",
        "balance",
        "drs",
        "energy",
        "thermal",
        "reference",
        "score",
        "montecarlo",
        "assumptions",
        "limits",
    ):
        assert f'<section id="{sec}">' in html, sec
    assert html.count("<img") >= 16
    out = _ok(["report", "--quick", "--out", str(tmp_path / "r.html")])
    assert (tmp_path / "r.html").stat().st_size > 100_000 and "saved" in out


def test_montecarlo_command(tmp_path):
    out = _ok(
        [
            "montecarlo",
            "--n",
            "3",
            "--workers",
            "1",
            "--retrim-front",
            "--plot",
            str(tmp_path / "mc.png"),
            "--out",
            str(tmp_path / "mc.csv"),
        ]
    )
    assert "ranking robustness" in out and "DUMMY" in out
    assert (tmp_path / "mc.png").exists() and len(pd.read_csv(tmp_path / "mc.csv")) == 3 * 9 * 2
    out = _ok(
        [
            "montecarlo",
            "--n",
            "2",
            "--workers",
            "1",
            "--no-candidates",
            "--plot",
            str(tmp_path / "h.png"),
        ]
    )
    assert "distribution" in out and (tmp_path / "h.png").exists()


def test_gps_command(tmp_path):
    out = _ok(["gps", "--plot", str(tmp_path / "g.png"), "--export-track", str(tmp_path / "t.csv")])
    assert "DUMMY" in out and "lap_time_ratio" in out and "laps: #0" in out
    assert (tmp_path / "g.png").exists() and len(pd.read_csv(tmp_path / "t.csv")) > 100
    out = _ok(["gps", "--method", "position", "--lap", "5"])
    assert "lap_time_sim_s" in out
    res = runner.invoke(app, ["gps", "--gate", "0.001"])
    assert res.exit_code != 0


def test_gps_low_rate_warning(tmp_path, baseline):
    from fsk_lapsim.gps import synth_nmea
    from fsk_lapsim.trackgen import load_track

    tr = load_track("tracks/template_endurance_B.yaml")
    from fsk_lapsim.solver import simulate

    lines = synth_nmea(tr, simulate(baseline, tr), laps=3, rate_hz=1.0, seed=0)
    p = tmp_path / "slow.nmea"
    p.write_text("\n".join(lines), encoding="ascii")
    out = _ok(["gps", str(p)])
    assert "WARNING" in out and "1 Hz" in out


def test_thermal_command(tmp_path):
    out = _ok(["thermal", "--plot", str(tmp_path / "t.png"), "--map-plot", str(tmp_path / "m.png")])
    assert "required_r_k_per_w" in out and "implied motor+inverter efficiency" in out
    assert (tmp_path / "t.png").exists() and (tmp_path / "m.png").exists()


def test_score_command():
    out = _ok(["score", "--with-candidates"])
    assert "PLACEHOLDER" in out and "points vs RW01" in out and "d_total_pts" in out
