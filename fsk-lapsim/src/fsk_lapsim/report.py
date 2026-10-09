"""Single-file HTML report (Korean text, English figure labels, images embedded as base64)."""

from __future__ import annotations

import datetime as _dt
import html
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import __version__, plots
from .active_aero import DRSEventGains, DRSSettings, drs_event_gains, lead_distance_table
from .aero import AeroComponent, components_from_config
from .analysis.balance_opt import BalanceOptResult, optimise_balance
from .analysis.compare import Comparison, compare_candidates
from .analysis.sensitivity import (
    CORE_PARAMS,
    EXTRA_PARAMS,
    SensitivityResult,
    equivalence_table,
    measurement_priority,
    sensitivity,
)
from .analysis.sweep import SweepResult, sweep_cla_cda
from .analysis.uncertainty import Uncertainty
from .config import (
    Param,
    assumed_parameters,
    default_config_dir,
    default_data_dir,
    load_aero_components_config,
    load_drs_config,
    load_events_config,
    load_vehicle_config,
    resolve_path,
)
from .events import EventResults, EventSetup, run_events
from .ggv import cached_ggv
from .solver import LapResult, SolverOptions, simulate
from .trackgen import load_track
from .vehicle import Vehicle, load_vehicle
from .wing import (
    ROUGH_NOTE,
    WingConversion,
    candidate_table,
    candidate_vehicle,
    read_wing_candidates,
)

EXTRA_SECTIONS: list[Callable[[ReportData], str]] = []
"""Hooks that append extra HTML sections (used by extension modules)."""


@dataclass
class ReportOptions:
    """What to compute for the report (paths default to the bundled examples)."""

    vehicle: str | None = None
    events: str | None = None
    components: str | None = None
    candidates: str | None = None
    drs: str | None = None
    uncertainty: str | None = None
    model: str = "axle"
    sweep_cla: tuple[float, ...] = (2.5, 3.0, 3.5, 4.0, 4.5)
    sweep_cda: tuple[float, ...] = (1.0, 1.2, 1.4, 1.6, 1.8)
    extra_sensitivity: bool = True
    reference_track: str | None = "tracks/fsk2026_endurance.yaml"
    montecarlo_samples: int = 0
    quick: bool = False

    def __post_init__(self) -> None:
        if self.quick:
            self.sweep_cla = (2.5, 3.5, 4.5)
            self.sweep_cda = (1.0, 1.4, 1.8)
            self.extra_sensitivity = False


@dataclass
class ReportData:
    """Everything the report shows."""

    opts: ReportOptions
    vehicle: Vehicle
    setup: EventSetup
    components: list[AeroComponent]
    events: EventResults
    sens: SensitivityResult
    priority: pd.DataFrame
    sweep: SweepResult
    comparison: Comparison
    wing_table: pd.DataFrame
    balance: BalanceOptResult
    drs: DRSEventGains
    assumptions: pd.DataFrame
    reference_lap: LapResult | None = None
    extras: dict[str, Any] = field(default_factory=dict)


def collect_assumptions(opts: ReportOptions) -> pd.DataFrame:
    """All parameters flagged ``assumed: true`` in the configuration files used."""
    rows = []

    def add(src: str, items: list[tuple[str, Param]]) -> None:
        for k, p in items:
            rows.append(
                {"file": src, "parameter": k, "value": p.value, "unit": p.unit, "note": p.note}
            )

    add("vehicle.yaml", assumed_parameters(load_vehicle_config(opts.vehicle)))
    add("aero_components.yaml", assumed_parameters(load_aero_components_config(opts.components)))
    add("events.yaml", assumed_parameters(load_events_config(opts.events)))
    add("drs.yaml", assumed_parameters(load_drs_config(opts.drs)))
    return pd.DataFrame(rows)


def build_report_data(
    opts: ReportOptions | None = None, progress: Callable[[str], None] | None = None
) -> ReportData:
    """Run all analyses needed by the report."""
    opts = opts or ReportOptions()
    say = progress or (lambda _m: None)
    veh = load_vehicle(opts.vehicle)
    setup = EventSetup.load(opts.events)
    sopt = SolverOptions(ds_m=setup.ds_m, model=opts.model)
    comps = components_from_config(load_aero_components_config(opts.components))
    say("events")
    ev = run_events(veh, setup, sopt)
    say("sensitivity")
    params = CORE_PARAMS + (EXTRA_PARAMS if opts.extra_sensitivity else ())
    sens = sensitivity(veh, setup, sopt, params)
    unc = Uncertainty.load(opts.uncertainty)
    prio = measurement_priority(sens, unc.sigmas())
    say("rear-wing comparison")
    cand_path = (
        resolve_path(opts.candidates)
        if opts.candidates
        else (default_data_dir() / "rear_wing_candidates_DUMMY.csv")
    )
    cands = read_wing_candidates(cand_path)
    cmp = compare_candidates(veh, comps, cands, setup, sopt)
    rear = next(c for c in comps if c.name == "rear_wing")
    wtab = candidate_table(cands, rear, WingConversion())
    say("sweep")
    points = [("current", veh)] + [
        (c.config_id, candidate_vehicle(veh, comps, c)[0]) for c in cands
    ]
    sw = sweep_cla_cda(veh, opts.sweep_cla, opts.sweep_cda, setup, sopt, points)
    say("balance")
    bal = optimise_balance(veh, setup, sopt, n_scan=9 if opts.quick else 15)
    say("active aero")
    drs = drs_event_gains(veh, DRSSettings.load(opts.drs), setup, sopt)
    extras: dict[str, Any] = {}
    if opts.montecarlo_samples > 1:
        import os

        from .analysis.montecarlo import run_montecarlo

        say("monte carlo")
        extras["mc"] = run_montecarlo(
            veh,
            opts.montecarlo_samples,
            0,
            setup,
            unc,
            cands,
            comps,
            workers=min(4, os.cpu_count() or 1),
        )
    ref_lap = None
    if opts.reference_track:
        say("reference track")
        ref_lap = simulate(veh, load_track(opts.reference_track), sopt)
    data = ReportData(
        opts,
        veh,
        setup,
        comps,
        ev,
        sens,
        prio,
        sw,
        cmp,
        wtab,
        bal,
        drs,
        collect_assumptions(opts),
        ref_lap,
    )
    return data


# --------------------------------------------------------------------------------------------
# HTML rendering
# --------------------------------------------------------------------------------------------
CSS = """
:root{--surface:#fcfcfb;--card:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--line:#e4e3df;
--accent:#2a78d6;--warn-bg:#fff4dc;--warn-line:#eda100;--dummy-bg:#fde8e8;--dummy-line:#d03b3b}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--surface:#1a1a19;
--card:#232322;--ink:#ffffff;--ink2:#c3c2b7;--line:#3a3a37;--accent:#3987e5;
--warn-bg:#3a2f12;--warn-line:#c98500;--dummy-bg:#3b1f1f;--dummy-line:#e66767}}
:root[data-theme="dark"]{--surface:#1a1a19;--card:#232322;--ink:#ffffff;--ink2:#c3c2b7;
--line:#3a3a37;--accent:#3987e5;--warn-bg:#3a2f12;--warn-line:#c98500;--dummy-bg:#3b1f1f;
--dummy-line:#e66767}
*{box-sizing:border-box}
body{margin:0;background:var(--surface);color:var(--ink);font:14px/1.6 system-ui,-apple-system,
"Segoe UI","Noto Sans KR","Apple SD Gothic Neo","Malgun Gothic",sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;margin:36px 0 8px;padding-top:8px;
border-top:1px solid var(--line)}h3{font-size:15px;margin:20px 0 6px}
p,li{color:var(--ink)}.muted{color:var(--ink2);font-size:13px}
.box{border-left:4px solid var(--warn-line);background:var(--warn-bg);padding:10px 14px;
margin:12px 0;border-radius:4px}.dummy{border-left-color:var(--dummy-line);
background:var(--dummy-bg)}
.fig{background:#fff;border:1px solid var(--line);border-radius:6px;padding:8px;margin:10px 0;
overflow-x:auto}.fig img{max-width:100%;height:auto;display:block;margin:0 auto}
.tblwrap{overflow-x:auto;margin:8px 0}
table.tbl{border-collapse:collapse;font-size:12.5px;font-variant-numeric:tabular-nums}
table.tbl th,table.tbl td{border-bottom:1px solid var(--line);padding:4px 10px;text-align:right;
white-space:nowrap}table.tbl th{color:var(--ink2);font-weight:600;text-align:right}
table.tbl td:first-child,table.tbl th:first-child{text-align:left}
nav ol{columns:2;font-size:13px}nav a{color:var(--accent)}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;
margin:12px 0}.kpi{background:var(--card);border:1px solid var(--line);border-radius:6px;
padding:10px 12px}.kpi .v{font-size:22px;font-weight:650}.kpi .l{color:var(--ink2);font-size:12px}
"""


def _tbl(df: pd.DataFrame, fmt: str = "{:.4g}", index: bool = False) -> str:
    def f(x: Any) -> str:
        if isinstance(x, bool | np.bool_):
            return "yes" if x else "no"
        if isinstance(x, float | np.floating):
            return "" if np.isnan(x) else fmt.format(x)
        return html.escape(str(x))

    body = df.to_html(
        index=index,
        border=0,
        classes="tbl",
        escape=False,
        formatters={c: f for c in df.columns},
        na_rep="",
    )
    return f'<div class="tblwrap">{body}</div>'


def _img(fig: Any, alt: str) -> str:
    return (
        f'<div class="fig"><img alt="{html.escape(alt)}" '
        f'src="data:image/png;base64,{plots.fig_to_base64(fig)}"></div>'
    )


def _section(anchor: str, title: str, body: str) -> str:
    return f'<section id="{anchor}"><h2>{html.escape(title)}</h2>{body}</section>'


def render_html(d: ReportData) -> str:
    """Render the report."""
    veh, ev, setup = d.vehicle, d.events, d.setup
    times = ev.times()
    m = ev.metrics()
    dummy = d.comparison.dummy
    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    parts: list[str] = []
    parts.append(f"""<h1>FSK 랩타임·공력 민감도 리포트</h1>
<p class="muted">fsk-lapsim {__version__} · 생성 {now} · 차량 모델: {html.escape(d.opts.model)}
(모델 B = 축 단위 준정상) · 차량: {html.escape(veh.name)}</p>
<div class="box"><b>가정값 기반 결과입니다.</b> 차량 파라미터는 모두 지시서의 가정값이며 실측값이
아닙니다
(아래 "가정 목록" 참고). 트랙은 FS 일반 규정 범위로 만든 합성 트랙입니다 (FSK 규정으로 교체 필요).
절대값보다 <b>설계안 사이의 차이와 순위</b>를 보는 용도로 쓰십시오.</div>""")
    if dummy:
        parts.append(
            '<div class="box dummy"><b>리어윙 후보 데이터는 DUMMY 입니다.</b> 실제 ANSYS 결과가 '
            "아니라 형식 확인용 임의값이므로 후보 비교 결과는 의미가 없습니다.</div>"
        )
    toc = [
        ("summary", "이벤트 결과 요약"),
        ("vehicle", "차량·트랙"),
        ("ggv", "GGV"),
        ("sens", "민감도·등가 환산"),
        ("priority", "먼저 실측할 파라미터"),
        ("sweep", "ClA–CdA 스윕"),
        ("wing", "리어윙 후보 비교"),
        ("balance", "공력 밸런스"),
        ("drs", "능동 공력 (DRS)"),
        ("energy", "에너지·모터 운전점"),
        ("reference", "참고: FSK 2026 중심선"),
        ("assumptions", "가정 목록"),
        ("limits", "모델 한계"),
    ]
    parts.append(
        "<nav><ol>" + "".join(f'<li><a href="#{a}">{t}</a></li>' for a, t in toc) + "</ol></nav>"
    )

    # -- summary
    kpis = [
        ("가속 75 m", f"{times['acceleration']:.3f} s"),
        ("스키드패드", f"{times['skidpad']:.3f} s"),
        ("오토크로스", f"{times['autocross']:.2f} s"),
        ("엔듀런스 (총)", f"{times['endurance']:.1f} s"),
        ("엔듀런스 에너지", f"{m['endurance_energy_kwh']:.2f} kWh"),
    ]
    kp = (
        '<div class="kpis">'
        + "".join(
            f'<div class="kpi"><div class="v">{v}</div><div class="l">{k}</div></div>'
            for k, v in kpis
        )
        + "</div>"
    )
    met = pd.DataFrame([{"항목": k, "값": v} for k, v in m.items()])
    warn = (
        "<p>타당성 경고: " + "; ".join(html.escape(w) for w in ev.warnings) + "</p>"
        if ev.warnings
        else "<p class='muted'>타당성 경고 없음 (FS EV 일반 범위: 가속 3.2–5.0 s, "
        "스키드패드 4.4–6.0 s, 오토크로스 평균 40–65 km/h).</p>"
    )
    parts.append(_section("summary", "1. 이벤트 결과 요약 (가정값 기반)", kp + _tbl(met) + warn))

    # -- vehicle and tracks
    vt = pd.DataFrame(
        [
            {"항목": "총질량 [kg]", "값": veh.mass_kg},
            {"항목": "CG 높이 [m]", "값": veh.cg_height_m},
            {"항목": "정적 전륜 비율", "값": veh.static_front_fraction},
            {"항목": "휠베이스 [m]", "값": veh.wheelbase_m},
            {"항목": "LLTD 전륜", "값": veh.lltd_front},
            {
                "항목": "μ0 / Fz0 [N] / k",
                "값": f"{veh.tire.mu0} / {veh.tire.fz0_n} / {veh.tire.load_sensitivity}",
            },
            {"항목": "ClA / CdA [m²]", "값": f"{veh.aero.cla_m2:.3f} / {veh.aero.cda_m2:.3f}"},
            {"항목": "공력 밸런스 (전륜)", "값": veh.aero.balance_front},
            {"항목": "출력 상한 [kW]", "값": veh.powertrain.max_power_w / 1e3},
            {"항목": "최고속도 (회전수) [km/h]", "값": veh.v_max_mps * 3.6},
        ]
    )
    tracks = []
    for name, tr in (("오토크로스", setup.autocross_track), ("엔듀런스", setup.endurance_track)):
        sm = tr.summary()
        tracks.append(
            {
                "이벤트": name,
                "트랙": tr.name,
                "길이 m": sm["length_m"],
                "최소 반경 m": sm["min_radius_m"],
                "최장 직선 m": sm["longest_straight_m"],
                "직선 비율": sm["straight_fraction"],
            }
        )
    body = (
        _tbl(vt)
        + _tbl(pd.DataFrame(tracks))
        + _img(plots.track_map(setup.autocross_track, ev.autocross), "autocross map")
        + _img(
            plots.speed_trace([("autocross", ev.autocross)], "Autocross speed trace"), "speed trace"
        )
        + _img(
            plots.speed_trace(
                [("endurance flying lap", ev.endurance.flying_lap)], "Endurance flying lap"
            ),
            "endurance trace",
        )
    )
    parts.append(_section("vehicle", "2. 차량·트랙", body))

    # -- GGV
    parts.append(
        _section(
            "ggv",
            "3. GGV",
            "<p>속도별 g-g 한계 (일정 속도 유지 기준 횡한계, 위쪽 평평한 "
            "부분은 출력 한계).</p>" + _img(plots.ggv_plot(cached_ggv(veh, d.opts.model)), "ggv"),
        )
    )

    # -- sensitivity
    piv = d.sens.pivot().reset_index()
    disp = d.sens.table.drop_duplicates("label").set_index("label")["display_unit"]
    piv.insert(1, "표시 단위", piv["label"].map(disp))
    eq = equivalence_table(d.sens, "autocross", "mass_kg")
    eq_txt = []
    for r in eq.itertuples():
        if r.param == "mass_kg" or not np.isfinite(r.equivalent_change):
            continue
        eq_txt.append(f"{html.escape(r.label)} {r.equivalent_change:+.4g} {html.escape(r.unit)}")
    body = (
        "<p>중앙차분 민감도: 표시 단위만큼 늘렸을 때 시간 변화 [s] (엔듀런스는 총 시간, "
        "endurance_energy 는 kWh). 양수 = 느려짐.</p>"
        + _tbl(piv, "{:+.4f}")
        + _img(plots.sensitivity_bars(d.sens, "autocross"), "sens autocross")
        + _img(plots.sensitivity_bars(d.sens, "endurance"), "sens endurance")
        + "<h3>등가 환산 (오토크로스 기준, +1 kg 과 같은 시간 손실)</h3><p>+1 kg ≡ "
        + " ≡ ".join(eq_txt)
        + "</p>"
        + _tbl(eq, "{:+.4g}")
    )
    parts.append(_section("sens", "4. 민감도·등가 환산", body))

    # -- priority
    body = (
        "<p>가정값의 불확실성 σ (configs/uncertainty.yaml, 이 값도 가정) × 민감도 = 그 가정이 "
        "틀렸을 때의 "
        "시간 오차. 점수 = 가속 + 스키드패드 + 오토크로스 + 엔듀런스/20 [s]. 위에 있을수록 먼저 "
        "실측해야 "
        "결과를 믿을 수 있습니다.</p>" + _tbl(d.priority.reset_index(names="순위"), "{:.4g}")
    )
    parts.append(_section("priority", "5. 먼저 실측해야 할 파라미터 (민감도 × 불확실성)", body))

    # -- sweep
    body = (
        "<p>밸런스·질량 고정, ClA/CdA 만 변경. 점 = 현재 차량(검정 원)과 리어윙 후보(사각형, "
        + ("DUMMY" if dummy else "")
        + "; 후보는 질량·밸런스도 달라 격자와 조건이 약간 다름).</p>"
        + _img(plots.sweep_contours(d.sweep), "sweep")
        + _tbl(d.sweep.points, "{:.4f}")
    )
    parts.append(_section("sweep", "6. ClA–CdA 스윕", body))

    # -- wing comparison
    cols = (
        [
            "config_id",
            "flap1_id",
            "flap2_id",
            "source",
            "rw_ClA_m2",
            "rw_CdA_m2",
            "balance_front",
            "mass_kg",
        ]
        + [f"{e}_time_s" for e in d.comparison.events]
        + [f"d_{e}_s" for e in d.comparison.events]
        + [f"rank_{e}" for e in d.comparison.events]
        + (["endurance_energy_kwh"] if "endurance" in d.comparison.events else [])
    )
    ct = d.comparison.table[[c for c in cols if c in d.comparison.table.columns]]
    body = (
        (
            '<div class="box dummy"><b>DUMMY 데이터</b> — 순위와 차이는 형식 확인용입니다.</div>'
            if dummy
            else ""
        )
        + f"<p>기준 = {html.escape(d.comparison.reference)}. Δ = 후보 − 기준 (음수 = 빠름). "
        f"리어윙 질량·높이 변화가 총질량·CG 에, 압력중심·항력 높이가 밸런스에 반영됩니다. "
        f"{html.escape(ROUGH_NOTE)}.</p>"
        + _tbl(ct, "{:.4f}")
        + _img(plots.compare_deltas(d.comparison), "compare")
        + "<h3>2D → 3D 변환 (대략적 추정)</h3>"
        + _tbl(d.wing_table, "{:.4g}")
    )
    parts.append(_section("wing", "7. 리어윙 후보 비교", body))

    # -- balance
    b = d.balance
    body = (
        f"<p>목적함수 {b.objective}: 최적 밸런스 <b>{b.best_balance * 100:.1f} %</b> "
        f"({b.best_value:.3f} s), 현재 {b.base_balance * 100:.1f} % ({b.base_value:.3f} s), "
        f"차이 {b.base_value - b.best_value:.3f} s. 준정상 최적점은 전·후축이 동시에 포화되는 "
        f"점이라 "
        "실차는 안정성(오버스티어 방지) 마진만큼 뒤쪽으로 둡니다.</p>"
        + _img(plots.balance_scan(b), "balance")
        + "<p>최적점의 제약별 횡한계 [g]:</p>"
        + _tbl(b.limits_at_best.reset_index(names="speed"), "{:.3f}")
    )
    parts.append(_section("balance", "8. 공력 밸런스 최적화", body))

    # -- DRS
    dt = d.drs.table
    endu = d.drs.laps["endurance_lap"]
    lead = lead_distance_table([60, 80, 100, 120, 140], [0.1, 0.2, 0.25, 0.3, 0.5])
    body = (
        "<p>저항 저감 상태 = 리어윙 플랩 개방 (ClA·CdA 감소는 drs.yaml 가정값). 개방 조건: |ay| "
        "< 임계값, "
        "최대 가속 중, 그리고 개방이 가속에 이득이 되는 속도 이상 (자동 교차 속도). 닫힘은 제동 "
        "시작 "
        "<b>전환 시간 T 이전</b>에 명령해야 합니다 (리드 거리 ≈ v·T).</p>"
        + _tbl(dt, "{:.3f}")
        + _img(plots.drs_trace(endu), "drs")
        + "<h3>엔듀런스 랩 개방 구간</h3>"
        + _tbl(endu.zones, "{:.2f}")
        + "<h3>닫힘 명령 리드 거리 [m] (속도 × 전환 시간) — 페일세이프 설계 참고</h3>"
        + _tbl(lead, "{:.2f}")
    )
    parts.append(_section("drs", "9. 능동 공력 (DRS)", body))

    # -- energy
    fl = ev.endurance.flying_lap
    e = fl.energy
    en = pd.DataFrame(
        [
            {"항목": "랩 시간 [s]", "값": fl.lap_time_s},
            {"항목": "랩당 순 에너지 [kWh]", "값": e.battery_net_j / 3.6e6},
            {"항목": "랩당 회생 [kWh]", "값": e.regen_j / 3.6e6},
            {"항목": "항력 일 [kJ]", "값": e.work_drag_j / 1e3},
            {"항목": "구름저항 일 [kJ]", "값": e.work_rolling_j / 1e3},
            {"항목": "마찰 브레이크 열 [kJ]", "값": e.friction_brake_j / 1e3},
            {"항목": "평균 출력 [kW]", "값": e.battery_net_j / fl.lap_time_s / 1e3},
            {"항목": "최대 출력 [kW]", "값": float(e.p_battery_w.max()) / 1e3},
            {"항목": "엔듀런스 총 에너지 (회생 포함) [kWh]", "값": m["endurance_energy_kwh"]},
            {
                "항목": "엔듀런스 총 에너지 (회생 미포함) [kWh]",
                "값": m["endurance_energy_no_regen_kwh"],
            },
        ]
    )
    parts.append(
        _section(
            "energy",
            "10. 에너지·모터 운전점 (엔듀런스 플라잉 랩)",
            _tbl(en) + _img(plots.motor_map(fl, veh), "motor map"),
        )
    )

    # -- reference
    if d.reference_lap is not None:
        rl = d.reference_lap
        tr = load_track(d.opts.reference_track) if d.opts.reference_track else None
        body = (
            "<p>저장소에 있던 FSK 2026 엔듀런스 cone2track 중심선 (측량 정확도 미확인, 중심선 "
            "주행이라 "
            f"레이싱 라인보다 느림). 플라잉 랩 {rl.lap_time_s:.2f} s, 평균 "
            f"{rl.mean_speed_mps * 3.6:.1f} km/h. 합성 트랙보다 최소 반경이 훨씬 작습니다.</p>"
        )
        if tr is not None:
            body += _img(plots.track_map(tr, rl), "fsk map")
        parts.append(_section("reference", "11. 참고: FSK 2026 엔듀런스 중심선", body))

    parts.append(_score_section(d))
    if "mc" in d.extras:
        parts.append(_mc_section(d))
    for hook in EXTRA_SECTIONS:
        parts.append(hook(d))

    parts.append(
        _section(
            "assumptions",
            "12. 가정 목록 (assumed: true)",
            "<p>설정 파일에서 assumed: true 인 값. 실측되면 YAML 에서 숫자로 바꾸십시오.</p>"
            + _tbl(d.assumptions, "{:.6g}"),
        )
    )
    parts.append(_section("limits", "13. 모델 한계 — 결과를 신뢰할 수 없는 영역", LIMITS_HTML))
    return (
        "<!doctype html><html lang='ko'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>FSK Lap Report</title><style>"
        + CSS
        + "</style></head><body><main>"
        + "".join(parts)
        + "</main></body></html>"
    )


def _thermal_section(d: ReportData) -> str:
    from .thermal import LossModel, endurance_heat

    lm = LossModel.load()
    h = endurance_heat(d.events.endurance, lm)  # type: ignore[arg-type]
    body = (
        "<p>손실 모델 (thermal.yaml, 전부 가정값): 모터 P = k_cu·T² + k_fe·ω^1.5 + k_w·ω², "
        "인버터 P = P0 + a·I + b·I² (I = |T|/k_t), 배터리 I²R. "
        "부품마다 열용량 C·열저항 R 집중 모델로 "
        "엔듀런스 전체를 적분했다. required_r = 정상상태에서 한계 온도를 지키는 데 필요한 열저항 "
        "(assumed_r 보다 작으면 냉각 강화 필요).</p>"
        + _tbl(h.table, "{:.3f}")
        + f"<p>구동 중 모터+인버터 종합 효율 {h.implied_motor_inverter_efficiency:.3f} "
        f"(랩 모델은 배터리→휠 {d.vehicle.powertrain.efficiency:.2f} 상수).</p>"
        + _img(plots.thermal_trace(h), "thermal")
        + _img(
            plots.motor_efficiency_map(
                lm,
                d.events.endurance.flying_lap,  # type: ignore[union-attr]
                d.vehicle,
            ),
            "eff map",
        )
    )
    return _section("thermal", "10b. 모터·인버터·어큐뮬레이터 발열 (E3, 가정값)", body)


def _score_section(d: ReportData) -> str:
    from .scoring import ScoringModel, comparison_points, sensitivity_points

    sm = ScoringModel.load()
    cal = sm.calibrate(d.events)
    sp = sensitivity_points(cal, d.sens.table, d.events)
    cp = comparison_points(sm, d.comparison.table, d.comparison.reference)
    body = (
        '<div class="box"><b>자리표시자 점수 공식</b> (configs/scoring.yaml) — FS 계열 공식 형태와 '
        "기준 기록 (기준 차량 시간 × 비율) 을 임의로 둔 것. FSK 규정 공식으로 교체 전에는 "
        "점수 절대값이 아니라 설계안 간 점수 차이의 크기 감각으로만 쓸 것.</div>"
        + _tbl(cal.table(d.events), "{:.3f}")
        + "<h3>민감도 (표시 단위당 점수 변화, 선형화)</h3>"
        + _tbl(sp, "{:+.3f}")
        + f"<h3>리어윙 후보 점수 차이 (기준 {html.escape(d.comparison.reference)})"
        + (" — DUMMY" if d.comparison.dummy else "")
        + "</h3>"
        + _tbl(cp, "{:+.3f}")
    )
    return _section("score", "11c. 점수 환산 (E4, 자리표시자)", body)


def _mc_section(d: ReportData) -> str:
    mc = d.extras["mc"]
    dist = mc.distribution()
    body = (
        f"<p>{mc.n} 샘플. 차량 가정값(uncertainty.yaml), 2D→3D 변환 계수"
        "(엔드플레이트 계수·e·k 보정, 후보 공통), "
        "후보별 CFD 오차(Cl 2 %, Cd 4 %)를 함께 뽑았다. 범위 자체가 가정이다.</p>"
        + _tbl(dist, "{:.3f}")
        + _img(
            plots.histogram(
                mc.matrix("autocross")["base"].to_numpy(),
                "autocross time [s]",
                "Autocross lap time distribution (base car)",
                [("nominal", float(mc.nominal.loc["base", "autocross"]))],
            ),
            "mc hist",
        )
    )
    if mc.candidates():
        dummy = (
            '<div class="box dummy"><b>DUMMY 후보</b> — 확률은 형식 확인용입니다.</div>'
            if d.comparison.dummy
            else ""
        )
        for metric in ("autocross", "endurance_lap"):
            body += (
                f"<h3>{metric}: 순위 견고성 — 전체 순위가 명목과 같을 확률 "
                f"{mc.p_ranking_unchanged(metric):.2f}</h3>"
                + dummy
                + _tbl(mc.rank_stats(metric), "{:.3f}")
                + "<p>가까운 후보 쌍의 순위 역전 확률:</p>"
                + _tbl(mc.pairwise_flip(metric).head(10), "{:.3f}")
            )
        body += _img(plots.rank_probability(mc.rank_stats("autocross"), "autocross"), "mc rank")
    return _section("montecarlo", "11b. 몬테카를로 불확실성 (E1)", body)


LIMITS_HTML = """<ul>
<li><b>주행선 = 트랙 기하.</b> 레이싱 라인 최적화가 없어 헤어핀·슬라럼 속도는 보수적이며, 실제
코스 중심선을
쓰면 더 느리게 나온다. 트랙 간 절대 비교는 의미가 작고 설계안 간 상대 비교에 쓴다.</li>
<li><b>준정상(QSS).</b> 요 관성·과도 하중이동·댐퍼·조향 응답이 없다. 슬라럼·시케인처럼 곡률이
빨리 바뀌는
구간은 낙관적일 수 있다. 공력 밸런스 최적점은 두 축 동시 포화점이라 안정성 마진이 반영되지
않는다.</li>
<li><b>타이어.</b> μ(Fz) 선형 하중민감도 + 마찰 타원의 실효 모델. 슬립·온도·마모·캠버 효과 없음.
μ0 와 k 가 결과를 가장 크게 좌우한다 (실측 우선순위 1·2위).</li>
<li><b>공력.</b> ClA·CdA·밸런스가 차고·피치·요·롤에 무관한 상수. 2D→3D 변환은 대략적 추정이며
후보 간
상대 비교도 유효 종횡비·k·e 가정에 따라 바뀔 수 있다 (몬테카를로로 순위 역전 확률 확인).</li>
<li><b>파워트레인.</b> 종합 효율 상수 (토크 경로에도 적용 — 보수적), 배터리 전압 강하·열 디레이팅·
인버터 전류 한계 없음. 회생은 후축 분담 × 모터 토크·출력 한계로 단순화.</li>
<li><b>에너지.</b> 엔듀런스 에너지는 전 랩 최대 성능 주행 가정 (에너지 관리 없음, 옵션으로 출력
제한 가능).
실제 팩 용량보다 크면 출력 제한 주행이 필요하다.</li>
<li><b>가속.</b> 75 m 정지 출발, 반응 시간·트랙션 컨트롤 과도·휠 슬립 없음. 스키드패드는 정상선회
해석해.</li>
</ul>"""


def write_report(
    path: str | Path,
    opts: ReportOptions | None = None,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Build the data, render and write the HTML report; returns the path."""
    data = build_report_data(opts, progress)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(data), encoding="utf-8")
    return out


__all__ = [
    "EXTRA_SECTIONS",
    "ReportData",
    "ReportOptions",
    "build_report_data",
    "collect_assumptions",
    "default_config_dir",
    "render_html",
    "write_report",
]
