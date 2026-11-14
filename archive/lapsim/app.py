"""Streamlit GUI for the lap time simulator.

    python -m lapsim gui            (or: streamlit run app.py)
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from lapsim import figures, study, targets
from lapsim.channels import channels
from lapsim.line import driven_path
from lapsim.solver import PHASES, simulate
from lapsim.track import track_from_dict
from lapsim.vehicle import G, vehicle_from_dict, vehicle_to_dict

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="FSK Lap Sim", page_icon="🏁", layout="wide")


# ---------------------------------------------------------------------------
# cached heavy steps (keyed on JSON text so edits invalidate them)
@st.cache_data(show_spinner=False)
def load_track_cached(text: str, name: str, ds: float, smooth: float):
    data = json.loads(text)
    if isinstance(data, list):
        data = {"centerline": data}
    return track_from_dict(data, ds=ds, smooth_m=smooth, name=name)


@st.cache_data(show_spinner=False)
def line_cached(text, name, ds, smooth, method, car_width, margin, track_width, slalom, car_json):
    trk = load_track_cached(text, name, ds, smooth)
    return driven_path(vehicle_from_dict(json.loads(car_json)), trk, method, car_width, margin,
                       default_width=track_width, slalom=slalom, smooth_m=smooth)


@st.cache_data(show_spinner=False)
def run_cached(car_json: str, track_key: str, _track):
    veh = vehicle_from_dict(json.loads(car_json))
    res = simulate(veh, _track)
    return veh, res, channels(veh, res)


def car_json_of(d: dict) -> str:
    return json.dumps(d, sort_keys=True)


def num(label, value, key, step=None, fmt=None, help=None):
    return st.number_input(label, value=float(value), key=key, step=step, format=fmt, help=help)


# ---------------------------------------------------------------------------
# sidebar: files and path
st.sidebar.title("🏁 FSK Lap Sim")
car_files = sorted((ROOT / "cars").glob("*.json"))
track_files = sorted((ROOT / "tracks").glob("*.json"))
default_track = next((i for i, f in enumerate(track_files) if "fsk" in f.name), 0)

car_pick = st.sidebar.selectbox("차량 파일", [f.name for f in car_files])
car_up = st.sidebar.file_uploader("…또는 차량 JSON 업로드", type="json", key="car_up")
if car_up is not None:
    car_src, car_base = car_up.name, json.loads(car_up.getvalue())
else:
    car_src, car_base = car_pick, json.loads((ROOT / "cars" / car_pick).read_text(encoding="utf-8"))

track_pick = st.sidebar.selectbox("트랙 파일", [f.name for f in track_files], index=default_track)
track_up = st.sidebar.file_uploader("…또는 트랙 JSON 업로드", type="json", key="track_up")
if track_up is not None:
    track_name, track_text = Path(track_up.name).stem, track_up.getvalue().decode("utf-8")
else:
    track_name = Path(track_pick).stem
    track_text = (ROOT / "tracks" / track_pick).read_text(encoding="utf-8")

st.sidebar.subheader("주행 경로")
LINE_LABELS = {"center": "중심선", "compromise": "주행 라인 (곡률/길이 절충)",
               "mincurv": "주행 라인 (최소 곡률)", "best": "주행 라인 (둘 중 빠른 것)"}
line_mode = st.sidebar.radio("경로", list(LINE_LABELS), format_func=LINE_LABELS.get)
car_width = st.sidebar.number_input("차폭 [m]", value=1.45, step=0.05)
line_margin = st.sidebar.number_input("트랙 경계 여유 [m]", value=0.25, step=0.05)
track_width = st.sidebar.number_input("트랙 폭 (파일에 없을 때) [m]", value=4.0, step=0.5)
use_slalom = st.sidebar.checkbox("슬라럼 콘 S자 통과", value=True,
                                 help="트랙 JSON의 center 콘(슬라럼)을 좌우 번갈아 통과. 첫 방향은 랩타임으로 선택")
with st.sidebar.expander("이산화"):
    ds = st.number_input("격자 간격 ds [m]", value=0.5, step=0.1, min_value=0.1)
    smooth = st.number_input("센터라인 스무딩 [m]", value=4.0, step=0.5, min_value=0.5)

# ---------------------------------------------------------------------------
tab_car, tab_lap, tab_cmp, tab_coef, tab_tgt, tab_sweep, tab_loads = st.tabs(
    ["🚗 차량 설정", "⏱ 랩 결과", "Δ 비교", "📈 공력 계수", "🎯 목표치", "🗺 스윕", "📋 하중표"])

# ---- vehicle editor --------------------------------------------------------
with tab_car:
    k = car_src + ":"
    st.caption(f"편집 중: **{car_src}** — 값을 바꾸면 모든 탭이 바로 다시 계산돼요.")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("**질량 / 기하**")
        mass = num("질량 (드라이버 포함) [kg]", car_base["mass"], k + "mass", 1.0)
        cg = num("무게중심 높이 [m]", car_base["cg_height"], k + "cg", 0.01)
        wb = num("휠베이스 [m]", car_base["wheelbase"], k + "wb", 0.01)
        wdf = num("앞축 정적 하중 비율", car_base["weight_dist_front"], k + "wdf", 0.01)
        iz = num("요 관성 Iz [kg·m²]", car_base.get("yaw_inertia", 0.0), k + "iz", 5.0,
                 help="0이면 과도 요 응답 무시")
        crr = num("구름저항 계수", car_base.get("rolling_resistance", 0.015), k + "crr", 0.001, "%.3f")
    with c2:
        st.markdown("**타이어**")
        ty = car_base.get("tyre", {})
        mux = num("μx (종방향)", ty.get("mu_x", 1.55), k + "mux", 0.01)
        muy = num("μy (횡방향)", ty.get("mu_y", 1.60), k + "muy", 0.01)
        ks = num("하중 민감도", ty.get("load_sensitivity", 0.1), k + "ks", 0.01)
        fz0 = num("기준 휠 하중 Fz0 [N]", ty.get("Fz0", 700.0), k + "fz0", 10.0)
    with c3:
        st.markdown("**파워트레인 (EV)**")
        pt = car_base.get("powertrain", {})
        pmax = num("최대 출력 [kW]", pt.get("max_power", 80000.0) / 1e3, k + "pmax", 1.0)
        tmax = num("모터 토크 [Nm]", pt.get("max_torque", 140.0), k + "tmax", 5.0)
        rpm = num("최고 회전수 [rpm]", pt.get("max_rpm", 6500.0), k + "rpm", 100.0)
        gr = num("감속비", pt.get("gear_ratio", 4.0), k + "gr", 0.1)
        rw = num("타이어 반경 [m]", pt.get("wheel_radius", 0.2), k + "rw", 0.005, "%.3f")
        eff = num("효율", pt.get("efficiency", 0.9), k + "eff", 0.01)
    with c4:
        st.markdown("**서스펜션**")
        sp = car_base.get("suspension") or {}
        use_susp = st.checkbox("좌우 하중이동 / 차고 모델 사용", value=bool(sp), key=k + "susp")
        tf = num("앞 트레드 [m]", sp.get("track_front", 1.22), k + "tf", 0.01)
        tr = num("뒤 트레드 [m]", sp.get("track_rear", 1.18), k + "tr", 0.01)
        ksf = num("앞 롤 강성 [Nm/deg]", sp.get("roll_stiffness_front", 450.0), k + "ksf", 10.0)
        ksr = num("뒤 롤 강성 [Nm/deg]", sp.get("roll_stiffness_rear", 400.0), k + "ksr", 10.0)
        st.caption(f"LLTD 앞 = {ksf / max(ksf + ksr, 1e-9) * 100:.1f} %")
        rhf = num("앞 정적 차고 [mm]", sp.get("ride_height_front", 30.0), k + "rhf", 1.0)
        rhr = num("뒤 정적 차고 [mm]", sp.get("ride_height_rear", 35.0), k + "rhr", 1.0)
        khf = num("앞 히브 강성 [N/mm]", sp.get("heave_rate_front", 0.0), k + "khf", 5.0,
                  help="0 = 차고 고정")
        khr = num("뒤 히브 강성 [N/mm]", sp.get("heave_rate_rear", 0.0), k + "khr", 5.0)
        hmin = num("범프스톱 최소 차고 [mm]", sp.get("min_ride_height", 8.0), k + "hmin", 1.0)

    st.markdown("**공력 장치** — 행 추가/삭제 가능. `x_cp`: 앞차축 기준 압력중심 [m] (음수 = 앞차축 앞), "
                "`rh_cl`/`rh_cd`: 차고 맵 계수 `c_f, c_r, c_ff, c_rr, c_fr` (mm당 상대 변화)")
    rows = []
    for name, d in car_base["aero"]["devices"].items():
        rows.append({"device": name, "ClA": d.get("ClA", 0.0), "CdA": d.get("CdA", 0.0),
                     "x_cp": d.get("x_cp"), "h_cp": d.get("h_cp", 0.35),
                     "ClA_min": d.get("ClA_min"), "ClA_max": d.get("ClA_max"),
                     "mass_per_cla": d.get("mass_per_cla", 0.0),
                     "rh_cl": ", ".join(str(c) for c in d.get("rh_cl") or []),
                     "rh_cd": ", ".join(str(c) for c in d.get("rh_cd") or [])})
    dev_df = st.data_editor(pd.DataFrame(rows), num_rows="dynamic", width="stretch",
                            key=k + "devices")
    rho = num("공기 밀도 ρ [kg/m³]", car_base["aero"].get("rho", 1.2), k + "rho", 0.01)

    def _coeffs(text):
        text = "" if text is None or (isinstance(text, float) and np.isnan(text)) else str(text)
        vals = [float(x) for x in text.replace(";", ",").split(",") if x.strip()]
        return vals or None

    devices = {}
    for _, r in dev_df.iterrows():
        if not isinstance(r["device"], str) or not r["device"].strip():
            continue
        d = {"ClA": float(r["ClA"] or 0.0), "CdA": float(r["CdA"] or 0.0),
             "h_cp": float(r["h_cp"]) if pd.notna(r["h_cp"]) else 0.35,
             "mass_per_cla": float(r["mass_per_cla"]) if pd.notna(r["mass_per_cla"]) else 0.0}
        d["x_cp"] = float(r["x_cp"]) if pd.notna(r["x_cp"]) else wb / 2
        for col in ("ClA_min", "ClA_max"):
            if pd.notna(r[col]):
                d[col] = float(r[col])
        for col in ("rh_cl", "rh_cd"):
            c = _coeffs(r[col])
            if c:
                d[col] = c
        devices[r["device"].strip()] = d

    car_dict = {
        "name": car_base.get("name", car_src), "mass": mass, "cg_height": cg, "wheelbase": wb,
        "weight_dist_front": wdf, "rolling_resistance": crr, "yaw_inertia": iz,
        "tyre": {"mu_x": mux, "mu_y": muy, "load_sensitivity": ks, "Fz0": fz0},
        "aero": {"rho": rho, "devices": devices},
        "powertrain": {"max_power": pmax * 1e3, "max_torque": tmax, "max_rpm": rpm,
                       "gear_ratio": gr, "wheel_radius": rw, "efficiency": eff},
    }
    if use_susp:
        car_dict["suspension"] = {
            "track_front": tf, "track_rear": tr, "roll_stiffness_front": ksf,
            "roll_stiffness_rear": ksr, "ride_height_front": rhf, "ride_height_rear": rhr,
            "heave_rate_front": khf, "heave_rate_rear": khr, "min_ride_height": hmin}
    try:
        veh = vehicle_from_dict(car_dict)
        st.code(veh.aero.table(), language=None)
    except Exception as e:  # noqa: BLE001
        st.error(f"차량 설정 오류: {e}")
        st.stop()
    st.download_button("편집한 차량 JSON 저장", json.dumps(vehicle_to_dict(veh), indent=2, ensure_ascii=False),
                       file_name=f"{Path(car_src).stem}_edited.json", mime="application/json")

car_json = car_json_of(car_dict)

# ---- track / path ------------------------------------------------------------
try:
    with st.spinner("트랙 / 주행 경로 계산 중… (슬라럼 방향 선택, 최소 곡률 라인은 수십 초 걸릴 수 있어요)"):
        limits = load_track_cached(track_text, track_name, ds, smooth)
        track, path_info = line_cached(track_text, track_name, ds, smooth, line_mode, car_width,
                                       line_margin, track_width, use_slalom, car_json)
except Exception as e:  # noqa: BLE001
    st.error(f"트랙 오류: {e}")
    st.stop()
track_key = (f"{track_name}|{hash(track_text)}|{ds}|{smooth}|{line_mode}|{car_width}|{line_margin}|"
             f"{track_width}|{use_slalom}|{hash(car_json)}")
st.sidebar.caption(f"**{track.name}** · {track.length:.1f} m · {'폐회로' if track.closed else '개회로'}")
if path_info.get("slalom_sides"):
    sides = " | ".join("".join("L" if s > 0 else "R" for s in g) for g in path_info["slalom_sides"])
    st.sidebar.caption(f"슬라럼 {len(path_info['slalom_sides'])}개 · 콘 기준 통과 방향 {sides} · "
                       f"최소 콘 거리 {np.abs(path_info['cone_clearance']).min():.2f} m "
                       f"(목표 {path_info['pass_distance']:.2f} m)")
if path_info.get("line_times"):
    st.sidebar.caption(" · ".join(f"{m}: {t:.3f} s" for m, t in path_info["line_times"].items()))

# ---- lap result ----------------------------------------------------------------
with tab_lap:
    veh, res, ch = run_cached(car_json, track_key, track)
    runs = {"baseline": (veh, res, ch)}
    pt = res.phase_time()
    lt = res.corner_limiter_time()
    m = st.columns(6)
    m[0].metric("랩타임 [s]", f"{res.lap_time:.3f}")
    m[1].metric("평균 속도 [km/h]", f"{res.s[-1] / res.lap_time * 3.6:.1f}")
    m[2].metric("최고 속도 [km/h]", f"{res.v.max() * 3.6:.1f}")
    m[3].metric("최대 횡가속도 [g]", f"{np.abs(res.a_y).max() / G:.2f}")
    m[4].metric("배터리 [Wh/lap]", f"{ch['energy_battery_Wh'][-1]:.0f}")
    m[5].metric("코너 한계 앞/뒤 [s]", f"{lt['front']:.1f}/{lt['rear']:.1f}")
    st.caption("시간 비율: " + ", ".join(f"{p} {pt[p] / res.lap_time * 100:.1f}%" for p in PHASES))

    c1, c2 = st.columns([3, 2])
    with c1:
        st.plotly_chart(figures.speed(runs), width="stretch")
        st.plotly_chart(figures.lines(runs, ["ax_g", "ay_g"], "Acceleration", "g"),
                        width="stretch")
    with c2:
        map_ch = st.selectbox("지도 색상 채널", [k_ for k_, v_ in ch.items() if np.asarray(v_).dtype.kind == "f"
                                             and k_ not in ("s_m", "t_s", "x_m", "y_m")], index=0)
        st.plotly_chart(figures.track_map(runs, limits, map_ch, map_ch), width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(figures.gg(runs), width="stretch")
        st.plotly_chart(figures.lines(runs, ["Fz_FL_N", "Fz_FR_N", "Fz_RL_N", "Fz_RR_N"],
                                      "Wheel loads", "N"), width="stretch")
        st.plotly_chart(figures.lines(runs, ["power_battery_kW"], "Battery power", "kW"),
                        width="stretch")
    with c2:
        st.plotly_chart(figures.lines(runs, ["ride_height_front_mm", "ride_height_rear_mm"],
                                      "Ride height", "mm"), width="stretch")
        st.plotly_chart(figures.lines(runs, ["aero_balance_front_pct"], "Aero balance (live)",
                                      "% front"), width="stretch")
        st.plotly_chart(figures.utilisation_hist(runs), width="stretch")
    st.plotly_chart(figures.aero_devices(veh, res, ch), width="stretch")

    st.subheader("채널 데이터")
    df = pd.DataFrame(ch)
    st.dataframe(df, width="stretch", height=300)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    st.download_button("CSV 다운로드", buf.getvalue(), file_name=f"{track_name}_lap.csv", mime="text/csv")

# ---- comparison ----------------------------------------------------------------
with tab_cmp:
    st.markdown("기준 차량 대비 공력 변경의 효과. 장치별 % 변경 + (선택) 밸런스 재트림.")
    c1, c2 = st.columns([2, 1])
    with c1:
        chg = st.data_editor(pd.DataFrame({"device": list(veh.aero.devices), "ΔClA %": 0.0, "ΔCdA %": 0.0}),
                             disabled=["device"], width="stretch", key="cmp_edit")
    with c2:
        all_cl = st.number_input("전체 ΔClA %", value=0.0, step=1.0)
        all_cd = st.number_input("전체 ΔCdA %", value=0.0, step=1.0)
        retrim = st.checkbox("밸런스 재트림")
        bal_t = st.slider("목표 밸런스 [% F]", 30.0, 70.0, float(round(veh.aero.balance_front * 100, 1)), 0.5,
                          disabled=not retrim)
    changes = {}
    if all_cl or all_cd:
        changes[None] = (all_cl / 100, all_cd / 100)
    for _, r in chg.iterrows():
        if r["ΔClA %"] or r["ΔCdA %"]:
            changes[r["device"]] = (r["ΔClA %"] / 100, r["ΔCdA %"] / 100)
    try:
        mod = study.modify(veh, changes, bal_t / 100 if retrim else None)
    except Exception as e:  # noqa: BLE001
        st.error(str(e))
        mod = None
    if mod is not None:
        mod_json = car_json_of(vehicle_to_dict(mod))
        _, res_m, ch_m = run_cached(mod_json, track_key, track)
        cruns = {"baseline": (veh, res, ch), "modified": (mod, res_m, ch_m)}
        d = res_m.lap_time - res.lap_time
        m = st.columns(4)
        m[0].metric("기준", f"{res.lap_time:.3f} s")
        m[1].metric("변경", f"{res_m.lap_time:.3f} s", f"{d * 1e3:+.1f} ms", delta_color="inverse")
        m[2].metric("ClA / CdA", f"{mod.aero.ClA:.2f} / {mod.aero.CdA:.2f}",
                    f"{mod.aero.ClA - veh.aero.ClA:+.2f} / {mod.aero.CdA - veh.aero.CdA:+.2f}",
                    delta_color="off")
        m[3].metric("밸런스", f"{mod.aero.balance_front * 100:.1f} % F",
                    f"{(mod.aero.balance_front - veh.aero.balance_front) * 100:+.1f}", delta_color="off")
        st.plotly_chart(figures.speed(cruns), width="stretch")
        st.plotly_chart(figures.time_delta(cruns), width="stretch")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(figures.gg(cruns), width="stretch")
        with c2:
            st.plotly_chart(figures.lines(cruns, ["ride_height_front_mm"], "Front ride height", "mm"),
                            width="stretch")

# ---- coefficients ----------------------------------------------------------------
with tab_coef:
    if st.button("랩타임 계수 계산", type="primary"):
        with st.spinner("계산 중…"):
            c = study.coefficients(veh, track)
            dc = study.device_coefficients(veh, track)
        st.session_state["coef"] = (car_json, track_key, c, dc)
    if "coef" in st.session_state and st.session_state["coef"][:2] == (car_json, track_key):
        _, _, c, dc = st.session_state["coef"]
        m = st.columns(4)
        m[0].metric("+1 % ClA", f"{c.dt_pct_cl * 1e3:+.1f} ms")
        m[1].metric("+1 % CdA", f"{c.dt_pct_cd * 1e3:+.1f} ms")
        m[2].metric("교환비", f"드래그 1 % = 다운포스 {c.exchange_rate:.2f} %")
        m[3].metric("랩타임 중립 L/D", f"{c.efficiency_equivalent:.2f}")
        st.dataframe(pd.DataFrame([{"device": x.name, "+0.1 ClA [ms]": x.dt_dcla * 100,
                                    "+0.1 CdA [ms]": x.dt_dcda * 100,
                                    "넘어야 할 ΔClA/ΔCdA": x.break_even_ld} for x in dc]),
                     width="stretch")
    elif "coef" in st.session_state:
        st.info("차량/트랙이 바뀌었어요. 다시 계산하세요.")

# ---- targets ----------------------------------------------------------------------
with tab_tgt:
    st.markdown("장치별 다운포스–드래그 폴라(`CdA = CdA0 + k·ClA²`)와 `ClA_min/max`, `mass_per_cla` 범위 안에서 "
                "랩타임이 최소가 되는 장치별 목표치를 찾아요.")
    c1, c2 = st.columns(2)
    with c1:
        use_win = st.checkbox("밸런스 창 제한", value=True)
        win = st.slider("허용 밸런스 [% F]", 30.0, 70.0, (45.0, 49.0), 0.5, disabled=not use_win)
    with c2:
        tunable = [n for n, d in veh.aero.devices.items() if d.tunable]
        sel = st.multiselect("최적화할 장치", list(veh.aero.devices), default=tunable)
    if st.button("목표치 계산", type="primary"):
        with st.spinner("최적화 중… (랩 수십 회 계산)"):
            tr_ = targets.optimise_targets(veh, track, sel or None,
                                           (win[0] / 100, win[1] / 100) if use_win else None)
        st.session_state["tgt"] = (car_json, track_key, tr_)
    if "tgt" in st.session_state and st.session_state["tgt"][:2] == (car_json, track_key):
        tr_ = st.session_state["tgt"][2]
        m = st.columns(3)
        m[0].metric("랩타임", f"{tr_.target_time:.3f} s", f"{tr_.gain * 1e3:+.1f} ms", delta_color="inverse")
        m[1].metric("밸런스", f"{tr_.target.aero.balance_front * 100:.1f} % F")
        m[2].metric("차량 질량", f"{tr_.target.mass:.1f} kg", f"{tr_.target.mass - veh.mass:+.1f}",
                    delta_color="off")
        st.code(tr_.summary(), language=None)
        res_t = simulate(tr_.target, track)
        st.code(targets.loads_table(targets.device_loads(tr_.target, res_t), tr_.target), language=None)
        st.download_button("목표 차량 JSON 저장",
                           json.dumps(vehicle_to_dict(tr_.target), indent=2, ensure_ascii=False),
                           file_name="target_car.json", mime="application/json")

# ---- sweeps -------------------------------------------------------------------------
with tab_sweep:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**ClA × CdA 스윕**")
        cl_rng = st.slider("ΔClA 범위 [%]", -40, 40, (-20, 20), 5)
        cd_rng = st.slider("ΔCdA 범위 [%]", -40, 40, (-20, 20), 5)
        step = st.select_slider("간격 [%]", [2.5, 5.0, 10.0], value=10.0)
        devs = st.multiselect("대상 장치 (비우면 전체)", list(veh.aero.devices))
        if st.button("스윕 실행"):
            d_cl = np.arange(cl_rng[0], cl_rng[1] + step / 2, step) / 100
            d_cd = np.arange(cd_rng[0], cd_rng[1] + step / 2, step) / 100
            with st.spinner(f"{len(d_cl) * len(d_cd)} 랩 계산 중…"):
                st.session_state["sweep"] = (car_json, track_key, d_cl, d_cd,
                                             study.sweep(veh, track, d_cl, d_cd, devs or None))
        if "sweep" in st.session_state and st.session_state["sweep"][:2] == (car_json, track_key):
            _, _, d_cl, d_cd, grid = st.session_state["sweep"]
            st.plotly_chart(figures.sweep(d_cl, d_cd, grid), width="stretch")
    with c2:
        st.markdown("**공력 밸런스 스윕** (총 ClA 고정, 앞·뒤 장치 재트림)")
        b_rng = st.slider("밸런스 범위 [% F]", 30.0, 70.0, (40.0, 60.0), 1.0)
        if st.button("밸런스 스윕 실행"):
            bals = np.arange(b_rng[0], b_rng[1] + 1e-9, 2.0) / 100
            with st.spinner("계산 중…"):
                st.session_state["bal"] = (car_json, track_key, bals, study.balance_sweep(veh, track, bals))
        if "bal" in st.session_state and st.session_state["bal"][:2] == (car_json, track_key):
            _, _, bals, times = st.session_state["bal"]
            st.plotly_chart(figures.balance(bals, times), width="stretch")
            i = int(np.nanargmin(times))
            st.success(f"최적 밸런스 ≈ {bals[i] * 100:.1f} % F ({times[i]:.3f} s)")

# ---- device loads -------------------------------------------------------------------
with tab_loads:
    sp_text = st.text_input("기준 속도 [km/h]", "40, 60, 80, 100")
    speeds = [float(x) for x in sp_text.split(",") if x.strip()]
    loads = targets.device_loads(veh, res, speeds)
    rows = []
    for l_ in loads:
        row = {"device": l_.name, "ClA": l_.ClA, "CdA": l_.CdA, "front %": l_.front_share * 100}
        for s_ in speeds:
            row[f"DF @{s_:g} [N]"] = l_.at_speed[s_][0]
            row[f"drag @{s_:g} [N]"] = l_.at_speed[s_][1]
        row.update({"lap mean DF [N]": l_.mean_downforce, "lap mean drag [N]": l_.mean_drag,
                    "peak DF [N]": l_.peak_downforce, "drag energy [Wh/lap]": l_.drag_energy / 3600})
        rows.append(row)
    st.dataframe(pd.DataFrame(rows).round(2), width="stretch")
    st.caption("정적 ClA/CdA 기준 (차고 맵에 따른 실제 하중은 ⏱ 랩 결과의 채널 참고)")
