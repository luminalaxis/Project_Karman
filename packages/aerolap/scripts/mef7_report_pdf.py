"""MEF7 (final) report PDF: aero numbers, event times and figures of the final package only.

Needs the outputs of aero_targets.py / aero_optimize.py --grid / aero_report_plots.py for the same
AERO_VEHICLE / AERO_TRACK / AERO_OUT (see docs/results_mef7_v6.md).

    python scripts/mef7_report_pdf.py      # -> out/<AERO_OUT>/MEF7_report.pdf
"""
import re

import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from aero_report_plots import FIG, OUT, RHO, TRACK, plt, save

MEF7_COLOR = "#eda100"
PARTS = {"프론트윙": (0.891, 0.238), "리어윙 (하단 윙 포함)": (0.935, 0.275), "차체·타이어": (-0.15, 0.193)}
CLA, CDA, CDA_PES, BAL = 1.676, 0.706, 0.721, 0.55
CDA_EXACT = 0.7056  # unrounded total of the MEF7 parts (nominal polar): OptimumLap cd input
SPEEDS = (40, 60, 80, 100)
PART_COLORS = ["#2a78d6", "#1baf7a", "#8a8984"]  # categorical slots 1, 3 + neutral for the bare car


def q(v_kmh):
    return 0.5 * RHO * (np.asarray(v_kmh, float) / 3.6) ** 2


def fig_forces():
    v = np.linspace(0, 110, 111)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for y, label, col, ls in ((q(v) * CLA, f"다운포스 (ClA {CLA:.2f})", MEF7_COLOR, "-"),
                              (q(v) * CLA * BAL, f"앞축 다운포스 ({BAL:.0%})", "#2a78d6", "--"),
                              (q(v) * CLA * (1 - BAL), f"뒤축 다운포스 ({1 - BAL:.0%})", "#1baf7a", "--"),
                              (q(v) * CDA, f"드래그 (CdA {CDA:.3f})", "#e34948", "-")):
        ax.plot(v, y, ls, color=col, lw=2, label=label)
    for s in SPEEDS:
        for y, col in ((q(s) * CLA, MEF7_COLOR), (q(s) * CDA, "#e34948")):
            ax.plot(s, y, "o", ms=7, color=col, mec="#fcfcfb", mew=2)
    ax.annotate(f"{q(80) * CLA:.0f} N", (80, q(80) * CLA), (6, 6), textcoords="offset points", fontsize=9)
    ax.annotate(f"{q(80) * CDA:.0f} N", (80, q(80) * CDA), (6, -14), textcoords="offset points", fontsize=9)
    ax.set_xlabel("속도 [km/h]")
    ax.set_ylabel("힘 [N]")
    ax.set_title("MEF7 (final) 속도별 다운포스 · 드래그", loc="left", fontsize=12)
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "mef7/forces_vs_speed.png")


def fig_parts():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    x = np.arange(len(SPEEDS))
    w = 0.26
    for ax, k, title in ((axes[0], 0, "파츠별 다운포스 [N]"), (axes[1], 1, "파츠별 드래그 [N]")):
        for i, ((name, c), col) in enumerate(zip(PARTS.items(), PART_COLORS)):
            vals = q(SPEEDS) * c[k]
            ax.bar(x + (i - 1) * w, vals, w * 0.92, color=col, label=name)
        ax.axhline(0, color="#8a8984", lw=1)
        ax.set_xticks(x, [f"{s} km/h" for s in SPEEDS])
        ax.set_title(title, loc="left", fontsize=12)
        ax.grid(axis="x", visible=False)
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    save(fig, "mef7/parts_by_speed.png")


def fig_lap():
    d = np.genfromtxt(OUT / "recommended" / "trace_endurance.csv", delimiter=",", names=True)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True)
    a1.plot(d["s_m"], d["v_kmh"], color=MEF7_COLOR, lw=1.6)
    a1.set_ylabel("속도 [km/h]")
    a1.set_title("MEF7 (final) 내구 1랩: 속도와 다운포스·드래그 (배터리 한계 출력 제한 포함)", loc="left", fontsize=12)
    a2.plot(d["s_m"], d["downforce_N"], color=MEF7_COLOR, lw=1.4, label="다운포스")
    a2.plot(d["s_m"], d["drag_N"], color="#e34948", lw=1.4, label="드래그")
    a2.set_ylabel("힘 [N]")
    a2.set_xlabel("거리 [m]")
    a2.legend(frameon=False, ncol=2)
    fig.tight_layout()
    save(fig, "mef7/endurance_lap.png")


def events():
    txt = (OUT / "recommended" / "events.txt").read_text(encoding="utf-8")
    t = {k: float(re.search(rf"^{k}\s+([\d.]+)", txt, re.M).group(1))
         for k in ("acceleration", "skidpad", "autocross", "endurance")}
    t["lap"] = float(re.search(r"(\d+) flying laps x ([\d.]+) s", txt).group(2))
    run = (FIG / "aerolap" / "MEF7" / "lap" / "summary.txt").read_text(encoding="utf-8")
    t["flying"] = float(re.search(r"lap_time_s\s+([\d.]+)", run).group(1))
    t["cap"] = float(re.search(r"power cap ([\d.]+) kW", txt).group(1))
    l0 = (OUT / "optimumlap_L0.txt").read_text(encoding="utf-8")
    t["L0"] = float(re.search(r"MEF7 \(final\).*?([\d.]+)\s*$", l0, re.M).group(1))
    track = re.search(r"\(([^()]*(?:\([^()]*\))?[^()]*)\); battery", txt).group(1)
    return t, track


def build_pdf(t, track):
    pdfmetrics.registerFont(TTFont("Malgun", "C:/Windows/Fonts/malgun.ttf"))
    pdfmetrics.registerFont(TTFont("MalgunB", "C:/Windows/Fonts/malgunbd.ttf"))
    H1 = ParagraphStyle("h1", fontName="MalgunB", fontSize=17, leading=22, spaceAfter=4)
    H2 = ParagraphStyle("h2", fontName="MalgunB", fontSize=12.5, leading=17, spaceBefore=10, spaceAfter=5,
                        textColor=colors.HexColor("#0b0b0b"))
    P = ParagraphStyle("p", fontName="Malgun", fontSize=9.3, leading=14, alignment=TA_LEFT,
                       textColor=colors.HexColor("#2b2b29"))
    S = ParagraphStyle("s", parent=P, fontSize=8.2, leading=12, textColor=colors.HexColor("#52514e"))

    def table(rows, widths, bold_last=False):
        tb = Table([[Paragraph(str(c), P) for c in r] for r in rows], colWidths=[w * mm for w in widths])
        st = [("FONTNAME", (0, 0), (-1, -1), "Malgun"),
              ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0efec")),
              ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor("#8a8984")),
              ("LINEBELOW", (0, 1), (-1, -1), 0.3, colors.HexColor("#e4e3df")),
              ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
              ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
        if bold_last:
            st.append(("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#fdf3dc")))
        tb.setStyle(TableStyle(st))
        return tb

    def img(path, width_mm):
        from reportlab.lib.utils import ImageReader
        iw, ih = ImageReader(str(path)).getSize()
        return Image(str(path), width=width_mm * mm, height=width_mm * mm * ih / iw)

    b = lambda s: f"<font name='MalgunB'>{s}</font>"  # noqa: E731
    story = [
        Paragraph("MEF7 (final) 공력 패키지 결과", H1),
        Paragraph(f"차량: MIP 228MV FD4 (OptimumLap 파일 기준) · 트랙: {track} · AeroLap L4 모델", S),
        Paragraph("공력 외 파라미터(질량·배터리·출력·서스펜션·타이어)는 OptimumLap 설계 파일 / fsk2026.yaml 값 그대로입니다.", S),
        Spacer(1, 6),
        Paragraph("1. 공력 계수", H2),
        table([["항목", "값"],
               ["ClA (다운포스 면적)", b(f"{CLA:.2f} m²")],
               ["CdA (항력 면적)", b(f"{CDA:.3f} m²") + f" (보수적 폴라 {CDA_PES:.3f})"],
               ["L/D", f"{CLA / CDA:.2f}"],
               ["공력 밸런스 (앞축)", f"{BAL:.2f}"],
               ["OptimumLap 입력 (전면 면적 0.635 m²)", f"cl {CLA / 0.635:.3f} / cd {CDA_EXACT / 0.635:.3f}"]], [70, 100]),
        Paragraph("2. 속도별 다운포스 · 드래그", H2),
        table([["속도", "다운포스", "드래그", "앞축 / 뒤축 다운포스"]]
              + [[f"{s} km/h", f"{q(s) * CLA:.0f} N", f"{q(s) * CDA:.0f} N (~{q(s) * CDA_PES:.0f})",
                  f"{q(s) * CLA * BAL:.0f} / {q(s) * CLA * (1 - BAL):.0f} N"] for s in SPEEDS], [30, 40, 50, 50]),
        Paragraph("F = ½ρv²·ClA, D = ½ρv²·CdA, ρ = 1.225 kg/m³. 드래그 괄호 값은 보수적 폴라(CdA 0.721) 기준.", S),
        Paragraph("3. 파츠별 다운포스 / 드래그", H2),
        table([["파츠 (ClA / CdA)"] + [f"{s} km/h" for s in SPEEDS]]
              + [[f"{n} ({c[0]:.2f} / {c[1]:.3f})"] + [f"{q(s) * c[0]:.0f} / {q(s) * c[1]:.0f} N" for s in SPEEDS]
                 for n, c in PARTS.items()]
              + [[b(f"합계 ({CLA:.2f} / {CDA:.3f})")] + [b(f"{q(s) * CLA:.0f} / {q(s) * CDA:.0f} N") for s in SPEEDS]],
              [55, 29, 29, 29, 29], bold_last=True),
        Paragraph("파츠별 값은 반올림 때문에 합계와 1 N 정도 차이가 날 수 있습니다. 차체·타이어의 음수 다운포스는 양력입니다.", S),
        Paragraph("4. 랩타임 (MEF7 final)", H2),
        table([["종목", "시간", "비고"],
               ["가속 75 m", f"{t['acceleration']:.3f} s", "정지 출발"],
               ["스키드패드", f"{t['skidpad']:.3f} s", "R 9.125 m 1바퀴"],
               ["오토크로스", f"{t['autocross']:.3f} s", "정지 출발 1랩"],
               ["내구 22 km", b(f"{t['endurance']:.1f} s"),
                f"26랩 × {t['lap']:.3f} s, 배터리 7.48 kWh 한계 → 출력 상한 {t['cap']:.1f} kW"],
               ["플라잉 1랩 (출력 제한 없음)", f"{t['flying']:.3f} s", "AeroLap L4, 배터리 한계 미적용 (lap.png)"],
               ["OptimumLap 동등 (L0) 1랩", f"{t['L0']:.3f} s", "점질량 모델, 밸런스·배터리 없음"]], [45, 30, 95]),
        PageBreak(),
        Paragraph("5. 그래프", H2),
        img(FIG / "mef7" / "forces_vs_speed.png", 160),
        Spacer(1, 6),
        img(FIG / "mef7" / "parts_by_speed.png", 170),
        Spacer(1, 6),
        img(FIG / "mef7" / "endurance_lap.png", 170),
        PageBreak(),
        Paragraph("AeroLap 1랩 결과 (aerolap run)", H2),
        img(FIG / "aerolap" / "MEF7" / "lap" / "lap.png", 128),
        img(FIG / "aerolap" / "MEF7" / "lap" / "track_map.png", 105),
        PageBreak(),
        Paragraph("교환비 (aerolap aero-exchange, 배터리 한계 내구)", H2),
        img(FIG / "aerolap" / "MEF7" / "exchange" / "aero_exchange.png", 150),
        Paragraph("ClA +0.1 → -1.42 s, CdA +0.1 → +2.67 s, 밸런스 +1 % 앞 → -0.40 s, 질량 +1 kg → +0.69 s. "
                  "추가 다운포스의 L/D가 1.88보다 좋아야 내구에서 이득입니다.", S),
        Paragraph("값을 정한 근거: 밸런스 × 받음각 그리드", H2),
        img(FIG / "balance_aoa_grid.png", 125),
        Paragraph("받음각 배율 1.10–1.15에서 기본·보수적 폴라 모두 이득이고, 1.20부터는 보수적 폴라에서 실속으로 손해. "
                  "밸런스는 0.55가 상한(0.60 이상은 언더스티어 지수 1.04 이하로 중립에 가까워짐) → 받음각 배율 1.10, 밸런스 0.55.", S),
        Paragraph("6. 가정과 한계", H2),
        Paragraph("• 파츠 폴라(받음각에 따른 ClA·CdA)는 가정값입니다(기본 p = 0.35/0.60, 실속 배율 1.3). CFD 2–3점으로 바꾸면 신뢰도가 올라갑니다.<br/>"
                  "• 밸런스 0.55는 정적 무게배분(앞 50 %)보다 앞입니다. 준정적 모델이라 과도 안정성은 실차 테스트로 확인해야 합니다.<br/>"
                  "• 무게중심·서스펜션·배터리(132S4P)는 fsk2026.yaml의 가정값입니다. 하단 윙 질량 0.6 kg 포함 차량 질량 310.6 kg.", P),
    ]

    def footer(c, doc):
        c.setFont("Malgun", 7.5)
        c.setFillColor(colors.HexColor("#8a8984"))
        c.drawString(18 * mm, 10 * mm, "MEF7 (final) 공력 패키지 · AeroLap")
        c.drawRightString(A4[0] - 18 * mm, 10 * mm, f"{doc.page}")

    path = OUT / "MEF7_report.pdf"
    SimpleDocTemplate(str(path), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm,
                      bottomMargin=16 * mm, title="MEF7 (final) 공력 패키지 결과").build(story, onFirstPage=footer,
                                                                                    onLaterPages=footer)
    return path


if __name__ == "__main__":
    (FIG / "mef7").mkdir(parents=True, exist_ok=True)
    fig_forces()
    fig_parts()
    fig_lap()
    t, track = events()
    print(t, track)
    print("written:", build_pdf(t, track))
