# FSK 랩타임 시뮬레이터 (Quasi-Steady-State)

차량 파라미터(질량·타이어·파워)와 공력(ClA·CdA·공력 밸런스)을 넣고 FSK 코스 랩타임을 계산하는 도구.
**"다운포스 +10%, 드래그 +5% 면 랩타임이 어떻게 되나?"** 를 숫자로 답하는 게 목적.

```bash
pip install -r requirements.txt          # numpy (필수), matplotlib (플롯), pytest

python -m lapsim gui                                          # GUI (브라우저에서 열림, streamlit)
python -m lapsim run    --plot lap.png --export lap.csv       # 기본: tracks/fsk_2026_endurance.json
python -m lapsim run    --line best                           # 주행 라인 (최소곡률 / 절충 중 빠른 것)
python -m lapsim delta  --dcl 10 --dcd 5                      # 전체 공력 ClA +10 %, CdA +5 %
python -m lapsim delta  --dev rear:cl=10,cd=5 --dev front:cl=5   # 장치별 변경 (조합 가능)
python -m lapsim delta  --dev rear:cl=10 --bal 48             # 변경 후 밸런스 48 % F 로 재트림
python -m lapsim coeffs                                       # 랩타임 계수: 전체 + 장치별
python -m lapsim sweep  --cl=-20:20:5 --cd=-20:20:5 --device rear --plot sweep.png
python -m lapsim balance --range 40:60:2.5                    # 공력 밸런스 스윕 (총 ClA 고정)
python -m lapsim forces --speeds 40,60,80,100                 # 장치별 다운포스·드래그 [N], 드래그 에너지
python -m lapsim targets --bal-range 45:49 --save cars/target.json   # 장치별 성능 목표치 자동 계산
python -m lapsim run --set aero.rear.ClA=1.6 --set mass=270   # 파라미터 즉석 변경
python -m lapsim run --track tracks/sample_autocross.json     # 합성 오토크로스 (정지 출발)
```
공통 옵션: `--car` (차량 JSON), `--track` (트랙 JSON), `--ds` (격자 간격, 기본 0.5 m),
`--smooth` (센터라인 스무딩 창, 기본 4 m), `--flying` (열린 코스를 플라잉 스타트로),
`--line center|compromise|mincurv|best` (주행 경로), `--car-width`, `--line-margin`, `--track-width`,
`--plot PNG` (리포트 그래프), `--export CSV` (전체 채널 데이터).

## GUI (`python -m lapsim gui`)
Streamlit 웹 앱. 사이드바에서 차량/트랙 파일 선택·업로드, 주행 경로 선택.
| 탭 | 내용 |
|---|---|
| 🚗 차량 설정 | 질량·타이어·파워트레인·서스펜션 입력, 공력 장치 표 편집(행 추가/삭제), 편집한 차량 JSON 저장 |
| ⏱ 랩 결과 | 랩타임·에너지 등 요약, 속도/가속도/g-g/휠 하중/차고/실시간 공력 밸런스/축 이용률/배터리 출력/장치별 공력 그래프, 트랙 맵(채널 색상 선택), 채널 표 + CSV 다운로드 |
| Δ 비교 | 장치별·전체 ΔClA/ΔCdA, 밸런스 재트림 → 랩타임 차, 속도 겹침, 누적 시간차 그래프 |
| 📈 공력 계수 | 1 % ClA/CdA 당 ms, 교환비, 장치별 0.1 ClA/CdA 당 ms 와 넘어야 할 L/D |
| 🎯 목표치 | 밸런스 창·대상 장치 선택 → 장치별 목표 ClA/CdA, 하중표, 목표 차량 JSON 저장 |
| 🗺 스윕 | ClA×CdA 랩타임 등고선, 공력 밸런스 스윕 |
| 📋 하중표 | 장치별 속도별 다운포스/드래그 [N], 랩 평균·최대, 드래그 에너지 |
음수로 시작하는 범위는 `--cl=-20:20:5` 처럼 `=` 를 붙인다.

## 공력 장치 (front / mid / rear)
차량 JSON 의 `aero.devices` 에 장치를 이름별로 넣는다. 각 장치는 자기 ClA·CdA·압력중심 위치를 가지며,
전체 ClA·CdA·밸런스·드래그 높이는 장치 합으로 자동 계산된다.

```jsonc
"aero": {
  "rho": 1.2,
  "devices": {
    "front": {"ClA": 1.20, "CdA": 0.33, "x_cp": -0.50, "h_cp": 0.25},   // 프론트 윙 (앞차축 앞)
    "mid":   {"ClA": 1.05, "CdA": 0.15, "x_cp":  0.85, "h_cp": 0.10},   // 언더트레이/디퓨저/사이드
    "rear":  {"ClA": 1.40, "CdA": 0.52, "x_cp":  1.90, "h_cp": 0.95},   // 리어 윙 (뒤차축 뒤)
    "body":  {"ClA": 0.00, "CdA": 0.32, "x_cp":  0.75, "h_cp": 0.45}    // 차체·드라이버·휠 드래그
  }
}
```
* `x_cp` : 앞차축 기준 압력중심 위치 [m] (음수 = 앞차축 앞). 앞축 분담률 = 1 − x_cp/L.
  프론트 윙은 100 % 넘게, 리어 윙은 음수(앞축 하중을 뺌)가 되는 게 정상. 대신 `front_share` 로 직접 줘도 된다.
* `h_cp` : 그 장치 드래그의 작용 높이 → 높은 리어 윙 드래그는 앞축 하중을 덜어내는 피칭 모멘트를 만든다.
* (선택) `ClA_min`, `ClA_max`, `CdA0`, `k_induced`, `induced_share`, `mass_per_cla` — 목표치 계산용 설계 범위 (아래 참고).
* 장치 이름은 자유 (`body` 는 공력 장치가 아닌 기본 드래그용). 옛 형식 `{"ClA", "CdA", "balance_front"}` 도 읽는다.
* `balance` / `--bal` 재트림은 가장 앞·가장 뒤 장치의 ClA 를 (CdA 도 같은 비율로) 조절해 총 ClA 를 유지한 채 목표 밸런스를 맞춘다 (윙 각도 조정에 해당).

## 출력 예 (기본 차량 × FSK 2026 Endurance, 850.8 m, 시계방향, 중심선)
```
lap time         :   56.911 s          (--line best: 54.092 s, 831.6 m)
max lat accel    :   2.34 g
corner limited by: front axle 6.10 s, rear axle 0.07 s

+1 % ClA             :    -28.7 ms
+1 % CdA             :     +8.2 ms
exchange rate        : +1 % drag is paid back by +0.29 % downforce

per device (absolute change on that device only; balance shifts with it):
  device       +0.1 ClA   +0.1 CdA  break-even L/D
  front        -160.3ms    +42.9ms            0.27
  mid           -77.3ms    +33.9ms            0.44
  rear           -9.9ms    +86.6ms            8.78
  body          -81.2ms    +55.5ms            0.68
```
읽는 법: 기본 차량은 코너에서 **앞축이 먼저 한계** → 프론트 다운포스 0.1 이 리어의 16배 가치.
리어 윙은 ΔClA/ΔCdA > 8.8 이어야 이득 (리어 다운포스는 밸런스를 뒤로 보내고, 높은 드래그는 앞축 하중을 더 덜어냄).
좌우 하중이동(LLTD 앞 53 %)과 요 관성이 들어가면서 앞축 의존도가 더 커졌다.

## 공력 장치 성능 목표치 자동 계산 (`forces`, `targets`)
**`forces`** — 현재 차량의 장치별 하중표: 기준 속도별 다운포스/드래그 [N], 랩 평균(시간 가중) 하중, 최대 다운포스,
드래그가 한 랩에 먹는 에너지 [Wh/lap] (EV 엔듀런스 에너지 예산용).

**`targets`** — 랩타임을 최소화하는 장치별 ClA 를 찾아 목표치를 낸다.
* 각 장치는 자기 폴라 `CdA = CdA0 + k_induced·ClA²` 위에서 움직인다 → 다운포스를 늘리면 유도항력이 제곱으로 늘어남.
  `CdA0`/`k_induced` 를 안 주면 현재 점을 지나고 드래그의 `induced_share`(기본 50 %)가 유도항력이라고 가정.
* `ClA_min`/`ClA_max` = 패키징·규정 범위 (없으면 0 ~ 현재×2), `mass_per_cla` = ClA 1.0 당 구조물 질량 [kg].
* `--bal-range 45:49` = 허용 공력 밸런스 창 (QSS 는 안정성 마진을 모르므로 지정 권장). `--device rear` 로 일부 장치만 최적화.
* 탐색은 패턴 서치(무기울기, numpy만) — FSK 코스 기준 약 60~70 랩 계산, 수 초.
* `--save` 로 목표 차량 JSON 저장 → `run/delta/coeffs --car` 로 바로 재사용.

출력 (기본 차량, FSK 2026, 밸런스 45–49 %):
```
  device    ClA now  target      Δ  CdA now  target   L/D    F %  +0.1ClA  +0.1CdA  min L/D  note
  front        1.20    1.30  +0.10     0.33    0.36  3.62  132.7   -123ms    +42ms     0.34  set by balance window
  mid          1.05    1.60  +0.55     0.15    0.25  6.42   44.4    -67ms    +36ms     0.53  at ClA_max (package limit)
  rear         1.40    1.39  -0.01     0.52    0.52  2.69  -24.2    -24ms    +67ms     2.83  set by balance window
  total        3.65    4.29  +0.64     1.32    1.44  2.97   49.0
  lap time       : 56.911 s -> 56.450 s (-461.3 ms)
```
* `target` ClA/CdA = 장치 설계 목표. 아래 하중표에 같은 목표를 속도별 다운포스/드래그 [N] 로 환산해 준다 (풍동·CFD 목표값으로 사용).
* `+0.1ClA`/`+0.1CdA` = 목표점에서 그 장치의 한계 가치, `min L/D` = 그 장치를 더 바꿀 때 ΔClA/ΔCdA 가 넘어야 하는 값.
* `note` 가 `at ClA_max` 면 패키징이 병목 (상한을 풀면 더 빨라짐), `optimum` 이면 내부 최적점, `set by balance window` 면 밸런스가 병목.
* 밸런스 창 상한(49 %)에서 프론트·리어가 묶이고, 언더트레이(mid)는 패키징 상한까지 — 밸런스 영향이 작은 mid 가 가장 싸게 다운포스를 더한다.

## 모델
| 항목 | 내용 |
|---|---|
| 공력 | 장치별 F_down = ½ρ·ClA_i·f_cl(Δh_f, Δh_r)·v², F_drag 동일, 앞축 몫 = 1 − x_cp,i/L, 장치별 드래그 피칭 모멘트(h_cp,i) 합산. f = 차고 다항식 맵 |
| 차고 | h = h_0 − ΔFz_axle / 히브 강성, 범프스톱 하한. 공력 ↔ 차고 고정점 반복 |
| 축하중 | 정적 배분 + 공력 + 종방향 하중이동(m·a_x·h/L + 드래그 모멘트) — 앞/뒤 축 분리 |
| 좌우 하중이동 | 바퀴당 ΔFz = LLTD_axle·m·a_y·h / 트레드_axle, LLTD = 롤 강성 배분 (안쪽 바퀴 들림까지) |
| 타이어 | 바퀴별 하중 민감 μ(Fz) = μ0·(1 − k·(Fz−Fz0)/Fz0) → 하중이동이 많은 축이 그립을 더 잃음, 축별 마찰 타원 (μx, μy) |
| 요 관성 | ṙ ≈ v²·dκ/ds, 앞축 횡력 = (m·a_y·b + Iz·ṙ)/L, 뒤축 = (m·a_y·a − Iz·ṙ)/L → 진입은 앞축, 탈출은 뒤축 부담 증가 |
| 코너 한계속도 | 위 횡력이 각 축 한계 이내인 최대 v (벡터화 이분법). 어느 축이 먼저 포화되는지 기록 |
| 가속 | min(모터 토크, 80 kW, 후륜 잔여 그립) − 드래그 − 구름저항, 하중이 빠지는 앞축 횡그립 한계로 제한 |
| 제동 | 4륜, 축별 잔여 그립 (이상적 배분) + 드래그, 하중이 빠지는 뒤축 횡그립 한계로 제한. a_x ↔ 하중이동은 할선법으로 수렴 |
| 랩 | v = min(전진, 후진). 닫힌 코스는 가장 느린 정점에서 시작하는 플라잉 랩, 열린 코스는 정지 출발 |
| 주행 라인 | `mincurv`: 정확한 3점 곡률의 ∫κ² ds 최소화 (가우스-뉴턴 + 박스 QP), `compromise`: 기준 격자 2차 차분 최소화 (곡률과 경로 길이 절충), `best`: 둘 다 돌려 빠른 것 |

남은 단순화 (QSS 특성상): 댐퍼·서스펜션 과도 응답 없음 (하중이동은 정상상태), 요 관성은 곡률 변화율 기반 준정적 근사,
주행 라인은 최소 시간 최적해가 아님 (최소 곡률/절충), 타이어는 마찰 타원 (슬립·온도 모델 없음), 배터리 전압 강하·열 디레이팅 없음.
공력 밸런스 최적점은 "두 축 동시 포화"점이라 실제 차는 안정성 마진만큼 뒤로 둔다.
센터라인 스무딩(`--smooth`)이 크면 헤어핀이 펴져 랩타임이 크게 빨라지므로(FSK: 4 m 56.9 s, 8 m 53.6 s) 주행 라인 옵션으로 비교할 것.

## 파일
```
lapsim/vehicle.py   차량·타이어·공력 장치(AeroDevice)·EV 파워트레인 (Vehicle, load_vehicle)
lapsim/track.py     트랙 JSON 로더 → 등간격 곡률 프로파일
lapsim/physics.py   공통 물리 코어 (공력·차고·축하중·바퀴별 그립·요 관성)
lapsim/solver.py    QSS 솔버 (simulate → LapResult)
lapsim/line.py      주행 라인 (mincurv / compromise / best)
lapsim/channels.py  랩 채널 계산, CSV 내보내기, 누적 시간차
lapsim/figures.py   GUI 용 plotly 그래프
app.py              Streamlit GUI
lapsim/study.py     modify / delta / sweep / balance_sweep / coefficients / device_coefficients
lapsim/targets.py   장치별 하중표(device_loads) / 목표치 최적화(optimise_targets)
lapsim/plot.py      CLI 리포트 그래프 (속도, 시간차, 가속도, g-g, 맵, 차고, 이용률), 스윕, 밸런스 (matplotlib)
cars/fsk_ev_baseline.json   기본 FS EV (280 kg, front/mid/rear/body, ClA 3.65, CdA 1.32, 47 % F, 80 kW)
tracks/fsk_2026_endurance.json  FSK 2026 엔듀런스 (cone2track export, 850.8 m, 시계방향)
tracks/sample_autocross.json  테스트용 합성 오토크로스 (실제 FSK 코스 아님)
```

## 트랙 JSON 형식 (자동 감지)
```jsonc
{"centerline": {"closed": true, "points": [{"x":, "y":}]}, "meta": {"run": "clockwise"}}  // cone2track export
{"cones": [{"side": "left", "lat":, "lng":}, ...]}       // 측량 콘만 있을 때 (centerline 없으면 대체)
{"x": [...], "y": [...], "closed": true}                 // 센터라인 좌표 [m]
{"centerline": [[x, y], ...]}                            // 동일 (또는 [{"x":..,"y":..}])
{"left": [[x, y], ...], "right": [[x, y], ...]}          // 콘 좌표 (blue/yellow, inner/outer 도 가능)
{"segments": [{"type": "straight", "length": 60},
              {"type": "corner", "radius": 9, "angle": 180, "direction": "left"}]}
{"s": [...], "curvature": [...]}                         // 거리-곡률
```
점/콘 형식은 기본 `closed: true` (엔듀런스), segments/곡률 형식은 기본 `closed: false` (오토크로스).
주행 방향: `meta.run` / `direction` (`clockwise`·`counterclockwise`) 또는 `reverse: true` 로 점 순서를 맞춘다 (x 동쪽, y 북쪽).

## 차량 JSON
`cars/fsk_ev_baseline.json` 참고. 단위는 SI (kg, m, N, W, rpm), 차고만 mm. `ClA` 는 다운포스 양수.
```jsonc
"yaw_inertia": 100.0,                       // Iz [kg m²], 0 = 요 관성 무시
"suspension": {                             // 없으면 좌우 하중이동 없음, 차고 고정
  "track_front": 1.22, "track_rear": 1.18,  // 트레드 [m]
  "roll_stiffness_front": 450.0, "roll_stiffness_rear": 400.0,   // [Nm/deg] → LLTD (또는 "lltd_front": 0.53)
  "ride_height_front": 30.0, "ride_height_rear": 35.0,           // 정적 차고 [mm]
  "heave_rate_front": 120.0, "heave_rate_rear": 140.0,           // 축 히브 강성 [N/mm], 0 = 차고 고정
  "min_ride_height": 8.0                                         // 범프스톱 [mm]
}
// 장치별 차고 맵: f = 1 + c0·Δh_f + c1·Δh_r + c2·Δh_f² + c3·Δh_r² + c4·Δh_f·Δh_r  (Δh [mm], 정적 차고 기준)
"mid": {..., "rh_cl": [-0.008, -0.004, -0.0002], "rh_cd": [-0.004, -0.002]}
```

## 테스트
```bash
python -m pytest -q
```
해석해 검증: 무공력 원 주행 v = √(μgR), 드래그-파워 평형 최고속, 모터 rpm 제한, ClA↑/CdA↑ 단조성,
밸런스 내부 최적점, 격자 수렴(0.5 m vs 0.25 m < 0.5 %), 트랙 JSON 형식별 곡률,
장치 합산·밸런스 재트림·옛 형식 호환, FSK 코스 길이/방향.
