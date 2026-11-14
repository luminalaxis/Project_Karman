# fsk-lapsim — FSK 랩타임 시뮬레이터 + 공력 민감도 분석 도구

공력 설계안 (ClA, CdA, 공력 밸런스, 부품 질량)을 **랩타임과 에너지**로 평가하는 준정상(QSS) 시뮬레이터.
Formula Student Korea 전기차 (후륜 단일 모터) 용.

> **모든 기본 수치는 가정값이다.** 차량 파라미터·트랙·리어윙 후보(DUMMY)는 실측이 아니다.
> `fsk-lapsim assumptions` 로 목록을 보고, 실측값이 생기면 YAML 에 숫자로 넣는다. 결과는 **설계안 사이의
> 차이와 순위**를 보는 용도로 쓰고, 절대값은 실측 우선순위 상위 항목 (타이어 μ0, 하중민감도, 질량 …) 이
> 정해질 때까지 믿지 않는다.

## 설치

```bash
cd fsk-lapsim
python -m venv .venv && . .venv/bin/activate       # Python 3.11+
pip install -e ".[dev]"                             # 런타임: numpy scipy pandas matplotlib pyyaml pydantic typer
python -m pytest -q                                 # 테스트 (약 4 분)
```

## 빠른 시작

```bash
fsk-lapsim run                        # 이벤트 4종 (가속·스키드패드·오토크로스·엔듀런스)
fsk-lapsim report -o out/report.html  # 전체 분석 HTML 리포트 (그래프·표·가정 목록 내장, 약 1–2 분)
fsk-lapsim compare --candidates my_wings.csv --plot out/compare.png   # 리어윙 후보 비교
fsk-lapsim init-configs my_cfg        # 예시 설정·데이터를 복사해서 편집
fsk-lapsim run -v my_cfg/configs/vehicle.yaml -e my_cfg/configs/events.yaml
```

`python -m fsk_lapsim ...` 도 같다. 모든 명령은 `--vehicle/-v`, `--events/-e`, `--model/-m (axle|point_mass)`, `--ds` 를 받는다.

| 명령 | 내용 |
|---|---|
| `run` | 이벤트 결과 (가정값 기반 표시), `--track` 로 트랙 1 랩, `--channels` 채널 CSV, `--plot-dir` 그래프 |
| `sensitivity` | 중앙차분 민감도 (ClA, CdA, 밸런스, 질량, CG 높이, μ, 출력 상한; `--extra` 로 7 개 추가), 등가 환산표, 실측 우선순위 |
| `sweep` | ClA × CdA 격자 → 오토크로스 시간·엔듀런스 에너지 등고선 (`--cla 2.5:4.5:0.5 --cda 1.0:1.8:0.2`), 현재 차량·후보 표시 |
| `compare` | 리어윙 후보 CSV → 이벤트별 시간·Δ·순위 (`--reference`, `--retrim-front` 로 밸런스 유지 비교) |
| `balance-opt` | 랩타임 최소 공력 밸런스 (`--objective autocross|endurance_lap|combined`) |
| `active-aero` | DRS: 이벤트별 이득, 전환 횟수, 개방 비율, 구간별 닫힘 명령 리드 (m, s) |
| `report` | 단일 HTML 리포트 (`--quick` 빠른 버전, `--mc N` 몬테카를로 포함) |
| `assumptions` | `assumed: true` 파라미터 목록 |
| `track` | 트랙 요약·규칙 검사, `--generate autocross --seed 3` 생성, `--export-csv`, `--plot` |
| `wing2d3d` | 2D Cl/Cd → 3D ClA/CdA 대략 추정 |
| `montecarlo`, `gps`, `thermal`, `score` | 확장 기능 E1–E4 (추가되는 대로 아래 "확장 기능" 절에 설명, 진행은 PROGRESS.md) |

## 입력 파일

기본 파일은 패키지 안 `src/fsk_lapsim/configs/` 와 `src/fsk_lapsim/data/` 에 있다 (`init-configs` 로 복사).
상대 경로는 (1) 그 YAML 파일 위치 (2) 현재 디렉터리 (3) 패키지 configs/ (4) 패키지 data/ 순으로 찾는다.

### 수치 형식 (모든 YAML 공통)
```yaml
total_mass_kg: {value: 320.0, unit: kg, assumed: true, note: "드라이버 포함"}   # 가정값
total_mass_kg: 318.5                                                          # 숫자만 → 실측/확정값 (assumed: false)
```

### `vehicle.yaml`
`mass` (total_mass_kg, cg_height_m, static_front_weight_fraction), `geometry` (wheelbase_m, track_front_m,
track_rear_m, lltd_front), `tire` (radius_m, mu0, fz0_n, load_sensitivity, mu_lon_lat_ratio, rolling_resistance_coeff),
`powertrain` (max_power_w, motor_max_torque_nm, motor_max_rpm, gear_ratio, efficiency, regen_max_power_w,
regen_enabled, differential: lsd|open), `brakes` (distribution: ideal|fixed, front_bias),
`aero` (cla_m2, cda_m2, balance_front, air_density_kg_m3, components_file 선택).

### `aero_components.yaml`
부품별 `cla_m2, cda_m2, x_cp_m` (앞차축 기준 뒤쪽 +), `z_cp_m` (항력 높이), `mass_kg, x_mass_m, z_mass_m`.
기준 부품 질량은 vehicle.yaml 총질량에 포함된 것으로 본다. 합성 밸런스는 −CdA·z/L 항력 피칭을 포함한다 (MODEL.md 1.1).

### 리어윙 후보 CSV (ANSYS 2D 결과)
```csv
config_id,flap1_id,flap2_id,flap1_aoa_deg,flap2_aoa_deg,Cl_2d,Cd_2d,Cm_2d,cm_ref_x_m,chord_ref_m,span_m,mass_kg,notes
RW01,F1A,F2A,22,42,2.78,0.098,0.973,1.70,0.70,1.10,4.9,flap A/A
```
- 필수: `config_id, flap1_id, flap2_id, Cl_2d, Cd_2d, chord_ref_m, span_m` (Cl 은 다운포스 양수, chord_ref 기준 2D 계수)
- 선택: `Cm_2d` + `cm_ref_x_m`, `notes`, `mass_kg`, `x_cp_m`, `z_cp_m`, `x_mass_m`, `z_mass_m`, `ar_eff`, `k_lift`, `oswald_e`,
  **`cla_3d_m2`, `cda_3d_m2` (3D CFD 값 — 있으면 2D→3D 변환보다 우선)**
- 그 밖의 열 (플랩 각도, 갭, 위치 …) 은 자유 파라미터로 표에 그대로 나온다. `#` 주석 줄 허용.
- 비어 있는 질량·위치는 `aero_components.yaml` 의 rear_wing 값을 쓴다.
- 예시 `data/rear_wing_candidates_DUMMY.csv` 는 **DUMMY** (임의값) 이다.

### `events.yaml`
가속 길이 (75 m), 스키드패드 주행선 반경 (9.125 m), 오토크로스·엔듀런스 트랙 YAML, 오토크로스 출발 방식,
엔듀런스 총거리 (22 km), 선택적 `endurance_max_power_w` (에너지 관리용 출력 제한), `ds_m`.

### 트랙 YAML
```yaml
# (1) 생성기 (시드 고정): kind autocross = 열린 코스, endurance = Dubins 폐합 루프
type: generated
generator: {kind: endurance, seed: 7, target_length_m: 1000, rules: {max_straight_m: 80, hairpin_radius_m: [4.5, 7]}}
# (2) 세그먼트 (손 설계): straight / arc / slalom, 닫힌 코스는 closure: dubins | adjust_straights
type: segments
closure: adjust_straights      # 직선 두 개 길이를 풀어서 정확히 폐합 (총 회전 360° 필요)
adjust: [14, 16]
segments:
  - {type: straight, length_m: 50}
  - {type: arc, radius_m: 8, angle_deg: 120, direction: left}
  - {type: slalom, n_cones: 6, spacing_m: 10, weave_m: 2.0, direction: right}
# (3) (x, y) CSV 임포트: 스플라인 평활화 후 곡률 계산
type: csv
path: tracks/fsk2026_endurance_centerline.csv
closed: true
reverse: false
smoothing_sigma_m: 0.3         # 위치 노이즈 σ [m] (splprep s = n·σ²). 크면 코너가 펴져 빨라진다
```
트랙 기하는 **주행선**으로 해석된다 (레이싱 라인 최적화 없음). 기본 생성 규칙 (직선 ≤ 80 m, 정상선회 지름 ≤ 50 m,
헤어핀 주행선 반경 4.5–7 m, 슬라럼 7.5–12 m) 은 FS 일반 범위의 가정값이다 → FSK 규정으로 교체.

### `drs.yaml`, `uncertainty.yaml`, `thermal.yaml`, `scoring.yaml`
각 파일 주석 참고. 모두 가정값.

## 확장 기능

### E1 몬테카를로 (`fsk-lapsim montecarlo`)
```bash
fsk-lapsim montecarlo --n 200 --workers 4                 # 가정 범위 → 랩타임 분포, 후보 순위 역전 확률
fsk-lapsim montecarlo --n 200 --retrim-front --plot mc.png  # 후보마다 프론트윙 재트림 (밸런스 효과 제거)
fsk-lapsim report --mc 100                                 # 리포트에 몬테카를로 절 추가
```
범위는 `configs/uncertainty.yaml` (σ, 균등 범위, 클립). 출력: 기준 차량 랩타임 분포 (p5/p50/p95), 후보별 P(1위), 평균 순위,
P(명목 순위 유지), 가까운 후보 쌍의 순위 역전 확률, 전체 순위가 그대로일 확률.

### E2 GPS 로그 (`fsk-lapsim gps LOG.nmea`)
```bash
fsk-lapsim gps                                   # 내장 DUMMY 로그 (합성)
fsk-lapsim gps run1.nmea --lap 2 --plot gps.png --export-track run1_track.csv
```
NEO-6M 등 NMEA 0183 (RMC/GGA/VTG, 토커 GP/GN/GL) → 랩 분할 (첫 고정점 = 시작점, `--gate` 반경) → 트랙 재구성
(기본 `course`: 도플러 진행방향·속도로 곡률 κ = dψ/ds, `position`: 위치 스플라인) → 시뮬레이션 → 실측 속도와 비교
(랩타임 비, 속도 RMSE, 평균 차이). **수신기는 5–10 Hz 로 설정**할 것 (1 Hz 기본값은 헤어핀을 놓쳐 시뮬레이션이 7–14 % 빨라짐).
만든 트랙 CSV 는 트랙 YAML `type: csv` 로 다시 쓸 수 있다 (`smoothing_sigma_m` 작게).

### E3 발열 (`fsk-lapsim thermal`)
```bash
fsk-lapsim thermal --plot heat.png --map-plot motor_eff.png      # 계수: configs/thermal.yaml (가정값)
```
엔듀런스 운전점에서 모터 (k_cu·T² + k_fe·ω^1.5 + k_w·ω²), 인버터 (P0 + a·I + b·I², I = |T|/k_t), 팩 (I²R) 손실을 계산하고,
부품별 열용량·열저항 집중 모델로 온도를 적분한다. 출력: 총 발열 [kJ], 평균/최대 발열률 [kW], 종료·최고 온도, 정상상태 온도,
한계 온도를 지키는 데 필요한 열저항 (`required_r_k_per_w` < `assumed_r_k_per_w` 이면 냉각 강화 필요), 모터 효율 맵.

### E4 점수 환산 (`fsk-lapsim score`)
```bash
fsk-lapsim score                       # 기준 차량 이벤트 점수, 0.1 s 당 점수
fsk-lapsim score --with-candidates     # 리어윙 후보 점수 차이 (DUMMY)
```
`configs/scoring.yaml` 은 **자리표시자**: FS 계열 시간 비율 공식 score = a·((T_max/T)^p − 1)/((T_max/T_min)^p − 1) + b
(T_max = 계수 × T_min) 과 효율 계수 공식의 형태만 따랐고, 기준 기록 T_min 은 기준 차량 시뮬레이션 시간 × 비율이다.
FSK 규정 공식·만점·실제 우승 기록 (`t_min_s`) 으로 교체하면 시간 차이 → 점수 차이를 바로 읽을 수 있다. 리포트 11c 절에
민감도 (표시 단위당 점수)·후보 점수 차이가 들어간다.

### E5 성능
`sweep`, `sensitivity`, `compare`, `report` 는 기본으로 CPU 수만큼 (최대 8) 프로세스를 쓴다 (`--workers N`, 1 = 직렬).
결과는 직렬과 같다. 4 코어 기준: 25 점 스윕 0.75 s, 민감도 2.2 s, 전체 리포트 약 18 s, 몬테카를로 200 샘플 × 9 변형 약 90 s.

## 결과 읽는 법
- **민감도 표**: "표시 단위만큼 늘렸을 때 몇 초" (양수 = 느려짐). 엔듀런스는 총 시간.
- **등가 환산**: "+1 kg 과 같은 오토크로스 시간 손실을 내는 ClA/CdA 변화" — 무게와 공력의 교환 비율.
- **실측 우선순위**: |민감도| × 가정값 불확실성 σ. 위에 있는 가정이 틀리면 결과가 가장 크게 바뀐다.
- **리어윙 비교**: 후보 질량·높이 → 총질량·CG, 압력중심·항력 높이 → 밸런스가 자동 반영된다. `--retrim-front` 는
  프론트윙을 스케일해 기준 밸런스를 유지 (밸런스 효과와 효율 효과 분리).
- **DRS 리드**: 닫힘 명령은 제동 시작보다 전환 시간 T 만큼 먼저 (≈ v·T m) 나가야 한다. 페일세이프는 최고속 × T 를 기준으로.

## 모델 가정과 한계 (요약 — 자세히는 MODEL.md, 리포트 13절)
- 준정상: 요 관성·과도 응답·서스펜션 동역학 없음. 공력 밸런스 최적점은 두 축 동시 포화점 (안정성 마진 없음).
- 주행선 = 트랙 기하 (레이싱 라인 없음) → 헤어핀·슬라럼 보수적.
- 타이어: 선형 하중민감도 + 마찰 타원 실효 모델, 슬립·온도 없음. 공력 계수는 차고·자세 무관 상수.
- 파워트레인: 종합 효율 상수 (토크 경로에도 적용), 배터리 전압·열 디레이팅 없음.
- 2D→3D 변환은 대략적 추정 (유효 종횡비·k·e 가정에 민감) — 3D CFD 값이 있으면 직접 입력.

## 파일 구조
```
src/fsk_lapsim/
  config.py        YAML 스키마 (pydantic, Param{value, unit, assumed, note})
  vehicle.py tire.py aero.py powertrain.py   차량 데이터클래스, 부품 합성, 파라미터 변경
  models.py        모델 A (점질량), 모델 B (축 단위)
  ggv.py           GGV 표 + 빠른 보간
  solver.py energy.py   QSS 솔버, 에너지·모터 운전점
  track.py trackgen.py  트랙 (세그먼트·슬라럼·CSV), 생성기·Dubins·규칙 검사
  events.py        가속·스키드패드·오토크로스·엔듀런스
  wing.py          2D→3D, 리어윙 후보
  active_aero.py   DRS
  analysis/        sensitivity, sweep, compare, balance_opt, uncertainty, montecarlo
  gps.py thermal.py scoring.py   확장 기능 (E2–E4, 추가 예정)
  plots.py report.py cli.py
  configs/  data/  예시 설정과 데이터
tests/             검증 1–9 + 단위·CLI 테스트
```
문서: MODEL.md (수식), ASSUMPTIONS.md (가정값과 근거), DECISIONS.md (설계 결정), PROGRESS.md (진행·인계), FINAL_SUMMARY.md.
