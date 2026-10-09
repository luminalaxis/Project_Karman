# FSK 랩타임 시뮬레이터 (Quasi-Steady-State)

차량 파라미터(질량·타이어·파워)와 공력(ClA·CdA·공력 밸런스)을 넣고 FSK 코스 랩타임을 계산하는 도구.
**"다운포스 +10%, 드래그 +5% 면 랩타임이 어떻게 되나?"** 를 숫자로 답하는 게 목적.

```bash
pip install -r requirements.txt          # numpy (필수), matplotlib (플롯), pytest

python -m lapsim run    --plot lap.png                        # 기본: tracks/fsk_2026_endurance.json
python -m lapsim delta  --dcl 10 --dcd 5                      # 전체 공력 ClA +10 %, CdA +5 %
python -m lapsim delta  --dev rear:cl=10,cd=5 --dev front:cl=5   # 장치별 변경 (조합 가능)
python -m lapsim delta  --dev rear:cl=10 --bal 48             # 변경 후 밸런스 48 % F 로 재트림
python -m lapsim coeffs                                       # 랩타임 계수: 전체 + 장치별
python -m lapsim sweep  --cl=-20:20:5 --cd=-20:20:5 --device rear --plot sweep.png
python -m lapsim balance --range 40:60:2.5                    # 공력 밸런스 스윕 (총 ClA 고정)
python -m lapsim run --set aero.rear.ClA=1.6 --set mass=270   # 파라미터 즉석 변경
python -m lapsim run --track tracks/sample_autocross.json     # 합성 오토크로스 (정지 출발)
```
공통 옵션: `--car` (차량 JSON), `--track` (트랙 JSON), `--ds` (격자 간격, 기본 0.5 m),
`--smooth` (센터라인 스무딩 창, 기본 4 m), `--flying` (열린 코스를 플라잉 스타트로).
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
* 장치 이름은 자유 (`body` 는 공력 장치가 아닌 기본 드래그용). 옛 형식 `{"ClA", "CdA", "balance_front"}` 도 읽는다.
* `balance` / `--bal` 재트림은 가장 앞·가장 뒤 장치의 ClA 를 (CdA 도 같은 비율로) 조절해 총 ClA 를 유지한 채 목표 밸런스를 맞춘다 (윙 각도 조정에 해당).

## 출력 예 (기본 차량 × FSK 2026 Endurance, 850.8 m, 시계방향)
```
lap time         :   54.842 s
max lat accel    :   2.52 g
corner limited by: front axle 8.93 s, rear axle 0.00 s

+1 % ClA             :    -27.4 ms
+1 % CdA             :     +6.6 ms
exchange rate        : +1 % drag is paid back by +0.24 % downforce

per device (absolute change on that device only; balance shifts with it):
  device       +0.1 ClA   +0.1 CdA  break-even L/D
  front        -117.2ms    +39.3ms            0.34
  mid           -73.8ms    +34.1ms            0.46
  rear          -39.6ms    +62.5ms            1.58
  body          -77.2ms    +46.3ms            0.60
```
읽는 법: 기본 차량은 코너에서 **앞축이 먼저 한계** → 프론트 다운포스 0.1 이 리어의 3배 가치.
리어 윙은 ΔClA/ΔCdA > 1.58 이어야 이득 (리어 다운포스는 밸런스를 뒤로 보내고, 높은 드래그는 앞축 하중을 더 덜어냄).
`balance` 스윕 최적점은 약 50 % F.

## 모델
| 항목 | 내용 |
|---|---|
| 공력 | 장치별 F_down = ½ρ·ClA_i·v², F_drag = ½ρ·CdA_i·v², 앞축 몫 = 1 − x_cp,i/L, 장치별 드래그 피칭 모멘트(h_cp,i) 합산 |
| 축하중 | 정적 배분 + 공력 + 종방향 하중이동(m·a_x·h/L) — 앞/뒤 축 분리 |
| 타이어 | 하중 민감 μ(Fz) = μ0·(1 − k·(Fz−Fz0)/Fz0), 축별 마찰 타원 (μx, μy) |
| 코너 한계속도 | 앞축 횡력 m·v²·κ·b/L, 뒤축 횡력 m·v²·κ·a/L + 주행저항 구동력이 각 축 한계 이내인 최대 v (이분법). 어느 축이 먼저 포화되는지 기록 → 공력 밸런스 효과가 드러남 |
| 가속 | min(모터 토크 한계, 80 kW 출력 한계, 후륜 잔여 그립) − 드래그 − 구름저항, 모터 최고 rpm에서 차단 |
| 제동 | 4륜, 축별 잔여 그립 (이상적 브레이크 배분) + 드래그 |
| 랩 | v = min(전진 패스, 후진 패스), 닫힌 코스는 플라잉 랩(3랩 타일링 후 가운데 랩), 열린 코스는 정지 출발 |

단순화 (QSS 특성상): 좌우 하중이동·롤 강성 배분 없음, 요 관성/과도 응답 없음, 센터라인 주행(레이싱라인 최적화 없음),
라이드 하이트에 따른 공력 맵 없음, 공력 밸런스 최적점은 "두 축 동시 포화"점이라 실제 차는 안정성 마진만큼 뒤로 둔다.

## 파일
```
lapsim/vehicle.py   차량·타이어·공력 장치(AeroDevice)·EV 파워트레인 (Vehicle, load_vehicle)
lapsim/track.py     트랙 JSON 로더 → 등간격 곡률 프로파일
lapsim/solver.py    QSS 솔버 (simulate → LapResult)
lapsim/study.py     modify / delta / sweep / balance_sweep / coefficients / device_coefficients
lapsim/plot.py      속도 트레이스, 스윕 히트맵, 밸런스 곡선 (matplotlib)
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
`cars/fsk_ev_baseline.json` 참고. 단위는 SI (kg, m, N, W, rpm). `ClA` 는 다운포스 양수.

## 테스트
```bash
python -m pytest -q
```
해석해 검증: 무공력 원 주행 v = √(μgR), 드래그-파워 평형 최고속, 모터 rpm 제한, ClA↑/CdA↑ 단조성,
밸런스 내부 최적점, 격자 수렴(0.5 m vs 0.25 m < 0.5 %), 트랙 JSON 형식별 곡률,
장치 합산·밸런스 재트림·옛 형식 호환, FSK 코스 길이/방향.
