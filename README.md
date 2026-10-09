# FSK 랩타임 시뮬레이터 (Quasi-Steady-State)

차량 파라미터(질량·타이어·파워)와 공력(ClA·CdA·공력 밸런스)을 넣고 FSK 코스 랩타임을 계산하는 도구.
**"다운포스 +10%, 드래그 +5% 면 랩타임이 어떻게 되나?"** 를 숫자로 답하는 게 목적.

```bash
pip install -r requirements.txt          # numpy (필수), matplotlib (플롯), pytest

python -m lapsim run    --track tracks/sample_autocross.json --plot lap.png
python -m lapsim delta  --dcl 10 --dcd 5                 # ClA +10 %, CdA +5 % → Δ랩타임
python -m lapsim coeffs                                  # 랩타임 계수 (ms / 0.1 ClA, ms / 1 % CdA, 교환비)
python -m lapsim sweep  --cl=-20:20:5 --cd=-20:20:5 --plot sweep.png   # ΔCl × ΔCd 그리드
python -m lapsim balance --range 30:60:2.5               # 공력 밸런스 스윕
python -m lapsim run --set aero.ClA=4.2 --set mass=270   # 파라미터 즉석 변경
```
공통 옵션: `--car` (차량 JSON), `--track` (트랙 JSON), `--ds` (격자 간격, 기본 0.5 m),
`--smooth` (센터라인 스무딩 창, 기본 4 m), `--flying` (열린 코스를 플라잉 스타트로).

## 출력 예 (기본 차량 × 샘플 오토크로스)
```
+0.1 ClA (downforce) :    -67.8 ms
+0.1 CdA (drag)      :    +65.9 ms
exchange rate        : +1 % drag is paid back by +0.36 % downforce
lap-time neutral     : 1.0 CdA == 0.97 ClA (a device is worth it if its dCl/dCd exceeds this)
```
→ 이 코스에선 공력 부품을 추가할 때 **ΔClA/ΔCdA > 0.97** 이면 랩타임 이득. FS처럼 저속·코너 위주 코스에서
다운포스 가치가 드래그 비용보다 훨씬 큰 이유가 여기서 숫자로 나온다. (샘플 코스는 합성 코스이므로 실제 FSK 코스로 다시 계산할 것.)

## 모델
| 항목 | 내용 |
|---|---|
| 공력 | F_down = ½ρ·ClA·v², F_drag = ½ρ·CdA·v², 앞축 몫 = `balance_front`, 드래그 피칭 모멘트(`h_cp`) 포함 |
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
lapsim/vehicle.py   차량·타이어·공력·EV 파워트레인 (Vehicle, load_vehicle)
lapsim/track.py     트랙 JSON 로더 → 등간격 곡률 프로파일
lapsim/solver.py    QSS 솔버 (simulate → LapResult)
lapsim/study.py     delta / sweep / balance_sweep / coefficients
lapsim/plot.py      속도 트레이스, 스윕 히트맵, 밸런스 곡선 (matplotlib)
cars/fsk_ev_baseline.json   기본 FS EV (280 kg, ClA 3.5, CdA 1.3, 45 % F, 80 kW)
tracks/sample_autocross.json  테스트용 합성 오토크로스 (실제 FSK 코스 아님)
```

## 트랙 JSON 형식 (자동 감지)
```jsonc
{"x": [...], "y": [...], "closed": true}                 // 센터라인 좌표 [m]
{"centerline": [[x, y], ...]}                            // 동일 (또는 [{"x":..,"y":..}])
{"left": [[x, y], ...], "right": [[x, y], ...]}          // 콘 좌표 (blue/yellow, inner/outer 도 가능)
{"segments": [{"type": "straight", "length": 60},
              {"type": "corner", "radius": 9, "angle": 180, "direction": "left"}]}
{"s": [...], "curvature": [...]}                         // 거리-곡률
```
점/콘 형식은 기본 `closed: true` (엔듀런스), segments/곡률 형식은 기본 `closed: false` (오토크로스).

## 차량 JSON
`cars/fsk_ev_baseline.json` 참고. 단위는 SI (kg, m, N, W, rpm). `ClA` 는 다운포스 양수.

## 테스트
```bash
python -m pytest -q
```
해석해 검증: 무공력 원 주행 v = √(μgR), 드래그-파워 평형 최고속, 모터 rpm 제한, ClA↑/CdA↑ 단조성,
밸런스 내부 최적점, 격자 수렴(0.5 m vs 0.25 m < 0.5 %), 트랙 JSON 형식별 곡률.
