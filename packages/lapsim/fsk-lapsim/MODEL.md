# MODEL — 수식 정리

단위는 전부 SI (m, s, kg, N, W, rad). 기호: m 질량, g 중력가속도, L 휠베이스, a/b 무게중심→앞/뒤 차축 거리
(b = L·w_f, a = L·(1−w_f), w_f 정적 전륜 하중 비율), h CG 높이, t_f/t_r 트레드, q = ½ρv².

## 1. 공력

    F_down = q·ClA,   F_drag = q·CdA,   F_down,front = bal·F_down

`bal` 은 **지면 기준 항력 피칭모멘트를 포함한** 전륜 다운포스 비율 (CFD/풍동 관례, DECISIONS D-05).
그래서 차량 모델에는 별도의 항력 모멘트 항이 없다.

### 1.1 부품 합성 (`aero.synthesize`)
부품 i: ClA_i, CdA_i, 압력중심 x_i (앞차축 기준 뒤쪽 +), 항력 높이 z_i, 질량 m_i, 질량중심 (xm_i, zm_i).

    ClA = Σ ClA_i,   CdA = Σ CdA_i
    ClA_f = Σ ClA_i·(1 − x_i/L) − Σ CdA_i·z_i/L          bal = ClA_f / ClA

부품 교체 (기준 패키지 → 새 패키지, `apply_component_change`):

    m' = m + Σ(m_new) − Σ(m_old)
    x_cg' = (m·a + Σ m_new·xm − Σ m_old·xm) / m'      w_f' = 1 − x_cg'/L
    h'    = (m·h + Σ m_new·zm − Σ m_old·zm) / m'

### 1.2 2D → 3D 변환 (`wing.wing_2d_to_3d`, **대략적 추정**)

    S = c·b,  AR_eff = (b/c)·k_ep (기본 k_ep = 1.5, 엔드플레이트 효과)
    k = AR_eff/(AR_eff + 2)                     (양력선 이론 근사)
    Cl3D = k·Cl2D
    Cd3D = Cd2D + Cl3D² / (π·e·AR_eff)          (e = 0.9)
    ClA = Cl3D·S,  CdA = Cd3D·S

3D CFD/풍동 값 (`cla_3d_m2`, `cda_3d_m2`) 이 있으면 그 값을 그대로 쓴다. 압력중심은
`x_cp = cm_ref_x_m + (Cm_2d/Cl_2d)·c` (Cm 은 다운포스가 기준점 뒤에 작용하면 양수).

## 2. 타이어 (바퀴 단위)

    μ(Fz) = μ0·(1 + k·(Fz − Fz0)/Fz0)
    F_y,max(Fz) = μ(Fz)·Fz,   F_x,max = r_xy·F_y,max   (r_xy = 종/횡 μ 비)
    마찰 타원: (Fx/F_x,max)² + (Fy/F_y,max)² ≤ 1

k < 0 이면 F_y,max 가 Fz_peak = Fz0(1−k)/(−2k) 에서 최대 → 그 이상은 일정 (D-07). Fz < 0 은 0.

## 3. 차량 모델

공통: 구름저항 R = Crr·(m·g + F_down), 저항 D + R, 파워트레인 한계 (5절) F_pt(v).
요구 타이어 종방향 힘 F = m·ax + D + R (F ≥ 0 구동, F < 0 제동).

### 3.1 모델 A — 점질량 (검증 기준선)

    N = m·g + F_down,   ay_grip = μ0·N/m
    F_x,tyre(ay) = r_xy·μ0·N·√(1 − (ay/ay_grip)²)
    ax_acc = (min(F_x,tyre, F_pt) − D − R)/m,    ax_brk = −(F_x,tyre + D + R)/m

하중이동·하중민감도 없음. 구동도 전체 그립을 쓴다 (검증용 단순화).

### 3.2 모델 B — 축 단위 준정상 (주 모델)

    Fz_f = m·g·b/L + bal·F_down − m·ax·h/L
    Fz_r = m·g·a/L + (1−bal)·F_down + m·ax·h/L
    ΔFz_f = LLTD·m·|ay|·h/t_f,   ΔFz_r = (1−LLTD)·m·|ay|·h/t_r
    안쪽 = max(Fz_axle/2 − ΔFz, 0),  바깥쪽 = Fz_axle − 안쪽     (휠 리프트 시 0 으로 클램프)

축별 횡력 요구 (정상선회 요 평형): Fy_f = m·ay·b/L, Fy_r = m·ay·a/L.
축 용량 C_axle = F_y,max(안쪽) + F_y,max(바깥쪽), 이용률 u = Fy/C. 축 안에서 바퀴 용량에 비례해 횡력을
나누면 (이용률 동일) 축 단위 마찰 타원이 정확해진다:

    잔여 종방향 용량 R_axle = r_xy·C_axle·√(1 − u²)
    구동 (후륜): LSD → R_r,   오픈 디프 → 2·r_xy·F_y,max(안쪽 후륜)·√(1 − u_r²)
    제동 (4륜): 이상 배분 → F_b ≤ R_f + R_r,   고정 배분 β → β·F_b ≤ R_f, (1−β)·F_b ≤ R_r

상태 (v, ax, ay) 가능 조건: u_f ≤ 1, u_r ≤ 1, 그리고 F ≥ 0 이면 F ≤ min(R_drive, F_pt), F < 0 이면 −F ≤ 제동 용량.

### 3.3 코너 한계와 기준 상태 (D-08)

    ax_ref(v) = min(0, 직선 최대가속(v))      (최고속도 이하에서는 0 = 속도 유지)
    ay_max(v) = ax_ref 상태가 가능한 최대 |ay|        (이분법)

즉 ay_max 는 **구동력이 항력+구름저항을 감당하며 속도를 유지할 수 있는** 횡가속도 한계이다 (스키드패드
정상선회와 같은 정의). |ay| ≤ ay_max 이면 가능한 ax 는 ax_ref 를 포함하는 구간 [ax_brk, ax_acc] 이며
양 끝은 이분법으로 찾는다. 진단용으로 전륜 횡 / 후륜 횡 / 구동 제약 각각의 한계 (`ay_max_by_limit`) 를 준다.

## 4. GGV 표 (D-13, D-14)

- 속도 격자 0 … v_top (정확히 최고속도에서 끝남), 0.25 m/s 이하 간격.
- 횡방향 ay = sin θ·ay_max(v), θ ∈ [0, π/2] 73 점 (마찰 타원 √(1−u²) = cos θ 라 θ 로는 매끄러움).
- ay_max(v) 는 v² 에 대해 선형보간 (모델 A 에서는 정확).
- 가속 한계 = min(전륜 횡그립 제약 표, 후륜 트랙션 제약 표, F_pt 닫힌 식) — 제약별로 따로 보간해 꺾임점 오차 제거.
- 직접 계산 대비 오차: 모델 B 최대 0.0085 m/s², 모델 A 최대 0.028 m/s² (테스트 9).

## 5. 파워트레인·에너지

    F_pt(v) = min(T_max·G·η/r, P_max·η/max(v, 1e−3))   (v ≤ v_rpm),  0 (v > v_rpm)
    v_rpm = n_max·2π/60 / G · r

v = 0 에서 P/v 는 1 mm/s 로 하한 처리 (토크 한계가 먼저 걸리므로 결과에 영향 없음).
최고속도 v_top = v_rpm 또는 F_pt(v) = D + R 의 근 (이분법).

구간 (노드 i→i+1, 구간 내 v² 는 s 에 선형):

    v̄² = (v_i² + v_{i+1}²)/2,  ax = (v_{i+1}² − v_i²)/(2·ds),  dt = 2·ds/(v_i + v_{i+1})
    F = m·ax + ½ρCdA·v̄² + Crr·(m·g + ½ρClA·v̄²)
    F > 0:  E_batt = F·ds/η,  모터 토크 T = F·r/(G·η)
    F < 0:  후축 분담 (이상: Fz_r/(Fz_f+Fz_r), 고정: 1−β) × |F| 를 모터가 회생,
            F_regen ≤ min(T_max·G/r, P_regen/(η·v̄)),  E_regen = η·F_regen·ds
    랩 순 에너지 = Σ E_batt − Σ E_regen

에너지 수지 (테스트 8): Σ F⁺·ds − Σ F⁻·ds = ΔKE + W_drag + W_roll.

## 6. 솔버 (QSS)

1. 노드 s_i = i·L/N (N = ⌈L/ds⌉), κ_i.
2. 곡률 한계 v_lim,i: v²·|κ_i| = ay_max(v) 의 근 (ay_max(v)/v² 단조 감소 → 유일), v_top 으로 상한.
3. 전진 패스 (Heun, v² 기준):
   a₁ = ax_acc(v_i, v_i²κ_i), p = v_i² + 2a₁ds (≤ v_lim²), a₂ = ax_acc(√p, p·κ_{i+1}),
   v_{i+1}² = min(v_i² + (a₁+a₂)ds, v_lim,i+1²).
4. 후진 패스: 같은 방식으로 감속 한계 사용.
5. v = min(전진, 후진).
6. 폐곡선 플라잉 랩: 전진 시작속도 ← 전진 끝속도, 후진 끝속도 ← 후진 시작속도 를 수렴할 때까지 반복.
7. 시간 t = Σ 2ds/(v_i + v_{i+1}).

## 7. 이벤트

- 가속: 75 m 직선, v₀ = 0.
- 스키드패드: v²/R = ay_max(v) 를 이분법으로 풀고 t = 2πR/v (R = 9.125 m), 좌우 대칭 모델이라 좌/우 동일 → 평균.
- 오토크로스: 1 랩, 정지 출발 (기본) 또는 플라잉.
- 엔듀런스: n = round(총거리/랩길이) 랩. t = t_첫랩(정지 출발, 끝속도 = 플라잉 랩 시작속도) + (n−1)·t_플라잉.
  에너지도 같은 방식. 선택적 출력 제한 `endurance_max_power_w`.

## 8. 능동 공력 (DRS)

- 개방 상태: 리어윙 ΔClA = −f_cl·ClA_rw, ΔCdA = −f_cd·CdA_rw, 앞축 다운포스 변화
  ΔClA_f = ΔClA·(1 − x_rw/L) − ΔCdA·z_rw/L → 밸런스 자동 이동.
- 개방 조건: |ay| < ay_th, 최대 가속 중 (전진 패스가 한계), v ≥ v_min (기본 = 개방 상태가 직선 가속에서 더 빨라지는 교차 속도).
- 구간별 가중치 w(t) = min((t − t_open)/T, (t_close − t)/T, 1); 계수(정확히는 두 GGV 의 한계값)를 w 로 선형 보간.
  t_close = 구간 끝 (제동 시작 또는 |ay| 초과 지점) → 닫힘 명령은 T 초 전, 리드 거리 ≈ ∫v dt ≈ v·T.
- 구간 검출 ↔ 재시뮬레이션을 반복.

## 9. 분석

- 민감도: dt/dp ≈ (t(p+h) − t(p−h))/(2h). 등가 환산: Δp_eq = (dt/dm)/(dt/dp) (+1 kg 과 같은 시간 손실).
- 실측 우선순위: 점수 = Σ_event w_e·|dt_e/dp|·σ_p (w: 가속·스키드패드·오토크로스 1, 엔듀런스 1/20).
- 밸런스 최적화: 15 점 스캔 + 최적 주변 bounded Brent.
- 스윕: ClA × CdA 격자 (밸런스·질량 고정).

## 10. 검증 결과 요약 (tests/)

| # | 항목 | 기준 | 결과 |
|---|---|---|---|
| 1 | 다운포스 포함 원 선회 해석해 (모델 A) | 상대오차 < 0.1 % | ~1e−9 (반경 6–40 m) |
| 2 | 최고속도 (P·η = (D+R)·v / 회전수 한계) | — | 1e−6, 회전수 한계 정확 |
| 3 | 75 m 가속 vs solve_ivp | < 0.5 % | A 2e−5, B 2e−6 |
| 4 | ds 0.25 → 0.125 랩타임 변화 | < 0.1 % | < 1e−4 |
| 5 | 단조성 (출력·μ↑ 안 느려짐, CdA↑ 안 빨라짐, ClA↑ 안 느려짐) | — | 통과 (상대 1e−9 수치잡음 허용) |
| 6 | 좌우 반전 트랙 | 동일 | 1e−12 |
| 7 | 밸런스 극단 → 해당 축 먼저 한계 | — | 통과 |
| 8 | 에너지 수지 | 0.5 % (독립 시간적분) | 6e−6 |
| 9 | GGV 보간 vs 직접 계산 | 최대 0.03 m/s² | 0.0085 (B) |
