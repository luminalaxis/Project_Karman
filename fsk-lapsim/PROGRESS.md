# PROGRESS — 진행 상황 (세션 인계용)

## 이어받는 방법
```bash
cd fsk-lapsim
pip install -e ".[dev]"
python -m pytest -q --cov=fsk_lapsim     # 전체 테스트 + 커버리지
ruff check src tests
```
- 지시서: 저장소 루트 대화의 "FSK 랩타임 시뮬레이터 + 에어로 민감도 분석 도구 — 무인 개발 지시서".
- 결정 기록: DECISIONS.md, 가정값: ASSUMPTIONS.md, 수식: MODEL.md.
- 각 Phase 끝에 테스트 통과 → 커밋 → 이 파일 갱신.

## Phase 상태
| Phase | 내용 | 상태 |
|---|---|---|
| P0 | 저장소, 문서 뼈대, 가정값 설정 파일 | 완료 |
| P1 | 트랙 모듈 (세그먼트·슬라럼, 생성기, 템플릿, CSV 임포트) | 완료 |
| P2 | 타이어 + 모델 A + 솔버 + 테스트 1~6 | 완료 |
| P3 | 모델 B 테스트 7, 9 + 축 한계 진단 | 완료 |
| P4 | 파워트레인·에너지 + 테스트 8 | 완료 |
| P5 | 공력 부품 합성 + 2D→3D + 리어윙 비교 | 다음 |

## 완료 내용
### P0
- `pyproject.toml` (패키지 `fsk_lapsim`, CLI 엔트리 `fsk-lapsim`), src 레이아웃.
- pydantic 설정 스키마 `config.py` (모든 수치 = `Param{value, unit, assumed, note}`), 예시 YAML
  (`vehicle.yaml`, `aero_components.yaml`, `events.yaml`), 런타임 데이터클래스 (`Vehicle`, `Tire`, `Aero`, `Powertrain`).
- 부품 합성이 ClA 3.5 / CdA 1.4 / 밸런스 45 % 를 정확히 재현하도록 배치.

### P1
- `track.py`: `Straight`/`Arc`/`Slalom`(κ = κmax·cos, Struve/Bessel 로 정확한 끝점) 세그먼트, `SegmentTrack`, `SampledTrack`,
  `track_from_xy` (splprep 평활화, σ 파라미터, 호길이 재표본, 곡률), 미러, 등간격 이산화.
- `trackgen.py`: 규칙(`LayoutRules`) 기반 시드 고정 생성기 (오토크로스=열린 코스, 엔듀런스=Dubins CSC 폐합),
  규칙 검사(`check_layout`: 최장 직선, 최소 반경, 자기 간격, 폐합 오차), 직선 2개 길이를 풀어 정확히 폐합하는 `adjust_straights`.
- 트랙 YAML: `autocross_generated`(seed 42, 814 m), `endurance_generated`(seed 7, 918 m), 템플릿 A(709 m)/B(587 m)/C(514 m),
  `fsk2026_endurance`(저장소 실측 기반 중심선 CSV, 843 m, 시계방향, 최소 반경 2.5 m — FS 규정 범위보다 훨씬 타이트).

### P2
- `models.py`: 모델 A(점질량)와 모델 B(축 단위, 하중이동·하중민감도·LLTD·LSD/오픈 디프·이상/고정 제동배분)를 같이 구현.
  한계 정의는 일정 속도 기준 (D-08, 처음의 코스팅 정의에서 수정 — 이유 기록).
- `ggv.py`: (v, θ) 격자 GGV + 순수 파이썬 스칼라 보간, 차량별 lru_cache. `solver.py`: 곡률 한계(이분법), Heun 전진/후진,
  플라잉 랩 반복, `LapResult`. `energy.py`: 구간별 힘·에너지·회생·모터 운전점.
- 검증 1~6 통과: 원 선회 해석해 (상대오차 ~1e-9), 최고속도 (출력·항력 해석해 1e-6, 회전수 제한), 75 m 가속 vs solve_ivp
  (A 2e-5, B 2e-6), 격자 수렴 (ds 0.25→0.125 변화 < 1e-4), 단조성 (출력·μ·CdA·ClA), 미러 대칭 (1e-12).
- 기본 차량(모델 B) 참고값: 가속 4.20 s, 스키드패드 5.10 s, 합성 오토크로스 55.9 s (평균 52 km/h).

### P3
- 모델 B 진단 `ay_max_by_limit` (전륜 횡 / 후륜 횡 / 구동 각각의 한계), 가속 한계의 제약별 분해 `ax_accel_parts`.
- GGV 정확도 개선 (D-14): 최고속도 격자 끝 정렬, 제약별 표 + 출력 한계 닫힌 식. 직접 계산 대비 최대 0.0085 m/s².
- 테스트 7 (밸런스 0 → 전륜 먼저, 1 → 후륜 먼저, 랩타임 내부 최적), 9 (GGV vs 직접), 하중 평형·LLTD·휠 리프트·오픈 디프·
  고정 제동배분·하중민감도·타이어/파워트레인 단위 테스트. 총 84 테스트 통과.
- 관찰: 기본 차량은 저속에서 후륜 구동 한계(항력 상쇄 구동력 + 횡력)가 근소하게 먼저 걸리고 전/후 횡한계가 거의 같다 (균형).

### P4
- 에너지 (`energy.py`): 구간별 v̄² 로 항력·다운포스 정확 평균, 구동/제동 일, 회생(후축 분담 × 모터 토크·20 kW 한계),
  어큐뮬레이터 순 에너지, 출력 시계열, 모터 토크·rpm, 시간가중 운전점 히스토그램.
- 테스트 8: (a) 구동 일 − 제동 일 = ΔKE + 항력 + 구름 (구간 정의, 1e-9) (b) 노드 속도로 시간 적분한 손실과 비교
  (허용 0.5 %, 관측 6e-6). 출력·회생 한계, 회생 on/off, 정지 출발 유한성, 모터 한계 테스트.
- 참고 (가정값 기반): 엔듀런스 합성 루프 1 랩 0.361 kWh (회생 0.050 kWh 포함), FSK 2026 중심선 0.333 kWh/랩.

## 다음 할 일
- P5: 2D→3D 변환 헬퍼, 리어윙 후보 CSV (DUMMY) 로더, 부품 교체 → 차량 갱신 → 이벤트별 비교 표.
