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
| P3 | 모델 B 테스트 7, 9 + 축 한계 진단 | 다음 |

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

## 다음 할 일
- P3: 테스트 7 (밸런스 극단 → 해당 축 먼저 한계), 9 (GGV 보간 vs 직접 계산), 모델 B 추가 검증 (오픈 디프, 고정 배분).
