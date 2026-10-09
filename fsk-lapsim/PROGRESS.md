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
| P2 | 타이어 + 모델 A + 솔버 + 테스트 1~6 | 다음 |

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

## 다음 할 일
- P2: 모델 A (점질량) + GGV + QSS 솔버 (전진/후진, 플라잉 랩) + 테스트 1~6.
