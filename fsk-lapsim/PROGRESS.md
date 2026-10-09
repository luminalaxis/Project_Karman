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
| P1 | 트랙 모듈 | 다음 |

## 완료 내용
### P0
- `pyproject.toml` (패키지 `fsk_lapsim`, CLI 엔트리 `fsk-lapsim`), src 레이아웃.
- pydantic 설정 스키마 `config.py` (모든 수치 = `Param{value, unit, assumed, note}`), 예시 YAML
  (`vehicle.yaml`, `aero_components.yaml`, `events.yaml`), 런타임 데이터클래스 (`Vehicle`, `Tire`, `Aero`, `Powertrain`).
- 부품 합성이 ClA 3.5 / CdA 1.4 / 밸런스 45 % 를 정확히 재현하도록 배치.

## 다음 할 일
- P1: 트랙 모듈 (세그먼트/슬라럼, 시드 고정 생성기 + Dubins 폐합, 템플릿, CSV 임포트, 미러, 스키드패드·가속 정의).
