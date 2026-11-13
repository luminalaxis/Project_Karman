# Project_Karman

서울과학기술대학교 e_formula (MIP) 공력팀의 Formula Student Korea(FSK) 공력 개발 도구를 한곳에 모은 모노레포입니다.

> 이름은 카르만 소용돌이 열(Kármán vortex street)에서 따왔습니다.

## 도구 지도

| 패키지 | 하는 일 | 상태 | 원본 레포 (보관용) |
|---|---|---|---|
| [`packages/aero-automation`](packages/aero-automation) | 2D 다요소 날개(메인 + 플랩) Fluent 배치 해석 자동화: 케이스 생성 → 저충실도 스크리닝 → gmsh 메쉬 → Fluent 실행 → Cl/Cd 정리·순위. 단일 익형 데이터셋과 CL/CD 회귀 모델 예제 포함 | 사용 중 | `AERO_AUTOMATION` |
| [`packages/aerolap`](packages/aerolap) | 공력·서스펜션 연성 4륜 랩타임 시뮬레이터: 축별 그립, 하중이동, 차고-aero map 연성, DRS, 배터리·열 모델, 셋업 최적화 | 사용 중 | `4_wheel_simulation` |
| [`packages/lapsim`](packages/lapsim) | 준정적(QSS) FSK 랩타임 시뮬레이터: ClA·CdA·밸런스 변화가 랩타임에 주는 영향, 장치별 목표치 계산, Streamlit GUI | 사용 중 | `LAP_TIME_OPTIMIZATION--AERO` |
| [`packages/vortex`](packages/vortex) | 터프트 영상 ↔ CFD ↔ AI 대리모델의 유동 재현 정확도 비교 | 설계 단계 (v0.3) | `PROJECT_VORTEX` |

계획 중인 도구는 [`docs/planned/`](docs/planned)에 메모로 둡니다.

## 흐름

```
에어포일 / 형상 후보
   │
   ▼
aero-automation ── 2D Cl/Cd, 형상 순위
   │
   ▼
lapsim / aerolap ── ClA·CdA·밸런스 → 랩타임, 장치별 성능 목표
   │
   ▼
vortex ── 실차 터프트 테스트로 CFD·대리모델 검증
```

## 사용법

각 패키지는 독립적으로 실행합니다. 패키지 폴더로 이동한 뒤 해당 README를 따르세요.

```bash
cd packages/aerolap
pip install -e .
```

## 규칙

- 커밋 메시지는 패키지 이름으로 시작합니다. 예: `aerolap: 히브 강성 스윕 추가`
- CFD 결과, 대용량 메쉬, 영상은 레포에 넣지 않습니다. 작은 샘플만 넣고, 큰 파일은 외부 저장소에 두고 경로만 기록합니다.
- 대회 시즌 기준 상태는 태그로 남깁니다. 예: `fsk2027`
