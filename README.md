<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/brand/karman_lockup_dark.svg">
  <img src="assets/brand/karman_lockup_light.svg" width="420" alt="Project Karman">
</picture>

서울과학기술대학교 e_formula (MIP) 공력팀의 Formula Student Korea(FSK) 공력 개발 도구를 한곳에 모은 모노레포입니다.

> 이름은 카르만 소용돌이 열(Kármán vortex street)에서 따왔습니다. 로고의 K는 세로 줄기(물체) 뒤로 서로 엇갈려 말리는 두 소용돌이로 이루어져 있습니다. 로고 원본은 [`assets/brand/`](assets/brand)에 있습니다.

## 도구 지도

| 패키지 | 하는 일 | 상태 | 원본 레포 (보관용) |
|---|---|---|---|
| [`packages/aero-automation`](packages/aero-automation) | 2D 다요소 날개(메인 + 플랩) Fluent 배치 해석 자동화: 케이스 생성 → 저충실도 스크리닝 → gmsh 메쉬 → Fluent 실행 → Cl/Cd 정리·순위. 단일 익형 데이터셋과 CL/CD 회귀 모델 예제 포함 | 사용 중 | `AERO_AUTOMATION` |
| [`packages/aerolap`](packages/aerolap) | 공력·서스펜션 연성 4륜 랩타임 시뮬레이터: 축별 그립, 하중이동, 차고-aero map 연성, DRS, 배터리·열 모델, 셋업 최적화 | 사용 중 | `4_wheel_simulation` |
| [`packages/fsk-lapsim`](packages/fsk-lapsim) | FSK 이벤트(가속·스키드패드·오토크로스·엔듀런스) 기준 랩타임·에너지 평가: 설계안 후보 비교·순위, 민감도, 밸런스 최적화, HTML 리포트 | 사용 중 | `LAP_TIME_OPTIMIZATION--AERO` |
| [`packages/vortex`](packages/vortex) | 터프트 영상 ↔ CFD ↔ AI 대리모델의 유동 재현 정확도 비교 | 설계 단계 (v0.3) | `PROJECT_VORTEX` |

계획 중인 도구는 [`docs/planned/`](docs/planned)에 메모로 둡니다. 더 이상 개발하지 않는 도구(이전 LapSim 등)는 [`archive/`](archive)에 보관합니다.

## 흐름

```
에어포일 / 형상 후보
   │
   ▼
aero-automation ── 2D Cl/Cd, 형상 순위
   │
   ▼
aerolap ── 공력·서스펜션 연성 정밀 랩타임, 셋업
fsk-lapsim ── 설계안 후보를 대회 이벤트 기준으로 비교·순위
   │
   ▼
vortex ── 실차 터프트 테스트로 CFD·대리모델 검증
```

## 사용법

### 런처 (추천)

레포 루트의 **`Karman.bat`** 을 더블클릭하면 도구를 고르는 창이 뜹니다.

- 카드의 **실행** 버튼, 카드 더블클릭, 또는 숫자키 `1`–`4` 로 바로 실행합니다.
- 필요한 모듈이 없으면 **설치 필요** 표시와 함께 버튼이 **설치** 로 바뀝니다. 설치가 끝나면 `F5` 로 다시 확인하세요.
- 새 도구를 추가하려면 [`launcher/tools.json`](launcher/tools.json) 에 항목을 하나 더 넣으면 됩니다.

### 직접 실행

각 패키지는 독립적으로 실행합니다. 패키지 폴더로 이동한 뒤 해당 README를 따르세요.

```bash
cd packages/aerolap
pip install -e .
```

## 규칙

- 커밋 메시지는 패키지 이름으로 시작합니다. 예: `aerolap: 히브 강성 스윕 추가`
- CFD 결과, 대용량 메쉬, 영상은 레포에 넣지 않습니다. 작은 샘플만 넣고, 큰 파일은 외부 저장소에 두고 경로만 기록합니다.
- 대회 시즌 기준 상태는 태그로 남깁니다. 예: `fsk2027`

## 폴더 아이콘 (Windows)

clone한 뒤 `scripts/set-folder-icon.bat`을 한 번 실행하면 탐색기에서 폴더 아이콘이 바뀝니다.
