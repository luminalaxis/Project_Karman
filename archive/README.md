# archive

더 이상 개발하지 않지만 참고용으로 남겨 둔 도구입니다. 런처에는 나오지 않습니다.

| 폴더 | 내용 | 대체 도구 |
|---|---|---|
| [`lapsim`](lapsim) | 준정적(QSS) 랩타임 시뮬레이터 + Streamlit GUI. 공력 장치별(front/mid/rear) ΔClA·ΔCdA 비교, 장치별 목표치 계산 | [`packages/aerolap`](../packages/aerolap) |
| [`fsk-lapsim`](fsk-lapsim) | FSK 이벤트 기준 랩타임·에너지 평가 (점질량/축 단위 모델, 트랙 생성기, HTML 리포트). 2026-11 aerolap과 기능이 겹쳐 보관. 몬테카를로 순위 신뢰도·점수 환산·GPS 비교는 aerolap으로 옮김 (`aerolap mc`, `score`, `gps`) | [`packages/aerolap`](../packages/aerolap) |

필요하면 폴더에 들어가 각 README대로 실행할 수 있습니다.
