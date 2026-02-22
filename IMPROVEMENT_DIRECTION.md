# 한국형 GDPNow 2.0 개선 방향 제안

## 1) 현재 코드베이스 진단 요약

현재 `최종.py`는 **데이터 수집 → 전처리 → 요인추출(PCA) → 시계열 예측(AR/VAR)**까지 단일 파일에서 일괄 처리하는 구조입니다. 이 구조는 빠르게 실험할 때는 유리하지만, 운영/고도화 단계에서는 아래 한계가 큽니다.

- 단일 대형 스크립트 구조로 인해 재사용성과 테스트 용이성이 낮음
- 전처리 규칙(정상성 검정/차분/로그차분)이 규칙 기반으로 하드코딩되어 설명 가능성은 있으나 유연성이 부족함
- API 호출/결측/빈 응답에 대한 예외 처리는 있으나, 재시도 정책·버전관리·실시간 nowcast 스냅샷 관리가 약함
- 평가 파이프라인(실시간 vintages 기반 backtest, RMSE/MAE, 발표일 기준 nowcast 성능 비교)이 코드상 분리되어 있지 않음

---

## 2) 새롭게 만들 때의 핵심 원칙

1. **파이프라인 분리**: `ingest`, `feature`, `model`, `evaluate`, `serve`를 분리
2. **실시간성 반영**: 발표시차(ragged edge)와 데이터 빈티지(vintage) 중심 설계
3. **모델 앙상블**: 단일 PCA+AR에서 벗어나 DFM/MIDAS/Gradient Boosting 등 병렬 운용
4. **운영 관점 내장**: 데이터 품질검사, 모델 모니터링, 재학습 주기 자동화
5. **설명 가능성 강화**: nowcast 변화분 기여도(contribution)와 시나리오 민감도 제공

---

## 3) 권장 타깃 아키텍처

### A. 데이터 계층
- 소스: BOK ECOS, KOSIS, 관세/무역, 금융시장(금리·환율), 글로벌 선행지표
- 저장: `raw/`, `staging/`, `feature_store/`, `vintage_store/` 계층화
- 메타데이터: 시계열 주기(월/분기), 발표지연, revision 정책, 결측 규칙을 테이블로 관리

### B. 피처 엔지니어링 계층
- 주기 혼합(M/Q) 정합: 분기 타깃 대비 월별 설명변수 집계 규칙 표준화
- 자동 변환 후보군: level/diff/log-diff/yoy/qoq saar를 모두 생성 후 모델이 선택
- 릴리즈 캘린더 기반 ragged-edge 마스킹 자동화

### C. 모델 계층(앙상블)
- Baseline: AR(분기 GDP 단변량), Bridge Equation
- Core: Dynamic Factor Model(DFM), MIDAS 회귀
- ML 보완: XGBoost/LightGBM(특징 중요도·비선형 포착)
- 결합: 시점별 성능가중 평균(rolling window weighting)

### D. 평가/검증 계층
- Pseudo real-time backtest (vintage 기준)
- 하위지표: GDP 총량 + 민간소비/투자/순수출 기여도
- 지표: RMSE, MAE, 방향성 정확도, turning-point 탐지율

### E. 서빙/리포팅 계층
- 주간/월간 자동 리포트 생성
- nowcast 변경분 decomposition(“이번주 +0.1%p의 원인”)
- 대시보드(API + Streamlit/FastAPI) 제공

---

## 4) 단계별 실행 로드맵 (권장)

### Phase 1 (2~3주): 재설계 기반 구축
- 레포 구조 재편: `src/kgdpnow/{ingest,features,models,eval,report}`
- 데이터 스키마 정의 + 유효성 검사(pydantic/pandera)
- 빈티지 저장 규칙 확정(예: `asof_date` 파티션)

### Phase 2 (3~5주): 모델 다변화 + 백테스트 자동화
- DFM, MIDAS, Bridge 기본모델 구현
- 발표일 캘린더 반영한 pseudo real-time 평가 파이프라인 구축
- 모델별 성능 대시보드 및 앙상블 가중치 자동 업데이트

### Phase 3 (2~4주): 운영화
- 스케줄러(Airflow/Prefect/GitHub Actions) 연동
- 데이터 이상탐지/모델 드리프트 모니터링
- 보고서 자동 생성 및 배포

---

## 5) 빠른 PoC 우선순위 (실전형)

1. **목표변수**: 실질 GDP qoq saar nowcast
2. **설명변수**: 산업생산, 소매판매, 수출입, 고용, 금리/환율, 심리지수
3. **모델 3종**: Bridge + DFM + XGBoost
4. **평가기간**: 최근 8~10년 vintage 백테스트
5. **성공기준**: 기존 방식 대비 RMSE 10% 이상 개선 + turning point 조기탐지

---

## 6) 현재 코드 기준 우선 개선 포인트

- `최종.py`를 기능 단위 모듈로 분할하여 실험/운영 경계를 분리
- 전처리 규칙을 하드코딩 로직에서 설정 파일(`yaml`) 기반으로 전환
- 데이터 수집 단계에 재시도/지수백오프/응답 캐싱 도입
- 월별 데이터와 분기 타깃 결합 시점의 ragged-edge 처리 명시화
- 모델 성능 평가를 함수화해 실행마다 자동 리포트 산출

---

## 7) 추천 기술 스택

- **언어/패키지**: Python, pandas/polars, statsmodels, linearmodels, lightgbm/xgboost
- **파이프라인**: Prefect 또는 Airflow
- **품질관리**: pytest, pandera, pre-commit, ruff
- **서빙**: FastAPI + Streamlit
- **저장소**: Parquet + DuckDB (초기), 필요 시 Postgres 확장

---

## 8) 결론

기존 프로젝트는 "지표 수집과 기본 nowcast 실험" 관점에서 좋은 출발점입니다. 다만 새롭게 만들 때는
**(1) 빈티지 기반 평가체계, (2) 모델 앙상블, (3) 운영 자동화**를 중심으로 재설계해야 실제 활용 가능한 한국형 GDPNow로 발전할 수 있습니다.
