# Macro Regime Allocation

거시경제 국면 인식 기반 투자전략 및 자산배분 연구를 위한 Python 프로젝트입니다.

## 구성

- `src/data`: 매크로/자산 데이터 로딩, 발표시차 정렬
- `src/features`: 전처리, 변환율, rolling z-score, 복합지수
- `src/models`: nowcasting(VBAR/VAR 스타일) 베이스라인
- `src/regime`: 국면 분류 로직
- `src/strategy`: 국면별 목표 비중 규칙
- `src/backtest`: 월간 리밸런싱 기반 백테스트
- `src/visualization`: 시각화 및 리포트 생성 유틸리티

## 빠른 시작

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
python run_mvp.py
```

## 데이터 범위

- 샘플 데이터는 월별 **2000-01 ~ 2025-12**(총 312개월)로 구성되어,
  초기 MVP 대비 더 긴 구간에서 국면별 통계와 백테스트를 확인할 수 있습니다.

## 결과물

`python run_mvp.py` 실행 시 아래 파일이 생성됩니다.

- `data/processed/composites.csv`
- `data/processed/regimes.csv`
- `data/processed/backtest_results.csv`
- `reports/mvp_report.md`
- `reports/composite_scores.png`
- `reports/regime_distribution.png`
- `reports/wealth_curve.png`

## MVP 범위

1. 월별 매크로 지표 샘플 로딩 및 발표시차 정렬
2. rolling z-score 및 하위 인덱스 산출
3. 성장/물가 2축 기반 4개 국면 분류
4. 국면별 자산배분 룰 적용 백테스트
5. 섹션형 설명 + 표/차트 포함 리포트 자동 생성

## 다음 단계

- 실제 데이터 소스(FRED, ECOS, OECD 등) 커넥터 추가
- 발표일/관측일 분리 기반 real-time dataset 확장
- HMM/Markov regime model 비교
- 팩터(가치/모멘텀/캐리/디펜시브) 성과 분해 추가
