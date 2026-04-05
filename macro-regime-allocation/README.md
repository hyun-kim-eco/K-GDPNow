# Macro Regime Allocation

거시경제 국면 인식 기반 투자전략 및 자산배분 연구를 위한 Python 프로젝트입니다.

## 구성

- `src/data`: 매크로/자산 데이터 로딩, 발표시차 정렬
- `src/features`: 전처리, 변환율, rolling z-score, 복합지수
- `src/models`: nowcasting(VBAR/VAR 스타일) 베이스라인
- `src/regime`: 국면 분류 로직
- `src/strategy`: 국면별 목표 비중 규칙
- `src/backtest`: 월간 리밸런싱 기반 백테스트
- `src/visualization`: 시각화 유틸리티

## 빠른 시작

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
```

## MVP 범위

1. 월별 매크로 지표 샘플 로딩
2. rolling z-score 및 하위 인덱스 산출
3. 성장/물가 2축 기반 4개 국면 분류
4. 국면별 자산배분 룰 적용 백테스트

## 다음 단계

- 실제 데이터 소스(FRED, ECOS, OECD 등) 커넥터 추가
- 발표일/관측일 분리 기반 real-time dataset 확장
- HMM/Markov regime model 비교
