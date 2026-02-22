# K-GDPNow

한국형 GDPNow 재구축 프로젝트입니다.

## Data Layer 실행

1) `.env` 파일 생성

```bash
cp .env.example .env
# .env 파일에서 BOK_API_KEY 값을 본인 키로 수정
```

2) 실행 환경 준비 후 수집 실행

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_data_layer.py
```

> `run_data_layer.py`와 `BOKECOSIngestor` 모두 `.env`에서 `BOK_API_KEY`를 읽습니다.

실행 후 데이터는 `data/raw`, `data/staging`, `data/vintage_store`에 `as_of_date=YYYY-MM-DD` 파티션으로 저장됩니다.
