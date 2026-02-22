# K-GDPNow

한국형 GDPNow 재구축 프로젝트입니다.

## Data Layer 실행

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export BOK_API_KEY="YOUR_KEY"
python run_data_layer.py
```

실행 후 데이터는 `data/raw`, `data/staging`, `data/vintage_store`에 `as_of_date=YYYY-MM-DD` 파티션으로 저장됩니다.
