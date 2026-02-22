# K-GDPNow

한국형 GDPNow 재구축 프로젝트입니다.

## Data Layer 실행

### 1) `.env` 파일 생성
1) `.env` 파일 생성

```bash
cp .env.example .env
# .env 파일에서 BOK_API_KEY 값을 본인 키로 수정
```

### 2) 실행 환경 준비 후 수집 실행
2) 실행 환경 준비 후 수집 실행

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_data_layer.py
```

> `run_data_layer.py`와 `BOKECOSIngestor` 모두 `.env`에서 `BOK_API_KEY`를 읽습니다.

실행 후 데이터는 `data/raw`, `data/staging`, `data/vintage_store`에 `as_of_date=YYYY-MM-DD` 파티션으로 저장됩니다.

## 지표 CSV

- 기본 월간 nowcast 후보 지표는 `data_list.csv`에 포함되어 있습니다.
- 컬럼: `Name`, `API_Code`, `Item_code`, `Item_code_2`
- ECOS 코드 체계 변경 가능성이 있으므로, 실제 운영 전 `API_Code/Item_code` 유효성 점검을 권장합니다.

## 코드 유효성 점검 (중요)

19개 이상 스킵되는 경우 대부분 `API_Code`/`Item_code` 조합이 틀린 경우입니다.
아래 스크립트로 통계코드별 유효한 ITEM_CODE를 먼저 조회하세요.

```bash
python discover_ecos_items.py 901Y009 --out ecos_901Y009_items.csv
```

조회된 `ITEM_CODE`를 `data_list.csv`에 반영한 뒤 다시 `python run_data_layer.py`를 실행하면 됩니다.
