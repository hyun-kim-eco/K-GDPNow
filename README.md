# K-GDPNow

한국형 GDPNow 재구축 프로젝트입니다.

## Data Layer 실행

### 1) `.env` 파일 생성

```bash
cp .env.example .env
# .env 파일에서 BOK_API_KEY 값을 본인 키로 수정
```

### 2) 실행 환경 준비 후 수집 실행

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
- `data_list.csv`는 **UTF-8 BOM(utf-8-sig)**로 저장되어 윈도우 엑셀에서 한글 깨짐을 줄였습니다.

## 코드 유효성 점검 (중요)

다수 지표가 스킵되는 경우 대부분 `API_Code`/`Item_code` 조합 문제입니다.
아래처럼 월간(`--cycle M`) + 키워드(`--contains`)로 좁혀서 필요한 항목만 조회하세요.

```bash
python discover_ecos_items.py 901Y009 --cycle M --contains 계절 --out ecos_901Y009_m.csv
```

- 기본값으로 `--cycle M`이 적용되어, 불필요한 분기/연간 항목을 줄입니다.
- 필터가 너무 강해서 결과가 없으면 `--cycle ''`로 전체를 조회하세요.
