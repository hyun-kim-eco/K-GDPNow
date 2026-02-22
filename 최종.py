import pandas as pd
import numpy as np
import os
import requests
from datetime import datetime
from dotenv import load_dotenv
from statsmodels.tsa.stattools import adfuller
from scipy.stats import linregress
from pandas.tseries.offsets import MonthEnd
import warnings
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from statsmodels.tsa.ar_model import AutoReg
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from statsmodels.api import OLS
from statsmodels.api import add_constant
from sklearn.preprocessing import StandardScaler
from itertools import product
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.var_model import VARResults

warnings.filterwarnings("ignore")

# 데이터 저장 경로 설정
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

def create_data_list():
    """CSV 파일을 통해 API에 사용할 data_list를 생성"""
    # CSV 파일 불러오기
    csv_path = os.path.join(os.path.dirname(__file__), "data_list.csv")
    df = pd.read_csv(csv_path).fillna("")

    # 리스트 생성
    data_list = []

    for _, row in df.iterrows():
        entry = {
            "name": row["Name"].strip(),
            "api_code": row["API_Code"].strip(),
            "item_code": row["Item_code"].strip(),
            "item_code_2": str(row["Item_code_2"]).strip()
        }
        data_list.append(entry)

    return data_list

def create_quarterly_data_list():
    """분기별 GDP 구성요소 데이터를 위한 data_list 생성"""
    quarterly_data_list = [
        {"name": "민간소비", "api_code": "200Y108", "item_code": "1010110"},
        {"name": "정부소비", "api_code": "200Y108", "item_code": "1010120"},
        {"name": "건설투자", "api_code": "200Y108", "item_code": "1020111"},
        {"name": "설비투자", "api_code": "200Y108", "item_code": "1020112"},
        {"name": "지식재산생산물투자", "api_code": "200Y108", "item_code": "1020113"},
        {"name": "재화수출", "api_code": "200Y108", "item_code": "1030110"},
        {"name": "서비스수출", "api_code": "200Y108", "item_code": "1030120"},
        {"name": "재화수입", "api_code": "200Y108", "item_code": "1040110"},
        {"name": "서비스수입", "api_code": "200Y108", "item_code": "1040120"}
    ]
    return quarterly_data_list

def collect_quarterly_data(data_list):
    """분기별 GDP 구성요소 데이터 수집"""
    # 1. API 키 로드
    load_dotenv()
    API_KEY = os.getenv("BOK_API_KEY")

    if not API_KEY:
        print("❌ .env 파일에 BOK_API_KEY가 없습니다.")
        return None

    # 2. 기본 설정
    base_url = "https://ecos.bok.or.kr/api/StatisticSearch"
    start_date = "1995Q1"  # 데이터 조회 시작 분기
    current_date = datetime.today()
    end_date = f"{current_date.year}Q{(current_date.month-1)//3 + 1}"  # 현재 분기

    df_list = []

    if not data_list:
        print("❗ quarterly_data_list가 비어있습니다.")
        return None

    for item in data_list:
        name = item["name"]
        api_code = item["api_code"]
        item_code = item["item_code"]

        url = f"{base_url}/{API_KEY}/json/kr/1/1000/{api_code}/Q/{start_date}/{end_date}/{item_code}"
        print(f"📡 API 요청: {name} ({url})")

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.ReadTimeout:
            print(f"❌ 타임아웃 발생: {name}")
            continue
        except requests.exceptions.HTTPError as http_err:
            print(f"❌ HTTP 에러 발생: {name} (상태 코드: {response.status_code})")
            try:
                error_content = response.json()
                if "RESULT" in error_content and "MESSAGE" in error_content["RESULT"]:
                    print(f"API 메시지: {error_content['RESULT']['MESSAGE']}")
            except:
                pass
            continue
        except Exception as e:
            print(f"❌ 요청 실패 또는 JSON 에러: {name}\n에러: {e}")
            continue

        if 'StatisticSearch' in data and 'row' in data['StatisticSearch']:
            rows = data['StatisticSearch']['row']
            if not rows:
                print(f"ℹ️ 데이터 없음: {name}")
                continue

            result = []
            for row in rows:
                date = row.get('TIME')
                value = row.get('DATA_VALUE', '')
                # None 값을 안전하게 처리
                if value is not None:
                    value = str(value).replace(',', '').strip()
                else:
                    value = ''
                if 'Q' in str(date) and value:
                    year, qtr = date.split('Q')
                    # 분기말 날짜로 변환 (3/31, 6/30, 9/30, 12/31)
                    month = int(qtr) * 3
                    quarter_end_date = pd.Timestamp(int(year), month, 1) + pd.offsets.QuarterEnd(0)
                    result.append({
                        'date': quarter_end_date,
                        'value': float(value)
                    })

            if result:
                df = pd.DataFrame(result)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df = df.sort_index()
                df.columns = [name]
                df_list.append(df)
                print(f"✅ 데이터 수집 성공: {name} ({len(df)}개 행)")
            else:
                print(f"ℹ️ 데이터 없음: {name}")

    # 3. 결과 저장
    if df_list:
        result_df = pd.concat(df_list, axis=1)
        result_df = result_df.sort_index()
        output_file = os.path.join(DATA_DIR, "quarterly_gdp_components.csv")
        result_df.to_csv(output_file, encoding="utf-8-sig", float_format='%.6f')
        print(f"🎉 분기별 데이터 수집 및 정제 완료! 저장됨: {output_file}")
        return result_df
    else:
        print("❌ 수집된 분기별 데이터가 없습니다.")
        return None

def collect_raw_data(data_list):
    """BOK API를 통해 Raw data를 수집"""
    # 1. API 키 로드 (.env 파일에서 불러오기)
    load_dotenv()
    API_KEY = os.getenv("BOK_API_KEY")

    # 2. 기본 설정
    base_url = "https://ecos.bok.or.kr/api/StatisticSearch"
    start_date = "199501"  # 데이터 조회 시작 월 (YYYYMM)
    end_date = datetime.today().strftime("%Y%m")  # 데이터 조회 종료 월 (현재 월)

    df_list = []

    if not data_list:
        print("❗ data_list가 비어있습니다. 수집할 변수 정보를 data_list에 정의해주세요.")
        return None

    for item in data_list:
        name = item["name"]
        api_code = item["api_code"]
        item_code = item["item_code"]
        item_code_2 = item.get("item_code_2", "").strip()

        if item_code_2:
            url = f"{base_url}/{API_KEY}/json/kr/1/100000/{api_code}/M/{start_date}/{end_date}/{item_code}/{item_code_2}"
        else:
            url = f"{base_url}/{API_KEY}/json/kr/1/100000/{api_code}/M/{start_date}/{end_date}/{item_code}"

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.ReadTimeout:
            print(f"❌ 타임아웃 발생: {name}\nURL: {url}")
            continue
        except requests.exceptions.HTTPError as http_err:
            print(f"❌ HTTP 에러 발생: {name} (상태 코드: {response.status_code})\nURL: {url}\n에러: {http_err}")
            try:
                error_content = response.json()
                if "RESULT" in error_content and "MESSAGE" in error_content["RESULT"]:
                    print(f"API 메시지: {error_content['RESULT']['MESSAGE']}")
            except:
                pass
            continue
        except Exception as e:
            print(f"❌ 요청 실패 또는 JSON 에러: {name}\nURL: {url}\n에러: {e}")
            continue

        if 'StatisticSearch' in data and 'row' in data['StatisticSearch']:
            rows = data['StatisticSearch']['row']
            if not rows:
                print(f"ℹ️ 데이터 없음 (API 응답 row 비어있음): {name}")
                continue

            df_api = pd.DataFrame(rows)

            if not df_api.empty and 'TIME' in df_api.columns and 'DATA_VALUE' in df_api.columns:
                try:
                    s_data = df_api.set_index('TIME')['DATA_VALUE']
                    s_data = s_data.map(lambda x: str(x).replace(",", "").replace(" ", "").strip() if pd.notna(x) else x)
                    s_data = pd.to_numeric(s_data, errors='coerce')

                    df_row = pd.DataFrame([s_data])
                    df_row.index = [name]

                    df_row.columns = pd.to_datetime(df_row.columns, format="%Y%m") + pd.offsets.MonthEnd(0)
                    df_row.columns = df_row.columns.strftime("%Y%m%d")

                    df_list.append(df_row)
                except Exception as e:
                    print(f"❗ 데이터 처리 실패: {name}\n에러: {e}")
            else:
                print(f"ℹ️ 데이터 형식 오류 (TIME 또는 DATA_VALUE 컬럼 누락): {name}")
        elif 'RESULT' in data and 'MESSAGE' in data['RESULT']:
            print(f"❌ API 에러: {name} - {data['RESULT']['MESSAGE']}")
        else:
            print(f"ℹ️ 데이터 없음 (알 수 없는 형식 또는 빈 데이터): {name}")

    # 3. 결과 저장
    if df_list:
        result_df = pd.concat(df_list)
        result_df = result_df.sort_index(axis=1)
        output_file = os.path.join(DATA_DIR, "monthly_economic_indicators.csv")
        result_df.to_csv(output_file, encoding="utf-8-sig", float_format='%.6f')
        print(f"🎉 수집 및 정제 완료! 저장됨: {output_file}")
        return result_df
    else:
        print("❌ 수집된 데이터가 없습니다. 파라미터 또는 data_list를 확인하세요.")
        return None

def preprocess_raw_data():
    """Raw data 전처리 및 정상성 분석"""
    # 파일 불러오기
    input_path = os.path.join(DATA_DIR, "monthly_economic_indicators.csv")
    df = pd.read_csv(input_path, index_col=0)
    df = df.T

    # 인덱스를 날짜로 변환
    try:
        df.index = pd.to_datetime(df.index, format="%Y%m%d")
    except ValueError:
        df.index = pd.to_datetime(df.index, format="%Y%m") + MonthEnd(0)

    # 문자열 → float 변환
    df = df.applymap(lambda x: str(x).replace(",", "").replace(" ", "").strip() if isinstance(x, str) else x)
    df = df.apply(pd.to_numeric, errors='coerce')

    # 결과 저장용
    results = []
    transformed = {}
    standardized = {}

    # 변수별 처리
    for col in df.columns:
        series = df[col].astype(float)
        
        # 초기화
        p_value = np.nan
        is_stationary = False
        trend_type = ""
        trans_type = "exclude"
        
        can_log_transform_flag = False
        actual_pos_series_len = np.nan
        lin_r2_report = np.nan
        exp_r2_report = np.nan

        # 로그 변환 가능 여부 판단
        series_dropna_for_check = series.dropna()
        if not series_dropna_for_check.empty:
            can_log_transform_flag = (series_dropna_for_check > 0).all()

        if series_dropna_for_check.shape[0] < 20:
            results.append({
                "variable": col,
                "adf_pvalue": np.nan,
                "stationary": "Insufficient Data",
                "can_log_transform": can_log_transform_flag,
                "pos_series_len": np.nan,
                "lin_r2": np.nan,
                "exp_trend_r2": np.nan,
                "trend_type": "",
                "transformation": "exclude"
            })
            transformed[col] = series
            standardized[col] = series
            continue

        # ADF 검정
        if len(series_dropna_for_check) > 0:
            p_value = adfuller(series_dropna_for_check)[1]
            is_stationary = p_value < 0.05

        if is_stationary:
            transformed[col] = series
            standardized[col] = (series - series.mean()) / series.std()
            trend_type = "-"
            trans_type = "level"
        else:
            # 비정상 시계열 처리
            t = np.arange(len(series_dropna_for_check))
            
            if len(series_dropna_for_check) >= 2:
                try:
                    lin_r2_report = linregress(t, series_dropna_for_check).rvalue ** 2
                except ValueError:
                    lin_r2_report = 0.0
            
            pos_series = series[series > 0].dropna()
            actual_pos_series_len = len(pos_series)

            if len(pos_series) >= 2:
                try:
                    log_transformed_pos_series = np.log(pos_series)
                    if np.var(log_transformed_pos_series) > 1e-12:
                        exp_r2_report = linregress(np.arange(len(pos_series)), log_transformed_pos_series).rvalue ** 2
                    else:
                        exp_r2_report = 0.0 if len(pos_series) > 0 else np.nan
                except ValueError:
                    exp_r2_report = 0.0

            lin_r2_for_decision = lin_r2_report if not np.isnan(lin_r2_report) else 0
            log_r2_for_decision = exp_r2_report if (not np.isnan(exp_r2_report) and actual_pos_series_len > 10) else 0
            
            if can_log_transform_flag and log_r2_for_decision > lin_r2_for_decision:
                trend_type = "exponential"
                trans_type = "log_diff"
                transformed[col] = np.log(series).diff()
                standardized[col] = (transformed[col] - transformed[col].mean()) / transformed[col].std()
            else:
                trend_type = "linear"
                trans_type = "level_diff"
                transformed[col] = series.diff()
                standardized[col] = (transformed[col] - transformed[col].mean()) / transformed[col].std()

        results.append({
            "variable": col,
            "adf_pvalue": round(p_value, 4) if not np.isnan(p_value) else np.nan,
            "stationary": "Yes" if is_stationary else "No",
            "can_log_transform": can_log_transform_flag,
            "pos_series_len": actual_pos_series_len,
            "lin_r2": round(lin_r2_report, 4) if not np.isnan(lin_r2_report) else np.nan,
            "exp_trend_r2": round(exp_r2_report, 4) if not np.isnan(exp_r2_report) else np.nan,
            "trend_type": trend_type,
            "transformation": trans_type
        })

    # 변환된 데이터프레임 생성
    df_transformed = pd.DataFrame(transformed)
    df_standardized = pd.DataFrame(standardized)

    for col in df.columns:
        if col not in df_transformed:
            df_transformed[col] = df[col]
            df_standardized[col] = df[col]

    # 저장
    df_transformed.to_csv(os.path.join(DATA_DIR, "monthly_indicators_transformed.csv"), float_format="%.6f")
    df_standardized.to_csv(os.path.join(DATA_DIR, "monthly_indicators_standardized.csv"), float_format="%.6f")
    pd.DataFrame(results).to_excel(os.path.join(DATA_DIR, "monthly_indicators_stationarity_report.xlsx"), index=False)

    print("✅ 전처리 및 정상성 분석 완료! (로그 변환 가능 여부(NaN제외 모든 값 양수), 양수 시리즈 길이, R2 값 포함)")
    print(f"결과 파일 저장 위치: {os.path.abspath(DATA_DIR)}")
    
    return df_transformed, df_standardized

def preprocess_quarterly_data():
    """분기별 데이터 전처리 및 정상성 분석"""
    # 파일 불러오기
    input_file = os.path.join(DATA_DIR, "quarterly_gdp_components.csv")
    df = pd.read_csv(input_file, index_col=0, parse_dates=True)
    df = df.interpolate(method='linear').dropna()

    # 결과 저장용
    results = []
    transformed = pd.DataFrame(index=df.index)

    for col in df.columns:
        series = df[col].copy()
        result = {"변수": col}

        try:
            adf_result = adfuller(series, autolag='AIC')
            pvalue = adf_result[1]
            result["정상성"] = "정상" if pvalue < 0.05 else "비정상"
            result["ADF p-value"] = round(pvalue, 4)
        except Exception as e:
            result["정상성"] = "오류"
            result["ADF p-value"] = None
            result["오류 메시지"] = str(e)
            results.append(result)
            continue

        if (series <= 0).any():
            transformed[col] = series.diff()
            result["변환방법"] = "level_diff (log 불가)"
            result["선형 R2"] = None
            result["로그 R2"] = None
        else:
            try:
                t = np.arange(len(series))
                df_trend = pd.DataFrame({"y": series.values, "t": t})
                df_trend["log_y"] = np.log(df_trend["y"])

                linear_model = OLS(df_trend["y"], add_constant(df_trend["t"])).fit()
                log_model = OLS(df_trend["log_y"], add_constant(df_trend["t"])).fit()

                linear_r2 = linear_model.rsquared
                log_r2 = log_model.rsquared

                result["선형 R2"] = round(linear_r2, 4)
                result["로그 R2"] = round(log_r2, 4)

                if log_r2 >= 0.8:
                    transformed[col] = np.log(series).diff()
                    result["변환방법"] = "log_diff"
                else:
                    transformed[col] = series.diff()
                    result["변환방법"] = "level_diff (log R2 낮음)"
            except Exception as e:
                transformed[col] = series.diff()
                result["변환방법"] = "level_diff (log 오류)"
                result["선형 R2"] = None
                result["로그 R2"] = None
                result["오류 메시지"] = str(e)

        results.append(result)

    # 결측치 제거
    transformed.dropna(inplace=True)

    # 분기말 날짜로 변환
    transformed.index = transformed.index + pd.offsets.QuarterEnd(0)

    # 저장
    transformed.to_csv(os.path.join(DATA_DIR, "quarterly_gdp_components_transformed.csv"), encoding='utf-8-sig', float_format='%.6f')
    pd.DataFrame(results).to_excel(os.path.join(DATA_DIR, "quarterly_gdp_components_stationarity_report.xlsx"), index=False)

    print("✅ 분기별 데이터 전처리 완료")
    print(f"- 변환된 데이터: {os.path.join(DATA_DIR, 'quarterly_gdp_components_transformed.csv')}")
    print(f"- 정상성 보고서: {os.path.join(DATA_DIR, 'quarterly_gdp_components_stationarity_report.xlsx')}")
    
    return transformed

def run_pca_ar_forecast(data_df, k_factors, factor_ar_order, current_date_for_run, as_of_time_str):
    """PCA 및 AR 모델 기반 요인 추출/예측"""
    print(f"\n--- Running PCA & AR Forecast using data up to {current_date_for_run.strftime('%Y-%m-%d')} (for files as of {as_of_time_str}) ---")
    
    min_obs_needed = factor_ar_order + 20
    if data_df.shape[0] < min_obs_needed:
        print(f"데이터가 충분하지 않아 ({data_df.shape[0]} < {min_obs_needed}) PCA 및 AR 모델을 실행할 수 없습니다.")
        return None, None

    # 정규화된 데이터 사용
    scaled_df = data_df.copy()
    
    # PCA를 위한 결측치 처리
    imputer = SimpleImputer(strategy='mean')
    scaled_df_imputed_for_pca = imputer.fit_transform(scaled_df)
    scaled_df_imputed_for_pca = pd.DataFrame(scaled_df_imputed_for_pca, index=scaled_df.index, columns=scaled_df.columns)

    if scaled_df_imputed_for_pca.isnull().values.any():
        print("오류: PCA를 위한 데이터 imputation 후에도 NaN이 남아있습니다.")
        return None, None
        
    try:
        # PCA 실행
        pca = PCA(n_components=k_factors)
        factors_pca_estimated = pca.fit_transform(scaled_df_imputed_for_pca)
        
        factor_est_cols = [f'Factor_PCA_Est_{i+1}' for i in range(k_factors)]
        factors_df = pd.DataFrame(factors_pca_estimated, index=scaled_df.index, columns=factor_est_cols)

    except Exception as e:
        print(f"PCA 실행 중 오류 발생: {e}")
        return None, None

    # 추출된 요인에 AR(p) 모델 적합 및 미래 요인 예측
    last_obs_date_in_current_run = scaled_df.index[-1]
    
    # 마지막 관측일이 속한 분기의 마지막 달까지 예측
    last_obs_quarter_end = (last_obs_date_in_current_run + pd.offsets.QuarterEnd(0)).normalize()
    forecast_start_date = last_obs_date_in_current_run + pd.offsets.MonthEnd(0) + pd.offsets.MonthBegin(1)
    forecast_end_date = last_obs_quarter_end
    forecast_index = pd.date_range(start=forecast_start_date, end=forecast_end_date, freq='M')
    forecast_horizon = len(forecast_index)

    print(f"  데이터 마지막 관측일: {last_obs_date_in_current_run.strftime('%Y-%m-%d')}")
    print(f"  예측 시작일: {forecast_start_date.strftime('%Y-%m-%d')}")
    print(f"  예측 종료일: {forecast_end_date.strftime('%Y-%m-%d')}")
    print(f"  예측할 기간(월 단위): {forecast_horizon}")

    factor_forecasts_df = pd.DataFrame() 
    if forecast_horizon > 0:
        try:
            # 각 요인별로 AR(factor_ar_order) 모델 적합 및 예측
            all_factor_forecasts = []
            for i in range(k_factors):
                factor_series = factors_df.iloc[:, i]
                ar_model = AutoReg(factor_series, lags=factor_ar_order, old_names=False)
                ar_results = ar_model.fit()
                forecast_start_idx = len(factor_series) 
                forecast_end_idx = len(factor_series) + forecast_horizon - 1
                factor_forecast_values = ar_results.predict(start=forecast_start_idx, end=forecast_end_idx)
                all_factor_forecasts.append(factor_forecast_values)
            
            factor_forecast_values_combined = pd.concat(all_factor_forecasts, axis=1).values
            
            factor_fcst_cols = [f'Factor_PCA_Fcst_{i+1}' for i in range(k_factors)]
            if len(forecast_index) == factor_forecast_values_combined.shape[0]:
                factor_forecasts_df = pd.DataFrame(factor_forecast_values_combined, index=forecast_index, columns=factor_fcst_cols)
            else:
                print(f"  Warning: PCA AR Forecast index length ({len(forecast_index)}) and forecast values length ({factor_forecast_values_combined.shape[0]}) mismatch. Skipping forecast DF creation.")
                min_len = min(len(forecast_index), factor_forecast_values_combined.shape[0])
                if min_len > 0:
                    factor_forecasts_df = pd.DataFrame(factor_forecast_values_combined[:min_len], index=forecast_index[:min_len], columns=factor_fcst_cols)

        except Exception as e:
            print(f"  요인 AR 모델 적합 또는 예측 중 오류 발생: {e}")
            factor_forecasts_df = pd.DataFrame()
    
    # 결과 저장
    factors_df.columns = ['Factor']  # 컬럼명을 'Factor'로 통일
    factors_df.to_csv(os.path.join(DATA_DIR, f"common_factor_historical.csv"), float_format="%.6f")
    if not factor_forecasts_df.empty:
        factor_forecasts_df.columns = ['Factor']  # 컬럼명을 'Factor'로 통일
        factor_forecasts_df.to_csv(os.path.join(DATA_DIR, f"common_factor_forecast.csv"), float_format="%.6f")
        print(f"  예측된 PCA 요인 (다음 {factor_forecasts_df.shape[0]} 기간, 시작: {factor_forecasts_df.index[0].strftime('%Y-%m-%d')}):\n{factor_forecasts_df.head()}")
        # 실측과 예측을 하나로 합쳐서 저장 (인덱스 중복 방지)
        combined_df = pd.concat([factors_df, factor_forecasts_df]).loc[~pd.concat([factors_df, factor_forecasts_df]).index.duplicated(keep='first')]
        combined_df.to_csv(os.path.join(DATA_DIR, f"common_factor_combined.csv"), float_format="%.6f")
        print(f"  실측+예측 공통요인 통합 데이터 저장 완료: common_factor_combined.csv")
    else:
        print("  PCA 요인 예측에 실패했습니다.")
        # 예측이 없을 때도 실측만 저장
        factors_df.to_csv(os.path.join(DATA_DIR, f"common_factor_combined.csv"), float_format="%.6f")
        print(f"  실측 공통요인만 저장: common_factor_combined.csv")
    print(f"추정된 PCA 요인 (가장 최근 값은 {factors_df.index[-1].strftime('%Y-%m-%d')}):\n{factors_df.tail()}")

    return factors_df, factor_forecasts_df

def create_monthly_data_for_arx():
    """ARX 모형을 위한 월별 데이터 준비
    
    Returns:
    --------
    pd.DataFrame
        월별 경제지표 데이터
    """
    print("\n=== ARX 모형을 위한 월별 데이터 준비 ===")
    
    # 파일 읽기
    input_file = os.path.join(DATA_DIR, "monthly_economic_indicators.csv")
    df = pd.read_csv(input_file, index_col=0)
    
    # 행/열 전환 (지표명이 행, 날짜가 열 → 날짜가 행, 지표명이 열)
    df = df.T
    
    # 인덱스를 날짜로 변환
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    
    # 데이터 정렬
    df = df.sort_index()
    
    print(f"데이터 기간: {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"변수 수: {len(df.columns)}")
    print(f"관측치 수: {len(df)}")
    
    return df

def fit_arx_model(monthly_data, factor_data):
    """ARX 모형 적합 및 최적 lag 선택
    
    Parameters:
    -----------
    monthly_data : pd.DataFrame
        월별 경제지표 데이터
    factor_data : pd.DataFrame
        공통요인 데이터
        
    Returns:
    --------
    dict
        각 변수별 최적 lag 조합과 모형 결과
    """
    results = {}
    
    # 모든 변수에 대해 ARX 모형 적합
    for target_var in monthly_data.columns:
        print(f"\n=== ARX 모형 결과 (타겟 변수: {target_var}) ===")
        try:
            # 로그 변환
            log_monthly = np.log(monthly_data)
            y = log_monthly[target_var].dropna()
            
            # x1: 타겟 변수의 lag 1~12
            X_lags = pd.concat([y.shift(lag) for lag in range(1, 13)], axis=1)
            X_lags.columns = [f'{target_var}_lag{lag}' for lag in range(1, 13)]
            
            # x2: 공통요인(예: PCA factor)의 lag 0~3
            factor = factor_data.loc[y.index]  # y와 인덱스 맞추기
            factor_lags = pd.concat([factor.shift(lag) for lag in range(0, 4)], axis=1)
            factor_lags.columns = [f'factor_lag{lag}' for lag in range(0, 4)]
            
            # AIC로 최적 lag 조합 선택
            best_aic = float('inf')
            best_lags = None
            best_model = None
            
            # lag 조합 생성 (타겟 변수 lag 1~12, 공통요인 lag 0~3)
            target_lag_range = range(1, 13)
            factor_lag_range = range(0, 4)
            
            for t_lag, f_lag in product(target_lag_range, factor_lag_range):
                # 현재 lag 조합으로 X 생성
                X_current = pd.concat([X_lags.iloc[:, :t_lag], factor_lags.iloc[:, :f_lag+1]], axis=1)
                
                # 결측치 제거 (X와 y의 인덱스 맞추기)
                X_current = X_current.dropna()
                y_current = y.loc[X_current.index]
                
                # OLS 회귀
                model = OLS(y_current, add_constant(X_current)).fit()
                aic = model.aic
                
                # AIC가 더 낮으면 업데이트
                if aic < best_aic:
                    best_aic = aic
                    best_lags = (t_lag, f_lag)
                    best_model = model
            
            # 결과 저장
            results[target_var] = {
                'best_lags': best_lags,
                'best_aic': best_aic,
                'model': best_model
            }
            
            # 최적 모형 결과 출력
            print(f"최적 lag 조합: 타겟 변수 lag {best_lags[0]}, 공통요인 lag {best_lags[1]}")
            print(f"최적 AIC: {best_aic:.2f}")
            
        except Exception as e:
            print(f"Error processing {target_var}: {str(e)}")
            continue
    
    # 결과를 CSV 파일로 저장
    results_df = pd.DataFrame(results).T
    results_df.to_csv(os.path.join(DATA_DIR, 'arx_model_results.csv'), index=True)
    
    return results

def create_forecast_dataframe(monthly_data):
    """마지막 월이 속한 분기의 마지막 월 말일 하나만 예측하도록 데이터프레임을 생성합니다.
    기존 monthly_data의 컬럼을 유지하고, 인덱스에 예측 월을 추가합니다."""
    last_obs_date = monthly_data.index[-1]
    last_obs_quarter_end = (last_obs_date + pd.offsets.QuarterEnd(0)).normalize()
    # 기존 인덱스 + 예측 월
    new_index = monthly_data.index.append(pd.DatetimeIndex([last_obs_quarter_end]))
    # 기존 monthly_data와 동일한 컬럼, 값은 모두 NaN
    forecast_df = pd.DataFrame(index=new_index, columns=monthly_data.columns)
    # 기존 데이터는 monthly_data에서 복사
    forecast_df.loc[monthly_data.index] = monthly_data.values
    return forecast_df

def forecast_missing_values(monthly_data, factor_data, arx_results, n_months=5):
    """
    최근 n개월 구간 내 결측치를 ARX 모델을 사용해, 가장 과거 시점부터 순차적으로 예측하여 모두 채웁니다.
    마지막 관측월이 속한 분기의 마지막 월까지 예측합니다.
    """
    print(f"\n=== ARX 모델을 사용한 결측치 순차 예측 시작 ===")
    forecast_df = monthly_data.copy()
    last_date = forecast_df.index[-1]
    
    # 마지막 관측월이 속한 분기의 마지막 월 계산
    last_quarter_end = (last_date + pd.offsets.QuarterEnd(0)).normalize()
    
    # 예측 기간 설정 (최근 n개월과 분기말 중 더 긴 기간 사용)
    start_date = min(last_date - pd.DateOffset(months=n_months), last_quarter_end - pd.DateOffset(months=n_months))
    end_date = max(last_date, last_quarter_end)
    
    # 예측 기간의 모든 월말 날짜 생성
    period_index = pd.date_range(start=start_date, end=end_date, freq='M')
    period_index = [date + pd.offsets.MonthEnd(0) for date in period_index]
    
    print(f"예측 기간: {period_index[0].strftime('%Y-%m-%d')} ~ {period_index[-1].strftime('%Y-%m-%d')}")

    for iter_num in range(n_months):  # 최대 n_months번 반복(무한루프 방지)
        changed = False
        for current_date in period_index:
            for target_var in forecast_df.columns:
                if pd.isna(forecast_df.loc[current_date, target_var]):
                    if target_var not in arx_results:
                        continue
                    try:
                        best_lags = arx_results[target_var]['best_lags']
                        target_lag, factor_lag = best_lags
                        log_data = np.log(forecast_df[target_var])
                        # 타겟 변수 lag
                        target_lags = pd.concat([log_data.shift(lag) for lag in range(1, target_lag + 1)], axis=1)
                        target_lags.columns = [f'{target_var}_lag{lag}' for lag in range(1, target_lag + 1)]
                        # 공통요인 lag
                        factor_lags = pd.concat([factor_data.shift(lag) for lag in range(0, factor_lag + 1)], axis=1)
                        factor_lags.columns = [f'factor_lag{lag}' for lag in range(0, factor_lag + 1)]
                        # 예측용 X
                        X = pd.concat([target_lags, factor_lags], axis=1)
                        X = X.loc[:current_date].iloc[-1:]
                        if X.isna().any().any():
                            continue
                        model = arx_results[target_var]['model']
                        X_array = np.ones((1, len(model.params)))
                        X_array[0, 1:] = X.values[0]
                        log_forecast = model.predict(X_array)[0]
                        forecast = np.exp(log_forecast)
                        forecast_df.loc[current_date, target_var] = forecast
                        print(f"✅ {current_date.strftime('%Y-%m-%d')} {target_var} 예측: {forecast:.4f}")
                        changed = True
                    except Exception as e:
                        print(f"❌ {current_date.strftime('%Y-%m-%d')} {target_var} 예측 오류: {str(e)}")
                        continue
        if not changed:
            print(f"반복 {iter_num+1}회 후 더 이상 채울 결측치가 없습니다.")
            break
    print(f"\n=== ARX 모델을 사용한 결측치 순차 예측 완료 ===")
    return forecast_df

def calculate_quarterly_averages(monthly_data, quarterly_gdp):
    """월별 데이터를 분기별 평균으로 변환하고, 분기별 GDP 데이터의 인덱스에 맞춤
    
    Parameters:
    -----------
    monthly_data : pd.DataFrame
        월별 데이터
    quarterly_gdp : pd.DataFrame
        분기별 GDP 데이터 (인덱스 기준)
    """
    print("\n=== 데이터 인덱스 확인 ===")
    print(f"월별 데이터 기간: {monthly_data.index[0]} ~ {monthly_data.index[-1]}")
    print(f"분기별 GDP 데이터 기간: {quarterly_gdp.index[0]} ~ {quarterly_gdp.index[-1]}")
    
    # 분기말 날짜로 변환
    monthly_data.index = monthly_data.index + pd.offsets.MonthEnd(0)
    
    # 분기별 평균 계산
    quarterly_data = monthly_data.resample('Q').mean()
    print(f"분기별 평균 데이터 기간: {quarterly_data.index[0]} ~ {quarterly_data.index[-1]}")
    
    # 분기별 GDP 데이터의 인덱스에 맞춤
    quarterly_data = quarterly_data.loc[quarterly_gdp.index]
    print(f"최종 데이터 기간: {quarterly_data.index[0]} ~ {quarterly_data.index[-1]}")
    print(f"최종 데이터 행 수: {len(quarterly_data)}")
    
    return quarterly_data

def find_best_monthly_indicators_manual():
    """경제적 논리에 맞는 브리지 지표를 수동으로 지정 (포괄적 버전)"""
    print("\n=== 경제적 논리 기반 브리지 지표 매핑 (포괄적) ===")
    
    # 포괄적 브리지 지표 로드 (상위 폴더에서)
    try:
        manual_indicators = pd.read_csv('comprehensive_bridge_indicators.csv', encoding='utf-8-sig')
        print(f"✅ 포괄적 브리지 지표 로드 완료: {len(manual_indicators)}개")
    except FileNotFoundError:
        print("❌ comprehensive_bridge_indicators.csv 파일을 찾을 수 없습니다.")
        return {}
    
    # GDP 구성요소별로 그룹화
    results = {}
    for gdp_component in manual_indicators['GDP_Component'].unique():
        component_indicators = manual_indicators[
            manual_indicators['GDP_Component'] == gdp_component
        ]
        
        results[gdp_component] = []
        
        print(f"\n{gdp_component}: {len(component_indicators)}개 지표")
        for i, (_, row) in enumerate(component_indicators.iterrows(), 1):
            indicator = row['Monthly_Indicator']
            logic = row['Economic_Logic']
            
            results[gdp_component].append({
                'indicator': indicator,
                'logic': logic,
                'r2': None,  # R2는 계산하지 않음
                'pvalue': None,
                'coef': None
            })
            
            if i <= 5:  # 처음 5개만 출력
                print(f"  {i}. {indicator}")
                print(f"     논리: {logic}")
            elif i == 6:
                print(f"  ... 외 {len(component_indicators) - 5}개 지표")
    
    return results

def build_bridge_equations_and_forecast_manual(quarterly_gdp, quarterly_monthly, manual_indicators):
    """수동으로 지정한 브리지 지표를 사용하여 GDP 구성요소를 예측합니다.
    
    Parameters:
    -----------
    quarterly_gdp : pd.DataFrame
        분기별 GDP 구성요소 데이터 (훈련용)
    quarterly_monthly : pd.DataFrame
        분기별로 평균낸 월별 지표 데이터 (예측용 포함)
    manual_indicators : dict
        수동으로 지정한 브리지 지표들
        
    Returns:
    --------
    dict
        각 GDP 구성요소별 예측값과 모델 정보
    """
    print(f"\n=== 수동 지정 브리지 지표를 사용한 GDP 구성요소 예측 ===")
    
    # 예측 결과 저장
    forecast_results = {}
    
    # 각 GDP 구성요소별로 브리지 방정식 구축
    for gdp_component in quarterly_gdp.columns:
        if gdp_component not in manual_indicators:
            print(f"❌ {gdp_component}에 대한 수동 브리지 지표가 없습니다.")
            continue
            
        print(f"\n=== {gdp_component} 브리지 방정식 구축 및 예측 ===")
        
        # 해당 GDP 구성요소에 대한 월별 지표들 선별
        component_indicators = [item['indicator'] for item in manual_indicators[gdp_component]]
        
        # 실제 데이터에 있는 지표만 필터링
        available_indicators = [ind for ind in component_indicators if ind in quarterly_monthly.columns]
        
        if not available_indicators:
            print(f"❌ {gdp_component}에 대한 사용 가능한 브리지 지표가 없습니다.")
            continue
        
        print(f"사용할 월별 지표 수: {len(available_indicators)}")
        for i, indicator in enumerate(available_indicators, 1):
            # 경제적 논리 출력
            logic = next(item['logic'] for item in manual_indicators[gdp_component] if item['indicator'] == indicator)
            print(f"  {i}. {indicator}")
            print(f"     논리: {logic}")
        
        try:
            # 훈련 데이터 준비 (전체 quarterly_gdp 사용)
            # 공통 인덱스 범위에서만 작업
            common_index = quarterly_gdp.index.intersection(quarterly_monthly.index)
            train_quarterly_monthly = quarterly_monthly.loc[common_index]
            train_quarterly_gdp = quarterly_gdp.loc[common_index]
            
            # X: 월별 지표들, y: GDP 구성요소
            X_train = train_quarterly_monthly[available_indicators].dropna()
            y_train = train_quarterly_gdp[gdp_component].loc[X_train.index]
            
            if len(X_train) < 5:  # 최소 5개 관측치 필요
                print(f"❌ {gdp_component}: 훈련 데이터가 부족합니다 ({len(X_train)}개)")
                continue
            
            # 예측 데이터 준비 (quarterly_monthly의 마지막 분기)
            latest_quarter = quarterly_monthly.index[-1]
            X_forecast = quarterly_monthly[available_indicators].loc[[latest_quarter]]
            
            # X_forecast에 결측치가 있는지 확인
            if X_forecast.isna().any().any():
                print(f"❌ {gdp_component}: 예측용 데이터에 결측치가 있습니다.")
                print("결측치가 있는 지표:")
                for col in X_forecast.columns:
                    if X_forecast[col].isna().any():
                        print(f"  - {col}")
                continue
            
            # 상수항 수동 추가 (더 안전한 방법)
            X_train_array = X_train.values
            X_train_const = np.column_stack([np.ones(X_train_array.shape[0]), X_train_array])
            
            X_forecast_array = X_forecast.values
            X_forecast_const = np.column_stack([np.ones(X_forecast_array.shape[0]), X_forecast_array])
            
            # 다중회귀 모델 구축
            model = OLS(y_train, X_train_const).fit()
            
            print(f"모델 R²: {model.rsquared:.4f}")
            print(f"조정된 R²: {model.rsquared_adj:.4f}")
            print(f"사용된 지표 수: {len(available_indicators)}")
            print(f"X_train_const shape: {X_train_const.shape}")
            print(f"X_forecast_const shape: {X_forecast_const.shape}")
            print(f"model.params length: {len(model.params)}")
            
            # 예측
            forecast_value = model.predict(X_forecast_const)[0]
            
            # 예측 구간 계산
            prediction = model.get_prediction(X_forecast_const)
            forecast_ci = prediction.conf_int(alpha=0.05)  # 95% 신뢰구간
            
            # 결과 저장
            forecast_results[gdp_component] = {
                'forecast': forecast_value,
                'lower_ci': forecast_ci[0, 0],
                'upper_ci': forecast_ci[0, 1],
                'model_r2': model.rsquared,
                'model_adj_r2': model.rsquared_adj,
                'n_indicators': len(available_indicators),
                'indicators': available_indicators,
                'model': model,
                'forecast_quarter': latest_quarter
            }
            
            print(f"✅ 예측 분기: {latest_quarter}")
            print(f"✅ 예측값: {forecast_value:.2f}")
            print(f"   95% 신뢰구간: [{forecast_ci[0, 0]:.2f}, {forecast_ci[0, 1]:.2f}]")
            
        except Exception as e:
            print(f"❌ {gdp_component} 예측 중 오류: {str(e)}")
            continue
    
    return forecast_results

def forecast_gdp_components_with_bvar(quarterly_gdp, n_lags=5):
    """BVAR 모델을 사용하여 분기별 GDP 구성요소의 다음 분기를 예측합니다.
    
    Parameters:
    -----------
    quarterly_gdp : pd.DataFrame
        분기별 GDP 구성요소 데이터
    n_lags : int
        VAR 모델의 래그 수 (기본값: 5)
        
    Returns:
    --------
    pd.DataFrame
        예측값이 추가된 분기별 GDP 구성요소 데이터
    """
    print(f"\n=== BVAR 모델을 사용한 GDP 구성요소 예측 ===")
    print(f"데이터 기간: {quarterly_gdp.index[0]} ~ {quarterly_gdp.index[-1]}")
    print(f"구성요소 수: {len(quarterly_gdp.columns)}")
    print(f"사용할 래그 수: {n_lags}")
    
    # 결측치 제거
    gdp_data = quarterly_gdp.dropna()
    print(f"결측치 제거 후 데이터 행 수: {len(gdp_data)}")
    
    if len(gdp_data) < n_lags + 10:  # 최소 데이터 요구량 확인
        print(f"❌ 데이터가 부족합니다. 최소 {n_lags + 10}개 필요, 현재 {len(gdp_data)}개")
        return quarterly_gdp
    
    try:
        # VAR 모델 적합 (BVAR의 근사치로 사용)
        model = VAR(gdp_data)
        
        # 최적 래그 선택 (AIC 기준)
        lag_order = model.select_order(maxlags=min(8, len(gdp_data)//4))
        optimal_lags = lag_order.aic
        print(f"AIC 기준 최적 래그: {optimal_lags}")
        
        # 사용자 지정 래그와 최적 래그 중 선택
        final_lags = min(n_lags, optimal_lags) if optimal_lags <= 6 else n_lags
        print(f"최종 사용 래그: {final_lags}")
        
        # VAR 모델 적합
        var_model = model.fit(final_lags)
        print(f"VAR({final_lags}) 모델 적합 완료")
        
        # 모델 진단 정보
        print(f"AIC: {var_model.aic:.2f}")
        print(f"BIC: {var_model.bic:.2f}")
        
        # 다음 분기 예측
        forecast_steps = 1
        forecast = var_model.forecast(gdp_data.values[-final_lags:], steps=forecast_steps)
        
        # 예측 구간 계산
        try:
            forecast_ci = var_model.forecast_interval(gdp_data.values[-final_lags:], steps=forecast_steps, alpha=0.05)
            # forecast_ci의 형태 확인 및 처리
            if len(forecast_ci) == 2:  # (lower, upper) 튜플 형태인 경우
                forecast_lower_vals = forecast_ci[0][0]  # 첫 번째 예측 시점의 하한값
                forecast_upper_vals = forecast_ci[1][0]  # 첫 번째 예측 시점의 상한값
            else:  # 3차원 배열 형태인 경우
                forecast_lower_vals = forecast_ci[0, :, 0]
                forecast_upper_vals = forecast_ci[0, :, 1]
        except Exception as ci_error:
            print(f"예측 구간 계산 중 오류: {ci_error}")
            # 신뢰구간 계산 실패 시 임시값 사용
            forecast_lower_vals = forecast[0] * 0.9  # 예측값의 90%
            forecast_upper_vals = forecast[0] * 1.1  # 예측값의 110%
        
        # 다음 분기 날짜 계산
        last_date = quarterly_gdp.index[-1]
        next_quarter = last_date + pd.offsets.QuarterEnd(1)
        
        # 예측 결과를 DataFrame으로 변환
        forecast_df = pd.DataFrame(
            forecast, 
            index=[next_quarter], 
            columns=gdp_data.columns
        )
        
        # 신뢰구간 정보 추가
        forecast_lower = pd.DataFrame(
            [forecast_lower_vals], 
            index=[next_quarter], 
            columns=[f"{col}_lower" for col in gdp_data.columns]
        )
        
        forecast_upper = pd.DataFrame(
            [forecast_upper_vals], 
            index=[next_quarter], 
            columns=[f"{col}_upper" for col in gdp_data.columns]
        )
        
        # 원본 데이터에 예측값 추가
        extended_gdp = pd.concat([quarterly_gdp, forecast_df])
        
        # 예측 결과 출력
        print(f"\n=== {next_quarter.strftime('%Y년 %m월')} 분기 GDP 구성요소 예측 결과 ===")
        for i, component in enumerate(gdp_data.columns):
            pred_val = forecast[0, i]
            lower_ci = forecast_lower_vals[i]
            upper_ci = forecast_upper_vals[i]
            print(f"{component}: {pred_val:.2f} [{lower_ci:.2f}, {upper_ci:.2f}]")
        
        # 예측 결과와 신뢰구간을 별도 파일로 저장
        forecast_with_ci = pd.concat([forecast_df, forecast_lower, forecast_upper], axis=1)
        forecast_with_ci.to_csv(os.path.join(DATA_DIR, 'bvar_gdp_forecast.csv'), 
                               encoding='utf-8-sig', float_format="%.6f")
        
        print(f"\n✅ BVAR 예측 완료! 결과 저장: bvar_gdp_forecast.csv")
        
        return extended_gdp
        
    except Exception as e:
        print(f"❌ BVAR 모델 적합 중 오류 발생: {str(e)}")
        return quarterly_gdp

def combine_bridge_and_bvar_forecasts(bridge_forecasts, bvar_forecast_file, bridge_weight=0.6):
    """브리지 방정식과 BVAR 예측을 결합합니다.
    
    Parameters:
    -----------
    bridge_forecasts : dict
        브리지 방정식 예측 결과
    bvar_forecast_file : str
        BVAR 예측 결과 파일 경로
    bridge_weight : float
        브리지 방정식의 가중치 (0~1, 기본값: 0.6)
        
    Returns:
    --------
    pd.DataFrame
        결합된 예측 결과
    """
    print(f"\n=== 브리지 방정식과 BVAR 예측 결합 ===")
    print(f"브리지 방정식 가중치: {bridge_weight:.1f}")
    print(f"BVAR 가중치: {1-bridge_weight:.1f}")
    print(f"※ 정부소비는 BVAR만 사용 (브리지 가중치 0)")
    
    try:
        # BVAR 예측 결과 로드
        bvar_results = pd.read_csv(bvar_forecast_file, encoding='utf-8-sig', index_col=0)
        print(f"BVAR 예측 파일 로드 완료: {bvar_forecast_file}")
        
        # 결합 결과 저장용
        combined_results = []
        
        # 각 GDP 구성요소별로 결합
        for component in bridge_forecasts.keys():
            if component in bvar_results.columns:
                # 브리지 예측값
                bridge_forecast = bridge_forecasts[component]['forecast']
                bridge_lower = bridge_forecasts[component]['lower_ci']
                bridge_upper = bridge_forecasts[component]['upper_ci']
                bridge_r2 = bridge_forecasts[component]['model_r2']
                
                # BVAR 예측값
                bvar_forecast = bvar_results[component].iloc[0]
                
                # BVAR 신뢰구간 (만약 있다면)
                bvar_lower_col = f"{component}_lower"
                bvar_upper_col = f"{component}_upper"
                
                if bvar_lower_col in bvar_results.columns:
                    bvar_lower = bvar_results[bvar_lower_col].iloc[0]
                    bvar_upper = bvar_results[bvar_upper_col].iloc[0]
                else:
                    # 신뢰구간이 없으면 예측값의 ±10%로 가정
                    bvar_lower = bvar_forecast * 0.9
                    bvar_upper = bvar_forecast * 1.1
                
                # 정부소비는 BVAR만 사용 (브리지 가중치 0)
                if component == '정부소비':
                    actual_bridge_weight = 0.0
                    actual_bvar_weight = 1.0
                    print(f"\n{component}: BVAR 전용 모드")
                else:
                    actual_bridge_weight = bridge_weight
                    actual_bvar_weight = 1 - bridge_weight
                
                # 가중평균으로 결합
                combined_forecast = actual_bridge_weight * bridge_forecast + actual_bvar_weight * bvar_forecast
                combined_lower = actual_bridge_weight * bridge_lower + actual_bvar_weight * bvar_lower
                combined_upper = actual_bridge_weight * bridge_upper + actual_bvar_weight * bvar_upper
                
                # 예측 불확실성 계산 (두 예측의 차이를 반영)
                forecast_divergence = abs(bridge_forecast - bvar_forecast)
                uncertainty_adjustment = forecast_divergence * 0.1  # 차이의 10%를 불확실성으로 추가
                
                final_lower = combined_lower - uncertainty_adjustment
                final_upper = combined_upper + uncertainty_adjustment
                
                combined_results.append({
                    'GDP_Component': component,
                    'Bridge_Forecast': bridge_forecast,
                    'BVAR_Forecast': bvar_forecast,
                    'Combined_Forecast': combined_forecast,
                    'Combined_Lower_CI': final_lower,
                    'Combined_Upper_CI': final_upper,
                    'Forecast_Divergence': forecast_divergence,
                    'Bridge_R2': bridge_r2,
                    'Bridge_Weight': actual_bridge_weight,
                    'BVAR_Weight': actual_bvar_weight
                })
                
                if component == '정부소비':
                    print(f"  브리지: {bridge_forecast:.2f} (사용 안함)")
                    print(f"  BVAR:   {bvar_forecast:.2f} [{bvar_lower:.2f}, {bvar_upper:.2f}] ✓")
                    print(f"  결합:   {combined_forecast:.2f} [{final_lower:.2f}, {final_upper:.2f}] (BVAR 전용)")
                else:
                    print(f"\n{component}:")
                    print(f"  브리지: {bridge_forecast:.2f} [{bridge_lower:.2f}, {bridge_upper:.2f}]")
                    print(f"  BVAR:   {bvar_forecast:.2f} [{bvar_lower:.2f}, {bvar_upper:.2f}]")
                    print(f"  결합:   {combined_forecast:.2f} [{final_lower:.2f}, {final_upper:.2f}]")
                    print(f"  예측 차이: {forecast_divergence:.2f}")
                
            else:
                print(f"❌ {component}: BVAR 결과에서 찾을 수 없음")
        
        # 결과를 DataFrame으로 변환
        combined_df = pd.DataFrame(combined_results)
        
        if not combined_df.empty:
            # 결과 저장
            combined_df.to_csv(os.path.join(DATA_DIR, 'combined_gdp_forecasts.csv'), 
                             index=False, float_format="%.6f", encoding="utf-8-sig")
            
            print(f"\n✅ 결합 예측 완료! 결과 저장: combined_gdp_forecasts.csv")
            print(f"결합된 구성요소 수: {len(combined_df)}")
            
            # 요약 통계
            avg_divergence = combined_df['Forecast_Divergence'].mean()
            max_divergence = combined_df['Forecast_Divergence'].max()
            print(f"평균 예측 차이: {avg_divergence:.2f}")
            print(f"최대 예측 차이: {max_divergence:.2f}")
            
            return combined_df
        else:
            print("❌ 결합할 수 있는 구성요소가 없습니다.")
            return pd.DataFrame()
        
    except Exception as e:
        print(f"❌ 예측 결합 중 오류 발생: {str(e)}")
        return pd.DataFrame()

def create_complete_gdp_data(quarterly_gdp, combined_forecasts):
    """결합 예측을 기존 GDP 데이터와 통합하여 완전한 분기별 GDP 데이터를 생성합니다.
    
    Parameters:
    -----------
    quarterly_gdp : pd.DataFrame
        기존 분기별 GDP 구성요소 데이터
    combined_forecasts : pd.DataFrame
        결합 예측 결과
        
    Returns:
    --------
    pd.DataFrame
        예측값이 통합된 완전한 분기별 GDP 데이터
    """
    print(f"\n=== 완전한 분기별 GDP 데이터 생성 ===")
    
    # 기존 데이터 복사
    complete_gdp = quarterly_gdp.copy()
    
    # 다음 분기 날짜 계산
    last_date = quarterly_gdp.index[-1]
    next_quarter = last_date + pd.offsets.QuarterEnd(1)
    
    print(f"기존 데이터 마지막 분기: {last_date.strftime('%Y년 %m월')}")
    print(f"예측 분기: {next_quarter.strftime('%Y년 %m월')}")
    
    # 예측값을 다음 분기에 추가
    forecast_row = {}
    
    for _, row in combined_forecasts.iterrows():
        component = row['GDP_Component']
        forecast_value = row['Combined_Forecast']
        
        if component in complete_gdp.columns:
            forecast_row[component] = forecast_value
            print(f"  {component}: {forecast_value:.2f}")
        else:
            print(f"❌ {component}: 기존 데이터에서 찾을 수 없음")
    
    # 새로운 행 추가
    if forecast_row:
        # 예측되지 않은 구성요소는 NaN으로 설정
        for col in complete_gdp.columns:
            if col not in forecast_row:
                forecast_row[col] = np.nan
                print(f"  {col}: NaN (예측값 없음)")
        
        # DataFrame에 새 행 추가
        forecast_series = pd.Series(forecast_row, name=next_quarter)
        complete_gdp = pd.concat([complete_gdp, forecast_series.to_frame().T])
        
        print(f"\n✅ 예측값 통합 완료!")
        print(f"최종 데이터 기간: {complete_gdp.index[0].strftime('%Y년 %m월')} ~ {complete_gdp.index[-1].strftime('%Y년 %m월')}")
        print(f"총 분기 수: {len(complete_gdp)}")
        print(f"예측된 구성요소 수: {len([v for v in forecast_row.values() if not pd.isna(v)])}")
        
        # 신뢰구간 정보도 별도로 저장
        confidence_intervals = {}
        for _, row in combined_forecasts.iterrows():
            component = row['GDP_Component']
            confidence_intervals[f"{component}_lower"] = row['Combined_Lower_CI']
            confidence_intervals[f"{component}_upper"] = row['Combined_Upper_CI']
        
        ci_series = pd.Series(confidence_intervals, name=next_quarter)
        ci_df = ci_series.to_frame().T
        
        return complete_gdp, ci_df
    else:
        print("❌ 통합할 예측값이 없습니다.")
        return complete_gdp, pd.DataFrame()

def calculate_gdp_growth():
    """BVAR 예측값과 브리지 예측값을 평균내서 GDP 성장률 계산"""
    
    # 1. 기존 분기별 GDP 데이터 로드 (2025-03-31까지)
    quarterly_gdp = pd.read_csv(os.path.join(DATA_DIR, 'quarterly_gdp_components.csv'), 
                               index_col=0, parse_dates=True)
    
    # 2. BVAR 예측값 로드 (2025-06-30)
    bvar_data = pd.read_csv(os.path.join(DATA_DIR, 'quarterly_gdp_components_with_bvar_forecast.csv'), 
                           index_col=0, parse_dates=True)
    bvar_forecast = bvar_data.iloc[-1]  # 마지막 행 (2025-06-30)
    
    # 3. 브리지 예측값 로드
    bridge_data = pd.read_csv(os.path.join(DATA_DIR, 'combined_gdp_forecasts.csv'), encoding='utf-8-sig')
    
    # 4. BVAR와 브리지 예측값을 평균내서 2025-06-30 값 생성
    print("\n=== BVAR와 브리지 예측값 평균 계산 ===")
    
    # 다음 분기 날짜 (2025-06-30)
    next_quarter = pd.Timestamp('2025-06-30')
    
    # 평균 예측값 계산 (모형으로 추정한 항목들만)
    averaged_forecast = {}
    for _, row in bridge_data.iterrows():
        component = row['GDP_Component']
        bridge_value = row['Bridge_Forecast']
        
        if component in bvar_forecast.index:
            bvar_value = bvar_forecast[component]
            # 두 예측값의 평균
            averaged_value = (bridge_value + bvar_value) / 2
            averaged_forecast[component] = averaged_value
            
            print(f"{component}:")
            print(f"  브리지: {bridge_value:.2f}")
            print(f"  BVAR:   {bvar_value:.2f}")
            print(f"  평균:   {averaged_value:.2f}")
        else:
            print(f"❌ {component}: BVAR 데이터에서 찾을 수 없음")
    
    # 새로 추가된 항목들은 전기 값을 그대로 사용
    print(f"\n=== 새로 추가된 항목들 (전기 값 그대로 사용) ===")
    prev_quarter_data = quarterly_gdp.iloc[-1]  # 2025-03-31 데이터
    
    additional_components = ['재고증감 및 귀중품 순취득', '통계상 불일치']
    
    for component in additional_components:
        if component == '통계상 불일치':
            # 통계상 불일치는 0으로 설정
            averaged_forecast[component] = 0.0
            print(f"{component}: 0.0 (0으로 설정)")
        elif component in ['재고증감 및 귀중품 순취득']:
            # 4분기 평균으로 설정
            if component in quarterly_gdp.columns and len(quarterly_gdp) >= 4:
                recent_4q_avg = quarterly_gdp[component].iloc[-4:].mean()
                averaged_forecast[component] = recent_4q_avg
                print(f"{component}: {recent_4q_avg:.2f} (최근 4분기 평균)")
            elif component in prev_quarter_data.index:
                # 데이터가 부족하면 전기 값 사용
                prev_value = prev_quarter_data[component]
                averaged_forecast[component] = prev_value
                print(f"{component}: {prev_value:.2f} (전기 값, 데이터 부족)")
            else:
                print(f"❌ {component}: 데이터를 찾을 수 없음")
        else:
            # 기타 구성요소는 전기 값 사용
            if component in prev_quarter_data.index:
                prev_value = prev_quarter_data[component]
                averaged_forecast[component] = prev_value
                print(f"{component}: {prev_value:.2f} (전기 값 그대로)")
            else:
                print(f"❌ {component}: 전기 데이터에서 찾을 수 없음")
    
    # 5. 기존 데이터에 평균 예측값 + 전기 값 추가
    forecast_series = pd.Series(averaged_forecast, name=next_quarter)
    df = pd.concat([quarterly_gdp, forecast_series.to_frame().T])
    
    print(f"\n=== ✅ BVAR + 브리지 평균 예측값이 적용된 GDP 구성요소 데이터 ===")
    print(f"데이터 기간: {df.index[0].strftime('%Y년 %m월')} ~ {df.index[-1].strftime('%Y년 %m월')}")
    print(f"총 분기 수: {len(df)}")
    print(f"GDP 구성요소: {list(df.columns)}")
    
    # 평균 예측값이 적용된 전체 데이터 저장
    df.to_csv(os.path.join(DATA_DIR, 'quarterly_gdp_components_averaged_forecast.csv'), float_format='%.6f', encoding='utf-8-sig')
    
    print(f"\n✅ GDP 구성요소 예측값 저장 완료!")
    print(f"- 저장된 데이터: {os.path.join(DATA_DIR, 'quarterly_gdp_components_averaged_forecast.csv')}")
    
    return df

def GDP_FULL():
    """분기별 GDP 구성요소 데이터를 위한 data_list 생성"""
    quarterly_data_list = [
        {"name": "재고증감 및 귀중품 순취득", "api_code": "200Y108", "item_code": "1020120"},
        {"name": "통계상 불일치", "api_code": "200Y108", "item_code": "10501"},
    ]
    return quarterly_data_list

def collect_and_merge_additional_gdp_components():
    """예측 완료 후 추가 2개 GDP 구성요소를 수집하고 기존 데이터와 합치기
    
    예측 방법:
    - 재고증감 및 귀중품 순취득: 최근 4분기 평균
    - 통계상 불일치: 0으로 설정
    """
    print("\n=== 추가 GDP 구성요소 수집 및 병합 ===")
    
    # 1. 추가 2개 구성요소 수집
    additional_data_list = GDP_FULL()
    additional_quarterly_data = collect_quarterly_data(additional_data_list)
    
    if additional_quarterly_data is None:
        print("❌ 추가 GDP 구성요소 수집에 실패했습니다.")
        return None
    
    # 2. 기존 예측이 완료된 GDP 데이터 로드
    try:
        # BVAR + 브리지 평균 예측이 포함된 데이터 로드
        existing_gdp_data = pd.read_csv(
            os.path.join(DATA_DIR, 'quarterly_gdp_components_averaged_forecast.csv'), 
            index_col=0, parse_dates=True
        )
        print(f"✅ 기존 GDP 데이터 로드 완료: {existing_gdp_data.shape}")
        print(f"기존 구성요소: {list(existing_gdp_data.columns)}")
        
    except FileNotFoundError:
        print("❌ 기존 GDP 예측 데이터를 찾을 수 없습니다. 먼저 예측을 완료해주세요.")
        return None
    
    # 3. 추가 구성요소의 마지막 분기 값을 예측 분기에 복사
    # (추가 구성요소는 예측하지 않고 전기 값을 그대로 사용)
    last_existing_date = existing_gdp_data.index[-2]  # 예측 전 마지막 실제 데이터
    forecast_date = existing_gdp_data.index[-1]  # 예측 분기
    
    print(f"마지막 실제 데이터 분기: {last_existing_date.strftime('%Y년 %m월')}")
    print(f"예측 분기: {forecast_date.strftime('%Y년 %m월')}")
    
    # 추가 구성요소의 예측 분기 값 설정
    for component in additional_quarterly_data.columns:
        if component == '통계상 불일치':
            # 통계상 불일치는 0으로 설정
            additional_quarterly_data.loc[forecast_date, component] = 0.0
            print(f"  {component}: 0.0 (0으로 설정)")
        elif component in ['재고증감 및 귀중품 순취득']:
            # 4분기 평균으로 설정
            if len(additional_quarterly_data) >= 4:
                recent_4q_avg = additional_quarterly_data[component].iloc[-4:].mean()
                additional_quarterly_data.loc[forecast_date, component] = recent_4q_avg
                print(f"  {component}: {recent_4q_avg:.2f} (최근 4분기 평균)")
            else:
                # 데이터가 4분기 미만이면 전체 평균 사용
                avg_value = additional_quarterly_data[component].mean()
                additional_quarterly_data.loc[forecast_date, component] = avg_value
                print(f"  {component}: {avg_value:.2f} (전체 평균, 데이터 부족)")
        else:
            # 기타 구성요소는 전기 값 사용
            if last_existing_date in additional_quarterly_data.index:
                last_value = additional_quarterly_data.loc[last_existing_date, component]
                additional_quarterly_data.loc[forecast_date, component] = last_value
                print(f"  {component}: {last_value:.2f} (전기 값 그대로)")
            else:
                print(f"❌ {component}: 전기 데이터를 찾을 수 없음")
    
    # 4. 데이터 병합 (공통 인덱스 기준)
    common_index = existing_gdp_data.index.intersection(additional_quarterly_data.index)
    
    # 기존 데이터에서 공통 인덱스만 추출
    existing_aligned = existing_gdp_data.loc[common_index]
    additional_aligned = additional_quarterly_data.loc[common_index]
    
    # 데이터 병합
    merged_gdp_data = pd.concat([existing_aligned, additional_aligned], axis=1)
    
    print(f"\n✅ 데이터 병합 완료!")
    print(f"최종 데이터 기간: {merged_gdp_data.index[0].strftime('%Y년 %m월')} ~ {merged_gdp_data.index[-1].strftime('%Y년 %m월')}")
    print(f"총 GDP 구성요소: {len(merged_gdp_data.columns)}개")
    print(f"구성요소 목록: {list(merged_gdp_data.columns)}")
    
    # 5. 결과 저장
    merged_gdp_data.to_csv(
        os.path.join(DATA_DIR, 'quarterly_gdp_components_complete.csv'), 
        float_format='%.6f', encoding='utf-8-sig'
    )
    
    print(f"최종 데이터 저장: quarterly_gdp_components_complete.csv")
    
    return merged_gdp_data

def calculate_final_gdp_growth_with_all_components():
    """11개 구성요소가 모두 합쳐진 데이터로 최종 GDP 성장률 계산"""
    print(f"\n=== 11개 구성요소를 사용한 최종 GDP 성장률 계산 ===")
    
    # 1. 완전한 분기별 GDP 데이터 로드 (11개 구성요소)
    try:
        df = pd.read_csv(os.path.join(DATA_DIR, 'quarterly_gdp_components_complete.csv'), 
                        index_col=0, parse_dates=True)
        print(f"✅ 완전한 GDP 데이터 로드: {df.shape}")
        print(f"구성요소: {list(df.columns)}")
    except FileNotFoundError:
        print("❌ 완전한 GDP 데이터를 찾을 수 없습니다.")
        return None
    
    # 2. 중복 컬럼 처리 및 데이터 정리
    # 통계상 불일치.1 컬럼에 실제 데이터가 있으므로 이를 사용
    if '통계상 불일치' in df.columns and '통계상 불일치.1' in df.columns:
        # 기존 통계상 불일치 컬럼 삭제하고 통계상 불일치.1을 통계상 불일치로 rename
        df = df.drop('통계상 불일치', axis=1)
        df = df.rename(columns={'통계상 불일치.1': '통계상 불일치'})
        print("✅ 통계상 불일치.1 컬럼을 통계상 불일치로 변경 완료")
    
    # 통계상 불일치의 결측치를 0으로 처리 (마지막 분기만)
    df['통계상 불일치'] = pd.to_numeric(df['통계상 불일치'], errors='coerce').fillna(0)
    print("✅ 통계상 불일치 결측치를 0으로 처리 완료")
    
    # 3. GDP 계산 (지출접근법 - CIGXM 분류)
    # C: 민간소비
    df['C_민간소비'] = df['민간소비']
    
    # I: 총투자 (모든 투자 구성요소 합계)
    df['I_총투자'] = (df['건설투자'] + df['설비투자'] + df['지식재산생산물투자'] + 
                   df['재고증감 및 귀중품 순취득'])
    
    # G: 정부소비
    df['G_정부소비'] = df['정부소비']
    
    # X: 총수출
    df['X_총수출'] = df['재화수출'] + df['서비스수출']
    
    # M: 총수입
    df['M_총수입'] = df['재화수입'] + df['서비스수입']
    
    # NX: 순수출 (X - M)
    df['NX_순수출'] = df['X_총수출'] - df['M_총수입']
    
    # GDP 계산: GDP = 통계상불일치 + C + I + G + NX
    df['GDP_11개구성요소'] = df['통계상 불일치'] + df['C_민간소비'] + df['I_총투자'] + df['G_정부소비'] + df['NX_순수출']
    
    print("✅ GDP 계산 공식: GDP = 통계상불일치 + C + I + G + NX")
    
    # 4. 성장률 계산
    df['전분기대비_성장률'] = df['GDP_11개구성요소'].pct_change() * 100
    df['전년동기대비_성장률'] = df['GDP_11개구성요소'].pct_change(periods=4) * 100
    
    # 5. 최근 5개 분기 GDP 데이터 출력
    print("\n=== 최근 5개 분기 GDP (11개 구성요소, 조원) ===")
    recent_data = df[['GDP_11개구성요소']].tail(5).copy()
    recent_data['GDP_조원'] = recent_data['GDP_11개구성요소'] / 1000  # 조원 단위로 변환
    for date, row in recent_data.iterrows():
        print(f"{date.strftime('%Y년 %m월')}: {row['GDP_조원']:.1f}조원")
    
    # 6. 최신 분기 정보
    latest_quarter = df.index[-1]
    latest_gdp = df['GDP_11개구성요소'].iloc[-1] / 1000  # 조원 단위
    prev_quarter_gdp = df['GDP_11개구성요소'].iloc[-2] / 1000
    prev_year_gdp = df['GDP_11개구성요소'].iloc[-5] / 1000  # 1년 전 (4분기 전)
    
    qoq_growth = df['전분기대비_성장률'].iloc[-1]  # Quarter over Quarter
    yoy_growth = df['전년동기대비_성장률'].iloc[-1]  # Year over Year
    
    print(f"\n=== 🎯 2025년 2분기 최종 GDP 성장률 (11개 구성요소) ===")
    print(f"예측 분기: {latest_quarter.strftime('%Y년 %m월')}")
    print(f"GDP 규모: {latest_gdp:.1f}조원")
    print(f"전분기({df.index[-2].strftime('%Y년 %m월')}) 대비: {qoq_growth:.2f}%")
    print(f"전년 동기({df.index[-5].strftime('%Y년 %m월')}) 대비: {yoy_growth:.2f}%")
    
    # 7. 연율화 성장률 (전분기 대비를 연율로 환산)
    annualized_growth = ((1 + qoq_growth/100)**4 - 1) * 100
    print(f"전분기 대비 연율화 성장률: {annualized_growth:.2f}%")
    
    # 8. GDP 구성요소별 기여도 분석 (CIGXM 분류)
    print(f"\n=== 2025년 2분기 GDP 구성요소별 기여도 분석 (CIGXM) ===")
    
    # 각 구성요소별 전분기 대비 변화와 GDP 기여도
    components = {
        '통계상 불일치': '통계상 불일치',
        'C (민간소비)': 'C_민간소비', 
        'I (총투자)': 'I_총투자',
        'G (정부소비)': 'G_정부소비',
        'NX (순수출)': 'NX_순수출'
    }
    
    for comp_name, comp_col in components.items():
        if comp_col in df.columns:
            current_val = df[comp_col].iloc[-1] / 1000
            prev_val = df[comp_col].iloc[-2] / 1000
            contribution = ((current_val - prev_val) / prev_quarter_gdp) * 100
            
            if prev_val != 0:
                growth_rate = ((current_val/prev_val-1)*100)
            else:
                growth_rate = 0.0 if current_val == 0 else float('inf')
            
            print(f"{comp_name}: {current_val:.1f}조원 "
                  f"(전분기대비 {growth_rate:.2f}%, "
                  f"GDP 기여도 {contribution:.2f}%p)")
    
    # 9. 결과를 CSV 파일로 저장 (CIGXM 구성요소 포함)
    result_columns = ['GDP_11개구성요소', '전분기대비_성장률', '전년동기대비_성장률',
                     '통계상 불일치', 'C_민간소비', 'I_총투자', 'G_정부소비', 'NX_순수출', 'X_총수출', 'M_총수입']
    result_df = df[result_columns].copy()
    result_df.to_csv(os.path.join(DATA_DIR, 'final_gdp_growth_rates_11_components_cigxm.csv'), 
                    float_format='%.6f', encoding='utf-8-sig')
    
    # 전체 데이터도 저장 (모든 구성요소 포함)
    df.to_csv(os.path.join(DATA_DIR, 'quarterly_gdp_components_final_complete_cigxm.csv'), 
             float_format='%.6f', encoding='utf-8-sig')
    
    print(f"\n✅ 최종 GDP 성장률 계산 완료! (11개 구성요소, CIGXM 분류)")
    print(f"- GDP 공식: 통계상불일치 + C + I + G + NX")
    print(f"- 최종 성장률 결과 (CIGXM): final_gdp_growth_rates_11_components_cigxm.csv")
    print(f"- 최종 전체 데이터 (CIGXM): quarterly_gdp_components_final_complete_cigxm.csv")
    
    return df

if __name__ == "__main__":
    print("\n=== 🚀 GDP Now 전체 파이프라인 시작 ===")
    
    # 1. 월별 경제지표 데이터 수집
    print("\n=== 1. 월별 경제지표 데이터 수집 ===")
    data_list = create_data_list()
    monthly_raw_data = collect_raw_data(data_list)
    
    if monthly_raw_data is None:
        print("❌ 월별 데이터 수집에 실패했습니다.")
        exit(1)
    
    # 2. 월별 데이터 전처리
    print("\n=== 2. 월별 데이터 전처리 ===")
    monthly_transformed, monthly_standardized = preprocess_raw_data()
    
    # 3. 분기별 GDP 구성요소 데이터 수집
    print("\n=== 3. 분기별 GDP 구성요소 데이터 수집 ===")
    quarterly_data_list = create_quarterly_data_list()
    quarterly_gdp_full = collect_quarterly_data(quarterly_data_list)
    
    if quarterly_gdp_full is None:
        print("❌ GDP 구성요소 데이터 수집에 실패했습니다.")
        exit(1)
    
    # 4. 분기별 데이터 전처리
    print("\n=== 4. 분기별 데이터 전처리 ===")
    quarterly_gdp_transformed = preprocess_quarterly_data()
    
    # 5. 공통요인 추출 및 예측 (PCA & AR)
    print("\n=== 5. 공통요인 추출 및 예측 ===")
    from datetime import datetime
    current_date = datetime.now()
    as_of_time = current_date.strftime("%Y%m%d_%H%M")
    
    factors_historical, factors_forecast = run_pca_ar_forecast(
        monthly_standardized, 
        k_factors=1, 
        factor_ar_order=3, 
        current_date_for_run=current_date,
        as_of_time_str=as_of_time
    )
    
    if factors_historical is None:
        print("❌ 공통요인 추출에 실패했습니다.")
        exit(1)

    print("\n=== 6. ARX 모형을 위한 월별 데이터 준비 ===")
    
    # 6. ARX 모형을 위한 월별 데이터 준비
    monthly_data = create_monthly_data_for_arx()
    
    # 공통요인 데이터 로드
    factor_data = pd.read_csv('data/common_factor_combined.csv', index_col=0, parse_dates=True)
    
    # ARX 모형 적합
    print("\n=== 7. ARX 모형 적합 ===")
    arx_results = fit_arx_model(monthly_data, factor_data)
    
    # monthly_complete_data.csv 불러오기 (있으면)
    try:
        forecast_df = pd.read_csv(os.path.join(DATA_DIR, 'monthly_complete_data.csv'), index_col=0, parse_dates=True)
    except FileNotFoundError:
        forecast_df = monthly_data.copy()
    
    # 마지막 관측일이 속한 분기의 마지막 월까지 인덱스 확장
    last_date = forecast_df.index[-1]
    last_quarter_end = (last_date + pd.offsets.QuarterEnd(0)).normalize()
    
    # 현재 인덱스에 없는 날짜들 추가
    new_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                            end=last_quarter_end, 
                            freq='M')
    new_dates = [date + pd.offsets.MonthEnd(0) for date in new_dates]
    
    # 새로운 날짜들을 NaN 값으로 추가
    for date in new_dates:
        if date not in forecast_df.index:
            forecast_df.loc[date] = np.nan
    
    # 결측치 예측
    print("\n=== 8. 결측치 예측 및 보간 ===")
    forecast_df = forecast_missing_values(forecast_df, factor_data, arx_results)
    
    # 결과를 CSV 파일로 저장
    forecast_df.to_csv(os.path.join(DATA_DIR, 'monthly_complete_data.csv'), index=True, float_format="%.6f")
    
    # 9. 분기별 GDP 구성요소 데이터 로드
    quarterly_gdp = pd.read_csv(os.path.join(DATA_DIR, 'quarterly_gdp_components.csv'), 
                              index_col=0, parse_dates=True)
    
    # 10. 분기별 평균 계산 (GDP 데이터의 인덱스에 맞춤)
    print("\n=== 10. 분기별 평균 계산 ===")
    quarterly_monthly = calculate_quarterly_averages(forecast_df, quarterly_gdp)
    
    # 11. 최적의 월별 지표 찾기
    print("\n=== 11. 브리지 방정식 지표 선별 ===")
    best_indicators = find_best_monthly_indicators_manual()
    
    # 수동 브리지 지표 결과 저장
    manual_results_list = []
    for component, indicators in best_indicators.items():
        for indicator in indicators:
            manual_results_list.append({
                'GDP_Component': component,
                'Monthly_Indicator': indicator['indicator'],
                'Economic_Logic': indicator['logic']
            })
    
    if manual_results_list:
        results_df = pd.DataFrame(manual_results_list)
        results_df.to_csv(os.path.join(DATA_DIR, 'bridge_equation_indicators_comprehensive.csv'), 
                         index=False, encoding="utf-8-sig")
        print(f"✅ 포괄적 브리지 지표 매핑 저장: bridge_equation_indicators_comprehensive.csv")
        print(f"총 {len(manual_results_list)}개 지표가 GDP 구성요소에 매핑되었습니다.")

    # 브리지 방정식을 사용한 GDP 구성요소 예측
    print("\n=== 브리지 방정식을 사용한 GDP 구성요소 예측 ===")
    
    # monthly_complete_data를 분기별 평균으로 변환
    forecast_df_copy = forecast_df.copy()
    forecast_df_copy.index = forecast_df_copy.index + pd.offsets.MonthEnd(0)
    quarterly_complete_data = forecast_df_copy.resample('Q').mean()
    
    # 인덱스를 분기말로 명시적으로 맞춤
    quarterly_complete_data.index = quarterly_complete_data.index + pd.offsets.QuarterEnd(0)
    
    print(f"분기별 변환된 데이터 기간: {quarterly_complete_data.index[0]} ~ {quarterly_complete_data.index[-1]}")
    print(f"분기별 데이터 행 수: {len(quarterly_complete_data)}")
    print(f"quarterly_gdp 기간: {quarterly_gdp.index[0]} ~ {quarterly_gdp.index[-1]}")
    
    # GDP 구성요소 예측 (수동 브리지 지표 사용)
    gdp_forecasts = build_bridge_equations_and_forecast_manual(
        quarterly_gdp, 
        quarterly_complete_data, 
        best_indicators
    )
    
    # 예측 결과 저장
    if gdp_forecasts:
        forecast_summary = []
        for component, result in gdp_forecasts.items():
            forecast_summary.append({
                'GDP_Component': component,
                'Forecast': result['forecast'],
                'Lower_CI': result['lower_ci'],
                'Upper_CI': result['upper_ci'],
                'Model_R2': result['model_r2'],
                'Model_Adj_R2': result['model_adj_r2'],
                'N_Indicators': result['n_indicators']
            })
        
        forecast_summary_df = pd.DataFrame(forecast_summary)
        forecast_summary_df.to_csv(os.path.join(DATA_DIR, 'gdp_component_forecasts.csv'), 
                                 index=False, float_format="%.6f", encoding="utf-8-sig")
        
        print(f"\n✅ GDP 구성요소 예측 완료! 결과 저장: gdp_component_forecasts.csv")
    else:
        print(f"\n❌ 브리지 방정식으로 예측된 구성요소가 없습니다. BVAR만으로 진행합니다.")
    
    # BVAR를 사용한 GDP 구성요소 예측
    print("\n=== BVAR를 사용한 GDP 구성요소 예측 ===")
    extended_quarterly_gdp = forecast_gdp_components_with_bvar(quarterly_gdp, n_lags=5)
    
    # BVAR 예측 결과를 포함한 전체 데이터 저장
    extended_quarterly_gdp.to_csv(os.path.join(DATA_DIR, 'quarterly_gdp_components_with_bvar_forecast.csv'), 
                                 encoding='utf-8-sig', float_format="%.6f")
    
    # 브리지 방정식과 BVAR 예측 결합
    if gdp_forecasts:
        print("\n=== 브리지 방정식과 BVAR 예측 결합 ===")
        combined_forecasts = combine_bridge_and_bvar_forecasts(
            gdp_forecasts, 
            os.path.join(DATA_DIR, 'bvar_gdp_forecast.csv'),
            bridge_weight=0.6  # 브리지 방정식에 60% 가중치, BVAR에 40% 가중치
        )
        
        if not combined_forecasts.empty:
            print(f"\n🎯 최종 결합 예측 결과:")
            print(f"총 {len(combined_forecasts)}개 GDP 구성요소 예측 완료")
            
            # 가장 큰 예측 차이를 보이는 구성요소 출력
            max_div_idx = combined_forecasts['Forecast_Divergence'].idxmax()
            max_div_component = combined_forecasts.loc[max_div_idx, 'GDP_Component']
            max_div_value = combined_forecasts.loc[max_div_idx, 'Forecast_Divergence']
            print(f"예측 차이가 가장 큰 구성요소: {max_div_component} (차이: {max_div_value:.2f})")
    else:
        print("\n=== 브리지 방정식이 실패하여 BVAR 결과만 사용 ===")
        # BVAR 결과를 combined_forecasts 형식으로 변환
        bvar_results = pd.read_csv(os.path.join(DATA_DIR, 'bvar_gdp_forecast.csv'), encoding='utf-8-sig', index_col=0)
        
        combined_results = []
        for component in bvar_results.columns:
            if not component.endswith('_lower') and not component.endswith('_upper'):
                forecast_value = bvar_results[component].iloc[0]
                
                # 신뢰구간이 있는지 확인
                lower_col = f"{component}_lower"
                upper_col = f"{component}_upper"
                
                if lower_col in bvar_results.columns:
                    lower_ci = bvar_results[lower_col].iloc[0]
                    upper_ci = bvar_results[upper_col].iloc[0]
                else:
                    # 신뢰구간이 없으면 예측값의 ±10%로 가정
                    lower_ci = forecast_value * 0.9
                    upper_ci = forecast_value * 1.1
                
                combined_results.append({
                    'GDP_Component': component,
                    'Bridge_Forecast': None,
                    'BVAR_Forecast': forecast_value,
                    'Combined_Forecast': forecast_value,
                    'Combined_Lower_CI': lower_ci,
                    'Combined_Upper_CI': upper_ci,
                    'Forecast_Divergence': 0.0,
                    'Bridge_R2': None,
                    'Bridge_Weight': 0.0,
                    'BVAR_Weight': 1.0
                })
        
        combined_forecasts = pd.DataFrame(combined_results)
        combined_forecasts.to_csv(os.path.join(DATA_DIR, 'combined_gdp_forecasts.csv'), 
                                 index=False, float_format="%.6f", encoding="utf-8-sig")
        print(f"✅ BVAR 전용 예측 결과 저장: combined_gdp_forecasts.csv")
    
    # GDP 성장률 계산
    print("\n=== GDP 성장률 계산 ===")
    gdp_data = calculate_gdp_growth()
    
    # 추가 GDP 구성요소 수집 및 병합
    print("\n=== 추가 GDP 구성요소 수집 및 병합 ===")
    complete_gdp_data = collect_and_merge_additional_gdp_components()
    
    if complete_gdp_data is not None:
        print(f"\n🎯 최종 완성된 GDP 데이터:")
        print(f"- 총 구성요소: {len(complete_gdp_data.columns)}개")
        print(f"- 데이터 기간: {complete_gdp_data.index[0].strftime('%Y년 %m월')} ~ {complete_gdp_data.index[-1].strftime('%Y년 %m월')}")
        print(f"- 저장 파일: quarterly_gdp_components_complete.csv")
        
        # 11개 구성요소로 최종 GDP 성장률 계산
        final_gdp_data = calculate_final_gdp_growth_with_all_components()
        
        if final_gdp_data is not None:
            print(f"\n🎉 최종 GDP 성장률 계산 완료! (11개 구성요소 사용)")
        else:
            print("❌ 최종 GDP 성장률 계산에 실패했습니다.")
    else:
        print("❌ 추가 구성요소 병합에 실패하여 9개 구성요소 결과를 사용합니다.")
    
    print("\n✅ 모든 처리가 완료되었습니다!")
