import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import seaborn as sns
import matplotlib.pyplot as plt
import platform
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import warnings

# 한글 깨짐 방지
def set_korean_font():
    # """운영체제에 따라 한글 폰트 설정 및 matplotlib 환경 설정"""
    if platform.system() == 'Darwin':  # macOS
        rc('font', family='AppleGothic')
    elif platform.system() == 'Windows':  # Windows
        font_path = 'C:/Windows/Fonts/NanumGothic.ttf'  # 실제 경로로 바꿔도 OK
        font_name = font_manager.FontProperties(fname=font_path).get_name()
        rc('font', family=font_name)
    else:  # Linux
        font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
        font_name = font_manager.FontProperties(fname=font_path).get_name()
        rc('font', family=font_name)
    
    # 한글 깨짐 방지 설정 및 경고 무시
    plt.rcParams['axes.unicode_minus'] = False
    warnings.filterwarnings('ignore')

# 실거래 데이터 전처리
def preprocess_base_df(df, year=2025):
    df = df.copy()
    df['dealAmount'] = df['dealAmount'].astype(str).str.replace(',', '', regex=False).astype(int)
    df['buildingAge'] = year - df['buildYear'].astype(int)
    df['floor'] = pd.to_numeric(df['floor'], errors='coerce')
    df['isHighFloor'] = (df['floor'] >= 15).astype(int)
    df['dealYearMonth'] = df['dealYear'].astype(str) + '-' + df['dealMonth'].astype(str).str.zfill(2)
    df['dealQuarter'] = pd.to_datetime(df['dealYearMonth']).dt.quarter.astype(int)
    df['umdNm'] = df['umdNm'].fillna('미상')
    df.dropna(subset=['excluUseAr', 'floor', 'dealAmount', 'buildYear'], inplace=True)
    return df

# 여러 모델 학습 및 성능 비교
def train_compare_models(df, features):
    X = pd.get_dummies(df[features])
    y = df['dealAmount']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'LinearRegression': LinearRegression(),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42),
        'LightGBM': lgb.LGBMRegressor(
            n_estimators=1000,         
            learning_rate=0.03,         
            max_depth=6,                
            num_leaves=31,              
            min_data_in_leaf=20,        
            subsample=0.8,              
            colsample_bytree=0.8,       
            reg_alpha=1,                
            reg_lambda=1,               
            random_state=42
        )
    }

    
    results = []
    fitted_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        results.append((name, mae, r2))
        fitted_models[name] = model
        print(f"{name} → MAE: {mae:.2f}, R²: {r2:.4f}")

    result_df = pd.DataFrame(results, columns=['Model', 'MAE', 'R2'])
    return result_df, fitted_models, X_train, X_test, y_train, y_test

# dealAmount 상관관계 시각화
def plot_dealamount_correlation(df):
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', square=True, linewidths=0.5)
    plt.title('Correlation Matrix Heatmap (Including dealAmount)')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # dealAmount 기준 상관계수 정렬 출력
    target_corr = corr_matrix['dealAmount'].sort_values(ascending=False)
    print(target_corr)

# 추가: df 숫자형 컬럼 정리 함수
def clean_numeric_df(df):
    unwanted_cols = ['jibun', 'sggCd', 'dealDay', 'dealMonth', 'dealYear', 'aptLat', 'aptLng']
    for col in unwanted_cols:
        if col in df.columns:
            df = df.drop(columns=[col])

    numeric_cols = ['dealAmount', 'excluUseAr', 'floor', 'buildingAge','dealQuarter',
                    'isStationNearby', 'isHighFloor', 'subwayDistance','nearestElementaryDistance', 'nearestMiddleDistance', 'isSchoolPremium']
    numeric_df = df[[col for col in numeric_cols if col in df.columns]].copy()
    return numeric_df

# 주소 → 좌표 변환 함수
import requests
from dotenv import load_dotenv
import os

# 환경변수 로딩
load_dotenv()
KAKAO_API_KEY = os.getenv('KAKAO_API_KEY')
headers = {'Authorization': f'KakaoAK {KAKAO_API_KEY}'}
def get_coords(address):
    url = 'https://dapi.kakao.com/v2/local/search/address.json'
    params = {'query': address}
    res = requests.get(url, headers=headers, params=params).json()
    try:
        coords = res['documents'][0]['address']
        return float(coords['y']), float(coords['x'])  # 위도, 경도
    except:
        return None
