"""
machine_learning.py
Coin AI Bot - 3단계: 머신러닝 모델 훈련 파이프라인

이 모듈은 피처 엔지니어링으로 준비된 데이터를 사용하여
XGBoost 모델을 훈련하고 Optuna을 이용해 하이퍼파라미터를 최적화합니다.
"""

import os
import re
import glob
import time
import logging
import json
import config as _cfg
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from datetime import datetime
from typing import Tuple, Dict, Optional, List
import asyncio

try:
    import lightgbm as lgb
    _LGBM_AVAILABLE = True
except ImportError:
    _LGBM_AVAILABLE = False

# ============================================================================
# 1. 환경 및 디렉토리 설정
# ============================================================================
INPUT_DIR = _cfg.DIRECTORIES["data_processed"]
MODEL_DIR = _cfg.DIRECTORIES["models"]
LOG_DIR   = _cfg.DIRECTORIES["logs"]

# 디렉토리 생성
for directory in [MODEL_DIR, LOG_DIR]:
    os.makedirs(directory, exist_ok=True)

# 피처 선택 시 제외할 컬럼 (모델 훈련 및 평가 공통 사용)
EXCLUDE_COLS = [
    'Target', 'target', 'timestamp', 'date',
    'open', 'high', 'low', 'close', 'volume',
    'SMA_Short', 'SMA_Long', 'Volume_SMA', 'Month',
    # 절대 가격 종속 컬럼 — 정규화 버전(Ratio)으로 대체됨
    'MACD', 'MACD_Signal', 'MACD_Hist',
    'BB_Upper', 'BB_Lower',
    'ATR',
    # HMM 원-핫 인코딩 전 원본 정수 컬럼 제외 (HMM_Bear/HMM_Sideways/HMM_Bull 사용)
    'HMM_State',
    # 시계열 식별자 — 시간 패턴 과적합 방지 (Month는 기존 제외, DayOfWeek/Hour 추가)
    'DayOfWeek', 'Hour',
    # 데이터 출처 식별자 — 예측에 무관
    'coin', 'interval',
    # 중복 피처 제거 (고상관 쌍에서 정보량 낮은 쪽)
    'MACD_Signal_Ratio',  # MACD_Ratio의 EMA — 상관계수 >0.95
    'Stochastic_D',       # Stochastic_K의 3봉 SMA — 지연된 복사본
    'ROC_10',             # ROC_5와 중복 (15분봉에서 5봉이 더 responsive)
    'OFI_Imbalance',      # OFI_BuyPressure.rolling(5)*2-1 — 선형 변환 중복
]

# ============================================================================
# 2. 모델 학습 통계 클래스
# ============================================================================
class ModelTrainingStats:
    """모델 학습 통계"""
    def __init__(self):
        self.n_samples = 0
        self.n_features = 0
        self.class_distribution = {}
        self.best_trial = {}
        self.training_time = 0.0
        self.cross_val_scores = {}
    
    def to_dict(self) -> Dict:
        """사전 형태로 변환"""
        return {
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "class_distribution": self.class_distribution,
            "best_trial": self.best_trial,
            "training_time": self.training_time,
            "cross_val_scores": self.cross_val_scores
        }

# Temporal-leakage detection thresholds (validate_time_series_integrity)
_LEAKAGE_SHIFT_DELTA = 0.05  # future-target corr must exceed current corr by at least this
_LEAKAGE_MIN_CORR    = 0.30  # minimum absolute future-target corr to flag as suspicious

# ============================================================================
# 2.5. 시계열 무결성 검증 함수 ✅ [누수 수정 + 블랙스완 보호]
# ============================================================================
def validate_time_series_integrity(df: pd.DataFrame) -> Dict[str, bool]:
    """
    시계열 데이터 무결성 검증 + 블랙스완 이벤트 감지
    
    Args:
        df (pd.DataFrame): 검증할 데이터프레임
        
    Returns:
        Dict: 검증 결과
    """
    checks = {
        'has_timestamp': 'timestamp' in df.columns,
        'has_coin': 'coin' in df.columns,
        'is_sorted': False,
        'no_temporal_leakage': False,
        'no_mixed_coins': True,
        'no_black_swan': True,  # 블랙스완 이벤트 감지
        'memory_efficient': True  # 메모리 사용량 검증
    }
    
    # 메모리 사용량 검증 (대용량 데이터 처리용)
    memory_usage = df.memory_usage(deep=True).sum() / 1024**2  # MB
    if memory_usage > 500:  # 500MB 초과 시 경고
        logger.warning(f"⚠️  메모리 사용량 높음: {memory_usage:.1f}MB — 청크 처리 권장")
        checks['memory_efficient'] = False

    # 블랙스완 이벤트 감지: 단일 봉 ±5% 이상 급변동 검출
    if 'close' in df.columns and len(df) > 100:
        try:
            returns = df['close'].pct_change().dropna()
            extreme_events = (returns.abs() > 0.05).sum()  # 단봉 ±5% 이상
            if extreme_events / len(returns) > 0.02:  # 2% 초과 시 경고
                logger.warning(f"⚠️  블랙스완 이벤트 다수 감지: {extreme_events}/{len(returns)} — 데이터 품질 검토 필요")
                checks['no_black_swan'] = False
        except Exception as e:
            logger.debug(f"블랙스완 검증 실패: {e}")
    
    # 기존 검증 로직 유지
    # 시간순 정렬 확인
    if 'timestamp' in df.columns:
        df_temp = df.copy()
        df_temp['timestamp'] = pd.to_datetime(df_temp['timestamp'])
        
        if 'coin' in df.columns:
            # 코인별로 시간순 확인
            for coin in df_temp['coin'].unique():
                coin_data = df_temp[df_temp['coin'] == coin].sort_index()
                time_diffs = coin_data['timestamp'].diff()
                is_sorted = (time_diffs.iloc[1:] >= pd.Timedelta(0)).all()
                if not is_sorted:
                    logger.warning(f"⚠️  {coin} 데이터가 시간순으로 정렬되지 않음")
                    checks['is_sorted'] = False
                    break
            else:
                checks['is_sorted'] = True
        else:
            # 전체 데이터 시간순 확인
            time_diffs = df_temp['timestamp'].diff()
            checks['is_sorted'] = (time_diffs.iloc[1:] >= pd.Timedelta(0)).all()
    
    # 코인별 데이터 분포 확인
    if 'coin' in df.columns:
        coin_counts = df['coin'].value_counts()
        logger.info(f"📊 코인별 데이터 분포: {coin_counts.to_dict()}")
        checks['no_mixed_coins'] = True
    
    # 시간적 누수 감지: 피처와 Target 간 미래 상관관계 탐지
    # 정상적 피처라면 shift(+1) 이후 Target과의 상관이 현저히 줄어야 함.
    # 그렇지 않으면 미래 정보가 피처에 흘러들어간 것으로 판단.
    if 'Target' in df.columns:
        target_col = 'Target'
    elif 'target' in df.columns:
        target_col = 'target'
    else:
        target_col = None

    if target_col:
        leaked_features = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c not in [target_col, 'timestamp']]
        for col in feature_cols[:50]:  # 최대 50개만 샘플 검사 (성능)
            try:
                corr_current = abs(df[col].corr(df[target_col]))
                # 현재 피처가 1봉 후 타겟을 예측하는 정도:
                # 정상 피처는 corr(feat[t], target[t]) ≥ corr(feat[t], target[t+1]) 여야 함.
                # target을 -1 shift → target[t+1]과 feat[t]의 상관.
                # 반대 방향(미래 타겟)과의 상관이 현재 타겟보다 훨씬 높으면
                # 피처가 미래 정보를 담고 있을 가능성 (롤링 윈도우 미래 누수 등).
                corr_next = abs(df[col].corr(df[target_col].shift(-1)))
                if corr_next > corr_current + _LEAKAGE_SHIFT_DELTA and corr_next > _LEAKAGE_MIN_CORR:
                    leaked_features.append((col, round(corr_current, 3), round(corr_next, 3)))
            except Exception:
                continue
        if leaked_features:
            logger.warning(
                f"⚠️ 시간적 누수 의심 피처 {len(leaked_features)}개 "
                f"(현재 타겟 상관 < 미래 타겟 상관): "
                + ", ".join(f"{c}({c0}→{cn})" for c, c0, cn in leaked_features[:5])
            )
            checks['no_temporal_leakage'] = False
        else:
            checks['no_temporal_leakage'] = True
    else:
        # Target 컬럼 없음 → 누수 검증 불가. True가 아닌 None으로 표기해 "검증 완료"와 구분.
        checks['no_temporal_leakage'] = None
        logger.warning("⚠️ Target 컬럼 없음 — 시간적 누수 검증 불가 (None으로 표기)")

    return checks

# ============================================================================
# 3. 로깅 설정
# ============================================================================
def setup_logging(name: str = 'ml_training') -> logging.Logger:
    """로깅 설정"""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.INFO)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 파일 핸들러
    fh = logging.FileHandler(f"{LOG_DIR}/model_training_{timestamp}.log", encoding='utf-8')
    # 콘솔 핸들러
    ch = logging.StreamHandler()
    
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', 
                                 datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

logger = setup_logging()

# ============================================================================
# 4. 데이터 로드 및 검증
# ============================================================================
def _coin_from_filename(fname: str) -> Optional[str]:
    """processed_KRW-BTC_minute60.csv → 'BTC'. 파싱 실패 시 None."""
    m = re.search(r'KRW-([A-Z0-9]+)_', os.path.basename(fname))
    return m.group(1) if m else None


def load_and_prepare_data(
    input_dir: str,
    chunk_size: int = 50000,
    coins: Optional[List[str]] = None,
    exclude_coins: Optional[set] = None,
    interval: Optional[str] = None,
) -> Tuple[Optional[pd.DataFrame], ModelTrainingStats]:
    """
    가공된 모든 데이터를 불러와 하나의 거대한 학습 셋으로 병합합니다.
    메모리 효율적 처리: chunk_size 단위로 청크 처리

    interval: 명시 시 해당 인터벌 파일만 로드. None이면 config.MODEL_MANAGEMENT["train_interval"] 폴백.
    """
    stats = ModelTrainingStats()

    file_list = glob.glob(f"{input_dir}/*.csv")
    if not file_list:
        logger.error(f"❌ '{input_dir}' 폴더에 가공된 CSV 파일이 없습니다.")
        return None, stats

    # 포함 필터: coins 지정 시 해당 코인만
    if coins:
        coins_upper = {c.upper() for c in coins}
        file_list = [f for f in file_list if _coin_from_filename(f) in coins_upper]
        logger.info(f"🗂️ 포함 필터 ({len(coins_upper)} 코인): {sorted(coins_upper)} → {len(file_list)}개 파일")
        if not file_list:
            logger.error(f"❌ 카테고리 코인 {sorted(coins_upper)}에 해당하는 파일이 없습니다.")
            return None, stats
    # 제외 필터: 기본 카테고리(C) — 명시 카테고리(A·B) 코인 제외 후 전체 사용
    elif exclude_coins:
        exclude_upper = {c.upper() for c in exclude_coins}
        before = len(file_list)
        file_list = [f for f in file_list if _coin_from_filename(f) not in exclude_upper]
        included = sorted({_coin_from_filename(f) for f in file_list if _coin_from_filename(f)})
        logger.info(f"🗂️ 제외 필터 ({len(exclude_upper)} 코인 제외): {before}개 → {len(file_list)}개 파일 | 학습 코인: {included}")

    # 인터벌 필터: 파라미터 > config.MODEL_MANAGEMENT["train_interval"] 순으로 적용
    _train_interval = interval or _cfg.MODEL_MANAGEMENT.get("train_interval", None)
    if _train_interval:
        filtered = [f for f in file_list if f"_{_train_interval}.csv" in os.path.basename(f)]
        logger.info(f"📅 인터벌 필터 적용: '{_train_interval}' → {len(filtered)}/{len(file_list)}개 파일")
        file_list = filtered if filtered else file_list

    _min_train_rows = _cfg.MODEL_MANAGEMENT.get("min_train_rows", 0)
    logger.info(f"📥 총 {len(file_list)}개의 가공된 데이터를 불러옵니다...")

    df_list = []
    total_memory = 0

    for file in file_list:
        try:
            # 메모리 효율적 로딩: 청크 단위 처리
            df_iter = pd.read_csv(file, index_col=0, chunksize=chunk_size)
            file_chunks = []

            for chunk in df_iter:
                memory_mb = chunk.memory_usage(deep=True).sum() / 1024**2
                total_memory += memory_mb

                # 메모리 사용량 모니터링 (500MB 초과 시 경고)
                if total_memory > 500:
                    logger.warning(f"⚠️  메모리 사용량 높음: {total_memory:.1f}MB — 청크 크기 조정 권장")

                file_chunks.append(chunk)

            # 파일별 청크 병합
            if file_chunks:
                df = pd.concat(file_chunks, axis=0).reset_index(drop=True)
                if _min_train_rows and len(df) < _min_train_rows:
                    logger.warning(
                        f"⚠️ 데이터 부족 스킵: {os.path.basename(file)} "
                        f"({len(df)}행 < 최소 {_min_train_rows}행)"
                    )
                    continue
                # 인터벌 태그 (object dtype → validate_features 자동 제외)
                m = re.search(r'_(minute\d+|days)(?:\.csv)?', os.path.basename(file))
                df['_interval'] = m.group(1) if m else 'unknown'
                df_list.append(df)
                logger.debug(f"  ✓ 로드: {os.path.basename(file)} ({len(df)} 행, {total_memory:.1f}MB)")

        except Exception as e:
            logger.warning(f"⚠️  파일 읽기 실패 ({file}): {type(e).__name__}")
    
    if not df_list:
        logger.error("❌ 로드된 파일이 없습니다.")
        return None, stats

    # 핵심 MTF 피처 미생성 파일 제외 (신규 상장 코인이 common_cols 교집합을 약화시키는 것 방지)
    _MTF_REQUIRED = {'MTF_1D_RSI', 'MTF_1D_BB_Pos'}
    complete = [df for df in df_list if _MTF_REQUIRED.issubset(df.columns)]
    if complete:
        excluded = len(df_list) - len(complete)
        if excluded:
            logger.warning(f"⚠️ MTF 핵심 피처 미생성 파일 {excluded}개 학습 제외 (신규 상장 코인, 데이터 부족)")
        df_list = complete
    # 스키마 불일치 방지: 모든 파일에 공통으로 존재하는 컬럼만 사용
    # (구버전 processed 파일에 레거시 컬럼이 남아있으면 concat 시 NaN 전파 → valid_idx 전체 False)
    common_cols = set.intersection(*[set(df.columns) for df in df_list])
    legacy_cols = set.union(*[set(df.columns) for df in df_list]) - common_cols
    if legacy_cols:
        logger.warning(f"⚠️ 일부 파일에만 존재하는 레거시 컬럼 제외 ({len(legacy_cols)}개): {sorted(legacy_cols)}")
        df_list = [df[sorted(common_cols)] for df in df_list]

    # 모든 코인의 데이터를 병합
    master_df = pd.concat(df_list, axis=0).reset_index(drop=True)
    
    # 데이터 검증
    if 'Target' not in master_df.columns and 'target' not in master_df.columns:
        logger.error("❌ Target 컬럼이 없습니다.")
        return None, stats
    
    # ✅ [누수 수정] 시간순으로 정렬 - 시계열 누수 방지
    if 'timestamp' in master_df.columns:
        try:
            master_df['timestamp'] = pd.to_datetime(master_df['timestamp'])
            # coin 컬럼이 있으면 코인별로 그룹화하여 정렬
            if 'coin' in master_df.columns:
                master_df = master_df.sort_values(['coin', 'timestamp']).reset_index(drop=True)
                logger.info("✅ 데이터를 코인별/시간순으로 정렬 완료")
            else:
                master_df = master_df.sort_values('timestamp').reset_index(drop=True)
                logger.info("✅ 데이터를 시간순으로 정렬 완료")
        except Exception as e:
            logger.warning(f"⚠️  시간순 정렬 실패: {e}")
    else:
        logger.warning("⚠️  timestamp 컬럼이 없어 정렬 불가 - 시계열 누수 위험")
    
    stats.n_samples = len(master_df)
    logger.info(f"✅ 데이터 로드 완료! (총 {stats.n_samples} 행)")
    
    return master_df, stats

def validate_features(df: pd.DataFrame, exclude_cols: List[str]) -> Tuple[List[str], int]:
    """
    피처 검증 및 추출
    """
    features = [col for col in df.columns if col not in exclude_cols and df[col].dtype != 'object']
    
    if not features:
        logger.error("❌ 추출된 피처가 없습니다.")
        return [], 0
    
    logger.info(f"📊 학습에 사용될 피처({len(features)}개): {', '.join(features[:5])}... 외 {len(features)-5}개")
    return features, len(features)

# ============================================================================
# 4.5. Purged TimeSeriesSplit — Triple Barrier 라벨 누수 방지
# ============================================================================
class PurgedTimeSeriesSplit:
    """
    TimeSeriesSplit에 purge gap을 추가한 CV 스플리터.
    Triple Barrier 라벨은 미래 `gap`봉까지 참조하므로,
    훈련 세트 끝 `gap`개 샘플을 제거해야 검증 세트와의 라벨 누수가 차단됩니다.
    scikit-learn CV 인터페이스(split / get_n_splits) 완전 호환.
    """
    def __init__(self, n_splits: int = 5, *, test_size=None, max_train_size=None, gap: int = 0):
        self.n_splits = n_splits
        self.test_size = test_size
        self.max_train_size = max_train_size
        self.gap = gap
        self._tscv = TimeSeriesSplit(
            n_splits=n_splits,
            test_size=test_size,
            max_train_size=max_train_size,
        )

    def split(self, X, y=None, groups=None):
        for train_idx, val_idx in self._tscv.split(X, y, groups):
            if self.gap > 0 and len(train_idx) > self.gap:
                train_idx = train_idx[:-self.gap]
            yield train_idx, val_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


# ============================================================================
# 5. Optuna 최적화 및 훈련 엔진 (개선됨)
# ============================================================================

def _train_regime_xgb(
    X: pd.DataFrame, y: pd.Series, best_params: dict, purge_gap: int
) -> dict:
    """HMM 국면별(Bull/Sideways/Bear) XGBoost CalibratedClassifier 학습.
    데이터 부족 국면은 건너뛰며, 호출자가 전체 모델 폴백으로 처리.
    """
    _min_n = _cfg.MODEL_MANAGEMENT.get("regime_min_samples", 300)
    has_bull = "HMM_Bull" in X.columns
    has_bear = "HMM_Bear" in X.columns
    if not (has_bull and has_bear):
        logger.warning("  ⚠️ HMM_Bull/HMM_Bear 컬럼 없음 — 국면 모델 스킵")
        return {}

    bull_mask     = (X["HMM_Bull"] == 1).values
    bear_mask     = (X["HMM_Bear"] == 1).values
    sideways_mask = ~bull_mask & ~bear_mask

    models = {}
    for name, mask in [("bull", bull_mask), ("sideways", sideways_mask), ("bear", bear_mask)]:
        X_r, y_r = X[mask], y[mask]
        pos_r = int((y_r == 1).sum())
        if len(X_r) < _min_n or pos_r < 20:
            logger.info(f"  ⚠️ 국면 '{name}': {len(X_r)}행/{pos_r}양성 부족 — 전체 모델 폴백")
            continue
        _bal = float(np.clip((y_r == 0).sum() / pos_r, 1.0, 10.0))
        base = xgb.XGBClassifier(
            **{**best_params, "scale_pos_weight": _bal},
            random_state=_cfg.RANDOM_SEED, eval_metric="auc", n_jobs=-1,
        )
        cv = PurgedTimeSeriesSplit(n_splits=3, gap=purge_gap)
        m = CalibratedClassifierCV(base, method="isotonic", cv=cv)
        m.fit(X_r, y_r)
        models[name] = m
        logger.info(f"  ✅ 국면 '{name}': {len(X_r)}행 / 양성 {pos_r}개 학습 완료")
    return models


def optimize_and_train_bot(
    df: pd.DataFrame,
    n_trials: int = 100,
    verbose: bool = True
) -> Tuple[Optional[xgb.XGBClassifier], float, pd.DataFrame, ModelTrainingStats]:
    """
    Optuna을 사용한 모델 최적화 및 훈련
    """
    stats = ModelTrainingStats()
    
    logger.info("🚀 Optuna을 사용한 하이퍼파라미터 최적화를 시작합니다...")
    
    # 마이크로초 단위 성능 모니터링 시작
    optimization_start = time.perf_counter()
    
    # ✅ [누수 수정] 시계열 무결성 검증
    integrity = validate_time_series_integrity(df)
    
    if not integrity['has_timestamp']:
        logger.warning("⚠️  timestamp 컬럼이 없습니다 - 시계열 무결성을 보장할 수 없습니다")
    
    if integrity['has_coin']:
        logger.info("✅ 코인별 데이터 분리 확인됨")
    
    if integrity['is_sorted']:
        logger.info("✅ 데이터가 시간순으로 정렬되어 있습니다")
    else:
        logger.warning("⚠️  데이터가 시간순으로 정렬되지 않았을 수 있습니다")
    
    # 1. 피처와 타겟 분리
    features, n_features = validate_features(df, EXCLUDE_COLS)
    
    if not features:
        return None, 0, pd.DataFrame(), stats
    
    X = df[features].copy()
    y = df['Target'] if 'Target' in df.columns else df['target']

    # NaN 제거
    valid_idx = ~(X.isna().any(axis=1) | y.isna())
    X = X[valid_idx]
    y = y[valid_idx]

    # timestamp 인덱스 보존 (최종 모델 rolling window용)
    _ts_series = pd.to_datetime(df['timestamp']).reindex(X.index) if 'timestamp' in df.columns else None

    stats.n_features = n_features
    stats.n_samples = len(X)

    if len(X) == 0:
        logger.error("❌ NaN 제거 후 유효 샘플 없음 — 훈련 생략")
        return None, 0, pd.DataFrame(), stats

    # 2. 클래스 불균형 분석
    positive_cases = int(sum(y == 1))
    negative_cases = int(sum(y == 0))

    if positive_cases == 0:
        logger.error(f"❌ 양성 샘플 0개 (음성 {negative_cases}개) — Triple Barrier 라벨 검토 필요, 훈련 생략")
        return None, 0, pd.DataFrame(), stats

    # 극단적 불균형 클램핑: 너무 높으면 FP 급증, 너무 낮으면 FN 급증
    raw_balance_ratio = negative_cases / positive_cases
    balance_ratio = float(np.clip(raw_balance_ratio, 1.0, 10.0))

    stats.class_distribution = {
        "positive": positive_cases,
        "negative": negative_cases,
        "positive_ratio": round(positive_cases / len(y), 3)
    }

    logger.info(
        f"⚖️  클래스 분포 (메타라벨링 후) — "
        f"음성: {negative_cases}, 양성: {positive_cases}, "
        f"양성 비율: {positive_cases/max(len(y),1)*100:.1f}%, "
        f"scale_pos_weight: {balance_ratio:.2f} (클램핑 전: {raw_balance_ratio:.2f})"
    )
    
    # config.LABELING 배수 — 학습 라벨링과 동일한 기댓값 계산 (하드코딩 제거)
    _tp_mult = _cfg.LABELING.get("atr_tp_mult", 2.0)
    _sl_mult = _cfg.LABELING.get("atr_sl_mult", 1.0)

    # 3. Optuna 목적 함수 (워크포워드 전진 분석)
    _min_recall_gate = 0.05  # recall < 5% → 아무것도 안하는 모델 거부

    def objective(trial):
        params = {
            'n_estimators':      trial.suggest_int('n_estimators', 100, 400),
            'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth':         trial.suggest_int('max_depth', 3, 5),        # 7→5: 금융 데이터 과적합 방지
            'min_child_weight':  trial.suggest_int('min_child_weight', 5, 50), # 1→5: 소수 샘플 분기 차단
            'subsample':         trial.suggest_float('subsample', 0.6, 0.9),
            'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.6, 0.9),
            'gamma':             trial.suggest_float('gamma', 0, 3),
            'reg_alpha':         trial.suggest_float('reg_alpha', 0.0, 5.0),  # L1 정규화 추가
            'reg_lambda':        trial.suggest_float('reg_lambda', 1.0, 10.0),# L2 정규화 추가
            'scale_pos_weight':  balance_ratio,
            'random_state':      _cfg.RANDOM_SEED,
            'eval_metric':       'auc',
            'n_jobs':            -1
        }

        # threshold 하한: calibrated 확률 분포(일반적으로 0.1~0.45)를 커버해야 Optuna가 유효 trial 탐색 가능
        # min_precision(0.42)보다 낮게 설정 — 기댓값은 objective의 expectancy/prec_factor가 자연 도태
        _t_min = _cfg.MODEL_MANAGEMENT.get("optuna_threshold_min", 0.20)
        threshold = trial.suggest_float('threshold', _t_min, 0.80)

        # 진정한 Walk-Forward 롤링 윈도우: 각 fold마다 동일한 크기의 train/test 창을 시간순으로 이동
        n_splits = 5  # 5개의 서로 다른 시장 국면을 테스트
        test_size = len(X) // (n_splits + 2)
        max_train_size = test_size * 3  # 테스트 기간의 3배만큼 최신 데이터로만 학습 (Concept Drift 방지)
        _purge_gap = _cfg.LABELING.get("forward_bars", 4)

        tscv = PurgedTimeSeriesSplit(n_splits=n_splits, test_size=test_size, max_train_size=max_train_size, gap=_purge_gap)
        _min_prec = _cfg.MODEL_MANAGEMENT.get("min_precision", 0.52)
        scores = {'expectancy': [], 'auc': [], 'num_trades': [], 'precision': []}

        for fold_num, (train_index, val_index) in enumerate(tscv.split(X)):
            if len(train_index) > 0 and len(val_index) > 0:
                if train_index.max() >= val_index.min():
                    continue

            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]

            try:
                # 메타라벨링 후 fold별 라벨 분포가 달라지므로 각 fold에서 재계산
                fold_pos = int((y_train == 1).sum())
                fold_neg = int((y_train == 0).sum())
                fold_params = {
                    **params,
                    'scale_pos_weight': fold_neg / fold_pos if fold_pos > 0 else 1.0,
                }
                model = xgb.XGBClassifier(**fold_params)
                model.fit(X_train, y_train, verbose=False)

                # 캘리브레이션과 임계값 탐색을 서로 다른 분할에서 수행 — 같은 데이터로 fit/score하면 threshold 과적합
                cal_size = max(10, len(X_val) // 2)
                X_cal, X_eval = X_val.iloc[:cal_size], X_val.iloc[cal_size:]
                y_cal, y_eval = y_val.iloc[:cal_size], y_val.iloc[cal_size:]
                if len(X_eval) < 10:
                    continue
                calibrated = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
                calibrated.fit(X_cal, y_cal)
                pred_probs = calibrated.predict_proba(X_eval)[:, 1]
                predictions = (pred_probs >= threshold).astype(int)

                num_trades = sum(predictions)
                if num_trades >= 10:
                    prec = precision_score(y_eval, predictions, zero_division=0)
                    rec  = recall_score(y_eval, predictions, zero_division=0)
                    # recall < 5% → skip 대신 강한 페널티 (skip 시 Optuna가 저재현율 임계값 선호하는 버그 방지)
                    if rec < _min_recall_gate:
                        scores['expectancy'].append(-10.0)
                        scores['num_trades'].append(num_trades)
                        scores['precision'].append(0.0)
                        continue
                    # F1 가중 기댓값: precision 단독 최적화(고정밀/저재현) 방지
                    f1 = f1_score(y_eval, predictions, zero_division=0)
                    expectancy = (prec * _tp_mult) - ((1 - prec) * _sl_mult)
                    # 과적합 페널티: train precision과 val precision 차이가 10% 초과 시 감점
                    train_probs = calibrated.predict_proba(X_train)[:, 1]
                    train_prec = precision_score(y_train, (train_probs >= threshold).astype(int), zero_division=0)
                    overfit_gap = train_prec - prec
                    overfit_penalty = max(0.0, overfit_gap - 0.10) * 3.0
                    # Adjusted_Sharpe: 기댓값 × (1 + F1 보너스) 로 recall도 보상
                    adjusted_expectancy = expectancy * (1.0 + 0.5 * f1) * max(0.0, 1.0 - overfit_penalty)

                    scores['expectancy'].append(adjusted_expectancy)
                    scores['num_trades'].append(num_trades)
                    scores['precision'].append(prec)
                    try:
                        scores['auc'].append(roc_auc_score(y_eval, pred_probs))
                    except Exception:
                        pass
            except (ValueError, RuntimeError) as e:
                logger.warning(f"⚠️  Fold {fold_num} 훈련 실패 ({type(e).__name__}): {e} — 건너뜀")
                continue
            except Exception as e:
                logger.error(f"❌ Fold {fold_num} 예상치 못한 오류: {type(e).__name__} - {e}", exc_info=True)
                continue

        # 5개 국면 중 4개 이상에서 거래가 발생해야 신뢰할 수 있는 파라미터로 인정
        if not scores['expectancy'] or len(scores['expectancy']) < (n_splits - 1):
            return -999.0  # 기댓값은 음수일 수 있으므로 극소값 반환

        exp_arr  = np.array(scores['expectancy'])
        exp_mean = float(np.mean(exp_arr))
        exp_std  = float(np.std(exp_arr))

        if len(exp_arr) > 1:
            sharpe_mean = exp_mean / (exp_std + 1e-8)
            cum = np.cumsum(exp_arr)
            max_dd_mean = float(np.min(cum - np.maximum.accumulate(cum)))
        else:
            sharpe_mean = 0.0
            max_dd_mean = 0.0

        avg_trades  = float(np.mean(scores['num_trades'])) if scores['num_trades'] else 10.0
        # freq_weight: 거래량 보너스 cap 1.0 — 이전엔 무한 증가로 저임계값 편향 유발
        freq_weight = min(1.0, np.sqrt(max(avg_trades, 1.0) / 10.0))
        robust_score = (exp_mean - 0.5 * exp_std + 0.1 * sharpe_mean - 0.2 * abs(max_dd_mean)) * freq_weight

        # 정밀도 하드게이트: 손익분기 미달 시 즉시 차단 (soft penalty → hard block)
        avg_prec = float(np.mean(scores['precision'])) if scores['precision'] else 0.0
        if avg_prec < _min_prec * 0.95:
            return -999.0
        prec_factor = min(1.5, (avg_prec / _min_prec) ** 3)
        robust_score = robust_score * prec_factor

        return robust_score

    # 4. 최적화 실행
    logger.info(f"⏳ {n_trials}번의 시뮬레이션을 통해 최적의 설정을 찾습니다...")
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    # MedianPruner: 처음 10회 탐색 후 중앙값보다 나쁜 trial을 5 step 이후 조기 종료
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
    
    # 💥 핵심 패치 1: 과거 실패한 캐시(DB) 강제 무시를 위해 스터디 이름에 현재 Timestamp 부착
    _db_path = os.path.join(MODEL_DIR, "optuna_xgb.db")
    _storage = f"sqlite:///{_db_path}"
    _study_name = f"xgb_bot_study_f{n_features}_{int(time.time())}"
    
    try:
        study = optuna.create_study(
            study_name=_study_name,
            storage=_storage,
            direction="maximize",
            pruner=pruner,
            load_if_exists=False,
        )
    except Exception as _e:
        logger.warning(f"Optuna SQLite 저장소 초기화 실패({_e}) — 인메모리 스터디로 폴백")
        study = optuna.create_study(direction="maximize", pruner=pruner)

    start_time = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose)
    elapsed_time = time.time() - start_time
    
    # 마이크로초 단위 정밀 측정
    optimization_elapsed = time.perf_counter() - optimization_start
    stats.training_time = optimization_elapsed
    
    logger.info(f"🎯 최적화 완료! (소요 시간: {elapsed_time:.2f}초, 정밀 측정: {optimization_elapsed:.6f}초)")
    
    # 5. 최적 파라미터 추출
    best_params = study.best_params.copy()
    best_threshold = best_params.pop('threshold')
    best_params['scale_pos_weight'] = balance_ratio
    
    stats.best_trial = {
        "trial_number": study.best_trial.number,
        "value": round(study.best_value, 4),
        "threshold": round(best_threshold, 4)
    }
    
    logger.info("-" * 50)
    logger.info("🏆 [최적화된 하이퍼파라미터]")
    for key, value in sorted(best_params.items()):
        if key not in ['random_state', 'eval_metric', 'n_jobs', 'scale_pos_weight']:
            if isinstance(value, float):
                logger.info(f"  • {key:20s}: {value:.4f}")
            else:
                logger.info(f"  • {key:20s}: {value}")
    logger.info(f"  • 최적 매수 임계값:        {best_threshold:.4f}")
    logger.info("-" * 50)
    
    # 6. 최종 모델 훈련: 90% 학습 + 10% final holdout 검증 후 전체 데이터 재학습
    # 100% 전체 데이터로만 학습하면 캘리브레이션이 과적합될 수 있음
    logger.info("\n🧠 최종 모델을 훈련하고 확률을 보정합니다...")
    _purge_n = _cfg.LABELING.get("forward_bars", 4)
    final_holdout_cutoff = int(len(X) * 0.90)
    _probe_cutoff = max(0, final_holdout_cutoff - _purge_n)
    X_train_90 = X.iloc[:_probe_cutoff]
    y_train_90 = y.iloc[:_probe_cutoff]
    X_holdout  = X.iloc[final_holdout_cutoff:]
    y_holdout  = y.iloc[final_holdout_cutoff:]

    base_model_90 = xgb.XGBClassifier(**best_params, random_state=_cfg.RANDOM_SEED, eval_metric='auc', n_jobs=-1)
    _cal_cv_90 = PurgedTimeSeriesSplit(n_splits=3, gap=_cfg.LABELING.get("forward_bars", 4))
    probe_model = CalibratedClassifierCV(base_model_90, method='isotonic', cv=_cal_cv_90)
    probe_model.fit(X_train_90, y_train_90)

    if len(X_holdout) >= 10:
        holdout_probs = probe_model.predict_proba(X_holdout)[:, 1]
        holdout_preds = (holdout_probs >= best_threshold).astype(int)
        n_pos_holdout = int(holdout_preds.sum())
        if n_pos_holdout >= 3:
            holdout_prec = precision_score(y_holdout, holdout_preds, zero_division=0)
            logger.info(
                f"  📊 Final 10% holdout 검증: 정밀도={holdout_prec:.4f} "
                f"(신호 {n_pos_holdout}개/{len(X_holdout)}개)"
            )
            if holdout_prec < _cfg.MODEL_MANAGEMENT.get("min_precision", 0.60) * 0.90:
                logger.warning(
                    f"  ⚠️ holdout 정밀도 {holdout_prec:.4f}가 OOS 기준 90% 미달 — 과적합 의심. "
                    f"max_depth={best_params.get('max_depth','?')}, "
                    f"min_child_weight={best_params.get('min_child_weight','?')} 검토 요망 "
                    f"(Optuna 범위: max_depth 3~5, min_child_weight 5~50)"
                )
        else:
            logger.info(f"  ℹ️ holdout 신호 부족({n_pos_holdout}개) — 정밀도 검증 스킵")

    # 최종 생산 모델: rolling_train_months 이내 최신 데이터만 사용 (Optuna CV는 전체 사용)
    _rolling = _cfg.MODEL_MANAGEMENT.get("rolling_train_months", 0)
    _rolling_applied = False
    X_final, y_final = X, y
    if _rolling > 0 and _ts_series is not None:
        _cutoff = _ts_series.max() - pd.DateOffset(months=_rolling)
        _mask = _ts_series >= _cutoff
        _pos_in_window = int((y[_mask] == 1).sum())
        if _mask.sum() >= 200 and _pos_in_window >= 20:
            X_final = X[_mask]
            y_final = y[_mask]
            _rolling_applied = True
            logger.info(f"🔄 최종 모델 롤링 창: 최근 {_rolling}개월 → {len(X_final)}행 / 양성 {_pos_in_window}개 (전체 {len(X)}행)")
        else:
            logger.warning(f"⚠️ 롤링 창 데이터 부족 ({_mask.sum()}행/{_pos_in_window}양성) — 전체 데이터 사용")

    base_model = xgb.XGBClassifier(**best_params, random_state=_cfg.RANDOM_SEED, eval_metric='auc', n_jobs=-1)
    # PurgedTimeSeriesSplit: 시계열 순서 보존 + purge gap으로 캘리브레이션 단계의 라벨 누수도 차단.
    _cal_cv = PurgedTimeSeriesSplit(n_splits=3, gap=_cfg.LABELING.get("forward_bars", 4))
    final_model = CalibratedClassifierCV(base_model, method='isotonic', cv=_cal_cv)
    final_model.fit(X_final, y_final)

    # 롤링 윈도우 적용 시 Optuna threshold가 전체-데이터 CV 기준으로 최적화됐으므로
    # 12개월 데이터로 재학습된 모델의 확률 분포와 미스매치 발생 → holdout으로 재탐색
    if _rolling_applied:
        _thresh_split = int(len(X_final) * 0.80)
        _thresh_purge = max(0, _thresh_split - _purge_n)
        X_tv = X_final.iloc[_thresh_purge + _purge_n:]  # purge gap 제거
        y_tv = y_final.iloc[_thresh_purge + _purge_n:]
        if len(X_tv) >= 30 and int((y_tv == 1).sum()) >= 5:
            _tv_probs = final_model.predict_proba(X_tv)[:, 1]
            _optuna_thresh = best_threshold
            _best_t, _best_f1 = best_threshold, -1.0
            for _t in np.arange(0.05, 0.81, 0.01):
                _preds = (_tv_probs >= _t).astype(int)
                if _preds.sum() < 2:
                    continue
                _prec = precision_score(y_tv, _preds, zero_division=0)
                _rec  = recall_score(y_tv, _preds, zero_division=0)
                _f1   = 2 * _prec * _rec / (_prec + _rec + 1e-8)
                if _f1 > _best_f1:
                    _best_f1, _best_t = _f1, float(_t)
            best_threshold = _best_t
            logger.info(
                f"🔄 롤링 threshold 재탐색: {_best_t:.4f} (Optuna: {_optuna_thresh:.4f}, "
                f"holdout F1: {_best_f1:.4f}, n={len(X_tv)}행)"
            )
        else:
            logger.warning(f"⚠️ threshold 재탐색 holdout 부족 ({len(X_tv)}행) — Optuna threshold 유지")

    # 7. 피처 중요도 분석 — 전체 calibrated fold 평균 (단일 fold는 과적합 편향 위험)
    try:
        all_imps = [
            cc.estimator.feature_importances_
            for cc in final_model.calibrated_classifiers_
            if hasattr(cc.estimator, 'feature_importances_')
        ]
        if all_imps:
            importances = np.mean(all_imps, axis=0)
        else:
            raise AttributeError("feature_importances_ 없음")
    except (AttributeError, IndexError):
        fallback = xgb.XGBClassifier(**best_params, random_state=_cfg.RANDOM_SEED, eval_metric='auc', n_jobs=-1)
        fallback.fit(X_final, y_final, verbose=False)
        importances = fallback.feature_importances_

    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    logger.info("\n📊 [수익에 기여도가 높은 상위 10개 지표]")
    for idx, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
        logger.info(f"  {idx:2d}. {row['Feature']:25s} → {row['Importance']*100:6.2f}%")

    # 2순위: HMM 국면별 분기 모델 학습 (best_params 재사용, Optuna 추가 없음)
    regime_models = {}
    if _cfg.MODEL_MANAGEMENT.get("regime_models_enabled", False):
        logger.info("\n🧩 [HMM 국면별 분기 모델 학습]")
        _purge = _cfg.LABELING.get("forward_bars", 4)
        regime_models = _train_regime_xgb(X_final, y_final, best_params, _purge)
        if regime_models:
            logger.info(f"  → 국면 모델 생성: {list(regime_models.keys())}")
        else:
            logger.info("  → 전체 단일 모델 사용 (국면 분리 조건 미충족)")

    return final_model, best_threshold, importance_df, stats, regime_models

# ============================================================================
# 5-B. LightGBM 최적화 훈련 + XGBoost 소프트보팅 앙상블
# ============================================================================

class EnsembleModel:
    """
    XGBoost + LightGBM 소프트보팅 앙상블.
    xgb_regime_models: HMM 국면별 XGBoost 딕셔너리 (없으면 전체 xgb_model 사용)
    recal_calibrator: 주간 온라인 재캘리브레이션용 post-hoc IsotonicRegression (None이면 미적용)
    """
    def __init__(self, xgb_model, lgbm_model, lgbm_weight: float = 0.4,
                 xgb_regime_models: dict = None):
        self.xgb_model = xgb_model
        self.lgbm_model = lgbm_model
        self.lgbm_weight = lgbm_weight
        self.xgb_weight = 1.0 - lgbm_weight
        self.xgb_regime_models = xgb_regime_models or {}
        self.recal_calibrator = None  # fit_recalibration()으로 주기적 갱신

    def _xgb_prob(self, X: pd.DataFrame) -> np.ndarray:
        """HMM 국면 모델이 있으면 행별 분기, 없으면 전체 모델 사용."""
        regime_models = getattr(self, "xgb_regime_models", {}) or {}
        if not regime_models or "HMM_Bull" not in X.columns or "HMM_Bear" not in X.columns:
            return self.xgb_model.predict_proba(X)[:, 1]

        probs = self.xgb_model.predict_proba(X)[:, 1].copy()
        bull_mask     = (X["HMM_Bull"] == 1).values
        bear_mask     = (X["HMM_Bear"] == 1).values
        sideways_mask = ~bull_mask & ~bear_mask

        for mask, name in [(bull_mask, "bull"), (sideways_mask, "sideways"), (bear_mask, "bear")]:
            if mask.any() and name in regime_models:
                probs[mask] = regime_models[name].predict_proba(X[mask])[:, 1]
        return probs

    def _raw_blend(self, X: pd.DataFrame) -> np.ndarray:
        xgb_prob  = self._xgb_prob(X)
        lgbm_prob = self.lgbm_model.predict_proba(X)[:, 1]
        return self.xgb_weight * xgb_prob + self.lgbm_weight * lgbm_prob

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        blended = self._raw_blend(X)
        # post-hoc 캘리브레이터가 있으면 적용 (구 pkl 호환: getattr 폴백)
        cal = getattr(self, 'recal_calibrator', None)
        if cal is not None:
            blended = np.clip(cal.predict(blended), 0.0, 1.0)
        return np.column_stack([1 - blended, blended])

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def fit_recalibration(self, X: pd.DataFrame, y: pd.Series) -> None:
        """최근 데이터로 post-hoc isotonic 재캘리브레이션. 기존 가중치 불변."""
        from sklearn.isotonic import IsotonicRegression
        raw = self._raw_blend(X)
        cal = IsotonicRegression(out_of_bounds='clip')
        cal.fit(raw, y)
        self.recal_calibrator = cal


def optimize_and_train_lgbm(
    df: pd.DataFrame,
    n_trials: int = 50,
) -> Tuple[Optional[object], float]:
    """
    LightGBM + Optuna Walk-Forward 최적화.
    """
    if not _LGBM_AVAILABLE:
        logger.warning("LightGBM 미설치 — pip install lightgbm. LightGBM 훈련 스킵.")
        return None, 0.65

    features, _ = validate_features(df, EXCLUDE_COLS)
    if not features:
        return None, 0.65

    X = df[features].copy()
    y = df["Target"] if "Target" in df.columns else df["target"]
    valid_idx = ~(X.isna().any(axis=1) | y.isna())
    X, y = X[valid_idx], y[valid_idx]

    _ts_series_lgbm = pd.to_datetime(df["timestamp"]).reindex(X.index) if "timestamp" in df.columns else None

    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    balance_ratio = float(np.clip(neg / pos if pos > 0 else 1.0, 1.0, 10.0))

    _tp_mult = _cfg.LABELING.get("atr_tp_mult", 2.0)
    _sl_mult = _cfg.LABELING.get("atr_sl_mult", 1.0)

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 400),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth":         trial.suggest_int("max_depth", 3, 5),          # 7→5
            "num_leaves":        trial.suggest_int("num_leaves", 15, 31),        # 63→31: 2^5=32 상한
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),# 최소 리프 샘플 추가
            "subsample":         trial.suggest_float("subsample", 0.6, 0.9),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 0.9),
            "reg_alpha":         trial.suggest_float("reg_alpha", 0.0, 5.0),    # L1 추가
            "reg_lambda":        trial.suggest_float("reg_lambda", 1.0, 10.0),  # L2 추가
            "scale_pos_weight":  balance_ratio,
            "random_state":      _cfg.RANDOM_SEED,
            "n_jobs":            -1,
            "verbose":           -1,
        }
        
        # Threshold 하한 0.25 강제: ai_exit_prob(0.15)보다 반드시 높아야 진입→즉시청산 버그 방지
        threshold = trial.suggest_float("threshold", 0.25, 0.80)
        
        n_splits = 5
        test_size = len(X) // (n_splits + 2)
        _purge_gap = _cfg.LABELING.get("forward_bars", 4)
        tscv = PurgedTimeSeriesSplit(n_splits=n_splits, test_size=test_size,
                                     max_train_size=test_size * 3, gap=_purge_gap)
        expectancies = []
        fold_prec_scores = []
        _lgbm_min_prec = _cfg.MODEL_MANAGEMENT.get("min_precision", 0.333)
        for train_idx, val_idx in tscv.split(X):
            if train_idx.max() >= val_idx.min():
                continue
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            try:
                m = lgb.LGBMClassifier(**params)
                m.fit(X_tr, y_tr)
                cal_size = max(10, len(X_val) // 2)
                X_cal, X_eval = X_val.iloc[:cal_size], X_val.iloc[cal_size:]
                y_cal, y_eval = y_val.iloc[:cal_size], y_val.iloc[cal_size:]
                if len(X_eval) < 10:
                    continue
                cal = CalibratedClassifierCV(m, method="isotonic", cv="prefit")
                cal.fit(X_cal, y_cal)
                probs = cal.predict_proba(X_eval)[:, 1]
                preds = (probs >= threshold).astype(int)
                n_trades = int(preds.sum())
                if n_trades >= 10:
                    prec = precision_score(y_eval, preds, zero_division=0)
                    rec  = recall_score(y_eval, preds, zero_division=0)
                    if rec < 0.05:
                        expectancies.append(-10.0)
                        continue
                    fold_prec_scores.append(prec)
                    f1 = f1_score(y_eval, preds, zero_division=0)
                    expectancy = (prec * _tp_mult) - ((1 - prec) * _sl_mult)
                    # 과적합 페널티: train precision과 val precision 차이가 10% 초과 시 감점
                    train_prec = precision_score(y_tr, (cal.predict_proba(X_tr)[:, 1] >= threshold).astype(int), zero_division=0)
                    overfit_gap = train_prec - prec
                    overfit_penalty = max(0.0, overfit_gap - 0.10) * 3.0
                    expectancies.append(expectancy * (1.0 + 0.5 * f1) * max(0.0, 1.0 - overfit_penalty))
            except Exception:
                continue
        if len(expectancies) < 4:
            return -999.0
        base_score = float(np.mean(expectancies) - 0.5 * np.std(expectancies))
        avg_prec_lgbm = float(np.mean(fold_prec_scores)) if fold_prec_scores else 0.0
        prec_factor_lgbm = min(1.5, (avg_prec_lgbm / _lgbm_min_prec) ** 2)
        return base_score * prec_factor_lgbm

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _lgbm_db = os.path.join(MODEL_DIR, "optuna_lgbm.db")
    
    # 💥 핵심 패치 1 (LGBM 버전): 과거 실패 스터디 캐시 우회 
    _study_name = f"lgbm_bot_study_f{len(features)}_{int(time.time())}"
    try:
        lgbm_study = optuna.create_study(
            study_name=_study_name,
            storage=f"sqlite:///{_lgbm_db}",
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
            load_if_exists=False,
        )
    except Exception:
        lgbm_study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        )
    study = lgbm_study
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params.copy()
    best_threshold = best_params.pop("threshold")
    best_params["scale_pos_weight"] = balance_ratio
    best_params["random_state"] = _cfg.RANDOM_SEED
    best_params["n_jobs"] = -1
    best_params["verbose"] = -1

    _rolling_lgbm = _cfg.MODEL_MANAGEMENT.get("rolling_train_months", 0)
    X_final_lgbm, y_final_lgbm = X, y
    if _rolling_lgbm > 0 and _ts_series_lgbm is not None:
        _cutoff_lgbm = _ts_series_lgbm.max() - pd.DateOffset(months=_rolling_lgbm)
        _mask_lgbm = _ts_series_lgbm >= _cutoff_lgbm
        _pos_lgbm = int((y[_mask_lgbm] == 1).sum())
        if _mask_lgbm.sum() >= 200 and _pos_lgbm >= 20:
            X_final_lgbm = X[_mask_lgbm]
            y_final_lgbm = y[_mask_lgbm]
            logger.info(f"🔄 LightGBM 롤링 창: 최근 {_rolling_lgbm}개월 → {len(X_final_lgbm)}행 / 양성 {_pos_lgbm}개 (전체 {len(X)}행)")
        elif _mask_lgbm.sum() < 200 or _pos_lgbm < 20:
            logger.warning(f"⚠️ LightGBM 롤링 창 데이터 부족 ({_mask_lgbm.sum()}행/{_pos_lgbm}양성) — 전체 데이터 사용")

    base = lgb.LGBMClassifier(**best_params)
    _cal_cv = PurgedTimeSeriesSplit(n_splits=3, gap=_cfg.LABELING.get("forward_bars", 4))
    final = CalibratedClassifierCV(base, method="isotonic", cv=_cal_cv)
    final.fit(X_final_lgbm, y_final_lgbm)

    logger.info(
        f"✅ LightGBM 훈련 완료 | threshold={best_threshold:.4f} "
        f"| best_score={study.best_value:.4f}"
    )
    return final, best_threshold


# ============================================================================
# 6. 모델 저장 (비동기 I/O 개선)
# ============================================================================
async def save_model_async(
    model: xgb.XGBClassifier,
    threshold: float,
    importance_df: pd.DataFrame,
    stats: ModelTrainingStats,
    model_name: str = "xgb_bot",
    train_cutoff_timestamp: str = None,
    oos_metrics: dict = None,
    output_dir: str = None,
    interval_thresholds: dict = None,
    interval: str = None,
) -> Tuple[str, str, str]:
    """
    모델과 설정 비동기 저장 (마이크로초 단위 성능 최적화)
    """
    start_time = time.perf_counter()  # 마이크로초 단위 측정
    
    _out = output_dir if output_dir else MODEL_DIR
    os.makedirs(_out, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    async def save_joblib_async(obj, path):
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, joblib.dump, obj, path)

    async def save_json_async(data, path):
        loop = asyncio.get_running_loop()
        def _save():
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
        await loop.run_in_executor(None, _save)

    async def save_csv_async(df, path):
        loop = asyncio.get_running_loop()
        def _save_csv():
            df.to_csv(path, index=False, encoding='utf-8')
        await loop.run_in_executor(None, _save_csv)

    # 병렬 비동기 저장
    tasks = []
    
    # 1. 모델 저장
    model_path = os.path.join(_out, f"{model_name}_{timestamp}.pkl")
    tasks.append(save_joblib_async(model, model_path))
    
    # 2. 설정 저장
    # XGBoost 부스터에서 실제 학습 순서 추출 (importance 정렬 순서와 다름)
    def _extract_training_order(m) -> list:
        for candidate in [getattr(m, 'xgb_model', None), m]:
            if candidate is None:
                continue
            try:
                names = candidate.calibrated_classifiers_[0].estimator.get_booster().feature_names
                if names:
                    return names
            except Exception:
                pass
            try:
                names = candidate.get_booster().feature_names
                if names:
                    return names
            except Exception:
                pass
        return []

    training_order = _extract_training_order(model)
    config_data = {
        "model_name": model_name,
        "timestamp": timestamp,
        "best_threshold": float(threshold),
        "train_cutoff_timestamp": train_cutoff_timestamp,
        "features": importance_df['Feature'].tolist(),
        "features_training_order": training_order,
        "feature_importances": importance_df['Importance'].tolist(),
        "training_stats": stats.to_dict(),
        "oos_metrics": oos_metrics or {},
        "interval_thresholds": interval_thresholds or {},
        "interval":          interval or _cfg.MODEL_MANAGEMENT.get("train_interval") or (
                                 _cfg.TARGET_INTERVALS[2] if len(_cfg.TARGET_INTERVALS) > 2 else "days"
                             ),
        "trailing_stop_pct": _cfg.BACKTEST.get("trailing_stop_pct", 0.03),
        "ai_exit_threshold": _cfg.BACKTEST.get("ai_exit_prob", 0.40),
        "ai_check_interval": 60,
    }
    config_path = os.path.join(_out, f"config_{timestamp}.json")
    tasks.append(save_json_async(config_data, config_path))
    
    # 3. 피처 중요도 저장
    importance_path = os.path.join(_out, f"importance_{timestamp}.csv")
    tasks.append(save_csv_async(importance_df, importance_path))
    
    # 모든 저장 작업 병렬 실행
    save_start = time.perf_counter()
    await asyncio.gather(*tasks)
    save_elapsed = time.perf_counter() - save_start
    
    total_elapsed = time.perf_counter() - start_time
    logger.info(f"💾 모델 저장 완료 (병렬 비동기, 저장: {save_elapsed:.6f}초, 총: {total_elapsed:.6f}초): {model_path}")

    # 오래된 모델 파일 정리 (비동기)
    await cleanup_old_models_async(_out)

    return model_path, config_path, importance_path

async def cleanup_old_models_async(output_dir: str):
    """오래된 모델 파일 비동기 정리"""
    try:
        retention = _cfg.MODEL_MANAGEMENT.get("retention_count", 5)
        all_pkls = sorted(
            glob.glob(os.path.join(output_dir, "*.pkl")),
            key=os.path.getctime
        )
        if len(all_pkls) > retention:
            loop = asyncio.get_running_loop()
            cleanup_tasks = []
            
            for old_pkl in all_pkls[:len(all_pkls) - retention]:
                m = re.search(r'(\d{8}_\d{6})\.pkl$', old_pkl)
                ts_part = m.group(1) if m else os.path.basename(old_pkl).replace(".pkl", "")
                
                companions = [
                    os.path.join(output_dir, f"config_{ts_part}.json"),
                    os.path.join(output_dir, f"importance_{ts_part}.csv"),
                ]
                
                for companion in companions:
                    if os.path.exists(companion):
                        cleanup_tasks.append(loop.run_in_executor(None, os.remove, companion))
                
                cleanup_tasks.append(loop.run_in_executor(None, os.remove, old_pkl))
            
            if cleanup_tasks:
                old_count = len(all_pkls) - retention
                await asyncio.gather(*cleanup_tasks)
                logger.info(f"🧹 오래된 모델 {old_count}세트 정리 완료")
                
    except Exception as e:
        logger.warning(f"모델 정리 실패: {e}")

def save_model(
    model: xgb.XGBClassifier,
    threshold: float,
    importance_df: pd.DataFrame,
    stats: ModelTrainingStats,
    model_name: str = "xgb_bot",
    train_cutoff_timestamp: str = None,
    oos_metrics: dict = None,
    output_dir: str = None,
    interval_thresholds: dict = None,
    interval: str = None,
) -> Tuple[str, str, str]:
    """
    모델과 설정 저장 (동기 래퍼 - 기존 호환성 유지)
    """
    coro = save_model_async(
        model, threshold, importance_df, stats,
        model_name, train_cutoff_timestamp, oos_metrics, output_dir,
        interval_thresholds, interval,
    )
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as _pool:
            return _pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)

# ============================================================================
# 7. 모델 로드 (새 기능)
# ============================================================================
def load_model(model_path: str) -> Optional[xgb.XGBClassifier]:
    try:
        model = joblib.load(model_path)
        logger.info(f"✅ 모델 로드 완료: {model_path}")
        return model
    except Exception as e:
        logger.error(f"❌ 모델 로드 실패: {type(e).__name__} - {e}")
        return None

def load_config(config_path: str) -> Optional[Dict]:
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logger.info(f"✅ 설정 로드 완료: {config_path}")
        return config
    except Exception as e:
        logger.error(f"❌ 설정 로드 실패: {type(e).__name__} - {e}")
        return None

# ============================================================================
# 8. 모델 평가 (새 기능)
# ============================================================================
def evaluate_model(
    model: xgb.XGBClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.5
) -> Dict:
    pred_probs = model.predict_proba(X)[:, 1]
    predictions = (pred_probs >= threshold).astype(int)
    
    metrics = {
        "precision": round(precision_score(y, predictions, zero_division=0), 4),
        "recall": round(recall_score(y, predictions, zero_division=0), 4),
        "f1": round(f1_score(y, predictions, zero_division=0), 4),
        "auc": round(roc_auc_score(y, pred_probs), 4) if len(np.unique(y)) > 1 else float("nan"),
        "total_predictions": int(len(predictions)),
        "positive_predictions": int(sum(predictions))
    }
    
    return metrics

# ============================================================================
# 9. 메인 파이프라인 실행
# ============================================================================
def _run_pipeline(
    input_dir: str = None,
    output_dir: str = None,
    coins: Optional[List[str]] = None,
    exclude_coins: Optional[set] = None,
    cat_label: str = "",
    cat_strict_oos_gate: Optional[bool] = None,
    interval: Optional[str] = None,
) -> None:
    """단일 카테고리(또는 전체) 훈련 파이프라인. main()에서 카테고리별로 호출.

    interval: 학습 인터벌 ('minute60' / 'minute15'). None이면 config 폴백.
    """
    _input_dir  = input_dir  or INPUT_DIR
    _output_dir = output_dir or MODEL_DIR

    logger.info("=" * 70)
    logger.info(f"🤖 Coin AI Bot - 머신러닝 모델 훈련 파이프라인 [{interval or 'default'}]")
    logger.info("=" * 70)

    start_time = time.time()

    try:
        # 1. 데이터 로드
        logger.info("\n📂 단계 1: 데이터 로드 및 준비")
        df, _ = load_and_prepare_data(_input_dir, coins=coins, exclude_coins=exclude_coins,
                                      interval=interval)
        
        if df is None or df.empty:
            logger.error("❌ 파이프라인 중단: 데이터 로드 실패")
            return

        # 1-B. B/C 카테고리: BTC 레퍼런스 피처 추가 (알트코인 예측 컨텍스트)
        if cat_label in ('B', 'C') and 'timestamp' in df.columns:
            _ivl_suffix = interval or 'minute60'
            _btc_csv = os.path.join(_input_dir, f'processed_KRW-BTC_{_ivl_suffix}.csv')
            _BTC_REF_FEATS = _cfg.MODEL_MANAGEMENT["btc_ref_feats"]
            if os.path.exists(_btc_csv):
                try:
                    _btc_df = pd.read_csv(_btc_csv, low_memory=False)
                    _btc_df['timestamp'] = pd.to_datetime(_btc_df['timestamp'])
                    _btc_df = _btc_df.sort_values('timestamp').drop_duplicates('timestamp')
                    _avail = [c for c in _BTC_REF_FEATS if c in _btc_df.columns]
                    _btc_ref = _btc_df[['timestamp'] + _avail].copy()
                    _btc_ref.columns = ['timestamp'] + [f'BTC_{c}' for c in _avail]
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.merge(_btc_ref, on='timestamp', how='left')
                    _btc_cols = [c for c in df.columns if c.startswith('BTC_')]
                    df[_btc_cols] = df[_btc_cols].ffill()
                    logger.info(f"  ✅ BTC 레퍼런스 피처 {len(_btc_cols)}개 추가 [{_ivl_suffix}]")
                except Exception as _e:
                    logger.warning(f"  ⚠️ BTC 레퍼런스 피처 로드 실패: {_e}")
            else:
                logger.warning(f"  ⚠️ BTC 기준 파일 없음: {_btc_csv}")

        # 2. 시간순 70/30 OOS 분할 (Look-ahead Bias 방지)
        logger.info("\n✂️  단계 2: 학습/OOS 시간순 분할 (70% 학습 / 30% 검증)")
        _purge_n = _cfg.LABELING.get("forward_bars", 4)
        cutoff_str = None
        if 'timestamp' in df.columns:
            all_ts = pd.to_datetime(df['timestamp'])
            _oos_months = _cfg.MODEL_MANAGEMENT.get("oos_months", 0)
            if _oos_months > 0:
                cutoff_time = all_ts.max() - pd.DateOffset(months=_oos_months)
            else:
                cutoff_time = all_ts.quantile(0.7)
            train_df = df[all_ts < cutoff_time].copy()
            if len(train_df) > _purge_n:
                train_df = train_df.iloc[:-_purge_n]
            oos_df   = df[all_ts >= cutoff_time].copy()
            cutoff_str = str(cutoff_time)
            logger.info(f"  • 컷오프: {cutoff_str[:19]}")
            logger.info(f"  • 학습 데이터: {len(train_df)}행 | OOS 데이터: {len(oos_df)}행")
        else:
            split_idx = int(len(df) * 0.7)
            train_df  = df.iloc[:split_idx - _purge_n].copy()
            oos_df    = df.iloc[split_idx:].copy()
            logger.warning("⚠️ timestamp 없어 행 인덱스로 70/30 분할 (시계열 보장 불가)")

        if len(train_df) < 200:
            logger.error(f"❌ 학습 데이터 부족: {len(train_df)}행 (최소 200 필요)")
            return

        # 3. 모델 훈련 및 최적화 (학습 데이터만 사용)
        logger.info("\n🎯 단계 3: 하이퍼파라미터 최적화 (XGBoost)")
        n_trials = int(_cfg.MODEL_MANAGEMENT.get("n_trials", 100))
        xgb_model, optimal_threshold, importance_df, train_stats, regime_models = optimize_and_train_bot(
            train_df, n_trials=n_trials, verbose=True
        )

        if xgb_model is None:
            logger.error("❌ 파이프라인 중단: XGBoost 모델 훈련 실패")
            return

        # 3-B. LightGBM 앙상블 (config.ENSEMBLE.enabled 시 병렬 훈련)
        ensemble_cfg = getattr(_cfg, "ENSEMBLE", {})
        model = xgb_model  # 기본값: XGBoost 단독
        final_threshold = optimal_threshold

        if ensemble_cfg.get("enabled", False) and _LGBM_AVAILABLE:
            logger.info("\n🎯 단계 3-B: LightGBM 보조 모델 훈련")
            lgbm_trials = max(30, n_trials // 2)
            lgbm_model, lgbm_threshold = optimize_and_train_lgbm(train_df, n_trials=lgbm_trials)
            if lgbm_model is not None:
                lgbm_weight = ensemble_cfg.get("lgbm_weight", 0.4)
                model = EnsembleModel(xgb_model, lgbm_model, lgbm_weight=lgbm_weight,
                                      xgb_regime_models=regime_models)
                # 두 모델 최적 임계값의 가중 평균
                final_threshold = (
                    (1 - lgbm_weight) * optimal_threshold + lgbm_weight * lgbm_threshold
                )
                logger.info(
                    f"✅ 앙상블 구성: XGBoost×{1-lgbm_weight:.1f} + "
                    f"LightGBM×{lgbm_weight:.1f} | threshold={final_threshold:.4f}"
                )
            else:
                logger.warning("⚠️ LightGBM 훈련 실패 — XGBoost 단독 사용")
        elif ensemble_cfg.get("enabled", False) and not _LGBM_AVAILABLE:
            logger.warning("⚠️ LightGBM 미설치 (pip install lightgbm) — XGBoost 단독 사용")

        # 4. OOS 평가 (순서 보정 패치 적용)
        logger.info("\n📈 단계 4: OOS(Out-of-Sample) 모델 평가")
        
        # 훈련 시 주입된 '원본 순서'의 피처 리스트 재추출
        trained_features, _ = validate_features(train_df, EXCLUDE_COLS)
        
        X_oos = oos_df[trained_features].copy()
        y_oos = oos_df['Target'] if 'Target' in oos_df.columns else oos_df['target']
        
        # 결측치 제거 및 정합성 보장
        valid_idx = ~(X_oos.isna().any(axis=1) | y_oos.isna())
        X_oos, y_oos = X_oos[valid_idx], y_oos[valid_idx]

        # 예측 및 채점
        metrics = evaluate_model(model, X_oos, y_oos, final_threshold)
        
        logger.info(f"  • [OOS] 정밀도 (Precision): {metrics['precision']:.4f}")
        logger.info(f"  • [OOS] 재현율 (Recall):    {metrics['recall']:.4f}")
        logger.info(f"  • [OOS] F1 스코어:          {metrics['f1']:.4f}")
        logger.info(f"  • [OOS] ROC-AUC:            {metrics['auc']:.4f}")
        logger.info(f"  • [OOS] 예측 개수: {metrics['total_predictions']} | 매수 신호: {metrics['positive_predictions']}")

        oos_positive_ratio = metrics['positive_predictions'] / max(metrics['total_predictions'], 1)
        if oos_positive_ratio < 0.05:
            logger.warning(
                f"⚠️ OOS 양성 신호 비율 {oos_positive_ratio*100:.1f}% < 5% "
                f"— 임계값({final_threshold:.4f})이 너무 높거나 모델이 거의 신호를 내지 않음"
            )

        _mm           = _cfg.MODEL_MANAGEMENT
        MIN_PRECISION = _mm.get("min_precision", 0.60)
        MIN_RECALL    = _mm.get("min_recall",    0.10)
        MIN_SIGNALS   = _mm.get("min_signals",   30)
        _strict_gate  = cat_strict_oos_gate if cat_strict_oos_gate is not None else _mm.get("strict_oos_gate", True)

        def _oos_fail(msg: str):
            if _strict_gate:
                raise ValueError(msg)
            logger.warning(msg + " — strict_oos_gate=False이므로 저장 강행")

        if metrics['precision'] < MIN_PRECISION:
            _oos_fail(f"[성능 미달] OOS 정밀도: {metrics['precision']:.4f} < {MIN_PRECISION}")
        elif metrics['recall'] < MIN_RECALL:
            _oos_fail(f"[성능 미달] OOS 재현율: {metrics['recall']:.4f} < {MIN_RECALL}")
        elif metrics['positive_predictions'] < MIN_SIGNALS:
            _oos_fail(f"[성능 미달] OOS 매수 신호 부족: {metrics['positive_predictions']}회 < {MIN_SIGNALS}")
        else:
            logger.info(
                f"  ✅ 성능 컷오프 통과: 정밀도 {metrics['precision']:.4f}, "
                f"재현율 {metrics['recall']:.4f}, 신호 {metrics['positive_predictions']}회"
            )

        # 인터벌별 OOS 정밀도 → adaptive threshold 계산
        interval_thresholds = {}
        _tp = float(_cfg.LABELING.get('atr_tp_mult', 1.5))
        _sl = float(_cfg.LABELING.get('atr_sl_mult', 1.0))
        # 손익분기 정밀도 = SL/(TP+SL) + 실비용 보정(0.04 = 수수료+슬리피지 round-trip ~0.4%)
        p_break_even = round(_sl / (_tp + _sl) + 0.04, 3)
        if '_interval' in oos_df.columns:
            oos_probs = model.predict_proba(X_oos)[:, 1]
            valid_oos = oos_df.loc[valid_idx].copy()
            valid_oos['_prob'] = oos_probs
            valid_oos['_y'] = y_oos.values
            logger.info(f"\n📐 인터벌별 OOS 정밀도 분석 (손익분기 기준: {p_break_even:.3f}):")
            for ivl, grp in valid_oos.groupby('_interval'):
                if len(grp) < 30:
                    continue
                probs_ivl = grp['_prob'].values
                y_ivl = grp['_y'].values
                best_thresh, best_prec = final_threshold, 0.0
                for t in np.arange(0.35, 0.90, 0.025):
                    preds_t = (probs_ivl >= t).astype(int)
                    if preds_t.sum() < 5:
                        break
                    prec_t = precision_score(y_ivl, preds_t, zero_division=0)
                    rec_t = recall_score(y_ivl, preds_t, zero_division=0)
                    if rec_t >= 0.05 and prec_t > best_prec:
                        best_prec = prec_t
                        best_thresh = float(t)
                preds_f = (probs_ivl >= best_thresh).astype(int)
                prec_f = float(precision_score(y_ivl, preds_f, zero_division=0))
                rec_f = float(recall_score(y_ivl, preds_f, zero_division=0))
                n_sig = int(preds_f.sum())
                skip = prec_f < p_break_even or n_sig < 5
                # 국면별 OOS 정밀도 (Bull 국면 재활성화 판단용)
                regime_precision: dict = {}
                if "HMM_Bear" in grp.columns and "HMM_Bull" in grp.columns:
                    for _rn, _rm in [
                        ("bull",     grp["HMM_Bull"] == 1),
                        ("sideways", (grp["HMM_Bear"] == 0) & (grp["HMM_Bull"] == 0)),
                        ("bear",     grp["HMM_Bear"] == 1),
                    ]:
                        _rg = grp[_rm]
                        if len(_rg) < 10:
                            continue
                        _rp = (_rg["_prob"].values >= best_thresh).astype(int)
                        if _rp.sum() < 3:
                            continue
                        regime_precision[_rn] = round(
                            float(precision_score(_rg["_y"].values, _rp, zero_division=0)), 4
                        )
                interval_thresholds[ivl] = {
                    'threshold': round(best_thresh, 4),
                    'precision': round(prec_f, 4),
                    'recall': round(rec_f, 4),
                    'n_signals': n_sig,
                    'break_even': p_break_even,
                    'skip': bool(skip),
                    'regime_precision': regime_precision,
                }
                icon = '✅' if not skip else '⛔'
                _rp_log = (
                    "  국면별: " + " | ".join(f"{k}={v:.3f}" for k, v in regime_precision.items())
                ) if regime_precision else ""
                logger.info(
                    f"  {icon} [{ivl}] prec={prec_f:.3f} thresh={best_thresh:.4f} "
                    f"신호={n_sig}회 {'→ 비활성' if skip else '→ 활성'}{_rp_log}"
                )

        # 학습 인터벌이 아닌 타임프레임: 피처 분포 불일치로 실전 적용 불가 → 강제 비활성
        train_ivl = interval or _cfg.MODEL_MANAGEMENT.get("train_interval", "minute60")
        for _ivl in _cfg.TARGET_INTERVALS:
            if _ivl not in interval_thresholds:
                interval_thresholds[_ivl] = {
                    'threshold': 1.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'n_signals': 0,
                    'break_even': p_break_even,
                    'skip': True,
                    'regime_precision': {},
                }
                logger.info(
                    f"  ⛔ [{_ivl}] 학습 인터벌({train_ivl})과 불일치 → 비활성 (피처 분포 불일치)"
                )

         # 5. 모델 저장
        logger.info("\n💾 단계 5: 모델 저장")
        _base_tag = "ensemble_bot" if "EnsembleModel" in str(type(model)) else "xgb_bot"
        model_tag = f"{cat_label}_{_base_tag}" if cat_label else _base_tag

        # 💥 핵심 패치: 반환값이 튜플이든 단일 문자열이든 모두 안전하게 받아내는 방탄 로직
        save_result = save_model(
            model=model,
            threshold=final_threshold,
            importance_df=importance_df,  # 반드시 중요도 DataFrame 원본을 전달
            stats=train_stats,
            model_name=model_tag,
            train_cutoff_timestamp=cutoff_str,
            oos_metrics=metrics,
            output_dir=_output_dir,
            interval_thresholds=interval_thresholds,
            interval=interval,
        )
        
        # 튜플 언패킹 에러 방어 및 안전한 파일명 추출
        if isinstance(save_result, tuple):
            final_model_path = str(save_result[0])
        else:
            final_model_path = str(save_result)
            
        safe_model_filename = os.path.basename(final_model_path) if final_model_path else "알_수_없는_파일.pkl"
        
        # 최종 요약
        elapsed_time = time.time() - start_time
        logger.info("\n" + "=" * 70)
        logger.info("✨ 머신러닝 훈련 및 모델 생성이 성공적으로 완료되었습니다!")
        logger.info("=" * 70)
        logger.info(f"  • 총 소요 시간: {elapsed_time:.2f}초")
        logger.info(f"  • 저장된 모델: {safe_model_filename}")
        logger.info(f"  • OOS 정밀도: {metrics.get('precision', 0):.4f}")
        logger.info("🚀 이제 'python main.py backtest'를 실행할 준비가 되었습니다.")
        
    except Exception as e:
        logger.error(f"❌ 파이프라인 실행 중 오류: {type(e).__name__} - {e}", exc_info=True)
        raise  # run_ml_only()가 False를 반환하게 해 후속 백테스트 차단


def main() -> None:
    """카테고리 × 인터벌 분리 모델 학습 (Model_A/B/C × minute60/minute15).
    MTF_ENSEMBLE.enabled=True면 두 인터벌 모두 학습. False면 minute60 단독."""
    categories = getattr(_cfg, "COIN_CATEGORIES", None)
    if not categories:
        _run_pipeline()
        return

    # 학습할 인터벌 목록 결정
    mtf_cfg = getattr(_cfg, "MTF_ENSEMBLE", {})
    if mtf_cfg.get("enabled", False):
        train_intervals = _cfg.MODEL_MANAGEMENT.get("train_intervals", ["minute60", "minute15"])
    else:
        train_intervals = [_cfg.MODEL_MANAGEMENT.get("train_interval", "minute60")]

    # 실제 수집된 코인 동적 스캔 (data_processed/*.csv)
    _available_coins: set = {
        _coin_from_filename(f)
        for f in glob.glob(f"{INPUT_DIR}/*.csv")
        if _coin_from_filename(f)
    }
    logger.info(f"📂 data_processed 수집 코인 ({len(_available_coins)}개): {sorted(_available_coins)}")

    # COIN_TIER_MAP으로 수집 코인을 A/B/C 동적 분배
    tier_coins: dict = {"A": [], "B": [], "C": []}
    get_cat = getattr(_cfg, "get_coin_category", lambda _: "C")
    for coin in sorted(_available_coins):
        tier = get_cat(coin)
        tier_coins.setdefault(tier, []).append(coin)
    for t, lst in tier_coins.items():
        logger.info(f"  Model_{t}: {lst}")

    results: dict = {}
    for ivl in train_intervals:
        is_15m = ivl == "minute15"
        dir_suffix = "_15m" if is_15m else ""
        logger.info(f"\n{'#'*70}")
        logger.info(f"⏱️  인터벌: {ivl} ({'타이밍 모델' if is_15m else '방향성 모델'})")
        logger.info(f"{'#'*70}")

        for cat_key, cat_info in categories.items():
            model_dir_key = f"model_dir{dir_suffix}"
            cat_dir   = cat_info.get(model_dir_key) or cat_info["model_dir"]
            cat_coins = tier_coins.get(cat_key, [])
            result_key = f"{cat_key}_{ivl}"

            if cat_info.get("skip_training"):
                logger.info(f"⏭️  Model_{cat_key}[{ivl}] skip_training=True — 학습 건너뜀")
                results[result_key] = "⏭️ 스킵 (skip_training)"
                continue

            if not cat_coins:
                logger.warning(f"⚠️  Model_{cat_key}[{ivl}] 학습 가능한 코인 없음 — 스킵")
                results[result_key] = "⏭️ 스킵 (수집 데이터 없음)"
                continue

            os.makedirs(cat_dir, exist_ok=True)
            logger.info(f"\n{'='*70}")
            logger.info(f"📦 Model_{cat_key}[{ivl}]: {cat_info['name']} — {cat_coins}")
            logger.info(f"{'='*70}")
            try:
                _run_pipeline(
                    input_dir=INPUT_DIR,
                    output_dir=cat_dir,
                    coins=cat_coins,
                    cat_label=cat_key,
                    cat_strict_oos_gate=cat_info.get("strict_oos_gate"),
                    interval=ivl,
                )
                results[result_key] = "✅ 성공"
            except Exception as exc:
                logger.error(f"❌ Model_{cat_key}[{ivl}] 학습 실패: {exc}", exc_info=True)
                results[result_key] = f"❌ 실패: {exc}"

    logger.info("\n" + "=" * 70)
    logger.info("📊 카테고리 × 인터벌 학습 결과 요약")
    logger.info("=" * 70)
    for k, v in results.items():
        logger.info(f"  {k}: {v}")


if __name__ == "__main__":
    main()