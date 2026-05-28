#feature_engineering.py
import os
import glob
import logging
import time
import warnings
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import RobustScaler
from typing import Optional, Dict, Tuple
import config as _cfg

# ============================================================================
# 1. 기술 지표 파라미터 (config.py 단일 소스 — 중복 하드코딩 제거)
# ============================================================================
_ti = _cfg.TECHNICAL_INDICATORS
SMA_SHORT    = _ti["SMA_SHORT"]
SMA_LONG     = _ti["SMA_LONG"]
RSI_PERIOD   = _ti["RSI_PERIOD"]
MACD_FAST    = _ti["MACD_FAST"]
MACD_SLOW    = _ti["MACD_SLOW"]
MACD_SIGNAL  = _ti["MACD_SIGNAL"]
BB_PERIOD    = _ti["BB_PERIOD"]
BB_STD_DEV   = _ti["BB_STD_DEV"]
VOLUME_PERIOD = _ti["VOLUME_PERIOD"]
ATR_PERIOD   = _ti["ATR_PERIOD"]
CMO_PERIOD   = _ti["CMO_PERIOD"]
del _ti
_lbl               = _cfg.LABELING                        # config.LABELING 단일 소스
FEE_THRESHOLD      = _lbl["fee_threshold"]
SLIPPAGE_ROUNDTRIP = _lbl["slippage_roundtrip"]
LABEL_COST_BUFFER  = _cfg.LABEL_COST_BUFFER  # fee×2 + slippage (config 단일 소스)
FORWARD_BARS       = _lbl["forward_bars"]
TB_WINDOW          = FORWARD_BARS
ATR_TP_MULT        = _lbl["atr_tp_mult"]
ATR_SL_MULT        = _lbl["atr_sl_mult"]

# 디렉토리 설정 (config.DIRECTORIES 단일 소스)
DATA_DIR   = _cfg.DIRECTORIES["data"]
OUTPUT_DIR = _cfg.DIRECTORIES["data_processed"]
LOG_DIR    = _cfg.DIRECTORIES["logs"]

# WFO HMM 결과 캐시: runtime inference 매 분 재실행 방지 (TTL 1시간)
_HMM_WFO_CACHE: dict = {}  # key: (n_rows, close_tail_hash) → (states_array, expire_epoch)
_HMM_WFO_CACHE_TTL: int  = 1800  # seconds (30분 — 메모리 GC용, 실무효화는 캐시 키 변경으로 처리)

# ============================================================================
# 2. 로깅 설정
# ============================================================================
def setup_logger(name: str = 'Coin_AI_Bot.feature_eng') -> logging.Logger:
    """로거 설정. 'Coin_AI_Bot.*' 계층으로 main.py FileHandler에 자동 전파됨"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

logger = setup_logger()

# ============================================================================
# 3. 피처 엔지니어링 통계 클래스
# ============================================================================
class FeatureEngineeringStats:
    """피처 엔지니어링 처리 통계"""
    def __init__(self):
        self.input_rows = 0
        self.output_rows = 0
        self.removed_null_rows = 0
        self.features_added = []
        self.processing_time = 0.0
    
    def __str__(self):
        return (f"입력행:{self.input_rows}, 출력행:{self.output_rows}, "
                f"제거행:{self.removed_null_rows}, 피처수:{len(self.features_added)}")

# ============================================================================
# 4. Triple Barrier 라벨링 함수
# ============================================================================
def _get_primary_signal_mask(df: pd.DataFrame, enable_signal_c: bool = True) -> pd.Series:
    """
    Meta-Labeling 1단계: Primary Signal 정의.

    Signal A — BB 스퀴즈 탈출 (추세 전환):
      BB_Width 20봉 하위 30% 수축 후 1~3봉 내 확장 & Volume_Surge > 1.5.
      MTF_4H_Trend 동조 시 Volume_Surge > 1.5, 역방향이라도 Volume_Surge > 2.0이면 허용.

    Signal B — SMA 골든크로스 zone (모멘텀 추종):
      SMA 크로스 후 PRIMARY_SIGNAL_ZONE봉 이내 & Volume_Surge > 1.2.

    Signal C — 과매도 반등 (하락장·횡보장 저점 포착, enable_signal_c=True 시만):
      RSI_Short 3봉 내 < 30 이후 상승 전환 & Volume_Surge > 1.3.
      B·C 티어만 활성 (A 티어는 ATR 작아 2×ATR 목표 미도달 → 노이즈).

    Filter — RSI 30~75: 기존 40~70 완화. 과매도 반등·모멘텀 연장 포착.
    MTF_4H_Trend: 전역 AND 차단에서 제거 → Signal A 내 소프트 조건으로 이동.
    """
    required_cols = ['SMA_Cross', 'RSI', 'RSI_Short', 'BB_Width', 'Volume_Surge']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise AssertionError(
            f"_get_primary_signal_mask: 필수 컬럼 미존재 {missing}. "
            f"add_technical_indicators 내부 계산 순서를 확인하세요."
        )

    # Signal A: BB 스퀴즈 탈출 (MTF 동조 → 기존 기준 / 역방향 → 강한 거래량 요구)
    bb_squeeze = df['BB_Width'] < df['BB_Width'].rolling(20, min_periods=10).quantile(0.30)
    was_in_squeeze = (
        bb_squeeze.shift(1, fill_value=False)
        | bb_squeeze.shift(2, fill_value=False)
        | bb_squeeze.shift(3, fill_value=False)
    )
    bb_expanding = df['BB_Width'] > df['BB_Width'].shift(1)
    if 'MTF_4H_Trend' in df.columns and df['MTF_4H_Trend'].max() > 0:
        mtf_ok = df['MTF_4H_Trend'] == 1
        signal_a = was_in_squeeze & bb_expanding & (
            (mtf_ok & (df['Volume_Surge'] > 1.5))
            | (~mtf_ok & (df['Volume_Surge'] > 2.0))  # 역방향은 강한 거래량 필수
        )
    else:
        signal_a = was_in_squeeze & bb_expanding & (df['Volume_Surge'] > 1.5)

    # Signal B: SMA 골든크로스 zone + 거래량 중간 확인
    sma_cross_rising = (df['SMA_Cross'] == 1) & (df['SMA_Cross'].shift(1) == 0)
    in_primary_zone = (
        sma_cross_rising.rolling(window=PRIMARY_SIGNAL_ZONE, min_periods=1).max()
        .fillna(0).astype(bool)
    )
    signal_b = in_primary_zone & (df['Volume_Surge'] > 1.2)

    # RSI 30~75 필터 (기존 40~70 완화). Signal C는 하단 차단 제외 (과매도 포착 목적)
    rsi_not_overbought = df['RSI'] <= 75
    rsi_not_oversold   = df['RSI'] >= 30

    base = (signal_a | signal_b) & rsi_not_oversold & rsi_not_overbought

    if not enable_signal_c:
        return base

    # Signal C: 과매도 반등 (B·C 티어만 — ATR 작은 A 티어는 노이즈)
    rsi_short_was_oversold = (
        (df['RSI_Short'] < 30).shift(1, fill_value=False)
        | (df['RSI_Short'] < 30).shift(2, fill_value=False)
        | (df['RSI_Short'] < 30).shift(3, fill_value=False)
    )
    rsi_short_turning_up = df['RSI_Short'] > df['RSI_Short'].shift(1)
    signal_c = rsi_short_was_oversold & rsi_short_turning_up & (df['Volume_Surge'] > 1.3)

    return base | (signal_c & rsi_not_overbought)


def triple_barrier_label_dynamic(
    df: pd.DataFrame,
    tp_mult: float = ATR_TP_MULT,
    sl_mult: float = ATR_SL_MULT,
    window: int = TB_WINDOW,
    use_meta_labeling: bool = True,
    enable_signal_c: bool = True,
) -> pd.Series:
    """
    ATR 기반 동적 Triple Barrier Method + Meta-Labeling.
    고정 비율 대신 각 봉의 ATR_Ratio로 익절/손절선을 계산 →
    변동성 높을 때 넓게, 횡보장엔 타이트하게 자동 적응.

    use_meta_labeling=True (기본): SMA 골든크로스 + RSI가 성립한 Primary Zone 봉에만 라벨 생성.
    나머지 봉은 NaN → dropna()로 제거. 횡보 노이즈 학습 방지.

    전제: df에 'ATR_Ratio', 'SMA_Cross', 'RSI' 컬럼이 이미 존재해야 함.
    Pessimistic tie-breaking: 같은 봉에서 둘 다 터치 시 저가(손절) 우선.
    """
    required = ['ATR_Ratio', 'SMA_Cross', 'RSI'] if use_meta_labeling else ['ATR_Ratio']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"{col} 컬럼이 없습니다. add_technical_indicators에서 계산 후 호출하세요.")

    n = len(df)
    labels = np.full(n, np.nan, dtype=float)
    close_v     = df['close'].values
    high_v      = df['high'].values
    low_v       = df['low'].values
    atr_ratio_v = df['ATR_Ratio'].values

    # Meta-Labeling: Primary Signal이 없는 봉은 처음부터 스킵
    if use_meta_labeling:
        primary_mask = _get_primary_signal_mask(df, enable_signal_c=enable_signal_c).values
    else:
        primary_mask = np.ones(n, dtype=bool)

    for i in range(n - window):
        if not primary_mask[i]:
            continue  # NaN 유지 → dropna()로 제거

        atr_pct = atr_ratio_v[i]
        if np.isnan(atr_pct) or atr_pct <= 0:
            continue

        # 순수익(Net Profit) 필터: 왕복 수수료+슬리피지(0.8%)를 커버 못하면 NaN 유지
        if atr_pct * tp_mult <= LABEL_COST_BUFFER:
            continue

        entry = close_v[i]
        tp = entry * (1.0 + atr_pct * tp_mult)
        sl = entry * (1.0 - atr_pct * sl_mult)

        window_highs = high_v[i + 1:i + window + 1]
        window_lows  = low_v[i + 1:i + window + 1]
        sl_hits = window_lows <= sl
        tp_hits = window_highs >= tp
        # Pessimistic tie-breaking: SL takes priority on the same bar
        first_sl = int(np.argmax(sl_hits)) if sl_hits.any() else window
        first_tp = int(np.argmax(tp_hits)) if tp_hits.any() else window

        if sl_hits.any() and first_sl <= first_tp:
            labels[i] = 0
        elif tp_hits.any() and first_tp < first_sl:
            labels[i] = 1
        # else: timeout → labels[i] remains np.nan → dropna 시 훈련 제외 (모호한 결과 학습 방지)

    return pd.Series(labels, index=df.index, dtype=float)


MIN_CANDLE_VALUE_KRW = int(_cfg.LABELING.get("min_candle_value_krw", 10_000_000))
PRIMARY_SIGNAL_ZONE = _cfg.LABELING.get("primary_signal_zone", 3)  # config 단일 소스

def _apply_volume_filter(df: pd.DataFrame) -> pd.DataFrame:
    """거래대금 기준 미달 캔들을 NaN으로 마스킹하여 좀비 코인 펌핑 노이즈 차단.
    threshold = max(코인 중앙값 × 5%, MIN_CANDLE_VALUE_KRW) → 저유동성 코인 과다 제거 방지."""
    candle_value = df['volume'] * df['close']
    median_val = float(candle_value.median())
    threshold = max(median_val * 0.05, MIN_CANDLE_VALUE_KRW)
    invalid_mask = candle_value < threshold
    if invalid_mask.any():
        df.loc[invalid_mask, ['open', 'high', 'low', 'close', 'volume']] = np.nan
    return df

# ============================================================================
# 5. 고급 기술 지표 계산 함수
# ============================================================================
# 상관 제거 우선순위: 낮은 인덱스 = 높은 우선순위 (쌍에서 살아남음)
_CORR_DROP_PRIORITY = [
    'RSI', 'ADX_14', 'MTF_1D_RSI', 'MTF_1D_BB_Pos', 'RSI_Short',
    'MACD_Ratio', 'ATR_Ratio', 'BB_Width', 'BB_Position',
    'Volume_Surge', 'OFI_BuyPressure', 'OFI_CumDelta_20', 'OFI_Imbalance', 'SMA_Cross',
    'Close_to_SMA_Short_Ratio', 'Close_to_SMA_Long_Ratio', 'Volume_Trend',
    'ROC_5', 'ROC_10', 'Stochastic_K',
    'MACD_Signal_Ratio', 'MACD_Hist_Ratio', 'Stochastic_D',
    'OFI_DeltaProxy', 'OFI_CumDelta_10',
    'HMM_Bear', 'HMM_Sideways', 'HMM_Bull', 'Macro_Trend_Up',
]

_CORR_EXCLUDE_SET = {
    # 원시 OHLCV / 중간 계산값 (ML EXCLUDE_COLS와 동일)
    'open', 'high', 'low', 'close', 'volume',
    'SMA_Short', 'SMA_Long', 'Volume_SMA', 'HMM_State',
    'MACD', 'MACD_Signal', 'MACD_Hist', 'BB_Upper', 'BB_Lower', 'ATR',
    'DayOfWeek', 'Hour', 'Month',
    # 모델 후보 피처 전체 보호 — 코인별 상관 제거로 인한 피처 불일치 방지
    'Close_to_SMA_Short_Ratio', 'Close_to_SMA_Long_Ratio', 'SMA_Cross',
    'RSI', 'MACD_Ratio', 'MACD_Signal_Ratio', 'MACD_Hist_Ratio',
    'BB_Width', 'BB_Position',
    'ATR_Ratio',
    'Volume_Surge', 'Volume_Trend',
    'Stochastic_K', 'Stochastic_D', 'ROC_5', 'ROC_10',
    'OFI_BuyPressure', 'OFI_DeltaProxy', 'OFI_CumDelta_10', 'OFI_CumDelta_20', 'OFI_Imbalance',
    'OFI_CumDelta_5', 'OFI_BuyPressure_MA', 'CMF_20', 'MFI_14',
    'Macro_Trend_Up',
    'HMM_Bear', 'HMM_Sideways', 'HMM_Bull',
    'ATR_Pctile', 'ADX_14', 'RSI_Short',
    'MTF_4H_RSI', 'MTF_4H_Trend', 'MTF_4H_BB_Pos',
    'MTF_1D_RSI', 'MTF_1D_BB_Pos',
}


def _drop_correlated_features(df: pd.DataFrame, threshold: float = 0.85) -> pd.DataFrame:
    """|R| > threshold인 피처 쌍에서 우선순위 낮은 피처를 drop."""
    feat_cols = [
        c for c in df.select_dtypes(include='number').columns
        if c != 'Target' and c not in _CORR_EXCLUDE_SET
    ]
    if len(feat_cols) < 2:
        return df
    corr = df[feat_cols].corr().abs()
    to_drop: set = set()
    for i, c1 in enumerate(feat_cols):
        for c2 in feat_cols[i + 1:]:
            if c1 in to_drop or c2 in to_drop:
                continue
            if corr.loc[c1, c2] > threshold:
                p1 = _CORR_DROP_PRIORITY.index(c1) if c1 in _CORR_DROP_PRIORITY else len(_CORR_DROP_PRIORITY)
                p2 = _CORR_DROP_PRIORITY.index(c2) if c2 in _CORR_DROP_PRIORITY else len(_CORR_DROP_PRIORITY)
                drop_col = c2 if p1 <= p2 else c1
                keep_col = c1 if drop_col == c2 else c2
                to_drop.add(drop_col)
                logger.info(f"  고상관 제거: {drop_col} (|R|={corr.loc[c1, c2]:.3f}, 유지: {keep_col})")
    if to_drop:
        df = df.drop(columns=list(to_drop))
        logger.warning(f"⚠️ 고상관 피처 {len(to_drop)}개 제거 완료: {sorted(to_drop)}")
    return df


def _calculate_walk_forward_hmm(df: pd.DataFrame, n_states: int = 3,
                                 train_window: int = None, step: int = None) -> np.ndarray:
    """
    Look-ahead bias 방지를 위한 Walk-Forward 방식의 HMM 국면 산출 
    (💥수학적 붕괴 패치, 수렴 최적화 및 찌꺼기 중복 코드 완벽 제거)
    """
    if train_window is None:
        train_window = _cfg.HMM_REGIME.get("lookback", 1440)
    if step is None:
        step = _cfg.HMM_REGIME.get("wfo_step") or max(60, train_window // 4)

    n = len(df)
    states = np.full(n, 1.0)  # 기본값: 1 (횡보)

    _orig_train_window = train_window
    _orig_step = step
    _adaptive_applied = False

    if n <= train_window + step:
        adaptive_train = max(n // 3, 100)
        adaptive_step  = max(adaptive_train // 6, 20)
        if n <= adaptive_train + adaptive_step:
            logger.warning(
                f"HMM WFO 스킵: 데이터({n}행) 최소 요건({adaptive_train + adaptive_step}행) 미달. "
                f"전체 국면을 횡보(1)로 유지합니다."
            )
            return states
        train_window = adaptive_train
        step = adaptive_step
        _adaptive_applied = True

    close = df['close'].values.astype(float)
    # close[-1]은 미완성 현재 캔들 — 포함 시 매 틱마다 캐시 무효화됨
    _cache_key = (n, int(train_window), hash(close[-11:-1].tobytes()))
    _now = time.time()
    if _cache_key in _HMM_WFO_CACHE:
        _cached, _exp = _HMM_WFO_CACHE[_cache_key]
        if _now < _exp:
            return _cached.copy()

    # 캐시 미스 시에만 어댑티브 축소 로그 출력
    if _adaptive_applied:
        logger.info(
            f"HMM WFO 어댑티브: 데이터({n}행) 부족 → "
            f"train_window {_orig_train_window}→{train_window}, step {_orig_step}→{step}"
        )
    log_ret = np.diff(np.log(np.maximum(close, 1e-10)))
    log_ret = np.insert(log_ret, 0, 0.0)
    vol20 = pd.Series(log_ret).rolling(20, min_periods=1).std().fillna(0).values
    
    X_all = np.column_stack([log_ret, vol20])
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)

    for start_idx in range(train_window, n, step):
        end_idx = min(start_idx + step, n)
        train_start = start_idx - train_window
        X_train_raw = X_all[train_start:start_idx]
        X_predict_raw = X_all[start_idx:end_idx]
        # per-window RobustScaler: 이상치에 강건하고 Look-ahead bias 없음
        _scaler = RobustScaler()
        X_train = _scaler.fit_transform(X_train_raw)
        X_predict = _scaler.transform(X_predict_raw)

        try:
            _hmmlearn_log = logging.getLogger('hmmlearn')

            def _fit_hmm(n_comp, X_tr):
                m = GaussianHMM(
                    n_components=n_comp,
                    covariance_type="diag",
                    n_iter=500,
                    tol=1e-3,
                    random_state=42
                )
                _prev_level = _hmmlearn_log.level
                _hmmlearn_log.setLevel(logging.ERROR)
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=UserWarning)
                        m.fit(X_tr)
                finally:
                    _hmmlearn_log.setLevel(_prev_level)
                return m

            model = _fit_hmm(n_states, X_train)
            # 수렴 실패 시 n_states=2 폴백
            if not model.monitor_.converged and n_states > 2:
                model = _fit_hmm(2, X_train)
                if not model.monitor_.converged:
                    states[start_idx:end_idx] = 1.0
                    continue
                # 2-state 결과를 3-state 공간으로 매핑 (낮은쪽=Bear, 높은쪽=Bull, Sideways 미배정)
                pred_states = model.predict(X_predict)
                train_states = model.predict(X_train)
                means_2 = np.array([
                    X_train_raw[train_states == s, 0].mean() if (train_states == s).any() else 0.0
                    for s in range(2)
                ])
                low_s, high_s = int(np.argmin(means_2)), int(np.argmax(means_2))
                map_2to3 = {low_s: 0, high_s: 2}  # Bear=0, Bull=2, Sideways=1(default)
                states[start_idx:end_idx] = np.array([map_2to3.get(s, 1) for s in pred_states], dtype=float)
                continue

            pred_states = model.predict(X_predict)
            train_states = model.predict(X_train)

            # 상태를 재정렬하기 위해 원본(X_all)의 로그 수익률을 참조
            state_means = np.array([
                X_train_raw[train_states == s, 0].mean() if (train_states == s).any() else 0.0
                for s in range(n_states)
            ])
            sorted_states = np.argsort(state_means)
            state_map = {old: new for new, old in enumerate(sorted_states)}
            states[start_idx:end_idx] = np.vectorize(state_map.get)(pred_states)

        except Exception:
            pass  # 스케일링 후 에러 발생 시 조용히 횡보(1) 유지

    _HMM_WFO_CACHE[_cache_key] = (states.copy(), _now + _HMM_WFO_CACHE_TTL)
    return states


def add_technical_indicators(
    df: pd.DataFrame,
    optimize_memory: bool = True,
    include_advanced: bool = True
) -> Tuple[Optional[pd.DataFrame], FeatureEngineeringStats]:
    """
    실전 머신러닝 학습을 위해 고도화된 기술적 지표, 시간 피처를 추가합니다.
    
    Args:
        df (pd.DataFrame): OHLCV 데이터 (timestamp 컬럼이 있어야 함)
        optimize_memory (bool): float64를 float32로 변환 여부 (기본값: True)
        include_advanced (bool): 고급 지표 포함 여부 (기본값: True)
        
    Returns:
        Tuple[Optional[pd.DataFrame], FeatureEngineeringStats]: (처리된 df, 통계)
    """
    start_time = time.time()
    stats = FeatureEngineeringStats()
    
    try:
        if df is None or len(df) == 0:
            logger.warning("빈 데이터프레임 입력됨")
            return None, stats
            
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"필수 컬럼 누락: {required_cols}")
            return None, stats
        
        df = df.copy()
        # pyupbit get_ohlcv() → DatetimeIndex, 'timestamp' 컬럼 없음 → 자동 변환
        if 'timestamp' not in df.columns and isinstance(df.index, pd.DatetimeIndex):
            df.insert(0, 'timestamp', df.index)
            df = df.reset_index(drop=True)
        stats.input_rows = len(df)

        # ✅ [누수 수정] timestamp 기반 시계열 처리
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)

        # 거래대금 미달 캔들 NaN 마스킹 (좀비 코인 펌핑 노이즈 원천 차단)
        df = _apply_volume_filter(df)

        # 1. 시계열 패턴 피처
        try:
            if 'timestamp' in df.columns:
                df['DayOfWeek'] = df['timestamp'].dt.dayofweek
                df['Hour'] = df['timestamp'].dt.hour
                df['Month'] = df['timestamp'].dt.month
            else:
                logger.warning("timestamp 컬럼 없어 시계열 피처 생성 불가")
            stats.features_added.extend(['DayOfWeek', 'Hour', 'Month'])
        except Exception as e:
            logger.warning(f"시계열 피처 생성 실패: {e}")
        
        # 2. 가격 정규화 (이격도)
        df['SMA_Short'] = df['close'].rolling(window=SMA_SHORT).mean()
        df['SMA_Long'] = df['close'].rolling(window=SMA_LONG).mean()
        df['Close_to_SMA_Short_Ratio'] = (df['close'] / df['SMA_Short']) - 1
        df['Close_to_SMA_Long_Ratio'] = (df['close'] / df['SMA_Long']) - 1
        df['SMA_Cross'] = (df['SMA_Short'] > df['SMA_Long']).astype(int)
        stats.features_added.extend(['SMA_Short', 'SMA_Long', 'Close_to_SMA_Short_Ratio', 
                                     'Close_to_SMA_Long_Ratio', 'SMA_Cross'])
        
        # 3. RSI (Wilder 평활화: alpha=1/period, adjust=False — 표준 거래소 RSI와 동일)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/RSI_PERIOD, min_periods=RSI_PERIOD, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/RSI_PERIOD, min_periods=RSI_PERIOD, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df['RSI'] = 100 - (100 / (1 + rs))
        stats.features_added.append('RSI')

        # RSI_Short (7봉: 표준 RSI-14보다 빠른 반응, 단기 모멘텀 전환 포착)
        avg_gain_s = gain.ewm(alpha=1/7, min_periods=7, adjust=False).mean()
        avg_loss_s = loss.ewm(alpha=1/7, min_periods=7, adjust=False).mean()
        rs_s = avg_gain_s / avg_loss_s.replace(0, np.nan)
        df['RSI_Short'] = (100 - (100 / (1 + rs_s))).clip(0, 100)
        stats.features_added.append('RSI_Short')
        
        # 4. MACD (절대값 → 가격 대비 비율로 정규화, 가격 스케일 무관하게 만듦)
        exp1 = df['close'].ewm(span=MACD_FAST, adjust=False).mean()
        exp2 = df['close'].ewm(span=MACD_SLOW, adjust=False).mean()
        macd = exp1 - exp2
        macd_signal = macd.ewm(span=MACD_SIGNAL, adjust=False).mean()
        df['MACD_Ratio'] = macd / df['close']
        df['MACD_Signal_Ratio'] = macd_signal / df['close']
        df['MACD_Hist_Ratio'] = (macd - macd_signal) / df['close']
        stats.features_added.extend(['MACD_Ratio', 'MACD_Signal_Ratio', 'MACD_Hist_Ratio'])

        # 5. 볼린저 밴드 (BB_Upper/BB_Lower는 절대 가격이므로 컬럼 저장 제외)
        bb_middle = df['close'].rolling(window=BB_PERIOD).mean()
        bb_std = df['close'].rolling(window=BB_PERIOD).std()
        bb_upper = bb_middle + (BB_STD_DEV * bb_std)
        bb_lower = bb_middle - (BB_STD_DEV * bb_std)
        df['BB_Width'] = (4 * bb_std) / bb_middle.replace(0, np.nan)
        df['BB_Position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)
        stats.features_added.extend(['BB_Width', 'BB_Position'])

        # 6. ATR (절대값 → 가격 대비 비율로 정규화)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = np.maximum(np.maximum(high_low, high_close), low_close)
        atr = pd.Series(true_range, index=df.index).ewm(alpha=1/ATR_PERIOD, min_periods=ATR_PERIOD, adjust=False).mean()
        df['ATR_Ratio'] = atr / df['close']
        stats.features_added.append('ATR_Ratio')

        # ADX-14: 추세 강도 (0=완전 횡보, 1=강한 추세) — 횡보장 진입 노이즈 구분 핵심 지표
        try:
            _up  = df['high'].diff()
            _dn  = -df['low'].diff()
            _pdm = _up.where((_up > _dn) & (_up > 0), 0.0)
            _mdm = _dn.where((_dn > _up) & (_dn > 0), 0.0)
            _pdi = 100 * _pdm.ewm(alpha=1/14, adjust=False).mean() / atr.replace(0, np.nan)
            _mdi = 100 * _mdm.ewm(alpha=1/14, adjust=False).mean() / atr.replace(0, np.nan)
            _dx  = ((_pdi - _mdi).abs() / (_pdi + _mdi).replace(0, np.nan)) * 100
            df['ADX_14'] = _dx.ewm(alpha=1/14, adjust=False).mean().clip(0, 100) / 100
            stats.features_added.append('ADX_14')
        except Exception as e:
            logger.warning(f"ADX 피처 생성 실패: {e}")
        
        # 7. 거래량 지표
        df['Volume_SMA'] = df['volume'].rolling(window=VOLUME_PERIOD).mean()
        df['Volume_Surge'] = df['volume'] / df['Volume_SMA'].replace(0, np.nan)
        df['Volume_Trend'] = (df['Volume_SMA'].diff() > 0).astype(int)
        stats.features_added.extend(['Volume_SMA', 'Volume_Surge', 'Volume_Trend'])
        
        # 8. 고급 지표 (옵션)
        if include_advanced:
            # Stochastic 지표 (표준: 14봉 내 저가/고가 범위 기준)
            stoch_low  = df['low'].rolling(window=14).min()
            stoch_high = df['high'].rolling(window=14).max()
            df['Stochastic_K'] = 100 * (df['close'] - stoch_low) / (stoch_high - stoch_low).replace(0, np.nan)
            df['Stochastic_D'] = df['Stochastic_K'].rolling(window=3).mean()
            stats.features_added.extend(['Stochastic_K', 'Stochastic_D'])
            
            # 가격 변화율 (ROC)
            df['ROC_5'] = ((df['close'] / df['close'].shift(5)) - 1) * 100
            df['ROC_10'] = ((df['close'] / df['close'].shift(10)) - 1) * 100
            stats.features_added.extend(['ROC_5', 'ROC_10'])
        
        # 8-B. Order Flow Imbalance (OFI) 프록시 피처 — OHLCV 근사
        # (close-low)/(high-low): 단일봉 내 핀바 구조·매수압력 포착 (평활화 없이 즉각 반응)
        # OFI_DeltaProxy/CumDelta_10은 Volume_SMA로 정규화 → 코인 간 거래량 스케일 차이 제거
        try:
            _hl = (df['high'] - df['low']).replace(0, np.nan)
            _buy_pres  = (df['close'] - df['low'])  / _hl   # 0=완전 하락봉, 1=완전 상승봉
            _delta_dir = (df['close'] - df['open'])  / _hl  # 시가 대비 이동 방향 (양수=순매수)
            _vol_norm  = df['Volume_SMA'].replace(0, np.nan)

            df['OFI_BuyPressure']  = _buy_pres.clip(0, 1)
            df['OFI_DeltaProxy']   = (_delta_dir * df['volume'] / _vol_norm).fillna(0)
            df['OFI_CumDelta_5']   = df['OFI_DeltaProxy'].rolling(5).sum()
            df['OFI_CumDelta_10']  = df['OFI_DeltaProxy'].rolling(10).sum()
            df['OFI_CumDelta_20']  = df['OFI_DeltaProxy'].rolling(20).sum()
            df['OFI_Imbalance']    = df['OFI_BuyPressure'].rolling(5).mean() * 2 - 1
            df['OFI_BuyPressure_MA'] = _buy_pres.rolling(10).mean()
            # Chaikin Money Flow: 거래량 가중 가격 위치 (OFI_Imbalance의 거래량 강화판)
            _cmf_num = ((2 * df['close'] - df['high'] - df['low']) / _hl * df['volume']).fillna(0)
            df['CMF_20'] = (_cmf_num.rolling(20).sum() /
                            df['volume'].rolling(20).sum().replace(0, np.nan)).fillna(0)
            # Money Flow Index: 거래량 가중 RSI (순매수 지속성 측정)
            _tp   = (df['high'] + df['low'] + df['close']) / 3
            _mf   = _tp * df['volume']
            _tp_s = _tp.shift(1)
            _pos  = _mf.where(_tp > _tp_s, 0.0).rolling(14).sum()
            _neg  = _mf.where(_tp < _tp_s, 0.0).rolling(14).sum()
            df['MFI_14'] = (100 - 100 / (1 + _pos / _neg.replace(0, np.nan))).fillna(50)
            stats.features_added.extend([
                'OFI_BuyPressure', 'OFI_DeltaProxy', 'OFI_CumDelta_5', 'OFI_CumDelta_10',
                'OFI_CumDelta_20', 'OFI_Imbalance', 'OFI_BuyPressure_MA', 'CMF_20', 'MFI_14',
            ])
        except Exception as e:
            logger.warning(f"OFI 프록시 피처 생성 실패: {e}")

        # 8-C. ATR 백분위 피처 — 현재 변동성이 역사적으로 높은지/낮은지
        # ATR_Pctile ≈ 0: 역사적 저변동성 구간, ≈ 1: 역사적 고변동성 구간
        try:
            if 'ATR_Ratio' in df.columns:
                df['ATR_Pctile'] = (
                    df['ATR_Ratio'].rolling(100, min_periods=20).rank(pct=True).fillna(0.5)
                )
                stats.features_added.append('ATR_Pctile')
        except Exception as e:
            logger.warning(f"ATR_Pctile 피처 생성 실패: {e}")

        # 9. MTFA: 일봉 기준 거시적 추세 피처 (Macro Trend)
        # 15분봉 데이터를 일봉으로 리샘플링 → 현재가가 일봉 SMA 위에 있는지 판별
        # trade_bot.py에서 count=480(≈5일)으로 호출하므로 추론 시에도 의미 있는 SMA 확보
        try:
            if 'timestamp' in df.columns:
                close_series = pd.Series(df['close'].values,
                                         index=pd.to_datetime(df['timestamp']))
            elif isinstance(df.index, pd.DatetimeIndex):
                close_series = df['close']
            else:
                close_series = None

            if close_series is not None:
                # .shift(1): 당일 종가(미래)가 아닌 전일 종가만 당일 장중에 사용 → Look-ahead Bias 제거
                daily_close = close_series.resample('1D').last().dropna().shift(1)
                if len(daily_close) >= 2:
                    sma_window = min(20, len(daily_close))
                    daily_sma = daily_close.rolling(window=sma_window, min_periods=2).mean()
                    # 15분봉 인덱스로 forward-fill 매핑 (당일 종가 → 다음날 봉에 반영)
                    expanded_sma = daily_sma.reindex(close_series.index, method='ffill')
                    df['Macro_Trend_Up'] = (close_series.values > expanded_sma.values).astype(float)

                    # MTF_1D 확장: 일봉 RSI + BB 위치 (이진 Macro_Trend_Up을 연속값으로 보완)
                    if len(daily_close) >= 14:
                        try:
                            _dd   = daily_close.diff()
                            _dg   = _dd.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
                            _dl   = (-_dd.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
                            _drsi = (100 - 100 / (1 + _dg / _dl.replace(0, np.nan))).shift(1) / 100
                            _ds20 = daily_close.rolling(20).mean().shift(1)
                            _dstd = daily_close.rolling(20).std().shift(1).replace(0, np.nan)
                            _dbbp = ((daily_close.shift(1) - _ds20) / (2 * _dstd)).clip(-1.5, 1.5)
                            df['MTF_1D_RSI']    = _drsi.reindex(close_series.index, method='ffill').fillna(0.5).values
                            df['MTF_1D_BB_Pos'] = _dbbp.reindex(close_series.index, method='ffill').fillna(0.0).values
                            stats.features_added.extend(['MTF_1D_RSI', 'MTF_1D_BB_Pos'])
                        except Exception as _e:
                            logger.warning(f"MTF_1D 피처 생성 실패: {_e}")
                else:
                    df['Macro_Trend_Up'] = 0.0
            else:
                df['Macro_Trend_Up'] = 0.0

            df['Macro_Trend_Up'] = df['Macro_Trend_Up'].fillna(0.0)
        except Exception as e:
            logger.warning(f"MTFA 피처 생성 실패: {e}")
            df['Macro_Trend_Up'] = 0.0
        finally:
            stats.features_added.append('Macro_Trend_Up')

        # 9-B. MTF 4H 피처 — 4시간봉 RSI·추세·BB위치 (Look-ahead bias 방지: shift(1) 적용)
        try:
            if 'timestamp' in df.columns:
                _ts_idx = pd.to_datetime(df['timestamp'])
            elif isinstance(df.index, pd.DatetimeIndex):
                _ts_idx = df.index
            else:
                _ts_idx = None

            if _ts_idx is not None:
                _close_s = pd.Series(df['close'].values, index=_ts_idx)
                _h4 = _close_s.resample('4h').last().dropna()
                if len(_h4) >= 14:
                    # 4H RSI-14 (Wilder 방식)
                    _d    = _h4.diff()
                    _gain = _d.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
                    _loss = (-_d.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
                    _h4_rsi = (100 - 100 / (1 + _gain / _loss.replace(0, np.nan))).shift(1) / 100

                    # 4H SMA10 추세
                    _h4_sma = _h4.rolling(10).mean().shift(1)
                    _h4_trend = (_h4.shift(1) > _h4_sma).astype(float)

                    # 4H BB위치 (SMA20 기준, -1.5~+1.5 클리핑)
                    _h4_s20  = _h4.rolling(20).mean().shift(1)
                    _h4_std  = _h4.rolling(20).std().shift(1).replace(0, np.nan)
                    _h4_bbpos = ((_h4.shift(1) - _h4_s20) / (2 * _h4_std)).clip(-1.5, 1.5)

                    df['MTF_4H_RSI']    = _h4_rsi.reindex(_ts_idx, method='ffill').fillna(0.5).values
                    df['MTF_4H_Trend']  = _h4_trend.reindex(_ts_idx, method='ffill').fillna(0.0).values
                    df['MTF_4H_BB_Pos'] = _h4_bbpos.reindex(_ts_idx, method='ffill').fillna(0.0).values
                    stats.features_added.extend(['MTF_4H_RSI', 'MTF_4H_Trend', 'MTF_4H_BB_Pos'])
        except Exception as e:
            logger.warning(f"MTF 4H 피처 생성 실패: {e}")

        # 10. HMM 국면 피처 추가 (Walk-Forward WFO — dropna 이전에 배치하여 인덱스 정합성 보장)
        # log_return + rolling_vol_20 피처로 학습 (market_context.HMMRegimeDetector와 동일)
        # 원-핫 인코딩: XGBoost 트리가 연속값(0/1/2)을 서수 관계로 오해하는 것을 방지
        try:
            hmm_states = _calculate_walk_forward_hmm(
                df,
                n_states=_cfg.HMM_REGIME["n_states"],  # config 단일 소스 (기본 3)
            )
            df['HMM_State'] = hmm_states  # 원본 보존 (EXCLUDE_COLS에서 제외할 것)
            df['HMM_Bear']    = (hmm_states == 0).astype(float)
            df['HMM_Sideways']= (hmm_states == 1).astype(float)
            df['HMM_Bull']    = (hmm_states == 2).astype(float)
        except Exception as e:
            logger.error(f"HMM 피처 생성 실패: {e}")
            df['HMM_State']   = 1.0
            df['HMM_Bear']    = 0.0
            df['HMM_Sideways']= 1.0
            df['HMM_Bull']    = 0.0
        finally:
            stats.features_added.extend(['HMM_Bear', 'HMM_Sideways', 'HMM_Bull'])

        # 11. Target 라벨 생성 (Meta-Labeling + ATR 동적 Triple Barrier)
        # Primary Signal(SMA 골든크로스 + RSI)이 발생한 봉에만 라벨 생성 →
        # 횡보 노이즈 샘플을 학습 데이터에서 원천 배제 (de Prado Meta-Labeling)
        _coin_sym    = str(df['coin'].iloc[0]) if 'coin' in df.columns else ''
        _coin_cat    = _cfg.get_coin_category(_coin_sym) if _coin_sym else 'C'
        df['Target'] = triple_barrier_label_dynamic(
            df, window=TB_WINDOW, use_meta_labeling=True,
            enable_signal_c=(_coin_cat != 'A'),
        )
        stats.features_added.append('Target')

        # 11-b. 고상관 피처 제거 (|R| > 0.85 → 우선순위 낮은 피처 drop)
        df = _drop_correlated_features(df, threshold=0.85)

        # 12. 결측치 제거 (Target 컬럼 제외)
        initial_len = len(df)
        # 💥 핵심 패치 3: Target 라벨이 없다고(NaN) 해서 원본 시계열 캔들을 날려버리는 재앙 차단. 
        # 오직 보조지표(SMA 등) 계산 시 앞부분에 생기는 빈칸만 제거함.
        indicator_cols = [c for c in df.columns if c != 'Target']
        df = df.dropna(subset=indicator_cols)
        
        stats.removed_null_rows = initial_len - len(df)
        stats.output_rows = len(df)

        if stats.removed_null_rows > 0:
            logger.debug(f"결측치 제거: {stats.removed_null_rows}개 행 (초기 지표 Lookback 윈도우)")

        # 13. 메모리 최적화
        if optimize_memory:
            float64_cols = df.select_dtypes(include=['float64']).columns
            if len(float64_cols) > 0:
                df[float64_cols] = df[float64_cols].astype('float32')
                logger.debug(f"메모리 최적화: {len(float64_cols)}개 float64 컬럼 변환")
        
        stats.processing_time = time.time() - start_time
        return df, stats
        
    except Exception as e:
        logger.error(f"기술적 지표 추가 중 오류: {type(e).__name__} - {e}")
        stats.processing_time = time.time() - start_time
        return None, stats

def process_all_data(input_dir: str = DATA_DIR, output_dir: str = OUTPUT_DIR) -> Dict:
    """
    data 폴더의 모든 원본 CSV를 읽어 실전형 지표를 추가한 뒤 data_processed 폴더에 저장합니다.
    
    Args:
        input_dir (str): 입력 데이터 폴더명
        output_dir (str): 출력 데이터 폴더명
        
    Returns:
        Dict: {"success": int, "failed": int, "total_features": int, "total_time": float}
    """
    try:
        # 출력 폴더 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 원본 데이터 파일 목록 조회
        file_list = glob.glob(f'{input_dir}/*.csv')
        
        if not file_list:
            logger.warning(f"'{input_dir}' 폴더에 CSV 파일이 없습니다.")
            return {"success": 0, "failed": 0, "total_features": 0, "total_time": 0.0}
        
        logger.info(f"총 {len(file_list)}개의 데이터에 피처 엔지니어링을 시작합니다.")
        
        processed_count = 0
        failed_files = []
        total_features = 0
        total_processing_time = 0.0

        _min_median_krw = int(_cfg.LABELING.get("min_median_candle_krw", 50_000_000))

        def _process_one(args):
            idx, file_path = args
            file_name = os.path.basename(file_path)
            progress = f"[{idx}/{len(file_list)}]"
            try:
                df = pd.read_csv(file_path)
                # 코인 레벨 유동성 게이트: 중앙값 거래대금 < 50M KRW/봉 (≈50억/일) 코인 제외
                if 'volume' in df.columns and 'close' in df.columns:
                    _median_val = float((df['volume'] * df['close']).median())
                    if _median_val < _min_median_krw:
                        logger.info(
                            f"{progress} 유동성 부족 스킵: {file_name} "
                            f"(중앙값 {_median_val:,.0f} KRW/봉 < {_min_median_krw:,.0f})"
                        )
                        return file_name, None, "유동성 부족"
                processed_df, stats = add_technical_indicators(df, include_advanced=True)
                if processed_df is None or len(processed_df) == 0:
                    logger.warning(f"{progress} 처리 실패: {file_name}")
                    return file_name, None, None
                save_path = os.path.join(output_dir, f'processed_{file_name}')
                tmp_path = save_path + ".tmp"
                processed_df.to_csv(tmp_path)
                os.replace(tmp_path, save_path)
                logger.info(f"{progress} ✅ {file_name} → 저장 완료 ({stats})")
                return file_name, stats, None
            except Exception as e:
                logger.error(f"{progress} 파일 처리 실패 ({file_name}): {type(e).__name__} - {e}")
                return file_name, None, str(e)

        # HMM은 내부적으로 BLAS 멀티스레드를 사용하므로 ThreadPoolExecutor로 감싸면
        # BLAS 코어 경합이 발생해 오히려 느려짐 → 순차 처리
        futures = [_process_one(args) for args in enumerate(file_list, 1)]

        for file_name, stats, err in futures:
            if err == "유동성 부족":
                continue  # 정상 스킵 — 실패로 집계하지 않음
            if err is not None or stats is None:
                failed_files.append(file_name)
            else:
                processed_count += 1
                total_features += len(stats.features_added)
                total_processing_time += stats.processing_time
        
        # 최종 결과 요약
        logger.info(f"처리 완료: 성공 {processed_count}개, 실패 {len(failed_files)}개")
        if failed_files:
            logger.warning(f"실패 파일: {', '.join(failed_files)}")
        
        return {
            "success": processed_count,
            "failed": len(failed_files),
            "total_features": total_features,
            "total_time": total_processing_time
        }
        
    except Exception as e:
        logger.error(f"데이터 처리 중 오류: {type(e).__name__} - {e}")
        return {"success": 0, "failed": 0, "total_features": 0, "total_time": 0.0}

if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("🚀 피처 엔지니어링 파이프라인 시작")
    logger.info("=" * 80)
    
    result = process_all_data()
    
    if result["success"] > 0:
        logger.info(f"✨ {result['success']}개 파일 처리 완료!")
        logger.info(f"📊 추가된 총 피처 수: {result['total_features']}")
        logger.info(f"⏱️  총 처리 시간: {result['total_time']:.2f}초")