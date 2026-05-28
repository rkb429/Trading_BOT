# ============================================================================
# Coin AI Bot - 중앙 설정 파일(config.py)
# ============================================================================

import json
import os
import logging
import threading

# ============================================================================
# 1. 데이터 수집 설정
# ============================================================================

# 거래대금 기반 동적 코인풀 추출 → market_context.py로 이동
# import market_context 후 market_context.get_dynamic_target_coins() 사용할 것

# 수집할 시간대
TARGET_INTERVALS = [
    "days",          # 일봉
    "minute60",     # 1시간봉
    "minute15",     # 15분봉
]

# 수집 파라미터
DATA_COLLECTION = {
    # minute15: 70080개 ≒ 2년 / minute60: 35040개 ≒ 4년 / day: 35040개 ≒ 96년
    # 퀀트 최소 요건: 최소 1년 이상 상승/하락/횡보 사이클 포함 필요
    # MTF 앙상블: 15m 모델에 충분한 데이터 확보를 위해 인터벌별 count 분리
    "count": {
        "minute15": 70_080,   # 2년치 (4×365×24×4 = 140160의 절반; 수집 속도 균형)
        "minute60": 35_040,   # 4년치 (기존 유지)
        "days":     35_040,   # 96년치 (기존 유지, 실제 거래소 개장 이후만 반환)
    },
    "overwrite": False,
    "max_retries": 3,
    "api_rate_limit_delay": 0.5,
}

# ============================================================================
# 2. 코인별 슬리피지 설정
# ============================================================================

# 시장가 주문을 허용할 유동성 충분 코인 (호가창 두께 기준 — 스프레드 0.05% 이내)
LIQUID_COINS = {'BTC', 'ETH', 'XRP', 'SOL'}

# 코인별 예상 슬리피지율 (백테스트 및 실거래 P&L 계산 공통 사용)
# Model_A (대형): 0.10~0.15% / Model_B (중형): 0.20~0.30% / Model_C (소형): 0.40~0.50%
SLIPPAGE_BY_COIN = {
    # ── Model_A: 대형 효율 시장 ──────────────────────────────────────────────
    'BTC':     0.001,   # 0.10% - 호가 두께 최상
    'ETH':     0.0015,  # 0.15%
    # ── Model_B: 중형 알트 ───────────────────────────────────────────────────
    'XRP':     0.002,   # 0.20% - 업비트 거래대금 Top3 상주
    'SOL':     0.002,   # 0.20%
    'ADA':     0.002,   # 0.20% - 안정적 호가
    'DOGE':    0.002,   # 0.20% - 밈 대비 높은 유동성
    'SUI':     0.0025,  # 0.25% - L1 신흥, 거래대금 Top10 진입 빈번
    # ── Model_C: 소형 알트 알파 ─────────────────────────────────────────────
    'ONDO':    0.003,   # 0.30% - RWA 중형
    'TRUMP':   0.004,   # 0.40% - 내러티브 driven, 변동성 높음
    'VIRTUAL': 0.004,   # 0.40% - AI Agent, 중소형
    'PENGU':   0.004,   # 0.40% - 밈/NFT 소형
    'CPOOL':   0.005,   # 0.50% - DeFi 소형
    'CFG':     0.005,   # 0.50% - Centrifuge RWA 소형
    'SAHARA':  0.005,   # 0.50% - AI 소형
    'KITE':    0.005,   # 0.50%
    'SPK':     0.005,   # 0.50%
    'OPEN':    0.005,   # 0.50% - 백테스트 검증 편입
    # ── 기타 ─────────────────────────────────────────────────────────────────
    'IP':      0.005,   # 0.50% - 신규 상장 소형
}
SLIPPAGE_DEFAULT = 0.004  # 목록 외 소형 알트 기본값 0.40% (유동성 필터 통과 최소 기준)

# ============================================================================
# 3. 피처 엔지니어링 설정
# ============================================================================

# 기술 지표 파라미터
TECHNICAL_INDICATORS = {
    "SMA_SHORT": 10,
    "SMA_LONG": 20,
    "RSI_PERIOD": 14,
    "MACD_FAST": 12,
    "MACD_SLOW": 26,
    "MACD_SIGNAL": 9,
    "BB_PERIOD": 20,
    "BB_STD_DEV": 2,
    "VOLUME_PERIOD": 10,
    "ATR_PERIOD": 14,
    "CMO_PERIOD": 14,
    # FEE_THRESHOLD는 LABELING["fee_threshold"] 정의 후 아래에서 단일 참조로 주입
}

# 피처 엔지니어링 옵션
FEATURE_ENGINEERING = {
    "optimize_memory": True,    # float64를 float32로 변환
    "include_advanced": True,   # 고급 지표 포함 (Stochastic, ROC 등)
}

# ============================================================================
# 4. 디렉토리 설정
# ============================================================================

DIRECTORIES = {
    "data": "data",                         # 현물(업비트) 원본 데이터 (모든 인터벌 공용)
    "data_processed": "data_processed",     # 현물 처리된 데이터 (모든 인터벌 공용)
    "data_futures": "data_futures",         # 선물(바이낸스) 원본 데이터 — 모델 분리
    "data_futures_processed": "data_futures_processed",  # 선물 처리된 데이터
    "logs": "logs",
    "models": "models",                     # 현물 모델 저장 (레거시 — 카테고리 미분리 시 사용)
    "models_futures": "models_futures",     # 선물 전용 모델 저장
    "models_A": "models_A",                 # Model_A 60m: BTC, ETH (대형 효율 시장)
    "models_B": "models_B",                 # Model_B 60m: XRP, SOL, ADA, DOGE (중형 알트)
    "models_C": "models_C",                 # Model_C 60m: 소형 알트 핵심 알파 구간
    "models_A_15m": "models_A_15m",         # Model_A 15m: BTC, ETH (MTF 타이밍 모델)
    "models_B_15m": "models_B_15m",         # Model_B 15m: 중형 알트 (MTF 타이밍 모델)
    "models_C_15m": "models_C_15m",         # Model_C 15m: 소형 알트 (MTF 타이밍 모델)
}

# ============================================================================
# 5. 로깅 설정
# ============================================================================

LOGGING = {
    "level": "INFO",  # 로깅 레벨 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    "format": "[%(asctime)s] %(levelname)s: %(message)s",
    "datefmt": "%Y-%m-%d %H:%M:%S",
}

# ============================================================================
# 6. 데이터 검증 설정
# ============================================================================

DATA_VALIDATION = {
    "min_rows": None,        # HMM_REGIME.lookback 정의 후 아래에서 단일 참조로 주입
    "max_null_ratio": 0.1,  # 최대 결측치 비율 (10%)
}

# ============================================================================
# 7. 바이낸스 선물 컨텍스트 (펀딩비 + 롱숏 비율)
# ============================================================================

BINANCE_CONTEXT = {
    "enabled": True,
    "cache_ttl": 300,                  # 캐시 유효 시간 (초)
    "funding_rate_threshold": 0.001,   # 0.10% 이상 → 롱 과열로 진입 차단
    "ls_ratio_overbought": 1.8,        # 롱숏 비율 1.8 이상 → 과매수
    "funding_rate_oversold": -0.0005,  # -0.05% 미만 → 숏 패닉 (반등 기대)
    # funding_risk_threshold, trailing_stop_callback_rate, exchange_info_refresh_interval
    # 은 리스크 관리 책임이므로 FUTURES_RISK 섹션(23번)에서만 관리
}

# ============================================================================
# 8. 김치 프리미엄 추적
# ============================================================================

KIMCHI_PREMIUM = {
    "enabled": True,
    "cache_ttl": 60,                   # 캐시 유효 시간 (초)
    "surge_threshold": 0.002,          # 가속도 0.20% 이상 → 수급 폭발
    "history_size": 5,                 # 가속도 계산용 이력 크기
    "threshold_boost": 0.02,           # 가속 감지 시 진입 임계값 완화폭
}

# ============================================================================
# 9. 공포-탐욕 지수 연동
# ============================================================================

FEAR_GREED = {
    "enabled": True,
    "cache_ttl": 3600,                     # 하루 1회 갱신이므로 1시간 캐시
    "extreme_greed": 80,                   # 이 이상: 트레일링 스탑 × 0.5
    "greed": 65,                           # 이 이상: 트레일링 스탑 × 0.75
    "trailing_stop_extreme_ratio": 0.5,
    "trailing_stop_greed_ratio": 0.75,
}

# ============================================================================
# 10. HMM 기반 시장 국면 감지
# ============================================================================

HMM_REGIME = {
    "enabled": True,
    "n_states": 3,                     # bull / sideways / bear
    # 3국면 통계 분리 최소 요건: 국면당 30개 독립 관측 × 7일 지속 × 96봉/일 = 2016봉
    # 30일(2880봉)으로 여유 확보 — 이전 1440=15일은 통계적으로 국면 분리 불충분
    "lookback": 2880,                  # WFO 훈련 윈도우 (15분봉 2880개 ≒ 30일)
    "wfo_step": 480,                   # WFO 슬라이딩 스텝 (5일마다 재학습)
    "retrain_interval": 3600,          # 추론용 HMM 재학습 주기 (초)
    "threshold_bear": 0.85,            # 약세 국면 진입 임계값 (사실상 차단 수준)
    "threshold_sideways": 0.0,         # 횡보 국면 임계값: 0.0 = 모델 base threshold 그대로 사용
    "threshold_bull": 0.65,            # 강세 국면 진입 임계값 (= 기본값)
    "position_multiplier_bear": 0.3,   # 약세 국면 포지션 축소
    "position_multiplier_sideways": 0.7,  # 횡보 국면 포지션 축소 (bear보다 완화)
}
# DATA_VALIDATION과 단일 참조 — lookback 변경 시 자동 반영
DATA_VALIDATION["min_rows"] = HMM_REGIME["lookback"]

# ============================================================================
# 11. 메타 라벨링 / Triple Barrier 설정 (feature_engineering.py)
# ============================================================================

LABELING = {
    # [비용 구조 현실화]
    # 업비트 실제 수수료 0.05% 적용 (기존 0.10%는 과대 추정)
    # 슬리피지 왕복 0.20%: SLIPPAGE_DEFAULT(0.15%×2=0.30%)보다 타이트하지만
    # 유동성 게이트(50M KRW 중앙값) 통과 코인 대상이므로 0.20% 현실적
    "fee_threshold":       0.0005,  # 편도 수수료 0.05% (업비트 실제)
    "slippage_roundtrip":  0.002,   # 왕복 슬리피지 0.20% (편도 0.10% × 2)
    "forward_bars":        16,      # 10→16: 16시간 추세 발전 여유, 노이즈 라벨 감소

    # 손익 구조: TP×2.0 / SL×0.75 → 손익비 2.67:1
    # 실효 손익비(비용 반영): ATR=2% 기준 net_win(3.7%) / net_loss(1.8%) ≈ 2.06 → 손익분기 33%
    # holdout 35% > 33% → 기댓값 흑자 전환 (기존 TP×1.5 → 손익분기 40% 초과로 기댓값 음수)
    "atr_tp_mult":         2.0,
    "atr_sl_mult":         0.75,
    
    "primary_signal_zone":   7,
    "min_candle_value_krw":  5_000_000,  # 개별 캔들 최소 거래대금 (5M KRW)
    "min_median_candle_krw": 50_000_000, # 코인 전체 중앙값 기준 유동성 게이트 (≒50억/일 기준: 50M×96봉)
    "min_atr_ratio":         0.015,
}

# TECHNICAL_INDICATORS와 단일 참조 — 두 값이 항상 동기화됨
TECHNICAL_INDICATORS["FEE_THRESHOLD"] = LABELING["fee_threshold"]

# ============================================================================
# 12. 유동성 로테이션 스캐너
# ============================================================================

LIQUIDITY_SCANNER = {
    "enabled": True,
    "check_interval": 300,             # 스캔 주기 (초)
    "top_n": 10,                       # 모니터링 상위 N개 종목
    "surge_rank_jump": 5,              # 이 이상 순위 상승 시 후보 등록
    "threshold_boost": 0.02,           # 현재 티커가 급등 후보일 때 진입 임계값 완화폭
}

# ============================================================================
# 13. 분할 진입/청산 (기본 OFF — 사용자 선택)
# ============================================================================

SPLIT_ORDER = {
    "enabled": False,                  # True로 변경하면 활성화 (ATR 분할 + Depth-TWAP 함께 활성화)
    "n_splits": 3,                     # 분할 횟수
    "split_delay_sec": 2,              # 분할 간격 (초)
    "atr_threshold": 0.015,            # ATR > 1.5%일 때만 ATR 기반 분할 적용
}

# ============================================================================
# 14. 켈리 배팅 비율 (backtest.py / trade_bot.py 공통)
# ============================================================================

# 풀 켈리 추정값: 승률 60% × 손익비 1.33 기준
#   kelly_f = (p × b - (1-p)) / b = (0.60 × 1.33 - 0.40) / 1.33 ≈ 0.30
# backtest.MLStrategy.trade_size_pct = KELLY_FRACTION × 0.5 = 15% (Half-Kelly)
KELLY_FRACTION = 0.30

# ============================================================================
# 15. 멀티봇 오케스트레이터 (orchestrator.py)
# ============================================================================

ORCHESTRATOR = {
    "enabled": False,                  # True: orchestrator.py 실행 시 활성화
    "max_concurrent_bots": 5,          # 동시 운용 종목 수 (자본금에 맞게 조절)
    "rescan_interval": 300,            # 유동성 순위 재평가 주기 (초)
    # ── 포트폴리오 상관계수 리스크 관리 ─────────────────────────────────────────
    "corr_lookback_bars": 96,          # 상관계수 계산 구간 (96 × 15분 = 24시간)
    "corr_high_threshold": 0.85,       # 이 이상: 봇 1개로 강제 축소 (사실상 동일 종목 노출)
    "corr_medium_threshold": 0.70,     # 이 이상: 봇 최대 2개로 제한
    # ── 시장 과열 현금 보유 ──────────────────────────────────────────────────────
    "overheat_fg_threshold": FEAR_GREED["extreme_greed"],  # FEAR_GREED.extreme_greed 단일 참조
}

# ============================================================================
# 16. 백테스트 설정 (backtest.py) - 라벨링과 동기화 필수
# ============================================================================
BACKTEST = {
    "initial_cash": 5_000_000,
    "commission": LABELING["fee_threshold"],  # 반드시 위 LABELING 수수료와 동기화
    "stress_slippage": 0.008,                 # 0.010 에서 0.008로 현실화
    "time_stop_bars": LABELING["forward_bars"],
    "trailing_stop_pct": 0.03,
    "ai_exit_prob": 0.15,                     # Optuna threshold(0.25~0.80) 보다 반드시 낮아야 함 — 0.45는 진입 즉시 청산 유발
    "hmm_bear_multiplier": 0.3,        # HMM_REGIME.position_multiplier_bear와 동기화 필수
    "short_max_up_prob": 0.20,
    "atr_risk_pct": 0.02,
    "time_stop_profit_exempt_pct": 0.005,
}

# ============================================================================
# 17. 모델 파일 관리 (machine_learning.py / auto_retrain.py)
# ============================================================================

MODEL_MANAGEMENT = {
    "retention_count": 5,             # models/ 폴더에 유지할 최대 파일 세트 수
                                      # (세트 = .pkl + config_.json + importance_.csv 3파일)
    # ── OOS 성능 컷오프 ───────────────────────────────────────────────────────
    # Triple Barrier (TP×2.0 / SL×0.75) 기준 이론 손익분기 = 27.3%
    # 실효 손익비(비용 반영): ATR=2% 기준 net_win(3.7%) / net_loss(1.8%) ≈ 2.06 → 손익분기 33%
    # 게이트 = 비용 조정 손익분기 33% + 마진 9%p = 42%
    "min_precision": 0.42,            # OOS 정밀도 미달 시 모델 저장 거부 (TP×2.0/SL×0.75 비용반영 손익분기 33% + 버퍼)
    "optuna_threshold_min": 0.20,     # Optuna threshold 탐색 하한 — min_precision보다 낮아야 calibrated 확률 분포 커버 가능
    "min_recall": 0.05,               # OOS 재현율 미달 시 거부 (per-interval 필터가 과매매 방지하므로 완화)
    "min_signals": 30,                # OOS 매수 신호 최소 횟수 (통계적 신뢰성)
    "strict_oos_gate": True,          # True=ValueError 차단, False=경고 후 저장 강행
    "regime_min_precision": 0.42,     # 국면별 정밀도 최소값 — skip=True 모델이 해당 국면에서만 재활성화
    # MTF 앙상블: minute60(추세) + minute15(타이밍) 두 인터벌 모두 학습
    # minute15 2년치(70,080봉)로 소형 알트 데이터 부족 문제 해결
    "train_intervals": ["minute60", "minute15"],
    "train_interval": "minute60",     # 하위 호환용 — trade_bot 레거시 참조 시 사용
    "min_train_rows": 2000,           # 파일당 최소 행 수 미달 시 학습 스킵 (≈20일, 신규 상장 노이즈 코인 제외)
    # ── 재학습 스케줄 ─────────────────────────────────────────────────────────
    "freshness_days": 3,              # 이 일수 이내 모델이면 정기 재학습 스킵 (drift 감지 시 무시)
    "n_trials": 150,                  # Optuna 하이퍼파라미터 탐색 횟수 (30 이하는 랜덤서치와 무차별)
    # ── 드리프트 감지 (auto_retrain.py) ──────────────────────────────────────
    "drift_payoff_threshold": 1.0,    # payoff_ema < 1.0 → Tier-3 폴백 임계 (거래 부족 봇용)
    "drift_ratio_trigger":    0.5,    # Tier-3: 임계 미달 봇 비율 ≥ 50% → 즉시 재학습
    "oos_months":             4,      # OOS 창 크기 (최근 N개월 고정) — 0이면 quantile(0.7) 폴백
    "adwin_min_trades":       10,     # Wilcoxon/t-검정 참여 최소 거래 수 (미달 시 Tier-3 폴백)
    "adwin_significance":     0.05,   # 통계 검정 유의 수준 (p < 이 값이면 드리프트 판정)
    "rolling_train_months":   12,     # 최종 모델 fit 시 최근 N개월만 사용 (Optuna CV는 전체 train_df 유지)
    "regime_models_enabled":  True,   # HMM 국면별(Bull/Sideways/Bear) XGBoost 분기 학습
    "regime_min_samples":     300,    # 국면별 최소 학습 샘플 수 미달 시 전체 모델 폴백
    # ── 온라인 재캘리브레이션 (auto_retrain.py recalibrate_models) ─────────────
    "recal_weeks": 3,                 # isotonic 재적합 창 (최근 N주 × 24봉/주)
    "btc_ref_feats": [                # B/C 카테고리 BTC 레퍼런스 피처 — machine_learning·auto_retrain 공유
        "RSI", "RSI_Short", "BB_Width", "BB_Position",
        "HMM_Bear", "HMM_Bull", "OFI_CumDelta_5", "OFI_CumDelta_10",
        "Macro_Trend_Up", "ATR_Ratio", "Volume_Surge", "MACD_Hist_Ratio",
    ],
}

# ============================================================================
# 18. 서킷 브레이커 (Circuit Breaker) — 일일 손실 한도 자동 트레이딩 차단
# ============================================================================

CIRCUIT_BREAKER = {
    "enabled": True,
    # 일일 포트폴리오 손실 -3% 초과 시 당일 신규 진입 차단
    # 연속 3패 후 60분 냉각: 과적합 연속 손절 방어
    "daily_loss_pct": -0.03,          # -3% 일일 손실 한도
    "consecutive_loss_count": 3,      # 연속 패배 허용 횟수
    "cooldown_minutes": 60,           # 연속 패배 후 냉각 시간 (분)
    "reset_hour": 9,                  # 매일 오전 9시 초기화 (KST)
    "max_drawdown_pct": -0.20,        # 누적 자본 Peak 대비 -20% 낙폭 시 봇 완전 정지 (0 = 비활성)
}

# ============================================================================
# 19. 재현성 시드 (모든 모델/분할에 통일 적용)
# ============================================================================

RANDOM_SEED = 42  # XGBoost, LightGBM, HMM, KMeans, Optuna sampler 공통 시드

# ============================================================================
# 20. 거래 신호 우선순위 계층
# ============================================================================
# 신호 간 충돌 시 아래 순서대로 우선 적용 (높을수록 절대 우선)
# 1(최우선): 리스크 게이트    — 지갑 락업, DAXA 경고, 킬 스위치
# 2:         회로차단기       — 일일 손실 한도, 연속 패배 냉각
# 3:         펀딩비/과열 게이트— 펀딩비 폭탄, 시장 과열(F&G+펀딩비)
# 4:         HMM 레짐        — bear=포지션 절반, bear+chop=진입 차단
# 5:         K-Means Chop   — 고변동성 횡보 차단
# 6:         BTC 거시 추세   — BTC 하락 시 진입 보류
# 7(최하):   AI 확률         — 진입/청산 최종 의사결정
SIGNAL_PRIORITY = {
    "risk_gate":         1,
    "circuit_breaker":   2,
    "market_overheat":   3,
    "hmm_regime":        4,
    "liquidity_flow":    4.5,  # 유동성 흐름 게이트 (HMM 직후)
    "kmeans_chop":       5,
    "btc_macro":         6,
    "ai_probability":    7,
}

# ============================================================================
# 21-A. 유동성 흐름 게이트 (LiquidityFlowMonitor)
# ============================================================================
# 매크로 유동성 2신호 + 코인 볼륨 1신호, min_signals 이상 충족 시 진입 허용
# 신호 1: 암호화폐 총 시총 total_mcap_days 변화 > 0  (신규 자금 유입)
# 신호 2: BTC 도미넌스 btc_dom_days 변화 ≤ btc_dom_threshold  (알트 로테이션)
# 신호 3: 코인 24h 거래대금 > 7일 평균 × volume_surge_ratio  (로컬 축적)
LIQUIDITY_FLOW = {
    "enabled":            True,
    "cache_ttl":          3600,   # CoinGecko 매크로 신호 갱신 주기 (초)
    "total_mcap_days":    7,      # 총 시총 변화 측정 구간 (일)
    "btc_dom_days":       5,      # BTC 도미넌스 변화 측정 구간 (일)
    "btc_dom_threshold":  -0.5,   # BTC.D 하락 기준 (%p, 음수) — -0.5%p 이상 하락
    "volume_surge_ratio": 1.5,    # 코인 24h 거래대금 > 7일 평균 × 1.5배
    "min_signals":        2,      # 3개 중 충족 최소 신호 수
}

# ============================================================================
# 21-B. 선물 헤지 봇 설정 (orchestrator 연동)
# ============================================================================
# Bear 국면에서 BTC 선물 숏을 자동 진입, 자본 유휴 없이 수익 창출
# futures_bot.AsyncFuturesBot을 orchestrator가 daemon 스레드로 구동
FUTURES_HEDGE = {
    "enabled":             True,
    "symbol":              "BTC/USDT",
    "leverage":            2,                # MAX_LEVERAGE(3) 이하 하드코딩 안전장치 있음
    "paper_balance_usdt":  1000.0,           # DRY_RUN 시 가상 초기 잔고 (USDT)
    "watchdog_interval":   120,              # 스레드 생존 점검 주기 (초)
}

# ============================================================================
# 21. LABEL_COST_BUFFER 실행 시 검증
# ============================================================================

def _validate_label_cost() -> float:  # noqa: D401
    """
    모듈 로드 시 왕복 비용 합산이 현실적 수준인지 검증.
    현재 설정: 수수료 0.05%×2 + 슬리피지 0.20% = 0.30% (LABEL_COST_BUFFER 최소 임계)
    이 값보다 낮으면 수익 없는 샘플을 승리로 학습하게 됨.
    """
    total = LABELING["fee_threshold"] * 2 + LABELING["slippage_roundtrip"]
    if total < 0.003:  # 총 왕복 비용 0.3% 미만은 비현실적
        import warnings
        warnings.warn(
            f"[config] LABEL_COST_BUFFER={total:.3f} < 0.3% — "
            "왕복 비용 과소 추정 → 수익 없는 샘플 학습 위험. "
            "slippage_roundtrip을 현실적 값으로 상향하세요.",
            UserWarning, stacklevel=2
        )
    return total

LABEL_COST_BUFFER = _validate_label_cost()

# ============================================================================
# 22. 업비트 리스크 관리 (지갑 락업 + DAXA 투자유의종목)
# ============================================================================

UPBIT_RISK = {
    "enabled": True,
    "cache_ttl": 120,          # 지갑/경고 상태 캐시 유효 시간 (초)
    "block_on_api_failure": True,   # API 실패 시 보수적으로 진입 차단
}

# ============================================================================
# 23. 선물 리스크 관리
# ============================================================================

FUTURES_RISK = {
    "funding_risk_threshold": 0.00025,  # 펀딩비 절대값 0.025% 초과 시 진입 차단
    "exchange_info_refresh_interval": 3600,  # 틱사이즈·수량단위 갱신 주기 (초)
    "trailing_stop_callback_rate": 2.0,      # 네이티브 TS 콜백 비율 (%)
    "use_native_trailing_stop": True,        # True: 거래소 TRAILING_STOP_MARKET 사용
    "daily_loss_limit_pct": 0.03,           # 서킷 브레이커: 일일 손실 3% 초과 시 당일 거래 중단
}

# ============================================================================
# 24. 앙상블 모델 설정 (XGBoost + LightGBM)
# ============================================================================

ENSEMBLE = {
    "enabled": True,             # False: XGBoost 단독 사용
    "lgbm_weight": 0.4,          # LightGBM 가중치 (XGBoost = 1 - lgbm_weight)
    "min_agreement_prob": 0.15,  # 두 모델 확률 차이가 이 이상이면 보수적 처리 (0.05는 항상 트리거)
}

# ============================================================================
# 24-B. 멀티 타임프레임 앙상블 (MTF Ensemble)
# ============================================================================
# 60분봉 모델(추세 방향) AND 15분봉 모델(진입 타이밍) 동시 동의 시에만 진입
# - 60m: 4년치 데이터 기반, 중장기 추세 포착 → 방향성 필터
# - 15m: 2년치 데이터 기반, 5시간 예측 horizon → 진입 타이밍 정밀화
# - AND 로직: 신호 수 감소 대신 정밀도 향상

MTF_ENSEMBLE = {
    "enabled": True,               # False: 60m 단독 모드 (레거시 동작)
    # 인터벌별 모델 디렉토리 매핑 (카테고리 접미사는 get_mtf_model_dir()에서 결합)
    "intervals": {
        "minute60": {
            "models_A": "models_A",
            "models_B": "models_B",
            "models_C": "models_C",
            "ohlcv_bars": 700,     # 예측용 OHLCV 조회 봉 수
            "role": "direction",   # 60m = 추세 방향 결정
        },
        "minute15": {
            "models_A": "models_A_15m",
            "models_B": "models_B_15m",
            "models_C": "models_C_15m",
            "ohlcv_bars": 2880,    # 2880봉 × 15min = 30일 — MTF_1D_RSI/BB_Pos(SMA_20=20일) + HMM lookback(2880봉) 충족
            "role": "timing",      # 15m = 진입 타이밍 정밀화
        },
    },
    # 인터벌별 독립 threshold — 각 모델 Optuna 최적값 사용, 미로드 시 이 기본값 적용
    "default_threshold_60m": 0.55,
    "default_threshold_15m": 0.50,  # 타이밍 모델은 다소 완화 (신호 확보)
}


def get_mtf_model_dir(coin: str, interval: str) -> str:
    """코인 심볼 + 인터벌 → MTF 모델 디렉토리 경로."""
    cat = get_coin_category(coin)  # 순환 참조 없음 — 아래 정의됨
    key = f"models_{cat}" if interval == "minute60" else f"models_{cat}_15m"
    return DIRECTORIES[key]


# ============================================================================
# 25. 선물 전용 모델 설정 (현물 모델과 완전 분리)
# ============================================================================

FUTURES_MODEL = {
    "symbols": ["BTC/USDT", "ETH/USDT"],  # 선물 훈련 대상 심볼
    "intervals": ["minute15", "minute60"],    # 선물 수집 인터벌 (TARGET_INTERVALS 일치)
    "count": 3500,                             # 수집 봉 수 (15분봉 ≒ 36일, OOS 신호 최소 요건 확보)
    "model_name": "futures_bot",            # 선물 모델 파일명 prefix
    # 선물은 펀딩비·숏스퀴즈 노이즈로 현물보다 낮은 정밀도 기준 적용
    "min_precision": 0.57,
    "min_recall": 0.08,
    "min_signals": 20,
    "purge_gap_bars": 16,  # Purged CV 갭 (LABELING.forward_bars와 동일하게 유지)
}

# ============================================================================
# 26. 동적 리스크 파라미터 핫-리로드
# ============================================================================
# dynamic_config.json 파일 mtime 변경 감지 → 봇 재시작 없이 핵심 리스크 파라미터 반영
# 허용 키: kelly_fraction / trailing_stop_pct / daily_loss_pct / cooldown_minutes
#
# 사용법:
#   kelly = config.dyn.get("kelly_fraction", config.KELLY_FRACTION)
#
# dynamic_config.json 예시:
#   { "kelly_fraction": 0.15, "trailing_stop_pct": 0.05, "daily_loss_pct": -0.02 }

class _DynamicConfigManager:
    _ALLOWED: dict = {
        "kelly_fraction":    (0.0,   1.0),
        "trailing_stop_pct": (0.001, 0.20),
        "daily_loss_pct":    (-0.50, -0.001),
        "cooldown_minutes":  (1,     1440),
    }

    def __init__(self, path: str = "dynamic_config.json") -> None:
        self._path = path
        self._mtime = 0.0
        self._lock = threading.Lock()
        self._values: dict = {}

    def _reload_if_changed(self) -> None:
        try:
            mtime = os.path.getmtime(self._path)
        except FileNotFoundError:
            return
        with self._lock:
            if mtime <= self._mtime:
                return
            try:
                with open(self._path, encoding="utf-8") as f:
                    raw = json.load(f)
                validated: dict = {}
                for key, (lo, hi) in self._ALLOWED.items():
                    if key not in raw:
                        continue
                    val = raw[key]
                    if not isinstance(val, (int, float)):
                        logging.warning(f"[DynamicConfig] {key}={val!r} 타입 오류 (숫자 필요) — 무시됨")
                        continue
                    if lo <= val <= hi:
                        validated[key] = val
                    else:
                        logging.warning(f"[DynamicConfig] {key}={val} out of [{lo}, {hi}] — 무시됨")
                self._values = validated
                self._mtime = mtime
                logging.info(f"[DynamicConfig] 리로드 완료: {list(validated.keys())}")
            except Exception as e:
                logging.warning(f"[DynamicConfig] 파싱 실패: {e}")

    def get(self, key: str, default):
        self._reload_if_changed()
        with self._lock:
            return self._values.get(key, default)


dyn = _DynamicConfigManager()

# ============================================================================
# 27. 런타임 파라미터 정합성 검증
# ============================================================================

def validate_config() -> None:
    """
    config 파라미터 간 상호 의존 불변식 검사.
    위반 시 ValueError — 봇 시작 전에 호출하여 잘못된 설정으로 인한 손실 방지.
    """
    errors: list[str] = []

    tp = LABELING["atr_tp_mult"]
    sl = LABELING["atr_sl_mult"]
    if tp <= sl:
        errors.append(
            f"LABELING: atr_tp_mult({tp}) must be > atr_sl_mult({sl}) "
            f"— TP가 SL보다 작으면 수익 기대값이 음수"
        )

    cost_buffer = LABELING["fee_threshold"] * 2 + LABELING["slippage_roundtrip"]
    min_expected_gain = (LABELING["atr_tp_mult"] - LABELING["atr_sl_mult"]) * LABELING["min_atr_ratio"]
    if cost_buffer >= min_expected_gain:
        errors.append(
            f"LABELING: 비용 버퍼({cost_buffer:.4f})가 최소 기대 수익({min_expected_gain:.4f})보다 큼 "
            f"— ATR 배수를 키우거나 수수료/슬리피지 설정을 재확인"
        )

    sma_short = TECHNICAL_INDICATORS["SMA_SHORT"]
    sma_long = TECHNICAL_INDICATORS["SMA_LONG"]
    if sma_short >= sma_long:
        errors.append(
            f"TECHNICAL_INDICATORS: SMA_SHORT({sma_short}) must be < SMA_LONG({sma_long})"
        )

    macd_fast = TECHNICAL_INDICATORS["MACD_FAST"]
    macd_slow = TECHNICAL_INDICATORS["MACD_SLOW"]
    if macd_fast >= macd_slow:
        errors.append(
            f"TECHNICAL_INDICATORS: MACD_FAST({macd_fast}) must be < MACD_SLOW({macd_slow})"
        )

    ensemble_w = ENSEMBLE["lgbm_weight"]
    if not (0.0 < ensemble_w < 1.0):
        errors.append(
            f"ENSEMBLE: lgbm_weight({ensemble_w}) must be in (0, 1)"
        )

    fg = FEAR_GREED
    if fg["greed"] >= fg["extreme_greed"]:
        errors.append(
            f"FEAR_GREED: greed({fg['greed']}) must be < extreme_greed({fg['extreme_greed']})"
        )

    if BACKTEST["commission"] != LABELING["fee_threshold"]:
        errors.append(
            f"BACKTEST: commission({BACKTEST['commission']}) must equal "
            f"LABELING.fee_threshold({LABELING['fee_threshold']}) "
            f"— 백테스트 수수료와 라벨링 비용 기준 불일치"
        )

    if BACKTEST["stress_slippage"] < SLIPPAGE_DEFAULT:
        errors.append(
            f"BACKTEST: stress_slippage({BACKTEST['stress_slippage']}) must be >= "
            f"SLIPPAGE_DEFAULT({SLIPPAGE_DEFAULT}) — 스트레스 슬리피지가 일반보다 낮으면 안 됨"
        )

    fr_risk = FUTURES_RISK["funding_risk_threshold"]
    fr_overheat = BINANCE_CONTEXT["funding_rate_threshold"]
    if fr_risk >= fr_overheat:
        errors.append(
            f"FUTURES_RISK: funding_risk_threshold({fr_risk}) must be < "
            f"BINANCE_CONTEXT.funding_rate_threshold({fr_overheat}) "
            f"— 선물 진입차단이 과열감지보다 느슨하면 안 됨"
        )

    if not (0.0 < KELLY_FRACTION < 1.0):
        errors.append(f"KELLY_FRACTION({KELLY_FRACTION}) must be in (0, 1)")

    hmm_bear = HMM_REGIME["threshold_bear"]
    hmm_bull = HMM_REGIME["threshold_bull"]
    if hmm_bear <= hmm_bull:
        errors.append(
            f"HMM_REGIME: threshold_bear({hmm_bear}) must be > threshold_bull({hmm_bull}) "
            f"— 약세 국면이 강세보다 높은 확률 필요 (bear > bull)"
        )

    if CIRCUIT_BREAKER["daily_loss_pct"] >= 0:
        errors.append(
            f"CIRCUIT_BREAKER: daily_loss_pct({CIRCUIT_BREAKER['daily_loss_pct']}) must be < 0"
        )

    if FUTURES_MODEL["purge_gap_bars"] != LABELING["forward_bars"]:
        errors.append(
            f"FUTURES_MODEL: purge_gap_bars({FUTURES_MODEL['purge_gap_bars']}) must equal "
            f"LABELING.forward_bars({LABELING['forward_bars']}) "
            f"— Purged CV 갭이 예측 시계와 다르면 데이터 누수 발생"
        )

    futures_count = FUTURES_MODEL["count"]
    futures_min_signals = FUTURES_MODEL["min_signals"]
    if futures_count < futures_min_signals * 50:
        errors.append(
            f"FUTURES_MODEL: count({futures_count}) < min_signals({futures_min_signals}) × 50 "
            f"— OOS 검증에 필요한 최소 봉 수 부족"
        )

    for ivl in FUTURES_MODEL["intervals"]:
        if ivl not in TARGET_INTERVALS:
            errors.append(
                f"FUTURES_MODEL: intervals에 '{ivl}' 이(가) TARGET_INTERVALS에 없음 — 데이터 수집 실패 위험"
            )

    for ivl in MODEL_MANAGEMENT.get("train_intervals", []):
        if ivl not in TARGET_INTERVALS:
            errors.append(
                f"MODEL_MANAGEMENT: train_intervals에 '{ivl}' 이(가) TARGET_INTERVALS에 없음"
                f" — 해당 인터벌 데이터 미수집으로 학습 실패"
            )

    bt_bear = BACKTEST["hmm_bear_multiplier"]
    hmm_bear = HMM_REGIME["position_multiplier_bear"]
    if bt_bear != hmm_bear:
        errors.append(
            f"BACKTEST.hmm_bear_multiplier({bt_bear}) != HMM_REGIME.position_multiplier_bear({hmm_bear}) "
            f"— 백테스트와 실거래 포지션 배율 불일치"
        )

    if errors:
        raise ValueError("config.py 파라미터 정합성 오류:\n" + "\n".join(f"  - {e}" for e in errors))


validate_config()


# ============================================================================
# 29. 코인 카테고리별 분리 모델
# ============================================================================
# Model_A: 대형 효율 시장   — 낮은 변동성·높은 유동성, 시장 효율 반영 피처 중심
# Model_B: 중형 알트        — 중간 유동성, 모멘텀·롤링 변동성 피처 강조
# Model_C: 소형 알트 알파   — 낮은 유동성·고변동성, 소량 데이터 보상 하이퍼파라미터
# COIN_TIER_MAP: 알려진 코인의 정적 티어 매핑. 미등록 코인 자동 → C

COIN_TIER_MAP: dict = {
    # ── A: 대형 (BTC·ETH만) ─────────────────────────────────────────────
    "BTC": "A", "ETH": "A",
    # ── B: 중형 알트 (업비트 거래대금 상위권 + 주요 L1/L2/DeFi) ──────────
    "XRP":  "B", "SOL":  "B", "ADA":  "B", "DOGE": "B", "SUI":  "B",
    "BCH":  "B", "LTC":  "B", "DOT":  "B", "AVAX": "B", "TRX":  "B",
    "LINK": "B", "XLM":  "B", "ATOM": "B", "ETC":  "B", "NEAR": "B",
    "OP":   "B", "ARB":  "B", "APT":  "B", "POL":  "B", "FIL":  "B",
    "UNI":  "B", "AAVE": "B", "ICP":  "B", "HBAR": "B", "VET":  "B",
    "THETA":"B", "ALGO": "B", "EOS":  "B", "QTUM": "B", "ICX":  "B",
    "ZIL":  "B", "WAVES":"B", "NEO":  "B", "FTM":  "B", "CRV":  "B",
    "MKR":  "B", "COMP": "B", "INJ":  "B", "TIA":  "B", "SEI":  "B",
    "IMX":  "B", "WLD":  "B", "TON":  "B", "SHIB": "B", "PEPE": "B",
    "FLOKI":"B", "WIF":  "B", "BONK": "B", "SAND": "B",
    "MANA": "B", "AXS":  "B", "GALA": "B", "ENJ":  "B", "BAT":  "B",
    "SNT":  "B", "ZEC":  "B", "DASH": "B", "XEM":  "B", "LSK":  "B",
    "STEEM":"B", "KNC":  "B", "GRT":  "B", "LDO":  "B", "SNX":  "B",
    "STORJ":"B", "IOST": "B", "KAVA": "B", "CELO": "B",
    # ── C 명시 (B에서 제외된 경계선 코인) ──────────────────────────────────
    "CHZ":  "C",  # 스포츠 팬토큰 — 수급 불규칙, B 신호 희석
    "ONT":  "C",  # 구형 L1 — 거래대금 감소 추세, 소형 특성
    # ── C: 소형 알파 — 미등록 코인 폴백 포함 (get_coin_category 참고) ──────
    # (명시 생략: 미등록 코인은 C 자동 배정)
}

COIN_CATEGORIES: dict = {
    "A": {
        "name": "대형 효율 시장",
        "coins": ["BTC", "ETH"],  # 데이터 수집 대상 유지 (BTC_* 레퍼런스 피처 + btc_macro 필터 필요)
        "model_dir": DIRECTORIES["models_A"],
        "model_dir_15m": DIRECTORIES["models_A_15m"],
        "strict_oos_gate": False,
        "skip_training": True,  # AUC 0.50 — 효율 시장에서 ML 엣지 없음. BTC는 매크로 필터(rule-based)로만 사용
    },
    "B": {
        "name": "중형 알트",
        "coins": ["XRP", "SOL", "ADA", "DOGE", "SUI", "BCH", "LINK"],  # 필수 수집·학습 대상 (AVAX/DOT 데이터 없음)
        "model_dir": DIRECTORIES["models_B"],
        "model_dir_15m": DIRECTORIES["models_B_15m"],
        "strict_oos_gate": True,   # TP×2.0 변경으로 손익분기 33%로 하향 — 이제 0.42 게이트 적용 가능
    },
    "C": {
        "name": "소형 알트 핵심 알파",
        "coins": ["OPEN"],  # 필수 수집·학습 대상 (백테스트 검증 코인만 등록)
        "model_dir": DIRECTORIES["models_C"],
        "model_dir_15m": DIRECTORIES["models_C_15m"],
        "default": True,
        "strict_oos_gate": True,   # C도 손익분기(0.333) 미달 시 저장 차단 — 미달 모델로 거래하면 기대수익 음수
    },
}


# 백테스트 실증 기반 영구 차단 코인 (MDD·Sharpe 복합 기준)
# SAHARA: ROI=-7.3%, MDD=19.2%, Sharpe=-0.95 (Sharpe 필터 임계값 상회하나 MDD 과대)
TRADING_BLACKLIST: set = {"SAHARA"}

# 동적 코인 자동 차단 임계값 (get_dynamic_target_coins → 최신 백테스트 결과 참조)
BACKTEST_PERFORMANCE_FILTER: dict = {
    "enabled": True,
    "sharpe_min": -1.0,   # 이 미만이면 다음 사이클 동적 편입 차단
    "min_trades": 10,      # 거래 수 미달 시 통계 불신뢰 → 필터 미적용
}

# 백테스트 결과 누적 기반 동적 블랙리스트 관리
PERFORMANCE_BLACKLIST_CRITERIA: dict = {
    "enabled": True,
    "blacklist_file": "logs/performance_blacklist.json",
    "sharpe_min": -1.0,         # 이 미만 시 실패 카운트
    "sharpe_extreme": -3.0,     # 이 미만 시 단회 즉시 차단 (MDD 무관)
    "mdd_max": 15.0,            # MDD 이 초과 시 즉시 차단 (단회)
    "min_trades": 3,            # 거래수 미달 시 판단 보류
    "consecutive_fail": 2,      # 연속 실패 횟수 충족 시 차단
    "sharpe_recover": 0.3,      # 차단 해제 기준 Sharpe (0.0→0.3: 마진 코인 bounce 방지)
    "roi_recover": 0.5,         # 차단 해제 기준 ROI % (두 조건 모두 충족 시 해제)
    "max_unblock_per_cycle": 4, # 사이클당 최대 probation 전환 수 (Sharpe 상위 N개)
}


def get_coin_category(coin: str) -> str:
    """코인 심볼 → 카테고리 키. COIN_TIER_MAP 조회 → 미등록 시 'C'."""
    return COIN_TIER_MAP.get(coin.upper(), "C")


def get_model_dir(coin: str) -> str:
    """코인 심볼 → 해당 카테고리 모델 디렉토리 경로."""
    return COIN_CATEGORIES[get_coin_category(coin)]["model_dir"]


# ============================================================================
# 28. 필수 환경변수 검증 (봇 시작 전 명시적 호출 필요)
# ============================================================================

def validate_env(mode: str = "spot") -> None:
    """
    mode: "spot" | "futures" | "all"
    봇 진입점(trade_bot.py / futures_bot.py / main.py)에서 호출.
    API 호출 시점이 아닌 시작 시점에 누락 ENV를 즉시 차단.
    """
    required: dict[str, list[str]] = {
        "spot":    ["UPBIT_ACCESS_KEY", "UPBIT_SECRET_KEY"],
        "futures": ["BINANCE_API_KEY", "BINANCE_SECRET_KEY"],
        "notify":  ["TELEGRAM_TOKEN", "TELEGRAM_CHAT_ID"],
    }
    missing: list[str] = []
    targets = ["spot", "notify"] if mode == "spot" else \
              ["futures", "notify"] if mode == "futures" else \
              ["spot", "futures", "notify"]
    for group in targets:
        for key in required.get(group, []):
            if not os.environ.get(key):
                missing.append(key)
    if missing:
        raise EnvironmentError(
            f"[config] 필수 환경변수 미설정: {', '.join(missing)}\n"
            "  .env 파일 또는 시스템 환경변수를 확인하세요."
        )