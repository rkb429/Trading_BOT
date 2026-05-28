"""
backtest.py
Coin AI Bot - 4단계: 실전 환경을 모사한 Backtrader 백테스팅
슬리피지, 수수료, 트레일링 스탑, 그리고 전문 분석 지표(MDD, Sharpe, 승률)를 모두 포함합니다.
"""

import os
import re
import json
import joblib
import glob
import traceback
import statistics as _st
import pandas as pd
import numpy as np
import backtrader as bt
import backtrader.analyzers as btanalyzers
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# 설정 import
import config

# ==========================================
# 1. 로깅 및 기본 설정
# ==========================================
def setup_logger():
    logger = logging.getLogger('Backtest')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', 
                                     datefmt='%Y-%m-%d %H:%M:%S')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

logger = setup_logger()

# ==========================================
# 2. 동적 파일 경로 설정
# ==========================================
def find_latest_file(directory, pattern):
    """디렉토리에서 최신 파일 찾기"""
    if not os.path.exists(directory):
        return None
    
    files = glob.glob(os.path.join(directory, pattern))
    if not files:
        return None
    
    return max(files, key=os.path.getctime)

def get_file_paths(model_dir: str = None):
    """사용할 파일 경로 결정. model_dir 미지정 시 config.DIRECTORIES['models'] 사용."""
    _model_dir = model_dir if model_dir else config.DIRECTORIES['models']
    data_dir = config.DIRECTORIES['data_processed']

    # 1. 모델 파일 찾기 (앙상블 우선, 없으면 XGBoost — trade_bot.py 초기화와 동일한 우선순위)
    model_path = (
        find_latest_file(_model_dir, "*ensemble_bot*.pkl")
        or find_latest_file(_model_dir, "*xgb_bot*.pkl")
    )
    config_path = find_latest_file(_model_dir, "config_*.json")
    
    # 2. 데이터 파일 선택 (사용자 입력 또는 자동)
    processed_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")), key=os.path.getmtime)
    
    return {
        'model': model_path,
        'config': config_path,
        'data_dir': data_dir,
        'processed_files': processed_files
    }


def _load_model_and_config(paths: dict):
    """모델·피처·임계값 로드 헬퍼."""
    try:
        model = joblib.load(paths['model'])
        with open(paths['config'], 'r', encoding='utf-8') as f:
            ml_config = json.load(f)
        return model, ml_config['features'], ml_config['best_threshold'], ml_config
    except Exception as e:
        logger.error(f"❌ 모델/설정 로드 실패: {type(e).__name__} - {e}")
        return None, None, None, None


def _extract_feature_order(model, features: list) -> list:
    """모델 내부 피처 순서 추출 — feature_names_in_ 또는 booster 기준."""
    try:
        if hasattr(model, 'feature_names_in_') and model.feature_names_in_ is not None:
            return list(model.feature_names_in_)
        if hasattr(model, 'xgb_model') and hasattr(model.xgb_model, 'calibrated_classifiers_'):
            return list(model.xgb_model.calibrated_classifiers_[0].estimator.get_booster().feature_names)
        if hasattr(model, 'calibrated_classifiers_'):
            return list(model.calibrated_classifiers_[0].estimator.get_booster().feature_names)
    except Exception:
        pass
    return features

# ✅ 파일 경로 (config.py에서 동적으로 설정됨)
DIRECTORIES = config.DIRECTORIES

# SharpeRatio 연환산 계수.
# timeframe=bt.TimeFrame.Days 설정 시 Backtrader가 15분봉을 일별로 먼저 집계한 뒤
# sqrt(factor)로 연환산하므로 365가 맞다. (bar 단위 계산이라면 96×365가 필요하지만
# 여기서는 daily aggregation을 사용한다.)
_SHARPE_ANNUAL_FACTOR = 365


def _row_count(path: str) -> int:
    """CSV 행 수 반환 (헤더 제외). 파일 크기 대신 실제 행 수로 데이터 길이를 비교한다."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return sum(1 for _ in f) - 1
    except OSError:
        return 0


def _create_cerebro(feed: "MLSignalData", slippage: float) -> bt.Cerebro:
    """Cerebro 인스턴스를 표준 설정으로 초기화한다.

    addstrategy / adddata / broker(cash·commission·slippage) / analyzer 3종을
    한 번에 구성해 세 개의 백테스트 함수 간 중복 설정을 제거한다.
    """
    cerebro = bt.Cerebro()
    cerebro.addstrategy(MLStrategy)
    cerebro.adddata(feed)
    cerebro.broker.setcash(float(config.BACKTEST["initial_cash"]))
    cerebro.broker.setcommission(commission=float(config.BACKTEST["commission"]))
    cerebro.broker.set_slippage_perc(slippage)
    cerebro.addanalyzer(btanalyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(
        btanalyzers.SharpeRatio, _name="sharpe",
        timeframe=bt.TimeFrame.Days, annualize=True,
        factor=_SHARPE_ANNUAL_FACTOR, riskfreerate=0.0,
    )
    return cerebro


def _parse_cerebro_results(cerebro: bt.Cerebro, strat, initial_cash: float) -> dict:
    """DrawDown / SharpeRatio / TradeAnalyzer 결과를 .get() 패턴으로 안전하게 추출."""
    end_value = cerebro.broker.getvalue()
    roi = (end_value / initial_cash - 1) * 100

    mdd = 0.0
    try:
        dd = strat.analyzers.drawdown.get_analysis()
        mdd = dd.get('max', {}).get('drawdown', 0.0) or 0.0
    except Exception:
        pass

    sharpe = 0.0
    try:
        sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio', 0.0) or 0.0
    except Exception:
        pass

    total = win = loss = 0
    win_rate = avg_win = avg_loss = 0.0
    try:
        t = strat.analyzers.trades.get_analysis()
        total = t.get('total', {}).get('closed', 0)
        if total:
            win      = t.get('won',  {}).get('total', 0)
            loss     = t.get('lost', {}).get('total', 0)
            win_rate = win / total * 100
            avg_win  = t.get('won',  {}).get('pnl', {}).get('average', 0.0)
            avg_loss = t.get('lost', {}).get('pnl', {}).get('average', 0.0)
    except Exception:
        pass

    return {
        'end_value': end_value, 'roi': roi,
        'mdd': mdd, 'sharpe': sharpe,
        'total_trades': total, 'win_trades': win, 'loss_trades': loss,
        'win_rate': win_rate, 'avg_win': avg_win, 'avg_loss': avg_loss,
    }

# ==========================================
# 2. Backtrader 커스텀 데이터 피드
# ==========================================
class MLSignalData(bt.feeds.PandasData):
    """머신러닝 예측값(ML_Signal), 예측 확률, HMM 국면, ATR_Ratio를 처리하는 커스텀 데이터 피드"""
    lines = ('ml_signal', 'ml_prob', 'hmm_state', 'atr_ratio', 'macro_trend',)
    params = (
        ('datetime', None),
        ('open', 'open'),
        ('high', 'high'),
        ('low', 'low'),
        ('close', 'close'),
        ('volume', 'volume'),
        ('ml_signal', 'ML_Signal'),
        ('ml_prob', 'ML_Prob'),
        ('hmm_state', 'HMM_State'),     # 0=Bear 1=Sideways 2=Bull (라이브봇과 동기화)
        ('atr_ratio', 'ATR_Ratio'),     # ATR/가격 비율 — 동적 포지션 크기 계산용
        ('macro_trend', 'Macro_Trend_Up'),  # 일봉 SMA20 대비 거시 추세 (0=하락 1=상승)
    )

# ==========================================
# 3. 실전 매매 전략 클래스
# ==========================================
class MLStrategy(bt.Strategy):
    """ML 신호 기반 매매 전략 — 라이브봇(trade_bot.py) 로직과 동기화"""
    params = (
        ('trailing_stop_pct', config.BACKTEST['trailing_stop_pct']),
        ('trade_size_pct', config.KELLY_FRACTION * 0.5),
        ('log_trades', True),
        ('ai_exit_prob', config.BACKTEST['ai_exit_prob']),
        ('hmm_bear_multiplier', config.BACKTEST['hmm_bear_multiplier']),
        ('time_stop_bars', config.LABELING.get('forward_bars', config.BACKTEST['time_stop_bars'])),
        # config 단일 소스 — trade_bot.py ATR_RISK_PCT와 반드시 동기화
        ('atr_risk_pct', config.BACKTEST.get('atr_risk_pct', 0.02)),
        ('time_stop_profit_exempt_pct', config.BACKTEST.get('time_stop_profit_exempt_pct', 0.005)),
    )

    # Half-Kelly + ATR 포지션 크기 파라미터 (trade_bot.py와 동일 로직)
    _KELLY_PAYOFF = float(config.LABELING.get("atr_tp_mult", 2.0)) / float(config.LABELING.get("atr_sl_mult", 1.0))
    _MIN_POS = 0.05   # 최소 포지션 비중 5%
    _MAX_POS = 0.40   # 최대 포지션 비중 40%
    _ATR_STOP_MULT = float(config.LABELING.get("atr_sl_mult", 1.0))

    @property
    def _ATR_RISK_PCT(self):
        return self.p.atr_risk_pct

    def __init__(self):
        self.order = None
        self.highest_price = 0.0
        self.buy_price = 0.0
        self.trades_count = 0
        self.wins = 0
        self.losses = 0
        self.trade_profits = []
        self.holding_periods = []
        self.buy_bar = 0
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.max_consecutive_wins = 0
        self.max_consecutive_losses = 0
        self._payoff_ema = self._KELLY_PAYOFF  # 실측 손익비 EMA (거래별 갱신)
        self._trade_count = 0  # EMA 신뢰도 판단용
        self.equity_curve: list = []  # (datetime, portfolio_value) — 포트폴리오 합산용

    def _compute_position_pct(self, ai_prob: float, atr_ratio: float) -> float:
        """Half-Kelly + ATR 동적 포지션 비중 — trade_bot._compute_position_size 로직 이식."""
        kelly_fraction = 0.0
        if ai_prob > 0.0:
            b = self._payoff_ema
            p = ai_prob
            raw_kelly = (p * b - (1 - p)) / b
            kelly_fraction = max(0.0, raw_kelly) * 0.5  # Half-Kelly

        if atr_ratio <= 0:
            raw_pct = kelly_fraction if kelly_fraction > 0 else self.p.trade_size_pct
        else:
            atr_raw_pct = self._ATR_RISK_PCT / (atr_ratio * self._ATR_STOP_MULT)
            raw_pct = min(kelly_fraction, atr_raw_pct) if kelly_fraction > 0 else atr_raw_pct

        return max(self._MIN_POS, min(self._MAX_POS, raw_pct))

    def notify_order(self, order):
        """주문 상태 변경 시 호출"""
        # 주문 제출/접수 상태는 무시
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        # 체결 완료 시
        if order.status in [order.Completed]:
            if order.isbuy():
                self.buy_price = order.executed.price
                self.highest_price = order.executed.price  # 진입 가격을 고점으로 초기화
                self.buy_bar = len(self)                   # 타임스탑 기산점

                if self.p.log_trades:
                    logger.debug(f"🟢 [BUY] 가격: {order.executed.price:,.0f} | "
                               f"수량: {order.executed.size:.4f} | "
                               f"총액: {order.executed.value:,.0f}")
            
            elif order.issell():
                self.trades_count += 1
                profit = order.executed.pnl
                profit_pct = (order.executed.price / self.buy_price - 1) * 100 if self.buy_price > 0 else 0

                # payoff EMA 갱신 (Half-Kelly 계산 품질 향상 — trade_bot과 동일 로직)
                if profit > 0 and self.buy_price > 0:
                    net_pct = (order.executed.price / self.buy_price - 1.0)
                    atr_now = float(self.data.atr_ratio[0])
                    trade_payoff = net_pct / max(atr_now * self._ATR_STOP_MULT, 0.001) if atr_now > 0 else self._KELLY_PAYOFF
                else:
                    trade_payoff = 0.0  # 손실 → 0.0 입력으로 EMA 희석 (trade_bot과 동일)
                self._payoff_ema = 0.1 * trade_payoff + 0.9 * self._payoff_ema
                self._payoff_ema = max(0.5, min(5.0, self._payoff_ema))
                self._trade_count += 1

                self.trade_profits.append(profit)
                if self.buy_bar > 0:
                    holding_period = len(self) - self.buy_bar
                    self.holding_periods.append(holding_period)
                
                # ✅ 연속 수익/손실 추적
                if profit > 0:
                    self.wins += 1
                    icon = "🔥"
                    self.consecutive_wins += 1
                    self.consecutive_losses = 0
                    self.max_consecutive_wins = max(self.max_consecutive_wins, self.consecutive_wins)
                else:
                    self.losses += 1
                    icon = "💧"
                    self.consecutive_losses += 1
                    self.consecutive_wins = 0
                    self.max_consecutive_losses = max(self.max_consecutive_losses, self.consecutive_losses)
                
                if self.p.log_trades:
                    logger.debug(f"{icon} [SELL] 가격: {order.executed.price:,.0f} | "
                               f"수익: {profit:,.0f} ({profit_pct:.2f}%) | "
                               f"거래#{self.trades_count} | 보유기간: {len(self) - self.buy_bar}봉")
                
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            if self.p.log_trades:
                logger.warning("⚠️  주문 거절/취소/증거금 부족")

        self.order = None  # 대기 주문 초기화

    def next(self):
        """각 바(캔들) 마다 호출되는 전략 로직"""
        self.equity_curve.append((self.data.datetime.datetime(0), self.broker.getvalue()))
        # 대기 중인 주문이 있으면 중복 주문 금지
        if self.order:
            return

        # 1. 포지션이 없을 때 (매수 로직)
        if not self.position:
            if self.data.ml_signal[0] == 1:
                # Half-Kelly + ATR 동적 포지션 크기 (trade_bot._compute_position_size 이식)
                ai_prob   = float(self.data.ml_prob[0])
                atr_ratio = float(self.data.atr_ratio[0])
                pos_pct   = self._compute_position_pct(ai_prob, atr_ratio)

                # HMM_State / Macro_Trend_Up 는 모델 학습 피처로 이미 확률에 반영됨.
                # 백테스트에서 같은 컬럼으로 이중 필터링하면 하락장 HMM Bull 구간
                # (Dead Cat Bounce)만 골라 진입 → 승률 역선택 발생. 제거.
                # 실전 trade_bot은 온라인 HMM(별도 재학습) + BTC 매크로(독립 API)를
                # 사용하므로 독립 게이트가 유효하다.
                target_value = self.broker.getcash() * pos_pct
                entry_price_estimate = self.data.close[0] * 1.005
                size_to_buy = target_value / entry_price_estimate
                self.order = self.buy(size=size_to_buy)

        # 2. 포지션이 있을 때 (청산 로직 - 트레일링 스탑 & AI 강제 청산 & 타임스탑)
        else:
            current_price = self.data.close[0]

            # 타임스탑: 라이브봇과 동일하게 forward_bars 초과 보유 시 강제 청산.
            # 단, +0.5% 이상 수익 중인 포지션은 면제 — 트레일링 스탑이 이미 보호하고 있음.
            bars_held = len(self) - self.buy_bar
            if bars_held >= self.p.time_stop_bars:
                profit_pct = (current_price / self.buy_price - 1.0) if self.buy_price > 0 else 0.0
                if profit_pct >= self.p.time_stop_profit_exempt_pct:
                    if self.p.log_trades:
                        logger.debug(
                            f"⏱️  타임스탑 도달 ({bars_held}봉) — 수익 중 ({profit_pct*100:.2f}%) "
                            f"→ 트레일링 스탑에 위임"
                        )
                else:
                    if self.p.log_trades:
                        logger.debug(f"⏱️  타임스탑 발동 ({bars_held}봉 보유 → 강제 청산)")
                    self.order = self.sell(size=self.position.size)
                    return

            # AI 위험 감지 강제 청산 로직 (예측 확률이 급락했을 때)
            if self.data.ml_prob[0] < self.p.ai_exit_prob:
                if self.p.log_trades:
                    logger.debug(f"🚨 AI 위험 감지 강제 청산 (현재 상승 확률: {self.data.ml_prob[0]*100:.1f}%)")
                self.order = self.sell(size=self.position.size)
                return

            bar_high = self.data.high[0]
            bar_low = self.data.low[0]
            atr_ratio = float(self.data.atr_ratio[0]) if self.data.atr_ratio[0] != 0 else 0.0

            # ATR 연동 동적 트레일링 스탑 (라이브봇과 전략 동기화)
            # 변동성 높을 때 넓게(최대 6%), 횡보장엔 타이트하게(최소 2%)
            if atr_ratio > 0:
                dynamic_ts = float(np.clip(atr_ratio * 1.5, 0.015, 0.05))
            else:
                dynamic_ts = self.p.trailing_stop_pct  # ATR 없을 때 config 기본값 폴백

            # Pessimistic intra-candle: 고점 갱신 전에 저점으로 스탑 먼저 체크.
            # 같은 봉에서 고가/저가가 둘 다 배리어를 건드릴 때, 현실에서는
            # 어느 쪽이 먼저인지 알 수 없으므로 불리한 쪽(저가 우선)을 가정.
            stop_price = self.highest_price * (1.0 - dynamic_ts)

            if bar_low <= stop_price or current_price <= stop_price:
                if self.p.log_trades:
                    logger.debug(f"🛡️  트레일링 스탑 발동 (고점: {self.highest_price:,.0f} → "
                               f"봉저점: {bar_low:,.0f}, 손절선: {stop_price:,.0f}, "
                               f"ATR연동 스탑률: {dynamic_ts*100:.1f}%)")
                self.order = self.sell(size=self.position.size)
                return

            # 스탑 미발동 확인 후에만 고점 갱신 (저가가 고가보다 먼저라는 비관적 가정 유지)
            if bar_high > self.highest_price:
                self.highest_price = bar_high

# ==========================================
# 4. 백테스팅 실행 및 평가 파이프라인
# ==========================================
def _backtest_run_one(
    data_path: str, model, feat_order: list, threshold: float,
    ml_config: dict, stress_mode: bool,
    model_15m=None, feat_order_15m: list = None, threshold_15m: float = 0.50,
    oos_start: str = None,
) -> "dict | None":
    """단일 코인 일반 백테스트 실행 — ThreadPoolExecutor 병렬 호출용.

    model_15m / feat_order_15m: MTF AND 게이트용 15m 모델 (없으면 60m 단독).
    """
    try:
        df = pd.read_csv(data_path)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp').sort_index()

        effective_start = oos_start or (ml_config.get('train_cutoff_timestamp') if ml_config else None)
        if effective_start:
            df = df[df.index >= pd.to_datetime(effective_start)]

        if len(df) < 20:
            logger.warning(f"⚠️ {os.path.basename(data_path)}: OOS 데이터 부족 ({len(df)}개)")
            return None

        ivl_m   = re.search(r'_(minute\d+|days)(?:\.csv)?$', os.path.basename(data_path))
        ivl     = ivl_m.group(1) if ivl_m else None
        ivl_cfg = ml_config.get('interval_thresholds', {}).get(ivl, {}) if ml_config else {}
        eff_thresh = ivl_cfg.get('threshold', threshold)
        if ivl_cfg.get('skip', False):
            # skip=True는 거래봇 전용 플래그 — 백테스트는 성능 평가를 위해 계속 실행
            eff_thresh = threshold  # 손익분기 미달 구간이므로 best_threshold 사용

        # BTC 레퍼런스 피처 자동 머지 (B/C 카테고리 모델이 BTC_ 피처를 학습한 경우)
        _btc_ref_needed = [f for f in feat_order if f.startswith('BTC_') and f not in df.columns]
        if _btc_ref_needed:
            _data_dir = os.path.dirname(data_path)
            _btc_path = os.path.join(_data_dir, f'processed_KRW-BTC_{ivl or "minute60"}.csv')
            if os.path.exists(_btc_path):
                try:
                    _btc_df = pd.read_csv(_btc_path)
                    _btc_df['timestamp'] = pd.to_datetime(_btc_df['timestamp'])
                    _btc_df = _btc_df.set_index('timestamp').sort_index()
                    _btc_rename = {c: f'BTC_{c}' for c in _btc_df.columns if f'BTC_{c}' in _btc_ref_needed}
                    if _btc_rename:
                        _btc_sub = _btc_df[list(_btc_rename.keys())].rename(columns=_btc_rename)
                        df = df.join(_btc_sub, how='left')
                        df[list(_btc_rename.values())] = df[list(_btc_rename.values())].ffill()
                except Exception as _e:
                    logger.warning(f"⚠️ BTC 레퍼런스 피처 로드 실패: {_e}")

        missing = [f for f in feat_order if f not in df.columns]
        if missing:
            _MTF_CRITICAL = {'MTF_1D_RSI', 'MTF_1D_BB_Pos', 'MTF_4H_RSI', 'MTF_4H_Trend', 'MTF_4H_BB_Pos'}
            critical_missing = [f for f in missing if f in _MTF_CRITICAL]
            if critical_missing:
                logger.warning(f"⚠️ {os.path.basename(data_path)} 핵심 MTF 피처 누락 — 스킵: {critical_missing}")
                return None
            logger.warning(f"⚠️ {os.path.basename(data_path)} 피처 누락 — 0으로 대체: {missing}")
            for f in missing:
                df[f] = 0.0
        df = df.dropna(subset=feat_order)
        if 'HMM_State' not in df.columns:
            df['HMM_State'] = 1.0
        if 'ATR_Ratio' not in df.columns:
            df['ATR_Ratio'] = 0.0
        if 'Macro_Trend_Up' not in df.columns:
            df['Macro_Trend_Up'] = 1.0

        pred_probs      = model.predict_proba(df[feat_order])[:, 1]
        df['ML_Prob']   = pred_probs
        df['ML_Signal'] = (pred_probs >= eff_thresh).astype(int)

        # MTF AND 게이트: 15m 신호를 60m 인덱스로 리샘플해 AND 적용
        if model_15m is not None and feat_order_15m:
            try:
                data_dir  = os.path.dirname(data_path)
                base_name = os.path.basename(data_path)
                path_15m  = os.path.join(data_dir, re.sub(r'_minute\d+\.csv$', '_minute15.csv', base_name))
                if os.path.exists(path_15m):
                    df15 = pd.read_csv(path_15m)
                    if 'timestamp' in df15.columns:
                        df15['timestamp'] = pd.to_datetime(df15['timestamp'])
                        df15 = df15.set_index('timestamp').sort_index()
                    # 15m 모델도 BTC_ 피처를 학습한 경우 머지
                    _btc15_needed = [f for f in feat_order_15m if f.startswith('BTC_') and f not in df15.columns]
                    if _btc15_needed:
                        _btc15_path = os.path.join(os.path.dirname(path_15m), 'processed_KRW-BTC_minute15.csv')
                        if os.path.exists(_btc15_path):
                            try:
                                _b15 = pd.read_csv(_btc15_path)
                                _b15['timestamp'] = pd.to_datetime(_b15['timestamp'])
                                _b15 = _b15.set_index('timestamp').sort_index()
                                _r15 = {c: f'BTC_{c}' for c in _b15.columns if f'BTC_{c}' in _btc15_needed}
                                if _r15:
                                    _bs15 = _b15[list(_r15.keys())].rename(columns=_r15)
                                    df15 = df15.join(_bs15, how='left')
                                    df15[list(_r15.values())] = df15[list(_r15.values())].ffill()
                            except Exception:
                                pass
                    missing_15m = [f for f in feat_order_15m if f not in df15.columns]
                    for f in missing_15m:
                        df15[f] = 0.0
                    df15 = df15.dropna(subset=feat_order_15m)
                    if len(df15) > 0:
                        prob15 = model_15m.predict_proba(df15[feat_order_15m])[:, 1]
                        sig15  = pd.Series((prob15 >= threshold_15m).astype(int), index=df15.index)
                        # 60m 인덱스로 forward-fill: 직전 15m 봉 신호를 60m에 반영
                        sig15_resampled = sig15.reindex(df.index, method='ffill').fillna(0).astype(int)
                        df['ML_Signal'] = (df['ML_Signal'] & sig15_resampled).astype(int)
                        logger.debug(f"MTF AND 적용: {os.path.basename(path_15m)}")
            except Exception as _e:
                logger.warning(f"⚠️ MTF 15m 백테스트 신호 생성 실패 ({_e}) — 60m 단독으로 진행")

        coin_match  = re.search(r'KRW-(\w+)', os.path.basename(data_path))
        coin        = coin_match.group(1) if coin_match else 'BTC'
        base_slip   = config.SLIPPAGE_BY_COIN.get(coin, config.SLIPPAGE_DEFAULT)
        STRESS_SLIP = float(config.BACKTEST.get('stress_slippage', 0.003))
        eff_slip    = STRESS_SLIP if stress_mode else base_slip

        cerebro  = _create_cerebro(MLSignalData(dataname=df), eff_slip)
        results  = cerebro.run()
        strat    = results[0]

        INITIAL_CASH = float(config.BACKTEST['initial_cash'])
        r   = _parse_cerebro_results(cerebro, strat, INITIAL_CASH)
        _hp = getattr(strat, 'holding_periods', [])

        logger.info(
            f"  [{coin:>6}] ROI={r['roi']:>7.1f}%  MDD={r['mdd']:>6.1f}%  "
            f"Sharpe={r['sharpe']:>5.2f}  거래={r['total_trades']:>4}회  승률={r['win_rate']:.1f}%"
        )
        return {
            "coin":          coin,
            "data_file":     os.path.basename(data_path),
            "interval":      ivl,
            "threshold":     round(eff_thresh, 4),
            "roi":           round(r['roi'],      2),
            "mdd":           round(r['mdd'],      2),
            "sharpe":        round(r['sharpe'],   2),
            "total_trades":  r['total_trades'],
            "win_rate":      round(r['win_rate'], 2),
            "win_trades":    int(r['win_trades']),
            "loss_trades":   int(r['loss_trades']),
            "avg_win":       round(r['avg_win'],  0),
            "avg_loss":      round(r['avg_loss'], 0),
            "avg_hold_bars": round(_st.mean(_hp), 2) if _hp else 0.0,
            "payoff_ema":    round(getattr(strat, '_payoff_ema', 0.0), 4),
            "equity_curve":  strat.equity_curve,  # 포트폴리오 합산용 (JSON 저장 제외)
        }
    except Exception as e:
        logger.error(f"❌ {os.path.basename(data_path)}: {e}")
        logger.debug(traceback.format_exc())
        return None


def _compute_portfolio_metrics(results: list, initial_cash: float, top_n: int = None) -> dict:
    """활성 60m 코인의 equity curve를 등가중 합산 → 포트폴리오 ROI/MDD/Sharpe.
    top_n: Sharpe 상위 N개만 사용 (orchestrator max_concurrent_bots 시뮬레이션)."""
    candidates = [r for r in results if r.get('equity_curve') and r.get('interval') == 'minute60']
    if top_n:
        candidates = sorted(candidates, key=lambda r: r.get('sharpe', 0), reverse=True)[:top_n]

    series_list = []
    for r in candidates:
        s = pd.Series({dt: val / initial_cash for dt, val in r['equity_curve']})
        series_list.append(s)

    if not series_list:
        return {}

    df = pd.concat(series_list, axis=1).sort_index().ffill()
    df = df.fillna(1.0)

    port = df.mean(axis=1)
    port_roi = (port.iloc[-1] - 1.0) * 100

    rolling_max = port.cummax()
    drawdown    = (port - rolling_max) / rolling_max * 100
    port_mdd    = float(abs(drawdown.min()))

    returns = port.pct_change().dropna()
    if len(returns) > 1 and returns.std() > 0:
        port_sharpe = float((returns.mean() / returns.std()) * np.sqrt(8760))
    else:
        port_sharpe = 0.0

    return {
        'roi':     round(port_roi,    2),
        'mdd':     round(port_mdd,    2),
        'sharpe':  round(port_sharpe, 2),
        'n_coins': len(series_list),
    }


def run_backtest(stress_mode: bool = False, start_date: str = None):
    """
    Args:
        stress_mode: True → stress_slippage 적용
        start_date:  OOS 시작일 수동 지정 (예: "2026-01-20"). None이면 각 모델의
                     train_cutoff_timestamp 자동 사용.
    카테고리(A/B/C)별 모델을 각각 로드해 코인별 라우팅 후 백테스트.
    """
    logger.info("=" * 70)
    logger.info("🚀 전체 코인 Backtrader 백테스팅 시작")
    if start_date:
        logger.info(f"📅 OOS 시작일 오버라이드: {start_date}")
    else:
        logger.info("📅 OOS 시작일: 각 모델 train_cutoff_timestamp 자동 적용")
    logger.info("=" * 70)

    categories = getattr(config, 'COIN_CATEGORIES', {})
    _mtf       = getattr(config, 'MTF_ENSEMBLE', {})
    mtf_enabled = _mtf.get('enabled', False)
    get_cat    = getattr(config, 'get_coin_category', lambda _: 'C')

    # 카테고리별 모델 로드: 60m / 15m 독립 로드 (60m 없어도 15m 사용 가능)
    cat_models: dict    = {}  # 60m 모델이 있는 카테고리
    cat_15m_only: dict  = {}  # 15m 전용 (60m 없는 카테고리)

    def _load_15m(cat_key: str, cat_info: dict):
        """15m 모델 로드 헬퍼 → (model_15m, feat_order_15m, threshold_15m, ml_config_15m)."""
        if not mtf_enabled:
            return None, None, 0.50, None
        dir_15m = cat_info.get('model_dir_15m', cat_info['model_dir'] + '_15m')
        if not os.path.isdir(dir_15m):
            return None, None, 0.50, None
        p15 = {
            'model':  (find_latest_file(dir_15m, '*ensemble_bot*.pkl')
                       or find_latest_file(dir_15m, '*xgb_bot*.pkl') or ''),
            'config': find_latest_file(dir_15m, 'config_*.json') or '',
        }
        if not (p15['model'] and p15['config']):
            logger.warning(f"⚠️ Model_{cat_key} 15m 파일 없음")
            return None, None, 0.50, None
        m15, f15, t15, cfg15 = _load_model_and_config(p15)
        fo15 = _extract_feature_order(m15, f15) if m15 else None
        if m15:
            _m15_base = os.path.basename(p15['model'])
            logger.info(f"  📐 MTF 15m Model_{cat_key}: {_m15_base}")
            _ts15 = re.search(r'_(\d{8}_\d{6})\.pkl$', _m15_base)
            if _ts15:
                _dt15 = datetime.strptime(_ts15.group(1), '%Y%m%d_%H%M%S')
                _age15 = (datetime.now() - _dt15).total_seconds() / 3600
                if _age15 > 12:
                    logger.warning(f"  ⚠️ MTF 15m Model_{cat_key} 구모델 fallback ({_age15:.0f}h 전) — 오늘 훈련 실패")
        return m15, fo15, t15, cfg15

    for cat_key, cat_info in categories.items():
        m15, fo15, t15, cfg15 = _load_15m(cat_key, cat_info)

        paths_cat = get_file_paths(model_dir=cat_info['model_dir'])
        if not paths_cat['model'] or not paths_cat['config']:
            if m15 is not None:
                cat_15m_only[cat_key] = dict(model_15m=m15, feat_order_15m=fo15,
                                              threshold_15m=t15, ml_config_15m=cfg15)
                logger.warning(f"⚠️ Model_{cat_key} 60m 없음 — 15m 단독 등록")
            else:
                logger.warning(f"⚠️ Model_{cat_key} 모델 없음 — 스킵")
            continue

        model, features, threshold, ml_config = _load_model_and_config(paths_cat)
        if model is None:
            continue
        feat_order = _extract_feature_order(model, features)
        _model_basename = os.path.basename(paths_cat['model'])
        logger.info(f"✅ Model_{cat_key}: {_model_basename}")
        _ts_m = re.search(r'_(\d{8}_\d{6})\.pkl$', _model_basename)
        if _ts_m:
            _model_dt = datetime.strptime(_ts_m.group(1), '%Y%m%d_%H%M%S')
            _age_h = (datetime.now() - _model_dt).total_seconds() / 3600
            if _age_h > 12:
                logger.warning(f"  ⚠️ Model_{cat_key} 구모델 fallback ({_age_h:.0f}h 전 학습) — 오늘 훈련 실패로 이전 모델 사용 중")

        cat_models[cat_key] = dict(
            model=model, feat_order=feat_order,
            threshold=threshold, ml_config=ml_config,
            model_15m=m15, feat_order_15m=fo15,
            threshold_15m=t15, ml_config_15m=cfg15,
        )

    if not cat_models:
        logger.error("❌ 카테고리 모델을 하나도 로드하지 못했습니다")
        return

    data_dir = config.DIRECTORIES['data_processed']
    processed_files = sorted(glob.glob(os.path.join(data_dir, '*.csv')), key=os.path.getmtime)
    if not processed_files:
        logger.error("❌ 처리된 데이터 파일을 찾을 수 없습니다")
        return

    # 4순위: 카테고리별 train_cutoff 중 가장 늦은 날짜로 공통 OOS 기간 정규화
    # → 모든 코인이 동일한 평가 구간을 사용해 ROI 직접 비교 가능
    if not start_date:
        cutoffs = [
            cm['ml_config'].get('train_cutoff_timestamp')
            for cm in cat_models.values()
            if cm.get('ml_config') and cm['ml_config'].get('train_cutoff_timestamp')
        ]
        if cutoffs:
            start_date = max(cutoffs)
            logger.info(f"📅 공통 OOS 시작일 (최신 train_cutoff): {start_date}")

    logger.info(f"📂 대상 파일: {len(processed_files)}개")

    INITIAL_CASH = float(config.BACKTEST['initial_cash'])
    max_workers  = min(4, len(processed_files))
    all_results  = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        for p in processed_files:
            fname = os.path.basename(p)
            ivl_m = re.search(r'_(minute\d+|days)(?:\.csv)?$', fname)
            ivl   = ivl_m.group(1) if ivl_m else 'minute60'

            # minute15·days: 단독 매매 없음 (15m은 60m AND게이트 전용) → 스킵
            if ivl != 'minute60':
                continue

            coin_m = re.search(r'KRW-([^_]+)', fname)
            coin   = coin_m.group(1) if coin_m else ''
            if coin in config.TRADING_BLACKLIST:
                continue
            cat    = get_cat(coin)
            cm     = cat_models.get(cat) or cat_models.get('C')
            if cm is None:
                continue

            # minute60: 60m 모델 + 15m MTF AND 게이트
            fut = pool.submit(
                _backtest_run_one, p,
                cm['model'], cm['feat_order'], cm['threshold'], cm['ml_config'], stress_mode,
                cm['model_15m'], cm['feat_order_15m'], cm['threshold_15m'],
                start_date,
            )
            futures[fut] = p

        for fut in as_completed(futures):
            r = fut.result()
            if r:
                all_results.append(r)

    if not all_results:
        logger.error("❌ 실행 가능한 코인이 없습니다")
        return

    all_results.sort(key=lambda x: x['roi'], reverse=True)

    # 차단 코인 목록 로드 (활성/관찰 분리용)
    _bl_file = getattr(config, 'PERFORMANCE_BLACKLIST_CRITERIA', {}).get(
        'blacklist_file', 'logs/performance_blacklist.json')
    _blacklisted_coins: set = set()
    if os.path.exists(_bl_file):
        try:
            with open(_bl_file, 'r', encoding='utf-8') as _f:
                _bl_data = json.load(_f)
            _blacklisted_coins = {k for k, v in _bl_data.items() if v.get('status') == 'blacklisted'}
        except Exception:
            pass
    _static_bl = set(getattr(config, 'TRADING_BLACKLIST', set()))

    def _coin_sym(r):
        return re.sub(r'_minute\d+$', '', r['coin'])

    active_results = [r for r in all_results if _coin_sym(r) not in _blacklisted_coins and _coin_sym(r) not in _static_bl]
    watch_results  = [r for r in all_results if _coin_sym(r) in _blacklisted_coins]

    def _avg_log(rows: list):
        if not rows:
            return
        return (
            round(np.mean([r['roi']    for r in rows]), 2),
            round(np.mean([r['mdd']    for r in rows]), 2),
            round(np.mean([r['sharpe'] for r in rows]), 2),
            sum(r['total_trades'] for r in rows),
        )

    logger.info("\n" + "=" * 70)
    logger.info("🏁 전체 코인 백테스트 결과 (ROI 순)")
    logger.info("=" * 70)

    if active_results:
        logger.info("  ── 활성 코인 ──")
        for r in active_results:
            logger.info(
                f"  [{_coin_sym(r):>6}] ROI={r['roi']:>7.1f}%  MDD={r['mdd']:>6.1f}%  "
                f"Sharpe={r['sharpe']:>5.2f}  거래={r['total_trades']:>4}회  승률={r['win_rate']:.1f}%"
            )
        a  = _avg_log(active_results)
        p  = _compute_portfolio_metrics(active_results, INITIAL_CASH)
        max_bots = getattr(config, 'ORCHESTRATOR', {}).get('max_concurrent_bots', 2)
        pt = _compute_portfolio_metrics(active_results, INITIAL_CASH, top_n=max_bots)
        logger.info("-" * 70)
        logger.info(f"  [활성평균]   ROI={a[0]:>7.1f}%  MDD={a[1]:>6.1f}%  Sharpe={a[2]:>5.2f}  총거래={a[3]}회")
        if p:
            logger.info(
                f"  [포트폴리오] ROI={p['roi']:>7.1f}%  MDD={p['mdd']:>6.1f}%  "
                f"Sharpe={p['sharpe']:>5.2f}  코인={p['n_coins']}개 (등가중)"
            )
        if pt and pt['n_coins'] < (p or {}).get('n_coins', 0):
            logger.info(
                f"  [실전Top{max_bots}]  ROI={pt['roi']:>7.1f}%  MDD={pt['mdd']:>6.1f}%  "
                f"Sharpe={pt['sharpe']:>5.2f}  코인={pt['n_coins']}개 (Sharpe상위{max_bots})"
            )

    if watch_results:
        logger.info("  ── 관찰 중 (차단) ──")
        for r in watch_results:
            logger.info(
                f"  [{_coin_sym(r):>6}] ROI={r['roi']:>7.1f}%  MDD={r['mdd']:>6.1f}%  "
                f"Sharpe={r['sharpe']:>5.2f}  거래={r['total_trades']:>4}회  승률={r['win_rate']:.1f}%"
            )

    avg_roi      = round(np.mean([r['roi']    for r in all_results]), 2)
    avg_mdd      = round(np.mean([r['mdd']    for r in all_results]), 2)
    avg_sharpe   = round(np.mean([r['sharpe'] for r in all_results]), 2)
    total_trades = sum(r['total_trades'] for r in all_results)
    logger.info("=" * 70)

    try:
        model_files = {k: os.path.basename(get_file_paths(model_dir=v['model_dir'])['model'] or '')
                       for k, v in categories.items() if k in cat_models}
        # equity_curve는 메모리 전용 — JSON 저장 전 제거
        per_coin_clean = [{k: v for k, v in r.items() if k != 'equity_curve'} for r in all_results]
        port = _compute_portfolio_metrics(active_results, INITIAL_CASH)
        summary = {
            'timestamp':                 datetime.now().isoformat(),
            'model_files':               model_files,
            'stress_mode':               stress_mode,
            'coin_count':                len(all_results),
            'initial_cash_per_coin':     INITIAL_CASH,
            'portfolio_capital_separated': True,
            'portfolio': port,
            'avg_roi':                   avg_roi,
            'avg_mdd':                   avg_mdd,
            'avg_sharpe':                avg_sharpe,
            'total_trades':              total_trades,
            'per_coin':                  per_coin_clean,
        }
        result_path = os.path.join(
            DIRECTORIES['logs'],
            f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        os.makedirs(DIRECTORIES['logs'], exist_ok=True)
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)
        logger.info(f"💾 결과 저장: {result_path}")
    except Exception as e:
        logger.error(f"❌ 결과 저장 오류: {e}")

    _update_performance_blacklist(all_results)

def _update_performance_blacklist(results: list) -> None:
    """백테스트 결과 누적 → 동적 성능 블랙리스트 갱신.
    연속 실패 or MDD 과대 코인을 자동 차단, 회복 시 자동 해제."""
    criteria = getattr(config, 'PERFORMANCE_BLACKLIST_CRITERIA', {})
    if not criteria.get('enabled', True):
        return

    sharpe_min     = criteria.get('sharpe_min', -1.0)
    sharpe_extreme = criteria.get('sharpe_extreme', -3.0)
    mdd_max        = criteria.get('mdd_max', 15.0)
    min_trades     = criteria.get('min_trades', 5)
    consec_fail    = criteria.get('consecutive_fail', 2)
    sharpe_rec     = criteria.get('sharpe_recover', 0.0)
    roi_rec        = criteria.get('roi_recover', 0.0)
    max_unblock    = criteria.get('max_unblock_per_cycle', 4)
    bl_file        = criteria.get('blacklist_file', 'logs/performance_blacklist.json')

    bl_data: dict = {}
    if os.path.exists(bl_file):
        try:
            with open(bl_file, 'r', encoding='utf-8') as f:
                bl_data = json.load(f)
        except Exception:
            bl_data = {}

    now_str = datetime.now().strftime('%Y-%m-%d %H:%M')
    added, removed = [], []
    recover_candidates: list = []   # (sharpe, roi, coin) — 루프 후 일괄 처리

    for r in results:
        if r.get('interval') != 'minute60':
            continue
        coin   = re.sub(r'_minute\d+$', '', r['coin'])
        trades = r.get('total_trades', 0)
        sharpe = r.get('sharpe', 0.0)
        mdd    = r.get('mdd', 0.0)
        roi    = r.get('roi', 0.0)
        immediate  = mdd > mdd_max or sharpe < sharpe_extreme

        # 거래 수 미달 → 극단값 즉시차단만 허용, 일반 필터 스킵
        if trades < min_trades:
            if immediate:
                entry = bl_data.setdefault(coin, {'consecutive_failures': 0, 'status': 'tracking'})
                if entry.get('status') != 'blacklisted':
                    reason = f"Sharpe={sharpe:.2f} 극단값 즉시차단(거래{trades}회)" if sharpe < sharpe_extreme else f"MDD={mdd:.1f}% 즉시차단(거래{trades}회)"
                    entry.update({'status': 'blacklisted', 'added_date': now_str, 'reason': reason,
                                  'last_sharpe': sharpe, 'last_mdd': mdd, 'last_roi': roi, 'last_updated': now_str})
                    added.append(f"{coin}({reason})")
            continue

        is_fail    = sharpe < sharpe_min or mdd > mdd_max
        # MDD 초과 중이면 회복 불가 (immediate 위험 지속)
        is_recover = sharpe >= sharpe_rec and roi >= roi_rec and not immediate

        if is_recover and coin in bl_data:
            recover_candidates.append((sharpe, roi, coin))

        elif is_fail:
            entry = bl_data.setdefault(coin, {'consecutive_failures': 0, 'status': 'tracking'})
            entry['consecutive_failures'] = entry.get('consecutive_failures', 0) + 1
            entry.update({'last_sharpe': sharpe, 'last_mdd': mdd, 'last_roi': roi, 'last_updated': now_str})

            if entry['status'] != 'blacklisted':
                if immediate or entry['consecutive_failures'] >= consec_fail:
                    if mdd > mdd_max:
                        reason = f"MDD={mdd:.1f}% 즉시차단"
                    elif sharpe < sharpe_extreme:
                        reason = f"Sharpe={sharpe:.2f} 극단값 즉시차단"
                    else:
                        reason = f"연속 {entry['consecutive_failures']}회 Sharpe={sharpe:.2f}"
                    entry['status']     = 'blacklisted'
                    entry['added_date'] = now_str
                    entry['reason']     = reason
                    added.append(f"{coin}({reason})")

        elif coin in bl_data and not is_fail:
            bl_data[coin]['consecutive_failures'] = 0
            bl_data[coin]['last_updated'] = now_str

    # Sharpe 상위 max_unblock 개: blacklisted→probation / probation→완전해제
    # 나머지: 실패 카운트 초기화 후 다음 사이클 대기
    recover_candidates.sort(key=lambda x: x[0], reverse=True)
    for sharpe_c, roi_c, coin in recover_candidates[:max_unblock]:
        entry = bl_data[coin]
        if entry.get('status') == 'probation':
            removed.append(f"{coin}(Sharpe={sharpe_c:.2f}, ROI={roi_c:.1f}% 관찰 완료→활성)")
            del bl_data[coin]
        else:
            entry.update({'status': 'probation', 'last_updated': now_str,
                          'last_sharpe': sharpe_c, 'last_roi': roi_c, 'consecutive_failures': 0})
            removed.append(f"{coin}(Sharpe={sharpe_c:.2f}, ROI={roi_c:.1f}% →관찰중)")
    for _, _, coin in recover_candidates[max_unblock:]:
        if coin in bl_data:
            bl_data[coin]['consecutive_failures'] = 0
            bl_data[coin]['last_updated'] = now_str

    os.makedirs(os.path.dirname(bl_file) or '.', exist_ok=True)
    with open(bl_file, 'w', encoding='utf-8') as f:
        json.dump(bl_data, f, indent=2, ensure_ascii=False)

    if added:
        logger.warning(f"🚫 성능 블랙리스트 차단: {', '.join(added)}")
    if removed:
        logger.info(f"✅ 성능 블랙리스트 해제: {', '.join(removed)}")
    active = [k for k, v in bl_data.items() if v.get('status') == 'blacklisted']
    if active:
        logger.info(f"📋 현재 차단 코인: {active}")


def _stress_run_one(
    data_path: str, model, features: list, threshold: float,
    slippage_mult: float, start: str, end: str,
    model_15m=None, feat_order_15m: list = None, threshold_15m: float = 0.50,
) -> "dict | None":
    """단일 코인 스트레스 백테스트 실행 — ThreadPoolExecutor 병렬 호출용."""
    try:
        df = pd.read_csv(data_path)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp').sort_index()

        df = df[(df.index >= start) & (df.index <= end)]
        if len(df) < 20:
            logger.warning(
                f"⚠️ {os.path.basename(data_path)}: 스트레스 구간 데이터 부족 ({len(df)}개)"
            )
            return None

        # BTC 레퍼런스 피처 자동 머지 (_backtest_run_one과 동일 로직)
        _btc_ref_needed = [f for f in features if f.startswith('BTC_') and f not in df.columns]
        if _btc_ref_needed:
            _data_dir = os.path.dirname(data_path)
            ivl_m = re.search(r'_(minute\d+|days)(?:\.csv)?$', os.path.basename(data_path))
            ivl   = ivl_m.group(1) if ivl_m else 'minute60'
            _btc_path = os.path.join(_data_dir, f'processed_KRW-BTC_{ivl}.csv')
            if os.path.exists(_btc_path):
                try:
                    _btc_df = pd.read_csv(_btc_path)
                    _btc_df['timestamp'] = pd.to_datetime(_btc_df['timestamp'])
                    _btc_df = _btc_df.set_index('timestamp').sort_index()
                    _btc_rename = {c: f'BTC_{c}' for c in _btc_df.columns if f'BTC_{c}' in _btc_ref_needed}
                    if _btc_rename:
                        _btc_sub = _btc_df[list(_btc_rename.keys())].rename(columns=_btc_rename)
                        df = df.join(_btc_sub, how='left')
                        df[list(_btc_rename.values())] = df[list(_btc_rename.values())].ffill()
                except Exception as _e:
                    logger.warning(f"⚠️ BTC 레퍼런스 피처 로드 실패: {_e}")

        missing_feats = [f for f in features if f not in df.columns]
        if missing_feats:
            logger.error(f"❌ {os.path.basename(data_path)} 피처 누락 — 0으로 대체: {missing_feats}")
            for f in missing_feats:
                df[f] = 0.0
        df = df.dropna(subset=features)
        if 'HMM_State' not in df.columns:
            df['HMM_State'] = 1.0
        if 'ATR_Ratio' not in df.columns:
            df['ATR_Ratio'] = 0.0
        if 'Macro_Trend_Up' not in df.columns:
            df['Macro_Trend_Up'] = 1.0

        pred_probs      = model.predict_proba(df[features])[:, 1]
        df['ML_Prob']   = pred_probs
        df['ML_Signal'] = (pred_probs >= threshold).astype(int)

        if model_15m is not None and feat_order_15m:
            try:
                data_dir = os.path.dirname(data_path)
                base_name = os.path.basename(data_path)
                path_15m = os.path.join(data_dir, re.sub(r'_minute\d+\.csv$', '_minute15.csv', base_name))
                if os.path.exists(path_15m):
                    df15 = pd.read_csv(path_15m)
                    if 'timestamp' in df15.columns:
                        df15['timestamp'] = pd.to_datetime(df15['timestamp'])
                        df15 = df15.set_index('timestamp').sort_index()
                    df15 = df15[(df15.index >= start) & (df15.index <= end)]
                    missing_15m = [f for f in feat_order_15m if f not in df15.columns]
                    for f in missing_15m:
                        df15[f] = 0.0
                    df15 = df15.dropna(subset=feat_order_15m)
                    if len(df15) > 0:
                        prob15 = model_15m.predict_proba(df15[feat_order_15m])[:, 1]
                        sig15  = pd.Series((prob15 >= threshold_15m).astype(int), index=df15.index)
                        sig15_resampled = sig15.reindex(df.index, method='ffill').fillna(0).astype(int)
                        df['ML_Signal'] = (df['ML_Signal'] & sig15_resampled).astype(int)
            except Exception as _e:
                logger.warning(f"⚠️ MTF 15m 스트레스 신호 생성 실패 ({_e}) — 60m 단독으로 진행")

        coin_match  = re.search(r'KRW-(\w+)', os.path.basename(data_path))
        coin        = coin_match.group(1) if coin_match else 'BTC'
        base_slip   = config.SLIPPAGE_BY_COIN.get(coin, config.SLIPPAGE_DEFAULT)
        stress_slip = min(base_slip * slippage_mult, 0.03)

        cerebro = _create_cerebro(MLSignalData(dataname=df), stress_slip)
        results  = cerebro.run()
        strat    = results[0]
        r        = _parse_cerebro_results(cerebro, strat, float(config.BACKTEST['initial_cash']))

        logger.info(
            f"  [{coin}] ROI={r['roi']:.1f}% MDD={r['mdd']:.1f}% "
            f"Sharpe={r['sharpe']:.2f} 거래={r['total_trades']}회"
        )
        return {
            "coin": coin, "start": start, "end": end,
            "interval": "minute60",
            "roi":    round(r['roi'],    2),
            "mdd":    round(r['mdd'],    2),
            "sharpe": round(r['sharpe'], 2),
            "trades": int(r['total_trades']),
            "stress_slippage": round(stress_slip * 100, 2),
            "equity_curve": strat.equity_curve,
        }
    except Exception as e:
        logger.error(f"❌ {os.path.basename(data_path)} 스트레스 테스트 실패: {e}")
        return None


def run_stress_period_backtest(start: str, end: str,
                               slippage_mult: float = 3.0,
                               label: str = "stress",
                               model_dir: str = None) -> dict:
    """
    역사적 위기 구간 집중 백테스트.
    일반 백테스트와 동일한 로직이지만 특정 날짜 범위만 잘라서 실행.

    Args:
        start: 시작 날짜 "YYYY-MM-DD"
        end:   종료 날짜 "YYYY-MM-DD"
        slippage_mult: 일반 슬리피지 대비 배수 (극단 스트레스 = 3배)
        label: 결과 파일 구분용 레이블

    Returns:
        dict: roi, mdd, sharpe, total_trades 등 결과 요약
    """
    logger.info(f"🔥 [{label}] 스트레스 구간 백테스트: {start} ~ {end} (슬리피지 ×{slippage_mult})")

    # 카테고리별 모델 로드 (run_backtest와 동일한 라우팅)
    categories  = getattr(config, 'COIN_CATEGORIES', {})
    _mtf        = getattr(config, 'MTF_ENSEMBLE', {})
    mtf_enabled = _mtf.get('enabled', False)
    get_cat     = getattr(config, 'get_coin_category', lambda _: 'C')
    cat_models: dict = {}

    for cat_key, cat_info in categories.items():
        _dir = cat_info.get('model_dir', model_dir or config.DIRECTORIES.get('models', 'models'))
        _paths = get_file_paths(model_dir=_dir)
        if not _paths['model'] or not _paths['config']:
            logger.warning(f"⚠️ Model_{cat_key} 없음 — 스트레스 스킵")
            continue
        _model, _feats, _thresh, _ = _load_model_and_config(_paths)
        if _model is None:
            continue
        _fo = _extract_feature_order(_model, _feats)
        _m15 = None
        _fo15 = None
        _t15 = 0.50
        if mtf_enabled:
            _dir15 = cat_info.get('model_dir_15m', _dir + '_15m')
            if os.path.isdir(_dir15):
                _p15 = {'model': (find_latest_file(_dir15, '*ensemble_bot*.pkl')
                                  or find_latest_file(_dir15, '*xgb_bot*.pkl') or ''),
                        'config': find_latest_file(_dir15, 'config_*.json') or ''}
                if _p15['model'] and _p15['config']:
                    _m15, _f15, _t15, _ = _load_model_and_config(_p15)
                    _fo15 = _extract_feature_order(_m15, _f15) if _m15 else None
        cat_models[cat_key] = dict(model=_model, feat_order=_fo, threshold=_thresh,
                                   model_15m=_m15, feat_order_15m=_fo15, threshold_15m=_t15)

    if not cat_models:
        logger.error("❌ 카테고리 모델 로드 실패")
        return {}

    data_dir = config.DIRECTORIES['data_processed']
    processed_files = sorted(glob.glob(os.path.join(data_dir, '*.csv')), key=os.path.getmtime)

    all_results = []
    max_workers = min(4, len(processed_files))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        for p in processed_files:
            ivl_m = re.search(r'_(minute\d+|days)(?:\.csv)?$', os.path.basename(p))
            ivl   = ivl_m.group(1) if ivl_m else None
            if ivl != 'minute60':
                continue
            coin_m = re.search(r'KRW-([^_]+)', os.path.basename(p))
            coin   = coin_m.group(1) if coin_m else ''
            if coin in config.TRADING_BLACKLIST:
                continue
            cm = cat_models.get(get_cat(coin)) or cat_models.get('C')
            if cm is None:
                continue
            futures[pool.submit(
                _stress_run_one, p,
                cm['model'], cm['feat_order'], cm['threshold'],
                slippage_mult, start, end,
                cm['model_15m'], cm['feat_order_15m'], cm['threshold_15m'],
            )] = p
        for fut in as_completed(futures):
            r = fut.result()
            if r:
                all_results.append(r)

    if not all_results:
        return {}

    INITIAL_CASH = float(config.BACKTEST['initial_cash'])
    port = _compute_portfolio_metrics(all_results, INITIAL_CASH)
    max_bots = getattr(config, 'ORCHESTRATOR', {}).get('max_concurrent_bots', 2)
    port_top = _compute_portfolio_metrics(all_results, INITIAL_CASH, top_n=max_bots)

    per_coin_clean = [{k: v for k, v in r.items() if k != 'equity_curve'} for r in all_results]
    summary = {
        "label": label, "start": start, "end": end,
        "portfolio_capital_separated": True,
        "portfolio": port,
        f"portfolio_top{max_bots}": port_top,
        "avg_roi":     round(np.mean([r["roi"]    for r in all_results]), 2),
        "avg_mdd":     round(np.mean([r["mdd"]    for r in all_results]), 2),
        "avg_sharpe":  round(np.mean([r["sharpe"] for r in all_results]), 2),
        "total_trades": sum(r["trades"] for r in all_results),
        "per_coin": per_coin_clean,
    }
    logger.info(
        f"🏁 [{label}] 스트레스 요약  ROI={summary['avg_roi']:.1f}%  "
        f"MDD={summary['avg_mdd']:.1f}%  Sharpe={summary['avg_sharpe']:.2f}  "
        f"거래={summary['total_trades']}회"
    )
    if port:
        logger.info(
            f"  [포트폴리오] ROI={port['roi']:.1f}%  MDD={port['mdd']:.1f}%  "
            f"Sharpe={port['sharpe']:.2f}  코인={port['n_coins']}개"
        )
    if port_top and port_top['n_coins'] < (port or {}).get('n_coins', 0):
        logger.info(
            f"  [실전Top{max_bots}]  ROI={port_top['roi']:.1f}%  MDD={port_top['mdd']:.1f}%  "
            f"Sharpe={port_top['sharpe']:.2f}  코인={port_top['n_coins']}개"
        )

    try:
        result_path = os.path.join(
            DIRECTORIES['logs'],
            f"stress_{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        os.makedirs(DIRECTORIES['logs'], exist_ok=True)
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)
        logger.info(f"💾 스트레스 결과 저장: {result_path}")
    except Exception as e:
        logger.error(f"❌ 스트레스 결과 저장 실패: {e}")
    return summary


def run_rolling_window_backtest(
    train_months: int = 3,
    test_months: int = 1,
    max_windows: int = 6,
    retrain_weights: bool = False,
) -> dict:
    """
    Rolling Window Walk-Forward 백테스트.
    단일 기간 백테스트의 특정 국면 과적합 문제를 보완한다.

    방법:
        [기본 권장] Threshold Optimization: 가중치는 전역 모델 유지, 창별 임계값만 재보정
                   — 속도 우수 (창당 <1분)
        [심화 분석] Full WFO: machine_learning.main()을 창별 호출하여 가중치 완전 재학습
                   — 정확도 우수하나 시간 소요 (창당 10~30분) → retrain_weights=True

    Args:
        train_months: 훈련 창 크기 (월)
        test_months:  검증 창 크기 (월)
        max_windows:  최대 슬라이딩 횟수 (데이터 부족 시 자동 축소)
        retrain_weights: True→가중치 재학습 (Full WFO, 10~30분/창)
                        False→임계값만 재보정 (기본 권장)

    Returns:
        dict: 창별 결과 및 집계 통계 (avg_roi, avg_mdd, avg_sharpe, win_rate)
             ⚠️ "avg_*" 지표는 각 코인이 동일 자본으로 독립 실행되었으므로
                포트폴리오 레벨 결과 아님 (참고용)
    """
    # 행 수가 가장 많은 코인 파일을 선택한 뒤 해당 코인의 카테고리 모델 로드
    _data_dir = config.DIRECTORIES['data_processed']
    _all_files = sorted(glob.glob(os.path.join(_data_dir, '*_minute60.csv')), key=os.path.getmtime)
    if not _all_files:
        logger.error("❌ 롤링 윈도우 백테스트: 처리된 데이터 없음")
        return {}
    _rep_file = max(_all_files, key=_row_count)
    _coin_m   = re.search(r'KRW-(\w+)', os.path.basename(_rep_file))
    _rep_coin = _coin_m.group(1) if _coin_m else 'BTC'
    _get_cat  = getattr(config, 'get_coin_category', lambda _: 'C')
    _cat      = _get_cat(_rep_coin)
    _cats     = getattr(config, 'COIN_CATEGORIES', {})
    _cat_info = _cats.get(_cat, {})
    _model_dir = _cat_info.get('model_dir', config.DIRECTORIES.get('models', 'models'))
    paths = get_file_paths(model_dir=_model_dir)
    paths['processed_files'] = _all_files
    if not paths['model'] or not paths['config']:
        logger.error(f"❌ 롤링 윈도우 백테스트: Model_{_cat} 없음")
        return {}
    logger.info(f"📌 대표 코인: {_rep_coin} (Model_{_cat})")

    model, features, threshold, _ = _load_model_and_config(paths)
    if model is None:
        return {}
    features = _extract_feature_order(model, features)

    # MTF 15m 모델 로드
    model_15m = feat_order_15m = df_15m_full = None
    threshold_15m = 0.50
    _mtf = getattr(config, "MTF_ENSEMBLE", {})
    if _mtf.get("enabled", False):
        model_dir_60m = os.path.dirname(paths['model'])
        model_dir_15m = model_dir_60m + "_15m"
        if os.path.isdir(model_dir_15m):
            _paths_15m = {
                'model':  (find_latest_file(model_dir_15m, "*ensemble_bot*.pkl")
                           or find_latest_file(model_dir_15m, "*xgb_bot*.pkl") or ""),
                'config': find_latest_file(model_dir_15m, "config_*.json") or "",
            }
            if _paths_15m['model'] and _paths_15m['config']:
                model_15m, feats_15m, threshold_15m, _ = _load_model_and_config(_paths_15m)
                feat_order_15m = _extract_feature_order(model_15m, feats_15m) if model_15m else None

    data_path = _rep_file
    window_slippage = config.SLIPPAGE_BY_COIN.get(_rep_coin, config.SLIPPAGE_DEFAULT)
    logger.info(f"📌 대표 코인: {_rep_coin} (Model_{_cat}) | 슬리피지: {window_slippage*100:.2f}%")

    # 15m 전체 데이터 로드 (MTF AND 게이트용)
    if model_15m is not None and feat_order_15m:
        try:
            _data_dir = os.path.dirname(data_path)
            _base     = os.path.basename(data_path)
            _path_15m = os.path.join(_data_dir, re.sub(r'_minute\d+\.csv$', '_minute15.csv', _base))
            if os.path.exists(_path_15m):
                df_15m_full = pd.read_csv(_path_15m)
                if 'timestamp' in df_15m_full.columns:
                    df_15m_full['timestamp'] = pd.to_datetime(df_15m_full['timestamp'])
                    df_15m_full = df_15m_full.set_index('timestamp').sort_index()
                logger.debug(f"MTF 15m 데이터 로드: {os.path.basename(_path_15m)}")
        except Exception as _e:
            logger.warning(f"⚠️ MTF 15m 데이터 로드 실패 ({_e}) — AND 게이트 비활성화")
            model_15m = None

    try:
        df_full = pd.read_csv(data_path)
        if 'timestamp' in df_full.columns:
            df_full['timestamp'] = pd.to_datetime(df_full['timestamp'])
            df_full = df_full.set_index('timestamp').sort_index()
    except Exception as e:
        logger.error(f"❌ 데이터 로드 실패: {e}")
        return {}

    # BTC 레퍼런스 피처 자동 머지 (B/C 카테고리 모델)
    _btc_ref_needed = [f for f in features if f.startswith('BTC_') and f not in df_full.columns]
    if _btc_ref_needed:
        _data_dir = os.path.dirname(data_path)
        ivl_m = re.search(r'_(minute\d+|days)(?:\.csv)?$', os.path.basename(data_path))
        ivl   = ivl_m.group(1) if ivl_m else 'minute60'
        _btc_path = os.path.join(_data_dir, f'processed_KRW-BTC_{ivl}.csv')
        if os.path.exists(_btc_path):
            try:
                _btc_df = pd.read_csv(_btc_path)
                _btc_df['timestamp'] = pd.to_datetime(_btc_df['timestamp'])
                _btc_df = _btc_df.set_index('timestamp').sort_index()
                _btc_rename = {c: f'BTC_{c}' for c in _btc_df.columns if f'BTC_{c}' in _btc_ref_needed}
                if _btc_rename:
                    _btc_sub = _btc_df[list(_btc_rename.keys())].rename(columns=_btc_rename)
                    df_full = df_full.join(_btc_sub, how='left')
                    df_full[list(_btc_rename.values())] = df_full[list(_btc_rename.values())].ffill()
                    logger.info(f"📎 BTC 레퍼런스 피처 {len(_btc_rename)}개 머지 완료")
            except Exception as _e:
                logger.warning(f"⚠️ BTC 레퍼런스 피처 로드 실패: {_e}")
        _still_missing = [f for f in _btc_ref_needed if f not in df_full.columns]
        if _still_missing:
            logger.warning(f"⚠️ BTC 피처 {len(_still_missing)}개 여전히 누락 — 0으로 대체")
            for f in _still_missing:
                df_full[f] = 0.0

    df_full = df_full.dropna(subset=features)
    if 'HMM_State' not in df_full.columns:
        df_full['HMM_State'] = 1.0
    if 'ATR_Ratio' not in df_full.columns:
        df_full['ATR_Ratio'] = 0.0
    if 'Macro_Trend_Up' not in df_full.columns:
        df_full['Macro_Trend_Up'] = 1.0

    total_days  = (df_full.index[-1] - df_full.index[0]).days
    window_days = train_months * 30 + test_months * 30
    if total_days < window_days:
        logger.error(
            f"❌ 데이터 기간 부족 ({total_days}일 < 필요 {window_days}일)"
        )
        return {}

    step_days = test_months * 30
    n_windows = min(max_windows, (total_days - window_days) // step_days + 1)

    logger.info(
        f"\n{'='*70}\n"
        f"🔄 Rolling Window 백테스트: {n_windows}개 창 "
        f"(훈련 {train_months}개월 / 검증 {test_months}개월)\n"
        f"{'='*70}"
    )

    window_results = []
    start_date = df_full.index[0]

    for w in range(n_windows):
        train_start = start_date + pd.DateOffset(days=w * step_days)
        train_end   = train_start + pd.DateOffset(months=train_months)
        test_start  = train_end
        test_end    = test_start + pd.DateOffset(months=test_months)

        df_train = df_full[(df_full.index >= train_start) & (df_full.index < train_end)]
        df_test  = df_full[(df_full.index >= test_start)  & (df_full.index < test_end)]
        if len(df_test) < 20:
            logger.warning(f"  창 {w+1}: 검증 데이터 부족 ({len(df_test)}개) — 스킵")
            continue

        try:
            # ========== 기본: Threshold Optimization Only (권장) ==========
            # 모델 가중치는 전역값 유지, 창별로 임계값만 재보정 (속도: 창당 <1분)
            # ========== 고급: Full WFO 재학습 (retrain_weights=True) ==========
            # 창별 training data로 machine_learning.main() 호출하여 가중치 재학습
            # (속도: 창당 10~30분, 극도로 정확한 분석 필요 시만 권장)
            
            if retrain_weights:
                # TODO: machine_learning.py의 main() 호출 후 신규 모델·임계값 반환
                # model_window, threshold_window = retrain_model_on_window(df_train, features)
                # 현재 미구현 — 운영 환경에서는 임계값 재보정만 하는 기본 모드 권장
                logger.warning(
                    f"  창 {w+1}: retrain_weights=True 설정되었으나 미구현. "
                    f"기본 Threshold Optimization 모드로 진행합니다."
                )
            
            window_threshold = threshold  # 전역 threshold 폴백
            label_col = next((c for c in df_train.columns if c.lower() == "label"), None)
            if label_col and len(df_train) >= 50:
                try:
                    from sklearn.metrics import precision_recall_curve
                    tr_probs  = model.predict_proba(df_train[features])[:, 1]
                    tr_labels = df_train[label_col]
                    if tr_labels.nunique() >= 2:
                        prec, rec, threshs = precision_recall_curve(tr_labels, tr_probs)
                        min_prec = config.MODEL_MANAGEMENT.get("min_precision", 0.60)
                        valid = prec[:-1] >= min_prec
                        if valid.any():
                            best_idx = int(np.argmax(rec[:-1] * valid))
                            window_threshold = float(threshs[best_idx])
                            logger.debug(
                                f"  창 {w+1}: 임계값 재보정 "
                                f"{threshold:.3f} → {window_threshold:.3f}"
                            )
                except Exception as _e:
                    logger.debug(f"  창 {w+1}: 임계값 재보정 실패 ({_e}), 전역값 사용")

            pred_probs = model.predict_proba(df_test[features])[:, 1]
            df_test = df_test.copy()
            df_test['ML_Prob']   = pred_probs
            df_test['ML_Signal'] = (pred_probs >= window_threshold).astype(int)

            # MTF AND 게이트: 15m 신호를 60m 인덱스로 ffill하여 AND 적용
            if model_15m is not None and feat_order_15m and df_15m_full is not None:
                try:
                    df15_win = df_15m_full[
                        (df_15m_full.index >= test_start) & (df_15m_full.index < test_end)
                    ]
                    missing_15m = [f for f in feat_order_15m if f not in df15_win.columns]
                    for f in missing_15m:
                        df15_win[f] = 0.0
                    df15_win = df15_win.dropna(subset=feat_order_15m)
                    if len(df15_win) >= 10:
                        prob15 = model_15m.predict_proba(df15_win[feat_order_15m])[:, 1]
                        sig15  = pd.Series((prob15 >= threshold_15m).astype(int), index=df15_win.index)
                        sig15_resampled = sig15.reindex(df_test.index, method='ffill').fillna(0).astype(int)
                        df_test['ML_Signal'] = (df_test['ML_Signal'] & sig15_resampled).astype(int)
                except Exception as _e:
                    logger.debug(f"  창 {w+1}: MTF AND 실패 ({_e}) — 60m 단독 진행")

            INITIAL_CASH = float(config.BACKTEST['initial_cash'])
            cerebro = _create_cerebro(
                MLSignalData(dataname=df_test),
                window_slippage,
            )

            results = cerebro.run()
            strat = results[0]

            end_val = cerebro.broker.getvalue()
            roi     = (end_val / INITIAL_CASH - 1) * 100
            _r      = _parse_cerebro_results(cerebro, strat, INITIAL_CASH)
            mdd     = _r['mdd']
            sharpe  = _r['sharpe']
            n_total = _r['total_trades']
            win_rate = _r['win_rate']

            window_results.append({
                "window": w + 1,
                "test_start": str(test_start.date()),
                "test_end":   str(test_end.date()),
                "roi": round(roi, 2),
                "mdd": round(mdd, 2),
                "sharpe": round(sharpe, 2),
                "trades": n_total,
                "win_rate": round(win_rate, 1),
            })
            logger.info(
                f"  창 {w+1} [{test_start.date()} ~ {test_end.date()}]: "
                f"ROI={roi:.1f}% MDD={mdd:.1f}% Sharpe={sharpe:.2f} "
                f"거래={n_total}회 승률={win_rate:.0f}%"
            )
        except Exception as e:
            logger.error(f"  창 {w+1} 실패: {e}")

    if not window_results:
        logger.error("❌ 유효한 검증 창이 없습니다")
        return {}

    # ⚠️ 포트폴리오 설계 제한: 각 코인이 동일 초기자본으로 독립 실행되므로
    # avg_roi/avg_mdd/avg_sharpe는 "단순 산술평균"이며, 실제 포트폴리오 성과 아님
    # (참고용으로만 사용 권장, 가중치 있는 평균 필요 시 외부 계산 필요)
    summary = {
        "strategy":      "rolling_window",
        "wfo_mode":      "threshold_optimization" if not retrain_weights else "full_wfo",
        "train_months":  train_months,
        "test_months":   test_months,
        "n_windows":     len(window_results),
        "portfolio_capital_separated": False,  # 설계 제한 메타데이터
        "capital_per_coin": float(config.BACKTEST['initial_cash']),
        "avg_roi":       round(np.mean([r["roi"]   for r in window_results]), 2),
        "avg_mdd":       round(np.mean([r["mdd"]   for r in window_results]), 2),
        "avg_sharpe":    round(np.mean([r["sharpe"] for r in window_results]), 2),
        "avg_win_rate":  round(np.mean([r["win_rate"] for r in window_results]), 1),
        "total_trades":  sum(r["trades"] for r in window_results),
        "profitable_windows": sum(1 for r in window_results if r["roi"] > 0),
        "windows":       window_results,
        "timestamp":     datetime.now().isoformat(),
    }

    wfo_label = "[Threshold Optimization]" if not retrain_weights else "[Full WFO]"
    logger.info(
        f"\n{'='*70}\n"
        f"🏁 Rolling Window 집계 {wfo_label} ({len(window_results)}개 창)\n"
        f"   평균 ROI   : {summary['avg_roi']:.2f}% (⚠️ 참고용)\n"
        f"   평균 MDD   : {summary['avg_mdd']:.2f}% (⚠️ 참고용)\n"
        f"   평균 Sharpe: {summary['avg_sharpe']:.2f} (⚠️ 참고용)\n"
        f"   평균 승률  : {summary['avg_win_rate']:.1f}%\n"
        f"   수익 창 비율: {summary['profitable_windows']}/{len(window_results)}\n"
        f"   (각 코인 독립 자본 {float(config.BACKTEST['initial_cash']):,.0f}원 → 포트폴리오 아님)\n"
        f"{'='*70}"
    )

    try:
        result_path = os.path.join(
            DIRECTORIES['logs'],
            f"rolling_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        )
        os.makedirs(DIRECTORIES['logs'], exist_ok=True)
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)
        logger.info(f"💾 Rolling Window 결과 저장: {result_path}")
    except Exception as e:
        logger.error(f"❌ Rolling Window 결과 저장 실패: {e}")
    return summary


if __name__ == '__main__':
    run_backtest()