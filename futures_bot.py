"""
futures_bot.py
Coin AI Bot - 선물 시장 실행기 (Futures Executor)
- Binance Futures 기반 (CCXT Async)
- 필수 안전장치 하드코딩: 무조건 격리 마진(ISOLATED), 레버리지 최대 3배 통제
- 양방향(Long/Short) 수익 창출: HMM 국면에 따른 유동적 숏 포지션 진입
- 통합 두뇌 구독: 기존 market_context, AI 킬 스위치 100% 연동
"""

import os
import sys
import time
import json
import joblib
import logging
import asyncio
import requests
import numpy as np
import pandas as pd
import ccxt.async_support as ccxt
from datetime import datetime, timezone
from typing import Optional
from dotenv import load_dotenv

# 기존 현물의 두뇌(Brain)와 피처 엔지니어링 모듈 재사용
from feature_engineering import add_technical_indicators
from market_context import (
    BinanceFuturesContext,
    FearGreedIndex,
    HMMRegimeDetector,
    KMeansRegimeDetector,
    NewsSentimentAnalyzer,

)
import config as bot_config

# ============================================================================
# 필수 환경 변수 및 하드코딩 상수 (안전장치)
# ============================================================================
load_dotenv()
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET = os.getenv("BINANCE_SECRET_KEY")

# 🚨 [안전장치 1] 레버리지 및 마진 하드코딩 통제
MAX_LEVERAGE = 3                 # 절대 3배 초과 금지
MARGIN_MODE = 'ISOLATED'         # 절대 교차(CROSS) 마진 허용 안 함
HEDGE_MODE = False               # 단방향(One-way) 모드 강제

# 트레이딩 파라미터 (현물 시스템 로직 계승)
DEFAULT_THRESHOLD = 0.65
# config 단일 소스: atr_tp_mult / atr_sl_mult 비율로 이론 손익비 계산
# (하드코딩 1.33은 config 기준 2.5/1.2 = 2.08과 불일치 → 과소 포지션 진입 오류)
KELLY_PAYOFF_RATIO = (
    float(bot_config.LABELING["atr_tp_mult"]) /
    float(bot_config.LABELING["atr_sl_mult"])
)
ATR_RISK_PCT = float(bot_config.BACKTEST.get("atr_risk_pct", 0.02))  # config 단일 소스
MIN_POSITION_PCT = 0.05
MAX_POSITION_PCT = 0.40
_TARGET_VOL = 0.02               # 동적 레버리지 산출 기준 목표 변동성 (ATR 역수 스케일)
SHORT_MAX_UP_PROB = float(bot_config.BACKTEST.get("short_max_up_prob", 0.20))

MODEL_DIR  = bot_config.DIRECTORIES["models"]
_DATA_DIR  = bot_config.DIRECTORIES.get("data", "data")
_STATE_PATH = os.path.join(_DATA_DIR, "futures_bot_state.json")

TIMEFRAME_MAP = {
    'minute15': '15m',
    'minute60': '1h',
    'day': '1d'
}

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger("FuturesBot")

# ============================================================================
# 비동기 선물 트레이딩 봇 클래스
# ============================================================================
class AsyncFuturesBot:
    def __init__(self, symbol: str = "BTC/USDT", leverage: int = 2, dry_run: bool = False,
                 paper_balance_usdt: float = 10000.0):
        self.symbol = symbol
        self.leverage = min(leverage, MAX_LEVERAGE)  # 🚨 [안전장치 2] 요청 레버리지가 높아도 MAX로 클램핑
        self.dry_run = dry_run
        self._paper_balance = paper_balance_usdt
        self.is_running = True
        self.markets_loaded = False
        # CCXT 비동기 거래소 초기화 — binanceusdm: USDS-M 무기한 선물 전용 인스턴스
        # (ccxt.binance + defaultType:'future' 대비 심볼 처리·마진 계산이 더 정확함)
        # dry_run 시 공개 API만 사용하므로 키 없이 초기화
        self.exchange = ccxt.binanceusdm({
            'apiKey': BINANCE_API_KEY if not dry_run else '',
            'secret': BINANCE_SECRET if not dry_run else '',
            'enableRateLimit': True,
            'options': {'positionMode': HEDGE_MODE},
        })
        
        # 상태 관리
        self.position_side = None  # 'LONG', 'SHORT', or None
        self.position_amt = 0.0
        self.entry_price = 0.0
        self.last_atr_ratio = 0.0
        self._trailing_high = 0.0
        self._trailing_low = 0.0        # SHORT 소프트웨어 TS용
        self._trailing_stop_order_id: str = None  # 네이티브 TS 주문 ID 추적
        self._state_path = _STATE_PATH

        # 틱사이즈 자동 갱신 타이머
        _fr_cfg = bot_config.FUTURES_RISK
        self._exchange_info_refresh_interval: int = int(
            _fr_cfg.get("exchange_info_refresh_interval", 3600)
        )
        self._last_exchange_info_refresh: float = 0.0
        self._use_native_trailing_stop: bool = bool(
            _fr_cfg.get("use_native_trailing_stop", True)
        )
        self._trailing_callback_rate: float = float(
            _fr_cfg.get("trailing_stop_callback_rate", 2.0)
        )
        # 서킷 브레이커 (일일 손실 한도)
        self._daily_loss_limit: float = float(_fr_cfg.get("daily_loss_limit_pct", 0.03))
        self._daily_realized_pnl: float = 0.0
        self._circuit_breaker_date: str = ""
        self._circuit_breaker_active: bool = False
        # Telegram 실시간 알림
        self._tg_token: str = os.getenv("TELEGRAM_TOKEN", "")
        self._tg_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")

        # 기존 AI 봇 설정 파일 및 모델 로드
        self._load_brain()
        self._load_state()  # 크래시 복원: 이전 포지션 상태 복구

        # 시장 컨텍스트 (Universal Brain) 초기화
        _hmm_cfg = bot_config.HMM_REGIME
        self.binance_ctx = BinanceFuturesContext()
        self.fear_greed = FearGreedIndex()
        self.hmm = HMMRegimeDetector(
            n_states=3,
            lookback=_hmm_cfg.get("lookback", 500),
            retrain_interval=_hmm_cfg.get("retrain_interval", 3600),
        )
        # 주의: NewsSentimentAnalyzer는 이름과 달리 NLP/감성분석이 아니라
        # 'BTC 급락 킬 스위치' 역할을 한다. market_context.py 결함 목록 참조.
        self.news_analyzer = NewsSentimentAnalyzer(cache_ttl=300)
        # K-Means 보조 국면 감지기 (고변동성 횡보장 추세 신호 차단)
        self.kmeans_regime = KMeansRegimeDetector(n_clusters=3, lookback=200)

    def _load_brain(self):
        """
        머신러닝 모델 로드 — 우선순위:
        1) models_futures/ 선물 전용 모델 (futures_bot_*.pkl)
        2) models/ 앙상블 모델 (ensemble_bot_*.pkl)
        3) models/ XGBoost 단독 (xgb_bot_*.pkl) — 폴백 (현물 학습, 정밀도 저하 경고)
        """
        import glob
        FUTURES_MODEL_DIR = bot_config.DIRECTORIES.get("models_futures", "models_futures")
        os.makedirs(FUTURES_MODEL_DIR, exist_ok=True)

        futures_pkls = glob.glob(f"{FUTURES_MODEL_DIR}/*futures_bot*.pkl")
        futures_cfgs = glob.glob(f"{FUTURES_MODEL_DIR}/config_*.json")

        if futures_pkls and futures_cfgs:
            model_files  = futures_pkls
            config_files = futures_cfgs
            logger.info("✅ 선물 전용 모델 로드 (models_futures/)")
        else:
            model_files = (
                glob.glob(f"{MODEL_DIR}/*ensemble_bot*.pkl")
                or glob.glob(f"{MODEL_DIR}/*xgb_bot*.pkl")
            )
            config_files = glob.glob(f"{MODEL_DIR}/config_*.json")
            logger.warning(
                "⚠️ 선물 전용 모델 없음 — 현물 학습 모델 폴백 사용. "
                "run_futures_pipeline()으로 선물 모델 훈련 권장."
            )
        
        if not model_files or not config_files:
            logger.critical("❌ 모델이나 설정 파일을 찾을 수 없습니다.")
            sys.exit(1)
            
        model_path = max(model_files, key=os.path.getctime)
        config_path = max(config_files, key=os.path.getctime)
        
        self.model = joblib.load(model_path)
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
            
        self.features = self.config['features']
        self.interval = self.config.get('interval', 'minute15')
        self.ccxt_timeframe = TIMEFRAME_MAP.get(self.interval, '15m')
        self.threshold = self.config.get('best_threshold', DEFAULT_THRESHOLD)

    def _save_state(self):
        """포지션 상태를 JSON 파일에 원자적으로 저장 (크래시 복원용)."""
        try:
            os.makedirs(os.path.dirname(self._state_path), exist_ok=True)
            state = {
                "position_side": self.position_side,
                "position_amt": self.position_amt,
                "entry_price": self.entry_price,
                "trailing_high": self._trailing_high,
                "trailing_low": self._trailing_low,
            }
            tmp = self._state_path + ".tmp"
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(state, f)
            os.replace(tmp, self._state_path)
        except Exception as e:
            logger.warning(f"상태 저장 실패: {e}")

    def _load_state(self):
        """이전 실행에서 저장된 포지션 상태를 복원."""
        if not os.path.exists(self._state_path):
            return
        try:
            with open(self._state_path, 'r', encoding='utf-8') as f:
                state = json.load(f)
            self.position_side  = state.get("position_side")
            self.position_amt   = float(state.get("position_amt", 0.0))
            self.entry_price    = float(state.get("entry_price", 0.0))
            self._trailing_high = float(state.get("trailing_high", 0.0))
            self._trailing_low  = float(state.get("trailing_low", 0.0))
            logger.info(
                f"🔄 상태 복원: {self.position_side} | "
                f"수량={self.position_amt} | 평단가={self.entry_price:,.2f}"
            )
        except Exception as e:
            logger.warning(f"상태 로드 실패 (초기화 상태로 진행): {e}")

    async def _refresh_exchange_info_if_needed(self) -> None:
        """
        바이낸스 틱사이즈·수량단위 동적 갱신.
        2026년 3~4월 다수 심볼의 틱사이즈가 변경됨 — 하드코딩 금지.
        exchange_info_refresh_interval(기본 1시간) 경과 시 load_markets(reload=True) 호출.
        """
        now = time.time()
        if now - self._last_exchange_info_refresh < self._exchange_info_refresh_interval:
            return
        try:
            await self.exchange.load_markets(reload=True)
            self._last_exchange_info_refresh = now
            self.markets_loaded = True
            logger.info("🔄 거래소 심볼 정보(틱사이즈·수량단위) 갱신 완료")
        except Exception as e:
            logger.warning(f"거래소 정보 갱신 실패 (기존 캐시 유지): {e}")

    async def _apply_hardcoded_safety_measures(self):
        """🚨 [안전장치 3] 거래소 세팅 강제 초기화 (격리 마진 & 레버리지 통제)"""
        if self.dry_run:
            logger.info(f"[PAPER] 안전장치 설정 스킵 (dry_run) — 레버리지={self.leverage}x, 마진={MARGIN_MODE}")
            return
        try:
            # 1. 격리 마진 세팅
            try:
                await self.exchange.fapiprivate_post_margintype({
                    'symbol': self.symbol.replace('/', ''),
                    'marginType': MARGIN_MODE
                })
                logger.info(f"🛡️ 마진 모드 {MARGIN_MODE} 설정 완료")
            except Exception as e:
                if 'No need to change margin type' in str(e):
                    logger.info(f"🛡️ 마진 모드 이미 {MARGIN_MODE} 상태입니다.")
                else:
                    raise e
                    
            # 2. 레버리지 세팅
            await self.exchange.fapiprivate_post_leverage({
                'symbol': self.symbol.replace('/', ''),
                'leverage': self.leverage
            })
            logger.info(f"🛡️ 레버리지 {self.leverage}x 설정 완료 (안전 하드코딩 통과)")
            
        except Exception as e:
            logger.critical(f"❌ 안전장치 설정 실패! 거래를 중단합니다: {e}")
            await self.exchange.close()
            sys.exit(1)

    async def _fetch_position(self):
        """현재 포지션 상태 동기화"""
        if self.dry_run:
            state = self.position_side if self.position_side else "미보유"
            logger.info(f"[PAPER] 포지션: {state} | 수량: {self.position_amt} | 평단가: {self.entry_price:,.2f} | 잔고: {self._paper_balance:,.2f} USDT")
            return
        try:
            positions = await self.exchange.fetch_positions([self.symbol])
            for pos in positions:
                amt = float(pos['info']['positionAmt'])
                self.position_amt = amt
                self.entry_price = float(pos['info']['entryPrice'])

                if amt > 0:
                    self.position_side = 'LONG'
                elif amt < 0:
                    self.position_side = 'SHORT'
                else:
                    self.position_side = None
                    self.entry_price = 0.0

            state = self.position_side if self.position_side else "미보유"
            logger.info(f"🔄 현재 포지션: {state} | 수량: {self.position_amt} | 평단가: {self.entry_price:,.2f}")
        except Exception as e:
            logger.error(f"포지션 조회 실패: {e}")

    async def _get_ai_prediction_and_features(self) -> float:
        """CCXT 데이터를 Pandas DataFrame으로 변환하여 모델 추론"""
        try:
            # 700봉: WFO HMM train_window=500 초과 보장 (480봉 시 HMM 루프 미실행 → 전봉 횡보 폴백)
            ohlcv = await self.exchange.fetch_ohlcv(self.symbol, self.ccxt_timeframe, limit=700)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            # CCXT는 UTC 밀리초 반환; feature_engineering은 Upbit KST naive 기준으로 학습됨
            # UTC → KST 변환 미수행 시 Hour/DayOfWeek 피처가 9시간 어긋나 추론 오염
            df['timestamp'] = (
                pd.to_datetime(df['timestamp'], unit='ms', utc=True)
                .dt.tz_convert('Asia/Seoul')
                .dt.tz_localize(None)
            )
            df.set_index('timestamp', inplace=True)
            df['timestamp'] = df.index  # feature_engineering 용
            
            # 마지막 미완성 캔들 제외
            completed_df = df.iloc[:-1]
            
            df_features, _ = await asyncio.to_thread(add_technical_indicators, completed_df)
            if df_features is None or df_features.empty:
                return -1.0
            
            X = df_features.iloc[-1:][self.features]
            if 'ATR_Ratio' in df_features.columns:
                self.last_atr_ratio = float(df_features.iloc[-1]['ATR_Ratio'])
                
            ai_prob = float(self.model.predict_proba(X)[0][1])
            
            # HMM 국면 업데이트 (백그라운드)
            if self.hmm.needs_retrain():
                asyncio.create_task(asyncio.to_thread(self.hmm.fit, completed_df))
            self.hmm.predict(completed_df)
            
            return ai_prob
        except Exception as e:
            logger.error(f"AI 추론 오류: {e}")
            return -1.0

    def _compute_dynamic_leverage(self) -> int:
        """
        ATR 기반 동적 레버리지: 변동성이 높을수록 레버리지를 낮춰 명목 리스크를 일정하게 유지.
        공식: clip(TARGET_VOL / ATR_ratio, 1, MAX_LEVERAGE)
        """
        atr = self.last_atr_ratio
        if atr <= 0.001:
            return MAX_LEVERAGE
        dynamic_lev = int(np.clip(_TARGET_VOL / atr, 1, MAX_LEVERAGE))
        return dynamic_lev

    def get_position_info(self) -> dict:
        """오케스트레이터 호환 포지션 정보 반환."""
        return {
            "coin": self.symbol,
            "is_holding": self.position_side is not None,
            "position_side": self.position_side,
            "position_amt": self.position_amt,
            "entry_price": self.entry_price,
        }

    def _compute_futures_size(self, usdt_balance: float, prob_win: float) -> float:
        """
        선물용 동적 포지션 사이즈 (레버리지 반영)
        prob_win: 롱이면 상승확률, 숏이면 하락확률
        """
        atr = self.last_atr_ratio
        kelly_fraction = 0.0

        # 1. 반방향 켈리 공식 적용
        if prob_win > 0.0:
            raw_kelly = (prob_win * KELLY_PAYOFF_RATIO - (1 - prob_win)) / KELLY_PAYOFF_RATIO
            kelly_fraction = max(0.0, raw_kelly) * 0.5

        # 2. _apply_dynamic_leverage_if_needed()에서 거래소 반영 후 self.leverage 갱신됨
        lev = self.leverage

        # 3. 선물 레버리지를 감안한 ATR 리스크 계산
        if atr <= 0.001:  # ATR이 0.1% 이하인 극단적 횡보장 방어 (비정상적 포지션 비대화 차단)
            safe_pct = max(MIN_POSITION_PCT, min(MAX_POSITION_PCT, kelly_fraction)) if kelly_fraction > 0 else MIN_POSITION_PCT
            return usdt_balance * safe_pct * lev

        # 2% 리스크 / (ATR * 손절배수)
        atr_raw_pct = ATR_RISK_PCT / (atr * 1.5)
        raw_pct = min(kelly_fraction, atr_raw_pct) if kelly_fraction > 0 else atr_raw_pct

        # 레버리지를 곱하여 실제 투입할 명목 금액(Notional Value) 산출
        final_pct = max(MIN_POSITION_PCT, min(MAX_POSITION_PCT, raw_pct))
        return usdt_balance * final_pct * lev

    async def execute_order(self, side: str, amount: float, reduce_only: bool = False, use_maker: bool = True):
        """CCXT를 통한 주문 실행 (Maker 지정가 우선, 미체결 시 Market 전환)"""
        if self.dry_run:
            action = "매수(BUY)" if side == 'buy' else "매도(SELL)"
            logger.info(f"[PAPER] {action} {amount:.6f} {self.symbol} | reduce_only={reduce_only}")
            return True
        try:
            if not self.markets_loaded:
                await self.exchange.load_markets()
                self.markets_loaded = True
            
            market = self.exchange.market(self.symbol)
            formatted_amount = float(self.exchange.amount_to_precision(self.symbol, amount))
            min_amount = market['limits']['amount']['min']
            
            if not reduce_only and formatted_amount < min_amount:
                logger.warning(f"⚠️ 주문 수량 미달 ({formatted_amount} < {min_amount})")
                return False

            if use_maker:
                # 1. 최우선 호가 조회 및 Limit 주문
                ticker = await self.exchange.fetch_ticker(self.symbol)
                raw_price = ticker['bid'] if side == 'buy' else ticker['ask']
                target_price = float(self.exchange.price_to_precision(self.symbol, raw_price))

                logger.info(f"⏳ Maker 지정가 시도: {side} {formatted_amount} @ {target_price}")
                order = await self.exchange.create_order(
                    symbol=self.symbol, type='limit', side=side, amount=formatted_amount,
                    price=target_price, params={'reduceOnly': reduce_only, 'timeInForce': 'GTC'}
                )
                
                # 2. 체결 상태 폴링 (1초 간격, 최대 10회)
                order_status = None
                for _ in range(10):
                    await asyncio.sleep(1)
                    order_status = await self.exchange.fetch_order(order['id'], self.symbol)
                    if order_status['status'] == 'closed':
                        break
                if order_status and order_status['status'] == 'closed':
                    logger.info(f"✅ Maker 체결 완료: {side} {formatted_amount}")
                    return True
                else:
                    # 3. 미체결 시 취소 후 Market 전환 (추격 매수/매도)
                    await self.exchange.cancel_order(order['id'], self.symbol)
                    logger.warning("⚠️ Maker 미체결 → Market 주문으로 전환")
                    
            # Market 폴백 또는 즉시 Market 실행
            order = await self.exchange.create_order(
                symbol=self.symbol, type='market', side=side, 
                amount=formatted_amount, params={'reduceOnly': reduce_only}
            )
            logger.info(f"✅ Market 주문 체결: {side} {formatted_amount}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 주문 실행 실패: {e}")
            return False

    async def _place_native_trailing_stop(self, side: str, amount: float) -> Optional[str]:
        """
        거래소 엔진 기반 TRAILING_STOP_MARKET 주문 등록.
        소프트웨어 루프 TS 대비: 네트워크 지연·Rate Limit 무관, 슬리피지 최소화.
        Returns: 주문 ID (실패 시 None)
        """
        if self.dry_run:
            mock_id = f"paper_ts_{int(time.time())}"
            logger.info(f"[PAPER] 네이티브 TS 등록: {side} {amount:.6f} | callback={self._trailing_callback_rate}% | id={mock_id}")
            return mock_id
        if not self.markets_loaded:
            await self.exchange.load_markets()
            self.markets_loaded = True
        stop_side = 'sell' if side.lower() == 'buy' else 'buy'
        formatted_amount = float(self.exchange.amount_to_precision(self.symbol, amount))
        try:
            order = await self.exchange.create_order(
                symbol=self.symbol,
                type='TRAILING_STOP_MARKET',
                side=stop_side,
                amount=formatted_amount,
                price=None,
                params={
                    'callbackRate': self._trailing_callback_rate,
                    'reduceOnly': True,
                    'workingType': 'MARK_PRICE',
                }
            )
            order_id = order.get('id')
            logger.info(
                f"🛡️ 네이티브 TS 등록: {stop_side} {amount} "
                f"| callbackRate={self._trailing_callback_rate}% | id={order_id}"
            )
            return order_id
        except Exception as e:
            logger.warning(f"⚠️ 네이티브 TS 등록 실패 (소프트웨어 TS로 폴백): {e}")
            return None

    async def _cancel_trailing_stop_if_exists(self) -> None:
        """포지션 청산 전 기존 TS 주문 취소 (미취소 시 반대 포지션 생성 위험)."""
        if not self._trailing_stop_order_id:
            return
        if self.dry_run:
            logger.info(f"[PAPER] TS 주문 취소: id={self._trailing_stop_order_id}")
            self._trailing_stop_order_id = None
            return
        try:
            await self.exchange.cancel_order(self._trailing_stop_order_id, self.symbol)
            logger.info(f"🗑️ TS 주문 취소: id={self._trailing_stop_order_id}")
        except Exception:
            pass
        finally:
            self._trailing_stop_order_id = None

    async def _check_funding_rate_before_entry(self) -> bool:
        """
        바이낸스 동적 펀딩비 1시간 정산 전환 필터.
        abs(funding_rate) >= 0.025% 이면 진입 차단 → False 반환.
        """
        threshold = float(
            bot_config.FUTURES_RISK.get("funding_risk_threshold", 0.00025)
        )
        symbol_id = self.symbol.replace('/', '')
        safe = await asyncio.to_thread(
            self.binance_ctx.check_funding_rate_risk, symbol_id, threshold
        )
        return safe

    async def _send_telegram(self, message: str) -> None:
        """텔레그램 알림 비동기 전송 (실패 시 무시)."""
        if not self._tg_token or not self._tg_chat_id:
            return
        try:
            url = f"https://api.telegram.org/bot{self._tg_token}/sendMessage"
            await asyncio.to_thread(
                requests.post, url,
                json={"chat_id": self._tg_chat_id, "text": f"[FuturesBot|{self.symbol}] {message}"},
                timeout=5,
            )
        except Exception:
            pass

    def _check_circuit_breaker(self, usdt_balance: float) -> bool:
        """일일 손실 한도 초과 여부 반환. 자정(UTC)마다 PnL 카운터 자동 리셋."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self._circuit_breaker_date != today:
            self._circuit_breaker_date = today
            self._daily_realized_pnl = 0.0
            self._circuit_breaker_active = False
        if not self._circuit_breaker_active and usdt_balance > 0:
            if self._daily_realized_pnl < -(usdt_balance * self._daily_loss_limit):
                self._circuit_breaker_active = True
                logger.warning(
                    f"🚨 [서킷 브레이커] 일일 손실 {abs(self._daily_realized_pnl):.2f} USDT "
                    f"({self._daily_loss_limit*100:.0f}% 초과) → 당일 신규 진입 차단"
                )
        return self._circuit_breaker_active

    def _record_trade_pnl(self, entry: float, exit_price: float, side: str, amt: float) -> float:
        """실현 PnL 산출 및 일일 누적."""
        pnl = (exit_price - entry) * abs(amt) if side == 'LONG' else (entry - exit_price) * abs(amt)
        self._daily_realized_pnl += pnl
        if self.dry_run:
            # 증거금 반환 + PnL 반영
            notional = entry * abs(amt)
            margin_return = notional / self.leverage
            self._paper_balance += margin_return + pnl
            logger.info(
                f"[PAPER] 실현 PnL: {pnl:+.2f} USDT | 잔고 복원: {self._paper_balance:,.2f} USDT "
                f"(일일 누적: {self._daily_realized_pnl:+.2f})"
            )
        else:
            logger.info(f"💰 실현 PnL: {pnl:+.2f} USDT (일일 누적: {self._daily_realized_pnl:+.2f})")
        return pnl

    async def _apply_dynamic_leverage_if_needed(self) -> None:
        """ATR 기반 동적 레버리지를 거래소에 실시간 반영 (변경 없으면 API 미호출)."""
        new_lev = self._compute_dynamic_leverage()
        if new_lev == self.leverage:
            return
        if self.dry_run:
            logger.info(f"[PAPER] 동적 레버리지: {self.leverage}x → {new_lev}x (ATR={self.last_atr_ratio:.4f})")
            self.leverage = new_lev
            return
        try:
            await self.exchange.fapiprivate_post_leverage({
                'symbol': self.symbol.replace('/', ''),
                'leverage': new_lev,
            })
            logger.info(
                f"⚡ 동적 레버리지 거래소 반영: {self.leverage}x → {new_lev}x "
                f"(ATR={self.last_atr_ratio:.4f})"
            )
            self.leverage = new_lev
        except Exception as e:
            logger.warning(f"동적 레버리지 거래소 반영 실패 ({self.leverage}x 유지): {e}")

    async def panic_close_all(self) -> bool:
        """🚨 AI 킬 스위치 발동 시 모든 포지션 덤핑. 반환값: 청산 확인 여부."""
        if self.position_side == 'LONG':
            logger.critical("🚨 [킬 스위치] LONG 포지션 전량 시장가 매도!")
            await self.execute_order('sell', abs(self.position_amt), reduce_only=True)
        elif self.position_side == 'SHORT':
            logger.critical("🚨 [킬 스위치] SHORT 포지션 전량 시장가 환매수!")
            await self.execute_order('buy', abs(self.position_amt), reduce_only=True)
        else:
            return True  # 포지션 없음

        # 청산 확인: 최대 3회 재시도
        for attempt in range(3):
            await asyncio.sleep(3)
            await self._fetch_position()
            if not self.position_side:
                logger.critical("✅ [킬 스위치] 포지션 청산 확인 완료.")
                return True
            logger.warning(f"⚠️ [킬 스위치] 포지션 잔존 확인 ({attempt + 1}/3), 재시도...")

        logger.critical(
            f"🚨 [킬 스위치] 청산 미확인! 잔존 포지션={self.position_side} "
            f"수량={self.position_amt} — 수동 개입 필요!"
        )
        return False

    async def _strategy_loop(self):
        """선물 롱/숏 하이브리드 전략 루프"""
        while self.is_running:
            try:
                # ── 1. 0순위 AI 킬 스위치 감시 ──
                _, is_kill_switch = await asyncio.to_thread(self.news_analyzer.fetch)
                if is_kill_switch and self.position_side:
                    await self._send_telegram(f"🚨 AI 킬 스위치 발동 | {self.position_side} 강제 청산")
                    closed = await self.panic_close_all()
                    if not closed:
                        await asyncio.sleep(15)
                        continue
                    await asyncio.sleep(60)
                    continue

                # ── 2. 상태 및 잔고 동기화 ──
                await self._fetch_position()
                if self.dry_run:
                    usdt_balance = self._paper_balance
                else:
                    balance = await self.exchange.fetch_balance()
                    usdt_balance = float(balance.get('USDT', {}).get('free', 0.0))
                
                ticker = await self.exchange.fetch_ticker(self.symbol)
                current_price = ticker['last']
                
                # ── 3. AI 추론 및 HMM 국면 확인 ──
                ai_prob_up = await self._get_ai_prediction_and_features()
                if ai_prob_up < 0:  # 추론 실패
                    await asyncio.sleep(15)
                    continue

                regime = self.hmm.regime
                # ⚠️ ai_prob_down = 1 - ai_prob_up 은 잘못된 계산.
                # 모델은 "상승(1) vs 비상승(0)"으로 학습됨.
                # 1 - P(up) = P(비상승) = 횡보 + 하락 혼합이므로 독립적 숏 신호가 아님.
                # 숏 진입은 P(up) ≤ SHORT_MAX_UP_PROB + BEAR 국면 이중 게이트로 보호.
                # SHORT_MAX_UP_PROB: config.BACKTEST["short_max_up_prob"] → 모듈 상수로 관리

                logger.info(
                    f"📊 국면: {regime} | 상승확률: {ai_prob_up*100:.1f}% "
                    f"(숏 진입 기준 ≤ {SHORT_MAX_UP_PROB*100:.0f}%)"
                )

                # ── 4. 틱사이즈 동적 갱신 (1시간 주기) ──────────────────────
                await self._refresh_exchange_info_if_needed()

                # ── 5. K-Means 고변동성 횡보 필터 ────────────────────────────
                # CHOP 감지 시 추세 추종 신호를 차단하는 1차 필터
                if self.kmeans_regime.needs_retrain():
                    try:
                        ohlcv_km = await self.exchange.fetch_ohlcv(
                            self.symbol, self.ccxt_timeframe, limit=250
                        )
                        df_km = pd.DataFrame(
                            ohlcv_km,
                            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                        )
                        await asyncio.to_thread(self.kmeans_regime.fit_predict, df_km)
                    except Exception as e:
                        logger.debug(f"K-Means 갱신 실패(무시): {e}")

                kmeans_chop = self.kmeans_regime.is_chop()
                if kmeans_chop and self.position_side is None:
                    logger.info("⏸️ [K-Means] 고변동성 횡보 감지 → 신규 진입 보류")
                    await asyncio.sleep(15)
                    continue

                # ── 6. 포지션 없을 때 펀딩비 진입 필터 (동적 1h 정산 방어) ──
                if self.position_side is None:
                    funding_safe = await self._check_funding_rate_before_entry()
                    if not funding_safe:
                        logger.info("⏸️ [펀딩비] 고위험 펀딩비 → 신규 진입 보류")
                        await asyncio.sleep(60)
                        continue

                # ── 6b. 포지션 보유 중 극단적 펀딩비 긴급 청산 ──────────────
                # 동적 1h 정산 전환 임계값(0.025%)의 2배(0.05%) 초과 시 즉시 청산.
                if self.position_side is not None:
                    _hold_fr_threshold = float(
                        bot_config.FUTURES_RISK.get("funding_risk_threshold", 0.00025)
                    ) * 2.0
                    try:
                        _sym = self.symbol.replace('/', '')
                        def _fetch_fr():
                            return requests.get(
                                f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={_sym}",
                                timeout=5,
                            ).json()
                        _fr_data = await asyncio.to_thread(_fetch_fr)
                        _current_fr = abs(float(_fr_data.get("lastFundingRate", 0.0)))
                    except Exception:
                        _current_fr = 0.0
                    if _current_fr >= _hold_fr_threshold:
                        logger.warning(
                            f"🚨 [펀딩비 긴급청산] 보유 중 펀딩비 {_current_fr*100:.4f}% "
                            f"≥ ±{_hold_fr_threshold*100:.3f}% → {self.position_side} 즉시 청산"
                        )
                        await self._cancel_trailing_stop_if_exists()
                        await self.panic_close_all()
                        self.position_side = None
                        self._save_state()
                        await asyncio.sleep(60)
                        continue

                # ── 7. 네이티브 TS 미등록 상태 감지 (재부팅 후 포지션 복원 시) ──
                if (self.position_side is not None
                        and self._use_native_trailing_stop
                        and self._trailing_stop_order_id is None):
                    entry_side = 'buy' if self.position_side == 'LONG' else 'sell'
                    self._trailing_stop_order_id = await self._place_native_trailing_stop(
                        entry_side, abs(self.position_amt)
                    )

                # ── 8. 소프트웨어 TS 폴백 (네이티브 TS 미지원 환경 대비) ────────
                if not self._use_native_trailing_stop and self.position_side in ('LONG', 'SHORT'):
                    trail_pct = bot_config.BACKTEST.get("trailing_stop_pct", 0.03)
                    _soft_ts_triggered = False
                    if self.position_side == 'LONG':
                        if current_price > self._trailing_high:
                            self._trailing_high = current_price
                        if self._trailing_high > 0:
                            drop = (self._trailing_high - current_price) / self._trailing_high
                            if drop >= trail_pct:
                                logger.info(
                                    f"🛑 [소프트 TS] 고점 {self._trailing_high:,.2f} 대비 "
                                    f"{drop*100:.1f}% 하락 → LONG 청산"
                                )
                                await self._cancel_trailing_stop_if_exists()
                                if await self.execute_order('sell', abs(self.position_amt), reduce_only=True):
                                    pnl = self._record_trade_pnl(self.entry_price, current_price, 'LONG', self.position_amt)
                                    await self._send_telegram(f"🛑 소프트TS LONG청산 | PnL={pnl:+.2f}U")
                                    self.position_side = None
                                    self._trailing_high = 0.0
                                    self._save_state()
                                _soft_ts_triggered = True
                    elif self.position_side == 'SHORT':
                        if self._trailing_low == 0.0 or current_price < self._trailing_low:
                            self._trailing_low = current_price
                        if self._trailing_low > 0:
                            rise = (current_price - self._trailing_low) / self._trailing_low
                            if rise >= trail_pct:
                                logger.info(
                                    f"🛑 [소프트 TS] 저점 {self._trailing_low:,.2f} 대비 "
                                    f"{rise*100:.1f}% 반등 → SHORT 청산"
                                )
                                await self._cancel_trailing_stop_if_exists()
                                if await self.execute_order('buy', abs(self.position_amt), reduce_only=True):
                                    pnl = self._record_trade_pnl(self.entry_price, current_price, 'SHORT', self.position_amt)
                                    await self._send_telegram(f"🛑 소프트TS SHORT청산 | PnL={pnl:+.2f}U")
                                    self.position_side = None
                                    self._trailing_low = 0.0
                                    self._save_state()
                                _soft_ts_triggered = True
                    if _soft_ts_triggered:
                        await asyncio.sleep(15)
                        continue

                # ── 9. 양방향(Long/Short) 진입/청산 로직 ──────────────────────

                # [상승장 또는 횡보장] → LONG 전략
                if regime in [HMMRegimeDetector.BULL, HMMRegimeDetector.SIDEWAYS]:
                    if ai_prob_up >= self.threshold:
                        if self.position_side == 'SHORT':
                            logger.info("🔄 [스위칭] 상승 신호 → 기존 SHORT 청산")
                            await self._cancel_trailing_stop_if_exists()
                            if await self.execute_order('buy', abs(self.position_amt), reduce_only=True):
                                pnl = self._record_trade_pnl(self.entry_price, current_price, 'SHORT', self.position_amt)
                                await self._send_telegram(f"🔄 스위칭 SHORT청산 | PnL={pnl:+.2f}U")
                                self.position_side = None
                                self._trailing_low = 0.0
                                self._save_state()
                            await asyncio.sleep(2)

                        if self.position_side != 'LONG' and not self._check_circuit_breaker(usdt_balance):
                            await self._apply_dynamic_leverage_if_needed()
                            notional_size = self._compute_futures_size(usdt_balance, ai_prob_up)
                            qty = notional_size / current_price
                            logger.info("🚀 [LONG 진입] 상승 추세 확신 → 시장가 진입")
                            if await self.execute_order('buy', qty):
                                self.position_side = 'LONG'
                                self.entry_price = current_price
                                self._trailing_high = current_price
                                self._trailing_low = 0.0
                                if self.dry_run:
                                    margin_used = notional_size / self.leverage
                                    self._paper_balance -= margin_used
                                    logger.info(f"[PAPER] 증거금 차감: {margin_used:,.2f} USDT | 잔고: {self._paper_balance:,.2f} USDT")
                                self._save_state()
                                await self._send_telegram(
                                    f"{'[PAPER] ' if self.dry_run else ''}🚀 LONG진입 {qty:.4f} @ {current_price:,.2f} | lev={self.leverage}x"
                                )
                                if self._use_native_trailing_stop:
                                    self._trailing_stop_order_id = (
                                        await self._place_native_trailing_stop('buy', qty)
                                    )

                # [하락장] → SHORT 전략: BEAR 국면 + P(up) ≤ 20% 이중 게이트
                elif regime == HMMRegimeDetector.BEAR:
                    if ai_prob_up <= SHORT_MAX_UP_PROB:
                        fr, _ = await asyncio.to_thread(self.binance_ctx.fetch)
                        funding_bonus = max(0.0, fr * 20.0) if fr > 0 else 0.0
                        effective_max_up = SHORT_MAX_UP_PROB + funding_bonus

                        if ai_prob_up <= effective_max_up:
                            if self.position_side == 'LONG':
                                logger.info("🔄 [스위칭] BEAR 국면 + 약세 신호 → 기존 LONG 청산")
                                await self._cancel_trailing_stop_if_exists()
                                if await self.execute_order('sell', abs(self.position_amt), reduce_only=True):
                                    pnl = self._record_trade_pnl(self.entry_price, current_price, 'LONG', self.position_amt)
                                    await self._send_telegram(f"🔄 스위칭 LONG청산 | PnL={pnl:+.2f}U")
                                    self.position_side = None
                                    self._trailing_high = 0.0
                                    self._save_state()
                                await asyncio.sleep(2)

                            if self.position_side != 'SHORT' and not self._check_circuit_breaker(usdt_balance):
                                await self._apply_dynamic_leverage_if_needed()
                                # P(비상승) = P(횡보+하락) 혼합이므로 직접 사용 금지.
                                # threshold 이격 거리로 [0.50, 0.70] 범위 보수적 환산.
                                # ai_prob_up=0.20 → 0.50(최소), ai_prob_up=0.00 → 0.70(최대)
                                short_prob_win = min(0.70, 0.5 + (SHORT_MAX_UP_PROB - ai_prob_up))
                                notional_size = self._compute_futures_size(usdt_balance, short_prob_win)
                                qty = notional_size / current_price
                                logger.info("🩸 [SHORT 진입] BEAR + 강한 하락 신호 → 시장가 진입")
                                if await self.execute_order('sell', qty):
                                    self.position_side = 'SHORT'
                                    self.entry_price = current_price
                                    self._trailing_high = 0.0
                                    self._trailing_low = current_price
                                    if self.dry_run:
                                        margin_used = notional_size / self.leverage
                                        self._paper_balance -= margin_used
                                        logger.info(f"[PAPER] 증거금 차감: {margin_used:,.2f} USDT | 잔고: {self._paper_balance:,.2f} USDT")
                                    self._save_state()
                                    await self._send_telegram(
                                        f"{'[PAPER] ' if self.dry_run else ''}🩸 SHORT진입 {qty:.4f} @ {current_price:,.2f} | lev={self.leverage}x"
                                    )
                                    if self._use_native_trailing_stop:
                                        self._trailing_stop_order_id = (
                                            await self._place_native_trailing_stop('sell', qty)
                                        )

            except Exception as e:
                logger.error(f"전략 루프 오류: {e}")

            await asyncio.sleep(15)  # 15초마다 타점 감시

    async def run(self):
        mode_tag = "[PAPER TRADING] " if self.dry_run else ""
        logger.info(f"🚀 {mode_tag}선물 봇 시스템 부팅 중... | 심볼={self.symbol} | 레버리지={self.leverage}x")
        if self.dry_run:
            logger.info(f"[PAPER] 초기 가상 잔고: {self._paper_balance:,.2f} USDT — 실제 주문 없음")
        await self._apply_hardcoded_safety_measures()
        try:
            await self._strategy_loop()
        except asyncio.CancelledError:
            pass
        finally:
            logger.info("🛑 선물 봇 안전 종료 절차 진행 중...")
            await self.exchange.close()

    def run_sync(self):
        """동기 래퍼 — threading.Thread(target=bot.run_sync) 로만 호출할 것.

        설계 제약:
        - AsyncFuturesBot은 asyncio 전용이므로 동기(trade_bot.py) 스레드 컨텍스트와
          이벤트 루프를 공유해선 안 된다.
        - asyncio.run()은 새 이벤트 루프를 생성·실행·폐기하므로 스레드별 격리를 보장한다.
        - 이미 실행 중인 루프(예: Jupyter, FastAPI) 위에서 호출하면
          RuntimeError("This event loop is already running")가 발생한다.
          그런 환경에서는 asyncio.run_until_complete() 대신 asyncio.create_task()를 사용하라.
        - orchestrator.py는 반드시 이 봇을 독립 스레드(또는 독립 프로세스)로 실행해야 한다.
        """
        # asyncio.get_running_loop()는 실행 중인 루프가 있으면 반환,
        # 없으면 RuntimeError — 이를 이용해 루프 충돌을 선제 차단한다.
        try:
            asyncio.get_running_loop()
            # 여기 도달 = 이미 루프 실행 중 → 호출 컨텍스트가 잘못됨
            logger.critical(
                f"[FuturesBot:{self.symbol}] run_sync()가 이미 실행 중인 이벤트 루프 "
                "안에서 호출됐습니다. 독립 스레드(threading.Thread)로 실행하세요."
            )
            return
        except RuntimeError:
            pass  # 실행 중인 루프 없음 → 정상 경로
        try:
            asyncio.run(self.run())
        except Exception as e:
            logger.error(f"[FuturesBot:{self.symbol}] run_sync 종료: {e}")

    def stop_gracefully(self):
        """오케스트레이터 호환 종료 인터페이스 (AITradingBot.stop_gracefully와 동일 시그니처)."""
        logger.info(f"[FuturesBot:{self.symbol}] 종료 신호 수신")
        self.is_running = False

    def force_kill(self):
        """오케스트레이터 호환 강제 종료 인터페이스."""
        logger.warning(f"[FuturesBot:{self.symbol}] force_kill 호출")
        self.is_running = False


def _krw_to_usdt(krw: int) -> float:
    """업비트 KRW-USDT 시세로 원화 → USDT 환산 (공개 API, 인증 불필요)"""
    try:
        resp = requests.get(
            "https://api.upbit.com/v1/ticker?markets=KRW-USDT",
            timeout=5,
            headers={"Accept": "application/json"},
        )
        rate = float(resp.json()[0]["trade_price"])
        usdt = krw / rate
        logger.info(f"[PAPER] 환율: 1 USDT = {rate:,.0f} KRW | {krw:,}원 → {usdt:,.2f} USDT")
        return usdt
    except Exception as e:
        logger.warning(f"환율 조회 실패 ({e}) — 폴백 1 USDT=1380 KRW 적용")
        return krw / 1380.0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="선물 봇 실행",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "예시:\n"
            "  python futures_bot.py --dry-run --paper-balance-krw 1000000   # 100만원 테스트\n"
            "  python futures_bot.py --dry-run --paper-balance-krw 3000000   # 300만원 테스트\n"
            "  python futures_bot.py --dry-run --paper-balance-krw 5000000   # 500만원 테스트\n"
            "  python futures_bot.py --dry-run --paper-balance-krw 10000000  # 1000만원 테스트\n"
            "  python futures_bot.py --dry-run --paper-balance 5000          # USDT 직접 지정\n"
        ),
    )
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--leverage", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true", help="Paper Trading 모드 (실제 주문 없음)")
    parser.add_argument("--paper-balance", type=float, default=None, help="Paper 초기 잔고 (USDT)")
    parser.add_argument("--paper-balance-krw", type=int, default=None, help="Paper 초기 잔고 (KRW → 실시간 환율 변환)")
    args = parser.parse_args()

    paper_balance_usdt = None
    if args.paper_balance_krw is not None:
        if not args.dry_run:
            parser.error("--paper-balance-krw 는 --dry-run 과 함께 사용해야 합니다.")
        paper_balance_usdt = _krw_to_usdt(args.paper_balance_krw)
    elif args.paper_balance is not None:
        if not args.dry_run:
            parser.error("--paper-balance 는 --dry-run 과 함께 사용해야 합니다.")
        paper_balance_usdt = args.paper_balance

    bot = AsyncFuturesBot(symbol=args.symbol, leverage=args.leverage, dry_run=args.dry_run)
    if paper_balance_usdt is not None:
        bot._paper_balance = paper_balance_usdt

    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        bot.is_running = False