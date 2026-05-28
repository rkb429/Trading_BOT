"""
market_context.py
실시간 시장 컨텍스트 수집기
- get_dynamic_target_coins : 거래대금 기반 동적 코인 추출 (config.py에서 이동)
- BinanceFuturesContext : 펀딩비 + 롱숏 비율
- KimchiPremiumTracker  : 김치 프리미엄 + 가속도
- FearGreedIndex        : 공포-탐욕 지수
- HMMRegimeDetector     : HMM 기반 시장 국면
- LiquidityScanner      : 유동성 로테이션 스캐너
"""

import os
import re
import json
import glob
import uuid
import time
import pickle
import logging
import threading
import requests
import numpy as np
import pandas as pd
import pyupbit
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, List, Optional

logger = logging.getLogger(__name__)

# ─── 거래대금 기반 동적 코인 추출 (config.py → market_context.py) ───────────────

_FALLBACK_COINS  = ["KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-SOL"]
_EXCLUDE_COINS   = {"KRW-USDT", "KRW-USDC"}  # 스테이블 코인 제외


_COIN_CACHE_TTL  = 300  # 5분
_coin_cache: dict = {"all_coins": None, "ts": 0.0}
_coin_cache_lock = threading.Lock()
_wallet_cache: dict = {"suspended": None, "ts": 0.0}
_wallet_cache_lock = threading.Lock()


def _get_suspended_coins() -> set:
    """입출금 모두 중단된 코인 심볼 집합 반환 (paused / unsupported)."""
    now = time.time()
    if _wallet_cache["suspended"] is not None and now - _wallet_cache["ts"] < _COIN_CACHE_TTL:
        return _wallet_cache["suspended"]
    with _wallet_cache_lock:
        now = time.time()
        if _wallet_cache["suspended"] is not None and now - _wallet_cache["ts"] < _COIN_CACHE_TTL:
            return _wallet_cache["suspended"]
        try:
            resp = requests.get("https://api.upbit.com/v1/status/wallet", timeout=5).json()
            suspended = {
                f"KRW-{item['currency']}"
                for item in resp
                if item.get("wallet_state") in ("paused", "unsupported", "withdraw_only")
            }
            _wallet_cache["suspended"] = suspended
            _wallet_cache["ts"] = now
            return suspended
        except Exception:
            return set()


def get_all_krw_coins() -> list:
    """업비트 KRW 마켓 전체 코인 목록 반환 (스테이블코인 제외, 거래대금 순 정렬)."""
    now = time.time()
    if _coin_cache["all_coins"] is not None and now - _coin_cache["ts"] < _COIN_CACHE_TTL:
        return _coin_cache["all_coins"]

    with _coin_cache_lock:
        now = time.time()
        if _coin_cache["all_coins"] is not None and now - _coin_cache["ts"] < _COIN_CACHE_TTL:
            return _coin_cache["all_coins"]

        try:
            markets = requests.get(
                "https://api.upbit.com/v1/market/all?isDetails=true", timeout=5
            ).json()
            suspended = _get_suspended_coins()
            krw_markets = [
                m['market'] for m in markets
                if m['market'].startswith("KRW-")
                and m.get('market_warning', 'NONE') == 'NONE'
                and m['market'] not in suspended
            ]

            all_tickers = []
            for i in range(0, len(krw_markets), 100):
                chunk = krw_markets[i:i + 100]
                resp = requests.get(
                    "https://api.upbit.com/v1/ticker",
                    params={"markets": ",".join(chunk)},
                    timeout=5,
                ).json()
                if isinstance(resp, list):
                    all_tickers.extend(resp)

            sorted_tickers = sorted(all_tickers, key=lambda x: x['acc_trade_price_24h'], reverse=True)
            all_coins = [t['market'] for t in sorted_tickers if t['market'] not in _EXCLUDE_COINS]

            result = all_coins if all_coins else _FALLBACK_COINS
            _coin_cache["all_coins"] = result
            _coin_cache["ts"] = now
            return result
        except Exception as e:
            logging.warning(f"⚠️ 전체 코인 목록 추출 실패: {type(e).__name__} — {e} / 폴백 리스트 사용")
            return _FALLBACK_COINS


def _load_perf_blacklist() -> set:
    """TRADING_BLACKLIST + 누적 성능 블랙리스트 + 최신 백테스트 Sharpe 미달 코인 반환."""
    try:
        import config as _cfg
        blacklist = set(f"KRW-{c}" for c in getattr(_cfg, "TRADING_BLACKLIST", set()))
        log_dir = _cfg.DIRECTORIES.get("logs", "logs")

        # 1. 누적 성능 블랙리스트 (backtest.py가 갱신하는 persistent JSON)
        bl_criteria = getattr(_cfg, "PERFORMANCE_BLACKLIST_CRITERIA", {})
        if bl_criteria.get("enabled", True):
            bl_file = bl_criteria.get("blacklist_file", "logs/performance_blacklist.json")
            if os.path.exists(bl_file):
                with open(bl_file, "r", encoding="utf-8") as f:
                    bl_data = json.load(f)
                for coin, entry in bl_data.items():
                    if entry.get("status") in ("blacklisted", "probation"):
                        blacklist.add(f"KRW-{coin}")

        # 2. 최신 백테스트 JSON 단회 Sharpe 필터 (기존 로직 유지)
        pf = getattr(_cfg, "BACKTEST_PERFORMANCE_FILTER", {})
        if pf.get("enabled", False):
            files = sorted(glob.glob(os.path.join(log_dir, "backtest_*.json")))
            if files:
                with open(files[-1], "r", encoding="utf-8") as f:
                    data = json.load(f)
                sharpe_min = pf.get("sharpe_min", -1.0)
                min_trades = pf.get("min_trades", 10)
                for entry in data.get("per_coin", []):
                    if entry.get("total_trades", 0) < min_trades:
                        continue
                    if entry.get("sharpe", 0) < sharpe_min:
                        m = re.search(r'^([^_]+)', entry.get("coin", ""))
                        if m:
                            blacklist.add(f"KRW-{m.group(1)}")
        return blacklist
    except Exception:
        try:
            import config as _cfg
            return set(f"KRW-{c}" for c in getattr(_cfg, "TRADING_BLACKLIST", set()))
        except Exception:
            return set()


def get_observation_coins() -> list:
    """성능 블랙리스트(차단)에 있지만 회복 감지를 위해 데이터 수집은 계속할 코인 목록.
    TRADING_BLACKLIST(영구 정적 차단)는 제외 — 거래도 수집도 안 함."""
    try:
        import config as _cfg
        bl_criteria = getattr(_cfg, 'PERFORMANCE_BLACKLIST_CRITERIA', {})
        if not bl_criteria.get('enabled', True):
            return []
        bl_file = bl_criteria.get('blacklist_file', 'logs/performance_blacklist.json')
        if not os.path.exists(bl_file):
            return []
        with open(bl_file, 'r', encoding='utf-8') as f:
            bl_data = json.load(f)
        static_bl = set(getattr(_cfg, 'TRADING_BLACKLIST', set()))
        return [
            f"KRW-{coin}" for coin, entry in bl_data.items()
            if entry.get('status') in ('blacklisted', 'probation') and coin not in static_bl
        ]
    except Exception:
        return []


def get_dynamic_target_coins(limit: int = 5) -> list:
    """거래대금 상위 코인으로 limit까지 채워 반환 (모든 코인 유동적).
    TRADING_BLACKLIST + 누적 성능 블랙리스트 코인은 제외."""
    all_coins = get_all_krw_coins()
    perf_blacklist = _load_perf_blacklist()
    result = []
    for coin in all_coins:
        if coin not in perf_blacklist and len(result) < limit:
            result.append(coin)
    return result if result else _FALLBACK_COINS

# ─── Binance Futures Context ──────────────────────────────────────────────────

class BinanceFuturesContext:
    """바이낸스 선물: 펀딩비 + 글로벌 롱숏 비율 (무인증 Public API)"""

    _PREMIUM_URL = "https://fapi.binance.com/fapi/v1/premiumIndex?symbol=BTCUSDT"
    _LS_URL = (
        "https://fapi.binance.com/futures/data/globalLongShortAccountRatio"
        "?symbol=BTCUSDT&period=15m&limit=1"
    )
    _SPOT_URL = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"

    def __init__(self, cache_ttl: int = 300,
                 funding_threshold: float = 0.001,
                 ls_overbought: float = 1.8,
                 funding_oversold: float = -0.0005):
        self.cache_ttl = cache_ttl
        self.funding_threshold = funding_threshold
        self.ls_overbought = ls_overbought
        self.funding_oversold = funding_oversold

        self._funding_rate: float = 0.0
        self._ls_ratio: float = 1.0
        self._btc_usdt: float = 0.0
        self._last_fetch: float = 0.0
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(
            target=self._background_refresh, daemon=True, name="BinanceCtxRefresh"
        )
        self._thread.start()

    def _do_fetch(self) -> None:
        """바이낸스 3개 엔드포인트 병렬 조회 후 캐시 갱신 — 백그라운드 스레드 전용."""
        def _get_funding():
            r = requests.get(self._PREMIUM_URL, timeout=5)
            return float(r.json().get("lastFundingRate", 0.0))

        def _get_ls():
            r = requests.get(self._LS_URL, timeout=5)
            data = r.json()
            return float(data[0]["longShortRatio"]) if data else 1.0

        def _get_btc():
            r = requests.get(self._SPOT_URL, timeout=5)
            return float(r.json().get("price", 0.0))

        try:
            with ThreadPoolExecutor(max_workers=3) as ex:
                f_fund = ex.submit(_get_funding)
                f_ls   = ex.submit(_get_ls)
                f_btc  = ex.submit(_get_btc)
                funding_rate = f_fund.result()
                ls_ratio     = f_ls.result()
                btc_usdt     = f_btc.result()

            with self._lock:
                self._funding_rate = funding_rate
                self._ls_ratio = ls_ratio
                self._btc_usdt = btc_usdt
                self._last_fetch = time.time()

            logger.debug(
                f"[Binance] 펀딩비={funding_rate*100:.4f}% "
                f"L/S={ls_ratio:.2f} BTC/USDT={btc_usdt:,.1f}"
            )
        except Exception as e:
            logger.warning(f"바이낸스 선물 컨텍스트 조회 실패: {e}")

    def _background_refresh(self) -> None:
        while self._running:
            self._do_fetch()
            time.sleep(self.cache_ttl)

    def fetch(self) -> Tuple[float, float]:
        """캐시된 (funding_rate, long_short_ratio) 즉시 반환 (non-blocking).
        백그라운드 스레드가 cache_ttl마다 자동 갱신.
        """
        with self._lock:
            return self._funding_rate, self._ls_ratio

    def shutdown(self) -> None:
        self._running = False

    def get_btc_usdt(self) -> float:
        self.fetch()
        with self._lock:
            return self._btc_usdt

    @property
    def is_long_overheated(self) -> bool:
        """롱 과열: 펀딩비 > 0.1% AND 롱숏 비율 > 1.8"""
        fr, ls = self.fetch()
        return fr > self.funding_threshold and ls > self.ls_overbought

    @property
    def is_short_panic(self) -> bool:
        """숏 패닉(반등 기대): 펀딩비 < -0.05%"""
        fr, _ = self.fetch()
        return fr < self.funding_oversold

    def check_funding_rate_risk(self, symbol: str = "BTCUSDT",
                                threshold: float = 0.00025) -> bool:
        """
        2026년 바이낸스 동적 펀딩비 1시간 정산 전환 필터.
        abs(funding_rate) >= threshold(0.025%)이면 펀딩비 폭탄 위험 → False 반환.
        BTCUSDT는 캐시값을 우선 사용하여 중복 HTTP 호출 차단.
        """
        try:
            if symbol == "BTCUSDT":
                with self._lock:
                    if time.time() - self._last_fetch < self.cache_ttl:
                        rate = self._funding_rate
                        safe = abs(rate) < threshold
                        if not safe:
                            logger.warning(
                                f"[FundingRisk] {symbol} 펀딩비 {rate*100:.4f}% "
                                f"≥ ±{threshold*100:.3f}% → 진입 차단 (캐시)"
                            )
                        return safe

            url = f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={symbol}"
            data = requests.get(url, timeout=5).json()
            rate = float(data.get("lastFundingRate", 0.0))
            safe = abs(rate) < threshold
            if not safe:
                logger.warning(
                    f"[FundingRisk] {symbol} 펀딩비 {rate*100:.4f}% "
                    f"≥ ±{threshold*100:.3f}% → 진입 차단"
                )
            return safe
        except Exception as e:
            logger.warning(f"[FundingRisk] 펀딩비 조회 실패: {e} → 보수적 차단")
            return False


# ─── Kimchi Premium Tracker ───────────────────────────────────────────────────

class KimchiPremiumTracker:
    """
    김치 프리미엄 실시간 추적 + 가속도(1차 미분) 계산.
    업비트 BTC/KRW ÷ (바이낸스 BTC/USDT × 업비트 USDT/KRW) - 1
    """

    def __init__(self, cache_ttl: int = 60, history_size: int = 5,
                 surge_threshold: float = 0.002):
        self.cache_ttl = cache_ttl
        self.surge_threshold = surge_threshold
        self._history: deque = deque(maxlen=history_size)
        self._premium: float = 0.0
        self._acceleration: float = 0.0
        self._last_fetch: float = 0.0
        self._lock = threading.Lock()

    def update(self, upbit_btc_krw: float,
               btc_usdt: float, usdt_krw: float) -> Tuple[float, float]:
        """
        (premium, acceleration) 반환.
        btc_usdt  : 바이낸스 BTC/USDT 가격 (BinanceFuturesContext에서 제공)
        usdt_krw  : 업비트 KRW-USDT 현재가 (≈ 달러 환율)
        """
        with self._lock:
            if time.time() - self._last_fetch < self.cache_ttl:
                return self._premium, self._acceleration

        try:
            if btc_usdt <= 0 or usdt_krw <= 0 or upbit_btc_krw <= 0:
                logger.warning(
                    f"[KimchiPremium] 계산 불가 "
                    f"(btc_usdt={btc_usdt}, usdt_krw={usdt_krw}, "
                    f"upbit_btc_krw={upbit_btc_krw}) — 캐시값 반환"
                )
                # _last_fetch를 갱신하여 cache_ttl 이내 재호출 시 경고가 반복되지 않도록 한다.
                with self._lock:
                    self._last_fetch = time.time()
                    return self._premium, self._acceleration

            btc_krw_ref = btc_usdt * usdt_krw
            premium = (upbit_btc_krw / btc_krw_ref) - 1.0

            with self._lock:
                prev = self._history[-1] if self._history else premium
                self._history.append(premium)
                self._premium = premium
                self._acceleration = premium - prev
                self._last_fetch = time.time()

            logger.debug(
                f"[김치] 프리미엄={premium*100:.2f}% "
                f"가속도={self._acceleration*100:.3f}%"
            )
            return premium, self._acceleration

        except Exception as e:
            logger.warning(f"김치 프리미엄 계산 실패: {e}")
            with self._lock:
                return self._premium, self._acceleration

    @property
    def is_surging(self) -> bool:
        """프리미엄 가속도 > surge_threshold → 국내 수급 폭발 조짐"""
        with self._lock:
            return self._acceleration > self.surge_threshold


# ─── Fear & Greed Index ───────────────────────────────────────────────────────

class FearGreedIndex:
    """Alternative.me 크립토 공포-탐욕 지수 (0=극도공포, 100=극도탐욕)"""

    _URL = "https://api.alternative.me/fng/?limit=2"  # 2개: 전일 delta 계산용

    # 마지막 성공 fetch로부터 이 시간(초) 이상 경과하면 stale 플래그 ON
    STALENESS_THRESHOLD = 7200  # 2시간

    def __init__(self, cache_ttl: int = 3600,
                 extreme_greed: int = 80, greed: int = 65,
                 ts_extreme_ratio: float = 0.5,
                 ts_greed_ratio: float = 0.75):
        self.cache_ttl = cache_ttl
        self.extreme_greed = extreme_greed
        self.greed = greed
        self.ts_extreme_ratio = ts_extreme_ratio
        self.ts_greed_ratio = ts_greed_ratio

        self._value: int = 50
        self._prev_value: int = 50
        self._label: str = "Neutral"
        self._last_fetch: float = 0.0
        self._last_success: float = 0.0
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(
            target=self._background_refresh, daemon=True, name="FearGreedRefresh"
        )
        self._thread.start()

    def _do_fetch(self) -> None:
        """Alternative.me API 조회 후 캐시 갱신 — 백그라운드 스레드 전용."""
        try:
            r = requests.get(self._URL, timeout=10)
            data = r.json()["data"]
            value = int(data[0]["value"])
            label = data[0]["value_classification"]
            prev_value = int(data[1]["value"]) if len(data) >= 2 else value

            now = time.time()
            with self._lock:
                self._prev_value = prev_value
                self._value = value
                self._label = label
                self._last_fetch = now
                self._last_success = now

            delta = value - prev_value
            logger.info(f"[F&G] 공포탐욕 지수: {value} ({label}) Δ{delta:+d}")

        except Exception as e:
            logger.warning(f"공포-탐욕 지수 조회 실패: {e}")
            with self._lock:
                stale_secs = time.time() - self._last_success
                if self._last_success > 0 and stale_secs > self.STALENESS_THRESHOLD:
                    logger.warning(
                        f"[F&G] 데이터 {stale_secs/3600:.1f}시간 stale — "
                        f"신호 신뢰도 저하, 중립(50)으로 대체"
                    )
                    self._value = 50
                    self._label = "Neutral (stale)"

    def _background_refresh(self) -> None:
        while self._running:
            self._do_fetch()
            time.sleep(self.cache_ttl)

    def fetch(self) -> Tuple[int, str]:
        """캐시된 (value, label) 즉시 반환 (non-blocking).
        백그라운드 스레드가 cache_ttl마다 자동 갱신.
        """
        with self._lock:
            return self._value, self._label

    def shutdown(self) -> None:
        self._running = False

    @property
    def is_stale(self) -> bool:
        """마지막 성공 fetch가 STALENESS_THRESHOLD 초 이상 경과했으면 True.
        첫 fetch 전(_last_success==0)에는 False 반환 — stale 오판정 방지.
        """
        with self._lock:
            return self._last_success > 0 and (time.time() - self._last_success) > self.STALENESS_THRESHOLD

    def get_regime_signal(self) -> dict:
        """
        지수 수준 + 변화 속도(delta)를 결합한 복합 시장 신호.
        급격한 변화 방향이 단순 임계값보다 더 강력한 예측 신호.
        """
        value, label = self.fetch()
        with self._lock:
            prev = self._prev_value
        delta = value - prev  # 하루 변화량 (양수=탐욕 증가, 음수=공포 증가)

        return {
            "value": value,
            "label": label,
            "delta": delta,
            # 지수 ≥80 & 하루 +10 이상 급등: 과열 가속 → 포지션 축소 신호
            "is_overheating": value >= self.extreme_greed and delta >= 10,
            # 지수 ≤20 & 하루 -15 이상 급락: 패닉 매도 → 반등 기대 (반전 후보)
            "is_panic": value <= 20 and delta <= -15,
        }

    def dynamic_trailing_stop(self, base_pct: float) -> float:
        """
        공포-탐욕 지수에 따른 동적 트레일링 스탑.
        극도 탐욕(≥80): base × 0.5  (수익 빠르게 확정)
        탐욕(≥65)      : base × 0.75
        중립/공포       : base 유지
        """
        value, label = self.fetch()
        if value >= self.extreme_greed:
            result = base_pct * self.ts_extreme_ratio
            logger.debug(f"[F&G] 극도 탐욕({value}) → 트레일링 스탑 {result*100:.1f}%")
            return result
        if value >= self.greed:
            return base_pct * self.ts_greed_ratio
        return base_pct


# ─── HMM Regime Detector ─────────────────────────────────────────────────────

class HMMRegimeDetector:
    """
    GaussianHMM 기반 시장 국면 감지 (bull / sideways / bear).
    hmmlearn 미설치 시 → 볼린저 밴드 폭 + SMA 기울기 폴백.
    """

    BULL     = "bull"
    SIDEWAYS = "sideways"
    BEAR     = "bear"

    def __init__(self, n_states: int = 3, lookback: int = 500,
                 retrain_interval: int = 3600,
                 threshold_bear: float = 0.75,
                 threshold_sideways: float = 0.0,
                 threshold_bull: float = 0.65,
                 position_multiplier_bear: float = 0.5,
                 position_multiplier_sideways: float = 0.7):
        self.n_states = n_states
        self.lookback = lookback
        self.retrain_interval = retrain_interval
        self.threshold_bear = threshold_bear
        self.threshold_sideways = threshold_sideways
        self.threshold_bull = threshold_bull
        self.position_multiplier_bear = position_multiplier_bear
        self.position_multiplier_sideways = position_multiplier_sideways

        self._regime: str = self.SIDEWAYS
        self._model = None
        self._last_train: float = 0.0
        self._use_hmm: bool = self._check_hmmlearn()
        self._lock = threading.Lock()
        try:
            import config as _mc_cfg
            _models_dir = _mc_cfg.DIRECTORIES.get("models", "models")
        except Exception:
            _models_dir = "models"
        self._model_path = os.path.join(_models_dir, "hmm_regime_detector.pkl")

        self._load_persisted_model()

    # ------------------------------------------------------------------

    def _load_persisted_model(self) -> None:
        """재시작 시 직전 학습 모델 복원 — cold-start 방지."""
        if not os.path.exists(self._model_path):
            return
        try:
            with open(self._model_path, "rb") as f:
                saved = pickle.load(f)
            if not isinstance(saved, dict) or "model" not in saved:
                logger.warning(f"HMM 모델 파일 스키마 불일치, 무시하고 재학습: {self._model_path}")
                return
            model = saved["model"]
            last_train = saved.get("last_train", 0.0)
            if not isinstance(last_train, (int, float)):
                raise ValueError(f"last_train 타입 오류: {type(last_train)}")
            with self._lock:
                self._model = model
                self._last_train = float(last_train)
            logger.info(
                f"✅ HMM 모델 복원: {self._model_path} "
                f"(학습 시각: {time.strftime('%Y-%m-%d %H:%M', time.localtime(self._last_train))})"
            )
        except Exception as e:
            logger.warning(f"HMM 모델 로드 실패 (무시하고 재학습 대기): {e}")

    def _persist_model(self) -> None:
        """학습된 모델을 디스크에 저장 (원자적 쓰기, 락 보호)."""
        try:
            with self._lock:
                model_snap = self._model
                last_train_snap = self._last_train
            model_dir = os.path.dirname(self._model_path)
            if model_dir:
                os.makedirs(model_dir, exist_ok=True)
            tmp = self._model_path + ".tmp"
            with open(tmp, "wb") as f:
                pickle.dump({"model": model_snap, "last_train": last_train_snap}, f)
            os.replace(tmp, self._model_path)
        except Exception as e:
            logger.warning(f"HMM 모델 저장 실패: {e}")

    # ------------------------------------------------------------------

    def _check_hmmlearn(self) -> bool:
        try:
            import hmmlearn  # noqa: F401
            return True
        except ImportError:
            logger.warning(
                "hmmlearn 미설치 → BB폭+SMA 기울기 폴백 사용. "
                "'pip install hmmlearn'으로 HMM 활성화 가능."
            )
            return False

    def _features(self, df: pd.DataFrame) -> np.ndarray:
        """HMM 입력 피처: [log_return, rolling_vol_20]"""
        close = df["close"].values.astype(float)
        log_ret = np.diff(np.log(np.maximum(close, 1e-10)))
        log_ret = np.insert(log_ret, 0, 0.0)
        vol = pd.Series(log_ret).rolling(20, min_periods=1).std().fillna(0).values
        return np.column_stack([log_ret, vol])

    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> None:
        """최근 OHLCV 데이터로 HMM 재학습 (백그라운드 스레드에서 호출)"""
        if not self._use_hmm:
            return
        try:
            from hmmlearn.hmm import GaussianHMM
            X = self._features(df.tail(self.lookback))
            model = GaussianHMM(
                n_components=self.n_states,
                covariance_type="diag",
                n_iter=200,
                random_state=42,
                tol=1e-3,
            )
            model.fit(X)
            with self._lock:
                self._model = model
                self._last_train = time.time()
            self._persist_model()
            logger.info(f"✅ HMM 재학습 완료 ({self.n_states}개 상태 | lookback={len(X)}봉)")
        except Exception as e:
            logger.warning(f"HMM 학습 실패: {e}")

    def predict(self, df: pd.DataFrame) -> str:
        """현재 시장 국면 반환: 'bull' | 'sideways' | 'bear'"""
        if self._use_hmm:
            regime = self._predict_hmm(df)
        else:
            regime = self._predict_fallback(df)
        with self._lock:
            self._regime = regime
        return regime

    def _predict_hmm(self, df: pd.DataFrame) -> str:
        with self._lock:
            model = self._model
        if model is None:
            logger.info("HMMRegimeDetector: 모델 미학습 — BB+SMA 폴백 사용 (첫 학습 전)")
            return self._predict_fallback(df)
        try:
            window = min(120, len(df))
            X = self._features(df.tail(window))
            states = model.predict(X)
            current_state = int(states[-1])

            # 상태별 평균 log return으로 bull/bear 분류
            log_ret = X[:, 0]
            state_means = {
                s: log_ret[states == s].mean() if (states == s).any() else 0.0
                for s in range(self.n_states)
            }
            sorted_s = sorted(state_means, key=state_means.get)
            bear_s = sorted_s[0]
            bull_s = sorted_s[-1]

            if self.n_states == 1 or bull_s == bear_s:
                return self.SIDEWAYS
            if current_state == bull_s:
                return self.BULL
            if current_state == bear_s:
                return self.BEAR
            return self.SIDEWAYS

        except Exception as e:
            logger.warning(f"HMM 예측 실패: {e} → BB+SMA 폴백")
            return self._predict_fallback(df)

    def _predict_fallback(self, df: pd.DataFrame) -> str:
        """볼린저 밴드 폭 + SMA20 기울기 기반 폴백 국면 판단"""
        try:
            close = df["close"].tail(60)
            sma20 = close.rolling(20).mean()
            # SMA 기울기: 최근 5봉 변화율
            slope = (sma20.iloc[-1] - sma20.iloc[-6]) / sma20.iloc[-6] if len(sma20) >= 6 else 0
            above_sma = close.iloc[-1] > sma20.iloc[-1]

            if above_sma and slope > 0.002:
                return self.BULL
            if not above_sma and slope < -0.002:
                return self.BEAR
            return self.SIDEWAYS
        except Exception:
            return self.SIDEWAYS

    # ------------------------------------------------------------------

    @property
    def regime(self) -> str:
        with self._lock:
            return self._regime

    def get_entry_threshold(self, base_threshold: float) -> float:
        return self.get_entry_threshold_for(base_threshold, self.regime)

    def get_entry_threshold_for(self, base_threshold: float, regime: str) -> float:
        """스냅샷 regime 기준 진입 임계값 — 스레드 안전"""
        if regime == self.BEAR:
            return self.threshold_bear
        if regime == self.SIDEWAYS:
            return self.threshold_sideways if self.threshold_sideways > 0 else base_threshold
        return base_threshold

    def get_position_multiplier(self) -> float:
        return self.get_position_multiplier_for(self.regime)

    def get_position_multiplier_for(self, regime: str) -> float:
        """스냅샷 regime 기준 포지션 배수 — 스레드 안전"""
        if regime == self.BEAR:
            return self.position_multiplier_bear
        if regime == self.SIDEWAYS:
            return self.position_multiplier_sideways
        return 1.0

    def load_from_wfo_checkpoint(self, checkpoint_path: str) -> bool:
        """
        feature_engineering.py의 WFO HMM이 저장한 체크포인트를 로드.
        학습-추론 HMM 이중 시스템의 국면 불일치를 최소화하는 핵심 메서드.
        WFO 파이프라인 실행 후 market_context HMM 초기화에 사용.

        Args:
            checkpoint_path: feature_engineering WFO HMM 저장 경로 (pickle)

        Returns:
            bool: 로드 성공 여부
        """
        if not os.path.exists(checkpoint_path):
            logger.warning(f"[HMMRegime] WFO 체크포인트 없음: {checkpoint_path}")
            return False
        try:
            with open(checkpoint_path, "rb") as f:
                saved = pickle.load(f)
            if not isinstance(saved, dict) or "model" not in saved:
                logger.warning(f"[HMMRegime] WFO 체크포인트 스키마 불일치: {checkpoint_path}")
                return False
            last_train = saved.get("last_train", time.time())
            if not isinstance(last_train, (int, float)):
                raise ValueError(f"last_train 타입 오류: {type(last_train)}")
            with self._lock:
                self._model = saved["model"]
                self._last_train = float(last_train)
            logger.info(
                f"✅ [HMMRegime] WFO 체크포인트 로드 완료: {checkpoint_path} "
                f"(학습 시각: {time.strftime('%Y-%m-%d %H:%M', time.localtime(self._last_train))})"
            )
            return True
        except Exception as e:
            logger.warning(f"[HMMRegime] WFO 체크포인트 로드 실패: {e}")
            return False

    def needs_retrain(self) -> bool:
        with self._lock:
            return time.time() - self._last_train > self.retrain_interval


# ─── Liquidity Flow Monitor ───────────────────────────────────────────────────

class LiquidityFlowMonitor:
    """
    매크로 유동성 흐름 3단 게이트.

    신호 1: 암호화폐 총 시총 N일 변화 > 0  (신규 자금 유입)
    신호 2: BTC 도미넌스 M일 변화 ≤ threshold  (알트 로테이션)
    신호 3: 코인 24h 거래대금 > 7일 평균 × ratio  (로컬 축적, 호출자 전달)

    min_signals 이상 충족 시 check_gate() → True.
    API 실패·데이터 부족 → 보수적 True 폴백 (진입 차단 최소화).
    """

    _GLOBAL_URL = "https://api.coingecko.com/api/v3/global"

    def __init__(self, cfg: dict):
        self._enabled          = cfg.get("enabled", True)
        self._cache_ttl        = cfg.get("cache_ttl", 3600)
        self._total_mcap_days  = cfg.get("total_mcap_days", 7)
        self._btc_dom_days     = cfg.get("btc_dom_days", 5)
        self._btc_dom_threshold = cfg.get("btc_dom_threshold", -0.5)  # %p
        self._volume_surge_ratio = cfg.get("volume_surge_ratio", 1.5)
        self._min_signals      = cfg.get("min_signals", 2)

        max_hours = int(max(self._total_mcap_days, self._btc_dom_days) * 24) + 25
        self._history: deque = deque(maxlen=max_hours)  # (ts, total_mcap_usd, btc_dom_pct)
        self._last_fetch = 0.0
        self._lock = threading.Lock()

        if self._enabled:
            threading.Thread(target=self._refresh_loop, daemon=True,
                             name="LiqFlowMonitor").start()

    # ── 내부 ──────────────────────────────────────────────────────────────────

    def _refresh_loop(self) -> None:
        while True:
            try:
                self._do_fetch()
            except Exception as e:
                logger.warning(f"[LiqFlow] 갱신 실패: {e}")
            time.sleep(self._cache_ttl)

    def _do_fetch(self) -> None:
        resp = requests.get(self._GLOBAL_URL, timeout=10).json()
        data = resp["data"]
        total_mcap = float(data["total_market_cap"]["usd"])
        btc_dom    = float(data["market_cap_percentage"]["btc"])
        now = time.time()
        with self._lock:
            self._history.append((now, total_mcap, btc_dom))
            self._last_fetch = now
        logger.debug(f"[LiqFlow] 총시총={total_mcap/1e12:.2f}T$ BTC.D={btc_dom:.2f}%")

    def _compute_macro(self) -> Tuple[Optional[bool], Optional[bool]]:
        """스레드 안전 히스토리 분석 → (mcap_growing, btc_dom_declining)."""
        with self._lock:
            if not self._history:
                return None, None
            latest = self._history[-1]
            now_ts, now_mcap, now_dom = latest

            # Signal 1: 총 시총 N일 전 대비 성장
            cutoff1 = now_ts - self._total_mcap_days * 86400
            old1 = next(
                ((ts, mc, bd) for ts, mc, bd in reversed(self._history) if ts <= cutoff1),
                None
            )
            s1 = (now_mcap > old1[1]) if old1 else None

            # Signal 2: BTC.D M일 전 대비 하락
            cutoff2 = now_ts - self._btc_dom_days * 86400
            old2 = next(
                ((ts, mc, bd) for ts, mc, bd in reversed(self._history) if ts <= cutoff2),
                None
            )
            s2 = ((now_dom - old2[2]) <= self._btc_dom_threshold) if old2 else None

        return s1, s2

    # ── Public ────────────────────────────────────────────────────────────────

    def check_gate(self, volume_surge: bool) -> Tuple[bool, int, str]:
        """
        진입 허용 여부.
        volume_surge: 호출자가 Upbit OHLCV로 계산한 코인별 거래대금 급증 여부.
        Returns: (allowed, signal_count, reason_str)
        """
        if not self._enabled:
            return True, -1, "disabled"

        stale = time.time() - self._last_fetch > self._cache_ttl * 2
        s1, s2 = self._compute_macro()

        if stale and s1 is None and s2 is None:
            return True, -1, "stale/no_data"

        r1 = s1 if s1 is not None else True
        r2 = s2 if s2 is not None else True
        r3 = volume_surge

        count   = int(r1) + int(r2) + int(r3)
        allowed = count >= self._min_signals
        reason  = (
            f"{count}/3 [MCap{'↑' if r1 else '↓'}"
            f" BTC.D{'↓' if r2 else '↑'}"
            f" Vol{'↑' if r3 else '→'}]"
        )
        return allowed, count, reason

    def get_volume_surge_ratio(self) -> float:
        return self._volume_surge_ratio


# ─── Liquidity Scanner ────────────────────────────────────────────────────────

class LiquidityScanner:
    """
    업비트 KRW 전 종목 실시간 유동성 모니터링.
    유동성 필터(상위 30개) 후 알파 스코어 랭킹으로 목표 코인 선정.
    알파 = OOS Sharpe(0.4) + 모델 정밀도 엣지(0.3) + 5일 모멘텀(0.3)
    """

    _MARKET_URL   = "https://api.upbit.com/v1/market/all?isDetails=false"
    _TICKER_URL   = "https://api.upbit.com/v1/ticker"
    _EXCLUDE      = {"KRW-USDT", "KRW-USDC"}
    _LIQ_POOL_N   = 30    # 유동성 필터 통과 후보 수
    _ALPHA_W      = {"sharpe": 0.4, "precision": 0.3, "momentum": 0.3}
    _WHITELIST_PATH = os.path.join(os.path.dirname(__file__), "data", "coin_whitelist.json")

    def __init__(self, check_interval: int = 300, top_n: int = 10,
                 surge_rank_jump: int = 5):
        self.check_interval = check_interval
        self.top_n = top_n
        self.surge_rank_jump = surge_rank_jump

        self._candidates: List[str] = []
        self._top_coins: List[str] = []
        self._prev_top: List[str] = []
        self._lock = threading.Lock()           # top_coins / candidates 읽기/쓰기 보호
        self._fetch_lock = threading.Lock()     # 동시 API 호출 방지 (중복 요청 차단)
        self._last_fetch_ts: float = 0.0        # rate-limit: 최소 60초 간격 강제
        self._MIN_FETCH_INTERVAL = 60.0
        self._running = True
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="LiquidityScanner"
        )
        self._thread.start()

    def _get_krw_markets(self) -> List[str]:
        try:
            markets = requests.get(self._MARKET_URL, timeout=5).json()
            return [
                m["market"] for m in markets
                if m["market"].startswith("KRW-")
                and m["market"] not in self._EXCLUDE
            ]
        except Exception:
            return ["KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-SOL"]

    def _fetch_top(self) -> List[str]:
        if not self._fetch_lock.acquire(blocking=False):
            logger.debug("유동성 스캐너: 다른 스레드가 fetch 중 — 캐시 반환")
            with self._lock:
                return list(self._top_coins)
        try:
            now = time.time()
            if now - self._last_fetch_ts < self._MIN_FETCH_INTERVAL:
                with self._lock:
                    return list(self._top_coins)
            self._last_fetch_ts = now

            markets = self._get_krw_markets()
            # Upbit ticker API 최대 100개 — 100개씩 청크로 조회
            all_tickers = []
            for i in range(0, len(markets), 100):
                chunk = markets[i:i + 100]
                resp = requests.get(
                    self._TICKER_URL,
                    params={"markets": ",".join(chunk)},
                    timeout=5,
                ).json()
                if isinstance(resp, list):
                    all_tickers.extend(resp)
                time.sleep(0.1)

            sorted_t = sorted(
                all_tickers,
                key=lambda x: x.get("acc_trade_price_24h", 0),
                reverse=True,
            )
            return [t["market"] for t in sorted_t]
        except Exception as e:
            logger.warning(f"유동성 스캐너: 티커 조회 실패 {e}")
            return []
        finally:
            self._fetch_lock.release()

    def _load_whitelist_scores(self) -> dict:
        """coin_whitelist.json에서 {COIN: sharpe} 반환. 없으면 {}."""
        try:
            with open(self._WHITELIST_PATH, "r", encoding="utf-8") as f:
                wl = json.load(f)
            scores = wl.get("per_coin_scores", {})
            return {
                k.replace("_minute60", ""): v.get("sharpe", 0.0)
                for k, v in scores.items()
            }
        except Exception:
            return {}

    def _load_model_precision(self, coin: str) -> float:
        """카테고리별 최신 model config에서 OOS precision 반환. 없으면 0.0."""
        try:
            import config as _cfg
            model_dir = _cfg.get_model_dir(coin)
            configs = sorted(glob.glob(os.path.join(model_dir, "config_*.json")))
            if not configs:
                return 0.0
            with open(configs[-1], "r", encoding="utf-8") as f:
                cfg = json.load(f)
            return float(cfg.get("oos_metrics", {}).get("precision", 0.0))
        except Exception:
            return 0.0

    def _load_momentum(self, coin: str, days: int = 5) -> float:
        """5일 가격 모멘텀 (수익률). API 실패 시 0.0."""
        try:
            df = pyupbit.get_ohlcv(coin, interval="day", count=days + 1)
            if df is None or len(df) < days + 1:
                return 0.0
            return float(df["close"].iloc[-1] / df["close"].iloc[0] - 1)
        except Exception:
            return 0.0

    def _rank_by_alpha(self, coins: List[str]) -> List[str]:
        """유동성 필터 통과 코인을 알파 스코어 순으로 정렬."""
        sharpe_map = self._load_whitelist_scores()

        raw: dict = {c: {"sharpe": 0.0, "precision": 0.0, "momentum": 0.0} for c in coins}
        coin_sym = {c: c.split("-")[1] for c in coins}

        for c in coins:
            raw[c]["sharpe"]    = sharpe_map.get(coin_sym[c], 0.0)
            raw[c]["precision"] = self._load_model_precision(coin_sym[c])

        # 모멘텀은 병렬 페칭 후 cap 적용 (급등 코인 점수 독식 방지)
        _MOMENTUM_CAP = 0.30  # 5일 +30% 초과분은 점수 기여 없음
        with ThreadPoolExecutor(max_workers=8) as ex:
            futs = {ex.submit(self._load_momentum, c): c for c in coins}
            for fut in as_completed(futs):
                c = futs[fut]
                raw[c]["momentum"] = min(fut.result(), _MOMENTUM_CAP)

        def _minmax(values: list) -> list:
            lo, hi = min(values), max(values)
            if hi == lo:
                return [0.5] * len(values)
            return [(v - lo) / (hi - lo) for v in values]

        keys = list(raw.keys())
        for metric in ("sharpe", "precision", "momentum"):
            vals  = [raw[c][metric] for c in keys]
            norms = _minmax(vals)
            for c, n in zip(keys, norms):
                raw[c][metric] = n

        scores = {
            c: (self._ALPHA_W["sharpe"]    * raw[c]["sharpe"] +
                self._ALPHA_W["precision"] * raw[c]["precision"] +
                self._ALPHA_W["momentum"]  * raw[c]["momentum"])
            for c in keys
        }
        ranked = sorted(keys, key=lambda c: scores[c], reverse=True)
        logger.info(
            "🔭 [알파 랭킹] " +
            " | ".join(f"{c.split('-')[1]}:{scores[c]:.2f}" for c in ranked[:10])
        )
        return ranked

    def _run(self):
        while self._running:
            try:
                # 1단계: 24h 거래대금 상위 _LIQ_POOL_N개 — 최소 유동성 필터
                liq_pool = self._fetch_top()
                if not liq_pool:
                    time.sleep(self.check_interval)
                    continue
                liq_pool = liq_pool[: self._LIQ_POOL_N]

                # 1.5단계: OOS 화이트리스트 필터 — 백테스트 검증 완료 코인만 알파풀 진입
                try:
                    with open(self._WHITELIST_PATH, "r", encoding="utf-8") as _f:
                        _wl_raw = json.load(_f)
                    _wl_coins = {k.replace("_minute60", "") for k in _wl_raw.get("whitelist", [])}
                    if _wl_coins:
                        liq_pool = [c for c in liq_pool if c.split("-")[-1] in _wl_coins]
                except Exception:
                    pass

                # 2단계: 알파 스코어 랭킹으로 재정렬
                current = self._rank_by_alpha(liq_pool)

                with self._lock:
                    prev = self._prev_top or current
                    prev_rank = {coin: i for i, coin in enumerate(prev)}
                    new_candidates = []

                    for i, coin in enumerate(current[: self.top_n]):
                        old_rank = prev_rank.get(coin, len(prev))
                        if old_rank >= self.top_n + self.surge_rank_jump and i < self.top_n // 2:
                            new_candidates.append(coin)
                            logger.info(
                                f"🔥 [유동성 로테이션] {coin}: "
                                f"순위 {old_rank+1}위 → {i+1}위 급등"
                            )

                    self._candidates = new_candidates
                    self._prev_top   = current
                    self._top_coins  = current[: self.top_n]

            except Exception as e:
                logger.warning(f"유동성 스캐너 루프 오류: {e}")

            time.sleep(self.check_interval)

    def get_rotation_candidates(self) -> List[str]:
        with self._lock:
            return list(self._candidates)

    def get_top_coins(self, n: int = 5) -> List[str]:
        with self._lock:
            return self._top_coins[:n]

    def shutdown(self):
        self._running = False
        logger.info("유동성 스캐너 종료")
# ─── News Sentiment Analyzer (Kill-Switch) ───────────────────────────────────

class NewsSentimentAnalyzer:
    """
    무거운 LLM 감성 분석을 배제한 초경량 킬 스위치 어댑터.
    BTC 1시간 낙폭이 btc_drop_threshold 이하이면 킬 스위치를 ON으로 전환한다.
    (추후 특정 거래소 공지 웹훅이나 단순 패닉 지수 API로 확장이 가능하도록 인터페이스만 유지)
    """
    def __init__(self, cache_ttl: int = 300, btc_drop_threshold: float = -0.05):
        self.cache_ttl = cache_ttl
        self._BTC_DROP_THRESHOLD = btc_drop_threshold
        self._is_kill_switch_active = False
        self._sentiment_score = 0.0
        self._last_fetch = 0.0
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(
            target=self._background_refresh, daemon=True, name="KillSwitchRefresh"
        )
        self._thread.start()

    def _do_fetch(self) -> None:
        """BTC 1시간 낙폭 조회 후 캐시 갱신 — 백그라운드 스레드 전용."""
        with self._lock:
            score = self._sentiment_score
            kill = self._is_kill_switch_active

        try:
            df = pyupbit.get_ohlcv("KRW-BTC", interval="minute60", count=2)
            if df is not None and len(df) >= 2:
                prev_close = float(df["close"].iloc[-2])
                curr_close = float(df["close"].iloc[-1])
                if prev_close > 0:
                    change = (curr_close - prev_close) / prev_close
                    score = float(change)
                    kill = change <= self._BTC_DROP_THRESHOLD
                    if kill:
                        logger.warning(
                            f"[KillSwitch] BTC 1시간 낙폭 {change*100:.2f}% ≤ "
                            f"{self._BTC_DROP_THRESHOLD*100:.0f}% → 킬 스위치 ON"
                        )
            else:
                logger.warning("[KillSwitch] BTC OHLCV 데이터 부족 — 이전 상태 유지")
        except Exception as e:
            logger.warning(f"[KillSwitch] BTC 낙폭 조회 실패: {e} — 이전 킬 스위치 상태 유지")

        with self._lock:
            self._sentiment_score = score
            self._is_kill_switch_active = kill
            self._last_fetch = time.time()

    def _background_refresh(self) -> None:
        while self._running:
            self._do_fetch()
            time.sleep(self.cache_ttl)

    def fetch(self) -> tuple[float, bool]:
        """캐시된 (sentiment_score, is_kill_switch_active) 즉시 반환 (non-blocking).
        백그라운드 스레드가 cache_ttl마다 자동 갱신.
        """
        with self._lock:
            return self._sentiment_score, self._is_kill_switch_active

    def shutdown(self) -> None:
        self._running = False


# ─── Upbit Risk Manager ───────────────────────────────────────────────────────

class UpbitRiskManager:
    """
    업비트 거래소 리스크 안전장치.
    1) v1/status/wallet  : 입출금 락업 감지 (JWT 인증 필요)
    2) v1/market/all     : DAXA 투자유의종목(CAUTION) 지정 여부
    두 조건 중 하나라도 위험하면 is_safe_to_trade() → False.
    """

    _WALLET_URL = "https://api.upbit.com/v1/status/wallet"
    _MARKET_URL = "https://api.upbit.com/v1/market/all?isDetails=true"

    def __init__(self, access_key: str, secret_key: str, cache_ttl: int = 120):
        self.access_key = access_key
        self.secret_key = secret_key
        self.cache_ttl = cache_ttl
        self._cache: dict = {}          # ticker → (is_safe, expire_ts)
        self._lock = threading.Lock()

    def _jwt_header(self) -> dict:
        try:
            import jwt as pyjwt
        except ImportError:
            logger.warning("[UpbitRisk] PyJWT 미설치. pip install PyJWT")
            return {}
        payload = {"access_key": self.access_key, "nonce": str(uuid.uuid4())}
        token = pyjwt.encode(payload, self.secret_key, algorithm="HS256")
        if isinstance(token, bytes):
            token = token.decode("utf-8")
        return {"Authorization": f"Bearer {token}"}

    def _fetch_wallet_state(self, currency: str) -> bool:
        """wallet_state == 'working' AND block_state == 'normal' 이면 True."""
        try:
            resp = requests.get(self._WALLET_URL, headers=self._jwt_header(), timeout=5)
            if resp.status_code != 200:
                logger.warning(f"[UpbitRisk] 지갑 API {resp.status_code} → 보수적 차단")
                return False
            for asset in resp.json():
                if asset.get("currency") == currency:
                    ok = (
                        asset.get("wallet_state") == "working"
                        and asset.get("block_state") == "normal"
                    )
                    if not ok:
                        logger.warning(
                            f"[UpbitRisk] {currency} 지갑 락업 "
                            f"(wallet={asset.get('wallet_state')}, "
                            f"block={asset.get('block_state')})"
                        )
                    return ok
            return True  # 목록에 없으면 별도 제한 없음으로 판단
        except Exception as e:
            logger.warning(f"[UpbitRisk] 지갑 상태 조회 실패: {e} → 보수적 차단")
            return False

    def _fetch_market_warning(self, ticker: str) -> bool:
        """DAXA market_warning == 'CAUTION' 이면 False."""
        try:
            resp = requests.get(self._MARKET_URL, timeout=5)
            if resp.status_code != 200:
                return True  # 조회 실패 시 경고 없음으로 관대하게 처리
            for m in resp.json():
                if m.get("market") == ticker:
                    if m.get("market_warning") == "CAUTION":
                        logger.warning(f"[UpbitRisk] {ticker} DAXA 투자유의종목 지정 → 차단")
                        return False
                    return True
            return True
        except Exception as e:
            logger.warning(f"[UpbitRisk] 마켓 경고 조회 실패: {e}")
            return True

    def is_safe_to_trade(self, ticker: str) -> bool:
        """
        ticker 예: 'KRW-BTC'.
        캐시 TTL 내이면 캐시값 반환, 만료 시 wallet·market 조회 병렬 실행.
        """
        now = time.time()
        with self._lock:
            cached = self._cache.get(ticker)
            if cached and now < cached[1]:
                return cached[0]

        currency = ticker.split("-")[-1]
        with ThreadPoolExecutor(max_workers=2) as ex:
            f_wallet = ex.submit(self._fetch_wallet_state, currency)
            f_market = ex.submit(self._fetch_market_warning, ticker)
            wallet_ok = f_wallet.result()
            market_ok = f_market.result()
        result = wallet_ok and market_ok

        with self._lock:
            self._cache[ticker] = (result, now + self.cache_ttl)

        if result:
            logger.debug(f"[UpbitRisk] {ticker} 안전 확인 ✅")
        return result


# ─── K-Means Market Regime Detector ─────────────────────────────────────────

class KMeansRegimeDetector:
    """
    비지도 학습 기반 시장 국면 감지 (K-Means Clustering).
    HMMRegimeDetector 전단에서 고변동성 횡보장을 조기 차단하는 보조 필터.
    피처: [log_return, rolling_vol_14, rsi_14_norm]
    국면 0 = 저변동/횡보, 1 = 고변동, 2 = 추세 (평균 수익률 기준 정렬)
    """

    TREND   = "trend"
    CHOP    = "chop"
    UNKNOWN = "unknown"

    def __init__(self, n_clusters: int = 3, lookback: int = 200,
                 retrain_interval: int = 1800):
        self.n_clusters = n_clusters
        self.lookback = lookback
        self.retrain_interval = retrain_interval
        self._model = None
        self._regime = self.UNKNOWN
        self._last_train: float = 0.0
        self._lock = threading.Lock()
        try:
            import config as _mc_cfg
            _models_dir = _mc_cfg.DIRECTORIES.get("models", "models")
        except Exception:
            _models_dir = "models"
        self._MODEL_PATH = os.path.join(_models_dir, "kmeans_regime_detector.pkl")
        self._scaler = None
        self._load_persisted_model()

    def _load_persisted_model(self) -> None:
        """재시작 시 직전 모델 복원 — cold-start 불안정 구간(약 30분) 제거."""
        if not os.path.exists(self._MODEL_PATH):
            return
        try:
            with open(self._MODEL_PATH, "rb") as f:
                saved = pickle.load(f)
            if not isinstance(saved, dict) or "model" not in saved:
                logger.warning(f"KMeans 모델 파일 스키마 불일치, 무시하고 재학습: {self._MODEL_PATH}")
                return
            model = saved["model"]
            last_train = saved.get("last_train", 0.0)
            if not isinstance(last_train, (int, float)):
                raise ValueError(f"last_train 타입 오류: {type(last_train)}")
            with self._lock:
                self._model = model
                self._scaler = saved.get("scaler", None)
                self._last_train = float(last_train)
            logger.info(
                f"✅ KMeans 모델 복원: {self._MODEL_PATH} "
                f"(학습 시각: {time.strftime('%Y-%m-%d %H:%M', time.localtime(self._last_train))})"
            )
        except Exception as e:
            logger.warning(f"KMeans 모델 로드 실패 (무시하고 재학습 대기): {e}")

    def _persist_model(self) -> None:
        """학습된 모델을 디스크에 저장 (원자적 쓰기, 락 보호). scaler 포함 저장 — leakage 방지."""
        try:
            with self._lock:
                model_snap = self._model
                scaler_snap = self._scaler
                last_train_snap = self._last_train
            model_dir = os.path.dirname(self._MODEL_PATH)
            if model_dir:
                os.makedirs(model_dir, exist_ok=True)
            tmp = self._MODEL_PATH + ".tmp"
            with open(tmp, "wb") as f:
                pickle.dump({
                    "model": model_snap,
                    "scaler": scaler_snap,
                    "last_train": last_train_snap,
                }, f)
            os.replace(tmp, self._MODEL_PATH)
        except Exception as e:
            logger.warning(f"KMeans 모델 저장 실패: {e}")

    @staticmethod
    def _extract_features(df: pd.DataFrame):
        """(X_scaled, scaler) 반환. 실패 시 (None, None). scaler를 함께 반환하여 재시작 후 재사용 가능."""
        try:
            from sklearn.preprocessing import StandardScaler
            close = df["close"].values.astype(float)
            log_ret = np.diff(np.log(np.maximum(close, 1e-10)))
            log_ret = np.insert(log_ret, 0, 0.0)
            vol = pd.Series(log_ret).rolling(14, min_periods=1).std().fillna(0).values

            delta = pd.Series(close).diff().fillna(0)
            gain = delta.clip(lower=0).rolling(14, min_periods=1).mean()
            loss = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
            rs = gain / (loss + 1e-10)
            rsi = (100 - 100 / (1 + rs)).values / 100.0

            X = np.column_stack([log_ret, vol, rsi])
            scaler = StandardScaler()
            return scaler.fit_transform(X), scaler
        except Exception as e:
            logger.warning(f"[KMeans] 피처 추출 실패: {e}")
            return None, None

    def fit_predict(self, df: pd.DataFrame) -> str:
        """재학습이 필요한 경우에만 재학습 후 현재 국면 반환. 불필요 시 캐시 즉시 반환."""
        if not self.needs_retrain():
            return self.regime

        if df is None or df.empty or "close" not in df.columns:
            return self.UNKNOWN

        try:
            from sklearn.cluster import KMeans
        except ImportError:
            logger.warning("[KMeans] scikit-learn 미설치")
            return self.UNKNOWN

        try:
            window = df.tail(self.lookback)
            X, scaler = self._extract_features(window)
            if X is None or len(X) < self.n_clusters * 10:
                return self.UNKNOWN

            model = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
            labels = model.fit_predict(X)

            # 클러스터별 평균 log_return으로 추세/횡보 분류
            close = window["close"].values.astype(float)
            log_ret = np.diff(np.log(np.maximum(close, 1e-10)))
            log_ret = np.insert(log_ret, 0, 0.0)

            cluster_ret = {
                c: log_ret[labels == c].mean() for c in range(self.n_clusters)
            }
            # 변동성 기준: 고변동 클러스터 = 횡보(CHOP)
            cluster_vol = {
                c: np.std(log_ret[labels == c]) for c in range(self.n_clusters)
            }
            max_vol_cluster = max(cluster_vol, key=cluster_vol.get)
            current = int(labels[-1])

            with self._lock:
                self._model = model
                self._scaler = scaler
                self._last_train = time.time()
                if current == max_vol_cluster:
                    self._regime = self.CHOP
                else:
                    self._regime = self.TREND

            self._persist_model()  # 재시작 후 cold-start 없이 즉시 사용
            logger.debug(
                f"[KMeans] 국면={self._regime} | 클러스터={current} "
                f"| 평균수익={cluster_ret[current]*100:.4f}%"
            )
            return self._regime

        except Exception as e:
            import traceback as _tb
            logger.warning(f"[KMeans] 국면 감지 실패: {e}\n{_tb.format_exc()}")
            return self.UNKNOWN

    @property
    def regime(self) -> str:
        with self._lock:
            return self._regime

    def is_chop(self) -> bool:
        """고변동성 횡보장이면 True → 추세 추종 신호 차단 권장."""
        return self.regime == self.CHOP

    def needs_retrain(self) -> bool:
        with self._lock:
            return time.time() - self._last_train > self.retrain_interval