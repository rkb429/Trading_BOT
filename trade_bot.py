"""
trade_bot.py
Coin AI Bot - 5단계: 실전용 실시간 자동 매매 봇 V11 (Infinity)
(Atomic Write, 더블 스펜딩(중복 주문) 원천 차단, WebSocket Queue Flush)
"""
import os
import sys
import time
import json
import glob
import random
import joblib
import pyupbit
import pandas as pd
import requests
import logging
import queue
import threading
import signal
import pytz
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime

import config as bot_config
from market_context import (
    BinanceFuturesContext,
    KimchiPremiumTracker,
    FearGreedIndex,
    HMMRegimeDetector,
    KMeansRegimeDetector,
    LiquidityFlowMonitor,
    LiquidityScanner,
    NewsSentimentAnalyzer,
    UpbitRiskManager,
)

# 🌟 2단계 피처 엔지니어링 모듈 불러오기
try:
    from feature_engineering import add_technical_indicators
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.critical(f"❌ feature_engineering.py를 찾을 수 없습니다: {e}")
    sys.exit(1)

# ============================================================================
# 상수 정의
# ============================================================================
DEFAULT_CONFIG = {
    'best_threshold': 0.65,
    # config 단일 소스 — 하드코딩 시 config.py와 drift 발생
    'trailing_stop_pct': bot_config.BACKTEST['trailing_stop_pct'],
    'ai_exit_threshold': bot_config.BACKTEST['ai_exit_prob'],
    'ai_check_interval': 60,
    'interval': 'minute15'
}

def _resolve_threshold(cfg: dict, interval: str) -> float:
    """interval_thresholds에서 OOS 캘리브레이션된 threshold 반환.
    skip=True면 1.0 (신호 불가 표시), 항목 없으면 best_threshold 폴백.
    """
    it = cfg.get("interval_thresholds", {}).get(interval, {})
    if it.get("skip", False):
        return 1.0
    cal = it.get("threshold")
    if cal is not None:
        return float(cal)
    return float(cfg.get("best_threshold", DEFAULT_CONFIG["best_threshold"]))


def _resolve_regime_precision(cfg: dict, interval: str, min_prec: float) -> dict:
    """interval_thresholds.regime_precision 중 min_prec 이상인 국면만 반환."""
    rp = cfg.get("interval_thresholds", {}).get(interval, {}).get("regime_precision", {})
    return {k: v for k, v in rp.items() if v >= min_prec}


def is_dry_run() -> bool:
    """DRY_RUN 환경변수를 호출 시점에 재평가한다.
    모듈 import 시 1회 평가하면 테스트 중 os.environ 변경이 반영되지 않아
    모듈 재import가 필요했던 문제를 해결한다.
    """
    return os.getenv("DRY_RUN", "true").lower() not in ("false", "0", "no")

MIN_ORDER_KRW = 5000
SAFE_MIN_KRW_BALANCE = 5010
FEE_RATE = 0.001
POSITION_SIZE_PCT = bot_config.KELLY_FRACTION * 0.5  # ATR 데이터 없을 때 폴백 — Half-Kelly 기준값과 일치
STRONG_CONVICTION_THRESHOLD = 0.85  # 이 이상이면 알트도 시장가로 즉시 진입
MIN_PROFIT_THRESHOLD = 0.006    # 최소 기대 수익률: 수수료 왕복(0.2%) + 슬리피지 방어선
ATR_TP_MULTIPLIER = float(bot_config.LABELING.get("atr_tp_mult", 2.0))  # config 단일 소스
KELLY_PAYOFF_RATIO = (
    float(bot_config.LABELING.get("atr_tp_mult", 2.0)) /
    float(bot_config.LABELING.get("atr_sl_mult", 1.0))
)  # config 단일 소스 — 이론 손익비 (tp_mult / sl_mult)
KELLY_EMA_ALPHA   = 0.1         # EMA 실측 payoff 갱신 속도 (최근 약 10회 거래 반영)
KELLY_EMA_MIN_TRADES = 5        # 이 이하면 이론값 유지 (데이터 부족 방어)
OBI_THRESHOLD = -0.2            # 호가창 불균형 하한: 이 이하면 매도 압력 우위(스푸핑 의심)
BTC_TREND_CACHE_TTL = 3600      # BTC 일봉 SMA 캐시 유효 시간 (1시간 — 일봉은 자주 바뀌지 않음)

# ATR 기반 동적 비중 조절 파라미터
ATR_RISK_PCT = float(bot_config.BACKTEST.get("atr_risk_pct", 0.02))  # config 단일 소스 — backtest.py와 동기화
ATR_STOP_MULTIPLE = 2.0         # ATR의 몇 배를 손절 거리로 볼지
MIN_POSITION_PCT = 0.05         # 최소 5%
MAX_POSITION_PCT = 0.40         # 최대 40%

# 코인별 슬리피지 설정 — config.py 단일 소스 사용 (중복 제거)
LIQUID_COINS     = bot_config.LIQUID_COINS
SLIPPAGE_BY_COIN = bot_config.SLIPPAGE_BY_COIN
SLIPPAGE_DEFAULT = bot_config.SLIPPAGE_DEFAULT

# 알트 지정가 매수 설정
LIMIT_ORDER_BUFFER = 0.002      # 지정가 = 현재가 × (1 + 0.2%) — 즉시 체결 유도
LIMIT_ORDER_TIMEOUT = 10        # 10초 내 미체결 시 시장가 전환
TELEGRAM_TIMEOUT = 5
TELEGRAM_MAX_RETRIES = 3
WS_TIMEOUT = 120
# STATE_FILE 모듈 상수 제거 — AITradingBot.__init__에서 self._state_file을
# f"bot_state_{ticker}.json" 형태로 ticker별 분리 관리하므로 모듈 레벨 고정값 불필요
INTERVAL_MINUTES = {
    'minute1': 1, 'minute3': 3, 'minute5': 5, 'minute10': 10,
    'minute15': 15, 'minute30': 30, 'minute60': 60, 'minute240': 240,
    'day': 1440, 'week': 10080, 'month': 43200
}
STATE_SAVE_THROTTLE = 30
MODEL_DIR = bot_config.DIRECTORIES["models"]  # 레거시 폴백 (카테고리 미분리 시)
MODEL_RELOAD_INTERVAL = 300  # 5분마다 새 모델 파일 감지
_WHITELIST_FILE  = os.path.join(bot_config.DIRECTORIES.get("data", "data"), "coin_whitelist.json")
_wl_cache: set   = set()    # 비어있으면 전체 허용 (파일 없을 때 안전 기본값)
_wl_mtime: float = 0.0
_wl_lock = threading.Lock()


def _load_whitelist() -> set:
    """coin_whitelist.json mtime 기반 캐시 로드. 파일 없거나 오류 시 빈 set(전체 허용) 반환."""
    global _wl_cache, _wl_mtime
    try:
        if not os.path.exists(_WHITELIST_FILE):
            return set()
        mtime = os.path.getmtime(_WHITELIST_FILE)
        if mtime == _wl_mtime:
            return _wl_cache
        with _wl_lock:
            if mtime == _wl_mtime:  # 락 획득 전 다른 스레드가 갱신했을 수 있음
                return _wl_cache
            with open(_WHITELIST_FILE, 'r', encoding='utf-8') as _f:
                data = json.load(_f)
            new_cache = set(data.get("whitelist", []))
            if new_cache != _wl_cache:
                blocked = data.get("blocked", [])
                logger.info(
                    f"[화이트리스트] 갱신 로드 — 허용 {len(new_cache)}개"
                    + (f" | 차단: {', '.join(blocked)}" if blocked else "")
                )
            _wl_cache = new_cache
            _wl_mtime = mtime
        return _wl_cache
    except Exception:
        return _wl_cache  # 오류 시 기존 캐시 유지 (안전)


def _resolve_model_dir(ticker: str) -> str:
    """'KRW-BTC' → 카테고리 모델 디렉토리. COIN_CATEGORIES 미설정 시 기본 models/ 사용."""
    coin = ticker.split("-")[-1].upper()  # "KRW-BTC" → "BTC"
    get_dir = getattr(bot_config, "get_model_dir", None)
    if callable(get_dir):
        return get_dir(coin)
    return MODEL_DIR

load_dotenv(override=True)  # os.getenv 호출 전 .env 로드, 시스템 환경변수보다 .env 우선

# ─── 자금 관리 / 리스크 제어 상수 ────────────────────────────────────────────
# .env로 재정의 가능; 기본값은 보수적 운용 기준
LIQUIDITY_IMPACT_CAP        = float(os.getenv("LIQUIDITY_IMPACT_CAP", "0.01"))
# 24h 거래대금 대비 최대 주문 비율 (1%). 복리로 자본이 커져도 슬리피지 폭발 방지
LIQ_CAP_CACHE_TTL           = 300          # 유동성 캡 캐시 TTL (초)
BOT_INITIAL_CAPITAL         = float(os.getenv("BOT_INITIAL_CAPITAL_KRW") or os.getenv("BOT_INITIAL_CAPITAL", "0"))
# 0 = 잔고 비례 모드 / 양수 = 이 KRW 금액을 봇 전체 예산 상한으로 고정
BOT_USE_FIXED_SEED          = os.getenv("BOT_USE_FIXED_SEED", "false").lower() == "true"
RECONCILIATION_DRIFT_THRESHOLD = float(os.getenv("RECONCILIATION_DRIFT_THRESHOLD", "0.001"))
# 장부-실제잔고 허용 오차 (기본 0.1%). 초과 시 Kill Switch

# 분할 진입/청산 설정 (config.py 연동 — SPLIT_ORDER["enabled"]로 ON/OFF)
_so = bot_config.SPLIT_ORDER
SPLIT_ORDER_ENABLED       = _so["enabled"]
SPLIT_ORDER_N_SPLITS      = _so["n_splits"]
SPLIT_ORDER_DELAY_SEC     = _so["split_delay_sec"]
SPLIT_ORDER_ATR_THRESHOLD = _so["atr_threshold"]
del _so


class _TokenBucket:
    """
    멀티봇 글로벌 API 속도 제한 — 동일 프로세스 내 모든 봇 인스턴스가 공유.
    Upbit 제한: 주문 ~10 req/s, 시세조회 ~30 req/s
    """
    def __init__(self, rate: float, capacity: int):
        self._rate = rate
        self._capacity = float(capacity)
        self._tokens = float(capacity)
        self._last = time.time()
        self._lock = threading.Lock()

    def wait(self, cost: float = 1.0):
        with self._lock:
            now = time.time()
            self._tokens = min(self._capacity, self._tokens + (now - self._last) * self._rate)
            self._last = now
            if self._tokens >= cost:
                self._tokens -= cost
            else:
                wait_sec = (cost - self._tokens) / self._rate
                self._tokens = 0.0
                time.sleep(wait_sec)

# 모듈 레벨 싱글톤 — 멀티봇 동시 스탑로스 발동 시 429 폭발 방지
_ORDER_BUCKET = _TokenBucket(rate=8.0, capacity=8)   # 주문 전용 (Upbit 10 req/s의 80%)
_QUERY_BUCKET = _TokenBucket(rate=18.0, capacity=18)  # 시세·잔고 조회 전용


def find_latest_file(directory: str, pattern: str):
    """디렉토리에서 가장 최근에 생성된 파일 찾기"""
    files = glob.glob(os.path.join(directory, pattern))
    if not files:
        return None
    return max(files, key=os.path.getctime)


def _get_feat_order(model, features: list) -> list:
    """XGBoost 부스터 학습 순서 기준 피처 순서 반환.
    EnsembleModel → CalibratedClassifierCV → XGBoost booster 계층 순으로 탐색.
    config['features']는 importance 정렬 순서로 저장되어 부스터 순서와 다를 수 있음.
    """
    for candidate in [getattr(model, 'xgb_model', None), model]:
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
    if hasattr(model, 'feature_names_in_'):
        return list(model.feature_names_in_)
    return features

# ============================================================================
# 1. 환경 설정 및 로깅
# ============================================================================
load_dotenv()
UPBIT_ACCESS = os.getenv("UPBIT_ACCESS_KEY")
UPBIT_SECRET = os.getenv("UPBIT_SECRET_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

required_env = ["UPBIT_ACCESS_KEY", "UPBIT_SECRET_KEY", "TELEGRAM_TOKEN", "TELEGRAM_CHAT_ID"]
missing_env = [var for var in required_env if not os.getenv(var)]
if missing_env and not is_dry_run():
    raise ValueError(f"❌ 필수 환경변수 누락: {', '.join(missing_env)}")

upbit = pyupbit.Upbit(UPBIT_ACCESS, UPBIT_SECRET)

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ============================================================================
# 2. 텔레그램 Worker Queue
# ============================================================================
class TelegramNotifier:
    _MAX_QUEUE = 50  # 큐 포화 시 가장 오래된 메시지를 드롭 (메시지 폭탄 방지)

    def __init__(self):
        self.msg_queue = queue.Queue(maxsize=self._MAX_QUEUE)
        self.worker_thread = threading.Thread(target=self._worker, daemon=True, name="TelegramWorker")
        self.is_running = True
        self.worker_thread.start()

    def _worker(self):
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        while self.is_running:
            try:
                text = self.msg_queue.get(timeout=1.0)
                if text is None:
                    self.msg_queue.task_done()
                    break
                try:
                    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text}
                    for attempt in range(TELEGRAM_MAX_RETRIES):
                        res = requests.post(url, json=payload, timeout=TELEGRAM_TIMEOUT)
                        if res.status_code == 429:
                            retry_after = res.json().get("parameters", {}).get("retry_after", 5)
                            logger.warning(f"텔레그램 트래픽 초과(429). {retry_after}초 대기 ({attempt+1}/{TELEGRAM_MAX_RETRIES})...")
                            time.sleep(retry_after)
                        elif res.status_code != 200:
                            try:
                                detail = res.json().get("description", res.text[:200])
                            except Exception:
                                detail = res.text[:200]
                            logger.error(f"텔레그램 전송 실패: HTTP {res.status_code} — {detail}")
                            break
                        else:
                            break
                except Exception as e:
                    logger.error(f"텔레그램 워커 에러: {e}")
                finally:
                    self.msg_queue.task_done()
                time.sleep(0.1)
            except queue.Empty:
                continue

    def notify(self, message: str):
        logger.info(message)
        try:
            self.msg_queue.put_nowait(message)
        except queue.Full:
            # 큐 포화 시 가장 오래된 메시지를 버리고 새 메시지 삽입 (drop-oldest)
            try:
                self.msg_queue.get_nowait()
                self.msg_queue.put_nowait(message)
            except queue.Empty:
                pass

    def shutdown(self):
        self.is_running = False
        self.msg_queue.put(None)
        self.worker_thread.join(timeout=3.0)

notifier = TelegramNotifier()
def notify(msg: str): notifier.notify(msg)

# ============================================================================
# 3. 실전 트레이딩 봇 클래스 (Infinity)
# ============================================================================
class AITradingBot:
    def __init__(self, ticker: str = "KRW-BTC", model_path: str = None, config_path: str = None,
                 capital_fraction: float = 1.0, setup_signals: bool = True,
                 liq_scanner: LiquidityScanner = None,
                 liq_flow: LiquidityFlowMonitor = None):
        self.ticker = ticker
        self.currency = ticker.split('-')[1]
        self.current_price = 0.0
        self.is_running = True
        self.wm = None
        self._wm_q = None  # WebSocketManager 큐 캐시 (pyupbit 버전 호환)

        # 멀티봇 지원: 자본 비중, 티커별 상태파일, 소유권 플래그
        self.capital_fraction = max(0.01, min(1.0, capital_fraction))
        self._state_file = f"bot_state_{ticker.replace('-', '_')}.json"
        self._state_file_tmp = f"bot_state_{ticker.replace('-', '_')}.json.tmp"
        self._trade_log_path = os.path.join("logs", f"trade_log_{ticker.replace('-', '_')}.csv")
        self._owns_liq_scanner = liq_scanner is None  # False면 오케스트레이터가 소유
        self._owns_notifier = setup_signals            # False면 오케스트레이터가 종료 담당

        if setup_signals:
            self._setup_signal_handlers()

        # 티커에서 카테고리 모델 디렉토리 자동 선택 (KRW-BTC → models_A 등)
        self._model_dir = _resolve_model_dir(ticker)

        # 경로 미지정 시 카테고리 모델 디렉토리에서 최신 파일 자동 탐색 (앙상블 우선)
        if model_path is None:
            model_path = (
                find_latest_file(self._model_dir, "*ensemble_bot*.pkl")
                or find_latest_file(self._model_dir, "*xgb_bot*.pkl")
            )
        if config_path is None:
            config_path = find_latest_file(self._model_dir, "config_*.json")

        if not model_path or not config_path:
            raise FileNotFoundError(f"❌ {self._model_dir}/ 디렉토리에서 모델 또는 설정 파일을 찾을 수 없습니다. 먼저 머신러닝 훈련을 실행하세요.")
        if not Path(model_path).exists() or not Path(config_path).exists():
            raise FileNotFoundError(f"❌ 모델 파일이 존재하지 않습니다: {model_path}, {config_path}")

        logger.info("🧠 AI 두뇌와 설정 파일을 로드합니다...")
        self.model = joblib.load(model_path)
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        self._model_path = model_path
        self._config_path = config_path
        self._last_model_check_time = time.time()

        self.features = self.config['features']
        self.feat_order = (
            self.config.get('features_training_order') or
            _get_feat_order(self.model, self.features)
        )
        self.interval = self.config.get('interval', DEFAULT_CONFIG['interval'])
        self.threshold = _resolve_threshold(self.config, self.interval)
        self._force_mtf = False
        if self.threshold >= 1.0:
            # skip=True지만 calibrated threshold가 존재하면 15m MTF 필수 모드로 시도
            # (Cat A/B: 60m 효율 시장 구조적 한계 → 15m AND 게이트 보완 설계)
            _it = self.config.get("interval_thresholds", {}).get(self.interval, {})
            _fallback = _it.get("threshold")
            if _fallback:
                self.threshold = _fallback
                self._force_mtf = True
                logger.warning(
                    f"⚠️ [{ticker}] {self.interval} skip=True"
                    f" (prec={_it.get('precision',0):.3f} < break_even={_it.get('break_even',0):.3f})"
                    f" — 15m MTF AND 게이트 필수 모드"
                )
            else:
                raise ValueError(
                    f"❌ {ticker}: {self.interval} 모델 interval_thresholds.skip=True "
                    f"— OOS 정밀도 손익분기 미달, 봇 배포 차단"
                )

        # ── MTF 15m 타이밍 모델 로드 ─────────────────────────────────────────────
        _mtf = getattr(bot_config, "MTF_ENSEMBLE", {})
        self._mtf_enabled = _mtf.get("enabled", False)
        self.model_15m = None
        self.features_15m: list = []
        self.feat_order_15m: list = []
        self.threshold_15m: float = _mtf.get("default_threshold_15m", 0.50)
        self._model_path_15m: str = ""
        self._model_dir_15m: str = ""
        if self._mtf_enabled:
            _get_mtf_dir = getattr(bot_config, "get_mtf_model_dir", None)
            coin = ticker.split("-")[-1].upper()
            self._model_dir_15m = _get_mtf_dir(coin, "minute15") if callable(_get_mtf_dir) else ""
            if self._model_dir_15m:
                _m15 = (find_latest_file(self._model_dir_15m, "*ensemble_bot*.pkl")
                        or find_latest_file(self._model_dir_15m, "*xgb_bot*.pkl"))
                _c15 = find_latest_file(self._model_dir_15m, "config_*.json")
                if _m15 and _c15:
                    self.model_15m = joblib.load(_m15)
                    with open(_c15, 'r', encoding='utf-8') as f:
                        _cfg15 = json.load(f)
                    self.features_15m = _cfg15.get('features', [])
                    self.feat_order_15m = (
                        _cfg15.get('features_training_order') or
                        _get_feat_order(self.model_15m, self.features_15m)
                    )
                    _t15 = _resolve_threshold(_cfg15, "minute15")
                    if _t15 >= 1.0:
                        logger.warning(
                            f"⚠️ [{ticker}] 15m 모델 minute15 skip=True "
                            f"— MTF AND 게이트 비활성화 (60m 단독 동작)"
                        )
                        self._mtf_enabled = False
                    else:
                        self.threshold_15m = _t15
                        self._model_path_15m = _m15
                        logger.info(f"📐 15m 타이밍 모델 로드: {os.path.basename(_m15)}")
                else:
                    logger.warning(f"⚠️ MTF 활성화 but 15m 모델 없음 ({self._model_dir_15m}) — 60m 단독 동작")
                    self._mtf_enabled = False
        # 국면 정밀도 오버라이드: skip=True이지만 특정 HMM 국면에서 손익분기 초과 시 재활성화
        _regime_min_p = getattr(bot_config, "MODEL_MANAGEMENT", {}).get("regime_min_precision", 0.40)
        self._regime_precision: dict = _resolve_regime_precision(self.config, self.interval, _regime_min_p)
        if self._force_mtf and not self._mtf_enabled:
            if not self._regime_precision:
                raise ValueError(
                    f"❌ {ticker}: {self.interval} skip=True + 15m MTF 보완 불가 — 봇 배포 차단"
                )
            logger.warning(
                f"⚠️ [{ticker}] skip=True + MTF 없음 → 국면 정밀도 오버라이드 활성화 "
                f"(허용 국면: {list(self._regime_precision.keys())})"
            )
        # ─────────────────────────────────────────────────────────────────────────

        self.trailing_stop_pct = self.config.get('trailing_stop_pct', DEFAULT_CONFIG['trailing_stop_pct'])
        self.ai_exit_threshold = self.config.get('ai_exit_threshold', DEFAULT_CONFIG['ai_exit_threshold'])
        self.ai_check_interval = self.config.get('ai_check_interval', DEFAULT_CONFIG['ai_check_interval'])
        
        self.highest_price = 0.0
        self._highest_price_lock = threading.Lock()   # 모니터 스레드와 메인 스레드 동시 접근 방지
        self.last_ai_check_time = 0
        self.is_holding = False
        self.buy_price = 0.0
        self.last_evaluated_candle_time = None

        # UUID 기반 포지션 격리: 봇이 직접 연 주문만 추적
        self.bot_uuid = None          # 봇이 제출한 매수 주문 UUID
        self.bot_quantity = 0.0       # 봇이 매수한 코인 수량 (개인 보유량과 분리)
        self.last_atr_ratio = 0.0     # ATR 기반 비중 계산용 (최근 AI 예측 시 갱신)

        # Kelly EMA 실측 Payoff 추적 (execute_sell 시 갱신, 상태파일에 영속)
        self.payoff_ema = KELLY_PAYOFF_RATIO   # 초기값: 이론값
        self.payoff_trade_count = 0            # 누적 청산 횟수 (신뢰도 판단용)

        # NAV 장부 추적 (P2b)
        self._initial_capital = 0.0            # 봇 최초 가용 예산 (첫 매수 시점에 확정)
        self._realized_pnl    = 0.0            # 누적 실현 손익 (수수료 포함 net KRW)
        self._peak_nav        = 0.0            # 역대 최고 NAV (절대 MDD 게이트 기준값)
        self._last_reconciliation_day = -1     # 마지막 장부 대사 실행일 (epoch day)
        self._liq_cap_cache: dict = {'krw': float('inf'), 'ts': 0.0}  # P0 캐시

        # ── 서킷 브레이커 (Circuit Breaker) 상태 ─────────────────────────────
        _cb = bot_config.CIRCUIT_BREAKER
        self._cb_enabled           = _cb.get("enabled", True)
        self._cb_daily_loss_pct    = _cb.get("daily_loss_pct", -0.03)      # -3%
        self._cb_consec_loss_limit = _cb.get("consecutive_loss_count", 3)  # 연속 패배 한도
        self._cb_cooldown_sec      = _cb.get("cooldown_minutes", 60) * 60  # 냉각 시간(초)
        self._cb_reset_hour_kst    = _cb.get("reset_hour", 9)              # KST 리셋 시각
        self._cb_daily_open_nav    = 0.0   # 당일 9시 기준 NAV (일일 손실 계산 분모)
        self._cb_consecutive_losses = 0    # 연속 패배 횟수
        self._cb_cooldown_until    = 0.0   # 냉각 종료 Unix timestamp (0=냉각 없음)
        self._cb_last_reset_day    = -1    # 마지막 리셋된 KST 날짜 (epoch day)

        # 타임 스탑: AI 예측 유효 시간(FORWARD_BARS × 캔들 간격) 초과 시 강제 청산
        self.buy_time = 0.0
        interval_min = INTERVAL_MINUTES.get(self.interval, 15)
        _fwd_bars = bot_config.LABELING.get("forward_bars", 4)
        self.time_stop_sec = interval_min * _fwd_bars * 60  # config.LABELING.forward_bars 봉
        self._timestop_cooldown_sec = interval_min * 3 * 60  # 타임스탑 후 3봉 재진입 금지
        self._timestop_cooldown_until = 0.0                  # 쿨다운 종료 Unix timestamp

        self._last_insufficient_funds_log_time = 0
        self._last_state_save_time = 0
        self._last_heartbeat_save_time = 0
        _trigger_interval = "minute15" if self._mtf_enabled else self.interval
        _candle_sec = INTERVAL_MINUTES.get(_trigger_interval, 15) * 60
        self.last_candle_boundary = (int(time.time()) // _candle_sec) * _candle_sec  # 거래소 시간 기반 캔들 경계 추적
        self._btc_trend_cache = {'is_uptrend': True, 'last_check': 0.0}  # BTC 거시 장세 캐시
        self._ohlcv_cache: dict = {'df': None, 'boundary': -1, 'count': 0}  # AI 700봉 캐시 (캔들 경계 기반 갱신)
        self._last_daily_report_day = -1       # 일일 Telegram 성과 리포트 마지막 전송일
        self._last_hmm_block_log: float = 0.0   # HMM 차단 로그 rate limit (5분 간격)
        self._last_btc_block_log: float = 0.0   # BTC 거시 차단 로그 rate limit (5분 간격)

        # 시장 컨텍스트 매니저 초기화
        _bc = bot_config.BINANCE_CONTEXT
        self.binance_ctx = BinanceFuturesContext(
            cache_ttl=_bc["cache_ttl"],
            funding_threshold=_bc["funding_rate_threshold"],
            ls_overbought=_bc["ls_ratio_overbought"],
            funding_oversold=_bc["funding_rate_oversold"],
        ) if _bc["enabled"] else None

        _kp = bot_config.KIMCHI_PREMIUM
        self.kimchi_tracker = KimchiPremiumTracker(
            cache_ttl=_kp["cache_ttl"],
            history_size=_kp["history_size"],
            surge_threshold=_kp["surge_threshold"],
        ) if _kp["enabled"] else None

        _fg = bot_config.FEAR_GREED
        self.fear_greed = FearGreedIndex(
            cache_ttl=_fg["cache_ttl"],
            extreme_greed=_fg["extreme_greed"],
            greed=_fg["greed"],
            ts_extreme_ratio=_fg["trailing_stop_extreme_ratio"],
            ts_greed_ratio=_fg["trailing_stop_greed_ratio"],
        ) if _fg["enabled"] else None

        _hmm = bot_config.HMM_REGIME
        self.hmm = HMMRegimeDetector(
            n_states=_hmm["n_states"],
            lookback=_hmm["lookback"],
            retrain_interval=_hmm["retrain_interval"],
            threshold_bear=_hmm["threshold_bear"],
            threshold_sideways=_hmm.get("threshold_sideways", 0.0),
            threshold_bull=_hmm["threshold_bull"],
            position_multiplier_bear=_hmm["position_multiplier_bear"],
            position_multiplier_sideways=_hmm.get("position_multiplier_sideways", 0.7),
        ) if _hmm["enabled"] else None

        if liq_scanner is not None:
            self.liq_scanner = liq_scanner  # 오케스트레이터로부터 공유 스캐너 주입
        else:
            _ls = bot_config.LIQUIDITY_SCANNER
            self.liq_scanner = LiquidityScanner(
                check_interval=_ls["check_interval"],
                top_n=_ls["top_n"],
                surge_rank_jump=_ls["surge_rank_jump"],
            ) if _ls["enabled"] else None

        _lf = bot_config.LIQUIDITY_FLOW
        if liq_flow is not None:
            self.liq_flow = liq_flow   # 오케스트레이터로부터 공유 모니터 주입
        else:
            self.liq_flow = LiquidityFlowMonitor(_lf) if _lf.get("enabled", True) else None

        self._current_regime: str = HMMRegimeDetector.SIDEWAYS
        self._btc_price_cache: dict = {'price': 0.0, 'ts': 0.0}
        self._usdt_krw_cache: dict = {'price': 0.0, 'ts': 0.0}

        # 업비트 리스크 매니저 (지갑 락업 + DAXA 투자유의종목 진입 전 차단)
        _ur = bot_config.UPBIT_RISK
        if _ur.get("enabled", True):
            self.upbit_risk = UpbitRiskManager(
                access_key=UPBIT_ACCESS,
                secret_key=UPBIT_SECRET,
                cache_ttl=_ur.get("cache_ttl", 120),
            )
        else:
            self.upbit_risk = None

        # K-Means 보조 국면 감지기 (고변동성 횡보 추세 신호 차단)
        self.kmeans_regime = KMeansRegimeDetector(n_clusters=3, lookback=200)

        # BTC 1h 낙폭 킬스위치 (0순위 진입 차단)
        self.news_analyzer = NewsSentimentAnalyzer(cache_ttl=300)

        # 독립 트레일링 스탑 감시 스레드: API 블로킹 중에도 10 Hz로 가격 감시
        self._ts_triggered = False       # 모니터 스레드 → 메인 루프 청산 신호
        self._ts_trigger_price = 0.0
        self._ts_active = threading.Event()  # 포지션 진입 시 set(), 청산 시 clear() → 미보유 중 절전
        self._monitor_thread = threading.Thread(
            target=self._trail_stop_monitor, daemon=True, name="TrailStopMonitor"
        )
        self._monitor_thread.start()
        self._hmm_retrain_thread: threading.Thread | None = None  # 중복 재학습 방지용 참조

        self._sync_account_state()
        self._validate_initialization()

        _budget_krw = self._get_bot_budget()
        notify(f"🤖 {self.ticker} 실전 매매 봇 V11 [Infinity] 가동!\n"
               f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
               f"📊 핵심 파라미터:\n"
               f"  • 자본비중: {self.capital_fraction*100:.0f}% (예산 {_budget_krw:,.0f}원)\n"
               f"  • 진입 컷: {self.threshold*100:.1f}% | 탈출 컷: {self.ai_exit_threshold*100:.1f}%\n"
               f"  • 트레일링 스탑: {self.trailing_stop_pct*100:.1f}%\n"
               f"  • 시스템: Atomic Write, Double-Spend 차단, WS Flush\n"
               f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    def _setup_signal_handlers(self):
        signal.signal(signal.SIGINT, self._handle_shutdown_signal)
        try:
            signal.signal(signal.SIGTERM, self._handle_shutdown_signal)
        except (OSError, ValueError):
            pass  # Windows에서 SIGTERM 미지원

    def _handle_shutdown_signal(self, signum, frame):  # noqa: ARG002
        logger.info("🛑 OS 종료 시그널 수신. 상태 저장 후 안전하게 종료합니다.")
        self.is_running = False

    def _validate_initialization(self):
        try:
            test_input = pd.DataFrame([[0] * len(self.feat_order)], columns=self.feat_order)
            self.model.predict_proba(test_input)
            del test_input
        except Exception as e:
            raise RuntimeError(f"❌ 모델 검증 실패: {e}")

    # =========================================================================
    # P0 / P1 / P2 리스크 헬퍼
    # =========================================================================

    def _get_bot_budget(self) -> float:
        """
        이 봇에 실제 사용 가능한 KRW 예산 반환.
        BOT_INITIAL_CAPITAL > 0이면 총 자본 기준으로 fraction 계산 (다른 봇 보유로 KRW 줄어도 예산 유지).
        BOT_INITIAL_CAPITAL = 0이면 현재 KRW 잔고 비례 모드 (복리 허용).
        """
        actual_krw = self._safe_api_get(upbit.get_balance, "KRW") or 0.0
        if BOT_INITIAL_CAPITAL > 0:
            fixed = BOT_INITIAL_CAPITAL * self.capital_fraction
            # DRY_RUN: 실잔고 0이어도 시뮬레이션 예산 그대로 사용
            return fixed if is_dry_run() else min(fixed, actual_krw)
        return actual_krw * self.capital_fraction

    def _get_liquidity_cap_krw(self) -> float:
        """
        24h 거래대금의 LIQUIDITY_IMPACT_CAP(기본 1%) 을 최대 주문금액 상한으로 반환.
        복리로 자본이 커져도 호가창 충격(슬리피지 폭발)을 원천 차단.
        결과는 LIQ_CAP_CACHE_TTL(5분) 동안 캐시 — OHLCV API 과호출 방지.
        조회 실패 시 inf 반환(안전 방향: 제한 없이 진행).
        """
        now = time.time()
        if now - self._liq_cap_cache['ts'] < LIQ_CAP_CACHE_TTL:
            return self._liq_cap_cache['krw']
        try:
            df = self._safe_api_get(pyupbit.get_ohlcv, self.ticker, interval="day", count=1)
            if df is not None and not df.empty:
                vol_24h = float(df['value'].iloc[-1])
                cap = vol_24h * LIQUIDITY_IMPACT_CAP
                self._liq_cap_cache = {'krw': cap, 'ts': now}
                logger.debug(
                    f"[{self.ticker}] 유동성 캡 갱신: 24h={vol_24h/1e8:.1f}억 → "
                    f"상한={cap/1e4:.0f}만원"
                )
                return cap
        except Exception as e:
            logger.debug(f"[{self.ticker}] 유동성 캡 조회 실패(제한 없음): {e}")
        return float('inf')

    def _daily_reconciliation(self) -> None:
        """
        장부(bot_quantity) vs 거래소 실제 잔고 대사.
        RECONCILIATION_DRIFT_THRESHOLD(기본 0.1%) 초과 불일치 시 Kill Switch 발동.
        포지션 미보유 상태면 즉시 반환.
        """
        if not self.is_holding or self.bot_quantity <= 0:
            return
        actual_qty = self._safe_api_get(upbit.get_balance, self.currency)
        if actual_qty is None:
            logger.warning(f"[{self.ticker}] 장부 대사: 잔고 조회 실패 — 이번 회차 스킵")
            return
        drift = abs(self.bot_quantity - actual_qty) / self.bot_quantity
        nav = self._initial_capital + self._realized_pnl
        logger.info(
            f"[{self.ticker}] 일일 대사: 장부={self.bot_quantity:.6f} / "
            f"실제={actual_qty:.6f} / 오차={drift:.4%} | "
            f"NAV={nav:,.0f}원 (초기={self._initial_capital:,.0f} + 실현PnL={self._realized_pnl:+,.0f})"
        )
        # 잉여(actual > bot_quantity)는 사용자 기존 보유분 — 오탐 방지, 부족분만 체크
        deficit = max(0.0, self.bot_quantity - actual_qty)
        drift = deficit / self.bot_quantity
        if drift > RECONCILIATION_DRIFT_THRESHOLD:
            msg = (
                f"🚨 [{self.ticker}] 장부 불일치 (부족)!\n"
                f"  장부: {self.bot_quantity:.6f} / 실제: {actual_qty:.6f}\n"
                f"  부족: {drift:.4%} (허용 {RECONCILIATION_DRIFT_THRESHOLD:.4%} 초과)\n"
                f"  → 봇 자동 정지. 수동 확인 필요."
            )
            notify(msg)
            logger.critical(msg)
            self.is_running = False

    def stop_gracefully(self):
        """오케스트레이터 종료 신호 수신 시 호출 — 보유 포지션 청산 후 루프 자연 종료"""
        logger.info(f"[{self.ticker}] 우아한 종료 신호 수신 — 포지션 청산 후 루프 종료 예정")
        self.is_running = False

    def force_kill(self):
        """
        오케스트레이터가 60초 join 이후에도 스레드가 살아있을 때 호출.
        is_running 및 _ts_active 이벤트 해제로 모니터 스레드 포함 전체 루프 즉시 중단.
        실제 주문 취소는 보장하지 않으므로 반드시 stop_gracefully() 우선 시도 후 사용.
        """
        logger.warning(f"[{self.ticker}] ⚠️ force_kill 호출 — 루프 강제 중단 (좀비 방지)")
        self.is_running = False
        self._ts_active.clear()  # 모니터 스레드 wait() 해제 → 루프 종료

    def get_position_info(self) -> dict:
        """orchestrator 호환 포지션 상태 반환."""
        unrealized_pnl = 0.0
        unrealized_pct = 0.0
        if self.is_holding and self.buy_price > 0 and self.current_price > 0:
            unrealized_pct = (self.current_price / self.buy_price) - 1.0
            unrealized_pnl = self.bot_quantity * self.current_price * (1 - FEE_RATE) \
                             - self.bot_quantity * self.buy_price * (1 + FEE_RATE)
        return {
            'ticker': self.ticker,
            'is_holding': self.is_holding,
            'buy_price': self.buy_price,
            'current_price': self.current_price,
            'highest_price': self.highest_price,
            'bot_quantity': self.bot_quantity,
            'bot_uuid': self.bot_uuid,
            'buy_time': self.buy_time,
            'unrealized_pnl_krw': unrealized_pnl,
            'unrealized_pnl_pct': unrealized_pct,
            'realized_pnl': self._realized_pnl,
            'nav': self._initial_capital + self._realized_pnl,
            'regime': self._current_regime,
            'is_running': self.is_running,
            'heartbeat': time.time(),
        }

    def _check_and_reload_model(self):
        """카테고리 모델 디렉토리에 새 파일 생성 시 포지션 비보유 상태에서 핫-리로드."""
        try:
            latest_model = (
                find_latest_file(self._model_dir, "*ensemble_bot*.pkl")
                or find_latest_file(self._model_dir, "*xgb_bot*.pkl")
            )
            latest_config = find_latest_file(self._model_dir, "config_*.json")
            if not latest_model or not latest_config:
                return
            if latest_model == self._model_path:
                return  # 변경 없음
            if self.is_holding:
                logger.debug("⏸️ 포지션 보유 중 — 모델 교체는 청산 후 다음 체크에서 수행")
                return
            logger.info(f"🔄 새 모델 감지: {os.path.basename(latest_model)} — 핫-리로드 시작")
            new_model = joblib.load(latest_model)
            with open(latest_config, 'r', encoding='utf-8') as f:
                new_config = json.load(f)
            new_interval  = new_config.get('interval', DEFAULT_CONFIG['interval'])
            new_threshold = _resolve_threshold(new_config, new_interval)
            _new_regime_p = _resolve_regime_precision(
                new_config, new_interval,
                getattr(bot_config, "MODEL_MANAGEMENT", {}).get("regime_min_precision", 0.40)
            )
            if new_threshold >= 1.0 and not _new_regime_p:
                logger.warning(
                    f"⚠️ [{self.ticker}] 새 모델 {new_interval} skip=True + 국면 정밀도 없음 "
                    f"— 핫-리로드 거부, 현재 모델 유지"
                )
                return
            self.model = new_model
            self.config = new_config
            self.features = new_config['features']
            self.feat_order = (
                new_config.get('features_training_order') or
                _get_feat_order(self.model, self.features)
            )
            self.interval         = new_interval
            self.threshold        = new_threshold if new_threshold < 1.0 else float(
                new_config.get("interval_thresholds", {}).get(new_interval, {}).get("threshold", 1.0)
            )
            self._regime_precision = _new_regime_p
            self._force_mtf       = (new_threshold >= 1.0)
            self.trailing_stop_pct = new_config.get('trailing_stop_pct', DEFAULT_CONFIG['trailing_stop_pct'])
            self.ai_exit_threshold = new_config.get('ai_exit_threshold', DEFAULT_CONFIG['ai_exit_threshold'])
            self._model_path = latest_model
            self._config_path = latest_config

            # MTF 15m 모델 핫-리로드
            if self._mtf_enabled and self._model_dir_15m:
                _m15 = (find_latest_file(self._model_dir_15m, "*ensemble_bot*.pkl")
                        or find_latest_file(self._model_dir_15m, "*xgb_bot*.pkl"))
                _c15 = find_latest_file(self._model_dir_15m, "config_*.json")
                if _m15 and _c15 and _m15 != self._model_path_15m:
                    self.model_15m = joblib.load(_m15)
                    with open(_c15, 'r', encoding='utf-8') as f:
                        _cfg15 = json.load(f)
                    _t15 = _resolve_threshold(_cfg15, "minute15")
                    if _t15 >= 1.0:
                        logger.warning(
                            f"⚠️ [{self.ticker}] 새 15m 모델 skip=True "
                            f"— MTF 비활성화"
                        )
                        self._mtf_enabled = False
                        self.model_15m = None
                    else:
                        self.features_15m = _cfg15.get('features', [])
                        self.feat_order_15m = (
                            _cfg15.get('features_training_order') or
                            _get_feat_order(self.model_15m, self.features_15m)
                        )
                        self.threshold_15m = _t15
                        self._model_path_15m = _m15
                        logger.info(f"📐 15m 모델 핫-리로드: {os.path.basename(_m15)}")

            notify(f"🧠 [{self.ticker}] 신규 모델 핫-리로드 완료!\n  {os.path.basename(latest_model)}")
        except Exception as e:
            logger.warning(f"모델 핫-리로드 오류: {e}")

    # 🌟 1. 더블 스펜딩(중복 매매) 방지를 위해 GET(조회) 용도로만 제한한 안전 호출
    def _safe_api_get(self, func, *args, retries=3, **kwargs):
        """
        상태 조회(GET) 전용: 지수 백오프 재시도.
        pyupbit는 HTTP 429/500 등을 예외로 올리거나 {'error': ...} dict로 반환하기도 함.
        두 경우 모두 감지해 재시도합니다.
        """
        fname = getattr(func, '__name__', repr(func))
        for i in range(retries):
            _QUERY_BUCKET.wait()
            try:
                result = func(*args, **kwargs)
                # pyupbit 일부 호출은 예외 대신 에러 딕셔너리를 반환함
                if isinstance(result, dict) and 'error' in result:
                    err_msg = result['error'].get('message', str(result['error']))
                    raise ValueError(f"Upbit API 에러: {err_msg}")
                return result
            except Exception as e:
                if i < retries - 1:
                    wait = 2 ** i
                    logger.warning(f"⚠️ API 조회 에러({fname}): {e}. {wait}초 대기 후 재시도...")
                    time.sleep(wait)
        logger.error(f"❌ API 조회 최종 실패({fname})")
        return None

    # 🌟 2. 원자적(Atomic) 상태 저장 로직
    def _save_local_state(self):
        """디스크 깨짐 방지: 임시 파일 기록 후 원자적 교체(Atomic Rename)"""
        try:
            state = {
                'is_holding': self.is_holding,
                'buy_price': self.buy_price,
                'highest_price': self.highest_price,
                'bot_uuid': self.bot_uuid,
                'bot_quantity': self.bot_quantity,
                'buy_time': self.buy_time,
                'payoff_ema': self.payoff_ema,
                'payoff_trade_count': self.payoff_trade_count,
                'initial_capital': self._initial_capital,
                'realized_pnl': self._realized_pnl,
                'last_reconciliation_day': self._last_reconciliation_day,
                'peak_nav': self._peak_nav,
                # 서킷 브레이커 상태 (재시작 후에도 당일 손실/연속패배 유지)
                'cb_daily_open_nav': self._cb_daily_open_nav,
                'cb_consecutive_losses': self._cb_consecutive_losses,
                'cb_cooldown_until': self._cb_cooldown_until,
                'cb_last_reset_day': self._cb_last_reset_day,
                'timestop_cooldown_until': self._timestop_cooldown_until,
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'heartbeat': time.time(),  # orchestrator 헬스체크용 Unix timestamp
            }
            # 임시 파일에 쓰기
            with open(self._state_file_tmp, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=4)
            # 단 0.0001초의 틈도 없이 덮어씌우기 (서버가 여기서 꺼져도 기존 파일 유지됨)
            os.replace(self._state_file_tmp, self._state_file)
        except Exception as e:
            logger.error(f"로컬 상태 원자적 저장 실패: {e}")

    def _load_local_state(self) -> dict:
        if Path(self._state_file).exists():
            try:
                with open(self._state_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    # 🌟 3. 웹소켓 큐 비우기 (Queue Flush)
    def _flush_ws_queue(self):
        """블로킹 작업 후 누적 틱 처리: 트레일링 스탑 이벤트는 버리지 않고 즉시 실행"""
        if self.wm and self._wm_q is not None:
            flushed = 0
            try:
                while True:
                    tick = self._wm_q.get_nowait()
                    price = tick.get('trade_price') if tick else None
                    if price and self.is_holding:
                        with self._highest_price_lock:
                            if price > self.highest_price:
                                self.highest_price = price
                            snap_high = self.highest_price
                        drop = (snap_high - price) / snap_high if snap_high > 0 else 0
                        if drop >= self.trailing_stop_pct:
                            logger.warning(f"⚠️ WS 플러시 중 트레일링 스탑 감지 ({price:,.0f}원) → 즉시 청산")
                            self.execute_sell(price, reason="WS 플러시 트레일링 스탑")
                            break
                    flushed += 1
            except queue.Empty:
                pass
            if flushed > 0:
                logger.debug(f"🧹 WS 큐 처리 완료: {flushed}개")

    def _trail_stop_monitor(self):
        """
        독립 트레일링 스탑 감시 스레드 (Improvement 2).
        전체 asyncio 리팩터링 없이 핵심 문제만 해결:
        - API 블로킹 중에도 10 Hz로 가격 감시
        - 트레일링 스탑 조건 충족 시 플래그 설정 → 메인 루프가 즉시 청산 실행
        - 실제 주문은 메인 스레드에서만 실행하여 스레드 안전성 보장 (GIL 활용)
        - 디바운스(3초): 스탑 라인 하회가 3초 이상 지속될 때만 진짜 하락으로 인정 → 스탑헌팅 휩쏘 방어
        """
        violation_start_time = None

        while self.is_running:
            # 포지션 없을 때는 Event 해제 대기 (10 Hz 폴링 대신 절전)
            # execute_buy 성공 시 _ts_active.set(), execute_sell 성공 시 _ts_active.clear()
            if not self._ts_active.wait(timeout=1.0):
                violation_start_time = None
                continue
            try:
                price = self.current_price
                if price > 0 and self.is_holding and not self._ts_triggered:
                    with self._highest_price_lock:
                        if price > self.highest_price:
                            self.highest_price = price
                        snap_high = self.highest_price
                    drop = (snap_high - price) / snap_high if snap_high > 0 else 0
                    if drop >= self.trailing_stop_pct:
                        if violation_start_time is None:
                            violation_start_time = time.time()
                        elif time.time() - violation_start_time >= 3.0:
                            self._ts_trigger_price = price
                            self._ts_triggered = True  # 메인 루프가 감지해 execute_sell 실행
                            violation_start_time = None
                            logger.warning(
                                f"🚨 [모니터 스레드] 트레일링 스탑 신호 ({price:,.0f}원, "
                                f"고점 대비 -{drop*100:.2f}%, 3초 지속 확인) — 메인 루프 청산 예약"
                            )
                    else:
                        violation_start_time = None  # 가격 회복 시 카운터 초기화
                else:
                    violation_start_time = None  # 포지션 없을 때 타이머 초기화
            except Exception:
                pass
            time.sleep(0.1)  # 10 Hz: API 블로킹 최대 0.1초 지연만 허용

    def _sync_account_state(self):
        """
        로컬 상태 파일을 신뢰하여 봇 포지션을 복원합니다.
        upbit.get_balance()로 전체 잔고를 보는 방식을 폐기하고,
        봇이 직접 발생시킨 주문 UUID/수량 기준으로 격리합니다.
        (사용자의 개인 장기 보유 물량에 절대 간섭하지 않음)
        """
        try:
            current_price = self._safe_api_get(pyupbit.get_current_price, self.ticker)
            self.current_price = current_price if current_price else 0.0

            local_state = self._load_local_state()

            # Kelly EMA payoff 복원 (재시작 시에도 누적 실측값 유지)
            self.payoff_ema = local_state.get('payoff_ema', KELLY_PAYOFF_RATIO)
            self.payoff_trade_count = int(local_state.get('payoff_trade_count', 0))
            # NAV 장부 복원
            self._initial_capital = float(local_state.get('initial_capital', 0.0))
            self._realized_pnl    = float(local_state.get('realized_pnl', 0.0))
            self._last_reconciliation_day = int(local_state.get('last_reconciliation_day', -1))
            self._peak_nav = float(local_state.get('peak_nav', 0.0))
            # 서킷 브레이커 상태 복원 (당일 손실·연속패배 재시작 후에도 유지)
            self._cb_daily_open_nav    = float(local_state.get('cb_daily_open_nav', 0.0))
            self._cb_consecutive_losses = int(local_state.get('cb_consecutive_losses', 0))
            self._cb_cooldown_until    = float(local_state.get('cb_cooldown_until', 0.0))
            self._cb_last_reset_day    = int(local_state.get('cb_last_reset_day', -1))
            self._timestop_cooldown_until = float(local_state.get('timestop_cooldown_until', 0.0))

            if local_state.get('is_holding'):
                self.is_holding = True
                self.buy_price = local_state.get('buy_price', 0.0)
                saved_highest = local_state.get('highest_price', 0.0)
                self.highest_price = max(self.buy_price, self.current_price, saved_highest)
                self.bot_uuid = local_state.get('bot_uuid')
                self.bot_quantity = local_state.get('bot_quantity', 0.0)
                self.buy_time = local_state.get('buy_time', 0.0)
                self._ts_active.set()   # 포지션 보유 → 모니터 스레드 활성화
                logger.info(
                    f"💾 봇 포지션 복원: 평단 {self.buy_price:,.0f}원 | "
                    f"고점 {self.highest_price:,.0f}원 | "
                    f"수량 {self.bot_quantity:.6f} {self.currency}"
                )
            else:
                # 미아 포지션 교차검증: UUID는 저장됐지만 is_holding=False인 경우 (크래시 복구)
                saved_uuid = local_state.get('bot_uuid')
                if saved_uuid:
                    is_split_order = str(saved_uuid).startswith('split_')
                    saved_buy_price = float(local_state.get('buy_price', 0.0))
                    saved_bot_quantity = float(local_state.get('bot_quantity', 0.0))

                    exec_vol = 0.0
                    avg_price = 0.0

                    if is_split_order:
                        # 분할 매수 UUID(split_xxx)는 Upbit에 없으므로 상태파일 수량·평단 신뢰 후 잔고 검증
                        if saved_bot_quantity > 0 and saved_buy_price > 0:
                            actual_balance = self._safe_api_get(upbit.get_balance, self.currency) or 0.0
                            if actual_balance >= saved_bot_quantity * 0.90:
                                exec_vol = saved_bot_quantity
                                avg_price = saved_buy_price
                    else:
                        order_detail = self._safe_api_get(upbit.get_order, saved_uuid)
                        if order_detail:
                            exec_vol = float(order_detail.get('executed_volume', 0) or 0)
                            avg_price = float(
                                order_detail.get('avg_price') or order_detail.get('price') or 0
                            )
                            if exec_vol > 0 and avg_price > 0:
                                actual_balance = self._safe_api_get(upbit.get_balance, self.currency) or 0.0
                                if actual_balance < exec_vol * 0.90:
                                    exec_vol = 0.0  # 잔고 없음 → 이미 청산됨

                    if exec_vol > 0 and avg_price > 0:
                        actual_balance = self._safe_api_get(upbit.get_balance, self.currency) or 0.0
                        self.is_holding = True
                        self.buy_price = avg_price
                        self.bot_uuid = saved_uuid
                        self.bot_quantity = min(exec_vol, actual_balance) if actual_balance > 0 else exec_vol
                        self.buy_time = local_state.get('buy_time', time.time())
                        self.highest_price = max(self.buy_price, self.current_price)
                        self._ts_active.set()  # 미아 포지션 복구 → 모니터 활성화
                        tag = "분할" if is_split_order else "단일"
                        logger.warning(
                            f"⚠️ [미아 포지션 복구({tag})] UUID={str(saved_uuid)[:12]}... "
                            f"qty={self.bot_quantity:.6f} @ {self.buy_price:,.0f}원"
                        )
                        notify(
                            f"⚠️ [{self.ticker}] 미아 포지션 자동 복구 ({tag})!\n"
                            f"  평단: {self.buy_price:,.0f}원 | 수량: {self.bot_quantity:.6f}"
                        )
                        self._save_local_state()
                        self._flush_ws_queue()
                        return

                self.is_holding = False
                self.buy_price = 0.0
                self.highest_price = 0.0
                self.bot_uuid = None
                self.bot_quantity = 0.0
                self._ts_active.clear()  # 포지션 없음 → 모니터 스레드 절전
                logger.info("🔄 계좌: [미보유] 포지션 대기 중")

            self._save_local_state()
            self._flush_ws_queue()

        except Exception as e:
            logger.error(f"계좌 동기화 에러: {e}")
            self.is_holding = False
            self._ts_active.clear()
            self.buy_price = 0.0
            self.highest_price = 0.0

    def _get_ai_prediction(self, is_entry=True) -> float:
        try:
            # 캔들 경계 직후(3초 이내) REST API 업데이트 미완료 방지
            interval_seconds = INTERVAL_MINUTES.get(self.interval, 15) * 60
            kst_now_dt = datetime.now(pytz.timezone('Asia/Seoul'))
            elapsed_in_interval = (
                kst_now_dt.hour * 3600 + kst_now_dt.minute * 60 + kst_now_dt.second
            ) % interval_seconds
            if elapsed_in_interval < 3:
                wait = 4 - elapsed_in_interval
                logger.debug(f"캔들 경계 감지, {wait}초 대기 후 조회")
                time.sleep(wait)

            # 700봉: WFO HMM train_window=500 초과 보장 (480봉 시 range(500,479,240) 공범위 → 전봉 횡보 폴백 버그)
            df = self._safe_api_get(pyupbit.get_ohlcv, self.ticker, interval=self.interval, count=700)
            if df is None or df.empty:
                return -1.0

            kst_now = pd.Timestamp.now(tz=pytz.timezone('Asia/Seoul')).replace(tzinfo=None)
            latest_forming_candle_time = df.index[-1]
            time_diff_seconds = (kst_now - latest_forming_candle_time).total_seconds()

            interval_minutes = INTERVAL_MINUTES.get(self.interval, 60)
            max_allowed_delay = (interval_minutes * 60) + 120

            if time_diff_seconds > max_allowed_delay or time_diff_seconds < -60:
                logger.warning("⚠️ 차트 지연 감지. 추론 보류")
                return -1.0

            completed_df = df.iloc[:-1]  # 항상 완성된 캔들만 사용 (미완성봉 Jittering 방지)

            self._ohlcv_cache = {'df': df, 'boundary': self.last_candle_boundary, 'count': len(df)}

            if is_entry:
                last_closed_time = completed_df.index[-1]
                # MTF: 60m OHLCV 마지막 시간은 :15/:30/:45 모두 동일 → 15m 경계값으로 중복 체크
                _eval_key = self.last_candle_boundary if self._mtf_enabled else last_closed_time
                if _eval_key == self.last_evaluated_candle_time:
                    logger.debug(f"[{self.ticker}] 중복 캔들 스킵 ({last_closed_time})")
                    return -1.0
                self.last_evaluated_candle_time = _eval_key

            target_df = completed_df

            df_features, _ = add_technical_indicators(target_df)
            if df_features is None or df_features.empty:
                return -1.0

            # BTC 컨텍스트 피처 주입 (모델이 BTC_* 피처로 학습된 경우)
            _btc_needed = [f for f in self.feat_order if f.startswith('BTC_')]
            if _btc_needed:
                df_features = self._inject_btc_features(df_features, _btc_needed)

            X = df_features.iloc[-1:][self.feat_order]
            if X.isnull().values.any():
                logger.warning("⚠️ 피처에 NaN 포함. 추론 보류")
                return -1.0

            # ATR_Ratio 캐싱: execute_buy의 동적 비중 계산에 사용
            if 'ATR_Ratio' in df_features.columns:
                self.last_atr_ratio = float(df_features.iloc[-1]['ATR_Ratio'])

            ai_prob = self.model.predict_proba(X)[0][1]

            # ── MTF AND 게이트: 15m 타이밍 모델 동시 동의 필요 (진입 시에만 적용) ──────
            if is_entry and self._mtf_enabled and self.model_15m and self.features_15m:
                try:
                    _mtf_bars = getattr(bot_config, "MTF_ENSEMBLE", {}).get(
                        "intervals", {}).get("minute15", {}).get("ohlcv_bars", 700)
                    df_15m = self._safe_api_get(
                        pyupbit.get_ohlcv, self.ticker, interval="minute15", count=_mtf_bars
                    )
                    if df_15m is not None and not df_15m.empty:
                        df_15m_feat, _ = add_technical_indicators(df_15m.iloc[:-1])
                        if df_15m_feat is not None and not df_15m_feat.empty:
                            _missing_15m = [f for f in self.feat_order_15m if f not in df_15m_feat.columns]
                            for _mf in _missing_15m:
                                df_15m_feat[_mf] = 0.0
                            X_15m = df_15m_feat.iloc[-1:][self.feat_order_15m]
                            if not X_15m.isnull().values.any():
                                prob_15m = self.model_15m.predict_proba(X_15m)[0][1]
                                if prob_15m < self.threshold_15m:
                                    logger.debug(
                                        f"📐 MTF AND 차단: 15m={prob_15m:.3f} < {self.threshold_15m:.3f}"
                                    )
                                    return -1.0
                                logger.debug(f"📐 MTF AND 통과: 15m={prob_15m:.3f} ≥ {self.threshold_15m:.3f}")
                except Exception as _e:
                    logger.warning(f"⚠️ MTF 15m 추론 오류 ({_e}) — 60m 단독으로 진행")
            # ──────────────────────────────────────────────────────────────────────

            # HMM 국면 갱신
            # 재학습 시 AI 추론용 480봉과 별도로 lookback 크기의 데이터를 추가 페칭
            # (480봉 ≒ 5일 → lookback=1440 ≒ 15일과 불일치 해소)
            if self.hmm is not None:
                if self.hmm.needs_retrain():
                    # 이전 재학습이 아직 실행 중이면 중복 시작 방지
                    if self._hmm_retrain_thread is None or not self._hmm_retrain_thread.is_alive():
                        hmm_count = self.hmm.lookback + 50
                        hmm_df = self._safe_api_get(
                            pyupbit.get_ohlcv, self.ticker,
                            interval=self.interval, count=hmm_count,
                        )
                        train_src = hmm_df if (hmm_df is not None and not hmm_df.empty) else completed_df
                        self._hmm_retrain_thread = threading.Thread(
                            target=self.hmm.fit,
                            args=(train_src,),
                            daemon=True,
                            name="HMMRetrain",
                        )
                        self._hmm_retrain_thread.start()
                    else:
                        logger.debug("⏸️ HMM 재학습 스레드 실행 중 — 중복 시작 건너뜀")
                self._current_regime = self.hmm.predict(completed_df)
                logger.debug(f"🔮 HMM 국면: {self._current_regime}")

            del df, target_df, df_features, X
            return float(ai_prob)

        except Exception as e:
            logger.error(f"AI 추론 오류: {e}")
            return -1.0

    def _compute_position_size(self, krw_balance: float, ai_prob: float = 0.0) -> float:
        """
        Half-Kelly 기반 포지션 비중 계산.
        1단계: Half-Kelly로 베팅 비율 산출 (과적합/파산 방지를 위해 풀 켈리의 50%)
          kelly_f = (p × b - (1-p)) / b  (b = 손익비, p = AI 상승 확률)
          half_kelly = kelly_f × 0.5
        2단계: ATR 기반 변동성 스케일링으로 실제 포지션 비중 결정
          포지션 = min(half_kelly, ATR_raw_pct)를 [5%, 40%]로 클램프
        ATR·AI확률 미제공 시 고정 POSITION_SIZE_PCT(20%) 폴백.
        """
        atr = self.last_atr_ratio

        # Half-Kelly 계산 (ai_prob > 0일 때만)
        # 누적 거래가 충분하면 실측 EMA payoff 사용, 부족하면 이론값 폴백
        kelly_fraction = 0.0
        if ai_prob > 0.0:
            b = self.payoff_ema if self.payoff_trade_count >= KELLY_EMA_MIN_TRADES else KELLY_PAYOFF_RATIO
            p = ai_prob
            raw_kelly = (p * b - (1 - p)) / b
            kelly_fraction = max(0.0, raw_kelly) * 0.5  # Half-Kelly, 음수 베팅 불가

        if atr <= 0:
            if kelly_fraction > 0:
                clamped_pct = max(MIN_POSITION_PCT, min(MAX_POSITION_PCT, kelly_fraction))
                logger.debug(f"📐 Half-Kelly 비중(ATR 없음): p={ai_prob:.2f} → {clamped_pct*100:.1f}%")
                return krw_balance * clamped_pct
            return krw_balance * POSITION_SIZE_PCT

        atr_raw_pct = ATR_RISK_PCT / (atr * ATR_STOP_MULTIPLE)
        # ATR·Kelly 모두 있으면 보수적인 쪽 채택 (양쪽 필터의 교집합)
        raw_pct = min(kelly_fraction, atr_raw_pct) if kelly_fraction > 0 else atr_raw_pct
        clamped_pct = max(MIN_POSITION_PCT, min(MAX_POSITION_PCT, raw_pct))
        logger.debug(
            f"📐 Half-Kelly + ATR 비중: p={ai_prob:.2f} kelly={kelly_fraction*100:.1f}% "
            f"atr={atr_raw_pct*100:.1f}% → {clamped_pct*100:.1f}% "
            f"(clamp {MIN_POSITION_PCT*100:.0f}~{MAX_POSITION_PCT*100:.0f}%)"
        )
        return krw_balance * clamped_pct

    # =========================================================================
    # 시장 컨텍스트 헬퍼
    # =========================================================================

    def _get_btc_price_cached(self) -> float:
        """BTC/KRW 현재가 — 60초 캐시 (비BTC 종목 거래 시 김치 프리미엄 계산용)"""
        if self.ticker == "KRW-BTC":
            return self.current_price
        now = time.time()
        if now - self._btc_price_cache['ts'] < 60 and self._btc_price_cache['price'] > 0:
            return self._btc_price_cache['price']
        price = self._safe_api_get(pyupbit.get_current_price, "KRW-BTC") or 0.0
        self._btc_price_cache = {'price': float(price), 'ts': now}
        return float(price)

    def _get_usdt_krw_cached(self) -> float:
        """USDT/KRW 현재가 — 60초 캐시 (≈ USD/KRW 환율 대체)"""
        now = time.time()
        if now - self._usdt_krw_cache['ts'] < 60 and self._usdt_krw_cache['price'] > 0:
            return self._usdt_krw_cache['price']
        price = self._safe_api_get(pyupbit.get_current_price, "KRW-USDT") or 0.0
        self._usdt_krw_cache = {'price': float(price), 'ts': now}
        return float(price)

    def _get_dynamic_trailing_stop(self) -> float:
        """ATR 및 공포-탐욕 지수에 따른 트레일링 스탑 비율 동적 조정 (Clamping 적용)"""
        # 기본 TS 비율 대신 최근 ATR Ratio를 활용 (예: ATR의 1.5배)
        base_ts_pct = self.last_atr_ratio * 1.5 if self.last_atr_ratio > 0 else self.trailing_stop_pct
        
        # 🚨 [개선] 극단적 횡보장 휩쏘(스탑헌팅) 및 폭주장 수익 반납 방어를 위한 하드 리미트
        MIN_TS_PCT = 0.015  # 최소 1.5% 보장 (노이즈에 털림 방지)
        MAX_TS_PCT = 0.050  # 최대 5.0% 제한 (과도한 이익 반납 방지)
        clamped_ts_pct = max(MIN_TS_PCT, min(MAX_TS_PCT, base_ts_pct))
        
        if self.fear_greed is None:
            return clamped_ts_pct
            
        return self.fear_greed.dynamic_trailing_stop(clamped_ts_pct)
    # =========================================================================
    # 분할 진입/청산 (SPLIT_ORDER_ENABLED = True 일 때만 호출됨)
    # =========================================================================

    def _execute_split_buy(self, current_price: float, total_amount: float) -> bool:
        """total_amount를 N조각으로 나눠 시장가 분할 매수"""
        n = SPLIT_ORDER_N_SPLITS
        chunk = total_amount / n
        total_qty = 0.0
        total_krw = 0.0

        notify(f"📊 [분할 매수 시작] {self.ticker} | {n}회 × {chunk:,.0f}원")

        for i in range(n):
            try:
                if i > 0:
                    time.sleep(SPLIT_ORDER_DELAY_SEC)
                    # 가격 역행 체크: 1% 이상 불리해지면 나머지 중단
                    new_price = self._safe_api_get(pyupbit.get_current_price, self.ticker)
                    if new_price and new_price > current_price * 1.01:
                        logger.warning(
                            f"⚠️ 분할 매수 {i+1}/{n}: 가격 역행 "
                            f"({new_price:,.0f}원 > {current_price*1.01:,.0f}원) → 중단"
                        )
                        break

                if not is_dry_run():
                    order = upbit.buy_market_order(self.ticker, chunk)
                    if not order or 'error' in order:
                        logger.error(f"분할 매수 {i+1}/{n} 실패: {order}")
                        continue
                    uuid = order.get('uuid')
                    if uuid:
                        time.sleep(0.5)
                        detail = self._safe_api_get(upbit.get_order, uuid)
                        if detail:
                            exec_vol = float(detail.get('executed_volume', 0))
                            exec_funds = float(detail.get('executed_funds', chunk))
                            total_qty += exec_vol
                            total_krw += exec_funds
                            logger.info(f"  ✅ 분할 {i+1}/{n}: {exec_vol:.6f} {self.currency}")
                else:
                    qty = chunk / current_price if current_price > 0 else 0
                    total_qty += qty
                    total_krw += chunk
                    logger.info(f"  [DRY] 분할 {i+1}/{n}: {qty:.6f} {self.currency}")

            except Exception as e:
                logger.error(f"분할 매수 {i+1}/{n} 오류: {e}")

        if total_qty > 0:
            self.bot_quantity = total_qty
            self.buy_price = total_krw / total_qty
            self.bot_uuid = f"split_{int(time.time())}"
            self.buy_time = time.time()
            self.is_holding = True
            self.highest_price = self.buy_price
            self._ts_active.set()   # 분할 매수 완료 → 트레일링 스탑 모니터 활성화
            self._save_local_state()
            notify(
                f"✅ [분할 매수 완료] {self.ticker}\n"
                f"  수량: {total_qty:.6f} {self.currency} | 평단: {self.buy_price:,.0f}원"
            )
            return True
        return False

    def _execute_split_sell(self, reason: str) -> bool:
        """bot_quantity를 N조각으로 나눠 시장가 분할 매도"""
        n = SPLIT_ORDER_N_SPLITS
        chunk_qty = self.bot_quantity / n
        total_sold = 0.0

        notify(f"📊 [분할 매도 시작] {self.ticker} | {n}회 × {chunk_qty:.6f} {self.currency}")

        for i in range(n):
            try:
                if i > 0:
                    time.sleep(SPLIT_ORDER_DELAY_SEC)

                if not is_dry_run():
                    order = upbit.sell_market_order(self.ticker, chunk_qty)
                    if not order or 'error' in order:
                        logger.error(f"분할 매도 {i+1}/{n} 실패: {order}")
                        continue
                    total_sold += chunk_qty
                    logger.info(f"  ✅ 분할 {i+1}/{n}: {chunk_qty:.6f} {self.currency} 매도")
                else:
                    total_sold += chunk_qty
                    logger.info(f"  [DRY] 분할 {i+1}/{n}: {chunk_qty:.6f} {self.currency} 매도")

            except Exception as e:
                logger.error(f"분할 매도 {i+1}/{n} 오류: {e}")

        if total_sold > 0:
            notify(
                f"✅ [분할 매도 완료] {self.ticker} | "
                f"총 {total_sold:.6f} {self.currency} | 사유: {reason}"
            )
            return True
        return False

    def execute_buy(self, current_price: float, ai_prob: float,
                    position_multiplier: float = 1.0) -> bool:
        try:
            # ── [안전장치] 업비트 지갑 락업 + DAXA 투자유의종목 검증 ──────────
            if self.upbit_risk is not None:
                if not self.upbit_risk.is_safe_to_trade(self.ticker):
                    logger.warning(
                        f"⛔ [UpbitRisk] {self.ticker} 지갑 락업 또는 DAXA 투자유의 → 매수 차단"
                    )
                    return False

            # P1a: 고정 시드 or 잔고 비례 예산 산출 (외부 입금 왜곡 방지)
            krw_balance = self._get_bot_budget()
            if krw_balance <= 0:
                return False
            # 최초 매수 시점에 초기 자본 확정 (NAV 추적 기준값 — CB baseline과 동일 기준 유지)
            if self._initial_capital <= 0:
                self._initial_capital = (BOT_INITIAL_CAPITAL * self.capital_fraction
                                         if BOT_INITIAL_CAPITAL > 0
                                         else krw_balance)

            if krw_balance < SAFE_MIN_KRW_BALANCE:
                now = time.time()
                if now - self._last_insufficient_funds_log_time > 3600:
                    logger.warning(f"⚠️ 잔고 부족 (보유: {krw_balance:,.0f}원 / 필요: {SAFE_MIN_KRW_BALANCE}원)")
                    self._last_insufficient_funds_log_time = now
                return False

            # Half-Kelly + ATR 기반 동적 비중 계산 (폴백: 고정 20%)
            # position_multiplier: HMM 약세/횡보 국면 시 0.5배 적용
            buy_amount = (
                self._compute_position_size(krw_balance, ai_prob)
                * (1 - FEE_RATE)
                * position_multiplier
            )

            # P0: 유동성 기반 주문 상한 — 24h 거래대금의 LIQUIDITY_IMPACT_CAP(1%) 초과 차단
            liq_cap = self._get_liquidity_cap_krw()
            if buy_amount > liq_cap:
                logger.warning(
                    f"⚠️ [{self.ticker}] 주문({buy_amount:,.0f}원) > 유동성 상한({liq_cap:,.0f}원) "
                    f"— 캡 적용 (슬리피지 방지)"
                )
                buy_amount = liq_cap

            if buy_amount < MIN_ORDER_KRW:
                return False

            conviction_tag = "🔥 초강세" if ai_prob >= STRONG_CONVICTION_THRESHOLD else "일반"
            regime_tag = f" | 국면:{self._current_regime}" if self.hmm else ""
            notify(f"🚀 [매수 {conviction_tag} {'시뮬레이션' if is_dry_run() else '요청'}] {self.ticker}\n"
                   f"- 현재가: {current_price:,.0f}원\n"
                   f"- 상승확률: {ai_prob*100:.1f}%{regime_tag}\n"
                   f"- 금액: {buy_amount:,.0f}원 (ATR={self.last_atr_ratio:.4f}"
                   f" | 비중배수={position_multiplier:.1f})\n"
                   f"- 자본비중: {self.capital_fraction*100:.0f}%")

            # 분할 매수: 변동성 높고 시장가 주문일 때 슬리피지 분산
            # (지정가 경로는 이미 내부에서 단일 주문 + 취소 로직을 가지므로 제외)
            if SPLIT_ORDER_ENABLED and self.last_atr_ratio >= SPLIT_ORDER_ATR_THRESHOLD:
                logger.info(
                    f"✂️ 분할 매수 활성화 (ATR={self.last_atr_ratio:.4f} ≥ {SPLIT_ORDER_ATR_THRESHOLD})"
                )
                return self._execute_split_buy(current_price, buy_amount)

            if not is_dry_run():
                # 미체결 주문 선검증 — 네트워크 순단 후 중복 매수 방지
                pending = self._safe_api_get(upbit.get_orders, self.ticker, state='wait')
                if pending:
                    logger.warning(f"⚠️ 미체결 주문 {len(pending)}개 발견 — 신규 매수 보류")
                    return False
                try:
                    # 확신도 분기: 초강세(85%↑) 또는 유동성 코인 → 호가창 Ask 뎁스 확인 후 시장가/지정가 결정
                    #              일반 알트 신호 → 지정가로 타점 대기 후 미체결 시 포기
                    high_conviction = ai_prob >= STRONG_CONVICTION_THRESHOLD
                    use_market_intent = high_conviction or self.currency in LIQUID_COINS

                    execute_as_market = False
                    _best_ask_size = 0.0  # Depth-Adaptive TWAP 판단용
                    if use_market_intent:
                        # 호가창 뎁스 확인: Ask 잔량이 얇으면 슬리피지 -1~2% 대참사 방지
                        orderbook = self._safe_api_get(pyupbit.get_orderbook, self.ticker)
                        if orderbook and 'orderbook_units' in orderbook[0]:
                            units = orderbook[0]['orderbook_units']
                            _best_ask_size = units[0]['ask_size']  # Depth-TWAP: 1호가 매도 잔량
                            # OBI(Order Book Imbalance): 매수/매도 호가 잔량 불균형 측정
                            # 음수(매도 우위)가 OBI_THRESHOLD(-0.2) 미만이면 스푸핑 의심 → 진입 보류
                            bid_vol = sum(u['bid_size'] for u in units[:5])
                            ask_vol_top5 = sum(u['ask_size'] for u in units[:5])
                            obi = (bid_vol - ask_vol_top5) / (bid_vol + ask_vol_top5) if (bid_vol + ask_vol_top5) > 0 else 0
                            acceptable_price = current_price * 1.005  # 최대 허용 슬리피지 +0.5%
                            available_vol = sum(u['ask_size'] for u in units if u['ask_price'] <= acceptable_price)
                            required_vol = buy_amount / current_price
                            depth_ok = available_vol >= required_vol
                            obi_ok = obi >= OBI_THRESHOLD
                            if depth_ok and obi_ok:
                                reason = "초강세 확신도" if high_conviction else "유동성 코인"
                                logger.info(
                                    f"⚡ {reason} | 뎁스 충분(필요:{required_vol:.2f}≤가능:{available_vol:.2f}) "
                                    f"OBI={obi:.2f} → 시장가 매수"
                                )
                                execute_as_market = True
                            elif not depth_ok:
                                logger.warning(f"⚠️ 매도 호가 얇음(필요:{required_vol:.2f}>가능:{available_vol:.2f}) → 지정가 폴백")
                            else:
                                logger.warning(
                                    f"⚠️ OBI={obi:.2f} < {OBI_THRESHOLD} (매도압력 우위/스푸핑 의심) → 지정가 폴백"
                                )
                        else:
                            execute_as_market = True  # 호가창 조회 실패 시 기존 동작 유지

                    if execute_as_market:
                        # Depth-Adaptive TWAP: 주문량이 1호가 잔량의 20% 초과 시 분할 매수로 가격 충격 최소화
                        # (available_vol 체크는 통과했더라도 1호가를 크게 먹으면 슬리피지 발생)
                        _required_vol = buy_amount / current_price if current_price > 0 else 0.0
                        if (SPLIT_ORDER_ENABLED and _best_ask_size > 0
                                and _required_vol > _best_ask_size * 0.2):
                            logger.warning(
                                f"⚠️ [Depth-TWAP] 주문량({_required_vol:.4f}) > "
                                f"1호가 잔량({_best_ask_size:.4f}) × 20% → 분할 매수 전환"
                            )
                            return self._execute_split_buy(current_price, buy_amount)
                        _ORDER_BUCKET.wait()
                        order = upbit.buy_market_order(self.ticker, buy_amount)
                    else:
                        # 알트 일반: 지정가 → 미체결 시 타점 포기
                        limit_price = round(current_price * (1 + LIMIT_ORDER_BUFFER))
                        limit_volume = buy_amount / limit_price
                        _ORDER_BUCKET.wait()
                        order = upbit.buy_limit_order(self.ticker, limit_price, limit_volume)
                        if order and 'error' not in order:
                            lmt_uuid = order.get('uuid')
                            logger.info(f"📋 지정가 매수 대기: {limit_price:,.0f}원 × {limit_volume:.4f} ({LIMIT_ORDER_TIMEOUT}초)")
                            time.sleep(LIMIT_ORDER_TIMEOUT)

                            # 거래소 API로 실제 체결 수량 직접 조회 (STATE_FILE 의존 제거 — 부분 체결 버그 수정)
                            order_status = self._safe_api_get(upbit.get_order, lmt_uuid) if lmt_uuid else None
                            exec_vol = float(order_status.get('executed_volume', 0)) if order_status else 0.0

                            if exec_vol > 0:
                                # 부분/전체 체결: 봇 포지션 즉시 등록 후 잔여 미체결 취소
                                self.bot_uuid     = lmt_uuid
                                self.bot_quantity = exec_vol
                                self.buy_price    = float(order_status.get('avg_price') or order_status.get('price') or limit_price)
                                self.buy_time     = time.time()
                                self.is_holding   = True
                                self.highest_price = self.buy_price
                                self._save_local_state()
                                if order_status.get('state') not in ('done', 'cancel'):
                                    try:
                                        upbit.cancel_order(lmt_uuid)
                                    except Exception:
                                        pass
                                fill_tag = "전체" if order_status.get('state') == 'done' else "부분"
                                logger.info(f"✅ 지정가 {fill_tag} 체결: {exec_vol:.6f} {self.currency} @ {self.buy_price:,.0f}원")
                                return True

                            # exec_vol == 0: 미체결 → 전량 취소 후 취소 완료 확인 (이중 매수 원천 차단)
                            remaining = self._safe_api_get(upbit.get_orders, self.ticker, state='wait')
                            if remaining:
                                for p in remaining:
                                    try:
                                        upbit.cancel_order(p['uuid'])
                                    except Exception:
                                        pass
                                time.sleep(1)
                                still_open = []
                                for p in remaining:
                                    status = self._safe_api_get(upbit.get_order, p['uuid'])
                                    if status and status.get('state') not in ('cancel', 'done'):
                                        still_open.append(p['uuid'])
                                if still_open:
                                    logger.error(
                                        f"❌ 지정가 취소 미확인 ({len(still_open)}개) — "
                                        f"이중 매수 방지를 위해 진입 포기"
                                    )
                                    return False

                            # ── 강세장 단일 재시도 (1회 한정) ────────────────────────
                            # 가격이 limit_price를 위로 돌파했다면 = 강세 모멘텀 확인.
                            # 이 경우에만 갱신된 지정가로 1회 재도전 (시장가 추격은 여전히 금지).
                            retry_price = self._safe_api_get(pyupbit.get_current_price, self.ticker)
                            if retry_price and retry_price > limit_price:
                                new_limit = round(retry_price * (1 + LIMIT_ORDER_BUFFER))
                                new_vol   = buy_amount / new_limit
                                logger.info(
                                    f"🔄 [강세장 재시도] 가격 상승 확인 "
                                    f"({limit_price:,.0f} → {retry_price:,.0f}원) "
                                    f"→ 갱신 지정가 {new_limit:,.0f}원 단 1회 재시도"
                                )
                                _ORDER_BUCKET.wait()
                                r_order = upbit.buy_limit_order(self.ticker, new_limit, new_vol)
                                if r_order and 'error' not in r_order:
                                    r_uuid = r_order.get('uuid')
                                    time.sleep(5)  # 단축 대기 (이미 늦었으므로 5초)
                                    r_status = self._safe_api_get(upbit.get_order, r_uuid) if r_uuid else None
                                    r_exec_vol = float(r_status.get('executed_volume', 0)) if r_status else 0.0
                                    if r_exec_vol > 0:
                                        self.bot_uuid     = r_uuid
                                        self.bot_quantity = r_exec_vol
                                        self.buy_price    = float(r_status.get('price', new_limit))
                                        self.buy_time     = time.time()
                                        self.is_holding   = True
                                        self.highest_price = self.buy_price
                                        self._save_local_state()
                                        fill_tag = "전체" if r_status.get('state') == 'done' else "부분"
                                        logger.info(
                                            f"✅ [재시도 {fill_tag} 체결] "
                                            f"{r_exec_vol:.6f} {self.currency} @ {self.buy_price:,.0f}원"
                                        )
                                        return True
                                    # 재시도도 미체결 → 취소 후 완전 포기
                                    if r_uuid:
                                        try:
                                            upbit.cancel_order(r_uuid)
                                        except Exception:
                                            pass
                                    logger.warning("⚠️ 갱신 지정가 재시도도 미체결 → 완전 포기")
                            # ─────────────────────────────────────────────────────

                            logger.warning(
                                f"⚠️ 지정가 미체결 컷 ({LIMIT_ORDER_TIMEOUT}초 초과) — "
                                f"불리한 시장가 추격 매수 포기"
                            )
                            notify(f"🛡️ [매수 취소] {self.ticker} 지정가 미체결 → 타점 포기")
                            return False

                    if order is None or 'error' in order:
                        logger.error(f"매수 실패: {order}")
                        return False

                    # 체결 후 UUID + 수량 저장 (봇 포지션 격리 핵심)
                    order_uuid = order.get('uuid')
                    if order_uuid:
                        # 크래시 방어: UUID 즉시 저장 (체결 확인 전이라도 재시작 시 복구 가능)
                        self.bot_uuid = order_uuid
                        self._save_local_state()
                        time.sleep(1)
                        order_detail = self._safe_api_get(upbit.get_order, order_uuid)
                        if order_detail:
                            exec_vol = float(order_detail.get('executed_volume', 0))
                            avg_price = float(order_detail.get('avg_price') or order_detail.get('price') or current_price)
                            if exec_vol > 0:
                                self.bot_quantity = exec_vol
                                self.buy_price    = avg_price if avg_price > 0 else current_price
                                self.buy_time     = time.time()
                                self.is_holding   = True
                                self.highest_price = self.buy_price
                                self._ts_active.set()  # 시장가 체결 확인 → 모니터 즉시 활성화
                                self._save_local_state()
                                logger.info(f"🔑 봇 포지션 등록: UUID={order_uuid[:8]}... qty={exec_vol:.6f} @ {self.buy_price:,.0f}원")

                except Exception as e:
                    logger.error(f"매수 API 예외: {e} — 계좌 재동기화로 체결 여부 확인")
                    time.sleep(3)
                    self._sync_account_state()
                    if self.is_holding:
                        logger.info("✅ 체결 확인됨 (예외 후 로컬 상태 재조회)")
                        return True
                    return False

            else:
                # DRY_RUN: 가상 포지션 등록
                # is_holding=True와 buy_price를 즉시 설정하지 않으면 _sync_account_state()가
                # 상태파일(is_holding=False)을 읽어 포지션을 등록하지 않고, 다음 캔들에도 매수를
                # 반복하는 무한 매수 루프가 발생한다.
                self.bot_quantity  = buy_amount / current_price if current_price > 0 else 0.0
                self.bot_uuid      = f"dry_run_{int(time.time())}"
                self.buy_time      = time.time()
                self.buy_price     = current_price
                self.is_holding    = True
                self.highest_price = current_price
                self._ts_active.set()   # 모니터 스레드 활성화
                self._save_local_state()  # 상태파일 갱신 후 _sync_account_state 호출

            time.sleep(2)
            self._sync_account_state()
            return True

        except Exception as e:
            logger.error(f"매수 프로세스 오류: {e}")
            return False

    def _log_trade(self, sell_price: float, quantity: float,
                   net_profit_pct: float, net_profit_krw: float, reason: str):
        """거래 완료 시 CSV에 한 행 추가 (파일 없으면 헤더 포함 생성)."""
        from datetime import datetime
        os.makedirs("logs", exist_ok=True)
        _hold_min = round((time.time() - self.buy_time) / 60, 1) if self.buy_time > 0 else 0.0
        row = {
            'timestamp': datetime.now().isoformat(),
            'ticker': self.ticker,
            'buy_time': datetime.fromtimestamp(self.buy_time).isoformat() if self.buy_time > 0 else '',
            'hold_duration_min': _hold_min,
            'buy_price': self.buy_price,
            'sell_price': sell_price,
            'quantity': quantity,
            'net_profit_pct': round(net_profit_pct * 100, 4),
            'net_profit_krw': round(net_profit_krw, 0),
            'reason': reason,
            'payoff_ema': round(self.payoff_ema, 4),
            'trade_count': self.payoff_trade_count,
        }
        write_header = not os.path.exists(self._trade_log_path)
        try:
            with open(self._trade_log_path, 'a', encoding='utf-8', newline='') as f:
                import csv
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
        except Exception as e:
            logger.warning(f"거래 로그 기록 실패: {e}")

    def execute_sell(self, current_price: float, reason: str) -> bool:
        try:
            # 봇이 매수한 수량만 청산 (개인 보유량 보호)
            sell_qty = self.bot_quantity
            if sell_qty <= 0:
                # 폴백: bot_quantity 미기록 시 전체 잔고 사용 (구버전 상태 파일 호환)
                sell_qty = self._safe_api_get(upbit.get_balance, self.ticker)
                if sell_qty is None or sell_qty <= 0:
                    return False
                logger.warning("⚠️ bot_quantity 미기록 — 전체 잔고로 청산 (구버전 상태)")

            # P2a: Dust 방어 — 수수료·분할체결 소수점 오차로 장부수량 > 실제잔고 시
            # '잔고 부족' 에러 발생을 원천 차단. 0.9999 계수로 부동소수점 끝자리 여유 확보.
            if not is_dry_run():
                actual_balance = self._safe_api_get(upbit.get_balance, self.currency)
                if actual_balance is not None and actual_balance > 0 and sell_qty > actual_balance:
                    logger.debug(
                        f"[{self.ticker}] Dust 조정: {sell_qty:.8f} → "
                        f"{actual_balance:.8f} {self.currency}"
                    )
                    sell_qty = actual_balance * 0.9999

            estimated_krw = sell_qty * current_price
            slippage = SLIPPAGE_BY_COIN.get(self.currency, SLIPPAGE_DEFAULT)
            conservative_exit_price = current_price * (1 - slippage)
            buy_cost = sell_qty * self.buy_price * (1 + FEE_RATE) if self.buy_price > 0 else 0
            sell_proceeds = sell_qty * conservative_exit_price * (1 - FEE_RATE)
            net_profit_krw = sell_proceeds - buy_cost if buy_cost > 0 else 0
            net_profit_pct = (net_profit_krw / buy_cost) if buy_cost > 0 else 0

            # P2b: NAV 장부 갱신 — 확정 손익(수수료 포함 net)을 누적
            # bot_nav = _initial_capital + _realized_pnl 이 다음 Kelly 계산의 기준값
            if buy_cost > 0:
                self._realized_pnl += net_profit_krw

            # Kelly EMA Payoff 갱신: 실현 손익비(win/loss pct) → EMA 누적
            # 진입 이유 불명 또는 buy_price 미기록 시 갱신 스킵
            if self.buy_price > 0 and buy_cost > 0:
                if net_profit_pct > 0:
                    trade_payoff = net_profit_pct / max(self.last_atr_ratio * 1.5, 0.001)
                else:
                    # 손실 거래: 0.0 입력으로 EMA 희석 (-5% 손실 시 0.95가 아닌 0.0 반영)
                    # 클램핑(max 0.5)이 최종 EMA 붕괴를 막으므로 입력 하한 불필요
                    trade_payoff = 0.0
                self.payoff_ema = (
                    KELLY_EMA_ALPHA * trade_payoff
                    + (1 - KELLY_EMA_ALPHA) * self.payoff_ema
                )
                self.payoff_ema = max(0.5, min(5.0, self.payoff_ema))  # 극단값 클램핑
                self.payoff_trade_count += 1
                logger.debug(
                    f"📊 Kelly EMA payoff 갱신: trade_payoff={trade_payoff:.3f} "
                    f"→ ema={self.payoff_ema:.3f} (누적 {self.payoff_trade_count}회)"
                )

            if estimated_krw >= MIN_ORDER_KRW:
                _nav = self._initial_capital + self._realized_pnl
                notify(f"💥 [{reason} {'시뮬레이션' if is_dry_run() else '청산'}] {self.ticker}\n"
                       f"- 매도단가: {current_price:,.0f}원\n"
                       f"- 평단가: {self.buy_price:,.0f}원\n"
                       f"- 실순수익: {net_profit_pct*100:.2f}% ({net_profit_krw:,.0f}원)\n"
                       f"- Kelly EMA payoff: {self.payoff_ema:.3f} ({self.payoff_trade_count}회 기반)\n"
                       f"- 자본비중: {self.capital_fraction*100:.0f}% | NAV {_nav:,.0f}원")

                # 분할 매도: ATR 높을 때 호가창에 던지는 충격 분산
                if SPLIT_ORDER_ENABLED and self.last_atr_ratio >= SPLIT_ORDER_ATR_THRESHOLD:
                    success = self._execute_split_sell(reason)
                    if success:
                        self._tick_circuit_breaker(net_profit_pct)
                        self._log_trade(current_price, sell_qty, net_profit_pct, net_profit_krw, reason)
                        self.is_holding = False
                        self.buy_price = 0.0
                        self.highest_price = 0.0
                        self.bot_uuid = None
                        self.bot_quantity = 0.0
                        self.buy_time = 0.0
                        self._ts_active.clear()
                        self._save_local_state()
                        time.sleep(2)
                        self._sync_account_state()
                    return success

                if not is_dry_run():
                    pending = self._safe_api_get(upbit.get_orders, self.ticker, state='wait')
                    if pending:
                        logger.warning(f"⚠️ 미체결 주문 {len(pending)}개 발견 — 신규 매도 보류")
                        return False
                    try:
                        _ORDER_BUCKET.wait()
                        order = upbit.sell_market_order(self.ticker, sell_qty)
                        if order is None or 'error' in order:
                            logger.error(f"매도 실패: {order}")
                            return False
                    except Exception as e:
                        logger.error(f"매도 API 예외: {e} — 계좌 재동기화로 체결 여부 확인")
                        time.sleep(3)
                        self._sync_account_state()
                        if not self.is_holding:
                            logger.info("✅ 청산 확인됨 (예외 후 로컬 상태 재조회)")
                            return True
                        return False

                # 청산 성공: 서킷 브레이커 상태 갱신 후 봇 포지션 초기화
                self._tick_circuit_breaker(net_profit_pct)
                self._log_trade(current_price, sell_qty, net_profit_pct, net_profit_krw, reason)
                self.is_holding = False
                self.buy_price = 0.0
                self.highest_price = 0.0
                self.bot_uuid = None
                self.bot_quantity = 0.0
                self.buy_time = 0.0
                self._ts_active.clear()
                self._save_local_state()  # is_holding=False 먼저 파일 반영 — _sync에서 포지션 재복원 방지
                time.sleep(2)
                self._sync_account_state()
                return True
            else:
                # MIN_ORDER_KRW 미만: 주문 불가, 상태만 초기화
                self.is_holding = False
                self.buy_price = 0.0
                self.highest_price = 0.0
                self.bot_uuid = None
                self.bot_quantity = 0.0
                self._save_local_state()
                return False
        except Exception as e:
            logger.error(f"매도 프로세스 오류: {e}")
            return False

    def run(self):
        last_ws_recv_time = time.time()
        consecutive_errors = 0

        logger.info("🚀 서버 레벨 무한 감시 루프 시작...")

        while self.is_running:
            try:
                current_time = time.time()

                if self.wm is None:
                    try:
                        self.wm = pyupbit.WebSocketManager("ticker", [self.ticker])
                        # 신규 pyupbit: 큐가 __q(private)로 변경, 프로세스 명시 시작 필요
                        if not self.wm.alive:
                            self.wm.alive = True
                            self.wm.start()
                        self._wm_q = (
                            getattr(self.wm, 'q', None) or
                            getattr(self.wm, '_WebSocketManager__q', None)
                        )
                        last_ws_recv_time = current_time
                        logger.info("✅ 웹소켓 스트림 연결 완료")
                        consecutive_errors = 0
                    except Exception as e:
                        raise ConnectionError(f"WS 연결 실패: {e}")

                data_received = False
                try:
                    data = self._wm_q.get(timeout=1)
                    data_received = True
                except queue.Empty:
                    if time.time() - last_ws_recv_time > WS_TIMEOUT:
                        _ws_alive = getattr(self.wm, 'alive', None) or (
                            hasattr(self.wm, 'is_alive') and self.wm.is_alive()
                        )
                        if _ws_alive:
                            # 스레드 살아있음 — 저거래량 틱 공백, 타임아웃만 리셋
                            last_ws_recv_time = time.time()
                        else:
                            logger.warning(
                                f"[{self.ticker}] 웹소켓 응답 멈춤({WS_TIMEOUT}초). 재연결 진행..."
                            )
                            self.wm.terminate()
                            self.wm = None
                            self._wm_q = None
                            continue

                candle_just_closed = False
                if data_received and data:
                    last_ws_recv_time = current_time
                    price = data.get('trade_price')
                    if price:
                        self.current_price = price
                        if self.is_holding:
                            with self._highest_price_lock:
                                if self.current_price > self.highest_price:
                                    self.highest_price = self.current_price
                                    now = time.time()
                                    if now - self._last_state_save_time > STATE_SAVE_THROTTLE:
                                        self._save_local_state()
                                        self._last_state_save_time = now

                    # 거래소 서버 시간 기반 캔들 경계 감지 (로컬 시간 드리프트 방지, 인터벌 동적 계산)
                    exchange_ts_ms = data.get('trade_timestamp')
                    if exchange_ts_ms:
                        exchange_ts_sec = int(exchange_ts_ms) / 1000.0
                        _trigger_interval = "minute15" if self._mtf_enabled else self.interval
                        _candle_sec = INTERVAL_MINUTES.get(_trigger_interval, 15) * 60
                        current_candle_close = (int(exchange_ts_sec) // _candle_sec) * _candle_sec
                        if current_candle_close > self.last_candle_boundary:
                            self.last_candle_boundary = current_candle_close
                            candle_just_closed = True
                            logger.info(f"⏰ [거래소 시간 동기화] {_trigger_interval} 캔들 마감 감지 → AI 분석 트리거")

                if not self.current_price or self.current_price <= 0:
                    continue

                # 독립 모니터 스레드가 트레일링 스탑 신호를 설정한 경우 즉시 청산
                # (API 블로킹 중 누락된 가격 하락을 10 Hz 감시로 포착 → 메인 스레드에서만 주문 실행)
                if self._ts_triggered and self.is_holding:
                    ts_price = self._ts_trigger_price
                    self._ts_triggered = False
                    if self.execute_sell(ts_price, reason="독립 모니터 트레일링 스탑"):
                        continue

                # 주기적 모델 핫-리로드 체크 (5분 간격, 포지션 비보유 시에만 교체)
                if current_time - self._last_model_check_time >= MODEL_RELOAD_INTERVAL:
                    self._last_model_check_time = current_time
                    self._check_and_reload_model()

                # 주기적 heartbeat 저장 (BOT_HEARTBEAT_STALE=120s 이내 갱신 보장)
                if current_time - self._last_heartbeat_save_time > 60:
                    self._save_local_state()
                    self._last_heartbeat_save_time = current_time

                # P1b: 일일 장부 대사 + Telegram 성과 리포트 (UTC 자정 기준, 하루 1회)
                _today_epoch_day = int(current_time) // 86400
                if _today_epoch_day != self._last_reconciliation_day:
                    self._last_reconciliation_day = _today_epoch_day
                    self._daily_reconciliation()
                if _today_epoch_day != self._last_daily_report_day:
                    self._last_daily_report_day = _today_epoch_day
                    if self._initial_capital > 0:
                        _nav = self._initial_capital + self._realized_pnl
                        _pnl_pct = (_nav / self._initial_capital - 1) * 100
                        _peak_str = f"{self._peak_nav:,.0f}원" if self._peak_nav > 0 else "미확정"
                        notify(
                            f"📊 [{self.ticker}] 일일 성과 리포트\n"
                            f"  자본비중   : {self.capital_fraction*100:.0f}%\n"
                            f"  NAV        : {_nav:,.0f}원 (최고 {_peak_str})\n"
                            f"  누적 손익  : {self._realized_pnl:+,.0f}원 ({_pnl_pct:+.2f}%)\n"
                            f"  Kelly EMA  : {self.payoff_ema:.3f} "
                            f"({self.payoff_trade_count}회 기반)"
                        )

                if self.is_holding:
                    self._handle_holding_position(self.current_price, current_time)
                else:
                    # 멀티봇 동시 API burst 방어: 캔들 마감 직후 모든 봇이 동시에 pyupbit.get_ohlcv를
                    # 호출하면 IP당 요청 제한을 순간 초과 가능. 최대 2초 랜덤 지연으로 분산.
                    if candle_just_closed:
                        time.sleep(random.uniform(0, 2.0))
                    self._handle_waiting_position(self.current_price, current_time, candle_just_closed)

                consecutive_errors = 0

            except Exception as e:
                consecutive_errors += 1
                sleep_time = min(300, 2 ** consecutive_errors)
                logger.error(f"메인 루프 에러 ({consecutive_errors}회 연속): {e}. {sleep_time}초 대기...")
                if self.wm:
                    try:
                        self.wm.terminate()
                    except Exception as wm_err:
                        logger.warning(f"WatchdogManager 종료 실패 (무시): {wm_err}")
                    self.wm = None
                    self._wm_q = None
                time.sleep(sleep_time)

        # Graceful Shutdown
        logger.info("🧹 봇 종료 절차 진행 중... (상태 저장)")
        self._save_local_state()
        if self.wm:
            try:
                self.wm.terminate()
            except Exception:
                pass
        if self._owns_liq_scanner and self.liq_scanner is not None:
            self.liq_scanner.shutdown()
        if self._owns_notifier:
            notifier.shutdown()
        logger.info("✅ 봇이 완전히 종료되었습니다.")

    def _handle_holding_position(self, current_price: float, current_time: float) -> None:
        # 1순위: 타임 스탑 — AI 예측 유효 시간 초과 시 청산
        # 단, 수익 중인 포지션(+0.5% 이상)은 면제 — 트레일링 스탑이 이미 보호하고 있음
        if self.buy_time > 0 and (current_time - self.buy_time) > self.time_stop_sec:
            elapsed_min = int((current_time - self.buy_time) / 60)
            profit_pct = (current_price / self.buy_price - 1.0) if self.buy_price > 0 else 0.0
            if profit_pct >= 0.005:
                logger.debug(
                    f"⏰ 타임스탑 도달 ({elapsed_min}분) — 수익 중 ({profit_pct*100:.2f}%) "
                    f"→ 트레일링 스탑에 위임"
                )
            else:
                logger.warning(
                    f"⏰ 타임 스탑: 예측 유효 시간({self.time_stop_sec//60}분) 초과 "
                    f"({elapsed_min}분 경과, 수익 {profit_pct*100:.2f}%) — 능동 청산"
                )
                if self.execute_sell(current_price, reason=f"타임 스탑 ({elapsed_min}분 경과)"):
                    self._timestop_cooldown_until = time.time() + self._timestop_cooldown_sec
                    return

        # 2순위: 트레일링 스탑 (공포-탐욕 지수 기반 동적 조정)
        dynamic_ts = self._get_dynamic_trailing_stop()
        drop_from_high = (self.highest_price - current_price) / self.highest_price if self.highest_price > 0 else 0
        if drop_from_high >= dynamic_ts:
            fgi_val, fgi_lbl = self.fear_greed.fetch() if self.fear_greed else (50, "Neutral")
            reason_ts = f"트레일링 스탑 {dynamic_ts*100:.1f}% (F&G={fgi_val}/{fgi_lbl})"
            if self.execute_sell(current_price, reason=reason_ts):
                return

        # 3순위: AI 능동 청산 (주기적 재평가)
        if current_time - self.last_ai_check_time > self.ai_check_interval:
            self.last_ai_check_time = current_time
            ai_prob = self._get_ai_prediction(is_entry=False)

            if ai_prob != -1.0 and 0 < ai_prob < self.ai_exit_threshold:
                self.execute_sell(current_price, reason=f"AI 능동 청산 (확률 {ai_prob*100:.1f}%)")

    def _tick_circuit_breaker(self, net_profit_pct: float) -> None:
        """
        청산 시 서킷 브레이커 상태 갱신.
        손실 거래 → 연속 패배 카운터 증가 + 한도 초과 시 냉각 타이머 설정.
        수익 거래 → 연속 패배 리셋.
        """
        if not self._cb_enabled:
            return
        if net_profit_pct < 0:
            self._cb_consecutive_losses += 1
            logger.info(f"[CB] 연속 패배 {self._cb_consecutive_losses}/{self._cb_consec_loss_limit}")
            if self._cb_consecutive_losses >= self._cb_consec_loss_limit:
                self._cb_cooldown_until = time.time() + self._cb_cooldown_sec
                self._cb_consecutive_losses = 0
                notify(
                    f"🚨 [{self.ticker}] 서킷 브레이커: 연속 {self._cb_consec_loss_limit}패\n"
                    f"  → {self._cb_cooldown_sec // 60}분 신규 진입 차단"
                )
        else:
            self._cb_consecutive_losses = 0

    def _check_circuit_breaker(self, current_time: float) -> bool:
        """
        신규 진입 허용 여부 반환. False이면 진입 차단.
        1) KST 오전 9시 일일 리셋
        2) 냉각 타이머 체크 (연속 패배)
        3) 일일 손실 한도 체크 (-3%)
        """
        if not self._cb_enabled:
            return True

        # ── 1. 매일 KST 09:00 리셋 ────────────────────────────────────────────
        kst_now = datetime.now(pytz.timezone('Asia/Seoul'))
        kst_day = kst_now.toordinal()
        if kst_day != self._cb_last_reset_day and kst_now.hour >= self._cb_reset_hour_kst:
            self._cb_last_reset_day = kst_day
            self._cb_consecutive_losses = 0
            self._cb_cooldown_until = 0.0
            nav = self._initial_capital + self._realized_pnl
            if nav <= 0:
                # 신규봇: 이론 예산 기준으로 초기화 (actual_krw 변동에 무관한 일관된 기준값)
                nav = (BOT_INITIAL_CAPITAL * self.capital_fraction
                       if BOT_INITIAL_CAPITAL > 0
                       else self._get_bot_budget())
            self._cb_daily_open_nav = max(nav, 0.0)
            logger.info(
                f"[CB] 일일 리셋 완료 (KST {self._cb_reset_hour_kst}시) "
                f"| 기준 NAV={self._cb_daily_open_nav:,.0f}원"
            )
            self._save_local_state()

        # ── 2. 연속 패배 냉각 체크 ───────────────────────────────────────────
        if self._cb_cooldown_until > current_time:
            remaining = int((self._cb_cooldown_until - current_time) / 60)
            logger.info(f"[CB] 냉각 중 — 잔여 {remaining}분. 신규 진입 차단")
            return False

        # ── 3. 일일 손실 한도 체크 ───────────────────────────────────────────
        if self._cb_daily_open_nav > 0:
            nav_now = self._initial_capital + self._realized_pnl
            if nav_now <= 0:
                # _initial_capital 미초기화(첫 거래 전) — 현재 NAV 산출 불가, 스킵
                return True
            daily_loss_pct = (nav_now - self._cb_daily_open_nav) / self._cb_daily_open_nav
            if daily_loss_pct <= self._cb_daily_loss_pct:
                logger.warning(
                    f"[CB] 일일 손실 한도 초과: {daily_loss_pct*100:.2f}% ≤ "
                    f"{self._cb_daily_loss_pct*100:.1f}% — 당일 신규 진입 전면 차단"
                )
                notify(
                    f"🚨 [{self.ticker}] 서킷 브레이커: 일일 손실 한도\n"
                    f"  현재 {daily_loss_pct*100:.2f}% (한도 {self._cb_daily_loss_pct*100:.1f}%)\n"
                    f"  → KST {self._cb_reset_hour_kst}시까지 신규 진입 차단"
                )
                return False

        # ── 4. 절대 MDD 한도 체크 (Peak-to-Valley) ───────────────────────────
        # peak_nav는 이 시점에 갱신하여 매 캔들마다 신고점 추적
        _nav_current = self._initial_capital + self._realized_pnl
        if _nav_current > self._peak_nav and _nav_current > 0:
            self._peak_nav = _nav_current
        _max_dd_limit = bot_config.CIRCUIT_BREAKER.get("max_drawdown_pct", -0.20)
        if _max_dd_limit < 0 and self._peak_nav > 0:
            _total_dd = (_nav_current - self._peak_nav) / self._peak_nav
            if _total_dd <= _max_dd_limit:
                msg = (
                    f"🚨🚨 [{self.ticker}] 절대 MDD 한도 초과!\n"
                    f"  최고NAV {self._peak_nav:,.0f}원 → 현재 {_nav_current:,.0f}원\n"
                    f"  낙폭 {_total_dd*100:.2f}% (한도 {_max_dd_limit*100:.1f}%)\n"
                    f"  → 봇 완전 정지. 수동 확인 후 재시작 필요."
                )
                logger.critical(msg)
                notify(msg)
                self.is_running = False
                return False

        return True

    def _inject_btc_features(self, df: pd.DataFrame, btc_cols: list) -> pd.DataFrame:
        """BTC OHLCV 기반 컨텍스트 피처를 df에 컬럼으로 추가."""
        try:
            btc_df = self._safe_api_get(
                pyupbit.get_ohlcv, "KRW-BTC", interval=self.interval, count=len(df) + 60
            )
            if btc_df is None or len(btc_df) < 30:
                for col in btc_cols:
                    df[col] = 0.0
                return df

            c = btc_df['close']
            h = btc_df['high']
            lo = btc_df['low']
            o = btc_df['open']
            v = btc_df['volume']

            feat: dict = {}

            # RSI
            def _rsi(series, period):
                delta = series.diff()
                gain = delta.clip(lower=0).rolling(period).mean()
                loss = (-delta.clip(upper=0)).rolling(period).mean()
                rs = gain / (loss + 1e-10)
                return 100 - (100 / (1 + rs))

            feat['BTC_RSI']       = _rsi(c, 14)
            feat['BTC_RSI_Short'] = _rsi(c, 7)

            # Bollinger Bands
            sma20 = c.rolling(20).mean()
            std20 = c.rolling(20).std()
            bb_upper = sma20 + 2 * std20
            bb_lower = sma20 - 2 * std20
            feat['BTC_BB_Width']    = (bb_upper - bb_lower) / (sma20 + 1e-10)
            feat['BTC_BB_Position'] = (c - bb_lower) / (bb_upper - bb_lower + 1e-10)

            # ATR
            tr = pd.concat([h - lo, (h - c.shift()).abs(), (lo - c.shift()).abs()], axis=1).max(axis=1)
            atr14 = tr.rolling(14).mean()
            feat['BTC_ATR_Ratio'] = atr14 / (c + 1e-10)

            # Volume surge
            feat['BTC_Volume_Surge'] = v / (v.rolling(20).mean() + 1e-10)

            # MACD histogram ratio
            ema12 = c.ewm(span=12, adjust=False).mean()
            ema26 = c.ewm(span=26, adjust=False).mean()
            macd  = ema12 - ema26
            signal = macd.ewm(span=9, adjust=False).mean()
            feat['BTC_MACD_Hist_Ratio'] = (macd - signal) / (c + 1e-10)

            # OFI cumulative delta (OHLCV 근사)
            delta_raw = (c - o) / (h - lo + 1e-10) * v
            feat['BTC_OFI_CumDelta_5']  = delta_raw.rolling(5).sum()
            feat['BTC_OFI_CumDelta_10'] = delta_raw.rolling(10).sum()

            # Macro trend (SMA50)
            feat['BTC_Macro_Trend_Up'] = (c > c.rolling(50).mean()).astype(float)

            # HMM 상태 (봇의 HMM이 BTC 국면 프록시로 사용)
            if self.hmm is not None:
                feat['BTC_HMM_Bear'] = float(self.hmm.regime == self.hmm.BEAR)
                feat['BTC_HMM_Bull'] = float(self.hmm.regime == self.hmm.BULL)
            else:
                feat['BTC_HMM_Bear'] = 0.0
                feat['BTC_HMM_Bull'] = 1.0

            btc_feat_df = pd.DataFrame(feat, index=btc_df.index)

            # 타임스탬프 기준 join (coin df 인덱스 기준 정렬)
            df = df.join(btc_feat_df[[c for c in btc_cols if c in btc_feat_df.columns]],
                         how='left')
            df[btc_cols] = df[btc_cols].ffill().fillna(0.0)

        except Exception as e:
            logger.warning(f"⚠️ BTC 컨텍스트 피처 주입 실패 ({e}) — 0으로 대체")
            for col in btc_cols:
                if col not in df.columns:
                    df[col] = 0.0
        return df

    def _check_btc_macro_trend(self) -> bool:
        """
        BTC 일봉 50일 SMA 기반 거시 장세 확인.
        True: 현재가 > SMA50 (상승장) → 매수 허용
        False: 현재가 < SMA50 (하락장) → 매수 전면 차단
        결과는 1시간 캐시하여 API 호출 최소화.
        """
        now = time.time()
        if now - self._btc_trend_cache['last_check'] < BTC_TREND_CACHE_TTL:
            return self._btc_trend_cache['is_uptrend']

        try:
            btc_daily = self._safe_api_get(pyupbit.get_ohlcv, "KRW-BTC", interval="day", count=55)
            if btc_daily is None or len(btc_daily) < 51:
                logger.warning("⚠️ BTC 일봉 데이터 부족 — 장세 필터 패스(허용)")
                return True

            sma50 = btc_daily['close'].rolling(window=50).mean().iloc[-1]
            current = btc_daily['close'].iloc[-1]
            is_uptrend = bool(current > sma50)

            self._btc_trend_cache = {'is_uptrend': is_uptrend, 'last_check': now}
            if not is_uptrend:
                logger.info(f"📉 BTC 거시 하락장: 현재가 {current:,.0f} < SMA50 {sma50:,.0f}")
            return is_uptrend
        except Exception as e:
            logger.warning(f"⚠️ BTC 장세 확인 실패: {e} — 필터 패스(허용)")
            return True  # 에러 시 거래 차단보다 허용이 더 안전 (봇 멈춤 방지)

    def _handle_waiting_position(self, current_price: float, current_time: float, candle_just_closed: bool = False) -> None:
        # 거래소 시간 기반 캔들 마감 트리거 우선, 폴백으로 시간 간격 체크
        if not candle_just_closed and (current_time - self.last_ai_check_time <= self.ai_check_interval):
            return
        self.last_ai_check_time = current_time

        # ── 0. 리스크 게이트 (SIGNAL_PRIORITY 1순위) ────────────────────────────
        _, kill_active = self.news_analyzer.fetch()
        if kill_active:
            logger.warning("🚨 킬스위치 ON (BTC 1h 급락) — 모든 신규 진입 차단")
            return
        if self.upbit_risk is not None and not self.upbit_risk.is_safe_to_trade(self.ticker):
            logger.warning(f"⛔ [UpbitRisk] {self.ticker} 지갑 락업 또는 DAXA 투자유의 → 진입 차단")
            return

        # ── 1. 서킷 브레이커 (SIGNAL_PRIORITY 2순위) ─────────────────────────
        if not self._check_circuit_breaker(current_time):
            return

        # ── 1-1. 타임스탑 쿨다운 ─────────────────────────────────────────────
        if current_time < self._timestop_cooldown_until:
            remaining = int((self._timestop_cooldown_until - current_time) / 60)
            logger.debug(f"⏸️ 타임스탑 쿨다운 중 — 잔여 {remaining}분. 재진입 보류")
            return

        # ── 2. OOS 화이트리스트 필터 ─────────────────────────────────────────
        _wl = _load_whitelist()
        if _wl:
            _coin = self.ticker.split('-')[1] if '-' in self.ticker else self.ticker
            _wl_key = f"{_coin}_{self.interval}"  # whitelist 형식: "ONDO_minute60"
            if _wl_key not in _wl:
                logger.info(f"[화이트리스트] {_coin} OOS 기준 미달 ({_wl_key}) — 진입 차단")
                return

        # ── 3. 시장 과열 필터 (SIGNAL_PRIORITY 3순위) ───────────────────────────
        if self.binance_ctx is not None and self.binance_ctx.is_long_overheated:
            fr, ls = self.binance_ctx.fetch()
            logger.warning(
                f"⚠️ 바이낸스 롱 과열 (펀딩비={fr*100:.3f}%, L/S={ls:.2f}) — 진입 보류"
            )
            return
        if self.fear_greed is not None and self.binance_ctx is not None:
            _fgs = self.fear_greed.get_regime_signal()
            if _fgs["is_overheating"] and not self.binance_ctx.check_funding_rate_risk():
                logger.warning(
                    f"⚠️ 극단적 탐욕 (F&G={_fgs['value']}) + 펀딩비 과열 → 진입 보류"
                )
                return

        # ── 4. HMM 국면 필터 (SIGNAL_PRIORITY 4순위) ────────────────────────
        if self.hmm is not None:
            _regime_snap = self.hmm.regime  # 스레드 안전: 이후 모든 HMM 판단에 동일 값 사용
            if _regime_snap == self.hmm.BEAR:
                _now_hmm = time.time()
                if _now_hmm - self._last_hmm_block_log > 300:
                    logger.warning("🚫 HMM bear 국면 — 신규 Long 진입 차단")
                    self._last_hmm_block_log = _now_hmm
                return
            effective_threshold = self.hmm.get_entry_threshold_for(self.threshold, _regime_snap)
            position_multiplier = self.hmm.get_position_multiplier_for(_regime_snap)
            if _regime_snap == self.hmm.SIDEWAYS:
                _now_hmm = time.time()
                if _now_hmm - self._last_hmm_block_log > 300:
                    logger.info(
                        f"⚠️ HMM sideways 국면 — 임계값 {effective_threshold*100:.1f}%"
                        f" | 비중 ×{position_multiplier:.1f} 로 완화 진입"
                    )
                    self._last_hmm_block_log = _now_hmm
        else:
            _regime_snap = self._current_regime
            effective_threshold = self.threshold
            position_multiplier = 1.0

        # ── 4.1. 국면 정밀도 오버라이드 게이트 ──────────────────────────────────
        # skip=True 모델이 MTF 없이 동작 중일 때 — 현재 국면의 regime_precision만 허용
        if self._force_mtf and not self._mtf_enabled:
            _r_prec = self._regime_precision.get(_regime_snap, 0.0)
            if _r_prec <= 0:
                logger.debug(
                    f"🔒 [{self.ticker}] 국면({_regime_snap}) 정밀도 미달 — 진입 보류"
                )
                return
            logger.info(
                f"📐 [{self.ticker}] 국면 오버라이드 활성: {_regime_snap} prec={_r_prec:.3f} — MTF 생략"
            )

        # ── 4.5. 유동성 흐름 게이트 (SIGNAL_PRIORITY 4.5순위) ───────────────────
        # 매크로 유동성 2신호 + 코인 볼륨 1신호 → min_signals 미만 시 진입 차단
        if self.liq_flow is not None:
            _vol_surge = False
            try:
                _ohlcv_d = self._safe_api_get(
                    pyupbit.get_ohlcv, self.ticker, interval="day", count=8
                )
                if _ohlcv_d is not None and len(_ohlcv_d) >= 7:
                    _avg_vol  = float(_ohlcv_d["value"].iloc[:-1].mean())
                    _today_vol = float(_ohlcv_d["value"].iloc[-1])
                    _surge_ratio = self.liq_flow.get_volume_surge_ratio()
                    _vol_surge = _today_vol > _avg_vol * _surge_ratio
            except Exception:
                _vol_surge = True  # 조회 실패 → 차단 없이 허용

            _lf_allowed, _, _lf_reason = self.liq_flow.check_gate(_vol_surge)
            if not _lf_allowed:
                logger.info(
                    f"💧 [LiqFlow] 유동성 게이트 차단 {_lf_reason} — 진입 보류"
                )
                return
            logger.debug(f"💧 [LiqFlow] 게이트 통과 {_lf_reason}")

        # ── 5. K-Means 횡보장 필터 (SIGNAL_PRIORITY 5순위) ───────────────────
        # 하드차단 대신 임계값 상향 — AI가 충분히 확신하면 진입 허용
        if self.kmeans_regime is not None:
            _ohlcv_km_raw = self._safe_api_get(pyupbit.get_ohlcv, self.ticker,
                                               interval=self.interval, count=220)
            _ohlcv_km = _ohlcv_km_raw if (_ohlcv_km_raw is not None and not _ohlcv_km_raw.empty) else pd.DataFrame()
            kmeans_result = self.kmeans_regime.fit_predict(_ohlcv_km)
            if kmeans_result == self.kmeans_regime.CHOP:
                effective_threshold = min(0.90, effective_threshold + 0.05)
                logger.info(
                    f"[KMeans] 고변동성 횡보장 ({kmeans_result}) "
                    f"→ 임계값 {effective_threshold*100:.1f}%로 상향"
                )

        # ── 6. AI 예측 (SIGNAL_PRIORITY 6순위) ──────────────────────────────
        ai_prob = self._get_ai_prediction(is_entry=True)
        if ai_prob == -1.0:
            logger.debug(f"[{self.ticker}] AI 예측 불가 (모델 오류 / MTF 차단 / 중복 캔들) — 스킵")
            return

        # ── 8. 김치 프리미엄 가속도 부스터 ──────────────────────────────────
        kimchi_boost = 0.0
        if self.kimchi_tracker is not None:
            btc_krw = self._get_btc_price_cached()
            btc_usdt = self.binance_ctx.get_btc_usdt() if self.binance_ctx else 0.0
            usdt_krw = self._get_usdt_krw_cached()
            if btc_usdt <= 0:
                logger.debug("⚠️ 김치 프리미엄 갱신 불가 (바이낸스 BTC/USDT 조회 실패) — 부스터 비활성")
            if btc_krw > 0 and btc_usdt > 0 and usdt_krw > 0:
                _, accel = self.kimchi_tracker.update(btc_krw, btc_usdt, usdt_krw)
                cfg_kp = bot_config.KIMCHI_PREMIUM
                if accel > cfg_kp["surge_threshold"]:
                    kimchi_boost = cfg_kp["threshold_boost"]
                    logger.info(
                        f"🌶️ 김치 프리미엄 가속 ({accel*100:.2f}%) → "
                        f"임계값 -{kimchi_boost*100:.0f}%p 완화"
                    )
        _threshold_floor = self.threshold - 0.04  # 캘리브레이션 기준 최대 4pp 완화
        effective_threshold = max(_threshold_floor, effective_threshold - kimchi_boost)

        # ── 9. 유동성 로테이션: 현재 티커가 급등 후보이면 임계값 완화 ────────────
        if self.liq_scanner is not None:
            candidates = self.liq_scanner.get_rotation_candidates()
            if candidates:
                logger.info(f"🔄 [유동성 로테이션 후보] {', '.join(candidates)}")
                if self.ticker in candidates:
                    _liq_boost = bot_config.LIQUIDITY_SCANNER.get("threshold_boost", 0.02)
                    effective_threshold = max(_threshold_floor, effective_threshold - _liq_boost)
                    logger.info(
                        f"💧 [{self.ticker}] 유동성 급등 감지 → "
                        f"임계값 {effective_threshold*100:.1f}%로 완화 (-{_liq_boost*100:.0f}%p)"
                    )

        logger.info(
            f"👀 [정각 AI 타점 분석] 📈 상승 확률 {ai_prob*100:.1f}% "
            f"(진입 컷 {effective_threshold*100:.1f}% | 국면: {_regime_snap} "
            f"| 비중배수: {position_multiplier:.1f})"
        )

        if ai_prob >= effective_threshold:
            # 극저 변동성 필터: ATR 기대 수익 < 0.6%이면 수수료+슬리피지 후 실질 순손실
            if self.last_atr_ratio > 0:
                expected_profit = self.last_atr_ratio * ATR_TP_MULTIPLIER
                if expected_profit < MIN_PROFIT_THRESHOLD:
                    logger.info(
                        f"🛑 기대 수익률 미달 ({expected_profit*100:.2f}% < {MIN_PROFIT_THRESHOLD*100:.1f}%) "
                        f"— 횡보장 수수료 누수 방지, 매수 스킵"
                    )
                    return
            self.execute_buy(current_price, ai_prob, position_multiplier=position_multiplier)

if __name__ == "__main__":
    if is_dry_run():
        logger.warning("⚠️  DRY_RUN 모드 활성화 — 실제 주문이 실행되지 않습니다. 실전 투자 시 DRY_RUN = False로 변경하세요.")

    if not os.path.exists(_WHITELIST_FILE):
        logger.warning("⚠️ coin_whitelist.json 없음 — 백그라운드에서 OOS 스트레스 백테스트 실행 중 (전체 허용 상태로 시작)")
        def _build_whitelist_bg():
            try:
                from auto_retrain import _update_coin_whitelist
                _update_coin_whitelist()
                logger.info("✅ coin_whitelist.json 생성 완료")
            except Exception as _e:
                logger.error(f"❌ 화이트리스트 생성 실패: {_e}")
        threading.Thread(target=_build_whitelist_bg, daemon=True, name="WhitelistInit").start()

    bot = AITradingBot(ticker="KRW-BTC")
    bot.run()