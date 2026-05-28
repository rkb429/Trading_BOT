"""
orchestrator.py : Multi-Agent Fleet Manager

유동성 스캐너 신호에 따라 종목별 AITradingBot 스레드를 동적으로 생성/종료합니다.
- 5분마다 업비트 전 종목 거래대금 순위를 재평가
- 상위 N개 코인에만 봇을 배치하고 순위 밀린 봇은 Graceful Shutdown
- 각 봇에 1/N 자본 비중 할당 → 이중 매수 방지
"""
import time
import random
import logging
import threading
import signal
import concurrent.futures
import json
import os
os.environ.setdefault('LOKY_MAX_CPU_COUNT', str(os.cpu_count() or 4))
# Windows 11: wmic deprecated → loky _count_physical_cores subprocess 에러 방지
try:
    import joblib.externals.loky.backend.context as _loky_ctx
    _loky_ctx._count_physical_cores = lambda: (os.cpu_count() or 4, None)
except Exception:
    pass
from typing import Optional
import pandas as pd
import pyupbit
import config as bot_config
from market_context import LiquidityScanner, LiquidityFlowMonitor, BinanceFuturesContext, FearGreedIndex, UpbitRiskManager
from trade_bot import AITradingBot, notifier, _load_whitelist

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger("Orchestrator")

_shutdown_event = threading.Event()


def _handle_signal(_signum, _frame):
    logger.info("⌨️ 종료 시그널 수신 — 전 봇 우아한 종료 시작")
    _shutdown_event.set()


# ─── 포트폴리오 리스크 헬퍼 ──────────────────────────────────────────────────

def _fetch_series(coin, lookback):
    for _ in range(2):
        if _shutdown_event.is_set():
            return coin, None
        df = pyupbit.get_ohlcv(coin, interval="minute15", count=lookback + 1)
        if df is not None and not df.empty:
            return coin, df['close'].pct_change().dropna()
        _shutdown_event.wait(timeout=0.5)
    return coin, None


def _fetch_atr(coin: str, lookback: int) -> tuple:
    """coin의 ATR/price 변동성 비율 반환. 실패 시 기본값 0.03."""
    for _ in range(2):
        if _shutdown_event.is_set():
            return coin, 0.03
        try:
            df = pyupbit.get_ohlcv(coin, interval="minute15", count=lookback + 1)
            if df is not None and not df.empty and len(df) > 14:
                high_low  = df['high'] - df['low']
                high_prev = (df['high'] - df['close'].shift(1)).abs()
                low_prev  = (df['low']  - df['close'].shift(1)).abs()
                tr = pd.concat([high_low, high_prev, low_prev], axis=1).max(axis=1)
                atr = tr.rolling(14).mean().iloc[-1]
                last_close = df['close'].iloc[-1]
                return coin, max(atr / last_close if last_close > 0 else 0.03, 1e-6)
        except Exception:
            pass
        _shutdown_event.wait(timeout=0.3)
    return coin, 0.03


def _compute_risk_parity_weights(coins: list, lookback: int = 96) -> dict:
    """
    ATR 역수 기반 리스크 파리티 자본 배분 비중.
    변동성(ATR/가격)이 높은 코인에 적은 자본을 배분하여 종목별 실질 리스크를 균등화.
    ThreadPoolExecutor로 병렬 페칭 — 메인 스레드 블로킹 5-10초 → ~2초로 단축.
    API 실패 시 균등 배분으로 폴백.
    """
    if not coins:
        return {}

    atrs: dict = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(5, len(coins))) as executor:
        futs = [executor.submit(_fetch_atr, coin, lookback) for coin in coins]
        for fut in concurrent.futures.as_completed(futs, timeout=15):
            try:
                coin, atr_val = fut.result()
                atrs[coin] = atr_val
            except Exception:
                pass

    for coin in coins:
        if coin not in atrs:
            atrs[coin] = 0.03

    inv_atrs = {coin: 1.0 / atr for coin, atr in atrs.items()}
    total = sum(inv_atrs.values())
    if total <= 0:
        n = len(coins)
        return {c: 1.0 / n for c in coins}
    return {coin: w / total for coin, w in inv_atrs.items()}


def _compute_avg_correlation(coins: list, lookback: int) -> float:
    if len(coins) < 2:
        return 0.0
    series = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(5, len(coins))) as executor:
        futures = [executor.submit(_fetch_series, coin, lookback) for coin in coins]
        try:
            for future in concurrent.futures.as_completed(futures, timeout=10):
                try:
                    coin, data = future.result()
                    if data is not None:
                        series[coin] = data
                except Exception:
                    continue
        except concurrent.futures.TimeoutError:
            logger.warning("⚠️ 상관계수 데이터 수집 타임아웃 — 부분 결과로 계산")

    if len(series) < 2:
        return 0.0
    ret_df = pd.DataFrame(series).dropna()
    if ret_df.empty or len(ret_df.columns) < 2:
        return 0.0
        
    corr = ret_df.corr().values
    n = len(corr)
    pairs = [corr[i][j] for i in range(n) for j in range(i + 1, n) if not pd.isna(corr[i][j])]
    return float(sum(pairs) / len(pairs)) if pairs else 0.0


def _is_market_overheated(binance_ctx: BinanceFuturesContext,
                           fear_greed: FearGreedIndex,
                           fg_threshold: int) -> bool:
    """바이낸스 펀딩비 과열 AND F&G 극도탐욕 동시 충족 시 True."""
    if binance_ctx is None or fear_greed is None:
        return False
    try:
        fr, _ = binance_ctx.fetch()
        fg_val, _ = fear_greed.fetch()
        if fr > 0.001 and fg_val >= fg_threshold:
            logger.warning(
                f"🔥 시장 과열: 펀딩비={fr*100:.3f}% | F&G={fg_val} ≥ {fg_threshold}"
                f" → 신규 봇 파견 차단, 현금 보유"
            )
            return True
        return False
    except Exception:
        return False


# ─── Fleet 상태 영속화 ────────────────────────────────────────────────────────

FLEET_STATE_FILE    = "active_fleet_state.json"
CORR_CACHE_TTL      = 3600   # 상관계수 재계산 주기: 1시간
BOT_HEARTBEAT_STALE = 120    # 봇 상태파일 heartbeat가 이 초 이상 갱신 없으면 이상 감지


def _save_fleet_state(active_fleet: dict):
    """재시작 후 이중 매수 방지를 위해 운용 중인 봇 목록을 파일로 저장 (원자적 쓰기)."""
    state = {coin: info['capital_fraction'] for coin, info in active_fleet.items()}
    tmp = FLEET_STATE_FILE + ".tmp"
    try:
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(state, f)
        os.replace(tmp, FLEET_STATE_FILE)
    except Exception as e:
        logger.warning(f"⚠️ fleet 상태 저장 실패: {e}")
        try:
            os.remove(tmp)
        except OSError:
            pass


def _load_fleet_state() -> dict:
    """이전 실행에서 저장된 fleet 상태를 반환. 파일 없으면 빈 dict."""
    if not os.path.exists(FLEET_STATE_FILE):
        return {}
    try:
        with open(FLEET_STATE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"⚠️ fleet 상태 로드 실패 (무시하고 새로 시작): {e}")
        return {}


# ─── Fleet Manager ────────────────────────────────────────────────────────────

def manage_fleet():
    try:
        bot_config.validate_config()
    except ValueError as e:
        logger.critical(f"설정 오류로 오케스트레이터 시작 중단:\n{e}")
        return

    cfg = bot_config.ORCHESTRATOR
    max_bots        = cfg["max_concurrent_bots"]
    rescan_interval = cfg["rescan_interval"]
    corr_lookback   = cfg["corr_lookback_bars"]
    corr_high       = cfg["corr_high_threshold"]
    corr_mid        = cfg["corr_medium_threshold"]
    fg_threshold    = cfg["overheat_fg_threshold"]

    signal.signal(signal.SIGINT, _handle_signal)
    try:
        signal.signal(signal.SIGTERM, _handle_signal)
    except (OSError, ValueError):
        pass

    # 시장 과열 감지용 컨텍스트 (봇과 독립적으로 오케스트레이터가 직접 조회)
    _bc = bot_config.BINANCE_CONTEXT
    binance_ctx = BinanceFuturesContext(
        cache_ttl=_bc["cache_ttl"],
        funding_threshold=_bc["funding_rate_threshold"],
        ls_overbought=_bc["ls_ratio_overbought"],
        funding_oversold=_bc["funding_rate_oversold"],
    ) if _bc["enabled"] else None

    _fg = bot_config.FEAR_GREED
    fear_greed = FearGreedIndex(
        cache_ttl=_fg["cache_ttl"],
        extreme_greed=_fg["extreme_greed"],
        greed=_fg["greed"],
    ) if _fg["enabled"] else None

    _ur = getattr(bot_config, "UPBIT_RISK", {})
    upbit_risk: Optional[UpbitRiskManager] = None
    if _ur.get("enabled", True):
        _access = os.getenv("UPBIT_ACCESS_KEY", "")
        _secret = os.getenv("UPBIT_SECRET_KEY", "")
        if _access and _secret:
            upbit_risk = UpbitRiskManager(
                access_key=_access,
                secret_key=_secret,
                cache_ttl=_ur.get("cache_ttl", 120),
            )
        else:
            logger.warning("⚠️ UPBIT_ACCESS_KEY/UPBIT_SECRET_KEY 미설정 — UpbitRiskManager 비활성화")

    logger.info(
        f"🚀 Multi-Agent Fleet 오케스트레이터 가동\n"
        f"   최대 봇: {max_bots}개 | 재평가 주기: {rescan_interval}초\n"
        f"   상관계수 기준: high={corr_high} mid={corr_mid} | F&G 과열 임계: {fg_threshold}"
    )

    # ── 화이트리스트 초기화: 파일 없으면 백그라운드 생성 ──────────────────────
    _wl_file = os.path.join(bot_config.DIRECTORIES.get("data", "data"), "coin_whitelist.json")
    if not os.path.exists(_wl_file):
        logger.warning(
            "⚠️ coin_whitelist.json 없음 — OOS 스트레스 백테스트로 백그라운드 생성 시작\n"
            "   생성 완료 전까지는 화이트리스트 필터 미적용 (전체 허용)"
        )
        def _build_whitelist_bg():
            try:
                from auto_retrain import _update_coin_whitelist
                _update_coin_whitelist()
                logger.info("✅ coin_whitelist.json 백그라운드 생성 완료")
            except Exception as _e:
                logger.error(f"❌ 화이트리스트 백그라운드 생성 실패: {_e}")
        threading.Thread(target=_build_whitelist_bg, daemon=True, name="WhitelistInit").start()

    scanner = LiquidityScanner(
        check_interval=rescan_interval,
        top_n=max(10, max_bots * 3),
        surge_rank_jump=5,
    )
    liq_flow_monitor = LiquidityFlowMonitor(bot_config.LIQUIDITY_FLOW)

    # ── 선물 헤지 봇 (Bear 국면 BTC 선물 숏 — daemon 스레드) ─────────────────
    _fh_cfg = getattr(bot_config, "FUTURES_HEDGE", {})
    _futures_thread: Optional[threading.Thread] = None

    def _spawn_futures_hedge() -> threading.Thread:
        from futures_bot import AsyncFuturesBot
        from trade_bot import is_dry_run
        _bot = AsyncFuturesBot(
            symbol=_fh_cfg.get("symbol", "BTC/USDT"),
            leverage=_fh_cfg.get("leverage", 2),
            dry_run=is_dry_run(),
            paper_balance_usdt=_fh_cfg.get("paper_balance_usdt", 1000.0),
        )
        _t = threading.Thread(target=_bot.run_sync, daemon=True, name="FuturesHedge")
        _t.start()
        mode = "PAPER" if is_dry_run() else "LIVE"
        logger.info(
            f"🔷 [FuturesHedge] 선물 헤지 봇 가동 "
            f"({_fh_cfg.get('symbol')} ×{_fh_cfg.get('leverage')}x | {mode})"
        )
        return _t

    if _fh_cfg.get("enabled", False):
        try:
            _futures_thread = _spawn_futures_hedge()
        except Exception as _fh_err:
            logger.error(f"❌ [FuturesHedge] 선물 헤지 봇 기동 실패: {_fh_err}")

    active_fleet: dict = {}
    retiring_threads: dict = {}  # {coin: (thread, enqueue_time)} — 5분 후 강제 해제
    _deploy_fail_cache: dict = {}  # {coin: timestamp} — 생성 실패 코인 쿨다운
    _DEPLOY_FAIL_COOLDOWN = 1800.0  # 30분 후 재시도
    _TARGET_UTILIZATION = 0.95     # 배포 봇 자본비중 합산 목표 (95%)
    _RETIRING_TIMEOUT = 300.0
    _corr_cache: dict = {"value": 0.0, "ts": 0.0}  # 상관계수 1시간 TTL 캐시
    _price_snapshot: dict = {}  # {coin: price} 가격 충격 감지용 직전 가격

    # ── 재시작 복원: 이전 실행에서 포지션을 열고 있던 봇 재구성 ─────────────
    prev_state = _load_fleet_state()
    if prev_state:
        logger.warning(
            f"⚠️ 이전 fleet 상태 감지 — {list(prev_state.keys())} 봇 복원 중 "
            f"(이중 매수 방지)"
        )
        current_top = scanner.get_top_coins(n=max(max_bots * 2, 10))
        for coin, fraction in prev_state.items():
            if current_top and coin not in current_top:
                logger.warning(
                    f"⚠️ [{coin}] 복원 대상이 현재 유동성 상위권 밖 — "
                    f"재파견 보류 (유동성 이탈 또는 스캐너 초기화 중)"
                )
                continue
            try:
                bot = AITradingBot(
                    ticker=coin,
                    capital_fraction=fraction,
                    setup_signals=False,
                    liq_scanner=scanner,
                    liq_flow=liq_flow_monitor,
                )
                t = threading.Thread(target=bot.run, daemon=True, name=f"Bot-{coin}")
                t.start()
                active_fleet[coin] = {'bot': bot, 'thread': t, 'capital_fraction': fraction}
                logger.info(f"✅ [{coin}] 봇 복원 완료 (자본 비중 {fraction*100:.0f}%)")
            except Exception as e:
                logger.error(f"❌ [{coin}] 봇 복원 실패: {e}")
        if active_fleet:
            _save_fleet_state(active_fleet)  # 복원 실패 코인 제거된 최신 상태 저장

    logger.info("⏳ 유동성 데이터 초기 수집 대기 (15초)...")
    _shutdown_event.wait(timeout=15)

    try:
        while not _shutdown_event.is_set():
            # 모델 실패 시 다음 순위 코인으로 폴백할 수 있도록 3× 후보 확보
            candidate_pool = scanner.get_top_coins(n=max(max_bots * 3, 10))
            # OOS 화이트리스트 필터: minute60 모델 기준 통과 코인만 배포 후보로 유지
            # BTC/ETH는 레퍼런스 피처 및 Model_A로 항상 허용
            _ALWAYS_ALLOW = {"BTC", "ETH"}
            _wl_set = _load_whitelist()
            if _wl_set:
                _primary_ivl = bot_config.MODEL_MANAGEMENT.get("train_interval", "minute60")
                candidate_pool = [
                    c for c in candidate_pool
                    if (c.split('-')[-1] in _ALWAYS_ALLOW)
                    or (f"{c.split('-')[-1]}_{_primary_ivl}" in _wl_set)
                ]
            target_coins = candidate_pool[:max_bots]  # 상관계수·과열 분석용

            if not target_coins:
                logger.warning("유동성 스캐너 데이터 없음 — 60초 후 재시도")
                for _ in range(60):
                    if _shutdown_event.is_set():
                        break
                    _shutdown_event.wait(timeout=1.0)
                continue

            if _shutdown_event.is_set():
                break

            # ── 포트폴리오 리스크 게이팅 ────────────────────────────────────────
            # 1단계: 시장 과열 체크 (F&G + 펀딩비) → 과열이면 봇 1개로 강제 제한
            if _is_market_overheated(binance_ctx, fear_greed, fg_threshold):
                effective_max = 1
            else:
                # 2단계: 가격 충격 감지 → ≥3% 급변 시 상관계수 캐시 즉시 무효화
                for _coin, _info in active_fleet.items():
                    _recent = getattr(_info['bot'], 'current_price', 0)
                    _prev = _price_snapshot.get(_coin, _recent)
                    if _prev > 0 and abs(_recent - _prev) / _prev >= 0.03:
                        logger.warning(
                            f"💥 [{_coin}] 가격 충격 감지 "
                            f"({_prev:,.0f} → {_recent:,.0f}, "
                            f"{(_recent - _prev) / _prev * 100:+.1f}%) "
                            f"→ 상관계수 캐시 즉시 무효화"
                        )
                        _corr_cache["ts"] = 0.0
                        break
                for _coin, _info in active_fleet.items():
                    _p = getattr(_info['bot'], 'current_price', 0)
                    if _p > 0:
                        _price_snapshot[_coin] = _p

                # 3단계: 상관계수 체크 — 1시간 TTL 캐시로 rescan마다 API 호출 방지
                _now = time.time()
                if _now - _corr_cache["ts"] >= CORR_CACHE_TTL:
                    try:
                        avg_corr = _compute_avg_correlation(target_coins, corr_lookback)
                        _corr_cache["value"] = avg_corr
                        _corr_cache["ts"] = _now
                        logger.info(f"📐 상관계수 갱신: r={avg_corr:.2f}")
                    except Exception as _corr_err:
                        avg_corr = _corr_cache["value"]
                        logger.warning(
                            f"⚠️ 상관계수 계산 실패({_corr_err}) — 캐시값 사용 (r={avg_corr:.2f})"
                        )
                else:
                    avg_corr = _corr_cache["value"]
                    remaining = int((CORR_CACHE_TTL - (_now - _corr_cache["ts"])) / 60)
                    logger.debug(f"📐 상관계수 캐시 사용 (r={avg_corr:.2f}, 갱신까지 {remaining}분)")
                if avg_corr >= corr_high:
                    effective_max = 1
                    logger.warning(
                        f"⚠️ 고상관 시장 (r={avg_corr:.2f} ≥ {corr_high})"
                        f" → 봇 1개로 제한 (분산 효과 없음)"
                    )
                elif avg_corr >= corr_mid:
                    effective_max = min(2, max_bots)
                    logger.info(f"📊 중간 상관 (r={avg_corr:.2f}) → 봇 최대 2개")
                else:
                    effective_max = max_bots
                    logger.info(f"✅ 저상관 (r={avg_corr:.2f}) → 봇 {max_bots}개 전체 운용")

            # 4단계: 목표 종목 확정 + ATR 역수 리스크파리티 자본 배분
            if _shutdown_event.is_set():
                break
            # deploy_pool: 모델 실패 시 다음 순위 코인으로 순차 시도할 전체 후보
            deploy_pool = candidate_pool
            try:
                rp_weights = _compute_risk_parity_weights(deploy_pool, lookback=corr_lookback)
            except Exception as _rp_err:
                logger.warning(f"⚠️ 리스크파리티 계산 실패({_rp_err}) → 균등 배분 폴백")
                n_dp = len(deploy_pool) or 1
                rp_weights = {c: 1.0 / n_dp for c in deploy_pool}

            # 배포 예상 코인 기반 가중치 정규화 — 실패 코인 제외 후 합산이 TARGET_UTILIZATION에 수렴
            _now_nu = time.time()
            _expected = [
                c for c in deploy_pool
                if c in active_fleet or _now_nu - _deploy_fail_cache.get(c, 0) > _DEPLOY_FAIL_COOLDOWN
            ][:effective_max]
            _raw_sum = sum(rp_weights.get(c, 0) for c in _expected)
            if _raw_sum > 0:
                _scale = _TARGET_UTILIZATION / _raw_sum
                rp_weights = {c: min(w * _scale, 1.0) for c, w in rp_weights.items()}

            wanted = target_coins[:effective_max]  # 이상적 top 코인 (로그용)
            weight_log = " | ".join(
                f"{c.split('-')[1]}:{rp_weights.get(c, 0)*100:.0f}%" for c in wanted
            )
            logger.info(
                f"📊 [Fleet 재평가] 목표: {wanted} | "
                f"봇 {effective_max}개 | 리스크파리티: [{weight_log}] | "
                f"현재 운용: {list(active_fleet.keys())}"
            )

            # ── 0. 이전 주기 좀비 스레드 정리 ────────────────────────────────
            for coin in list(retiring_threads.keys()):
                t_ret, enqueued = retiring_threads[coin]
                elapsed = time.time() - enqueued
                if not t_ret.is_alive():
                    del retiring_threads[coin]
                    logger.info(f"✅ [{coin}] 좀비 스레드 자연 종료 확인 — 추적 해제")
                elif elapsed >= _RETIRING_TIMEOUT:
                    del retiring_threads[coin]
                    logger.warning(
                        f"⚠️ [{coin}] 좀비 스레드 {elapsed/60:.0f}분 초과 — "
                        f"강제 추적 해제 (스레드는 데몬으로 OS가 최종 회수)"
                    )

            # fleet_changed는 0b 블록 이전에 초기화해야 한다.
            # 이전 코드는 0b에서 True로 설정 후 1블록에서 False로 덮어써
            # 무음 종료 봇 제거 후 상태 파일이 저장되지 않는 버그가 있었다.
            fleet_changed = False

            # ── 0b. 봇 헬스체크: 스레드 생존 + heartbeat 타임스탬프 이중 확인 ──
            for coin in list(active_fleet.keys()):
                info = active_fleet[coin]
                thread_dead = not info['thread'].is_alive()

                # heartbeat 체크: 봇이 _save_local_state()에 기록한 Unix timestamp 확인
                heartbeat_stale = False
                try:
                    bot = info['bot']
                    state_file = getattr(bot, '_state_file', None)
                    if state_file and os.path.exists(state_file):
                        with open(state_file, 'r', encoding='utf-8') as _sf:
                            _st = json.load(_sf)
                        hb = _st.get('heartbeat', 0)
                        if hb > 0 and (time.time() - hb) > BOT_HEARTBEAT_STALE:
                            heartbeat_stale = True
                except Exception:
                    pass

                if thread_dead or heartbeat_stale:
                    reason = "스레드 무음 종료" if thread_dead else f"heartbeat {BOT_HEARTBEAT_STALE}초 이상 갱신 없음"
                    logger.error(
                        f"❌ [{coin}] 봇 이상 감지 ({reason}) "
                        f"— fleet에서 제거 후 다음 주기에 재파견"
                    )
                    del active_fleet[coin]
                    fleet_changed = True
                    _save_fleet_state(active_fleet)  # 즉시 저장 — 이중 매수 방지

            # ── 1. 순위 밀린 봇 Graceful Shutdown (공유 60초 deadline으로 병렬 join) ──
            # wanted(상위 effective_max) 기준 교체.
            # 쿨다운 중인 실패 코인이 있으면 그 슬롯을 채운 fallback 봇을 보호
            # — 보호 없으면 fallback 봇이 매 주기 종료·재파견되는 무한루프 발생
            _now_fc = time.time()
            n_failed_wanted = sum(
                1 for c in wanted
                if _now_fc - _deploy_fail_cache.get(c, 0) <= _DEPLOY_FAIL_COOLDOWN
            )
            non_wanted_running = [c for c in active_fleet if c not in wanted]
            protected_fallbacks = set(non_wanted_running[:n_failed_wanted])
            effective_wanted = set(
                c for c in wanted
                if _now_fc - _deploy_fail_cache.get(c, 0) > _DEPLOY_FAIL_COOLDOWN
            )
            coins_to_stop = [
                c for c in active_fleet
                if c not in effective_wanted and c not in protected_fallbacks
            ]
            for coin in coins_to_stop:
                logger.warning(f"🛑 [{coin}] 목표 제외 → 종료 신호 전송")
                active_fleet[coin]['bot'].stop_gracefully()
            if coins_to_stop:
                _stop_deadline = time.time() + 60
                for coin in coins_to_stop:
                    remaining = max(0.0, _stop_deadline - time.time())
                    active_fleet[coin]['thread'].join(timeout=remaining)
                    if active_fleet[coin]['thread'].is_alive():
                        active_fleet[coin]['bot'].force_kill()
                        logger.error(
                            f"❌ [{coin}] 스레드 60초 내 미종료 — force_kill 호출 후 좀비 추적 등록"
                        )
                        retiring_threads[coin] = (active_fleet[coin]['thread'], time.time())
                    del active_fleet[coin]
                    fleet_changed = True

            # ── 2. 신규 종목 봇 파견 ─────────────────────────────────────────
            # deploy_pool 전체를 순서대로 시도 — 모델 실패 코인은 건너뛰고 다음 코인 파견
            # 이미 운용 중인 봇은 자본비중 갱신만 수행 (슬롯 소모로 카운트)
            new_bot_idx = 0
            for coin in deploy_pool:
                if _shutdown_event.is_set():
                    break
                coin_fraction = rp_weights.get(coin, 1.0 / len(deploy_pool))

                # 좀비 스레드가 아직 살아있으면 이중파견 차단
                if coin in retiring_threads:
                    t_ret, enqueued = retiring_threads[coin]
                    if t_ret.is_alive() and (time.time() - enqueued) < _RETIRING_TIMEOUT:
                        logger.warning(
                            f"⚠️ [{coin}] 이전 봇 스레드 아직 실행 중 — 새 봇 파견 보류 (이중 포지션 방지)"
                        )
                        continue
                    del retiring_threads[coin]

                if coin in active_fleet:
                    # 이미 운용 중인 봇의 자본 비중이 바뀌면 갱신 (슬롯 소모)
                    prev = active_fleet[coin].get('capital_fraction', coin_fraction)
                    if abs(prev - coin_fraction) > 0.01:
                        active_fleet[coin]['bot'].capital_fraction = coin_fraction
                        active_fleet[coin]['capital_fraction'] = coin_fraction
                        logger.info(
                            f"🔄 [{coin}] 자본 비중 갱신: "
                            f"{prev*100:.0f}% → {coin_fraction*100:.0f}% (리스크파리티)"
                        )
                        fleet_changed = True
                elif len(active_fleet) < effective_max:
                    # 슬롯 여유 있을 때만 신규 파견
                    if upbit_risk is not None:
                        _block_on_fail = _ur.get("block_on_api_failure", True)
                        try:
                            if not upbit_risk.is_safe_to_trade(coin):
                                logger.warning(
                                    f"🚫 [{coin}] 업비트 리스크 차단 "
                                    f"(지갑 락업 또는 DAXA 투자유의) — 파견 스킵"
                                )
                                continue
                        except Exception as _re:
                            if _block_on_fail:
                                logger.warning(
                                    f"⚠️ [{coin}] UpbitRiskManager API 실패({_re}) "
                                    f"— block_on_api_failure=True, 파견 스킵"
                                )
                                continue
                            logger.warning(
                                f"⚠️ [{coin}] UpbitRiskManager API 실패({_re}) "
                                f"— block_on_api_failure=False, 파견 진행"
                            )
                    if new_bot_idx > 0:
                        jitter = random.uniform(0.5, 2.0)
                        logger.debug(f"⏱️ [{coin}] 봇 파견 Jitter {jitter:.1f}초 대기...")
                        _shutdown_event.wait(timeout=jitter)
                        if _shutdown_event.is_set():
                            break
                    new_bot_idx += 1
                    logger.info(
                        f"🔥 [{coin}] 신규 투입 → 자본 비중 {coin_fraction*100:.0f}% (리스크파리티)"
                    )
                    try:
                        bot = AITradingBot(
                            ticker=coin,
                            capital_fraction=coin_fraction,
                            setup_signals=False,
                            liq_scanner=scanner,
                            liq_flow=liq_flow_monitor,
                        )
                        t = threading.Thread(
                            target=bot.run, daemon=True, name=f"Bot-{coin}"
                        )
                        t.start()
                        active_fleet[coin] = {'bot': bot, 'thread': t,
                                              'capital_fraction': coin_fraction}
                        _deploy_fail_cache.pop(coin, None)
                        logger.info(f"✅ [{coin}] 봇 가동")
                        fleet_changed = True
                    except Exception as e:
                        logger.error(f"❌ [{coin}] 봇 생성 실패: {e}")
                        _deploy_fail_cache[coin] = time.time()
                        logger.warning(f"⏳ [{coin}] 배포 실패 — {_DEPLOY_FAIL_COOLDOWN/60:.0f}분 쿨다운 후 재시도")

            # fleet 구성이 바뀌었을 때만 상태 파일 갱신
            if fleet_changed:
                _save_fleet_state(active_fleet)

            # ── 선물 헤지 봇 왓치독 ────────────────────────────────────────────
            if _fh_cfg.get("enabled", False) and _futures_thread is not None:
                if not _futures_thread.is_alive():
                    logger.warning("⚠️ [FuturesHedge] 스레드 비정상 종료 감지 — 재기동")
                    try:
                        _futures_thread = _spawn_futures_hedge()
                    except Exception as _fh_err:
                        logger.error(f"❌ [FuturesHedge] 재기동 실패: {_fh_err}")

            # Ctrl+C 즉시 반응: 1초 단위 폴링 (Windows에서 Event.wait 긴 타임아웃은
            # SIGINT 핸들러가 호출돼도 즉시 깨어나지 않을 수 있음)
            _deadline = time.time() + rescan_interval
            while not _shutdown_event.is_set() and time.time() < _deadline:
                _shutdown_event.wait(timeout=1.0)

    finally:
        logger.info("🧹 전 봇 종료 중...")
        bots_info = list(active_fleet.values())
        for info in bots_info:
            info['bot'].stop_gracefully()
        if bots_info:
            _shutdown_deadline = time.time() + 60
            for info in bots_info:
                remaining = max(0.0, _shutdown_deadline - time.time())
                info['thread'].join(timeout=remaining)
        # 정상 종료 시 상태 파일 제거 (재시작 시 불필요한 복원 방지)
        if os.path.exists(FLEET_STATE_FILE):
            os.remove(FLEET_STATE_FILE)
        scanner.shutdown()
        notifier.shutdown()
        logger.info("✅ 함대 완전 종료")


if __name__ == "__main__":
    manage_fleet()
