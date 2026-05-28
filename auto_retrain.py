"""
auto_retrain.py : 주간 자동 재학습 스케줄러
매주 일요일 새벽 3시에 최신 데이터로 모델을 재학습합니다.
trade_bot.py 와 별도 프로세스로 실행 — 봇 실행 중에도 백그라운드 가동 가능.
"""
import os
import glob
import json
import time
import shutil
import logging
import requests
import schedule
import traceback
from datetime import datetime, timedelta
from dotenv import load_dotenv
import config as _cfg

load_dotenv()

TELEGRAM_TOKEN  = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

LOCK_FILE              = "auto_retrain.lock"
LOCK_STALE_SECONDS     = 7200          # 2시간 이상 오래된 락은 고아로 간주
DRIFT_STATE_STALE_SECONDS = 7200       # 봇 상태 파일 heartbeat가 이 초 이상 오래됐으면 드리프트 판단 제외
DRIFT_PAYOFF_THRESHOLD = _cfg.MODEL_MANAGEMENT.get("drift_payoff_threshold", -0.005)
DRIFT_RATIO_TRIGGER    = _cfg.MODEL_MANAGEMENT.get("drift_ratio_trigger",    0.5)
MODEL_FRESHNESS_DAYS   = _cfg.MODEL_MANAGEMENT.get("freshness_days", 7)  # config 단일 소스
_STATE_DIR             = _cfg.DIRECTORIES.get("data", "data")
_MODEL_DIR             = _cfg.DIRECTORIES.get("models", "models")
_FUTURES_MODEL_DIR     = _cfg.DIRECTORIES.get("models_futures", "models_futures")
_FUTURES_ROLLBACK_DIR  = os.path.join(_FUTURES_MODEL_DIR, "rollback_backup")
RETRAIN_HISTORY_FILE         = "retrain_history.json"
_MODEL_UPDATED_FLAG          = os.path.join(_STATE_DIR, "model_updated.flag")
_WHITELIST_FILE              = os.path.join(_STATE_DIR, "coin_whitelist.json")
_WHITELIST_ROI_MIN           = 0.0     # OOS 스트레스 ROI 하한 (%) — 수익이어야 통과
_WHITELIST_MDD_MAX           = 20.0   # 최대 낙폭 상한 (%) — MDD는 양수 저장, 기존 -35 비교는 항상 통과였음
_WHITELIST_SHARPE_MIN        = 0.0    # 최소 Sharpe — 0 미만은 손실 기대값
_WHITELIST_MIN_TRADES        = 10     # 최소 거래 횟수 (통계적 유효성)
_WHITELIST_STRESS_MONTHS     = 6       # 최근 N개월 OOS 구간
_WHITELIST_SLIPPAGE_MULT     = 2.0     # 스트레스 슬리피지 배수 (중간 수준)
_WHITELIST_ALWAYS_ALLOW      = frozenset()  # 카테고리 A 예외 제거 — BTC/ETH도 스트레스 필터 통과 필요
_HISTORY_MAX_ENTRIES         = 104   # 약 1년치 주 2회 재학습 이력
_ROLLBACK_ESCALATION_THRESHOLD = 2   # 연속 롤백이 이 횟수 이상이면 수동 개입 경고
_last_drift_check: float     = 0.0   # 드리프트 체크 타임스탬프 (모듈 레벨 — main에서 갱신)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("AutoRetrain")


def _get_all_model_dirs() -> list:
    """활성 모델 디렉토리 목록 (카테고리 60m + 15m + 선물)"""
    dirs: list = []
    cats = getattr(_cfg, "COIN_CATEGORIES", None)
    if cats:
        for cat_info in cats.values():
            for key in ("model_dir", "model_dir_15m"):
                d = cat_info.get(key)
                if d and d not in dirs:
                    dirs.append(d)
    else:
        if _MODEL_DIR not in dirs:
            dirs.append(_MODEL_DIR)
    if _FUTURES_MODEL_DIR not in dirs:
        dirs.append(_FUTURES_MODEL_DIR)
    return dirs


def _load_latest_model_config() -> dict:
    """모든 카테고리 모델 디렉토리에서 가장 최근 config_*.json을 로드. 없으면 {} 반환."""
    configs = sorted(
        [c for d in _get_all_model_dirs()
         for c in glob.glob(os.path.join(d, "config_*.json"))],
        key=os.path.getmtime
    )
    if not configs:
        return {}
    try:
        with open(configs[-1], 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"모델 설정 로드 실패: {e}")
        return {}


def _backup_dir(src_dir: str, dst_dir: str) -> int:
    """src_dir의 pkl/json/csv를 dst_dir에 원자적으로 복사. 복사된 파일 수 반환.
    tmp 디렉토리에 먼저 복사 완료 후 rename — 복사 실패 시 기존 백업 보존.
    """
    tmp_dir = dst_dir + ".tmp"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)

    count = 0
    for ext in ("*.pkl", "*.json", "*.csv"):
        for src in glob.glob(os.path.join(src_dir, ext)):
            shutil.copy2(src, os.path.join(tmp_dir, os.path.basename(src)))
            count += 1

    if count > 0:
        if os.path.exists(dst_dir):
            shutil.rmtree(dst_dir)
        os.rename(tmp_dir, dst_dir)
    else:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return count


def _backup_current_model() -> bool:
    """재학습 전 모든 카테고리 모델 디렉토리를 각 dir/rollback_backup/ 에 복사."""
    try:
        n = 0
        for model_dir in _get_all_model_dirs():
            n += _backup_dir(model_dir, os.path.join(model_dir, "rollback_backup"))
        logger.info(f"💾 모델 백업 완료 — {n}개 파일")
        return n > 0
    except Exception as e:
        logger.warning(f"⚠️ 모델 백업 실패: {e}")
        return False


def _restore_dir(src_dir: str, dst_dir: str) -> int:
    """src_dir 파일을 dst_dir로 원자적으로 복원. 복원된 파일 수 반환."""
    files = glob.glob(os.path.join(src_dir, "*"))
    if not files:
        return 0
    tmp_dir = dst_dir + ".restore_tmp"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)
    for src in files:
        shutil.copy2(src, os.path.join(tmp_dir, os.path.basename(src)))
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    os.rename(tmp_dir, dst_dir)
    return len(files)


def _restore_model_backup() -> bool:
    """각 카테고리 dir/rollback_backup/ 파일을 해당 디렉토리로 복원."""
    try:
        n = 0
        for model_dir in _get_all_model_dirs():
            rollback_dir = os.path.join(model_dir, "rollback_backup")
            if os.path.exists(rollback_dir):
                n += _restore_dir(rollback_dir, model_dir)
        if n == 0:
            logger.warning("⚠️ 롤백 대상 없음 — 백업 파일이 존재하지 않습니다")
            return False
        logger.warning(f"🔁 모델 롤백 완료 — {n}개 파일 복원")
        return True
    except Exception as e:
        logger.error(f"❌ 모델 롤백 실패: {e}")
        return False


def _append_retrain_history(entry: dict):
    """retrain_history.json 에 재학습 이벤트 추가. 최대 _HISTORY_MAX_ENTRIES 개 유지."""
    history: list = []
    if os.path.exists(RETRAIN_HISTORY_FILE):
        try:
            with open(RETRAIN_HISTORY_FILE, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            if isinstance(raw, list):
                history = raw
        except Exception as e:
            logger.warning(f"⚠️ 재학습 이력 로드 실패 (새로 시작): {e}")

    history.append(entry)
    if len(history) > _HISTORY_MAX_ENTRIES:
        history = history[-_HISTORY_MAX_ENTRIES:]

    tmp = RETRAIN_HISTORY_FILE + ".tmp"
    try:
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        os.replace(tmp, RETRAIN_HISTORY_FILE)
    except Exception as e:
        logger.warning(f"⚠️ 재학습 이력 저장 실패: {e}")
        try:
            os.remove(tmp)
        except OSError:
            pass


def _is_model_fresh(max_age_days: int = MODEL_FRESHNESS_DAYS) -> bool:
    """현재 모델이 max_age_days 일 이내에 학습됐으면 True 반환."""
    cfg = _load_latest_model_config()
    ts_str = cfg.get("timestamp")
    if not ts_str:
        return False
    try:
        model_time = datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
        age_days = (datetime.now() - model_time).days
        logger.info(f"현재 모델 나이: {age_days}일 (신선도 임계값: {max_age_days}일)")
        return age_days < max_age_days
    except Exception as e:
        logger.warning(f"모델 타임스탬프 파싱 실패: {e}")
        return False


def _log_oos_delta(old_cfg: dict, new_cfg: dict):
    """구 모델 vs 신 모델 OOS 지표 차이를 로그로 출력."""
    old_m = old_cfg.get("oos_metrics", {})
    new_m = new_cfg.get("oos_metrics", {})
    if not old_m or not new_m:
        logger.info("OOS 지표 비교 불가 (이전 모델에 oos_metrics 없음)")
        return

    lines = ["📊 OOS 성능 비교 (구 모델 → 신 모델):"]
    for key in ("precision", "recall", "f1", "auc"):
        old_v = old_m.get(key, 0.0)
        new_v = new_m.get(key, 0.0)
        delta = new_v - old_v
        arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "─")
        lines.append(f"  {key:12s}: {old_v:.4f} → {new_v:.4f}  {arrow}{abs(delta):.4f}")
    _notify("\n".join(lines))  # _notify 내부에서 logger.info 처리


def _is_pid_running(pid: int) -> bool:
    """PID가 현재 실행 중인지 확인 (Windows/Linux 공통)."""
    try:
        if os.name == 'nt':
            import ctypes
            handle = ctypes.windll.kernel32.OpenProcess(0x100000, False, pid)
            if handle:
                ctypes.windll.kernel32.CloseHandle(handle)
                return True
            return False
        else:
            os.kill(pid, 0)
            return True
    except (OSError, ProcessLookupError):
        return False


def _acquire_lock() -> bool:
    """재학습 중복 실행 방지용 PID 파일 락. O_EXCL 원자적 생성으로 레이스 컨디션 제거."""
    if os.path.exists(LOCK_FILE):
        age = time.time() - os.path.getmtime(LOCK_FILE)
        try:
            with open(LOCK_FILE, 'r') as f:
                old_pid = int(f.read().strip())
        except (ValueError, Exception):
            old_pid = -1

        if age < LOCK_STALE_SECONDS and old_pid > 0 and _is_pid_running(old_pid):
            logger.warning(
                f"⚠️ 재학습 중복 실행 차단 — PID {old_pid} 실행 중 "
                f"(파일 나이 {age/60:.0f}분 < {LOCK_STALE_SECONDS//60}분)"
            )
            return False
        logger.warning(
            f"⚠️ 고아 락 파일 (PID {old_pid}, {age/60:.0f}분) — 제거 후 진행"
        )
        try:
            os.remove(LOCK_FILE)
        except OSError:
            pass

    try:
        fd = os.open(LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True
    except FileExistsError:
        logger.warning("⚠️ 재학습 중복 실행 차단 — 락 파일 경쟁 (race condition)")
        return False
    except Exception as e:
        logger.error(f"락 파일 생성 실패: {e}")
        return False


def _release_lock():
    try:
        if os.path.exists(LOCK_FILE):
            os.remove(LOCK_FILE)
    except Exception as e:
        logger.warning(f"락 파일 해제 실패: {e}")


def _check_performance_drift() -> bool:
    """
    3-tier 통계적 드리프트 감지 (scipy 기반, river 불필요).

    Tier 1 — Wilcoxon 단측검정: 거래 충분 봇 집합의 payoff_ema 중앙값이
              이론 payoff (tp_mult/sl_mult) 미만인지 검정 (H1: median < theoretical).
    Tier 2 — t-검정:  realized_pnl/initial_capital 평균이 음수인지 검정 (H1: mean < 0).
    Tier 3 — 폴백:    거래 부족 봇의 payoff_ema 단순 비율 임계 (기존 로직 유지).

    EMA alpha=0.1 + 초기값 2.0 구조에서 CB 3연패 차단으로 단일 봇이
    임계값 1.0에 도달 불가한 문제를 통계적 집합 검정으로 해결.
    """
    try:
        from scipy.stats import wilcoxon, ttest_1samp
    except ImportError:
        logger.warning("scipy 미설치 — Tier-3 폴백 전용 드리프트 체크")
        wilcoxon = ttest_1samp = None

    _min_trades  = _cfg.MODEL_MANAGEMENT.get("adwin_min_trades", 10)
    _alpha       = _cfg.MODEL_MANAGEMENT.get("adwin_significance", 0.05)
    _theoretical = (
        _cfg.LABELING.get("atr_tp_mult", 2.0)
        / _cfg.LABELING.get("atr_sl_mult", 1.0)
    )

    state_files = glob.glob(os.path.join(_STATE_DIR, "*_state.json"))
    if not state_files:
        return False

    payoff_samples: list = []   # Tier 1 표본
    nav_returns:    list = []   # Tier 2 표본
    fallback_below = 0
    total = 0
    now = time.time()

    for sf in state_files:
        try:
            with open(sf, 'r', encoding='utf-8') as fh:
                state = json.load(fh)
            hb = state.get("heartbeat", 0)
            ref_time = hb if hb > 0 else os.path.getmtime(sf)
            if now - ref_time > DRIFT_STATE_STALE_SECONDS:
                logger.debug(
                    f"⏭️ 스테일 상태 파일 무시: {os.path.basename(sf)} "
                    f"({(now - ref_time) / 3600:.1f}h 미갱신)"
                )
                continue
            if "payoff_ema" not in state:
                continue
            payoff      = float(state["payoff_ema"])
            trade_count = int(state.get("payoff_trade_count", 0))
            init_cap    = float(state.get("initial_capital", 0.0))
            pnl         = float(state.get("realized_pnl", 0.0))
            total += 1

            if trade_count >= _min_trades:
                payoff_samples.append(payoff)
                if init_cap > 0:
                    nav_returns.append(pnl / init_cap)
            else:
                if payoff < DRIFT_PAYOFF_THRESHOLD:
                    fallback_below += 1
                    logger.warning(
                        f"📉 {os.path.basename(sf)}: payoff_ema={payoff:.3f} "
                        f"< {DRIFT_PAYOFF_THRESHOLD} (거래 {trade_count}회)"
                    )
        except Exception:
            continue

    if total == 0:
        return False

    drift_signals: list = []

    # Tier 1: Wilcoxon 단측검정 — payoff_ema 집합 vs 이론 payoff
    if wilcoxon is not None and len(payoff_samples) >= 5:
        diffs = [p - _theoretical for p in payoff_samples]
        if len(set(diffs)) > 1:
            try:
                _, p_val = wilcoxon(diffs, alternative='less')
                if p_val < _alpha:
                    drift_signals.append(
                        f"Wilcoxon p={p_val:.4f} "
                        f"(유효봇 {len(payoff_samples)}개 payoff 중앙값 < {_theoretical:.2f})"
                    )
            except Exception as e:
                logger.debug(f"Wilcoxon 검정 실패: {e}")

    # Tier 2: t-검정 — NAV 수익률 평균 < 0
    if ttest_1samp is not None and len(nav_returns) >= 5:
        try:
            _, p_val = ttest_1samp(nav_returns, 0.0, alternative='less')
            if p_val < _alpha:
                avg_r = sum(nav_returns) / len(nav_returns)
                drift_signals.append(
                    f"NAV t-검정 p={p_val:.4f} (평균 수익률 {avg_r:.2%} < 0)"
                )
        except Exception as e:
            logger.debug(f"t-검정 실패: {e}")

    # Tier 3: 폴백 단순 비율 (거래 부족 봇, 기존 로직 유지)
    if total > 0 and fallback_below / total >= DRIFT_RATIO_TRIGGER:
        drift_signals.append(
            f"payoff 임계 미달 {fallback_below}/{total}개 봇 "
            f"({DRIFT_RATIO_TRIGGER * 100:.0f}% 초과)"
        )

    if drift_signals:
        summary = " | ".join(drift_signals)
        logger.warning(f"🚨 실전 성능 드리프트 — {summary} → 즉시 재학습")
        _notify(f"🚨 [드리프트 감지] {summary}\n→ 긴급 재학습 시작")
        return True

    logger.info(
        f"✅ 드리프트 없음 — 활성봇 {total}개 "
        f"(통계검정 대상: {len(payoff_samples)}개 / NAV: {len(nav_returns)}개)"
    )
    return False


def _notify(text: str):
    """텔레그램 알림 (실패해도 재학습 중단 없음)"""
    logger.info(text)
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=5)
    except Exception as te:
        logger.debug(f"텔레그램 전송 실패: {te}")


def _is_market_too_volatile(atr_threshold: float = 0.08) -> bool:
    """
    재학습 전 시장 극단 변동성 체크.
    BTC 일봉 ATR/가격 비율이 atr_threshold 이상이면 True 반환 → 재학습 연기.
    극단 변동 중 수집된 데이터로 학습하면 과적합 위험이 크다.
    """
    try:
        import pyupbit
        df = pyupbit.get_ohlcv("KRW-BTC", interval="day", count=15)
        if df is None or len(df) < 14:
            return False
        high_low  = df['high'] - df['low']
        high_prev = (df['high'] - df['close'].shift(1)).abs()
        low_prev  = (df['low']  - df['close'].shift(1)).abs()
        import pandas as pd
        tr    = pd.concat([high_low, high_prev, low_prev], axis=1).max(axis=1)
        price = float(df['close'].iloc[-1])
        ratio = float(tr.rolling(14).mean().iloc[-1]) / price if price > 0 else 0.0
        if ratio >= atr_threshold:
            logger.warning(
                f"⚠️ 시장 극단 변동성 감지 — BTC ATR/가격 {ratio*100:.2f}% ≥ "
                f"{atr_threshold*100:.0f}% → 재학습 연기 (다음 주기로 이월)"
            )
            _notify(
                f"⚠️ [재학습 연기] BTC ATR/가격 {ratio*100:.2f}% — "
                f"극단 변동성으로 이번 주기 재학습 건너뜀"
            )
            return True
        return False
    except Exception as e:
        logger.warning(f"변동성 체크 실패 ({e}) — 재학습 계속 진행")
        return False


def _update_coin_whitelist() -> None:
    """
    재학습 완료 후 최근 6개월 OOS 스트레스 백테스트로 코인 화이트리스트 갱신.
    카테고리별 전용 모델을 사용하여 각 코인의 실전 성능을 평가한다.
    실패해도 기존 whitelist 유지 — 실거래에 영향 없음.
    """
    try:
        from backtest import run_stress_period_backtest

        end_dt   = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        start_dt = (datetime.now() - timedelta(days=_WHITELIST_STRESS_MONTHS * 30)).strftime("%Y-%m-%d")

        result = run_stress_period_backtest(
            start=start_dt, end=end_dt,
            slippage_mult=_WHITELIST_SLIPPAGE_MULT,
            label="whitelist_oos",
        )
        all_per_coin: list = result.get("per_coin", [])

        if not all_per_coin:
            _notify("⚠️ [화이트리스트] 스트레스 결과 없음 — 업데이트 스킵")
            return

        # 기존 연속 실패 이력 로드
        existing: dict = {}
        if os.path.exists(_WHITELIST_FILE):
            try:
                with open(_WHITELIST_FILE, 'r', encoding='utf-8') as _wf:
                    existing = json.load(_wf)
            except Exception:
                pass
        consecutive_failures: dict = existing.get("consecutive_failures", {})

        whitelist, blocked, scores = [], [], {}

        for r in all_per_coin:
            coin = r["coin"]
            scores[coin] = {
                "roi": r["roi"], "mdd": r["mdd"],
                "sharpe": r["sharpe"], "trades": r["trades"],
            }

            # 카테고리 A 코어는 항상 허용
            if coin in _WHITELIST_ALWAYS_ALLOW:
                whitelist.append(coin)
                consecutive_failures[coin] = 0
                continue

            passed = (
                r["roi"]    >= _WHITELIST_ROI_MIN
                and r["mdd"] <= _WHITELIST_MDD_MAX
                and r["sharpe"] >= _WHITELIST_SHARPE_MIN
                and r["trades"] >= _WHITELIST_MIN_TRADES
            )

            if passed:
                whitelist.append(coin)
                consecutive_failures[coin] = 0
            else:
                consecutive_failures[coin] = consecutive_failures.get(coin, 0) + 1
                if consecutive_failures[coin] < 2:
                    # 1회 실패: 경고만, 아직 차단하지 않음 (2-strike 원칙)
                    whitelist.append(coin)
                else:
                    blocked.append(coin)

        # 최소 3개 보장 — 전원 차단 상황 방어
        if len(whitelist) < 3:
            for r in sorted(all_per_coin, key=lambda x: x["roi"], reverse=True):
                if r["coin"] not in whitelist:
                    whitelist.append(r["coin"])
                if len(whitelist) >= 3:
                    break

        whitelist_data = {
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "stress_period": {"start": start_dt, "end": end_dt},
            "slippage_mult": _WHITELIST_SLIPPAGE_MULT,
            "whitelist": sorted(set(whitelist)),
            "blocked": sorted(set(blocked)),
            "consecutive_failures": consecutive_failures,
            "per_coin_scores": scores,
        }

        tmp_path = _WHITELIST_FILE + ".tmp"
        with open(tmp_path, 'w', encoding='utf-8') as _wf:
            json.dump(whitelist_data, _wf, ensure_ascii=False, indent=2)
        os.replace(tmp_path, _WHITELIST_FILE)

        block_msg = f"차단: {', '.join(sorted(set(blocked)))}" if blocked else "전체 통과"
        _notify(
            f"📊 [화이트리스트] {len(set(whitelist))}개 허용 / {len(set(blocked))}개 차단\n"
            f"평가 기간: {start_dt}~{end_dt} (슬리피지 ×{_WHITELIST_SLIPPAGE_MULT})\n"
            f"{block_msg}"
        )
        logger.info(f"✅ 화이트리스트 갱신 완료 — 허용: {sorted(set(whitelist))} | 차단: {sorted(set(blocked))}")

    except Exception as e:
        _notify(f"⚠️ [화이트리스트] 갱신 실패 (실거래 영향 없음): {e}")
        logger.error(f"화이트리스트 갱신 실패: {e}\n{traceback.format_exc()}")


def recalibrate_models(weeks: int = 0) -> None:
    """XGBoost/LightGBM 가중치 불변, post-hoc isotonic 캘리브레이터만 최근 N주 데이터로 재적합.
    전체 재학습(~10분) 대비 ~30초로 확률 분포 드리프트를 저비용 교정.
    매주 목요일 KST 03:00 실행 (retrain_job 수·일과 겹치지 않음).
    weeks=0이면 config.MODEL_MANAGEMENT["recal_weeks"] 사용.
    """
    import joblib
    import pandas as pd

    try:
        from machine_learning import load_and_prepare_data
    except Exception as e:
        logger.error(f"recalibrate_models: machine_learning import 실패 — {e}")
        return

    if weeks <= 0:
        weeks = _cfg.MODEL_MANAGEMENT["recal_weeks"]
    recal_bars = weeks * 7 * 24  # N주 × 60m봉
    _input_dir = _cfg.DIRECTORIES["data_processed"]
    _BTC_REF_FEATS = _cfg.MODEL_MANAGEMENT["btc_ref_feats"]

    logger.info("🔄 [온라인 재캘리브레이션] 시작")
    results = {}

    for cat_key, cat_info in _cfg.COIN_CATEGORIES.items():
        if cat_info.get("skip_training"):
            continue

        model_dir = cat_info["model_dir"]
        pkls = sorted(glob.glob(os.path.join(model_dir, "*ensemble_bot*.pkl")))
        cfgs = sorted(glob.glob(os.path.join(model_dir, "config_*.json")))
        if not pkls or not cfgs:
            logger.warning(f"  ⚠️ {cat_key}: 모델 없음 — 스킵")
            results[cat_key] = "스킵(모델없음)"
            continue

        try:
            model = joblib.load(pkls[-1])
            with open(cfgs[-1]) as f:
                cfg_data = json.load(f)
            # 피처 순서는 config가 아닌 모델 부스터에서 직접 획득 (순서 불일치 방지)
            try:
                features = model.xgb_model.calibrated_classifiers_[0] \
                    .estimator.get_booster().feature_names
            except Exception:
                features = cfg_data.get("features", [])

            coins   = cat_info.get("coins") or None
            exclude = None
            if not coins:
                exclude = set()
                for k, v in _cfg.COIN_CATEGORIES.items():
                    if k != cat_key and v.get("coins"):
                        exclude.update(v["coins"])

            df, _ = load_and_prepare_data(
                input_dir=_input_dir,
                coins=coins,
                exclude_coins=exclude,
                interval="minute60",
            )
            if df is None or df.empty:
                logger.warning(f"  ⚠️ {cat_key}: 데이터 로드 실패 — 스킵")
                results[cat_key] = "스킵(데이터없음)"
                continue

            # B/C 카테고리 BTC 레퍼런스 피처 병합 (훈련 시와 동일 로직)
            if cat_key in ('B', 'C') and 'timestamp' in df.columns:
                _btc_csv = os.path.join(_input_dir, 'processed_KRW-BTC_minute60.csv')
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
                        df[[c for c in df.columns if c.startswith('BTC_')]] = \
                            df[[c for c in df.columns if c.startswith('BTC_')]].ffill()
                    except Exception as _e:
                        logger.warning(f"  ⚠️ {cat_key}: BTC 피처 병합 실패 — {_e}")

            target_col = "Target" if "Target" in df.columns else "target"
            avail = [f for f in features if f in df.columns]
            if len(avail) < len(features) * 0.7:
                logger.warning(f"  ⚠️ {cat_key}: 피처 {len(avail)}/{len(features)} 부족 — 스킵")
                results[cat_key] = f"스킵(피처부족 {len(avail)}/{len(features)})"
                continue

            df_recent = df[avail + [target_col]].dropna().iloc[-recal_bars:]
            if len(df_recent) < 100:
                logger.warning(f"  ⚠️ {cat_key}: 최근 데이터 {len(df_recent)}행 부족 — 스킵")
                results[cat_key] = f"스킵(행부족 {len(df_recent)})"
                continue

            X_cal = df_recent[avail]
            y_cal = df_recent[target_col]
            pos_rate = float(y_cal.mean())

            model.fit_recalibration(X_cal, y_cal)

            # 원자적 저장 (봇이 읽는 도중 깨지지 않도록)
            tmp_path = pkls[-1] + ".recal.tmp"
            joblib.dump(model, tmp_path)
            os.replace(tmp_path, pkls[-1])

            logger.info(
                f"  ✅ {cat_key}: 재캘리브레이션 완료 "
                f"(n={len(df_recent)}, pos={pos_rate:.1%}, "
                f"model={os.path.basename(pkls[-1])})"
            )
            results[cat_key] = f"✅ n={len(df_recent)} pos={pos_rate:.1%}"

        except Exception as e:
            logger.error(f"  ❌ {cat_key}: 재캘리브레이션 실패 — {e}\n{traceback.format_exc()}")
            results[cat_key] = f"실패({e})"

    summary = " | ".join(f"{k}:{v}" for k, v in results.items())
    logger.info(f"🔄 [온라인 재캘리브레이션] 완료: {summary}")
    _notify(f"🔄 [재캘리브레이션] {summary}")


def retrain_job(force: bool = False):
    """
    재학습 파이프라인: 데이터 수집 → 피처 엔지니어링 → 모델 훈련
    각 단계가 실패하면 알림 후 해당 주기는 건너뜀.
    PID 락으로 스케줄 실행과 RETRAIN_NOW 수동 실행이 동시에 실행되지 않도록 보호.

    Args:
        force: True이면 모델 신선도 체크를 건너뜀 (드리프트 긴급 재학습 시 사용).
    """
    if not _acquire_lock():
        return  # 이미 다른 프로세스가 재학습 중

    start = datetime.now()
    try:
        # ── 개선 1: 모델 신선도 체크 — 7일 이내 모델은 정기 재학습 스킵
        if not force and _is_model_fresh():
            logger.info(
                f"⏭️ 모델이 {MODEL_FRESHNESS_DAYS}일 이내에 학습됨 — 정기 재학습 스킵 "
                f"(드리프트 감지 시 force=True로 재실행)"
            )
            return

        # ── 개선 2: 극단 변동성 중에는 재학습 연기 (force=True이면 드리프트 긴급 재학습이므로 강행)
        if not force and _is_market_too_volatile():
            return

        logger.info("=" * 60)
        logger.info(f"🔄 재학습 시작: {start.strftime('%Y-%m-%d %H:%M:%S')}"
                    + (" [강제 실행]" if force else ""))
        logger.info("=" * 60)
        _notify(f"🤖 [재학습] {start.strftime('%m/%d %H:%M')} 시작"
                + (" (드리프트 감지)" if force else ""))

        try:
            # 단계별 임포트를 함수 내부에서 수행 — 모듈 로드 오류 격리
            from main import run_full_pipeline, run_ml_only
        except ImportError as e:
            _notify(f"❌ [재학습 실패] main.py 임포트 불가: {e}")
            logger.error(f"main.py 임포트 실패: {e}")
            return

        # Phase 1: 데이터 수집 + 피처 엔지니어링
        try:
            logger.info("📥 Phase 1: 데이터 수집 및 피처 엔지니어링...")
            result = run_full_pipeline(overwrite=True)  # 재학습 시 기존 데이터 파일 갱신 필수
            if not result.get("success"):
                raise RuntimeError("run_full_pipeline() 실패")
            logger.info(
                f"✅ 데이터 준비 완료 — "
                f"수집 {result['data_collection']['success']}개, "
                f"피처 {result['feature_engineering']['total_features']}개"
            )
        except Exception as e:
            _notify(f"❌ [재학습 실패] Phase 1 오류: {e}")
            logger.error(f"Phase 1 오류: {e}\n{traceback.format_exc()}")
            return

        # ── Phase 2 전에 구 모델 OOS 지표 캡처 + 백업
        old_model_cfg = _load_latest_model_config()
        backup_ok = _backup_current_model()

        # Phase 2: 모델 재학습
        try:
            logger.info("🧠 Phase 2: 머신러닝 모델 재학습...")
            run_ml_only()
            logger.info("✅ 모델 재학습 완료 — models/ 폴더에 저장됨")
        except Exception as e:
            _notify(f"❌ [재학습 실패] Phase 2 오류: {e}")
            logger.error(f"Phase 2 오류: {e}\n{traceback.format_exc()}")
            if backup_ok:
                if _restore_model_backup():
                    _notify("🔁 [모델 롤백] Phase 2 실패 — 이전 모델 복원 완료")
            _append_retrain_history({
                "timestamp": start.strftime("%Y-%m-%d %H:%M:%S"),
                "force": force,
                "success": False,
                "error": str(e),
                "old_precision": old_model_cfg.get("oos_metrics", {}).get("precision"),
                "new_precision": None,
                "elapsed_seconds": (datetime.now() - start).seconds,
            })
            return

        # Phase 2-B: 선물 모델 재학습 (비치명적 — 실패해도 현물 결과 유지)
        _fm = getattr(_cfg, "FUTURES_MODEL", {})
        if _fm:
            try:
                logger.info("🚀 Phase 2-B: 선물 모델 재학습...")
                from data_pipeline import collect_futures_ohlcv
                from feature_engineering import process_all_data as _fe_process
                from machine_learning import (
                    load_and_prepare_data as _ml_load,
                    optimize_and_train_bot as _ml_train,
                    save_model as _ml_save,
                )
                collect_futures_ohlcv(
                    symbols=_fm.get("symbols", ["BTC/USDT"]),
                    intervals=_fm.get("intervals", ["minute15", "minute60"]),
                    count=_fm.get("count", 1000),
                )
                _fe_process(
                    input_dir=_cfg.DIRECTORIES["data_futures"],
                    output_dir=_cfg.DIRECTORIES["data_futures_processed"],
                )
                futures_df, _ = _ml_load(_cfg.DIRECTORIES["data_futures_processed"])
                if futures_df is not None and not futures_df.empty:
                    n_trials = max(30, _fm.get("n_trials", 50))
                    f_model, f_threshold, f_imp, f_stats, _ = _ml_train(
                        futures_df, n_trials=n_trials, verbose=False
                    )
                    if f_model is not None:
                        _ml_save(
                            f_model, f_threshold, f_imp, f_stats,
                            model_name=_fm.get("model_name", "futures_bot"),
                            output_dir=_FUTURES_MODEL_DIR,
                        )
                        logger.info("✅ 선물 모델 재학습 완료 — models_futures/ 저장됨")
                    else:
                        logger.warning("⚠️ 선물 모델 훈련 실패 (OOS 기준 미달)")
                else:
                    logger.warning("⚠️ 선물 학습 데이터 없음 — Phase 2-B 스킵")
            except Exception as _fe:
                _notify(f"⚠️ [선물 재학습 실패] 현물 모델은 정상: {_fe}")
                logger.error(f"Phase 2-B (선물) 오류: {_fe}\n{traceback.format_exc()}")
                if _restore_dir(_FUTURES_ROLLBACK_DIR, _FUTURES_MODEL_DIR) > 0:
                    logger.info("🔁 선물 모델 롤백 완료 — 이전 버전 복원")

        # 신 모델 OOS 지표 로드 후 델타 비교 + 롤백 판단
        new_model_cfg = _load_latest_model_config()
        _log_oos_delta(old_model_cfg, new_model_cfg)

        old_m = old_model_cfg.get("oos_metrics", {})
        new_m = new_model_cfg.get("oos_metrics", {})
        old_precision = old_m.get("precision", 0.0)
        new_precision = new_m.get("precision", 0.0)
        # 복합 점수: precision 60% + AUC 40% — 단일 지표 롤백 오탐 방지
        old_score = old_precision * 0.6 + old_m.get("auc", old_precision) * 0.4
        new_score = new_precision * 0.6 + new_m.get("auc", new_precision) * 0.4
        rolled_back = False
        if old_score > 0 and new_score < old_score - 0.03 and backup_ok:
            logger.warning(
                f"⚠️ 신 모델 복합 점수 하락 ({old_score:.4f} → {new_score:.4f}) "
                f"— 이전 모델로 롤백"
            )
            if _restore_model_backup():
                _notify(
                    f"🔁 [모델 롤백] 복합 점수 하락 ({old_score:.3f} → {new_score:.3f}) "
                    f"— 이전 모델 복원"
                )
                rolled_back = True

        elapsed = int((datetime.now() - start).total_seconds())
        _append_retrain_history({
            "timestamp": start.strftime("%Y-%m-%d %H:%M:%S"),
            "force": force,
            "success": True,
            "rolled_back": rolled_back,
            "old_precision": old_precision if old_precision > 0 else None,
            "new_precision": new_precision if new_precision > 0 else None,
            "elapsed_seconds": elapsed,
        })
        # 연속 롤백 에스컬레이션: 임계 횟수 이상 연속 롤백이면 수동 개입 경고
        if rolled_back:
            try:
                rollback_streak = 0
                if os.path.exists(RETRAIN_HISTORY_FILE):
                    with open(RETRAIN_HISTORY_FILE, 'r', encoding='utf-8') as _hf:
                        _hist = json.load(_hf)
                    if isinstance(_hist, list):
                        for _e in reversed(_hist):
                            if _e.get("rolled_back"):
                                rollback_streak += 1
                            else:
                                break
                if rollback_streak >= _ROLLBACK_ESCALATION_THRESHOLD:
                    _notify(
                        f"🚨 [롤백 에스컬레이션] {rollback_streak}회 연속 롤백 발생 — "
                        f"모델 품질 저하 지속, 수동 점검 및 개입 필요"
                    )
            except Exception:
                pass

        # 화이트리스트 갱신 (새 카테고리 모델 기준 OOS 스트레스 평가)
        if not rolled_back:
            _update_coin_whitelist()

        # trade_bot이 폴링하는 플래그 파일로 새 모델 로드 시그널
        if not rolled_back:
            try:
                os.makedirs(_STATE_DIR, exist_ok=True)
                with open(_MODEL_UPDATED_FLAG, 'w') as _mf:
                    _mf.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            except Exception as _mfe:
                logger.debug(f"model_updated.flag 생성 실패: {_mfe}")

        _notify(
            f"✅ [재학습 완료] {start.strftime('%m/%d')}\n"
            f"소요 시간: {elapsed // 60}분 {elapsed % 60}초\n"
            + ("⚠️ 롤백됨 (복합 점수 하락)\n" if rolled_back else "")
            + "trade_bot 다음 구동 시 새 모델 자동 로드됩니다."
        )
        logger.info(f"✅ 재학습 완료 ({elapsed}초 소요)")
    finally:
        _release_lock()


def _register_kst_schedule(kst_day: str = "sunday", kst_hour: int = 3) -> None:
    """
    KST 기준 요일·시각을 서버 로컬 클럭으로 변환하여 schedule 등록.
    UTC 서버에서도 KST 일요일 03:00에 정확히 실행됨.
    자정 경계를 넘을 경우 요일도 자동 이동.
    """
    _DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    try:
        local_utc_offset_h = datetime.now().astimezone().utcoffset().total_seconds() / 3600
    except Exception:
        local_utc_offset_h = 9.0  # 폴백: KST 가정
    local_hour_float = kst_hour + (local_utc_offset_h - 9.0)
    day_shift = int(local_hour_float // 24)
    local_hour = int(local_hour_float % 24)
    local_day = _DAYS[(_DAYS.index(kst_day) + day_shift) % 7]
    local_time_str = f"{local_hour:02d}:00"
    getattr(schedule.every(), local_day).at(local_time_str).do(retrain_job)
    logger.info(
        f"⏳ 재학습 스케줄: 매주 {local_day} {local_time_str} (로컬) "
        f"= KST {kst_day} {kst_hour:02d}:00 "
        f"(서버 UTC{local_utc_offset_h:+.0f}h)"
    )


def main():
    logger.info("⏳ 자동 재학습 스케줄러 가동 중 (주 2회: 수요일·일요일 KST 02:00)...")
    _notify("⏳ [재학습 스케줄러] 가동 시작 — 수요일·일요일 KST 02:00 실행")

    _register_kst_schedule("wednesday", 2)
    _register_kst_schedule("sunday", 2)

    # 매주 목요일 KST 03:00 온라인 재캘리브레이션 (재학습 날과 겹치지 않음)
    def _register_recal_schedule(kst_day: str, kst_hour: int):
        _DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        try:
            local_utc_offset_h = datetime.now().astimezone().utcoffset().total_seconds() / 3600
        except Exception:
            local_utc_offset_h = 9.0
        local_hour_float = kst_hour + (local_utc_offset_h - 9.0)
        day_shift = int(local_hour_float // 24)
        local_hour = int(local_hour_float % 24)
        local_day = _DAYS[(_DAYS.index(kst_day) + day_shift) % 7]
        local_time_str = f"{local_hour:02d}:00"
        getattr(schedule.every(), local_day).at(local_time_str).do(recalibrate_models)
        logger.info(
            f"⏳ 재캘리브레이션 스케줄: 매주 {local_day} {local_time_str} (로컬) "
            f"= KST {kst_day} {kst_hour:02d}:00"
        )

    _register_recal_schedule("thursday", 3)

    # 즉시 실행 옵션: 환경변수 RETRAIN_NOW=1 로 테스트 가능
    if os.getenv("RETRAIN_NOW") == "1":
        logger.info("🔧 RETRAIN_NOW=1 감지 — 즉시 재학습 실행")
        retrain_job()

    _drift_check_interval = 3600  # 1시간마다 드리프트 체크
    global _last_drift_check

    while True:
        schedule.run_pending()

        now = time.time()
        if now - _last_drift_check >= _drift_check_interval:
            _last_drift_check = now
            if _check_performance_drift():
                _last_drift_check = time.time()  # 재학습 시작 즉시 갱신 — 완료 전 중복 트리거 차단
                retrain_job(force=True)
                _last_drift_check = time.time()  # 재학습 완료 후 갱신 — 봇 모델 로드 전 재감지 방지

        time.sleep(60)


if __name__ == "__main__":
    main()
