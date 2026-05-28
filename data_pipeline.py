#data_pipeline.py
import os
import time
import logging
import requests
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime
from typing import List, Optional, Dict, Tuple
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, as_completed
import config as _cfg

# ============================================================================
# 1. 환경 변수 및 상수 정의
# ============================================================================
load_dotenv()
UPBIT_ACCESS_KEY = os.getenv("UPBIT_ACCESS_KEY")
UPBIT_SECRET_KEY = os.getenv("UPBIT_SECRET_KEY")

# API 설정
DEFAULT_COUNT = 2000
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 1
API_RATE_LIMIT_DELAY = 0.5
NETWORK_ERROR_DELAY = 2
API_ERROR_DELAY = 1

# 디렉토리 설정 (config.DIRECTORIES 단일 소스)
DATA_DIR = _cfg.DIRECTORIES["data"]
LOG_DIR  = _cfg.DIRECTORIES["logs"]

# ============================================================================
# 2. 로깅 설정
# ============================================================================
def setup_logging(name: str = 'data_api', log_level: int = logging.INFO) -> logging.Logger:
    """표준 로깅 모듈을 사용한 로그 설정 (중복 방지)"""
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    
    logger = logging.getLogger(name)
    
    # 이미 핸들러가 있으면 반환
    if logger.handlers:
        return logger
    
    logger.setLevel(log_level)
    
    # 파일 핸들러
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(f"{LOG_DIR}/data_collection_{timestamp}.log", encoding='utf-8')
    
    # 콘솔 핸들러
    ch = logging.StreamHandler()
    
    # 포매터
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

logger = setup_logging()

# ============================================================================
# 3. 재시도 데코레이터 (기능적 개선)
# ============================================================================
_NON_RETRYABLE_EXCEPTIONS = (
    # 재시도해도 의미 없는 에러 유형 — 즉시 실패 처리
    KeyError,
    ValueError,
)

_RATE_LIMIT_BASE_DELAY = 60   # 429 수신 시 기본 대기 (초) — Retry-After 없을 때 사용
_RATE_LIMIT_MAX_DELAY  = 300  # 429 연속 수신 시 최대 대기 상한

def _get_retry_after(response) -> float:
    """응답 헤더에서 Retry-After 값을 파싱. 없으면 _RATE_LIMIT_BASE_DELAY 반환."""
    if response is None:
        return _RATE_LIMIT_BASE_DELAY
    header = response.headers.get("Retry-After") or response.headers.get("retry-after")
    if header:
        try:
            return min(float(header), _RATE_LIMIT_MAX_DELAY)
        except (ValueError, TypeError):
            pass
    return _RATE_LIMIT_BASE_DELAY

def retry_on_error(max_retries: int = MAX_RETRIES, initial_delay: int = INITIAL_RETRY_DELAY):
    """실패 시 자동으로 재시도하는 데코레이터.
    - 재시도 불가 에러(400/404/422/KeyError/ValueError)는 즉시 실패 처리.
    - HTTP 429 Rate Limit: Retry-After 헤더 또는 지수 백오프(최대 300초) 적용.
    - 그 외 HTTP/일반 에러: 지수 백오프.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except _NON_RETRYABLE_EXCEPTIONS as e:
                    logger.error(f"재시도 불가 에러 ({type(e).__name__}) — 즉시 중단: {func.__name__}")
                    raise
                except requests.HTTPError as e:
                    status = e.response.status_code if e.response is not None else 0
                    if status in (400, 404, 422):
                        logger.error(f"HTTP {status} — 재시도 불가, 즉시 중단: {func.__name__}")
                        raise
                    if status == 429:
                        # Retry-After는 서버가 지정한 대기 시간이므로 그대로 사용.
                        # 지수 배수를 곱하면 60초 헤더 × 4 = 240초로 과도하게 증폭됨.
                        wait = min(_get_retry_after(e.response), _RATE_LIMIT_MAX_DELAY)
                        logger.warning(
                            f"HTTP 429 Rate Limit — {attempt}/{max_retries}회, "
                            f"{wait:.0f}초 대기 후 재시도: {func.__name__}"
                        )
                        time.sleep(wait)
                        continue
                    if attempt == max_retries:
                        raise
                    delay = initial_delay * (2 ** (attempt - 1))
                    logger.warning(f"HTTP {status} 재시도 {attempt}/{max_retries}: {delay}초 대기")
                    time.sleep(delay)
                except Exception as e:
                    # pyupbit는 requests.HTTPError 대신 문자열로 rate limit 전달 가능
                    msg = str(e).lower()
                    if "429" in msg or "too many requests" in msg or "rate limit" in msg:
                        wait = min(
                            _RATE_LIMIT_BASE_DELAY * (2 ** (attempt - 1)),
                            _RATE_LIMIT_MAX_DELAY
                        )
                        logger.warning(
                            f"Rate Limit 감지 ({type(e).__name__}) — {attempt}/{max_retries}회, "
                            f"{wait:.0f}초 대기: {func.__name__}"
                        )
                        time.sleep(wait)
                        continue
                    if attempt == max_retries:
                        logger.error(f"최대 재시도 횟수({max_retries}) 초과: {func.__name__}")
                        raise
                    delay = initial_delay * (2 ** (attempt - 1))
                    logger.warning(f"재시도 {attempt}/{max_retries}: {delay}초 대기 후 재시도... ({type(e).__name__})")
                    time.sleep(delay)
        return wrapper
    return decorator

# ============================================================================
# 4. 데이터 품질 검증 및 통계 (기능적 개선)
# ============================================================================
class DataQualityStats:
    """데이터 품질 검증 통계"""
    def __init__(self):
        self.null_count = 0
        self.duplicates_count = 0
        self.removed_rows = 0
    
    def __str__(self):
        return f"결측치:{self.null_count}, 중복:{self.duplicates_count}, 제거행:{self.removed_rows}"

def validate_data_quality(df: pd.DataFrame, name: str) -> Tuple[Optional[pd.DataFrame], DataQualityStats]:
    """
    데이터 품질 검증 및 정제 (기능 개선: 통계 반환)
    
    Args:
        df (pd.DataFrame): 검사할 데이터프레임
        name (str): 데이터 식별명 (로그용)
        
    Returns:
        Tuple[Optional[pd.DataFrame], DataQualityStats]: (정제된 df, 통계)
    """
    stats = DataQualityStats()
    
    if df is None or df.empty:
        logger.warning(f"  ⚠️  [{name}] 빈 데이터프레임")
        return None, stats
    
    initial_len = len(df)
    
    # 1. 결측치 검사
    stats.null_count = df.isnull().sum().sum()
    if stats.null_count > 0:
        logger.warning(f"  ⚠️  [{name}] 결측치 {stats.null_count}개 감지, 제거 처리")
        df = df.dropna()
    
    # 2. 중복 데이터 검사
    stats.duplicates_count = df.index.duplicated().sum()
    if stats.duplicates_count > 0:
        logger.warning(f"  ⚠️  [{name}] 중복 데이터 {stats.duplicates_count}개 감지, 제거 처리")
        df = df[~df.index.duplicated(keep='last')]
    
    stats.removed_rows = max(0, initial_len - len(df))
    
    # 3. 정제 후 빈 데이터프레임 체크
    if df.empty:
        logger.error(f"  ❌ [{name}] 정제 후 데이터 없음")
        return None, stats
    
    # 4. 타임스탬프 연속성 검사 (timestamp 인덱스인 경우만)
    if pd.api.types.is_datetime64_any_dtype(df.index):
        gaps = df.index[1:] - df.index[:-1]
        try:
            gaps_series = pd.Series((gaps / np.timedelta64(1, 'm')).astype(int))
        except Exception:
            gaps_series = pd.Series((gaps.astype('timedelta64[ns]').astype('int64') // 60_000_000_000).astype(int))

        # 예상 간격 추론 (대부분이 같은 간격을 가짐)
        gap_mode = gaps_series.mode()
        if len(gap_mode) > 0:
            expected_gap = gap_mode[0]
            large_gaps = gaps_series[gaps_series > expected_gap]
            
            if len(large_gaps) > 0:
                logger.warning(
                    f"  ⚠️  [{name}] 타임스탑 갭 {len(large_gaps)}개 감지 "
                    f"(예상: {expected_gap}분, 최대: {large_gaps.max()}분)"
                )
                
                # 5봉 초과 갭은 경고, 5봉 이하는 forward-fill 처리
                large_gap_idx = [i for i, gap in enumerate(large_gaps) if gap > expected_gap * 5]
                if large_gap_idx:
                    logger.debug(
                        f"  [{name}] 대규모 갭 ({expected_gap * 5}분 초과): "
                        f"{len(large_gap_idx)}개 구간 — 데이터 연속성 확인 필요"
                    )
    
    return df, stats

def validate_tickers_intervals(tickers: List[str], intervals: List[str]) -> bool:
    """
    수집 대상 목록이 유효한지 검사합니다.
    
    Args:
        tickers (List[str]): 코인 목록
        intervals (List[str]): 시간대 목록
        
    Returns:
        bool: 유효하면 True, 아니면 False
    """
    if not tickers or not intervals:
        logger.error("❌ 코인 목록이나 시간대 목록이 비어 있습니다.")
        return False
    if not isinstance(tickers, list) or not isinstance(intervals, list):
        logger.error("❌ 코인 목록과 시간대 목록은 리스트여야 합니다.")
        return False
    return True

# ============================================================================
# 5. 핵심 데이터 수집 로직
# ============================================================================
@retry_on_error(max_retries=3)
def fetch_ohlcv(ticker: str, interval: str, count: int) -> Optional[pd.DataFrame]:
    """API에서 OHLCV 데이터 조회 - Raw API + 하드 타임아웃 (Pandas 모호성 에러 패치 완료)"""
    df_list = []
    to_datetime_utc = None
    remaining = count
    
    if interval.startswith("minute"):
        unit = interval.replace("minute", "")
        url = f"https://api.upbit.com/v1/candles/minutes/{unit}"
    else: # day
        url = "https://api.upbit.com/v1/candles/days"
        
    while remaining > 0:
        req_count = min(remaining, 200)
        headers = {"accept": "application/json"}
        params = {"market": ticker, "count": req_count}
        
        if to_datetime_utc:
            params["to"] = to_datetime_utc.strftime('%Y-%m-%d %H:%M:%S')
            
        try:
            res = requests.get(url, headers=headers, params=params, timeout=5)
            res.raise_for_status()
            data = res.json()
        except requests.HTTPError:
            raise
        except Exception as e:
            raise Exception(f"API 네트워크 장애 (Timeout 등): {e}")
            
        if not data:
            break
            
        df = pd.DataFrame(data)
        
        # 💥 핵심 수정: 인덱스 이름(timestamp) 명시적 제거로 기존 로직과의 충돌(Ambiguous) 원천 차단
        df.index = pd.to_datetime(df['candle_date_time_kst'])
        df.index.name = None 
        df['utc_time'] = pd.to_datetime(df['candle_date_time_utc'])
        
        df = df[['opening_price', 'high_price', 'low_price', 'trade_price', 'candle_acc_trade_volume', 'utc_time']]
        df.columns = ['open', 'high', 'low', 'close', 'volume', 'utc_time']
        df = df.sort_index()
        
        df_list.append(df[['open', 'high', 'low', 'close', 'volume']])
        remaining -= len(df)
        to_datetime_utc = df['utc_time'].iloc[0]
        
        if remaining > 0:
            time.sleep(0.2)
            
    if not df_list:
        return None
        
    final_df = pd.concat(df_list).sort_index()
    final_df = final_df[~final_df.index.duplicated(keep='first')]
    return final_df.tail(count)

def collect_ml_data(
    tickers: List[str],
    intervals: List[str],
    count=DEFAULT_COUNT,
    overwrite: bool = False,
    output_dir: str = DATA_DIR
) -> Dict:
    """
    업비트 API에서 OHLCV 데이터를 수집하여 CSV로 저장합니다.

    Args:
        tickers (List[str]): 코인 목록 (예: ["KRW-BTC", "KRW-ETH"])
        intervals (List[str]): 시간대 목록 (예: ["day", "minute60"])
        count (int | Dict[str, int]): 수집할 캔들 개수. 인터벌별 dict 가능
            예: {"minute15": 70080, "minute60": 35040, "days": 35040}
        overwrite (bool): 기존 파일 덮어쓰기 여부
        output_dir (str): 출력 디렉토리

    Returns:
        Dict: {
            "success": int,
            "failed": int,
            "skipped": int,
            "total_rows": int,
            "quality_issues": List[Dict]
        }
    """
    if not validate_tickers_intervals(tickers, intervals):
        return {"success": 0, "failed": 0, "skipped": 0, "total_rows": 0, "quality_issues": []}
    
    os.makedirs(output_dir, exist_ok=True)
    
    total_tasks = len(tickers) * len(intervals)
    current_task = 0
    
    logger.info(f"📥 총 {len(tickers)}개 코인, {len(intervals)}개 시간대에 대한 수집 시작 [총 {total_tasks}개 작업]")
    
    stats = {
        "success": 0,
        "failed": 0,
        "skipped": 0,
        "total_rows": 0,
        "quality_issues": []
    }
    
    # 타임프레임별 최소 수집 비율 (이 미만이면 심각한 갭으로 간주)
    _MIN_ROW_RATIO = {
        "days": 0.30,      # 일봉: 업비트 최대 ~3500봉 → effective_count=3500 기준
        "minute60": 0.40,
        "minute15": 0.40,  # 15분봉 35040개 요청 시 40% ≒ 14016개(≈146일) 이상이면 허용 (신규 상장 코인 포함)
    }
    # 이 비율 이상의 갭은 치명적 → 수집 실패(failed) 처리 (경고만 하면 결손 데이터로 학습)
    _GAP_ABORT_RATIO = {
        "days": 0.60,
        "minute60": 0.70,
        "minute15": 0.70,  # 30% 이상 결손이면 학습 데이터 품질 보장 불가
    }

    def _collect_one(ticker: str, interval: str) -> Tuple[str, str, Optional[pd.DataFrame], DataQualityStats, str]:
        """단일 (ticker, interval) 조합 수집 — ThreadPoolExecutor 작업 단위."""
        file_path = os.path.join(output_dir, f"{ticker}_{interval}.csv")
        MAX_STALE_DAYS = 7
        if os.path.exists(file_path) and not overwrite:
            file_age_days = (time.time() - os.path.getmtime(file_path)) / 86400
            if file_age_days <= MAX_STALE_DAYS:
                return ticker, interval, None, DataQualityStats(), "skipped"
            logger.warning(f"⚠️  [{ticker}] {interval} 파일 {file_age_days:.0f}일 경과 — 재수집")

        try:
            # 인터벌별 count 지원: dict이면 해당 인터벌 count, int이면 공통 count
            _raw_count = count.get(interval, count.get("default", DEFAULT_COUNT)) if isinstance(count, dict) else count
            # days 인터벌은 업비트 상장(2017) 이후 최대 ~3500봉만 존재
            effective_count = min(_raw_count, 3500) if interval == "days" else _raw_count
            df_raw = fetch_ohlcv(ticker, interval, effective_count)
            # 워커 스레드 내에서 슬립 → 실제 API 호출 속도를 제한 (메인 스레드 슬립은 효과 없음)
            time.sleep(API_RATE_LIMIT_DELAY)
            df_clean, qstats = validate_data_quality(df_raw, f"{ticker}_{interval}")

            if df_clean is None:
                return ticker, interval, None, qstats, "failed"

            # 최초 상장일 도달 여부: API가 요청보다 적은 봉을 반환 → 더 이상 과거 캔들 없음
            reached_history_start = len(df_raw) < effective_count

            # 절대 수집량 검증: 요청 봉 수 대비 실제 수집 비율
            min_ratio = _MIN_ROW_RATIO.get(interval, 0.50)
            if len(df_clean) < min_ratio * effective_count:
                if reached_history_start:
                    logger.info(
                        f"✅ [{ticker}] {interval} 최초 상장일 도달 — "
                        f"{len(df_clean)}봉 전량 수집 완료 (요청 {effective_count}봉의 {len(df_clean)/effective_count:.0%})"
                    )
                else:
                    logger.error(
                        f"❌ [{ticker}] {interval} 절대 수집량 부족 — "
                        f"요청 {effective_count}봉 중 {len(df_clean)}봉 수집 ({len(df_clean)/effective_count:.0%} < {min_ratio:.0%}) → 실패"
                    )
                    return ticker, interval, None, qstats, "failed"

            # 갭 감지: 상장 이후 실제 경과 시간 대비 결손율(Relative Gap) 계산
            abort_ratio = _GAP_ABORT_RATIO.get(interval, 0.70)

            if len(df_clean) > 2:
                time_diff = df_clean.index[-1] - df_clean.index[0]
                if interval == "minute15":
                    expected_bars = int(time_diff.total_seconds() / 900) + 1
                elif interval == "minute60":
                    expected_bars = int(time_diff.total_seconds() / 3600) + 1
                else: # days
                    expected_bars = int(time_diff.total_seconds() / 86400) + 1

                # 수집된 총 기간 대비 실제 들어있는 행의 비율
                internal_ratio = len(df_clean) / max(expected_bars, 1)

                if internal_ratio < abort_ratio:
                    if reached_history_start:
                        logger.info(
                            f"✅ [{ticker}] {interval} 최초 상장일 도달 — "
                            f"{len(df_clean)}봉 정상 완료 (기간 내 밀도 {internal_ratio:.0%})"
                        )
                    else:
                        logger.error(
                            f"❌ [{ticker}] {interval} 구간 내 데이터 훼손 심각 — "
                            f"예상 {expected_bars}행 중 {len(df_clean)}행 수집 ({internal_ratio:.0%}) → 실패 처리"
                        )
                        return ticker, interval, None, qstats, "failed"
                else:
                    # 상장 기간이 짧아 데이터가 적은 것은 정상으로 간주
                    logger.info(f"  [{ticker}] {interval} 수집 완료 (총 기간 내 보존율: {internal_ratio:.0%})")

            df_clean['timestamp'] = pd.to_datetime(df_clean.index)
            df_clean['coin'] = ticker.replace('KRW-', '')
            df_clean = df_clean.sort_values('timestamp').reset_index(drop=True)
            return ticker, interval, df_clean, qstats, "success"

        except Exception as e:
            logger.error(f"⚠️  수집 오류 ({ticker}-{interval}): {type(e).__name__} - {e}")
            return ticker, interval, None, DataQualityStats(), "failed"

    # ML 핵심 타임프레임(minute15) 먼저, 보조 데이터(minute60, days) 나중에
    # 동일 우선순위 내에서 worker 2개로 안정적 병렬 처리
    COLLECTION_PRIORITY = ["minute15", "minute60", "days"]
    prioritized_tasks: list = []
    for prio_interval in COLLECTION_PRIORITY:
        for t in tickers:
            if prio_interval in intervals:
                prioritized_tasks.append((t, prio_interval))
    # 우선순위 목록에 없는 interval 추가 (예: 사용자 커스텀 간격)
    for t in tickers:
        for i in intervals:
            if i not in COLLECTION_PRIORITY:
                prioritized_tasks.append((t, i))

    max_workers = min(2, len(prioritized_tasks))  # 2개로 줄여 API 안정성 확보
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_collect_one, t, i): (t, i) for t, i in prioritized_tasks}
        for future in as_completed(futures):
            current_task += 1
            progress = f"[{current_task}/{total_tasks}]"
            try:
                ticker, interval, df, quality_stats, status = future.result()
            except Exception as e:
                logger.error(f"{progress} future.result() 예외: {e}")
                stats["failed"] += 1
                continue

            if status == "skipped":
                logger.info(f"{progress} ⏭️  [{ticker}] {interval} 파일 이미 존재 (스킵)")
                stats["skipped"] += 1
            elif status == "success":
                file_path = os.path.join(output_dir, f"{ticker}_{interval}.csv")
                # 원자적 쓰기: 임시 파일 → os.replace() (크래시 시 기존 파일 보존)
                tmp_path = file_path + ".tmp"
                df.to_csv(tmp_path, index=False)
                os.replace(tmp_path, file_path)
                logger.info(f"{progress}  ✅ [{ticker}] {interval} 저장 (행:{len(df)}, 품질:{quality_stats})")
                stats["success"] += 1
                stats["total_rows"] += len(df)
            else:
                logger.error(f"{progress}  ❌ [{ticker}] {interval} 수집 실패 (status={status})")
                stats["failed"] += 1
                if quality_stats.null_count or quality_stats.duplicates_count:
                    stats["quality_issues"].append({
                        "ticker": ticker, "interval": interval,
                        "issue": str(quality_stats)
                    })
    
    # 최종 통계
    total = stats["success"] + stats["failed"] + stats["skipped"]
    logger.info(
        f"🎉 수집 완료 - 성공: {stats['success']}/{total}, "
        f"실패: {stats['failed']}, 스킵: {stats['skipped']}, "
        f"총 행 수: {stats['total_rows']}"
    )
    
    return stats

# ============================================================================
# 6. 퇴역 코인 정리
# ============================================================================

def cleanup_stale_coin_files(
    active_tickers: List[str],
    data_dir: str = DATA_DIR,
    archive_dir: str = None,
) -> int:
    """
    현재 TARGET_COINS에 없는 코인의 CSV 파일을 archive 폴더로 이동합니다.
    학습 데이터에 퇴역 코인이 포함되어 노이즈가 되는 것을 방지합니다.

    Returns:
        이동된 파일 수
    """
    if not os.path.exists(data_dir):
        return 0
    if archive_dir is None:
        archive_dir = os.path.join(data_dir, "_archived_coins")

    active_set = {t.replace("KRW-", "") for t in active_tickers}
    moved = 0
    for fname in os.listdir(data_dir):
        if not fname.endswith(".csv"):
            continue
        # futures_BTC_USDT_minute15.csv 같은 선물 파일은 현물 정리 대상 아님
        if fname.startswith("futures_"):
            continue
        parts = fname.split("_")
        if not parts:
            continue
        coin_slug = parts[0]  # 현물 파일명 형식: "KRW-BTC_minute15.csv"
        coin_id = coin_slug.replace("KRW-", "")
        if coin_id not in active_set:
            os.makedirs(archive_dir, exist_ok=True)
            src = os.path.join(data_dir, fname)
            dst = os.path.join(archive_dir, fname)
            try:
                os.replace(src, dst)
                moved += 1
                logger.info(f"🗄️  퇴역 코인 파일 아카이브: {fname} → _archived_coins/")
            except Exception as e:
                logger.warning(f"퇴역 파일 이동 실패 ({fname}): {e}")
    if moved:
        logger.info(f"🧹 퇴역 코인 정리 완료 — {moved}개 파일 아카이브")
    return moved


# ============================================================================
# 7. 통합 파이프라인 (기능적 개선)
# ============================================================================
def run_data_pipeline(target_coins: List[str], target_intervals: List[str],
                      count=DEFAULT_COUNT, overwrite: bool = False) -> Dict:
    """
    데이터 수집 파이프라인 실행.
    overwrite=False(기본): 이미 존재하는 파일 스킵 (초기 수집용).
    overwrite=True: 기존 파일 덮어쓰기 — auto_retrain.py 재학습 시 반드시 True로 호출해야 함.

    Args:
        target_coins (List[str]): 수집할 코인 목록
        target_intervals (List[str]): 수집할 시간대 목록
        count (int | Dict[str, int]): 수집할 캔들 개수. 인터벌별 dict 가능
        overwrite (bool): 기존 파일 덮어쓰기 여부

    Returns:
        Dict: 수집 결과
    """
    logger.info("=" * 80)
    logger.info(f"🚀 데이터 수집 파이프라인 시작 (overwrite={overwrite})")
    logger.info("=" * 80)

    start_time = time.time()

    # 퇴역 코인 CSV 정리 (수집 전 실행 — 학습 데이터에 노이즈 방지)
    cleanup_stale_coin_files(target_coins, data_dir=DATA_DIR)

    result = collect_ml_data(
        tickers=target_coins,
        intervals=target_intervals,
        count=count,
        overwrite=overwrite,
    )

    elapsed = time.time() - start_time
    logger.info(f"⏱️  총 소요 시간: {elapsed:.2f}초")

    return result

# ============================================================================
# 8. 바이낸스 USDT-M 선물 전용 데이터 수집 파이프라인
# ============================================================================

FUTURES_DATA_DIR = _cfg.DIRECTORIES.get("data_futures", "data_futures")
_FUTURES_TIMEFRAME_MAP = {
    "minute15": "15m",
    "minute60":  "1h",
    "day":       "1d",
}
_FUTURES_SYMBOLS = ["BTC/USDT", "ETH/USDT"]  # 선물 모델 훈련 기본 심볼
_FUTURES_BATCH_LIMIT = 1500  # 바이낸스 단일 요청 최대 봉 수
_FUTURES_INTERVAL_MS = {
    "minute15": 900_000,
    "minute60": 3_600_000,
    "day":      86_400_000,
}


def collect_futures_ohlcv(
    symbols: List[str] = None,
    intervals: List[str] = None,
    count: int = _cfg.FUTURES_MODEL["count"],
    overwrite: bool = True,
) -> Dict:
    """
    바이낸스 USDT-M 무기한 선물 OHLCV 데이터 수집.
    현물(업비트) 파이프라인과 완전히 분리된 별도 디렉토리에 저장.
    수집 데이터는 futures_bot.py 전용 모델 훈련에 사용된다.

    Returns:
        {"success": int, "failed": int, "files": List[str]}
    """
    try:
        import ccxt
    except ImportError:
        logger.error("ccxt 미설치 — pip install ccxt")
        return {"success": 0, "failed": 0, "files": []}

    symbols   = symbols   or _FUTURES_SYMBOLS
    intervals = intervals or ["minute15", "minute60"]
    os.makedirs(FUTURES_DATA_DIR, exist_ok=True)

    exchange = ccxt.binanceusdm({
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    })

    result = {"success": 0, "failed": 0, "files": []}

    def _collect_futures_one(symbol: str, interval: str) -> Tuple[bool, str]:
        tf = _FUTURES_TIMEFRAME_MAP.get(interval, "15m")
        interval_ms = _FUTURES_INTERVAL_MS.get(interval, 900_000)
        coin_slug = symbol.replace("/", "_")
        fname = os.path.join(FUTURES_DATA_DIR, f"futures_{coin_slug}_{interval}.csv")

        if not overwrite and os.path.exists(fname):
            logger.info(f"⏭️  건너뜀(기존): {os.path.basename(fname)}")
            return True, fname

        for attempt in range(3):
            try:
                # 페이지네이션: 바이낸스 단일 요청 최대 1500봉 한계 우회
                all_ohlcv: list = exchange.fetch_ohlcv(
                    symbol, tf, limit=min(count, _FUTURES_BATCH_LIMIT)
                )
                if not all_ohlcv:
                    raise ValueError("빈 응답")

                while len(all_ohlcv) < count:
                    batch_limit = min(count - len(all_ohlcv) + 1, _FUTURES_BATCH_LIMIT)
                    oldest_ts = all_ohlcv[0][0]
                    since = oldest_ts - batch_limit * interval_ms
                    new_batch = exchange.fetch_ohlcv(symbol, tf, since=since, limit=batch_limit)
                    if not new_batch:
                        break
                    new_batch = [c for c in new_batch if c[0] < oldest_ts]
                    if not new_batch:
                        break
                    all_ohlcv = new_batch + all_ohlcv

                ohlcv = all_ohlcv[-count:]

                df = pd.DataFrame(
                    ohlcv,
                    columns=["timestamp", "open", "high", "low", "close", "volume"],
                )
                df["timestamp"] = (
                    pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                    .dt.tz_convert("Asia/Seoul")
                    .dt.tz_localize(None)
                )
                df.sort_values("timestamp", inplace=True)
                df.drop_duplicates("timestamp", keep="last", inplace=True)
                df.set_index("timestamp", inplace=True)

                df_clean, _ = validate_data_quality(df, f"futures_{symbol}_{interval}")
                if df_clean is None:
                    return False, fname
                df_clean = df_clean.reset_index()

                tmp_fname = fname + ".tmp"
                df_clean.to_csv(tmp_fname, index=False, encoding="utf-8")
                os.replace(tmp_fname, fname)

                logger.info(
                    f"✅ 선물 수집: {symbol} {interval} "
                    f"({len(df_clean)}봉) → {os.path.basename(fname)}"
                )
                return True, fname

            except Exception as e:
                if attempt < 2:
                    wait = 2 ** attempt
                    logger.warning(
                        f"⚠️  {symbol}/{interval} 수집 실패({attempt+1}/3): {e} "
                        f"— {wait}초 후 재시도"
                    )
                    time.sleep(wait)
                else:
                    logger.error(f"❌ {symbol}/{interval} 최종 실패: {e}")
                    return False, fname
        return False, fname

    tasks = [(s, i) for s in symbols for i in intervals]
    max_workers = min(2, len(tasks))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures_map = {pool.submit(_collect_futures_one, s, i): (s, i) for s, i in tasks}
        for future in as_completed(futures_map):
            success, fname = future.result()
            if success:
                result["success"] += 1
                result["files"].append(fname)
            else:
                result["failed"] += 1

    try:
        exchange.close()
    except Exception:
        pass

    logger.info(
        f"📊 선물 데이터 수집 완료: 성공 {result['success']}개 / 실패 {result['failed']}개"
    )
    return result


# ============================================================================
# 8. 메인 실행
# ============================================================================
if __name__ == "__main__":
    # 수집 대상 설정
    TARGET_COINS = ["KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-SOL"]
    TARGET_INTERVALS = ["days", "minute60", "minute15"]
    
    # 파이프라인 실행
    result = run_data_pipeline(TARGET_COINS, TARGET_INTERVALS)
    
    # 결과 요약
    if result["success"] > 0:
        logger.info(f"✨ {result['success']}개 파일이 정상 수집되었습니다.")
        if result["quality_issues"]:
            logger.warning(f"⚠️  {len(result['quality_issues'])}개 항목에서 품질 문제 발생")