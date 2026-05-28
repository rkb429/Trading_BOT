# ============================================================================
# Coin AI Bot - 통합 메인 파이프라인(main.py)
# ============================================================================

import sys
import os
import io
import glob as _glob
import argparse
import logging
import time
from logging.handlers import RotatingFileHandler
from typing import Dict

# 로컬 모듈 import
from data_pipeline import run_data_pipeline
from feature_engineering import process_all_data
from machine_learning import main as run_ml_training
from backtest import run_backtest, run_stress_period_backtest, run_rolling_window_backtest
from market_context import get_dynamic_target_coins, get_observation_coins
import config

# ============================================================================
# 1. 로깅 설정
# ============================================================================

def setup_main_logger() -> logging.Logger:
    log_dir = config.DIRECTORIES["logs"]
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger("Coin_AI_Bot")

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        config.LOGGING["format"],
        datefmt=config.LOGGING["datefmt"]
    )

    utf8_stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", line_buffering=True
    ) if hasattr(sys.stdout, "buffer") else sys.stdout
    stream_handler = logging.StreamHandler(utf8_stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = RotatingFileHandler(
        os.path.join(log_dir, "coin_ai_bot.log"),
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

logger = setup_main_logger()

# ============================================================================
# 2. 파이프라인 함수
# ============================================================================

def print_header(title: str):
    """헤더 출력"""
    logger.info("=" * 80)
    logger.info(f"🚀 {title}")
    logger.info("=" * 80)

def print_section(title: str):
    """섹션 출력"""
    logger.info(f"\n{'─' * 80}")
    logger.info(f"📋 {title}")
    logger.info(f"{'─' * 80}")

def run_full_pipeline(overwrite: bool = False) -> Dict:
    """
    전체 파이프라인 실행.
    overwrite=True: auto_retrain.py에서 호출 시 기존 데이터 파일 갱신 (기본 False = 스킵).

    Returns:
        Dict: 파이프라인 결과 요약
    """
    print_header("Coin AI Bot - 전체 데이터 처리 파이프라인")

    start_time = time.time()
    pipeline_results = {
        "data_collection": None,
        "feature_engineering": None,
        "total_time": 0,
        "success": True
    }

    try:
        # ====================================================================
        # Phase 1: 데이터 수집
        # ====================================================================
        print_section("Phase 1: Upbit API에서 데이터 수집")

        target_coins = get_dynamic_target_coins(limit=30)
        obs_coins = [c for c in get_observation_coins() if c not in target_coins]
        collect_coins = target_coins + obs_coins
        if obs_coins:
            logger.info(f"👁 관찰 코인 {len(obs_coins)}개 수집 포함 (차단 중, 회복 감지용): {[c.replace('KRW-','') for c in obs_coins]}")
        logger.info(f"대상 코인: {len(collect_coins)}개 (카테고리 필수 {sum(len(v['coins']) for v in config.COIN_CATEGORIES.values())}개 + 거래대금 상위 보충 + 관찰)")
        logger.info(f"대상 시간대: {', '.join(config.TARGET_INTERVALS)}")
        logger.info(f"캔들 개수: {config.DATA_COLLECTION['count']}")

        collection_result = run_data_pipeline(
            target_coins=collect_coins,
            target_intervals=config.TARGET_INTERVALS,
            count=config.DATA_COLLECTION['count'],
            overwrite=overwrite,
        )
        
        pipeline_results["data_collection"] = collection_result

        logger.info("\n📊 데이터 수집 결과:")
        logger.info(f"  ✅ 성공: {collection_result['success']}")
        logger.info(f"  ❌ 실패: {collection_result['failed']}")
        logger.info(f"  ⏭️  스킵: {collection_result['skipped']}")
        logger.info(f"  📈 총 행 수: {collection_result['total_rows']}")

        # 단계 실패 중단 체인: 수집된 파일이 0개면 피처 엔지니어링 스킵
        total_attempts = collection_result['success'] + collection_result['failed']
        if collection_result['success'] == 0 and total_attempts > 0:
            logger.error(
                "❌ 데이터 수집 전 실패 — 피처 엔지니어링 및 이후 단계 중단 "
                "(수집 성공 파일이 0개입니다)"
            )
            pipeline_results["success"] = False
            pipeline_results["total_time"] = time.time() - start_time
            return pipeline_results

        # ====================================================================
        # Phase 2: 피처 엔지니어링
        # ====================================================================
        print_section("Phase 2: 피처 엔지니어링 및 데이터 전처리")

        # --overwrite 시 구형 가공 파일 전체 삭제 → 스키마 혼재 방지
        if overwrite:
            proc_dir = config.DIRECTORIES['data_processed']
            old_files = _glob.glob(os.path.join(proc_dir, 'processed_*.csv'))
            if old_files:
                for _f in old_files:
                    os.remove(_f)
                logger.info(f"🧹 --overwrite: data_processed 내 {len(old_files)}개 구형 가공 파일 삭제")
        
        logger.info(f"입력 폴더: {config.DIRECTORIES['data']}")
        logger.info(f"출력 폴더: {config.DIRECTORIES['data_processed']}")
        logger.info(f"고급 지표 포함: {config.FEATURE_ENGINEERING['include_advanced']}")
        logger.info(f"메모리 최적화: {config.FEATURE_ENGINEERING['optimize_memory']}")
        
        eng_result = process_all_data(
            input_dir=config.DIRECTORIES['data'],
            output_dir=config.DIRECTORIES['data_processed']
        )
        
        pipeline_results["feature_engineering"] = eng_result

        logger.info("\n📊 피처 엔지니어링 결과:")
        logger.info(f"  ✅ 성공: {eng_result['success']}")
        logger.info(f"  ❌ 실패: {eng_result['failed']}")
        logger.info(f"  🔧 추가된 피처 수: {eng_result['total_features']}")
        logger.info(f"  ⏱️  처리 시간: {eng_result['total_time']:.2f}초")

        total_fe_attempts = eng_result['success'] + eng_result['failed']
        if eng_result['success'] == 0 and total_fe_attempts > 0:
            logger.error(
                "❌ 피처 엔지니어링 전 실패 — ML 훈련에 사용할 파일이 없습니다"
            )
            pipeline_results["success"] = False
            pipeline_results["total_time"] = time.time() - start_time
            return pipeline_results
        
        # ====================================================================
        # 최종 요약
        # ====================================================================
        total_time = time.time() - start_time
        pipeline_results["total_time"] = total_time
        
        print_section("파이프라인 최종 결과 요약")
        
        logger.info("\n✨ 파이프라인 실행 완료!")
        logger.info("\n📊 최종 통계:")
        logger.info(f"  • 수집된 데이터 파일: {collection_result['success']}")
        logger.info(f"  • 처리된 데이터 파일: {eng_result['success']}")
        logger.info(f"  • 추가된 기술 지표: {eng_result['total_features']}")
        logger.info(f"  • 총 소요 시간: {total_time:.2f}초")
        logger.info(f"  • 평균 파일 처리시간: {eng_result['total_time']/max(1, eng_result['success']):.3f}초")
        
        # 다음 단계 안내
        logger.info("\n📚 다음 단계:")
        logger.info(f"  1. {config.DIRECTORIES['data_processed']}에서 처리된 데이터 확인")
        logger.info("  2. 머신러닝 모델 훈련을 위해 데이터 로드")
        logger.info(f"  3. {config.DIRECTORIES['models']}에 모델 저장")
        
        return pipeline_results
        
    except Exception:
        logger.exception("❌ 파이프라인 실행 중 예외 발생")
        pipeline_results["success"] = False
        pipeline_results["total_time"] = time.time() - start_time
        return pipeline_results

def run_collection_only():
    """데이터 수집만 실행"""
    print_header("Coin AI Bot - 데이터 수집 (Option)")

    start_time = time.time()
    target_coins = get_dynamic_target_coins(limit=30)
    obs_coins = [c for c in get_observation_coins() if c not in target_coins]
    collect_coins = target_coins + obs_coins
    if obs_coins:
        logger.info(f"👁 관찰 코인 {len(obs_coins)}개 수집 포함 (차단 중, 회복 감지용): {[c.replace('KRW-','') for c in obs_coins]}")

    result = run_data_pipeline(
        target_coins=collect_coins,
        target_intervals=config.TARGET_INTERVALS,
        count=config.DATA_COLLECTION['count'],
        overwrite=True,  # 수집 전용 실행은 항상 최신 데이터로 갱신
    )
    
    elapsed = time.time() - start_time
    logger.info(f"⏱️  총 소요 시간: {elapsed:.2f}초")
    
    return result

def run_ml_only() -> bool:
    """머신러닝 훈련만 실행. 성공 시 True, 예외 시 False 반환."""
    print_header("Coin AI Bot - 머신러닝 모델 훈련 (Option)")

    start_time = time.time()
    try:
        run_ml_training()
        logger.info(f"⏱️  총 소요 시간: {time.time() - start_time:.2f}초")
        return True
    except Exception:
        logger.exception("❌ ML 훈련 중 예외 발생")
        logger.info(f"⏱️  총 소요 시간: {time.time() - start_time:.2f}초")
        return False

def run_engineering_only():
    """피처 엔지니어링만 실행"""
    print_header("Coin AI Bot - 피처 엔지니어링 (Option)")

    start_time = time.time()
    result = process_all_data(
        input_dir=config.DIRECTORIES['data'],
        output_dir=config.DIRECTORIES['data_processed']
    )
    elapsed = time.time() - start_time

    logger.info(f"  ✅ 성공: {result['success']}")
    logger.info(f"  ❌ 실패: {result['failed']}")
    logger.info(f"  🔧 추가된 피처 수: {result['total_features']}")
    logger.info(f"  ⏱️  총 소요 시간: {elapsed:.2f}초")
    return result

def run_backtest_only(start_date: str = None):
    """백테스팅만 실행"""
    print_header("Coin AI Bot - 실전 환경 백테스팅 (Option)")
    start_time = time.time()
    run_backtest(start_date=start_date)
    elapsed = time.time() - start_time
    logger.info(f"⏱️  총 소요 시간: {elapsed:.2f}초")

def run_stress_only():
    """전체 코인 OOS 구간 스트레스 백테스트"""
    print_header("Coin AI Bot - 전체 코인 스트레스 백테스트 (Option)")
    import glob
    import json
    import datetime

    # 모든 카테고리 모델 디렉토리에서 config 스캔 → max train_cutoff = 공통 OOS 시작점
    cats = getattr(config, "COIN_CATEGORIES", None)
    model_dirs = []
    if cats:
        for cat_info in cats.values():
            for key in ("model_dir", "model_dir_15m"):
                d = cat_info.get(key)
                if d and d not in model_dirs:
                    model_dirs.append(d)
    else:
        model_dirs = [config.DIRECTORIES.get("models", "models")]

    cutoffs = []
    for d in model_dirs:
        for cfg_path in glob.glob(os.path.join(d, "config_*.json")):
            try:
                with open(cfg_path, encoding="utf-8") as f:
                    ts = json.load(f).get("train_cutoff_timestamp", "")
                if ts:
                    cutoffs.append(ts)
            except Exception:
                pass

    if not cutoffs:
        logger.error("❌ 모델 config 없음 — train 먼저 실행")
        return

    common_oos_start = max(cutoffs)          # 공통 OOS 시작 = 가장 늦은 cutoff
    start_date = common_oos_start[:10]
    end_date = datetime.date.today().strftime("%Y-%m-%d")
    logger.info(f"📅 OOS 구간: {start_date} ~ {end_date}  (공통 cutoff: {common_oos_start})")
    start_time = time.time()
    result = run_stress_period_backtest(start=start_date, end=end_date)
    elapsed = time.time() - start_time
    logger.info(f"⏱️  총 소요 시간: {elapsed:.2f}초")
    if result:
        logger.info(f"📊 평균 ROI: {result.get('avg_roi', 0):.2f}% | "
                    f"평균 MDD: {result.get('avg_mdd', 0):.2f}% | "
                    f"평균 Sharpe: {result.get('avg_sharpe', 0):.2f} | "
                    f"총 거래: {result.get('total_trades', 0)}회")

# ============================================================================
# 3. CLI 인터페이스 (argparse — 자동화/cron 환경에서 input() 블로킹 방지)
# ============================================================================

_COMMANDS = {
    "all":          "전체 파이프라인 (수집 + 피처 엔지니어링 + ML + 백테스트)",
    "collect":      "데이터 수집만 실행",
    "engineer":     "피처 엔지니어링만 실행",
    "train":        "머신러닝 훈련만 실행",
    "backtest":     "OOS 백테스팅 (train_cutoff 이후, --start_date 오버라이드 가능)",
    "stress":       "전체 코인 OOS 구간 스트레스 백테스트 (다중 코인 평균)",
    "rolling_wfo":  "롤링 WFO 백테스트 (--full_retrain 플래그로 완전 재학습 가능)",
}

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Coin AI Bot — 데이터 처리 & 머신러닝 & 백테스팅 통합 파이프라인",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(f"  {k:<14} {v}" for k, v in _COMMANDS.items()),
    )
    parser.add_argument(
        "command",
        choices=list(_COMMANDS.keys()),
        help="실행할 단계 선택",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="기존 데이터 파일 덮어쓰기 (all/collect에 적용)",
    )
    parser.add_argument(
        "--start_date",
        default=None,
        metavar="YYYY-MM-DD",
        help="backtest OOS 시작일 수동 지정 (기본: 각 모델 train_cutoff)",
    )
    parser.add_argument(
        "--full_retrain",
        action="store_true",
        default=False,
        help="rolling_wfo 시 모델 가중치 완전 재학습 (기본: threshold 재보정만)",
    )
    return parser

def main():
    parser = _build_parser()
    args = parser.parse_args()

    cmd = args.command
    if cmd == "all":
        result = run_full_pipeline(overwrite=args.overwrite)
        if result["success"]:
            logger.info("전체 파이프라인 완료 — ML 훈련 시작")
            ml_ok = run_ml_only()
            if ml_ok:
                logger.info("ML 훈련 완료 — 백테스팅 시작")
                run_backtest_only()
            else:
                logger.error("❌ ML 훈련 실패 — 백테스팅 스킵")
    elif cmd == "collect":
        run_collection_only()
    elif cmd == "engineer":
        run_engineering_only()
    elif cmd == "train":
        run_ml_only()
    elif cmd == "backtest":
        run_backtest_only(start_date=args.start_date)
    elif cmd == "stress":
        run_stress_only()
    elif cmd == "rolling_wfo":
        start_time = time.time()
        print_header("Coin AI Bot - 롤링 WFO 백테스트")
        result = run_rolling_window_backtest(retrain_weights=args.full_retrain)
        elapsed = time.time() - start_time
        if result:
            logger.info(
                f"  WFO 완료 — 평균 ROI={result.get('avg_roi', 0):.1f}%  "
                f"MDD={result.get('avg_mdd', 0):.1f}%  Sharpe={result.get('avg_sharpe', 0):.2f}"
            )
        logger.info(f"⏱️  총 소요 시간: {elapsed:.2f}초")

    logger.info("작업 완료.")

# ============================================================================
# 4. 실행
# ============================================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("사용자에 의해 중단됨")
        sys.exit(0)
    except Exception:
        logger.exception("예상치 못한 오류")
        sys.exit(1)
