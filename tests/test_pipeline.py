"""
tests/test_pipeline.py

Coin AI Bot 핵심 모듈 단위 테스트.
외부 API 의존성은 unittest.mock으로 격리합니다.

실행:
    python -m pytest tests/ -v
    python -m unittest tests/test_pipeline.py -v
"""

import sys
import os
import pickle
import time
import threading
import unittest
from unittest.mock import MagicMock, patch

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 핵심 데이터 과학 의존성: 없으면 해당 테스트를 skip
try:
    import pandas as pd
    import numpy as np
    _HAS_NUMPY_PANDAS = True
except ImportError:
    _HAS_NUMPY_PANDAS = False

def _require_numpy_pandas(test_func):
    """pandas/numpy 미설치 환경에서 테스트를 조건부 스킵."""
    return unittest.skipUnless(_HAS_NUMPY_PANDAS, "pandas/numpy 미설치")(test_func)

# ============================================================================
# 헬퍼: 최소 OHLCV DataFrame 생성
# ============================================================================

def _make_ohlcv(n: int = 200):
    """단위 테스트용 더미 OHLCV 데이터프레임. pandas/numpy 없으면 None 반환."""
    if not _HAS_NUMPY_PANDAS:
        return None
    np.random.seed(42)
    close = 50_000_000 + np.cumsum(np.random.randn(n) * 100_000)
    high  = close * (1 + np.abs(np.random.randn(n) * 0.005))
    low   = close * (1 - np.abs(np.random.randn(n) * 0.005))
    open_ = close * (1 + np.random.randn(n) * 0.002)
    vol   = np.abs(np.random.randn(n) * 1e10) + 1e9
    idx   = pd.date_range("2024-01-01", periods=n, freq="15min")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=idx,
    )


# ============================================================================
# 1. config.validate_config 테스트
# ============================================================================

class TestConfigValidation(unittest.TestCase):
    """config.validate_config() 정합성 검사 테스트."""

    def setUp(self):
        try:
            import config
            self.config = config
        except Exception as e:
            self.skipTest(f"config 모듈 로드 불가: {e}")

    def test_valid_default_config_passes(self):
        """기본 config는 검증을 통과해야 한다."""
        try:
            self.config.validate_config()
        except ValueError as e:
            self.fail(f"기본 config가 검증 실패: {e}")

    def test_invalid_tp_less_than_sl_raises(self):
        """atr_tp_mult < atr_sl_mult 이면 ValueError."""
        original_tp = self.config.LABELING["atr_tp_mult"]
        original_sl = self.config.LABELING["atr_sl_mult"]
        try:
            self.config.LABELING["atr_tp_mult"] = 1.0
            self.config.LABELING["atr_sl_mult"] = 2.0
            with self.assertRaises(ValueError):
                self.config.validate_config()
        finally:
            self.config.LABELING["atr_tp_mult"] = original_tp
            self.config.LABELING["atr_sl_mult"] = original_sl

    def test_invalid_sma_short_ge_long_raises(self):
        """SMA_SHORT >= SMA_LONG 이면 ValueError."""
        orig_short = self.config.TECHNICAL_INDICATORS["SMA_SHORT"]
        orig_long  = self.config.TECHNICAL_INDICATORS["SMA_LONG"]
        try:
            self.config.TECHNICAL_INDICATORS["SMA_SHORT"] = 30
            self.config.TECHNICAL_INDICATORS["SMA_LONG"]  = 10
            with self.assertRaises(ValueError):
                self.config.validate_config()
        finally:
            self.config.TECHNICAL_INDICATORS["SMA_SHORT"] = orig_short
            self.config.TECHNICAL_INDICATORS["SMA_LONG"]  = orig_long

    def test_invalid_ensemble_weight_out_of_range_raises(self):
        """lgbm_weight <= 0 또는 >= 1 이면 ValueError."""
        orig = self.config.ENSEMBLE["lgbm_weight"]
        try:
            self.config.ENSEMBLE["lgbm_weight"] = 1.5
            with self.assertRaises(ValueError):
                self.config.validate_config()
        finally:
            self.config.ENSEMBLE["lgbm_weight"] = orig

    def test_invalid_fear_greed_order_raises(self):
        """greed >= extreme_greed 이면 ValueError."""
        orig_g  = self.config.FEAR_GREED["greed"]
        orig_eg = self.config.FEAR_GREED["extreme_greed"]
        try:
            self.config.FEAR_GREED["greed"]         = 90
            self.config.FEAR_GREED["extreme_greed"] = 80
            with self.assertRaises(ValueError):
                self.config.validate_config()
        finally:
            self.config.FEAR_GREED["greed"]         = orig_g
            self.config.FEAR_GREED["extreme_greed"] = orig_eg


# ============================================================================
# 2. data_pipeline — 429 Rate Limit 백오프 테스트
# ============================================================================

class TestRetryDecorator(unittest.TestCase):
    """retry_on_error 데코레이터의 429 처리 로직 테스트."""

    def setUp(self):
        try:
            import data_pipeline
            self.dp = data_pipeline
        except Exception as e:
            self.skipTest(f"data_pipeline 모듈 로드 불가: {e}")

    def test_get_retry_after_parses_header(self):
        """Retry-After 헤더 값을 올바르게 파싱해야 한다."""
        mock_resp = MagicMock()
        mock_resp.headers = {"Retry-After": "45"}
        result = self.dp._get_retry_after(mock_resp)
        self.assertEqual(result, 45.0)

    def test_get_retry_after_caps_at_max(self):
        """Retry-After 가 상한(300초)을 초과하면 상한으로 클리핑."""
        mock_resp = MagicMock()
        mock_resp.headers = {"Retry-After": "9999"}
        result = self.dp._get_retry_after(mock_resp)
        self.assertEqual(result, self.dp._RATE_LIMIT_MAX_DELAY)

    def test_get_retry_after_none_response_returns_default(self):
        """응답 없을 때 기본값 반환."""
        result = self.dp._get_retry_after(None)
        self.assertEqual(result, self.dp._RATE_LIMIT_BASE_DELAY)

    def test_retry_succeeds_after_transient_failure(self):
        """일시적 실패 후 재시도 성공."""
        call_count = {"n": 0}

        @self.dp.retry_on_error(max_retries=3, initial_delay=0)
        def flaky():
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise ConnectionError("일시 오류")
            return "success"

        result = flaky()
        self.assertEqual(result, "success")
        self.assertEqual(call_count["n"], 3)

    def test_non_retryable_exception_raises_immediately(self):
        """ValueError는 재시도 없이 즉시 발생."""
        call_count = {"n": 0}

        @self.dp.retry_on_error(max_retries=3, initial_delay=0)
        def bad():
            call_count["n"] += 1
            raise ValueError("잘못된 인자")

        with self.assertRaises(ValueError):
            bad()
        self.assertEqual(call_count["n"], 1)

    def test_rate_limit_string_in_exception_triggers_long_wait(self):
        """예외 메시지에 '429' 포함 시 rate limit 대기 경로 진입."""
        sleep_calls = []

        @self.dp.retry_on_error(max_retries=2, initial_delay=0)
        def rate_limited():
            raise Exception("HTTP Error 429: Too Many Requests")

        with patch("data_pipeline.time") as mock_time:
            mock_time.sleep = lambda s: sleep_calls.append(s)
            try:
                rate_limited()
            except Exception:
                pass

        # 대기 시간이 일반 backoff(0~1초)보다 길어야 한다
        self.assertTrue(
            any(s >= self.dp._RATE_LIMIT_BASE_DELAY for s in sleep_calls),
            f"Rate limit 대기 미적용: sleep 호출값={sleep_calls}"
        )


# ============================================================================
# 3. market_context — NewsSentimentAnalyzer 테스트
# ============================================================================

@unittest.skipUnless(_HAS_NUMPY_PANDAS, "pandas/numpy 미설치")
class TestNewsSentimentAnalyzer(unittest.TestCase):
    """NewsSentimentAnalyzer 킬 스위치 로직 테스트."""

    def _make_analyzer(self, threshold=-0.05):
        from market_context import NewsSentimentAnalyzer
        return NewsSentimentAnalyzer(cache_ttl=300, btc_drop_threshold=threshold)

    def _make_ohlcv_2row(self, prev, curr):
        """2행 더미 OHLCV (prev → curr 종가)."""
        idx = pd.date_range("2024-01-01", periods=2, freq="1h")
        return pd.DataFrame({"close": [prev, curr]}, index=idx)

    def test_kill_switch_activates_on_large_drop(self):
        """BTC 5% 이상 낙폭 시 킬 스위치 ON."""
        ana = self._make_analyzer(threshold=-0.05)
        df = self._make_ohlcv_2row(prev=50_000_000, curr=47_000_000)  # -6%

        with patch("market_context.pyupbit.get_ohlcv", return_value=df):
            score, kill = ana.fetch()

        self.assertTrue(kill)
        self.assertLess(score, -0.05)

    def test_kill_switch_off_on_small_drop(self):
        """BTC 1% 하락은 킬 스위치 OFF."""
        ana = self._make_analyzer(threshold=-0.05)
        df = self._make_ohlcv_2row(prev=50_000_000, curr=49_500_000)  # -1%

        with patch("market_context.pyupbit.get_ohlcv", return_value=df):
            score, kill = ana.fetch()

        self.assertFalse(kill)

    def test_cache_prevents_duplicate_api_calls(self):
        """캐시 TTL 내 재호출은 API를 다시 조회하지 않는다."""
        ana = self._make_analyzer()
        df = self._make_ohlcv_2row(prev=50_000_000, curr=49_000_000)

        with patch("market_context.pyupbit.get_ohlcv", return_value=df) as mock_get:
            ana.fetch()
            ana.fetch()  # 두 번째 호출 — 캐시 hit

        self.assertEqual(mock_get.call_count, 1)

    def test_api_failure_returns_previous_state(self):
        """API 오류 시 이전 상태(킬 스위치 OFF)를 유지한다."""
        ana = self._make_analyzer()

        with patch("market_context.pyupbit.get_ohlcv", side_effect=Exception("네트워크 오류")):
            score, kill = ana.fetch()

        self.assertFalse(kill)

    def test_no_race_condition_under_concurrent_calls(self):
        """멀티 스레드에서 동시 호출 시 크래시 없이 동작."""
        ana = self._make_analyzer()
        df = self._make_ohlcv_2row(prev=50_000_000, curr=49_000_000)
        results = []

        def call():
            with patch("market_context.pyupbit.get_ohlcv", return_value=df):
                results.append(ana.fetch())

        threads = [threading.Thread(target=call) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(results), 10)
        for score, kill in results:
            self.assertIsInstance(score, float)
            self.assertIsInstance(kill, bool)


# ============================================================================
# 4. market_context — Pickle 스키마 검증 테스트
# ============================================================================

class TestPickleSchemaValidation(unittest.TestCase):
    """HMMRegimeDetector / KMeansRegimeDetector pickle 로드 스키마 검증."""

    def setUp(self):
        try:
            import market_context  # noqa: F401
            self._market_context_available = True
        except ImportError as e:
            self._market_context_available = False
            self._skip_reason = str(e)

    def _skip_if_unavailable(self):
        if not self._market_context_available:
            self.skipTest(f"market_context 로드 불가: {self._skip_reason}")

    def _make_hmm(self):
        from market_context import HMMRegimeDetector
        with patch("market_context.HMMRegimeDetector._load_persisted_model"):
            det = HMMRegimeDetector.__new__(HMMRegimeDetector)
            det._model = None
            det._last_train = 0.0
            det._lock = threading.Lock()
            det._model_path = "nonexistent.pkl"
            return det

    def _make_kmeans(self):
        from market_context import KMeansRegimeDetector
        with patch("market_context.KMeansRegimeDetector._load_persisted_model"):
            det = KMeansRegimeDetector.__new__(KMeansRegimeDetector)
            det._model = None
            det._last_train = 0.0
            det._lock = threading.Lock()
            return det

    def _write_pickle(self, path, data):
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def test_hmm_rejects_missing_model_key(self, tmp_path=None):
        """'model' 키 없는 pickle 파일은 무시해야 한다."""
        self._skip_if_unavailable()
        import tempfile, os
        det = self._make_hmm()
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            self._write_pickle(path, {"last_train": 12345.0})  # model 키 없음
            det._model_path = path
            det._load_persisted_model()
            self.assertIsNone(det._model)  # 로드 거부
        finally:
            os.unlink(path)

    def test_hmm_rejects_non_dict_payload(self):
        """dict가 아닌 파일은 무시해야 한다."""
        self._skip_if_unavailable()
        import tempfile, os
        det = self._make_hmm()
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            self._write_pickle(path, ["not", "a", "dict"])
            det._model_path = path
            det._load_persisted_model()
            self.assertIsNone(det._model)
        finally:
            os.unlink(path)

    def test_hmm_loads_valid_payload(self):
        """올바른 스키마는 정상 로드되어야 한다."""
        self._skip_if_unavailable()
        import tempfile, os
        det = self._make_hmm()
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            dummy_model = MagicMock()
            self._write_pickle(path, {"model": dummy_model, "last_train": 9999.0})
            det._model_path = path
            det._load_persisted_model()
            self.assertEqual(det._last_train, 9999.0)
        finally:
            os.unlink(path)

    def test_kmeans_rejects_bad_schema(self):
        """KMeans도 스키마 검증 적용."""
        self._skip_if_unavailable()
        import tempfile, os
        det = self._make_kmeans()
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            self._write_pickle(path, {"wrong_key": "value"})
            det._MODEL_PATH = path
            det._load_persisted_model()
            self.assertIsNone(det._model)
        finally:
            os.unlink(path)


# ============================================================================
# 5. feature_engineering — add_technical_indicators 기본 동작 테스트
# ============================================================================

@unittest.skipUnless(_HAS_NUMPY_PANDAS, "pandas/numpy 미설치")
class TestFeatureEngineering(unittest.TestCase):
    """add_technical_indicators 핵심 동작 테스트."""

    @classmethod
    def setUpClass(cls):
        try:
            from feature_engineering import add_technical_indicators
            cls.fn = staticmethod(add_technical_indicators)
            cls.available = True
        except ImportError:
            cls.available = False

    def _skip_if_unavailable(self):
        if not self.available:
            self.skipTest("feature_engineering 모듈 로드 불가 (의존성 미설치)")

    def test_returns_none_for_empty_df(self):
        self._skip_if_unavailable()
        result, stats = self.fn(pd.DataFrame())
        self.assertIsNone(result)

    def test_returns_none_for_missing_columns(self):
        self._skip_if_unavailable()
        df = pd.DataFrame({"close": [1, 2, 3]})
        result, stats = self.fn(df)
        self.assertIsNone(result)

    def test_produces_expected_feature_columns(self):
        self._skip_if_unavailable()
        df = _make_ohlcv(200)
        df["timestamp"] = df.index
        df["coin"] = "BTC"
        result, stats = self.fn(df)
        if result is None:
            self.skipTest("TA-Lib 미설치로 피처 생성 불가")
        expected = ["ATR_Ratio", "Volume_Surge", "BB_Width", "BB_Position"]
        for col in expected:
            self.assertIn(col, result.columns, f"피처 누락: {col}")

    def test_output_row_count_close_to_input(self):
        self._skip_if_unavailable()
        df = _make_ohlcv(200)
        df["timestamp"] = df.index
        result, stats = self.fn(df)
        if result is None:
            self.skipTest("TA-Lib 미설치")
        self.assertGreater(len(result), 100)

    def test_no_infinite_values_in_output(self):
        self._skip_if_unavailable()
        df = _make_ohlcv(200)
        df["timestamp"] = df.index
        result, stats = self.fn(df)
        if result is None:
            self.skipTest("TA-Lib 미설치")
        numeric = result.select_dtypes(include=[np.number])
        has_inf = np.isinf(numeric.values).any()
        self.assertFalse(has_inf, "출력에 무한값(inf) 포함")


# ============================================================================
# 6. main.py — argparse CLI 테스트
# ============================================================================

class TestMainArgparse(unittest.TestCase):
    """main.py argparse 인터페이스 테스트."""

    def setUp(self):
        try:
            # main.py 임포트 시 부작용 방지를 위해 의존 모듈 모킹
            self.patches = [
                patch("data_pipeline.run_data_pipeline", return_value={}),
                patch("feature_engineering.process_all_data", return_value={}),
            ]
            for p in self.patches:
                p.start()
        except Exception as e:
            self.skipTest(f"main 모듈 로드 불가: {e}")

    def tearDown(self):
        for p in getattr(self, "patches", []):
            try:
                p.stop()
            except Exception:
                pass

    def _get_parser(self):
        import main
        return main._build_parser()

    def test_all_command_parsed(self):
        parser = self._get_parser()
        args = parser.parse_args(["all"])
        self.assertEqual(args.command, "all")
        self.assertFalse(args.overwrite)

    def test_all_with_overwrite_flag(self):
        parser = self._get_parser()
        args = parser.parse_args(["all", "--overwrite"])
        self.assertTrue(args.overwrite)

    def test_each_command_valid(self):
        parser = self._get_parser()
        for cmd in ["collect", "engineer", "train", "backtest"]:
            args = parser.parse_args([cmd])
            self.assertEqual(args.command, cmd)

    def test_invalid_command_raises_system_exit(self):
        parser = self._get_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["invalid_command"])


# ============================================================================
# 실행
# ============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
