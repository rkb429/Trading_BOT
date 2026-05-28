# Coin AI Bot — Spot & Futures Hybrid Trading System

![Status](https://img.shields.io/badge/Status-In%20Development-red)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![asyncio](https://img.shields.io/badge/async-asyncio-green)
![ML](https://img.shields.io/badge/ML-XGBoost%20%C2%B7%20LightGBM-orange)
![Exchange](https://img.shields.io/badge/Exchange-Upbit%20%7C%20Binance-yellow)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> 업비트(현물) + 바이낸스(선물)를 동시 운용하는 ML 기반 자동 매매 시스템.

설계 원칙:
- **통계적 엄밀성**: Triple Barrier 라벨링으로 Look-ahead bias를 원천 차단하고, WFO(Walk-Forward Optimization)로 과적합을 방지
- **실전 리스크 설계**: 킬스위치 → 서킷브레이커 → AI 게이트까지 8단계 신호 우선순위 체계로 자산 보호 우선
- **자율 운용**: 드리프트 감지 → 재학습 → 롤백을 무인 자동화(MLOps)하여 모델 열화를 자동 교정

---

## 기술 스택

| 영역                  | 기술                                                        |
| --------------------- | ----------------------------------------------------------- |
| ML                    | XGBoost · LightGBM · Scikit-learn (CalibratedClassifierCV) |
| 하이퍼파라미터 최적화 | Optuna (SQLite 기반 study 캐싱)                             |
| 시장 국면 감지        | hmmlearn (HMM) · KMeans Chop 감지                          |
| 피처 엔지니어링       | pandas/numpy 직접 구현 (RSI, MACD, ATR 등) · Triple Barrier 라벨링 |
| 거래소 연동           | pyupbit (WebSocket 내장) · ccxt (Binance FAPI)             |
| 비동기 처리           | asyncio · ccxt.async_support (선물봇 전 구간)               |
| 영속성 레이어         | JSON 파일 (fleet state 원자적 교체, os.replace)             |
| 스케줄링              | schedule (수·일 KST 02:00 재학습)                           |
| 알림                  | python-telegram-bot 21.8                                    |
| 백테스트              | Backtrader (롤링 WFO · 스트레스 구간)                       |
| 설정 검증             | Pydantic v2 · pydantic-settings (런타임 타입 체크)          |
| 로깅                  | loguru (구조화 로깅 · 자동 로테이션)                        |
| 성능                  | orjson (C 기반 고속 JSON 파싱)                              |

---

## 아키텍처

```
OHLCV 수집 → 피처 엔지니어링 → ML 학습 (Optuna 튜닝) → 백테스트 (WFO)
                                                              ↓
드리프트 감지 ←──────────────── 실전 매매 (현물/선물 동시)
     ↓                              ↑
자동 재학습 → 복합 점수 평가 → 롤백 또는 배포
```

---

## 백테스트 결과 (OOS)

> WFO 기반 Out-of-Sample 구간 대표 지표. 스트레스 기간(급락·급등 구간) 포함.

| 지표 | 현물 (B/C 카테고리) | 선물 |
| ---- | ------------------- | ---- |
| Sharpe Ratio | | |
| MDD | | |
| Win Rate | | |
| 테스트 기간 | | |

---

## 주요 기능

### 1. ML 앙상블 + 자동 재학습 (MLOps)
- XGBoost · LightGBM Soft-Voting 앙상블, Optuna로 하이퍼파라미터 최적화
- 코인을 유동성별 A/B/C 카테고리로 분리, 카테고리별 독립 모델 운용 (60m + 15m MTF)
- `payoff_ema < drift_payoff_threshold(기본 1.0)` 봇이 전체의 50% 이상이면 드리프트로 판단 (Tier-1은 Wilcoxon 검정) → 자동 재학습 트리거
- 재학습 전 모델 백업, 복합 점수 `(precision×0.6 + AUC×0.4)`로 롤백 여부 자동 판단

### 2. Triple Barrier 라벨링
- ATR 기반 TP(2×) / SL(0.75×) 배리어 + 타임스탑(16봉) 구성
- 수수료·슬리피지 버퍼(0.3%) 반영, Look-ahead bias 방지를 위한 Purge Gap 적용
- 생성된 라벨로 모델·백테스트·실거래 파라미터 일관성 유지

### 3. 다층 신호 우선순위 게이트 (현물)
```
1. 킬스위치 (BTC 1h 낙폭 ≤ -5%) / UpbitRisk (지갑락업·DAXA 경고)
2. 서킷브레이커 (일손실 -3%, 연속 3패 60분 냉각, MDD -20% 봇 정지)
3. OOS 화이트리스트 (스트레스 백테스트 통과 코인만 진입 허용)
4. 바이낸스 롱 과열 (펀딩비 AND L/S 비율)
5. HMM Bear → Long 차단 / Sideways → 임계값 상향 후 완화 진입
6. KMeans CHOP 감지 → 임계값 +5%p 상향 (하드 차단 아님)
7. (예약 — 미구현)
8. AI 예측: 60m ≥ threshold AND 15m ≥ threshold (MTF AND 게이트)
```

### 4. 포지션 사이징 — Half-Kelly + ATR
- `position_size = min(half_kelly, atr_raw_pct)`, 범위 [5%, 40%] 클리핑
- 시장 충격 방지: 24h 거래대금 대비 주문 상한 `LIQUIDITY_IMPACT_CAP` 적용
- `ai_prob ≥ 0.85` 또는 유동성 상위 코인에만 시장가, 나머지 지정가

### 5. 선물봇 — 완전 비동기 (asyncio)
- Isolated Margin 강제, 최대 레버리지 3배 하드코딩
- HMM BEAR + `P(up) ≤ 0.20` 이중 조건 충족 시에만 SHORT 진입 (과매도 필터)
- `panic_close_all()` 비상 청산 인터페이스 제공

### 6. 백테스트 — Backtrader 기반
- 롤링 윈도우 WFO (train/test 슬라이딩, 창별 임계값 재보정)
- 스트레스 구간 백테스트: ROI > 0%, MDD ≤ 20%, Sharpe ≥ 0, 거래 ≥ 10회 통과 코인만 화이트리스트 등록
- 실거래 로직(트레일링 스탑 공식, payoff_ema 처리)을 백테스트와 동기화

---

## 프로젝트 구조

```
Coin_AI_Bot/
├── main.py                  # 파이프라인 CLI (all|collect|engineer|train|backtest|stress|rolling_wfo)
├── orchestrator.py          # 멀티봇 파견·교체
├── trade_bot.py             # 업비트 현물 트레이딩
├── futures_bot.py           # 바이낸스 선물 트레이딩 (async)
├── data_pipeline.py         # OHLCV 수집
├── feature_engineering.py   # 피처 엔지니어링 + 라벨링
├── machine_learning.py      # ML 학습 파이프라인
├── market_context.py        # 실시간 시장 국면 감지
├── auto_retrain.py          # 드리프트 감지 + 자동 재학습
├── backtest.py              # 백테스트 엔진
├── config.py                # 전역 설정 + 파라미터 검증
└── tests/
    └── test_pipeline.py
```

모델은 코인 카테고리 × 타임프레임별로 독립 저장된다.

| 카테고리             | 대상 코인                    | 디렉토리                       |
| -------------------- | ---------------------------- | ------------------------------ |
| A — 대형 효율 시장   | BTC, ETH                     | 데이터 수집만 (ML 학습 제외)   |
| B — 중형 알트        | XRP, SOL, ADA, DOGE 등       | `models_B/`, `models_B_15m/`   |
| C — 소형 알트        | 화이트리스트 통과 코인        | `models_C/`, `models_C_15m/`   |

---

## 설치

**Python 3.10+** 필요.

```bash
pip install -r requirements.txt
```

> Windows에서 TA-Lib 설치 오류 시: `pip install talib-binary`

프로젝트 루트에 `.env` 파일 생성:

```env
UPBIT_ACCESS_KEY=your_key
UPBIT_SECRET_KEY=your_key
BINANCE_API_KEY=your_key
BINANCE_SECRET_KEY=your_key
TELEGRAM_TOKEN=your_token
TELEGRAM_CHAT_ID=your_id

DRY_RUN=true               # 페이퍼 트레이딩 (기본값 true, 실전 전 반드시 검증)
BOT_INITIAL_CAPITAL_KRW=0  # 0 = 잔고 비례, 양수 = 고정 시드(KRW)
LIQUIDITY_IMPACT_CAP=0.01  # 24h 거래대금 대비 최대 주문 비율
```

---

## 실행

```bash
# 전체 파이프라인 (수집 + 피처 엔지니어링 + ML + 백테스트)
python main.py all

# 데이터 수집만
python main.py collect

# 피처 엔지니어링만
python main.py engineer

# ML 학습만 (A/B/C 카테고리 × 60m/15m 순차)
python main.py train

# OOS 백테스트
python main.py backtest

# 스트레스 백테스트 (전체 코인 OOS)
python main.py stress

# 롤링 WFO 백테스트
python main.py rolling_wfo

# 실전 매매 (DRY_RUN=false 설정 후)
python orchestrator.py

# 자동 재학습 데몬 (수·일 KST 02:00)
python auto_retrain.py
```

---

## 면책 고지

본 프로그램은 포트폴리오 및 연구 목적으로 공개되었습니다.  
실제 투자에 사용할 경우 모든 손익에 대한 책임은 사용자 본인에게 있습니다.  
반드시 `DRY_RUN=true` 환경에서 충분히 검증한 후 운용하십시오.
