import os
import yfinance as yf
import pandas as pd
import numpy as np
import random
import json

# Get path relative to the script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, "../data/train.jsonl"))


def _to_scalar(x):
    if isinstance(x, pd.Series):
        if len(x) == 0:
            return float('nan')
        return float(x.iloc[0])
    if isinstance(x, np.ndarray):
        if x.size == 0:
            return float('nan')
        return float(x.flatten()[0])
    try:
        return float(x)
    except Exception:
        return float('nan')

TICKERS = [
    "AAPL",   # apple
    "MSFT",   # Microsoft
    "AMZN",   # Amazon
    "GOOGL",  # Alphabet
    "NVDA",   # NVIDIA
    "TSLA",   # Tesla
    "META",    # Meta
    "AMD",    # AMD
    "PLTR"    # Palantir
]

#yfinance로 가격 데이터 불러오기
def fetch_price_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df = df.reset_index()
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    df.dropna(inplace=True)
    return df

#market_state 자동 생성 로직
def detect_trend(price, ma_short, ma_long, threshold=0.01):
    p = _to_scalar(price)
    s = _to_scalar(ma_short)
    l = _to_scalar(ma_long)

    if pd.isna(p) or pd.isna(s) or pd.isna(l):
        return "sideways"

    if (p > s) and (s > l * (1 + threshold)):
        return "bull"
    elif (p < s) and (s < l * (1 - threshold)):
        return "bear"
    else:
        return "sideways"

#risk_level 자동 생성 로직    
def detect_risk(volatility):
    v = _to_scalar(volatility)
    if pd.isna(v):
        return "medium"
    if v < 0.01:
        return "low"
    elif v < 0.02:
        return "medium"
    elif v < 0.035:
        return "high"
    else:
        return "panic"

#liquidity 간단 추정(거래량 기반)   
def detect_liquidity(volume, volume_ma, short_term_trend):
    """
    volume        : 오늘 거래량
    volume_ma     : 20일 평균 거래량
    short_term_trend : bull / bear / sideways
    """

    # 유동성 풍부
    if volume > volume_ma * 1.2:
        return "loose"

    # 거래량이 줄어든 상태
    if volume < volume_ma * 0.8:
        if short_term_trend == "bull":
            return "tight_but_easing"
        else:
            return "tight_and_rising"

    # 나머지는 중립
    return "neutral"

#market_state 생성 함수    
def generate_market_state(df, idx):
    close = _to_scalar(df.loc[idx, "Close"])

    ma5 = _to_scalar(df["Close"].rolling(5).mean().iloc[idx])
    ma20 = _to_scalar(df["Close"].rolling(20).mean().iloc[idx])
    ma60 = _to_scalar(df["Close"].rolling(60).mean().iloc[idx])

    returns = _to_scalar(df["Close"].pct_change().rolling(20).std().iloc[idx])

    volume = _to_scalar(df.loc[idx, "Volume"])
    volume_ma = _to_scalar(df["Volume"].rolling(20).mean().iloc[idx])

    short_trend = detect_trend(close, ma5, ma20)

    market_state = {
        "short_term": short_trend,
        "mid_term": detect_trend(close, ma20, ma60),
        "long_term": detect_trend(close, ma60, ma60),
        "risk_level": detect_risk(returns),
        "risk_score": int(min(6, returns * 200 if not pd.isna(returns) else 0)),
        "liquidity": detect_liquidity(volume, volume_ma, short_trend)
    }

    return market_state

#뉴스 상태
def generate_news_state():
    return {
        "sentiment_score": round(random.uniform(-0.3, 0.3), 2),
        "market_impact": random.randint(1, 4),
        "conflict_check": random.choice([True, False])
    }

#portfolio_state
class PortfolioState:
    def __init__(self):
        self.position_ratio = 0.0
        self.avg_entry_price = None
        self.holding_days = 0
        self.unrealized_pnl_pct = 0.0

    def update(self, close_price, target_ratio):
        if target_ratio == 0.0:
            self.__init__()
            return

        if self.position_ratio == 0.0:
            self.position_ratio = target_ratio
            self.avg_entry_price = close_price
            self.holding_days = 1
            return

        if target_ratio != self.position_ratio:
            self.avg_entry_price = (
                self.avg_entry_price * self.position_ratio +
                close_price * (target_ratio - self.position_ratio)
            ) / target_ratio
            self.position_ratio = target_ratio

        self.holding_days += 1
        self.unrealized_pnl_pct = (
            close_price - self.avg_entry_price
        ) / self.avg_entry_price

def trading_policy(market_state, portfolio):
    current = portfolio.position_ratio
    target = current

    if market_state["risk_level"] in ["panic", "high"]:
        target = 0.0

    elif market_state["short_term"] == "bull" and \
         market_state["liquidity"] in ["loose", "neutral"]:
        target = min(current + 0.2, 1.0)

    elif market_state["short_term"] == "bear":
        target = max(current - 0.2, 0.0)

    elif market_state["short_term"] == "sideways":
        target = current

    return {
        "position_target": round(target, 2),
        "delta_position": round(target - current, 2),
        "rebalance": abs(target - current) > 0.01
    }


def generate_multi_ticker_training_data(
    tickers,
    start="2014-01-01",
    end="2024-12-31",
    output_path=DEFAULT_OUTPUT_PATH
):
    with open(output_path, "w") as f:
        for ticker in tickers:
            print(f"Processing {ticker}")
            df = fetch_price_data(ticker, start, end)
            portfolio = PortfolioState()

            for i in range(200, len(df)):
                market_state = generate_market_state(df, i)
                news_state = generate_news_state()
                close_price = df.loc[i, "Close"]

                action = trading_policy(market_state, portfolio)
                portfolio.update(close_price, action["position_target"])

                unreal = _to_scalar(portfolio.unrealized_pnl_pct)

                record = {
                    "instruction": "You are a trading policy model. Decide today's position size.",
                    "input": {
                        "ticker": ticker,
                        "market_state": market_state,
                        "news_state": news_state,
                        "portfolio_state": {
                            "position_ratio": round(_to_scalar(portfolio.position_ratio), 2),
                            "holding_days": portfolio.holding_days,
                            "unrealized_pnl_pct": round(unreal, 4) if not pd.isna(unreal) else 0.0
                        }
                    },
                    "output": action
                }

                f.write(json.dumps(record) + "\n")


generate_multi_ticker_training_data(
    tickers=TICKERS,
    start="2014-01-01",
    end="2024-12-31"
)