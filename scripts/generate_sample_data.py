"""
Script to generate synthetic sample data CSVs for the Vega Trading Engine.
Run once from the project root: python scripts/generate_sample_data.py
"""
import random
import os
from datetime import date, timedelta, datetime

random.seed(42)


def is_trading_day(d: date) -> bool:
    """Simple weekday filter (Mon–Fri). Does not account for NSE holidays."""
    return d.weekday() < 5


def generate_trading_days(start: date, count: int) -> list[date]:
    days = []
    d = start
    while len(days) < count:
        if is_trading_day(d):
            days.append(d)
        d += timedelta(days=1)
    return days


# ── Daily OHLCV bars ─────────────────────────────────────────────────────────
def generate_daily_bars(start_price: float, trading_days: list[date]) -> list[str]:
    rows = ["timestamp,open,high,low,close,volume"]
    price = start_price
    for d in trading_days:
        gap       = random.uniform(-0.004, 0.004) * price
        open_p    = round(price + gap, 2)
        change    = random.uniform(-0.010, 0.013) * price
        close_p   = round(price + change, 2)
        high_p    = round(max(open_p, close_p) + random.uniform(10, 120), 2)
        low_p     = round(min(open_p, close_p) - random.uniform(10, 100), 2)
        volume    = random.randint(150_000, 420_000)
        rows.append(f"{d},{open_p},{high_p},{low_p},{close_p},{volume}")
        price = close_p
    return rows


# ── Tick data (one trading session) ──────────────────────────────────────────
def generate_ticks(trade_date: date, start_price: float, n_ticks: int) -> list[str]:
    rows = ["timestamp,price,volume"]
    # NSE session: 09:15 to 15:30 IST = 375 minutes
    session_start = datetime(trade_date.year, trade_date.month, trade_date.day, 9, 15, 0)
    price = start_price
    for i in range(n_ticks):
        # Distribute ticks through the session
        seconds_offset = int((i / n_ticks) * 375 * 60)
        ts = session_start + timedelta(seconds=seconds_offset)
        change = random.uniform(-0.0008, 0.0010) * price
        price  = round(max(1.0, price + change), 2)
        volume = random.randint(50, 500)
        rows.append(f"{ts.strftime('%Y-%m-%d %H:%M:%S')},{price},{volume}")
    return rows


# ── Macro data ────────────────────────────────────────────────────────────────
def generate_macro(trading_days: list[date]) -> list[str]:
    rows = ["date,nifty_trend,india_vix,usdinr"]
    vix    = 14.5
    usdinr = 82.5
    trend  = 0.02   # positive = NIFTY above 200d EMA (percentage)
    for d in trading_days:
        trend  = round(trend  + random.uniform(-0.005, 0.006), 4)
        vix    = round(max(10.0, min(30.0, vix    + random.uniform(-0.4, 0.5))), 2)
        usdinr = round(max(78.0, min(87.0, usdinr + random.uniform(-0.15, 0.18))), 2)
        rows.append(f"{d},{trend},{vix},{usdinr}")
    return rows


if __name__ == "__main__":
    trading_days = generate_trading_days(date(2023, 1, 2), count=120)

    os.makedirs("data/market", exist_ok=True)
    os.makedirs("data/macro",  exist_ok=True)

    # Daily bars
    daily_rows = generate_daily_bars(17_500.0, trading_days)
    with open("data/market/NIFTY50_daily.csv", "w") as f:
        f.write("\n".join(daily_rows) + "\n")
    print(f"Written {len(daily_rows)-1} daily bars -> data/market/NIFTY50_daily.csv")

    # Ticks (first trading day only, 60 ticks)
    tick_rows = generate_ticks(trading_days[0], 17_500.0, n_ticks=60)
    with open("data/market/NIFTY50_ticks.csv", "w") as f:
        f.write("\n".join(tick_rows) + "\n")
    print(f"Written {len(tick_rows)-1} ticks      -> data/market/NIFTY50_ticks.csv")

    # Macro
    macro_rows = generate_macro(trading_days)
    with open("data/macro/macro_data.csv", "w") as f:
        f.write("\n".join(macro_rows) + "\n")
    print(f"Written {len(macro_rows)-1} macro rows -> data/macro/macro_data.csv")
