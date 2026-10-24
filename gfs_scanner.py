#!/usr/bin/env python3
"""
GFS Scanner - daily morning scan for NSE stocks
================================================
Conditions
  1. Monthly RSI(14) > 60
  2. Weekly  RSI(14) > 60
  3. Latest COMPLETED daily candle: RSI(14) between 38 and 45 (inclusive)
     and close > open (green candle)

Trade levels (from the signal candle)
  Entry  = signal candle High
  Stop   = signal candle Low
  Target = Entry + 3 x (Entry - Stop)

Install:   pip install yfinance pandas requests
Run:       python gfs_scanner.py                      # Nifty 500 universe
           python gfs_scanner.py --symbols my.txt     # your own list (one symbol per line)
           python gfs_scanner.py --htf-completed-only # use only fully closed weekly/monthly candles

Notes
  * Weekly/monthly candles are built from daily data. By default the
    still-forming weekly/monthly candle is INCLUDED (same RSI you see on a chart).
    Use --htf-completed-only if you want closed candles only.
  * If run during market hours, today's partial daily candle is dropped
    automatically, so the signal candle is always a completed one.
  * RSI uses Wilder smoothing (same as TradingView / most charting platforms).
"""

import argparse
import io
import json
import sys
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yfinance as yf

# ----------------------------- SETTINGS ------------------------------------
RSI_LEN = 14
MONTHLY_RSI_MIN = 60
WEEKLY_RSI_MIN = 60
DAILY_RSI_LOW = 38
DAILY_RSI_HIGH = 45
REWARD_MULTIPLE = 3          # target = entry + 3 x risk
HISTORY_PERIOD = "10y"       # needed so monthly RSI(14) is well-seeded
MIN_DAILY_BARS = 800         # ~3+ years minimum for stable monthly RSI
CHUNK_SIZE = 50              # tickers per download batch
MARKET_CLOSE_IST = time(15, 45)   # after this, today's daily bar is treated as complete

NIFTY500_URL = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
IST = ZoneInfo("Asia/Kolkata")
# ---------------------------------------------------------------------------


def rsi(close: pd.Series, length: int = RSI_LEN) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss
    out = 100 - 100 / (1 + rs)
    out[(avg_loss == 0) & (avg_gain > 0)] = 100.0
    return out


def load_universe(symbols_file: str | None) -> list[str]:
    """Return list of NSE symbols (without .NS)."""
    if symbols_file:
        syms = [s.strip().upper() for s in Path(symbols_file).read_text().splitlines()]
        return [s.replace(".NS", "") for s in syms if s and not s.startswith("#")]

    cache = Path("nifty500_cache.csv")
    try:
        r = requests.get(NIFTY500_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        r.raise_for_status()
        cache.write_bytes(r.content)
        df = pd.read_csv(io.BytesIO(r.content))
    except Exception as e:
        if cache.exists():
            print(f"[warn] Could not download Nifty 500 list ({e}); using cached copy.")
            df = pd.read_csv(cache)
        else:
            sys.exit(f"Could not get Nifty 500 list ({e}). Pass --symbols yourfile.txt instead.")
    return df["Symbol"].dropna().astype(str).str.strip().tolist()


def resample_ohlc(daily: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Build weekly ('W-FRI') or monthly ('M') OHLC from daily data."""
    period = daily.index.to_period(freq)
    g = daily.groupby(period)
    out = pd.DataFrame({
        "Open": g["Open"].first(),
        "High": g["High"].max(),
        "Low": g["Low"].min(),
        "Close": g["Close"].last(),
    })
    return out


def rsi_of_timeframe(daily: pd.DataFrame, freq: str, completed_only: bool) -> float | None:
    bars = resample_ohlc(daily, freq)
    if completed_only and len(bars) > 0:
        last_period_end = bars.index[-1].end_time.date()
        if daily.index[-1].date() < last_period_end:   # last candle still forming
            bars = bars.iloc[:-1]
    r = rsi(bars["Close"]).dropna()
    return float(r.iloc[-1]) if len(r) else None


def evaluate(daily: pd.DataFrame, htf_completed_only: bool) -> dict | None:
    """Return a result dict if the stock passes all GFS conditions, else None."""
    daily = daily.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    if len(daily) < MIN_DAILY_BARS:
        return None

    # drop today's partial candle if market is still open / data not final
    now = datetime.now(IST)
    if daily.index[-1].date() == now.date() and now.time() < MARKET_CLOSE_IST:
        daily = daily.iloc[:-1]
        if len(daily) < MIN_DAILY_BARS:
            return None

    # --- cheapest filter first: daily candle ---
    d_rsi_series = rsi(daily["Close"])
    d_rsi = d_rsi_series.iloc[-1]
    sig = daily.iloc[-1]
    if pd.isna(d_rsi) or not (DAILY_RSI_LOW <= d_rsi <= DAILY_RSI_HIGH):
        return None
    if not sig["Close"] > sig["Open"]:
        return None

    # --- higher timeframes ---
    w_rsi = rsi_of_timeframe(daily, "W-FRI", htf_completed_only)
    if w_rsi is None or w_rsi <= WEEKLY_RSI_MIN:
        return None
    m_rsi = rsi_of_timeframe(daily, "M", htf_completed_only)
    if m_rsi is None or m_rsi <= MONTHLY_RSI_MIN:
        return None

    entry = float(sig["High"])
    stop = float(sig["Low"])
    risk = entry - stop
    if risk <= 0:
        return None
    target = entry + REWARD_MULTIPLE * risk

    return {
        "SignalDate": daily.index[-1].date().isoformat(),
        "Close": round(float(sig["Close"]), 2),
        "MonthlyRSI": round(m_rsi, 1),
        "WeeklyRSI": round(w_rsi, 1),
        "DailyRSI": round(float(d_rsi), 1),
        "Entry": round(entry, 2),
        "StopLoss": round(stop, 2),
        "Target": round(target, 2),
        "Risk/Share": round(risk, 2),
        "Risk%": round(risk / entry * 100, 2),
    }


def download_chunk(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Download daily OHLC for a batch and split into per-ticker frames."""
    raw = yf.download(
        tickers, period=HISTORY_PERIOD, interval="1d", auto_adjust=True,
        group_by="ticker", threads=True, progress=False,
    )
    frames = {}
    if raw is None or raw.empty:
        return frames
    if not isinstance(raw.columns, pd.MultiIndex):       # single ticker case
        frames[tickers[0]] = raw
        return frames
    available = set(raw.columns.get_level_values(0))
    for t in tickers:
        if t in available:
            df = raw[t].dropna(how="all")
            if not df.empty:
                frames[t] = df
    return frames


def main():
    ap = argparse.ArgumentParser(description="GFS Scanner (NSE)")
    ap.add_argument("--symbols", help="text file with one NSE symbol per line (default: Nifty 500)")
    ap.add_argument("--htf-completed-only", action="store_true",
                    help="use only fully closed weekly/monthly candles")
    ap.add_argument("--out", default="gfs_results", help="output folder for CSV")
    ap.add_argument("--json", default="docs/results.json", help="JSON file read by the dashboard")
    args = ap.parse_args()

    symbols = load_universe(args.symbols)
    tickers = [f"{s}.NS" for s in symbols]
    print(f"GFS Scanner | {datetime.now(IST):%d-%b-%Y %H:%M IST} | scanning {len(tickers)} stocks...")

    results, failed = [], 0
    for i in range(0, len(tickers), CHUNK_SIZE):
        batch = tickers[i:i + CHUNK_SIZE]
        try:
            frames = download_chunk(batch)
        except Exception as e:
            print(f"[warn] batch {i // CHUNK_SIZE + 1} failed: {e}")
            failed += len(batch)
            continue
        failed += len(batch) - len(frames)
        for t, df in frames.items():
            try:
                res = evaluate(df, args.htf_completed_only)
            except Exception as e:
                print(f"[warn] {t}: {e}")
                continue
            if res:
                results.append({"Symbol": t.replace(".NS", ""), **res})
        print(f"  processed {min(i + CHUNK_SIZE, len(tickers))}/{len(tickers)}  |  hits so far: {len(results)}")

    jpath = Path(args.json)
    jpath.parent.mkdir(parents=True, exist_ok=True)
    jpath.write_text(json.dumps({
        "generated": datetime.now(IST).isoformat(timespec="minutes"),
        "universe": len(tickers), "skipped": failed, "count": len(results),
        "stocks": sorted(results, key=lambda r: r["Risk%"]),
    }, indent=1))
    print(f"Dashboard data: {jpath.resolve()}")

    print()
    if not results:
        print("No stocks match all GFS conditions today.")
    else:
        df = pd.DataFrame(results).sort_values("Risk%").reset_index(drop=True)
        pd.set_option("display.width", 200)
        pd.set_option("display.max_columns", None)
        print(f"{len(df)} stock(s) match:\n")
        print(df.to_string(index=False))

        out_dir = Path(args.out)
        out_dir.mkdir(exist_ok=True)
        path = out_dir / f"gfs_{datetime.now(IST):%Y%m%d}.csv"
        df.to_csv(path, index=False)
        print(f"\nSaved: {path.resolve()}")

    if failed:
        print(f"\n({failed} symbols returned no/insufficient data and were skipped.)")


if __name__ == "__main__":
    main()
