import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests


BINANCE_URL = "https://api.binance.com/api/v3/klines"
COLS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "num_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
]
OUT_COLS = [
    "symbol",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "num_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
]


def _to_ms(date_like):
    if date_like is None:
        return None
    return int(pd.Timestamp(date_like, tz="UTC").timestamp() * 1000)


def _download_symbol(symbol, start_date, end_date):
    rows = []
    start_ms = _to_ms(start_date)
    end_ms = _to_ms(end_date)
    while True:
        params = {"symbol": symbol, "interval": "1d", "limit": 1000, "startTime": start_ms}
        if end_ms is not None:
            params["endTime"] = end_ms
        r = requests.get(BINANCE_URL, params=params, timeout=30)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        next_ms = int(batch[-1][0]) + 24 * 60 * 60 * 1000
        if next_ms == start_ms or (end_ms is not None and next_ms > end_ms):
            break
        start_ms = next_ms
        if len(batch) < 1000:
            break
    if not rows:
        return pd.DataFrame(columns=OUT_COLS)
    df = pd.DataFrame(rows, columns=COLS)
    df["symbol"] = symbol
    df["date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_localize(None).dt.normalize()
    return df[OUT_COLS]


def _clean(df):
    if df.empty:
        return df
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    num_cols = [c for c in OUT_COLS if c not in {"symbol", "date"}]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["symbol", "date", "open", "high", "low", "close", "volume"])
    df = df[(df["close"] > 0) & (df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["volume"] >= 0)]
    df = df.drop_duplicates(["symbol", "date"]).sort_values(["symbol", "date"])
    return df[OUT_COLS].reset_index(drop=True)


def load_or_download_binance_daily(symbols, start_date, end_date, raw_dir, use_cached=True, allow_download=True):
    raw = Path(raw_dir)
    raw.mkdir(parents=True, exist_ok=True)
    frames = []
    for symbol in symbols:
        path = raw / f"{symbol}.csv"
        df = None
        if use_cached and path.exists():
            try:
                df = pd.read_csv(path, parse_dates=["date"])
                logging.info("Loaded cached %s (%s rows)", symbol, len(df))
            except Exception as exc:
                logging.warning("Could not load cache for %s: %s", symbol, exc)
        if df is None and not allow_download:
            logging.warning("No cached CSV for %s and downloads are disabled", symbol)
            continue
        if df is None:
            try:
                df = _download_symbol(symbol, start_date, end_date)
                if not df.empty:
                    df.to_csv(path, index=False)
                    logging.info("Downloaded %s (%s rows)", symbol, len(df))
                else:
                    logging.warning("No Binance rows returned for %s", symbol)
            except Exception as exc:
                logging.warning("Binance download failed for %s: %s", symbol, exc)
                continue
        frames.append(_clean(df))
    if not frames:
        raise RuntimeError("No data loaded. Check Binance access or cached CSVs in data/raw/.")
    return _clean(pd.concat(frames, ignore_index=True))
