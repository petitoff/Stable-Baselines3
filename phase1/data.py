import ccxt
import pandas as pd
import time
from datetime import datetime, timedelta, timezone
from typing import Any, cast
import argparse
from pathlib import Path


def fetch_bybit_ohlcv(
    symbol: str = "BTC/USDT",
    timeframe: str = "1m",
    days: int = 120,
    limit: int = 1000,
    save_csv: bool = False,
    csv_path: str | None = None,
    exchange_kwargs: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Fetch OHLCV data from Bybit via CCXT in paginated requests.

    Parameters
    - symbol: Market symbol, e.g. 'BTC/USDT'
    - timeframe: CCXT timeframe, e.g. '1m'
    - days: Lookback period in days
    - limit: Number of candles per request (Bybit supports up to 1000)
    - save_csv: Whether to save the result to CSV
    - csv_path: Optional explicit CSV path; if None, an auto name is used
    - exchange_kwargs: Extra kwargs for ccxt.bybit (e.g., {"enableRateLimit": True})

    Returns
    - DataFrame with columns: [timestamp, datetime, open, high, low, close, volume]
    """

    kwargs: dict[str, Any] = {"enableRateLimit": True}
    if exchange_kwargs:
        kwargs.update(exchange_kwargs)

    # Instantiate without passing kwargs to satisfy the type checker,
    # then apply known config fields on the instance.
    exchange = ccxt.bybit()
    exchange.enableRateLimit = bool(kwargs.get("enableRateLimit", True))
    # Optional: honor ccxt 'options', e.g., {'defaultType': 'spot' | 'swap'}
    options = kwargs.get("options")
    if isinstance(options, dict):
        default_type = options.get("defaultType")
        if default_type:
            # CCXT Bybit supports setting default market type (spot/swap)
            # Build a safe local options dict and assign back to exchange
            local_opts = cast(dict[str, Any], (getattr(exchange, "options", None) or {}))
            local_opts["defaultType"] = str(default_type)
            exchange.options = local_opts
    # Preload markets for consistent symbol handling
    exchange.load_markets()

    # Compute time bounds
    now = datetime.now(timezone.utc)
    since_dt = now - timedelta(days=days)
    since_ms = int(since_dt.timestamp() * 1000)

    all_rows: list[list[int | float]] = []
    fetch_since = since_ms

    # Bybit/CCXT rate limit: respect enableRateLimit; also small sleep between loops
    while True:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=fetch_since, limit=limit)
        if not ohlcv:
            break

        all_rows.extend(ohlcv)

        # Advance since to last timestamp + 1ms to avoid duplicates
        last_ts = ohlcv[-1][0]
        next_since = last_ts + 1

        # Stop if we've reached 'now' or no forward progress
        if next_since <= fetch_since:
            break
        fetch_since = next_since

        # Optional short sleep to be gentle
        time.sleep(0.05)

        # Safety: stop if we've exceeded now by a margin
        if fetch_since >= int(now.timestamp() * 1000):
            break

    if not all_rows:
        raise RuntimeError("No data returned from Bybit. Try reducing 'days' or check network/API.")

    df: pd.DataFrame = pd.DataFrame(
        all_rows,
        columns=pd.Index(["timestamp", "open", "high", "low", "close", "volume"]),
    )
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.loc[:, ["timestamp", "datetime", "open", "high", "low", "close", "volume"]]

    # Drop potential duplicates and sort
    df = (
        df.drop_duplicates(subset="timestamp")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    if save_csv:
        if csv_path is None:
            csv_path = f"bybit_{symbol.replace('/', '')}_{timeframe}_{days}d.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved {len(df):,} rows to {csv_path}")

    print(f"Fetched {len(df):,} candles from {df['datetime'].min()} to {df['datetime'].max()} (UTC)")
    return df