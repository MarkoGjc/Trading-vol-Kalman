# download_spot_1m_simple.py
# Refactored to expose `get_candles()` which downloads and optionally saves CSV
import requests, time, sys, pandas as pd
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo               # stdlib Py ≥ 3.9
import os
from typing import Optional




# -------------------------------------------------------------
def fetch_chunk(symbol: str, start_ms: int, interval: str, limit: int,market_type: str):
    params = dict(symbol=symbol, interval=interval,
                  startTime=start_ms, limit=limit)
    if market_type == "Futures":
        API_URL   = "https://fapi.binance.com/fapi/v1/klines"
    if market_type == "Spot":
        API_URL   = "https://api.binance.com/api/v3/klines"
    return requests.get(API_URL, params=params, timeout=10).json()


def download(symbol: str, start_dt: datetime, end_dt: datetime,
             interval: str, limit, sleep: float, market_type: str) -> pd.DataFrame:
    PARIS_TZ  = ZoneInfo("Europe/Paris")
    rows, cur = [], start_dt
    while cur < end_dt:
        batch = fetch_chunk(symbol, int(cur.timestamp() * 1000), interval=interval, limit=limit, market_type=market_type)
        if not batch:
            break
        rows.extend(batch)

        last_open = datetime.fromtimestamp(batch[-1][0] / 1000, tz=timezone.utc)
        cur = last_open + timedelta(minutes=1)
        time.sleep(sleep)

    if not rows:
        raise RuntimeError("Aucune donnée reçue ; période peut-être trop ancienne.")

    df = pd.DataFrame(rows, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","qav","trades","tb_base","tb_quote","ignore"
    ])

    # Horodatage Europe/Paris, puis suppression de l'info fuseau (+02:00)
    df["timestamp"] = (
        pd.to_datetime(df["open_time"], unit="ms", utc=True)
          .dt.tz_convert(PARIS_TZ)
          .dt.tz_localize(None)
    )

    df[["open","high","low","close","volume"]] = (
        df[["open","high","low","close","volume"]].astype(float))

    return df[["timestamp","open","high","low","close","volume"]]


def get_candles(symbol: str = "BTCUSDT",
                days_back: int = 50,
                interval: str = "1m",
                market_type = "Spot") -> pd.DataFrame:
    """Télécharge les chandelles (1m par défaut) et optionnellement sauvegarde en CSV.

    Usage simple :
        df = get_candles("BTCUSDT", 50)
    """
    limit = 1000
    sleep = 0.12
    output_dir  = os.path.join(os.getcwd(), "output")
    # normalize market_type and prepare market-specific folder
    market_type = str(market_type).capitalize()
    end_utc = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start_utc = end_utc - timedelta(days=days_back)

    df = download(symbol, start_utc, end_utc, interval=interval, limit=limit, sleep=sleep, market_type=market_type)

    # Rename the 'timestamp' column to 'Time' and set as index
    columns = {"timestamp": "Time", "open": "Open", "high": "High",
               "low": "Low", "close": "Close", "volume": "Volume"}
    df.rename(columns=columns, inplace=True)
    df.set_index('Time', inplace=True)

    os.makedirs(output_dir, exist_ok=True)

    # create market-type subdirectory (Spot or Futures)
    market_dir = os.path.join(output_dir, market_type)
    os.makedirs(market_dir, exist_ok=True)

    # create per-symbol subdirectory inside market_dir
    symbol_dir = os.path.join(market_dir, symbol)
    os.makedirs(symbol_dir, exist_ok=True)

    output_file = os.path.join(symbol_dir, f"{symbol}_{interval}_{days_back}d.csv")

    df.to_csv(output_file, index=True)
    print(f"✅  Fichier créé :  {output_file}   ({len(df):,} lignes)")


    return df


# -------------------------------------------------------------
if __name__ == "__main__":
    symbol    = "BTCUSDT"
    interval  = "1m"
    days_back = 10
    market_type = "Futures"
    print(f"⏬  {symbol} | {interval} | {days_back} jours …")
    try:
        get_candles(symbol, days_back, interval, market_type)
    except Exception as e:
        print("❌  Erreur :", e, file=sys.stderr)
        sys.exit(1)