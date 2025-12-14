from __future__ import annotations

import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf




def _safe_filename(s: str) -> str:
    # Remplace les caractères invalides Windows: \ / : * ? " < > |
    return re.sub(r'[\\/:*?"<>|]+', "_", s)


def fetch_and_save_index_data(
    ticker: str,
    freq: str,
    days: int,
    base_dir: str | Path = r"C:\Users\Gajic\OneDrive - Université Paris-Dauphine\Trading\Trading-vol-Kalman\get_data\Stocks",
    pause_s: float = 0.5,
) -> Path:
    """
    Télécharge des données via yfinance (en chunks), puis sauvegarde en CSV dans:
      base_dir/<ticker_sanitized>/<ticker>_<freq>_...csv
    """
    _MAX_DAYS = {"1m": 30,"2m": 60,"5m": 60,"15m": 60,"30m": 60,"60m": 730,"1h": 730,"90m": 60,
                "1d": 36500,  "5d": 36500,"1wk": 36500,"1mo": 36500,"3mo": 36500,
    }

    # Taille de chunk (en jours) pour fiabiliser les downloads
    _CHUNK_DAYS = {"1m": 7,"2m": 30,"5m": 30,"15m": 30,"30m": 60,"60m": 180,"1h": 180,"90m": 60,
                    "1d": 3650,"5d": 3650,"1wk": 3650,"1mo": 3650,"3mo": 3650,
    }
    if days <= 0:
        raise ValueError("days doit être > 0")

    freq = freq.lower().strip()
    if freq not in _MAX_DAYS:
        raise ValueError(f"Fréquence non supportée: {freq}. Ex: {sorted(_MAX_DAYS.keys())}")

    max_days = _MAX_DAYS[freq]
    req_days = min(days, max_days)
    if req_days < days:
        print(f"[info] Demande réduite à {req_days} jours (limite typique pour {freq}).")

    chunk_days = _CHUNK_DAYS.get(freq, 30)

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=req_days)

    dfs: list[pd.DataFrame] = []
    cur = start

    while cur < end:
        nxt = min(cur + timedelta(days=chunk_days), end)

        df = yf.download(
            ticker,
            start=cur,
            end=nxt,
            interval=freq,
            progress=False,
            threads=False,
            auto_adjust=False,
        )

        if df is not None and not df.empty:
            dfs.append(df)

        cur = nxt
        time.sleep(pause_s)

    if not dfs:
        raise RuntimeError(
            f"Aucune donnée récupérée pour {ticker} en {freq} sur ~{req_days} jours. "
            f"Essaye un proxy (ex: SPY au lieu de ^GSPC) ou une fréquence moins fine."
        )

    data = pd.concat(dfs).sort_index()
    data = data[~data.index.duplicated(keep="last")]

    # Remove 'Adj Close' if present
    if "Adj Close" in data.columns:
        data = data.drop(columns=["Adj Close"])

    # Convert index (datetime) to formatted Time column: DD/MM/YYYY  HH:MM:SS
    idx = pd.to_datetime(data.index)
    # If timezone-naive, assume UTC; otherwise convert to UTC for consistent formatting
    try:
        if idx.tz is None:
            idx = idx.tz_localize(timezone.utc)
        else:
            idx = idx.tz_convert(timezone.utc)
    except Exception:
        # fallback: ignore timezone operations
        idx = pd.to_datetime(data.index)

    time_str = idx.strftime("%d/%m/%Y  %H:%M:%S")
    data = data.copy()
    data.insert(0, "Time", time_str)

    # Keep only the requested columns in order
    wanted = ["Time", "Open", "High", "Low", "Close", "Volume"]
    available = [c for c in wanted if c in data.columns]
    data = data[available]

    base_dir = Path(base_dir)
    # ensure an 'output' folder inside the Index folder, then per-ticker subfolder
    output_base = base_dir / "output"
    ticker_dir = output_base / _safe_filename(ticker)
    ticker_dir.mkdir(parents=True, exist_ok=True)
    fname = _safe_filename(f"{ticker}_{freq}_{req_days}d.csv")

    out_path = ticker_dir / fname

    data.columns = data.columns.droplevel(1)
    data.to_csv(out_path, index=False)

def main_stocks():

    stock_tickers = [
    "V", "MA",
    "KO", "PEP",
    "XOM", "CVX",
    "HD", "LOW",
    "JPM", "BAC",
    # optionnels (plus "news/earnings sensitive"):
    "AAPL", "MSFT",
    "NVDA", "AMD",
    ]

    yf_max_days_by_interval = {"1m": 7,"2m": 60,"5m": 60,"15m": 60,"30m": 60,"60m": 700,   
                               "1h": 700,"90m": 60,"1d": 1000,"5d": 1000,"1wk": 1000,
                               "1mo": 1000,"3mo": 1000}
    
    for ticker in stock_tickers:
        for freq, day_back in yf_max_days_by_interval.items():
            print(f"⏬  {ticker} | {freq} | {day_back} jours …")
            try:
                fetch_and_save_index_data(ticker, freq, day_back)
            except Exception as e:
                print(f"❌  Erreur pour {ticker} | {freq} | {day_back} jours : {e}")


    # #TESTS
    # fetch_and_save_index_data("AAPL", "1d", 5000)
    # fetch_and_save_index_data("AAPL", "15m", 60)
    # fetch_and_save_index_data("AAPL", "1h", 500)



if __name__ == "__main__":

    main_stocks()