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
    base_dir: str | Path = r"C:\Users\Gajic\OneDrive - Université Paris-Dauphine\Trading\Trading-vol-Kalman\get_data\Index",
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
    data.insert(0, "Ticker", ticker)

    base_dir = Path(base_dir)
    ticker_dir = base_dir / _safe_filename(ticker)
    ticker_dir.mkdir(parents=True, exist_ok=True)

    start_str = start.strftime("%Y%m%d")
    end_str = end.strftime("%Y%m%d")
    fname = _safe_filename(f"{ticker}_{freq}_{req_days}d_{start_str}-{end_str}.csv")

    out_path = ticker_dir / fname
    data.to_csv(out_path, index=True)



# --- Exemples ---
if __name__ == "__main__":
    fetch_and_save_index_data("^GSPC", "1d", 5000)
    fetch_and_save_index_data("SPY", "15m", 60)
    fetch_and_save_index_data("SPY", "1h", 500)
