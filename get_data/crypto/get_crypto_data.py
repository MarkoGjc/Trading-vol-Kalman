# download_spot_1m_simple.py
# ──────────────────────────────────────────────────────────────
# 1.  Règle SYMBOL et DAYS_BACK juste ici ➜
# 2.  Exécute :  python download_spot_1m_simple.py
#                → CSV avec date-heure *sans* indicateur de fuseau :
#                  2025-05-12 19:20:00
# ──────────────────────────────────────────────────────────────
import requests, time, sys, pandas as pd
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo               # stdlib Py ≥ 3.9
import os
# ───────  ➜  PERSONALISE ICI  ───────
SYMBOL     = "BTCUSDT"       # ex. "ETHUSDT", "SOLUSDT"…
DAYS_BACK  = 50              # nombre de jours à remonter (3 ans)
# ────────────────────────────────────
OUTPUT_DIR  = os.path.join(os.getcwd(), "output")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"{SYMBOL}_1m_{DAYS_BACK}d.csv")

API_URL   = "https://api.binance.com/api/v3/klines"
INTERVAL  = "1m"
LIMIT     = 1000
SLEEP     = 0.12
PARIS_TZ  = ZoneInfo("Europe/Paris")

# -------------------------------------------------------------
def fetch_chunk(symbol: str, start_ms: int):
    params = dict(symbol=symbol, interval=INTERVAL,
                  startTime=start_ms, limit=LIMIT)
    return requests.get(API_URL, params=params, timeout=10).json()

def download(symbol: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    rows, cur = [], start_dt
    while cur < end_dt:
        batch = fetch_chunk(symbol, int(cur.timestamp() * 1000))
        if not batch:
            break
        rows.extend(batch)

        last_open = datetime.fromtimestamp(batch[-1][0] / 1000, tz=timezone.utc)
        cur = last_open + timedelta(minutes=1)
        time.sleep(SLEEP)

    if not rows:
        raise RuntimeError("Aucune donnée reçue ; période peut-être trop ancienne.")

    df = pd.DataFrame(rows, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","qav","trades","tb_base","tb_quote","ignore"
    ])

    # Horodatage Europe/Paris, puis suppression de l’info fuseau (+02:00)
    df["timestamp"] = (
        pd.to_datetime(df["open_time"], unit="ms", utc=True)
          .dt.tz_convert(PARIS_TZ)       # passe UTC → Paris (gère DST)
          .dt.tz_localize(None)          # enlève le fuseau → 'naive'
    )

    df[["open","high","low","close","volume"]] = (
        df[["open","high","low","close","volume"]].astype(float))

    return df[["timestamp","open","high","low","close","volume"]]

# -------------------------------------------------------------
if __name__ == "__main__":
    end_utc   = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start_utc = end_utc - timedelta(days=DAYS_BACK)

    print(f"⏬  {SYMBOL} | 1-minute | {DAYS_BACK} jours …")
    try:
        df = download(SYMBOL, start_utc, end_utc)
    except Exception as e:
        print("❌  Erreur :", e, file=sys.stderr)
        sys.exit(1)


    # Rename the 'Timestamp' column to 'Time'
    columns = {"timestamp" : "Time","open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"}
    df.rename(columns=columns, inplace=True)
    # Set 'Time' as the index
    df.set_index('Time', inplace=True)

    # Créer le dossier output_spot si nécessaire
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    df.to_csv(OUTPUT_FILE, index=True)
    print(f"✅  Fichier créé :  {OUTPUT_FILE}   ({len(df):,} lignes)")