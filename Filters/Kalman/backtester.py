import numpy as np
import pandas as pd


def backtest_redpoints_entry_1m_exit_15m_discrete(
    df15m: pd.DataFrame,
    df1m: pd.DataFrame,
    *,
    time_col: str = "Time",                 # OPEN time
    filtered_col: str = "filtered_close",
    red_threshold: float = 0.0020,          # ex 20 bps = 0.0020
    entry_delay_min: int = 1,               # entrée = close+1min
    exit_delay_min: int = 1,                # sortie = close+1min
    fee_roundtrip_bps: float = 4.0,         # frais aller-retour
    tp_bps: float = 25.0,                   # take profit (net > 0 typiquement >= fees+buffer)
    sl_bps: float = 50.0,                   # stop loss
    max_hold_bars: int = 8,                 # time stop en bougies 15m (8=2h)
    allow_overlap: bool = False,
    require_entry_dislocation: bool = True, # à l’entrée, encore du bon côté du fair value ?
    min_entry_disloc_bps: float = 0.0,      # filtre optionnel
    debug: bool = True,
    debug_every: int = 50,
):
    """
    Stratégie baseline:
      - point rouge sur close 15m: abs(close-filtered)/abs(close) > red_threshold
      - direction MR: close>filtered => short ; close<filtered => long
      - entrée: close_15m + entry_delay_min, prix = Open 1m
      - gestion: on check TP/SL uniquement aux closes 15m suivants, puis on sort close+exit_delay_min sur Open 1m
    """

    # -----------------------
    # Prep / sort
    # -----------------------
    d15 = df15m.copy()
    d1 = df1m.copy()

    d15[time_col] = pd.to_datetime(d15[time_col])
    d1[time_col] = pd.to_datetime(d1[time_col])

    d15 = d15.sort_values(time_col).reset_index(drop=True)
    d1 = d1.sort_values(time_col).set_index(time_col)

    if filtered_col not in d15.columns:
        raise ValueError(f"df15m doit contenir '{filtered_col}'.")

    if debug:
        print("[DEBUG] Input ranges")
        print(f"  df15m: {d15[time_col].min()} -> {d15[time_col].max()} (rows={len(d15)})")
        print(f"  df1m : {d1.index.min()} -> {d1.index.max()} (rows={len(d1)})")
        print(f"  red_threshold={red_threshold} ({red_threshold*10000:.1f} bps), tp_bps={tp_bps}, sl_bps={sl_bps}, fees_rt={fee_roundtrip_bps} bps")
        print(f"  entry_delay={entry_delay_min}m, exit_delay={exit_delay_min}m, max_hold_bars={max_hold_bars}")
        print(f"  require_entry_dislocation={require_entry_dislocation}, min_entry_disloc_bps={min_entry_disloc_bps}")

    idx1 = d1.index.values  # datetime64[ns], sorted

    def next_1m_open(ts: pd.Timestamp):
        ts64 = np.datetime64(ts)
        pos = idx1.searchsorted(ts64, side="left")
        if pos >= len(idx1):
            return None
        return pd.Timestamp(idx1[pos])

    # -----------------------
    # Compute red points + side
    # -----------------------
    close15 = d15["Close"].astype(float).values
    filt15 = d15[filtered_col].astype(float).values

    denom = np.where(np.abs(close15) > 0, np.abs(close15), np.nan)
    pct_diff = np.abs(close15 - filt15) / denom
    is_red = pct_diff > float(red_threshold)

    # direction MR
    side = (-np.sign(close15 - filt15)).astype(int)  # +1 long, -1 short, 0 flat

    # timestamps
    open15 = pd.to_datetime(d15[time_col])
    close15_time = open15 + pd.Timedelta(minutes=15)

    if debug:
        print("[DEBUG] Red stats")
        print(f"  red_count={int(is_red.sum())}/{len(d15)} ({is_red.mean()*100:.2f}%)")
        if np.isfinite(pct_diff).any():
            print(f"  pct_diff: p50={np.nanmedian(pct_diff):.6f}, p95={np.nanpercentile(pct_diff,95):.6f}, max={np.nanmax(pct_diff):.6f}")

    # -----------------------
    # Backtest loop
    # -----------------------
    fee_frac = float(fee_roundtrip_bps) / 1e4
    tp_frac = float(tp_bps) / 1e4
    sl_frac = float(sl_bps) / 1e4

    trades = []
    in_trade = False
    trade_until_time = None  # last exit time (for overlap control)

    # stats counters
    c_red = int(is_red.sum())
    c_signal = 0
    c_entered = 0
    c_skip_overlap = 0
    c_skip_no1m = 0
    c_skip_cross = 0
    c_skip_small = 0

    i = 0
    while i < len(d15):
        if not is_red[i] or side[i] == 0:
            i += 1
            continue

        c_signal += 1

        signal_close_time = close15_time.iloc[i]
        entry_time = signal_close_time + pd.Timedelta(minutes=entry_delay_min)

        # overlap control
        if (not allow_overlap) and (trade_until_time is not None) and (entry_time <= trade_until_time):
            c_skip_overlap += 1
            i += 1
            continue

        et = next_1m_open(entry_time)
        if et is None:
            c_skip_no1m += 1
            break

        entry_px = float(d1.loc[et, "Open"])
        if not np.isfinite(entry_px) or entry_px <= 0:
            c_skip_no1m += 1
            i += 1
            continue

        s = int(side[i])
        filt_signal = float(filt15[i])

        # entry filters (important si entrée en retard)
        entry_disloc_bps = abs(entry_px - filt_signal) / entry_px * 10000.0

        if require_entry_dislocation:
            if s == 1 and not (entry_px < filt_signal):
                c_skip_cross += 1
                i += 1
                continue
            if s == -1 and not (entry_px > filt_signal):
                c_skip_cross += 1
                i += 1
                continue

        if entry_disloc_bps < float(min_entry_disloc_bps):
            c_skip_small += 1
            i += 1
            continue

        # Enter trade
        c_entered += 1
        entry_i = i
        entry_i_time = open15.iloc[i]
        entry_close_time = signal_close_time

        if debug and (c_entered % max(1, int(debug_every)) == 0):
            print(f"[DEBUG] Enter #{c_entered}: side={'LONG' if s==1 else 'SHORT'}")
            print(f"  signal_open={open15.iloc[i]} close={signal_close_time} entry_time={et} entry_px={entry_px:.2f}")
            print(f"  pct_diff={pct_diff[i]*10000:.1f}bps  entry_disloc={entry_disloc_bps:.1f}bps  filtered={filt_signal:.2f}")

        # monitor at future 15m closes
        exit_reason = "time"
        exit_i = min(entry_i + max_hold_bars, len(d15) - 1)

        for j in range(entry_i + 1, exit_i + 1):
            mark_px = float(d15.loc[j, "Close"])  # observable au close de la bougie j

            # mark-to-market PnL (net approximé, décision)
            if s == 1:
                pnl_mark = (mark_px / entry_px) - 1.0 - fee_frac
            else:
                pnl_mark = (entry_px / mark_px) - 1.0 - fee_frac

            if pnl_mark >= tp_frac:
                exit_reason = "tp"
                exit_i = j
                break
            if pnl_mark <= -sl_frac:
                exit_reason = "sl"
                exit_i = j
                break

        # execute exit at close+exit_delay_min
        exit_close_time = close15_time.iloc[exit_i]
        exit_time = exit_close_time + pd.Timedelta(minutes=exit_delay_min)
        xt = next_1m_open(exit_time)
        if xt is None:
            c_skip_no1m += 1
            break

        exit_px = float(d1.loc[xt, "Open"])
        if not np.isfinite(exit_px) or exit_px <= 0:
            c_skip_no1m += 1
            break

        # realized pnl
        if s == 1:
            pnl_gross = (exit_px / entry_px) - 1.0
        else:
            pnl_gross = (entry_px / exit_px) - 1.0
        pnl_net = pnl_gross - fee_frac
        win = int(pnl_net > 0)

        trade_until_time = xt

        trades.append({
            "signal_15m_open": open15.iloc[entry_i],
            "signal_15m_close": entry_close_time,
            "entry_time_1m": et,
            "exit_time_1m": xt,
            "side": s,
            "entry_px": entry_px,
            "exit_px": exit_px,
            "pct_diff": float(pct_diff[entry_i]),
            "entry_disloc_bps": float(entry_disloc_bps),
            "exit_reason": exit_reason,
            "hold_bars_15m": int(exit_i - entry_i),
            "hold_minutes": float((xt - et).total_seconds() / 60.0),
            "pnl_gross": float(pnl_gross),
            "pnl_net": float(pnl_net),
            "win": win,
        })

        i = exit_i + 1 if (not allow_overlap) else i + 1

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        if debug:
            print("[DEBUG] No trades executed.")
            print(f"  red_total={c_red}, signals={c_signal}, entered={c_entered}")
            print(f"  skips: overlap={c_skip_overlap}, no1m={c_skip_no1m}, entry_cross={c_skip_cross}, entry_small={c_skip_small}")
        return trades_df, {"n_trades": 0, "msg": "Aucun trade exécuté."}

    # summary
    n = len(trades_df)
    win_rate = float(trades_df["win"].mean())
    avg_pnl = float(trades_df["pnl_net"].mean())
    med_pnl = float(trades_df["pnl_net"].median())

    pf = (
        trades_df.loc[trades_df["pnl_net"] > 0, "pnl_net"].sum()
        / (-trades_df.loc[trades_df["pnl_net"] < 0, "pnl_net"].sum() + 1e-12)
    )
    pf = float(pf)

    eq = np.clip(1.0 + trades_df["pnl_net"].values, 1e-12, None)
    equity = np.cumprod(eq)
    peak = np.maximum.accumulate(equity)
    max_dd = float(np.min(equity / peak - 1.0))
    trades_df["equity"] = equity

    summary = {
        "n_trades": int(n),
        "win_rate": win_rate,
        "avg_pnl_net_bps": float(avg_pnl * 10000.0),
        "median_pnl_net_bps": float(med_pnl * 10000.0),
        "profit_factor": pf,
        "total_return": float(equity[-1] - 1.0),
        "max_drawdown": max_dd,
        "avg_hold_min": float(trades_df["hold_minutes"].mean()),
        "exit_reasons": trades_df["exit_reason"].value_counts().to_dict(),
        "skip_stats": {
            "red_total": c_red,
            "signals": c_signal,
            "entered": c_entered,
            "skip_overlap": c_skip_overlap,
            "skip_no1m": c_skip_no1m,
            "skip_entry_cross": c_skip_cross,
            "skip_entry_small": c_skip_small,
        }
    }

    if debug:
        print("[DEBUG] Summary")
        print(summary)

        mean_win = trades_df.loc[trades_df["win"] == 1, "pnl_net"].mean()
        mean_loss = trades_df.loc[trades_df["win"] == 0, "pnl_net"].mean()
        if np.isfinite(mean_win) and np.isfinite(mean_loss) and mean_win > 0 and mean_loss < 0:
            be_wr = float((-mean_loss) / ((-mean_loss) + mean_win))
            print(f"[DEBUG] mean_win={mean_win*10000:.2f}bps mean_loss={mean_loss*10000:.2f}bps break_even_wr≈{be_wr:.3f}")

    return trades_df, summary


def main():

    df15m = pd.read_csv(r"C:\Users\Gajic\OneDrive - Université Paris-Dauphine\Trading\Trading-vol-Kalman\Filters\Kalman\outpout_filter\BTCUSDT_15m_imm_filtered.csv")
    df1m = pd.read_csv(r"C:\Users\Gajic\OneDrive - Université Paris-Dauphine\Trading\Trading-vol-Kalman\get_data\Crypto\output\Futures\BTCUSDT\BTCUSDT_1m_400d.csv")
    trades, summary = backtest_redpoints_entry_1m_exit_15m_discrete(
        df15m, df1m,
        filtered_col="filtered_close",
        red_threshold=0.002,       # 20 bps
        fee_roundtrip_bps=4,
        tp_bps=25,
        sl_bps=50,
        max_hold_bars=8,
        require_entry_dislocation=True,
        min_entry_disloc_bps=15,    # optionnel
        debug=True,
    )



if __name__ == "__main__":
    main()
