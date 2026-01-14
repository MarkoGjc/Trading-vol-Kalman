import numpy as np
import pandas as pd

def backtest_redpoints_entry_1m_exit_15m_discrete_with_tp_sl_1m_and_filters(
    df15m: pd.DataFrame,
    df1m: pd.DataFrame,
    *,
    time_col: str = "Time",                 # OPEN time (15m et 1m)
    filtered_col: str = "filtered_close",

    # --- Signal rouge (15m) ---
    red_threshold: float = 0.0020,          # ex: 0.002 = 20 bps
    require_entry_dislocation: bool = True,
    min_entry_disloc_bps: float = 0.0,

    # --- Timing ---
    entry_delay_min: int = 1,               # entrée = close15 + 1min
    exit_delay_min: int = 1,                # sortie (si décision 15m) = close15 + 1min

    # --- Costs ---
    fee_roundtrip_bps: float = 4.0,         # frais aller-retour (bps)

    # --- Exits "lents" (discrets sur 15m) ---
    tp_bps: float = 25.0,                   # NET bps (décision sur close 15m)
    sl_bps: float = 50.0,                   # NET bps (décision sur close 15m)
    max_hold_bars: int = 8,                 # time stop en barres 15m

    # --- Exits "rapides" (intraminute 1m) ---
    use_tp_1m: bool = True,
    use_sl_1m: bool = True,
    tp_1min_bps: float = 15.0,              # NET bps (actif si use_tp_1m)
    sl_1min_bps: float = 30.0,              # NET bps (actif si use_sl_1m)
    one_min_uses_highlow: bool = True,      # True: High/Low, False: Close
    conservative_same_minute: bool = True,  # si TP et SL touchent même minute -> SL

    # --- Filters (df15m columns) ---
    use_pi_filter: bool = True,
    pi_col: str = "pi_high",
    pi_max: float = 0.85,

    use_slope_filter: bool = True,
    slope_col: str = "slope_pct_mix",       # fraction (ex: 0.0002)
    slope_mom_bps: float = 3.0,             # bps/bar

    # --- Overlap ---
    allow_overlap: bool = False,

    # --- Debug ---
    debug: bool = True,
    debug_every: int = 50,
):
    """
    Objectif:
      - Même logique que backtest_redpoints_entry_1m_exit_15m_discrete
      - + TP/SL 1m optionnels
      - + filtres pi_high / slope_pct_mix optionnels

    Important:
      - Time est OPEN time.
      - Signal calculé au close de la bougie 15m (Time+15m).
      - Entrée/exécution à close+1min via Open 1m.
      - Si TP/SL 1m déclenche, on sort au prix barrière (tp_px / sl_px) au timestamp de hit.

    Pour reproduire exactement (au mieux) la baseline:
      use_tp_1m=False, use_sl_1m=False, use_pi_filter=False, use_slope_filter=False
    """

    # -----------------------
    # Prep
    # -----------------------
    d15 = df15m.copy()
    d1 = df1m.copy()

    d15[time_col] = pd.to_datetime(d15[time_col])
    d1[time_col] = pd.to_datetime(d1[time_col])

    d15 = d15.sort_values(time_col).reset_index(drop=True)
    d1 = d1.sort_values(time_col).set_index(time_col)

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in d15.columns:
            raise ValueError(f"df15m doit contenir {col}")
        if col not in d1.columns:
            raise ValueError(f"df1m doit contenir {col}")
    if filtered_col not in d15.columns:
        raise ValueError(f"df15m doit contenir '{filtered_col}'.")

    if use_pi_filter and pi_col not in d15.columns:
        raise ValueError(f"use_pi_filter=True mais '{pi_col}' absent de df15m.")
    if use_slope_filter and slope_col not in d15.columns:
        raise ValueError(f"use_slope_filter=True mais '{slope_col}' absent de df15m.")

    idx1 = d1.index.values  # datetime64[ns], sorted

    def next_1m_open(ts: pd.Timestamp):
        pos = idx1.searchsorted(np.datetime64(ts), side="left")
        if pos >= len(idx1):
            return None
        return pd.Timestamp(idx1[pos])

    # Fractions
    fee_frac = float(fee_roundtrip_bps) / 1e4
    tp15_frac = float(tp_bps) / 1e4
    sl15_frac = float(sl_bps) / 1e4

    tp1_net = float(tp_1min_bps) / 1e4
    sl1_net = float(sl_1min_bps) / 1e4

    # -----------------------
    # Signal rouge + direction
    # -----------------------
    close15 = d15["Close"].astype(float).values
    filt15 = d15[filtered_col].astype(float).values
    denom = np.where(np.abs(close15) > 0, np.abs(close15), np.nan)
    pct_diff = np.abs(close15 - filt15) / denom
    is_red = pct_diff > float(red_threshold)

    # direction MR: close>filtered => short (-1), close<filtered => long (+1)
    side = (-np.sign(close15 - filt15)).astype(int)

    open15 = pd.to_datetime(d15[time_col])
    close15_time = open15 + pd.Timedelta(minutes=15)

    if debug:
        print("[DEBUG] Input ranges")
        print(f"  df15m: {d15[time_col].min()} -> {d15[time_col].max()} (rows={len(d15)})")
        print(f"  df1m : {d1.index.min()} -> {d1.index.max()} (rows={len(d1)})")
        print(f"  red_threshold={red_threshold} ({red_threshold*10000:.1f} bps)")
        print(f"  TP15={tp_bps}bps SL15={sl_bps}bps  fees_rt={fee_roundtrip_bps}bps  H={max_hold_bars} bars")
        print(f"  1m exits: use_tp_1m={use_tp_1m} tp_1min_bps={tp_1min_bps} | use_sl_1m={use_sl_1m} sl_1min_bps={sl_1min_bps}")
        print(f"  entry_delay={entry_delay_min}m exit_delay={exit_delay_min}m")
        print(f"  entry filters: require_entry_dislocation={require_entry_dislocation}, min_entry_disloc_bps={min_entry_disloc_bps}")
        if use_pi_filter:
            print(f"  pi_filter: {pi_col} <= {pi_max}")
        if use_slope_filter:
            print(f"  slope_filter: {slope_col} with |mom| threshold {slope_mom_bps} bps/bar")
        print("[DEBUG] Red stats")
        print(f"  red_count={int(is_red.sum())}/{len(d15)} ({is_red.mean()*100:.2f}%)")

    # -----------------------
    # Helpers: compute 1m barrier prices (NET thresholds)
    # -----------------------
    def compute_tp_sl_prices(entry_px: float, s: int):
        """
        Convert NET thresholds to price levels.
        pnl_net = pnl_gross - fee_frac

        LONG:
          pnl_gross = exit/entry - 1
          TP net: pnl_gross = fee + tp1_net  => exit = entry*(1 + fee + tp1_net)
          SL net: pnl_gross = fee - sl1_net  => exit = entry*(1 + fee - sl1_net)

        SHORT:
          pnl_gross = entry/exit - 1
          TP net: entry/exit - 1 - fee = tp1_net  => exit = entry / (1 + fee + tp1_net)
          SL net: entry/exit - 1 - fee = -sl1_net => exit = entry / (1 + fee - sl1_net)

        Si une barrière est désactivée ou invalide (<=0 / denom<=0), elle renvoie None.
        """
        tp_px = None
        sl_px = None

        if use_tp_1m:
            if s == 1:
                tp_px = entry_px * (1.0 + fee_frac + tp1_net)
            else:
                tp_px = entry_px / (1.0 + fee_frac + tp1_net)

        if use_sl_1m:
            if s == 1:
                sl_px = entry_px * (1.0 + fee_frac - sl1_net)
                if sl_px <= 0:
                    sl_px = None
            else:
                denom = (1.0 + fee_frac - sl1_net)
                if denom <= 0:
                    sl_px = None
                else:
                    sl_px = entry_px / denom

        return tp_px, sl_px

    def first_hit_time_1m(w1: pd.DataFrame, s: int, tp_px, sl_px):
        if w1.empty:
            return None, None

        tp_hit = None
        sl_hit = None

        if one_min_uses_highlow:
            hi = w1["High"].astype(float)
            lo = w1["Low"].astype(float)
            if tp_px is not None:
                tp_hit = (hi >= tp_px) if s == 1 else (lo <= tp_px)
            if sl_px is not None:
                sl_hit = (lo <= sl_px) if s == 1 else (hi >= sl_px)
        else:
            cl = w1["Close"].astype(float)
            if tp_px is not None:
                tp_hit = (cl >= tp_px) if s == 1 else (cl <= tp_px)
            if sl_px is not None:
                sl_hit = (cl <= sl_px) if s == 1 else (cl >= sl_px)

        tp_any = bool(tp_hit.any()) if tp_hit is not None else False
        sl_any = bool(sl_hit.any()) if sl_hit is not None else False

        if not tp_any and not sl_any:
            return None, None

        t_tp = tp_hit.index[tp_hit][0] if tp_any else None
        t_sl = sl_hit.index[sl_hit][0] if sl_any else None

        if tp_any and not sl_any:
            return "tp", t_tp
        if sl_any and not tp_any:
            return "sl", t_sl

        # both
        if t_tp < t_sl:
            return "tp", t_tp
        if t_sl < t_tp:
            return "sl", t_sl

        # same minute
        return ("sl" if conservative_same_minute else "tp"), t_tp

    # -----------------------
    # Backtest loop
    # -----------------------
    trades = []

    # counters
    c_red = int(is_red.sum())
    c_signal = 0
    c_entered = 0

    c_skip_pi = 0
    c_skip_slope = 0
    c_skip_overlap = 0
    c_skip_no1m = 0
    c_skip_cross = 0
    c_skip_small = 0

    c_exit_tp1m = 0
    c_exit_sl1m = 0

    trade_until_time = None

    i = 0
    while i < len(d15):
        if not is_red[i] or side[i] == 0:
            i += 1
            continue

        c_signal += 1

        # --- filters at signal time (close 15m of bar i) ---
        if use_pi_filter:
            pi = float(d15.loc[i, pi_col])
            if (not np.isfinite(pi)) or (pi > float(pi_max)):
                c_skip_pi += 1
                i += 1
                continue

        if use_slope_filter:
            slope_pct = float(d15.loc[i, slope_col])
            slope_bps = slope_pct * 10000.0
            s_dir = int(side[i])
            # short MR: skip if momentum up strong
            if s_dir == -1 and slope_bps > float(slope_mom_bps):
                c_skip_slope += 1
                i += 1
                continue
            # long MR: skip if momentum down strong
            if s_dir == 1 and slope_bps < -float(slope_mom_bps):
                c_skip_slope += 1
                i += 1
                continue

        # --- entry time ---
        signal_close_time = close15_time.iloc[i]
        entry_time = signal_close_time + pd.Timedelta(minutes=entry_delay_min)

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

        # entry dislocation checks
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

        c_entered += 1

        # 1m barrier prices (NET thresholds)
        tp1_px, sl1_px = compute_tp_sl_prices(entry_px, s)

        # horizon in 15m bars
        entry_i = i
        exit_i = min(entry_i + max_hold_bars, len(d15) - 1)
        exit_reason = "time"

        scan_start = et
        hit_1m = False
        exit_time = None
        exit_px = None

        # iterate bar-by-bar
        for j in range(entry_i + 1, exit_i + 1):
            bar_close = close15_time.iloc[j]
            bar_exec_time = bar_close + pd.Timedelta(minutes=exit_delay_min)

            # --- scan 1m for TP/SL ---
            if use_tp_1m or use_sl_1m:
                w1 = d1.loc[scan_start:bar_exec_time]
                hit_kind, t_hit = first_hit_time_1m(w1, s, tp1_px, sl1_px)
                if hit_kind is not None:
                    hit_1m = True
                    exit_time = pd.Timestamp(t_hit)
                    if hit_kind == "tp":
                        exit_reason = "tp_1m"
                        exit_px = tp1_px
                        c_exit_tp1m += 1
                    else:
                        exit_reason = "sl_1m"
                        exit_px = sl1_px
                        c_exit_sl1m += 1

                    # on associe l'exit_i au bar j (pour saut d'index comparable à la baseline)
                    exit_i = j
                    break

            # --- 15m discrete check at close ---
            mark_px = float(d15.loc[j, "Close"])
            if s == 1:
                pnl_mark = (mark_px / entry_px) - 1.0 - fee_frac
            else:
                pnl_mark = (entry_px / mark_px) - 1.0 - fee_frac

            if pnl_mark >= tp15_frac:
                exit_reason = "tp"
                exit_i = j
                break
            if pnl_mark <= -sl15_frac:
                exit_reason = "sl"
                exit_i = j
                break

            scan_start = bar_exec_time

        # Exit execution
        if hit_1m:
            # sortie au prix barrière
            if exit_px is None or (not np.isfinite(float(exit_px))) or float(exit_px) <= 0:
                # barrière invalide => fallback sur exit discret 15m
                hit_1m = False

        if not hit_1m:
            bar_close = close15_time.iloc[exit_i]
            exit_exec_time = bar_close + pd.Timedelta(minutes=exit_delay_min)
            xt = next_1m_open(exit_exec_time)
            if xt is None:
                c_skip_no1m += 1
                break
            exit_time = xt
            exit_px = float(d1.loc[xt, "Open"])

        trade_until_time = pd.Timestamp(exit_time)

        # Realized PnL
        if s == 1:
            pnl_gross = (float(exit_px) / entry_px) - 1.0
        else:
            pnl_gross = (entry_px / float(exit_px)) - 1.0
        pnl_net = pnl_gross - fee_frac
        win = int(pnl_net > 0)

        trades.append({
            "signal_15m_open": open15.iloc[entry_i],
            "signal_15m_close": close15_time.iloc[entry_i],
            "entry_time_1m": et,
            "exit_time_1m": pd.Timestamp(exit_time),
            "side": s,
            "entry_px": float(entry_px),
            "exit_px": float(exit_px),
            "pct_diff": float(pct_diff[entry_i]),
            "entry_disloc_bps": float(entry_disloc_bps),
            "exit_reason": exit_reason,
            "hold_minutes": float((pd.Timestamp(exit_time) - et).total_seconds() / 60.0),
            "pnl_gross": float(pnl_gross),
            "pnl_net": float(pnl_net),
            "win": win,
        })

        if debug and (c_entered % max(1, int(debug_every)) == 0):
            extra = ""
            if use_pi_filter:
                extra += f" pi={float(d15.loc[entry_i, pi_col]):.2f}"
            if use_slope_filter:
                extra += f" slope_bps={float(d15.loc[entry_i, slope_col])*10000.0:+.2f}"
            print(f"[DEBUG] Trade #{c_entered}: {'LONG' if s==1 else 'SHORT'} entry={et} exit={exit_time} reason={exit_reason}{extra}")
            print(f"  entry_px={entry_px:.2f} exit_px={float(exit_px):.2f} entry_disloc={entry_disloc_bps:.1f}bps pct_diff={pct_diff[entry_i]*10000:.1f}bps")
            print(f"  pnl_net={pnl_net*10000:.2f}bps")

        # Advance index (baseline-like)
        if not allow_overlap:
            i = exit_i + 1
        else:
            i += 1

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        if debug:
            print("[DEBUG] No trades executed.")
            print(f"  red_total={c_red}, signals={c_signal}, entered={c_entered}")
            print(f"  skips: overlap={c_skip_overlap}, no1m={c_skip_no1m}, entry_cross={c_skip_cross}, entry_small={c_skip_small}, pi={c_skip_pi}, slope={c_skip_slope}")
        return trades_df, {"n_trades": 0, "msg": "Aucun trade exécuté."}

    # Summary
    n = len(trades_df)
    win_rate = float(trades_df["win"].mean())
    avg_pnl = float(trades_df["pnl_net"].mean())
    med_pnl = float(trades_df["pnl_net"].median())

    pf = (
        trades_df.loc[trades_df["pnl_net"] > 0, "pnl_net"].sum()
        / (-trades_df.loc[trades_df["pnl_net"] < 0, "pnl_net"].sum() + 1e-12)
    )

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
        "profit_factor": float(pf),
        "total_return": float(equity[-1] - 1.0),
        "max_drawdown": max_dd,
        "avg_hold_min": float(trades_df["hold_minutes"].mean()),
        "exit_reasons": trades_df["exit_reason"].value_counts().to_dict(),
        "tp1m_count": int(c_exit_tp1m),
        "sl1m_count": int(c_exit_sl1m),
        "skip_stats": {
            "red_total": c_red,
            "signals": c_signal,
            "entered": c_entered,
            "skip_overlap": c_skip_overlap,
            "skip_no1m": c_skip_no1m,
            "skip_entry_cross": c_skip_cross,
            "skip_entry_small": c_skip_small,
            "skip_pi": c_skip_pi,
            "skip_slope": c_skip_slope,
        }
    }

    if debug:
        print("[DEBUG] Summary")
        print(summary)

    return trades_df, summary

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
        red_threshold=0.00035,       # 20 bps
        fee_roundtrip_bps=4,
        tp_bps=25,
        sl_bps=50,
        max_hold_bars=8,
        require_entry_dislocation=True,
        min_entry_disloc_bps=15,    # optionnel
        debug=True,
    )


    trades2, summary2 = backtest_redpoints_entry_1m_exit_15m_discrete_with_tp_sl_1m_and_filters(
        df15m, df1m,
        filtered_col="filtered_close",
        red_threshold=0.00035,
        fee_roundtrip_bps=4,
        tp_bps=15,
        sl_bps=15,
        max_hold_bars=8,
        require_entry_dislocation=True,
        min_entry_disloc_bps=0.25,

        use_tp_1m=True,
        use_sl_1m=True,
        use_pi_filter=True,
        use_slope_filter=True,
        slope_mom_bps = 1, 
        pi_max = 4,
        tp_1min_bps = 45.0,              # NET bps (actif si use_tp_1m)
        sl_1min_bps = 25.0, 

        debug=True
    )



if __name__ == "__main__":
    main()
