from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Tuple, Optional

import numpy as np
from pykalman import KalmanFilter

import pandas as pd
# ----------------------------
# 1) Fit base model (EM) on TRAIN ONLY (no-leak)
# ----------------------------
import numpy as np

import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

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



def plot_price_and_filtered_with_alerts(
    prices,
    filtered_prices,
    threshold=0.0375,
    x=None,
    title=None,
    figsize=(12, 6),
    show=True,
    markersize=4,
    linewidth=1.5,
    pi_high=None,
    slope=None,                 # slope_pct_mix (fraction) -> affiché en bps
    pi_fmt="{:.2f}",
    slope_bps_decimals=1,
    annotate_offset=(0, 8),
    lows=None,
    highs=None,
    extrema_color="blue",
    extrema_markersize=35,
    z_score=None,               # ✅ nouveau: liste/array même longueur
    z_thresh=2.0,               # ✅ seuil |z| pour point vert
    z_markersize=45,            # taille des points verts/noirs
):
    """
    - Trace prix réel vs prix filtré avec markers.
    - Points rouges si abs((price-filtered)/price) > threshold.
    - Points verts si abs(z_score) > z_thresh.
    - Si rouge ET vert au même index => point noir.
    Optionnel:
      - annote pi_high + slope (en bps) sur les points rouges (y compris noirs)
      - sur t+1 après chaque point rouge, plot low/high en bleu.
    """
    p = np.asarray(prices, dtype=float)
    f = np.asarray(filtered_prices, dtype=float)

    if p.shape != f.shape:
        raise ValueError(f"prices et filtered_prices doivent avoir la même longueur: {len(p)} vs {len(f)}")

    n = len(p)
    if x is None:
        x = np.arange(n)
    else:
        if len(x) != n:
            raise ValueError(f"x doit avoir la même longueur que prices: {len(x)} vs {n}")

    # pi_high validation
    if pi_high is not None:
        pi = np.asarray(pi_high, dtype=float)
        if len(pi) != n:
            raise ValueError(f"pi_high doit avoir la même longueur que prices: {len(pi)} vs {n}")
    else:
        pi = None

    # slope validation
    if slope is not None:
        sl = np.asarray(slope, dtype=float)
        if len(sl) != n:
            raise ValueError(f"slope doit avoir la même longueur que prices: {len(sl)} vs {n}")
    else:
        sl = None

    # lows/highs validation
    if (lows is None) ^ (highs is None):
        raise ValueError("Il faut fournir lows ET highs, ou aucun des deux.")
    if lows is not None and highs is not None:
        lo = np.asarray(lows, dtype=float)
        hi = np.asarray(highs, dtype=float)
        if len(lo) != n or len(hi) != n:
            raise ValueError(f"lows/highs doivent avoir la même longueur que prices: {len(lo)}, {len(hi)} vs {n}")
    else:
        lo = hi = None

    # z_score validation
    if z_score is not None:
        z = np.asarray(z_score, dtype=float)
        if len(z) != n:
            raise ValueError(f"z_score doit avoir la même longueur que prices: {len(z)} vs {n}")
    else:
        z = None

    # pct diff -> red mask
    denom = np.where(np.abs(p) > 0, np.abs(p), np.nan)
    pct_diff = np.abs(p - f) / denom
    mask_red = pct_diff > threshold

    # z mask -> green mask
    if z is not None:
        mask_green = np.abs(z) > float(z_thresh)
    else:
        mask_green = np.zeros(n, dtype=bool)

    # overlap -> black
    mask_black = mask_red & mask_green
    mask_red_only = mask_red & (~mask_black)
    mask_green_only = mask_green & (~mask_black)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(x, p, marker="o", linestyle="-", markersize=markersize, linewidth=linewidth, label="Prix (réel)")
    ax.plot(x, f, marker="o", linestyle="-", markersize=markersize, linewidth=linewidth, label="Prix filtré")

    x_arr = np.asarray(x)

    # Points: red / green / black
    if mask_red_only.any():
        ax.scatter(x_arr[mask_red_only], p[mask_red_only], s=40, color="red",
                   label=f"Rouge: |diff| > {threshold:.4f}")
    if mask_green_only.any():
        ax.scatter(x_arr[mask_green_only], p[mask_green_only], s=z_markersize, color="green",
                   label=f"Vert: |z| > {z_thresh:g}")
    if mask_black.any():
        ax.scatter(x_arr[mask_black], p[mask_black], s=max(50, z_markersize), color="black",
                   label="Noir: Rouge & Vert")

    # Annotations pi + slope sur les points rouges (rouges + noirs)
    mask_annot = mask_red  # inclut black
    if mask_annot.any() and (pi is not None or sl is not None):
        for idx in np.where(mask_annot)[0]:
            parts = []
            if pi is not None:
                parts.append(f"pi={pi_fmt.format(pi[idx])}")
            if sl is not None:
                slope_bps = sl[idx] * 10000.0
                parts.append(f"slope={slope_bps:+.{slope_bps_decimals}f} bps")
            ax.annotate(
                "\n".join(parts),
                (x_arr[idx], p[idx]),
                textcoords="offset points",
                xytext=annotate_offset,
                ha="center",
                fontsize=8
            )

    # sur t+1 après chaque point rouge (rouge + noir): low/high en bleu
    if lo is not None and hi is not None and mask_red.any():
        next_idx = np.where(mask_red)[0] + 1
        next_idx = next_idx[next_idx < n]
        next_idx = np.unique(next_idx)

        ax.scatter(
            x_arr[next_idx], lo[next_idx],
            s=extrema_markersize, color=extrema_color, marker="v",
            label="Low (t+1 après rouge)"
        )
        ax.scatter(
            x_arr[next_idx], hi[next_idx],
            s=extrema_markersize, color=extrema_color, marker="^",
            label="High (t+1 après rouge)"
        )

    ax.set_xlabel("Index" if x is None else "Temps")
    ax.set_ylabel("Prix")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    if show:
        plt.show()

    # je retourne aussi les masques pour debug
    return pct_diff, mask_red, mask_green, mask_black, fig, ax



def outlier_report(eps, name="eps"):
    eps = np.asarray(eps, float)
    eps = eps[np.isfinite(eps)]
    n = len(eps)

    def rate(th):
        return float(np.mean(np.abs(eps) > th))

    # kurtosis excess (sans scipy)
    m = eps.mean()
    v = eps.var()
    if v <= 0:
        ek = np.nan
    else:
        ek = float(np.mean(((eps - m)**4)) / (v*v) - 3.0)

    print(f"{name}: n={n}")
    print("  mean/std:", float(eps.mean()), float(eps.std()))
    print("  P(|eps|>3):", rate(3))
    print("  P(|eps|>4):", rate(4))
    print("  P(|eps|>5):", rate(5))
    print("  excess kurtosis:", ek)


@dataclass(frozen=True)
class BaseSSM:
    use_log: bool
    A: np.ndarray  # (n,n)
    H: np.ndarray  # (1,n)
    Q: np.ndarray  # (n,n)
    R: np.ndarray  # (1,1)
    m0: np.ndarray  # (n,)
    P0: np.ndarray  # (n,n)


def fit_base_local_linear_trend_em(
    prices_train: Iterable[float],
    use_log: bool = True,
    em_iter: int = 60,
) -> BaseSSM:
    """
    Base model: local linear trend (level+slope) on (log-)prices.
    Fit EM on TRAIN ONLY => parameters fixed afterwards (no-leak).
    """
    y = np.asarray(list(prices_train), dtype=float)
    if y.ndim != 1 or len(y) < 60:
        raise ValueError("prices_train doit être 1D et assez long (>= 60).")
    if not np.all(np.isfinite(y)):
        raise ValueError("prices_train contient NaN/inf.")
    if use_log and np.any(y <= 0):
        raise ValueError("Prix <= 0: impossible en log.")

    z = np.log(y) if use_log else y
    obs = z.reshape(-1, 1)

    # init heuristics (just for EM start)
    var0 = float(np.var(z[: min(len(z), 80)]))
    var0 = max(var0, 1e-6)
    r0 = max(var0 * 0.2, 1e-6)
    q0 = max(var0 * 0.02, 1e-8)

    # local linear trend: state=[level, slope]
    A = np.array([[1.0, 1.0],
                  [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q0 = np.array([[q0, 0.0],
                   [0.0, q0 * 0.1]])
    R0 = np.array([[r0]])

    kf0 = KalmanFilter(
        transition_matrices=A,
        observation_matrices=H,
        transition_covariance=Q0,
        observation_covariance=R0,
        initial_state_mean=np.array([float(z[0]), 0.0]),
        initial_state_covariance=np.array([[var0, 0.0],
                                           [0.0, var0]]),
    )

    kf = kf0.em(
        obs,
        n_iter=em_iter,
        em_vars=[
            "transition_covariance",
            "observation_covariance",
            "initial_state_mean",
            "initial_state_covariance",
        ],
    )

    A = np.asarray(kf.transition_matrices, dtype=float)
    H = np.asarray(kf.observation_matrices, dtype=float)
    Q = np.asarray(kf.transition_covariance, dtype=float)
    R = np.asarray(kf.observation_covariance, dtype=float)
    m0 = np.asarray(kf.initial_state_mean, dtype=float)
    P0 = np.asarray(kf.initial_state_covariance, dtype=float)

    # light PD safeguard
    def make_pd(M, jitter=1e-10):
        M = np.asarray(M, dtype=float)
        M = 0.5 * (M + M.T)
        return M + jitter * np.eye(M.shape[0])

    Q = make_pd(Q)
    P0 = make_pd(P0)
    R = make_pd(np.atleast_2d(R))

    return BaseSSM(use_log=use_log, A=A, H=H, Q=Q, R=R, m0=m0, P0=P0)


# ----------------------------
# 2) IMM 2-regime filter (online, no-leak)
# ----------------------------

class IMM2Regimes:
    """
    IMM 2 régimes (low/high vol) + robust gating optionnel (Huber-like).

    Attendu sur `base` :
      - base.use_log : bool
      - base.A : (n,n)
      - base.H : (1,n) ou (n,) compatible @
      - base.Q : (n,n)
      - base.R : scalaire ou (1,1)
      - base.m0 : (n,)
      - base.P0 : (n,n)
    """

    def __init__(
        self,
        base,
        *,
        q_scale_low: float = 0.2,
        r_scale_low: float = 0.8,
        q_scale_high: float = 2.0,
        r_scale_high: float = 1.4,
        p00: float = 0.995,
        p11: float = 0.995,
        mu0: Optional[np.ndarray] = None,
        jitter: float = 1e-10,
        # robust options
        robust: bool = True,
        huber_k: float = 3.0,
        robust_high_only: bool = True,
        robust_S_cap_mult: float = 1e6,
        min_r_scale: float = 1e-6,
    ):
        if not (0.0 < p00 < 1.0 and 0.0 < p11 < 1.0):
            raise ValueError("p00 et p11 doivent être dans (0,1).")
        if huber_k <= 0:
            raise ValueError("huber_k doit être > 0.")
        if robust_S_cap_mult < 1.0:
            raise ValueError("robust_S_cap_mult doit être >= 1.")

        # éviter R=0 (source de spikes / singularités)
        r_scale_low = max(float(r_scale_low), float(min_r_scale))
        r_scale_high = max(float(r_scale_high), float(min_r_scale))

        self.use_log = bool(base.use_log)

        self.A = np.asarray(base.A, dtype=float)
        self.H = np.asarray(base.H, dtype=float)

        self._n = self.A.shape[0]
        self._jitter = float(jitter)

        self.Q0 = np.asarray(base.Q, dtype=float) * float(q_scale_low)
        self.R0 = np.asarray(base.R, dtype=float) * float(r_scale_low)

        self.Q1 = np.asarray(base.Q, dtype=float) * float(q_scale_high)
        self.R1 = np.asarray(base.R, dtype=float) * float(r_scale_high)

        p01 = 1.0 - float(p00)
        p10 = 1.0 - float(p11)
        self.Ptrans = np.array([[p00, p01],
                                [p10, p11]], dtype=float)

        if mu0 is None:
            self.mu = np.array([0.5, 0.5], dtype=float)
        else:
            mu0 = np.asarray(mu0, dtype=float)
            mu0 = mu0 / mu0.sum()
            self.mu = mu0

        # états/covariances par régime
        self.m = np.stack([np.asarray(base.m0, dtype=float).copy(),
                           np.asarray(base.m0, dtype=float).copy()], axis=0)  # (2,n)
        self.P = np.stack([np.asarray(base.P0, dtype=float).copy(),
                           np.asarray(base.P0, dtype=float).copy()], axis=0)  # (2,n,n)

        # robust params
        self._robust = bool(robust)
        self._robust_high_only = bool(robust_high_only)
        self._huber_k = float(huber_k)
        self._robust_S_cap_mult = float(robust_S_cap_mult)

    @staticmethod
    def _log_gauss_1d(innov: float, S: float) -> float:
        return -0.5 * (math.log(2.0 * math.pi) + math.log(S) + (innov * innov) / S)

    def _ensure_pd(self, M: np.ndarray) -> np.ndarray:
        M = 0.5 * (M + M.T)
        M = M + self._jitter * np.eye(M.shape[0])
        return M

    def update(
        self, close_t: float
    ) -> Tuple[float, float, float, float, float, float, float, float, float]:
        # --- observation ---
        if not np.isfinite(close_t):
            raise ValueError("close_t NaN/inf.")
        if self.use_log:
            if close_t <= 0:
                raise ValueError("close_t <= 0 impossible en log.")
            z = math.log(float(close_t))
        else:
            z = float(close_t)

        # --- IMM mixing ---
        mu_prev = self.mu
        Ptrans = self.Ptrans

        c0 = Ptrans[0, 0] * mu_prev[0] + Ptrans[1, 0] * mu_prev[1]
        c1 = Ptrans[0, 1] * mu_prev[0] + Ptrans[1, 1] * mu_prev[1]
        c = np.array([c0, c1], dtype=float)
        c = np.maximum(c, 1e-300)

        mu_cond = np.empty((2, 2), dtype=float)  # i,j
        mu_cond[0, 0] = Ptrans[0, 0] * mu_prev[0] / c0
        mu_cond[1, 0] = Ptrans[1, 0] * mu_prev[1] / c0
        mu_cond[0, 1] = Ptrans[0, 1] * mu_prev[0] / c1
        mu_cond[1, 1] = Ptrans[1, 1] * mu_prev[1] / c1

        m_mix = np.empty_like(self.m)
        P_mix = np.empty_like(self.P)

        for j in (0, 1):
            w0, w1 = mu_cond[0, j], mu_cond[1, j]
            m0, m1 = self.m[0], self.m[1]

            m_j = w0 * m0 + w1 * m1
            m_mix[j] = m_j

            d0 = (m0 - m_j).reshape(-1, 1)
            d1 = (m1 - m_j).reshape(-1, 1)
            Pj = (
                w0 * (self.P[0] + d0 @ d0.T) +
                w1 * (self.P[1] + d1 @ d1.T)
            )
            P_mix[j] = self._ensure_pd(Pj)

        # --- Per-regime KF predict/update + robust likelihood ---
        A, H = self.A, self.H
        logL = np.empty(2, dtype=float)

        eps_raw = np.empty(2, dtype=float)  # sans gating
        eps_rob = np.empty(2, dtype=float)  # après gating (si activé)

        for j in (0, 1):
            Q = self.Q0 if j == 0 else self.Q1
            R = self.R0 if j == 0 else self.R1

            # predict
            m_pred = A @ m_mix[j]
            P_pred = A @ P_mix[j] @ A.T + Q

            # innovation
            y_pred = float((H @ m_pred).reshape(()))
            innov = z - y_pred

            s_x = float((H @ P_pred @ H.T).reshape(()))  # HPH'
            r_val = float(np.asarray(R).reshape(()))     # R scalaire
            S0 = s_x + r_val
            if S0 <= 0.0:
                S0 = 1e-12

            # RAW eps (avant robust)
            eps_raw[j] = innov / math.sqrt(S0)

            # Robust gating: inflation de S si |eps_raw| > k
            S = S0
            if self._robust and (not self._robust_high_only or j == 1):
                aeps = abs(eps_raw[j])
                if aeps > self._huber_k:
                    S_target = (innov * innov) / (self._huber_k * self._huber_k)
                    S = max(S0, min(S_target, S0 * self._robust_S_cap_mult))

            # ROBUST eps (après gating)
            eps_rob[j] = innov / math.sqrt(S)

            # likelihood + update avec S (robuste)
            logL[j] = self._log_gauss_1d(innov, S)

            K = (P_pred @ H.T) / S
            m_upd = m_pred + (K[:, 0] * innov)
            P_upd = P_pred - (K @ K.T) * S

            self.m[j] = m_upd
            self.P[j] = self._ensure_pd(P_upd)

        # --- Regime prob update ---
        logw0 = logL[0] + math.log(c[0])
        logw1 = logL[1] + math.log(c[1])
        mx = max(logw0, logw1)
        w0 = math.exp(logw0 - mx)
        w1 = math.exp(logw1 - mx)
        s = w0 + w1

        self.mu = np.array([w0 / s, w1 / s], dtype=float)
        mu = self.mu

        # --- aggregated state ---
        m_bar = mu[0] * self.m[0] + mu[1] * self.m[1]
        level = float(m_bar[0])
        slope_mix = float(m_bar[1])  # ✅ slope (log-price per bar si use_log=True)

        filtered_close = math.exp(level) if self.use_log else level
        pi_high = float(mu[1])

        # ✅ slope en % par barre (plus interprétable)
        slope_pct_mix = (math.exp(slope_mix) - 1.0) if self.use_log else slope_mix

        # --- outputs raw + robust ---
        eps_raw_low = float(eps_raw[0])
        eps_raw_high = float(eps_raw[1])
        eps_raw_mix = float(mu[0] * eps_raw_low + mu[1] * eps_raw_high)

        eps_robust_low = float(eps_rob[0])
        eps_robust_high = float(eps_rob[1])
        eps_robust_mix = float(mu[0] * eps_robust_low + mu[1] * eps_robust_high)

        return (
            float(filtered_close),
            pi_high,
            eps_raw_low, eps_raw_high, eps_raw_mix,
            eps_robust_low, eps_robust_high, eps_robust_mix,
            float(slope_pct_mix),   # ✅ ajouté (rendement attendu par barre)
        )



# ----------------------------
# 3) Example main (no-leak backtest plotting)
# ----------------------------

def main():
    # expects `prices` list/array of closes already defined

    data = pd.read_csv(r"C:\Users\Gajic\OneDrive - Université Paris-Dauphine\Trading\Trading-vol-Kalman\get_data\Crypto\output\Futures\BTCUSDT\BTCUSDT_15m_400d.csv") 
    df1m = pd.read_csv(r"C:\Users\Gajic\OneDrive - Université Paris-Dauphine\Trading\Trading-vol-Kalman\get_data\Crypto\output\Futures\BTCUSDT\BTCUSDT_1m_400d.csv")
    data = data.tail(10000)
    
    prices = np.array(data["Close"])
    lows = np.array(data["Low"])
    highs = np.array(data["High"])
    try:
        prices  # noqa: F821
    except NameError:
        raise NameError("Définis `prices = [...]` (closes) avant d'exécuter.")

    import matplotlib.pyplot as plt

    prices_arr = np.asarray(prices, dtype=float)
    if prices_arr.ndim != 1 or len(prices_arr) < 400:
        raise ValueError("`prices` doit être 1D et assez long (>= 400 recommandé).")
    if not np.all(np.isfinite(prices_arr)):
        raise ValueError("`prices` contient NaN/inf.")
    if np.any(prices_arr <= 0):
        raise ValueError("Prix <= 0: impossible si use_log=True.")

    # TRAIN/TEST split (index-based)
    split = int(0.7 * len(prices_arr))
    df15m = data.iloc[split:].reset_index(drop=True)
    train = prices_arr[:split]
    test = prices_arr[split:]
    test_lows = lows[split:]
    test_highs = highs[split:]

    # 1) Fit base parameters on TRAIN ONLY (no-leak)
    base = fit_base_local_linear_trend_em(train, use_log=True, em_iter=60)

    # 2) IMM engine (parameters fixed, regime estimated online)
    imm = IMM2Regimes(
        base,
        q_scale_low=0.2, r_scale_low=0.8,
        q_scale_high=2.0, r_scale_high=1.4,
        p00=0.995, p11=0.995,
        robust=True,
        huber_k=3.0,
        robust_high_only=True,      # recommandé: robuste surtout en high-vol
        robust_S_cap_mult=1e6,      # cap large = on peut ignorer un print extrême si besoin
    )


    # Warm-up on TRAIN (no storing)
    for p in train:
        imm.update(float(p))

    # Run on TEST (store outputs)
    filt_test = np.empty_like(test)
    pi_high_test = np.empty_like(test)

    eps_raw_low_test  = np.empty_like(test)
    eps_raw_high_test = np.empty_like(test)
    eps_raw_mix_test  = np.empty_like(test)

    eps_rob_low_test  = np.empty_like(test)
    eps_rob_high_test = np.empty_like(test)
    eps_rob_mix_test  = np.empty_like(test)

    slope_pct_mix_test = np.empty_like(test)

    for i, p in enumerate(test):
        (f, ph,
        erL, erH, erM,
        eL,  eH,  eM,
        slope_pct) = imm.update(float(p))

        filt_test[i] = f
        pi_high_test[i] = ph

        eps_raw_low_test[i] = erL
        eps_raw_high_test[i] = erH
        eps_raw_mix_test[i] = erM

        eps_rob_low_test[i] = eL
        eps_rob_high_test[i] = eH
        eps_rob_mix_test[i] = eM

        slope_pct_mix_test[i] = slope_pct
    df15m["filtered_close"] = filt_test

    # Quick sanity stats (sur test)
    print("eps_raw_mix mean/std:", float(np.mean(eps_raw_mix_test)), float(np.std(eps_raw_mix_test)))
    print("eps_rob_mix mean/std:", float(np.mean(eps_rob_mix_test)), float(np.std(eps_rob_mix_test)))

    print("eps_raw_low mean/std:", float(np.mean(eps_raw_low_test)), float(np.std(eps_raw_low_test)))
    print("eps_raw_high mean/std:", float(np.mean(eps_raw_high_test)), float(np.std(eps_raw_high_test)))

    print("eps_rob_low mean/std:", float(np.mean(eps_rob_low_test)), float(np.std(eps_rob_low_test)))
    print("eps_rob_high mean/std:", float(np.mean(eps_rob_high_test)), float(np.std(eps_rob_high_test)))
    
    # Align for plotting (NaN before split)
    filt_all = np.full_like(prices_arr, np.nan)
    pi_high_all = np.full_like(prices_arr, np.nan)
    filt_all[split:] = filt_test
    pi_high_all[split:] = pi_high_test

    eps_raw_mix_all = np.full_like(prices_arr, np.nan)
    eps_rob_mix_all = np.full_like(prices_arr, np.nan)

    eps_raw_mix_all[split:] = eps_raw_mix_test
    eps_rob_mix_all[split:] = eps_rob_mix_test

    print("Dernier obs       :", float(prices_arr[-1]))
    print("Dernier filt (IMM):", float(filt_test[-1]))
    print("Dernier pi_high   :", float(pi_high_test[-1]))
    print("Dernier eps_raw_mix   :", float(eps_raw_mix_test[-1]))
    print("Dernier eps_rob_mix   :", float(eps_rob_mix_test[-1]))

    diff_all = prices_arr - filt_all


    pct_diff, mask_red, mask_green, mask_black, fig, ax = plot_price_and_filtered_with_alerts(
    prices=test,
    filtered_prices= filt_test,
    threshold=0.0003,
    pi_high = pi_high_test,slope=slope_pct_mix_test,
    lows=test_lows,                  
    highs=test_highs,  
    z_score= eps_raw_mix_test,               
    z_thresh=2.7,
  # optionnel (timestamps), sinon indices
    title="Prix vs Prix filtré + alertes"
    )


    trades, summary = backtest_redpoints_entry_1m_exit_15m_discrete(
        df15m, df1m,
        filtered_col="filtered_close",
        red_threshold=0.0003,       # 20 bps
        fee_roundtrip_bps=4,
        tp_bps=25,
        sl_bps=50,
        max_hold_bars=8,
        require_entry_dislocation=True,
        min_entry_disloc_bps=15,    # optionnel
        debug=True,
    )


    print(pct_diff)
    
    plt.figure()
    plt.plot(eps_raw_mix_test, label="eps_raw_mix_test")
    plt.axvline(split, linestyle="--", label="Split train/test")
    plt.axhline(0.0, linestyle="--")
    plt.legend()
    plt.title("IMM — z-score du gap (cohérent log, EWMA causal)")
    plt.show()

    # FIGURE 1 : prix vs prix filtré
    plt.figure()
    plt.plot(prices_arr, label="Close observé")
    plt.plot(filt_all, label="Close filtré (test only)")
    plt.axvline(split, linestyle="--", label="Split train/test")
    plt.legend()
    plt.title("Prix vs prix filtré")
    plt.show(block=False)   # <-- important pour pouvoir déplacer la fenêtre avant la suivante

    plt.figure()
    plt.plot(diff_all, label="Close observé - Close filtré")
    plt.axvline(split, linestyle="--", label="Split train/test")
    plt.legend()
    plt.title("IMM — différence (prix réel - prix filtré)")
    plt.show()

    plt.figure()
    plt.plot(pi_high_all, label="P(régime high-vol)")
    plt.axvline(split, linestyle="--", label="Split train/test")
    plt.legend()
    plt.title("IMM 2 régimes — probabilité de régime (no-leak)")
    plt.show()
    
    plt.figure()
    plt.plot(eps_raw_mix_all, label="ε raw mix")
    plt.plot(eps_rob_mix_all, label="ε robust mix")
    plt.axvline(split, linestyle="--", label="Split train/test")
    plt.legend()
    plt.title("IMM 2 régimes — innovation normalisée (raw vs robust, test only)")
    plt.show()

    plt.figure()
    plt.plot(eps_rob_low_test, label="ε robust low")
    plt.plot(eps_rob_high_test, label="ε robust high")
    plt.legend()
    plt.title("IMM — ε robust par régime (test)")
    plt.show()

    plt.figure()
    plt.plot(eps_raw_mix_test, label="ε raw mix")
    plt.plot(eps_rob_mix_test, label="ε robust mix")
    plt.legend()
    plt.title("IMM — ε mix (test) raw vs robust")
    plt.show()

    maskH = (pi_high_test > 0.5)
    maskL = ~maskH

    print("\n--- Outlier reports (test) ---")
    outlier_report(eps_raw_mix_test, "eps_raw_mix")
    outlier_report(eps_rob_mix_test, "eps_robust_mix")

    outlier_report(eps_raw_low_test[maskL],  "eps_raw_low | low")
    outlier_report(eps_rob_low_test[maskL],  "eps_rob_low | low")

    outlier_report(eps_raw_high_test[maskH], "eps_raw_high | high")
    outlier_report(eps_rob_high_test[maskH], "eps_rob_high | high")






if __name__ == "__main__":
    main()
