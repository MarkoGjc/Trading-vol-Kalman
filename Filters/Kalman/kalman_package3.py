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
    IMM (Interacting Multiple Model) with 2 regimes:
      regime 0: low-vol (smoother)
      regime 1: high-vol (more reactive)

    Each regime is a Kalman model with same A,H but different Q,R.
    Regime probabilities updated online at each close (no future data).
    """

    __slots__ = (
        "use_log", "A", "H",
        "Q0", "R0", "Q1", "R1",
        "Ptrans",
        "mu", "m", "P",
        "_n", "_jitter"
    )

    def __init__(
        self,
        base: BaseSSM,
        *,
        q_scale_low: float = 0.3,
        r_scale_low: float = 1.0,
        q_scale_high: float = 3.0,
        r_scale_high: float = 2.0,
        p00: float = 0.985,
        p11: float = 0.985,
        mu0: Optional[np.ndarray] = None,
        jitter: float = 1e-10,
    ):
        if not (0.0 < p00 < 1.0 and 0.0 < p11 < 1.0):
            raise ValueError("p00 et p11 doivent être dans (0,1).")

        self.use_log = base.use_log
        self.A = np.asarray(base.A, dtype=float)
        self.H = np.asarray(base.H, dtype=float)

        self._n = self.A.shape[0]
        self._jitter = float(jitter)

        self.Q0 = np.asarray(base.Q, dtype=float) * float(q_scale_low)
        self.R0 = np.asarray(base.R, dtype=float) * float(r_scale_low)
        self.Q1 = np.asarray(base.Q, dtype=float) * float(q_scale_high)
        self.R1 = np.asarray(base.R, dtype=float) * float(r_scale_high)

        # Transition matrix P(s_t=j | s_{t-1}=i)
        p01 = 1.0 - p00
        p10 = 1.0 - p11
        self.Ptrans = np.array([[p00, p01],
                                [p10, p11]], dtype=float)

        # Regime probs
        if mu0 is None:
            self.mu = np.array([0.5, 0.5], dtype=float)
        else:
            mu0 = np.asarray(mu0, dtype=float)
            mu0 = mu0 / mu0.sum()
            self.mu = mu0

        # Per-regime states: m[j] shape (n,), P[j] shape (n,n)
        self.m = np.stack([base.m0.copy(), base.m0.copy()], axis=0)  # (2,n)
        self.P = np.stack([base.P0.copy(), base.P0.copy()], axis=0)  # (2,n,n)

    @staticmethod
    def _log_gauss_1d(innov: float, S: float) -> float:
        return -0.5 * (math.log(2.0 * math.pi) + math.log(S) + (innov * innov) / S)

    def _ensure_pd(self, M: np.ndarray) -> np.ndarray:
        M = 0.5 * (M + M.T)
        M = M + self._jitter * np.eye(M.shape[0])
        return M

    def update(self, close_t: float) -> Tuple[float, float, float, float, float]:
        """
        Ingest close_t and return:
          filtered_close_t (IMM-averaged, no-leak),
          pi_high_t (mu[1]),
          eps_low_t (standardized innov in low regime),
          eps_high_t,
          eps_mix_t (mu-weighted)
        """
        if not np.isfinite(close_t):
            raise ValueError("close_t NaN/inf.")
        if self.use_log:
            if close_t <= 0:
                raise ValueError("close_t <= 0 impossible en log.")
            z = math.log(float(close_t))
        else:
            z = float(close_t)

        # --- IMM mixing step ---
        mu_prev = self.mu
        Ptrans = self.Ptrans

        # predicted regime probs (normalizers)
        c0 = Ptrans[0, 0] * mu_prev[0] + Ptrans[1, 0] * mu_prev[1]
        c1 = Ptrans[0, 1] * mu_prev[0] + Ptrans[1, 1] * mu_prev[1]
        c = np.array([c0, c1], dtype=float)
        c = np.maximum(c, 1e-300)

        # mixing probabilities mu_{i|j}
        mu_cond = np.empty((2, 2), dtype=float)  # i,j
        mu_cond[0, 0] = Ptrans[0, 0] * mu_prev[0] / c0
        mu_cond[1, 0] = Ptrans[1, 0] * mu_prev[1] / c0
        mu_cond[0, 1] = Ptrans[0, 1] * mu_prev[0] / c1
        mu_cond[1, 1] = Ptrans[1, 1] * mu_prev[1] / c1

        # mixed initial states for each j
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

        # --- Per-regime KF predict/update + likelihood ---
        A, H = self.A, self.H
        logL = np.empty(2, dtype=float)
        eps = np.empty(2, dtype=float)

        for j in (0, 1):
            Q = self.Q0 if j == 0 else self.Q1
            R = self.R0 if j == 0 else self.R1

            # predict
            m_pred = A @ m_mix[j]
            P_pred = A @ P_mix[j] @ A.T + Q

            # innovation
            y_pred = float((H @ m_pred).reshape(()))
            innov = z - y_pred
            S = float((H @ P_pred @ H.T).reshape(()) + R.reshape(()))
            if S <= 0.0:
                S = 1e-12

            eps[j] = innov / math.sqrt(S)
            logL[j] = self._log_gauss_1d(innov, S)

            # update
            K = (P_pred @ H.T) / S  # (n,1)
            m_upd = m_pred + (K[:, 0] * innov)
            P_upd = P_pred - (K @ K.T) * S

            self.m[j] = m_upd
            self.P[j] = self._ensure_pd(P_upd)

        # --- Regime prob update ---
        # mu_t(j) ∝ exp(logL_j) * c_j
        # do in log-space
        logw0 = logL[0] + math.log(c[0])
        logw1 = logL[1] + math.log(c[1])
        mx = max(logw0, logw1)
        w0 = math.exp(logw0 - mx)
        w1 = math.exp(logw1 - mx)
        s = w0 + w1
        self.mu = np.array([w0 / s, w1 / s], dtype=float)

        # --- Aggregate state (recommended: soft output) ---
        mu = self.mu
        m_bar = mu[0] * self.m[0] + mu[1] * self.m[1]

        level = float(m_bar[0])
        filtered_close = math.exp(level) if self.use_log else level

        pi_high = float(mu[1])
        eps_low = float(eps[0])
        eps_high = float(eps[1])
        eps_mix = float(mu[0] * eps_low + mu[1] * eps_high)

        return float(filtered_close), pi_high, eps_low, eps_high, eps_mix

    def run(self, prices: Iterable[float]) -> dict[str, np.ndarray]:
        y = np.asarray(list(prices), dtype=float)
        T = len(y)

        filt = np.empty(T, dtype=float)
        pi_high = np.empty(T, dtype=float)
        eps_low = np.empty(T, dtype=float)
        eps_high = np.empty(T, dtype=float)
        eps_mix = np.empty(T, dtype=float)

        for t in range(T):
            f, ph, el, eh, em = self.update(float(y[t]))
            filt[t] = f
            pi_high[t] = ph
            eps_low[t] = el
            eps_high[t] = eh
            eps_mix[t] = em

        return {
            "observed": y,
            "filtered": filt,
            "pi_high": pi_high,
            "eps_low": eps_low,
            "eps_high": eps_high,
            "eps_mix": eps_mix,
        }


# ----------------------------
# 3) Example main (no-leak backtest plotting)
# ----------------------------

def main():
    # expects `prices` list/array of closes already defined

    data = pd.read_csv(r"C:\Users\Gajic\OneDrive - Université Paris-Dauphine\Trading\Trading-vol-Kalman\get_data\Index\output\SPY\SPY_15m_60d.csv") 
    prices = np.array(data["Close"])
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
    train = prices_arr[:split]
    test = prices_arr[split:]

    # 1) Fit base parameters on TRAIN ONLY (no-leak)
    base = fit_base_local_linear_trend_em(train, use_log=True, em_iter=60)

    # 2) IMM engine (parameters fixed, regime estimated online)
    imm = IMM2Regimes(
        base,
        q_scale_low=0.3, r_scale_low=1.0,
        q_scale_high=3.0, r_scale_high=2.0,
        p00=0.985, p11=0.985,
    )

    # Warm-up on TRAIN (no storing)
    for p in train:
        imm.update(float(p))

    # Run on TEST (store outputs)
    filt_test = np.empty_like(test)
    pi_high_test = np.empty_like(test)
    eps_mix_test = np.empty_like(test)
    eps_low_test = np.empty_like(test)
    eps_high_test = np.empty_like(test)

    for i, p in enumerate(test):
        f, ph, el, eh, em = imm.update(float(p))
        filt_test[i] = f
        pi_high_test[i] = ph
        eps_low_test[i] = el
        eps_high_test[i] = eh
        eps_mix_test[i] = em

    # Quick sanity stats (sur test)
    print("Mean(pi_high):", float(np.mean(pi_high_test)))
    print("Pct high-vol (pi_high>0.5):", float(np.mean(pi_high_test > 0.5)))
    print("eps_mix mean/std:", float(np.mean(eps_mix_test)), float(np.std(eps_mix_test)))
    print("eps_low mean/std:", float(np.mean(eps_low_test)), float(np.std(eps_low_test)))
    print("eps_high mean/std:", float(np.mean(eps_high_test)), float(np.std(eps_high_test)))

    # Align for plotting (NaN before split)
    filt_all = np.full_like(prices_arr, np.nan)
    pi_high_all = np.full_like(prices_arr, np.nan)
    eps_mix_all = np.full_like(prices_arr, np.nan)

    filt_all[split:] = filt_test
    pi_high_all[split:] = pi_high_test
    eps_mix_all[split:] = eps_mix_test

    print("Dernier obs       :", float(prices_arr[-1]))
    print("Dernier filt (IMM):", float(filt_test[-1]))
    print("Dernier pi_high   :", float(pi_high_test[-1]))
    print("Dernier eps_mix   :", float(eps_mix_test[-1]))

    plt.figure()
    plt.plot(prices_arr, label="Close observé")
    plt.plot(filt_all, label="Close filtré (IMM, x_{t|t}, no-leak, test only)")
    plt.axvline(split, linestyle="--", label="Split train/test")
    plt.legend()
    plt.title("IMM 2 régimes — close filtré (no-leak)")
    plt.show()

    plt.figure()
    plt.plot(pi_high_all, label="P(régime high-vol)")
    plt.axvline(split, linestyle="--", label="Split train/test")
    plt.legend()
    plt.title("IMM 2 régimes — probabilité de régime (no-leak)")
    plt.show()

    plt.figure()
    plt.plot(eps_mix_all, label="ε_t mix (innovation normalisée)")
    plt.axvline(split, linestyle="--", label="Split train/test")
    plt.legend()
    plt.title("IMM 2 régimes — innovation normalisée (no-leak)")
    plt.show()

    plt.figure()
    plt.plot(eps_low_test, label="ε low (standardized innov)")
    plt.plot(eps_high_test, label="ε high (standardized innov)")
    plt.legend()
    plt.title("IMM — ε par régime (test)")
    plt.show()

    plt.figure()
    plt.plot(eps_mix_test, label="ε mix")
    plt.legend()
    plt.title("IMM — ε mix (test)")
    plt.show()

    maskH = pi_high_test > 0.5
    maskL = ~maskH
    print("eps_mix mean/std:", eps_mix_test.mean(), eps_mix_test.std())
    print("eps_low  std | low :", eps_low_test[maskL].std(), "  (n=", maskL.sum(), ")")
    print("eps_high std | high:", eps_high_test[maskH].std(), " (n=", maskH.sum(), ")")


if __name__ == "__main__":
    main()
