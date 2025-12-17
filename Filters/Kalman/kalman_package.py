from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Literal, Optional, Tuple

import numpy as np
from pykalman import KalmanFilter

import pandas as pd 

ModelType = Literal["local_level", "local_linear_trend"]


@dataclass(frozen=True)
class KalmanParams:
    model: ModelType
    use_log: bool
    kf: KalmanFilter


def fit_kalman_params_em(
    prices_train: Iterable[float],
    model: ModelType = "local_linear_trend",
    use_log: bool = True,
    em_iter: int = 60,
) -> KalmanParams:
    y = np.asarray(list(prices_train), dtype=float)
    if y.ndim != 1 or len(y) < 30:
        raise ValueError("prices_train doit être 1D et assez long (>= 30).")
    if not np.all(np.isfinite(y)):
        raise ValueError("prices_train contient NaN/inf.")
    if use_log and np.any(y <= 0):
        raise ValueError("Prix <= 0 impossible en log.")

    z = np.log(y) if use_log else y
    obs = z.reshape(-1, 1)

    var0 = float(np.var(z[: min(len(z), 60)]))
    var0 = max(var0, 1e-6)
    r0 = max(var0 * 0.2, 1e-6)
    q0 = max(var0 * 0.02, 1e-8)

    if model == "local_level":
        kf0 = KalmanFilter(
            transition_matrices=np.array([[1.0]]),
            observation_matrices=np.array([[1.0]]),
            transition_covariance=np.array([[q0]]),
            observation_covariance=np.array([[r0]]),
            initial_state_mean=np.array([float(z[0])]),
            initial_state_covariance=np.array([[var0]]),
        )
        em_vars = ["transition_covariance", "observation_covariance",
                   "initial_state_mean", "initial_state_covariance"]

    elif model == "local_linear_trend":
        A = np.array([[1.0, 1.0],
                      [0.0, 1.0]])
        H = np.array([[1.0, 0.0]])
        Q = np.array([[q0, 0.0],
                      [0.0, q0 * 0.1]])
        R = np.array([[r0]])
        kf0 = KalmanFilter(
            transition_matrices=A,
            observation_matrices=H,
            transition_covariance=Q,
            observation_covariance=R,
            initial_state_mean=np.array([float(z[0]), 0.0]),
            initial_state_covariance=np.array([[var0, 0.0],
                                               [0.0, var0]]),
        )
        em_vars = ["transition_covariance", "observation_covariance",
                   "initial_state_mean", "initial_state_covariance"]
    else:
        raise ValueError("model inconnu.")

    kf = kf0.em(obs, n_iter=em_iter, em_vars=em_vars)
    return KalmanParams(model=model, use_log=use_log, kf=kf)


class OnlineKalmanWithInnovation:
    """
    Streaming NO-LEAK:
    - input: close_t
    - output: filtered_close_t (= x_{t|t} en prix),
              innovation_normalized_t (= epsilon_t)
    """

    __slots__ = ("params", "A", "H", "Q", "R", "m", "P")

    def __init__(self, params: KalmanParams):
        self.params = params
        kf = params.kf

        self.A = np.asarray(kf.transition_matrices, dtype=float)          # (n,n)
        self.H = np.asarray(kf.observation_matrices, dtype=float)         # (1,n)
        self.Q = np.asarray(kf.transition_covariance, dtype=float)        # (n,n)
        self.R = np.asarray(kf.observation_covariance, dtype=float)       # (1,1)

        self.m = np.asarray(kf.initial_state_mean, dtype=float).copy()    # (n,)
        self.P = np.asarray(kf.initial_state_covariance, dtype=float).copy()  # (n,n)

        if self.H.shape[0] != 1:
            raise ValueError("Cette implémentation suppose une observation 1D (un close).")

    def reset(self) -> None:
        kf = self.params.kf
        self.m = np.asarray(kf.initial_state_mean, dtype=float).copy()
        self.P = np.asarray(kf.initial_state_covariance, dtype=float).copy()

    def update(self, close_t: float) -> Tuple[float, float, float, float]:
        """
        Retourne:
          filtered_price_t,
          eps_t (innovation normalisée),
          innov_t (en log/prix selon use_log),
          S_t (variance innovation)
        """
        if not np.isfinite(close_t):
            raise ValueError("close_t NaN/inf.")
        if self.params.use_log:
            if close_t <= 0:
                raise ValueError("close_t <= 0 impossible en log.")
            z = math.log(float(close_t))
        else:
            z = float(close_t)

        # 1) Predict: m_pred = A m, P_pred = A P A' + Q
        m_pred = self.A @ self.m
        P_pred = self.A @ self.P @ self.A.T + self.Q

        # 2) Innovation: nu = z - H m_pred ; S = H P_pred H' + R
        y_pred = float((self.H @ m_pred).reshape(()))
        nu = z - y_pred

        S = float((self.H @ P_pred @ self.H.T).reshape(()) + self.R.reshape(()))
        if S <= 0:
            # ultra rare, mais on protège
            S = 1e-12

        eps = nu / math.sqrt(S)

        # 3) Update: K = P_pred H' / S (S scalaire)
        K = (P_pred @ self.H.T) / S  # (n,1)
        self.m = m_pred + (K[:, 0] * nu)
        self.P = P_pred - (K @ K.T) * S

        level = float(self.m[0])
        filtered_price = math.exp(level) if self.params.use_log else level

        return filtered_price, float(eps), float(nu), float(S)

    def run(self, prices: Iterable[float]) -> Tuple[np.ndarray, np.ndarray]:
        y = np.asarray(list(prices), dtype=float)
        filt = np.empty_like(y)
        eps = np.empty_like(y)

        for t, p in enumerate(y):
            fp, e, _, _ = self.update(float(p))
            filt[t] = fp
            eps[t] = e
        return filt, eps


def main():
    # `prices` doit exister : liste/array des closes S&P 500
    data = pd.read_csv(r"C:\Users\Gajic\OneDrive - Université Paris-Dauphine\Trading\Trading-vol-Kalman\get_data\Index\output\SPY\SPY_15m_60d.csv") 
    prices = np.array(data["Close"])
    try:
        prices  # noqa: F821
    except NameError:
        raise NameError("Définis `prices = [...]` avant d'exécuter.")

    prices = np.asarray(prices, dtype=float)

    # Calibration sans leak: uniquement sur un passé (train)
    split = int(0.7 * len(prices))
    train = prices[:split]

    params = fit_kalman_params_em(
        prices_train=train,
        model="local_linear_trend",
        use_log=True,
        em_iter=60,
    )

    # Live/streaming no-leak: x_{t|t} + innovation normalisée
    online = OnlineKalmanWithInnovation(params)
    filtered, eps = online.run(prices)

    print("Dernier close obs :", float(prices[-1]))
    print("Dernier close filt:", float(filtered[-1]))
    print("Dernière eps (z)  :", float(eps[-1]))

    # Plot (optionnel)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(prices, label="Close observé")
    plt.plot(filtered, label="Close filtré (x_{t|t}, no-leak)")
    plt.legend()
    plt.title("S&P 500 — Filtre online no-leak (état filtré)")
    plt.show()

    plt.figure()
    plt.plot(eps, label="Innovation normalisée ε_t")
    plt.legend()
    plt.title("Innovation normalisée (standardized forecast error)")
    plt.show()


if __name__ == "__main__":
    main()
