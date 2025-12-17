from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Literal, Tuple

import numpy as np
from pykalman import KalmanFilter
import pandas as pd

ModelType = Literal["local_level", "local_linear_trend"]


@dataclass(frozen=True)
class KalmanParams:
    model: ModelType
    use_log: bool
    kf: KalmanFilter  # pykalman object with fixed parameters


def fit_kalman_params_em(
    prices_train: Iterable[float],
    model: ModelType = "local_linear_trend",
    use_log: bool = True,
    em_iter: int = 50,
) -> KalmanParams:
    """
    Fit (EM) SUR TRAIN UNIQUEMENT => OK pour live / no-leak.
    Retourne un KalmanFilter paramétré (Q,R, init, etc.).
    """
    y = np.asarray(list(prices_train), dtype=float)
    if y.ndim != 1 or len(y) < 30:
        raise ValueError("prices_train doit être 1D et assez long (>= 30).")
    if not np.all(np.isfinite(y)):
        raise ValueError("prices_train contient NaN/inf.")
    if np.any(y <= 0) and use_log:
        raise ValueError("Prix <= 0: impossible en log.")

    z = np.log(y) if use_log else y
    obs = z.reshape(-1, 1)

    # Heuristiques robustes d'init (juste point de départ EM)
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
        em_vars = [
            "transition_covariance",
            "observation_covariance",
            "initial_state_mean",
            "initial_state_covariance",
        ]

    elif model == "local_linear_trend":
        # State = [level, slope], obs = level
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
        em_vars = [
            "transition_covariance",
            "observation_covariance",
            "initial_state_mean",
            "initial_state_covariance",
        ]
    else:
        raise ValueError("model inconnu.")

    kf = kf0.em(obs, n_iter=em_iter, em_vars=em_vars)

    return KalmanParams(model=model, use_log=use_log, kf=kf)


class OnlineKalmanCloseFilter:
    """
    Filtre ONLINE no-leak: à t, tu fournis close_t => tu obtiens filtered_close_t = x_{t|t}.
    - Aucun smooth
    - Aucun recalibrage sur le futur
    """

    __slots__ = ("params", "_kf", "_m", "_P", "_obs_buf")

    def __init__(self, params: KalmanParams):
        self.params = params
        self._kf = params.kf

        # buffers pour réduire allocations
        self._obs_buf = np.empty((1,), dtype=float)

        # initial state depuis kf (figé)
        self._m = np.array(self._kf.initial_state_mean, dtype=float)
        self._P = np.array(self._kf.initial_state_covariance, dtype=float)

    def reset(self) -> None:
        """Réinitialise l'état (utile si tu relances un backtest)."""
        self._m = np.array(self._kf.initial_state_mean, dtype=float)
        self._P = np.array(self._kf.initial_state_covariance, dtype=float)

    def update(self, close_t: float) -> float:
        """
        Ingestion du close en t -> retourne le close filtré en t (no-leak).
        """
        if not np.isfinite(close_t):
            raise ValueError("close_t NaN/inf.")
        if self.params.use_log:
            if close_t <= 0:
                raise ValueError("close_t <= 0 impossible en log.")
            z = math.log(close_t)
        else:
            z = float(close_t)

        self._obs_buf[0] = z

        # filter_update = (predict + update) sur UNE observation => online, sans futur
        self._m, self._P = self._kf.filter_update(
            filtered_state_mean=self._m,
            filtered_state_covariance=self._P,
            observation=self._obs_buf,
        )

        level = float(self._m[0])  # level latent
        return math.exp(level) if self.params.use_log else level

    def filter_series(self, prices: Iterable[float]) -> np.ndarray:
        """
        Applique le filtre en streaming sur toute la série.
        Sortie: filtered[t] = x_{t|t} (no-leak).
        """
        y = np.asarray(list(prices), dtype=float)
        out = np.empty_like(y)
        for i, p in enumerate(y):
            out[i] = self.update(float(p))
        return out


def main():
    # On suppose que `prices` (S&P 500 closes) existe déjà (liste/array)
    data = pd.read_csv(r"C:\Users\Gajic\OneDrive - Université Paris-Dauphine\Trading\Trading-vol-Kalman\get_data\Index\output\SPY\SPY_15m_60d.csv") 
    prices = np.array(data["Close"])
    try:
        prices  # noqa: F821
    except NameError:
        raise NameError("Définis `prices = [...]` avant d'exécuter.")

    prices = np.asarray(prices, dtype=float)

    # --- 1) Calibration "offline" SUR UN PASSE (train) ---
    # En live: tu calibres une fois avec l'historique disponible (ex: 2-5 ans),
    # puis tu figes les paramètres.
    split = int(0.7 * len(prices))
    train = prices[:split]

    params = fit_kalman_params_em(
        prices_train=train,
        model="local_linear_trend",  # généralement meilleur qu'un simple niveau
        use_log=True,
        em_iter=60,
    )

    # --- 2) Filtrage ONLINE sur toute la série (no-leak dans l'état) ---
    kf_live = OnlineKalmanCloseFilter(params)
    filtered = kf_live.filter_series(prices)

    # Exemple: close filtré à t (dernier point)
    print("Dernier close obs :", float(prices[-1]))
    print("Dernier close filt:", float(filtered[-1]))

    # (Optionnel) plot
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(prices, label="Close observé")
    plt.plot(filtered, label="Close filtré (x_{t|t}, no-leak)")
    plt.legend()
    plt.title("S&P 500 — Filtre Kalman ONLINE (pykalman, no-leak)")
    plt.show()


if __name__ == "__main__":
    main()
