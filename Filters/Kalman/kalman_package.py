import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
import pandas as pd

EPS = 1e-10


def _as_obs_log_prices(prices):
    y = np.asarray(prices, dtype=float)
    if y.ndim != 1 or len(y) < 10:
        raise ValueError("`prices` doit être une liste/array 1D (>= 10 points).")
    if np.any(~np.isfinite(y)):
        y = y.copy()
        y[~np.isfinite(y)] = np.nan
    if np.any(y <= 0):
        raise ValueError("Les prix doivent être strictement positifs pour passer en log.")
    z = np.log(y)
    obs = ma.masked_invalid(z.reshape(-1, 1))  # (T, 1)
    return y, z, obs


def _make_pd(mat, jitter=1e-9):
    """Force une matrice symétrique PD (utile après EM si dégénère)."""
    m = np.asarray(mat, dtype=float)
    m = 0.5 * (m + m.T)
    # Jitter sur la diagonale
    m = m + jitter * np.eye(m.shape[0])
    return m


def _fit_candidate(kf: KalmanFilter, obs, em_vars, n_iter=40):
    # EM pour estimer les paramètres listés
    kf_fit = kf.em(obs, n_iter=n_iter, em_vars=em_vars)

    # Sécuriser PD sur covariances
    if kf_fit.transition_covariance is not None:
        kf_fit.transition_covariance = _make_pd(kf_fit.transition_covariance)
    if kf_fit.observation_covariance is not None:
        # obs_cov est 1x1 ici, mais on garde générique
        kf_fit.observation_covariance = _make_pd(np.atleast_2d(kf_fit.observation_covariance))

    ll = float(kf_fit.loglikelihood(obs))
    return kf_fit, ll


def _aic(ll, k_params):
    return 2.0 * k_params - 2.0 * ll


def _candidates_initial_kfs(z):
    """
    Retourne une liste de (name, kf, em_vars, k_params_approx, state_dim, obs_matrix_level_index)
    Tous les modèles observent le "level" du state (ou unique state).
    """
    T = len(z)
    z0 = float(z[0])
    var0 = float(np.var(z[: min(T, 50)]) if T >= 5 else 1.0)
    var0 = max(var0, 1e-6)

    # Heuristiques initiales
    r0 = max(var0 * 0.2, 1e-6)
    q0 = max(var0 * 0.02, 1e-8)

    candidates = []

    # 1) Local Level: x_t = x_{t-1} + w
    kf_ll = KalmanFilter(
        transition_matrices=np.array([[1.0]]),
        observation_matrices=np.array([[1.0]]),
        transition_covariance=np.array([[q0]]),
        observation_covariance=np.array([[r0]]),
        initial_state_mean=np.array([z0]),
        initial_state_covariance=np.array([[var0]]),
    )
    # paramètres libres approx: Q(1) + R(1) + mu0(1) + P0(1) = 4
    candidates.append(("local_level", kf_ll,
                       ["transition_covariance", "observation_covariance",
                        "initial_state_mean", "initial_state_covariance"],
                       4, 1))

    # 2) Local Linear Trend: [level, slope]
    # level_t = level_{t-1} + slope_{t-1} + w1
    # slope_t = slope_{t-1} + w2
    A = np.array([[1.0, 1.0],
                  [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = np.array([[q0, 0.0],
                  [0.0, q0 * 0.1]])
    R = np.array([[r0]])

    kf_tr = KalmanFilter(
        transition_matrices=A,
        observation_matrices=H,
        transition_covariance=Q,
        observation_covariance=R,
        initial_state_mean=np.array([z0, 0.0]),
        initial_state_covariance=np.array([[var0, 0.0],
                                           [0.0, var0]]),
    )
    # libres approx: Q sym 2x2 (3) + R(1) + mu0(2) + P0 sym 2x2 (3) = 9
    candidates.append(("local_linear_trend", kf_tr,
                       ["transition_covariance", "observation_covariance",
                        "initial_state_mean", "initial_state_covariance"],
                       9, 2))

    # 3) AR(1) latent (mean-reverting en log): x_t = phi x_{t-1} + w
    # On laisse EM estimer phi via transition_matrices.
    phi0 = 0.995
    kf_ar1 = KalmanFilter(
        transition_matrices=np.array([[phi0]]),
        observation_matrices=np.array([[1.0]]),
        transition_covariance=np.array([[q0]]),
        observation_covariance=np.array([[r0]]),
        initial_state_mean=np.array([z0]),
        initial_state_covariance=np.array([[var0]]),
    )
    # libres approx: phi(1) + Q(1) + R(1) + mu0(1) + P0(1) = 5
    candidates.append(("ar1_latent", kf_ar1,
                       ["transition_matrices", "transition_covariance",
                        "observation_covariance", "initial_state_mean",
                        "initial_state_covariance"],
                       5, 1))

    return candidates


def best_kalman_underlying_from_prices(prices, em_iter=60, criterion="aic"):
    """
    Retourne le meilleur modèle (parmi 3) calibré par EM sur log-prix,
    et le sous-jacent latent en prix via smoothed state.
    """
    y, z, obs = _as_obs_log_prices(prices)
    candidates = _candidates_initial_kfs(z)

    fits = []
    for name, kf, em_vars, k_params, state_dim in candidates:
        kf_fit, ll = _fit_candidate(kf, obs, em_vars=em_vars, n_iter=em_iter)
        aic = _aic(ll, k_params)
        fits.append((name, kf_fit, ll, aic, k_params, state_dim))

    # Sélection
    if criterion.lower() != "aic":
        raise ValueError("criterion supporté: 'aic' uniquement (ici).")

    fits.sort(key=lambda x: x[3])  # tri sur AIC
    best_name, best_kf, best_ll, best_aic, best_k, best_dim = fits[0]

    # Filter + Smooth
    filt_means, filt_covs = best_kf.filter(obs)
    sm_means, sm_covs = best_kf.smooth(obs)

    # Le niveau latent = state[0] (log-prix)
    log_under_filt = np.asarray(filt_means[:, 0], dtype=float)
    log_under_sm = np.asarray(sm_means[:, 0], dtype=float)
    var_log_sm = np.asarray(sm_covs[:, 0, 0], dtype=float)

    underlying_filt = np.exp(log_under_filt)
    underlying_sm = np.exp(log_under_sm)

    # IC approx en log (±2σ), transformé en prix
    std_log = np.sqrt(np.maximum(var_log_sm, 0.0))
    lo = np.exp(log_under_sm - 2.0 * std_log)
    hi = np.exp(log_under_sm + 2.0 * std_log)

    return {
        "best_model": best_name,
        "best_ll": best_ll,
        "best_aic": best_aic,
        "kf": best_kf,
        "observed": y,
        "underlying_filtered": underlying_filt,
        "underlying_smoothed": underlying_sm,
        "ci_low": lo,
        "ci_high": hi,
        "all_models": [
            {"name": n, "loglik": ll, "aic": aic, "k_params": k, "state_dim": d}
            for (n, _, ll, aic, k, d) in fits
        ],
    }


def main():
    # On suppose que `prices` est déjà défini (liste des prix S&P 500)
    data = pd.read_csv(r"C:\Users\Gajic\OneDrive - Université Paris-Dauphine\Trading\Trading-vol-Kalman\get_data\Index\output\SPY\SPY_15m_60d.csv") 
    prices = list(np.array(data["Close"]))
    try:
        prices  # noqa: F821
    except NameError:
        raise NameError(
            "Définis d’abord `prices = [...]` (liste de prix S&P 500) avant d’exécuter."
        )

    res = best_kalman_underlying_from_prices(prices, em_iter=80, criterion="aic")

    print("=== Model selection (AIC) ===")
    for m in res["all_models"]:
        print(f"{m['name']:>18s} | AIC={m['aic']:.2f} | ll={m['loglik']:.2f} | k~{m['k_params']} | dim={m['state_dim']}")
    print("\nBEST =", res["best_model"])
    print("Dernier obs      :", res["observed"][-1])
    print("Dernier sous-jacent (smooth):", res["underlying_smoothed"][-1])

    y = res["observed"]
    x = res["underlying_smoothed"]
    lo, hi = res["ci_low"], res["ci_high"]

    plt.figure()
    plt.plot(y, label="Prix observé")
    plt.plot(x, label=f"Sous-jacent Kalman (smooth) — {res['best_model']}")
    plt.fill_between(np.arange(len(y)), lo, hi, alpha=0.2, label="IC ~95% (approx)")
    plt.legend()
    plt.title("S&P 500 — Sous-jacent latent (pykalman + EM + sélection AIC)")
    plt.show()


if __name__ == "__main__":
    main()
