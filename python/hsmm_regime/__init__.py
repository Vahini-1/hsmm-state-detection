"""
hsmm_regime: Bayesian online changepoint detection and hidden semi-Markov
regime modeling for dynamic asset allocation.

This module re-exports the compiled `_core` C++ extension (forward-backward,
EM, particle filter) and adds a few high-level Python convenience wrappers
that the scripts in python/scripts/ and the plotting module build on top of.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import _core  # compiled pybind11 extension; see cpp/src/bindings.cpp

__all__ = [
    "_core",
    "HSMMConfig",
    "init_params_from_kmeans",
    "fit_offline",
    "decode_regimes",
    "sample_emission",
    "OnlineFilter",
    "regime_label_map",
]


@dataclass
class HSMMConfig:
    """High-level knobs for building an initial HSMMParams and EM run."""

    n_states: int = 3
    max_duration: int = 252
    em_max_iters: int = 200
    em_tol: float = 1e-6
    em_seed: int = 42
    verbose: bool = True


def _make_transition_matrix(n_states: int) -> np.ndarray:
    """Uniform off-diagonal transition matrix (each regime equally likely
    to move to any other regime), zero diagonal since HSMMs handle
    self-persistence via the explicit duration distribution instead."""
    m = np.full((n_states, n_states), 1.0 / (n_states - 1) if n_states > 1 else 0.0)
    np.fill_diagonal(m, 0.0)
    return m


def init_params_from_kmeans(
    observations: np.ndarray,
    config: HSMMConfig,
    random_state: int | None = None,
) -> "_core.HSMMParams":
    """
    Build an initial HSMMParams guess by k-means clustering the (rolling
    volatility-augmented) observations into `n_states` groups, using each
    cluster's mean/std as the initial Student-t location/scale and a
    moderate default duration prior. This gives EM a much better starting
    point than random initialization, particularly for volatility-driven
    regimes where clusters are usually well separated on the raw series
    already.
    """
    try:
        from sklearn.cluster import KMeans
    except ImportError as e:
        raise ImportError(
            "scikit-learn is required for init_params_from_kmeans(); "
            "install via `pip install scikit-learn`, or construct "
            "HSMMParams manually and skip this helper."
        ) from e

    x = np.asarray(observations).reshape(-1, 1)
    km = KMeans(n_clusters=config.n_states, n_init=10, random_state=random_state)
    labels = km.fit_predict(x)

    # Order states by cluster mean so state indices are consistently
    # interpretable across runs (e.g. state 0 = lowest mean/"bear-ish",
    # last state = highest mean), rather than the arbitrary k-means label
    # order which can flip between runs/seeds.
    order = np.argsort(km.cluster_centers_.ravel())
    label_remap = {old: new for new, old in enumerate(order)}
    labels = np.array([label_remap[l] for l in labels])

    params = _core.HSMMParams()
    params.n_states = config.n_states
    params.transition = _make_transition_matrix(config.n_states)

    emissions = []
    durations = []
    for k in range(config.n_states):
        cluster_vals = x[labels == k].ravel()
        if len(cluster_vals) < 2:
            mu, sigma = float(x.mean()), float(x.std() + 1e-6)
        else:
            mu, sigma = float(cluster_vals.mean()), float(cluster_vals.std() + 1e-6)
        emissions.append(_core.EmissionParams(mu, sigma, 5.0))
        # Moderate persistence prior (mean duration ~20 trading days),
        # loose enough for EM to reshape substantially during fitting.
        durations.append(_core.DurationParams(2.0, 2.0 / 22.0, config.max_duration))

    params.emissions = emissions
    params.durations = durations
    params.initial_dist = np.full(config.n_states, 1.0 / config.n_states)

    return params


def fit_offline(
    observations: pd.Series | np.ndarray,
    config: HSMMConfig | None = None,
    init_params: "_core.HSMMParams | None" = None,
) -> "_core.EMResult":
    """
    Fit the offline HSMM via EM. If `init_params` is not supplied, builds
    one via k-means initialization from `config`.
    """
    config = config or HSMMConfig()
    obs = np.asarray(observations, dtype=float)

    if init_params is None:
        init_params = init_params_from_kmeans(obs, config)

    em_config = _core.EMConfig()
    em_config.max_iters = config.em_max_iters
    em_config.tol = config.em_tol
    em_config.seed = config.em_seed
    em_config.verbose = config.verbose

    return _core.fit_hsmm_em(obs, init_params, em_config)


def decode_regimes(
    observations: pd.Series | np.ndarray,
    params: "_core.HSMMParams",
) -> np.ndarray:
    """Segmental Viterbi decoding: most likely regime path."""
    obs = np.asarray(observations, dtype=float)
    return _core.most_likely_state_path(obs, params)


def sample_emission(
    emission_params: "_core.EmissionParams",
    n_samples: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Draw samples from a fitted Student-t emission distribution, for
    posterior predictive checks (see plotting.plot_posterior_predictive_check).
    Uses numpy's standard_t and rescales, matching the C++ side's
    parameterization y = mu + sigma * t_nu.
    """
    rng = rng or np.random.default_rng()
    base = rng.standard_t(emission_params.nu, size=n_samples)
    return emission_params.mu + emission_params.sigma * base


def regime_label_map(params: "_core.HSMMParams") -> dict[int, str]:
    """
    Heuristic, human-readable labels for regimes based on fitted emission
    parameters: highest-volatility state -> 'Crash/High-Vol', lowest mean
    with elevated vol -> etc. This is a convenience default for plotting
    titles/legends; callers with domain knowledge should override with
    their own explicit mapping rather than relying on this heuristic for
    anything that feeds back into strategy logic.
    """
    sigmas = [e.sigma for e in params.emissions]
    mus = [e.mu for e in params.emissions]
    n = len(sigmas)

    high_vol_state = int(np.argmax(sigmas))
    low_vol_high_mean_state = int(np.argmax([
        mus[k] - sigmas[k] if k != high_vol_state else -np.inf for k in range(n)
    ]))

    labels = {k: f"Regime {k}" for k in range(n)}
    labels[high_vol_state] = "High-Vol / Crash"
    if low_vol_high_mean_state != high_vol_state:
        labels[low_vol_high_mean_state] = "Low-Vol / Bull"
    return labels


class OnlineFilter:
    """
    Thin, Pythonic wrapper around _core.ParticleFilter that additionally
    tracks the running posterior history as a DataFrame, which is what
    the plotting and backtest code want rather than raw per-step arrays.
    """

    def __init__(self, params: "_core.HSMMParams", n_particles: int = 2000, seed: int = 42):
        cfg = _core.ParticleFilterConfig()
        cfg.n_particles = n_particles
        cfg.seed = seed
        self._pf = _core.ParticleFilter(params, cfg)
        self.n_states = params.n_states
        self._history: list[np.ndarray] = []
        self._ess_history: list[float] = []

    def step(self, y_t: float) -> np.ndarray:
        posterior = self._pf.step(float(y_t))
        self._history.append(posterior)
        self._ess_history.append(self._pf.effective_sample_size())
        return posterior

    def run(self, observations: pd.Series | np.ndarray) -> pd.DataFrame:
        """Run the filter over a full series, returning a DataFrame of
        P(z_t = k | y_1:t) indexed like the input (if a Series was passed)."""
        obs = np.asarray(observations, dtype=float)
        for y in obs:
            self.step(y)
        arr = np.array(self._history)
        index = observations.index if isinstance(observations, pd.Series) else range(len(obs))
        return pd.DataFrame(arr, index=index, columns=list(range(self.n_states)))

    @property
    def effective_sample_size_history(self) -> list[float]:
        return self._ess_history
