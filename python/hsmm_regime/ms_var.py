"""
Markov-Switching VAR (MS-VAR) for joint regime-conditioned dynamics of
asset returns and macro/cross-asset factors, plus impulse response
functions (IRFs).

Design note — why this is a separate module from the C++ engine
------------------------------------------------------------------
The C++ HSMM engine (`hsmm_regime._core`) is deliberately univariate: its
emission model is a scalar Student-t per regime (see `types.hpp`). Regimes
there are inferred from a single series (typically log returns or realized
vol) via the forward-backward/EM/particle-filter machinery already built
and tested in cpp/.

MS-VAR is a genuinely different, genuinely multivariate model: instead of
"regime k emits a scalar", it says "regime k has its own VAR(p) system
(intercept c_k, coefficient matrices A_k(1..p), and residual covariance
Sigma_k) governing the joint dynamics of several series at once" (Hamilton
1989/1994; Krolzig 1997). Fitting a *new* regime sequence via full MS-VAR
EM (Hamilton filter + Kim smoother jointly with VAR coefficient
regression) is its own substantial undertaking.

The pragmatic and honestly-documented choice made here: reuse the
already-fitted, already-validated HSMM regime path (from
`hsmm_regime.decode_regimes`, running on the univariate return series with
the full explicit-duration machinery) as the *given* regime assignment,
and condition VAR estimation on it. This is a standard two-step approach
in the regime-switching literature (e.g. fitting "regime-conditional VARs"
given an external state indicator) and lets us reuse the HSMM's superior
duration modeling rather than reimplementing a weaker geometric-duration
Hamilton filter for the multivariate case. It is *not* full joint MS-VAR
estimation (where the VAR likelihood would also inform the regime
posterior) — see the module docstring warning in `fit_ms_var` for the
practical implication of this simplification.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RegimeVARParams:
    """Fitted VAR(p) parameters for a single regime."""

    intercept: np.ndarray       # (n_vars,)
    coefficients: list[np.ndarray]  # length p, each (n_vars, n_vars); coefficients[l] multiplies lag l+1
    sigma: np.ndarray           # (n_vars, n_vars) residual covariance
    n_obs: int                  # number of observations used to fit this regime's VAR
    lag_order: int


@dataclass
class MSVARResult:
    regime_params: dict[int, RegimeVARParams]
    variable_names: list[str]
    lag_order: int
    regime_path: np.ndarray     # length T, aligned to the data used
    residuals: dict[int, np.ndarray]  # per-regime residuals, for diagnostics/PPCs


def _build_lagged_design(data: np.ndarray, lag_order: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the standard VAR(p) regression design: for T x n_vars data,
    returns (Y, X) where Y is (T - p) x n_vars (the "current" values) and
    X is (T - p) x (1 + p*n_vars) (a constant column plus p lags of every
    variable, most-recent lag first), so intercept/coefficients can be
    recovered via ordinary least squares per regime.
    """
    T, n_vars = data.shape
    if T <= lag_order:
        raise ValueError(f"Need more than {lag_order} observations, got {T}")

    Y = data[lag_order:]
    X_parts = [np.ones((T - lag_order, 1))]
    for l in range(1, lag_order + 1):
        X_parts.append(data[lag_order - l : T - l])
    X = np.hstack(X_parts)
    return Y, X


def fit_ms_var(
    data: pd.DataFrame,
    regime_path: np.ndarray,
    lag_order: int = 1,
    min_obs_per_regime: int = 20,
    ridge: float = 1e-6,
) -> MSVARResult:
    """
    Fit a separate VAR(lag_order) per regime, conditioning on a given
    regime path (see module docstring for why this is regime-*conditioned*
    VAR rather than full joint MS-VAR estimation).

    `data` must be T x n_vars (e.g. columns = [asset_log_return,
    macro_factor_change, ...]), aligned in time with `regime_path` (length
    T, integer state labels in [0, K)).

    IMPORTANT — look-ahead caveat: `regime_path` here is the *smoothed*
    (offline, full-sample) Viterbi/gamma-based path from the HSMM, which
    uses future information to label the past (that is the whole point of
    smoothing). This is appropriate for *historical* regime-conditional
    dynamics analysis (the Phase 2/deliverable use case: "what do returns
    and macro factors do together during a crash regime, historically?"),
    but a VAR fit this way must NOT be fed positions/signals into the
    Phase 4 backtester the way the online particle-filter posterior is —
    that would reintroduce exactly the look-ahead bug the backtester's
    lag discipline (see strategy.py) is built to prevent. Use
    `run_regime_backtest`'s online-posterior input for anything
    tradeable; use this module for descriptive/impulse-response analysis.

    Regimes with fewer than `min_obs_per_regime` observations raise a
    ValueError rather than silently returning a degenerate/overfit VAR —
    a K-regime split on daily data can easily starve a rare high-vol
    regime of enough rows to estimate n_vars^2 * lag_order coefficients.
    """
    if len(regime_path) != len(data):
        raise ValueError(
            f"regime_path length ({len(regime_path)}) must match data length ({len(data)})"
        )

    variable_names = list(data.columns)
    n_vars = len(variable_names)
    values = data.to_numpy(dtype=float)

    Y_full, X_full = _build_lagged_design(values, lag_order)
    # regime_path is aligned to the *original* data; after building the
    # lagged design we drop the first `lag_order` rows, so align the
    # regime labels the same way. We label each regression row by the
    # regime of its *current* (dependent-variable) timestep, which is the
    # standard convention: "the VAR active while generating y_t".
    aligned_regimes = regime_path[lag_order:]

    unique_states = sorted(set(int(s) for s in aligned_regimes))
    regime_params: dict[int, RegimeVARParams] = {}
    residuals: dict[int, np.ndarray] = {}

    for k in unique_states:
        mask = aligned_regimes == k
        n_obs_k = int(mask.sum())
        if n_obs_k < min_obs_per_regime:
            raise ValueError(
                f"Regime {k} has only {n_obs_k} observations after lag alignment "
                f"(< min_obs_per_regime={min_obs_per_regime}). Either fit fewer "
                f"regimes, use a longer sample, or lower min_obs_per_regime "
                f"(at the cost of a noisier VAR estimate for that regime)."
            )

        Y_k = Y_full[mask]
        X_k = X_full[mask]

        # Ridge-regularized OLS (closed form), one regression per equation
        # via the standard multivariate-OLS trick: coefficients (for all
        # n_vars equations at once) = (X'X + ridge*I)^-1 X'Y, applied
        # column-wise. A small ridge term guards against near-singular
        # X'X when a regime has few observations relative to n_vars^2 *
        # lag_order free parameters, without meaningfully biasing
        # well-conditioned regimes.
        XtX = X_k.T @ X_k
        XtX_reg = XtX + ridge * np.eye(XtX.shape[0])
        beta = np.linalg.solve(XtX_reg, X_k.T @ Y_k)  # (1 + p*n_vars) x n_vars

        intercept = beta[0, :]
        coefficients = [beta[1 + l * n_vars : 1 + (l + 1) * n_vars, :].T for l in range(lag_order)]

        fitted = X_k @ beta
        resid = Y_k - fitted
        # Degrees-of-freedom-corrected covariance where possible; falls
        # back to the biased (n) divisor if a regime is right at the
        # min_obs_per_regime floor to avoid inflating Sigma unreasonably.
        dof = max(n_obs_k - X_k.shape[1], 1)
        sigma = (resid.T @ resid) / dof

        regime_params[k] = RegimeVARParams(
            intercept=intercept,
            coefficients=coefficients,
            sigma=sigma,
            n_obs=n_obs_k,
            lag_order=lag_order,
        )
        residuals[k] = resid

    return MSVARResult(
        regime_params=regime_params,
        variable_names=variable_names,
        lag_order=lag_order,
        regime_path=aligned_regimes,
        residuals=residuals,
    )


def _companion_form(params: RegimeVARParams, n_vars: int) -> np.ndarray:
    """
    Build the VAR(p) companion matrix so IRFs can be computed via repeated
    matrix powers rather than re-deriving the MA(infinity) representation
    by hand: for y_t = c + A_1 y_{t-1} + ... + A_p y_{t-p} + e_t, the
    companion matrix F stacks [y_t; y_{t-1}; ...; y_{t-p+1}] into a single
    VAR(1) system Y_t = C + F Y_{t-1} + E_t, and the top-left n_vars x
    n_vars block of F^h gives the response at horizon h.
    """
    p = params.lag_order
    F = np.zeros((n_vars * p, n_vars * p))
    for l, A_l in enumerate(params.coefficients):
        F[0:n_vars, l * n_vars : (l + 1) * n_vars] = A_l
    if p > 1:
        F[n_vars:, : n_vars * (p - 1)] = np.eye(n_vars * (p - 1))
    return F


def compute_impulse_responses(
    result: MSVARResult,
    regime: int,
    horizon: int = 20,
    orthogonalize: bool = True,
) -> np.ndarray:
    """
    Compute impulse response functions for a given regime's fitted VAR:
    the response of every variable to a one-standard-deviation shock in
    each variable, at horizons 0..horizon.

    Returns an array of shape (horizon + 1, n_vars, n_vars), where
    result[h, i, j] is the response of variable i at horizon h to a shock
    in variable j at time 0.

    If `orthogonalize`, shocks are identified via a Cholesky decomposition
    of the regime's residual covariance (the standard, if identification-
    order-dependent, choice — see the docstring warning below). If False,
    returns the reduced-form (non-orthogonalized) MA coefficients, i.e.
    the response to a raw one-unit shock in each residual, which avoids
    the ordering-dependence issue but conflates contemporaneously
    correlated shocks.

    IMPORTANT — Cholesky ordering caveat: orthogonalized IRFs from a
    Cholesky decomposition depend on the *order* of columns in `data`
    passed to `fit_ms_var` (the first variable is assumed to affect all
    others contemporaneously but not vice versa, the second affects all
    but the first, etc.). This is the standard simplifying assumption in
    applied MS-VAR work, but it is an assumption, not a result — different
    orderings can give different contemporaneous responses. Report the
    ordering used alongside any IRF plot.
    """
    if regime not in result.regime_params:
        raise KeyError(f"No fitted VAR for regime {regime}; available: {list(result.regime_params)}")

    params = result.regime_params[regime]
    n_vars = len(result.variable_names)
    p = params.lag_order

    if orthogonalize:
        # Cholesky of Sigma gives the "structural" impact matrix B0 such
        # that B0 @ B0.T = Sigma; a unit structural shock in variable j
        # then has contemporaneous impact B0[:, j] on all variables.
        try:
            B0 = np.linalg.cholesky(params.sigma)
        except np.linalg.LinAlgError as e:
            raise np.linalg.LinAlgError(
                f"Regime {regime}'s residual covariance is not positive-definite "
                f"(likely too few observations relative to n_vars for a stable "
                f"covariance estimate); cannot Cholesky-orthogonalize. Consider "
                f"raising min_obs_per_regime or increasing ridge in fit_ms_var()."
            ) from e
    else:
        B0 = np.eye(n_vars)

    F = _companion_form(params, n_vars)
    responses = np.zeros((horizon + 1, n_vars, n_vars))

    F_power = np.eye(n_vars * p)
    for h in range(horizon + 1):
        phi_h = F_power[:n_vars, :n_vars]  # reduced-form MA coefficient at horizon h
        responses[h] = phi_h @ B0
        F_power = F_power @ F

    return responses


def compare_regime_dynamics(
    result: MSVARResult,
    horizon: int = 20,
    orthogonalize: bool = True,
) -> dict[int, np.ndarray]:
    """Convenience wrapper: compute IRFs for every fitted regime, so callers
    can directly compare e.g. how a return shock propagates in a
    high-vol/crash regime vs. a low-vol/bull regime."""
    return {
        k: compute_impulse_responses(result, k, horizon=horizon, orthogonalize=orthogonalize)
        for k in result.regime_params
    }
