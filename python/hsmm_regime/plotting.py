"""
Plotting utilities for the HSMM regime pipeline.

Covers the Phase 2 validation deliverable (overlay the model's most likely
hidden states on the price/return series) plus general regime diagnostics:
duration distributions, transition matrices, and posterior predictive
checks comparing simulated vs. observed return distributions per regime.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# A colorblind-friendlier qualitative palette; extended/cycled if n_states
# exceeds the palette length rather than raising, since exploratory runs
# often sweep n_states.
_REGIME_COLORS = [
    "#d62728",  # red    - conventionally "high vol / crash"
    "#2ca02c",  # green  - conventionally "low vol / bull"
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#9467bd",  # purple
    "#8c564b",  # brown
]


def _color_for_state(k: int) -> str:
    return _REGIME_COLORS[k % len(_REGIME_COLORS)]


def plot_regime_overlay(
    prices: pd.Series,
    state_path: np.ndarray,
    n_states: int,
    state_labels: dict[int, str] | None = None,
    ax: Axes | None = None,
    title: str = "Price with Decoded Regimes",
) -> Figure:
    """
    Plot a price (or return/vol) series with the background shaded by the
    most-likely regime at each timestep (Viterbi path from
    `hsmm_regime.model.decode_regimes`, or the argmax of posterior gammas).

    `state_path` must be the same length as `prices` and contain integer
    state labels in [0, n_states).
    """
    if len(state_path) != len(prices):
        raise ValueError(
            f"state_path length ({len(state_path)}) must match prices length ({len(prices)})"
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=(13, 5))
    else:
        fig = ax.figure

    ax.plot(prices.index, prices.values, color="black", linewidth=1.0, zorder=3)

    # Shade contiguous regime segments as background spans, rather than
    # per-point scatter coloring, so the chart reads as "regime blocks"
    # per the Phase 2 spec (red = high-vol, green = low-vol, etc.).
    idx = prices.index
    seg_start = 0
    for t in range(1, len(state_path) + 1):
        if t == len(state_path) or state_path[t] != state_path[seg_start]:
            state = int(state_path[seg_start])
            ax.axvspan(
                idx[seg_start],
                idx[t - 1] if t < len(state_path) else idx[-1],
                color=_color_for_state(state),
                alpha=0.15,
                zorder=1,
            )
            seg_start = t

    # Legend built from unique states actually present, not all n_states,
    # so sparsely-visited regimes don't clutter it and mislabeled unused
    # states aren't implied to occur.
    present_states = sorted(set(int(s) for s in state_path))
    handles = []
    for k in present_states:
        label = state_labels.get(k, f"Regime {k}") if state_labels else f"Regime {k}"
        handles.append(plt.Rectangle((0, 0), 1, 1, color=_color_for_state(k), alpha=0.3, label=label))
    ax.legend(handles=handles, loc="upper left")

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(prices.name or "Value")
    fig.tight_layout()
    return fig


def plot_regime_probabilities(
    dates: pd.DatetimeIndex,
    posterior: np.ndarray,
    n_states: int,
    state_labels: dict[int, str] | None = None,
    ax: Axes | None = None,
    title: str = "Posterior Regime Probabilities",
) -> Figure:
    """
    Stacked-area plot of P(z_t = k | data) over time — useful both for the
    offline posterior gammas and for the online particle-filter posterior,
    since both are length-T x K probability matrices.
    """
    if posterior.shape[0] != len(dates):
        raise ValueError("posterior rows must match len(dates)")

    if ax is None:
        fig, ax = plt.subplots(figsize=(13, 3.5))
    else:
        fig = ax.figure

    labels = [
        (state_labels.get(k, f"Regime {k}") if state_labels else f"Regime {k}")
        for k in range(n_states)
    ]
    colors = [_color_for_state(k) for k in range(n_states)]

    ax.stackplot(dates, posterior.T, labels=labels, colors=colors, alpha=0.7)
    ax.set_ylim(0, 1)
    ax.set_ylabel("P(regime)")
    ax.set_title(title)
    ax.legend(loc="upper left", ncol=min(n_states, 4))
    fig.tight_layout()
    return fig


def plot_duration_diagnostics(
    durations,  # list[hsmm._core.DurationParams]
    state_labels: dict[int, str] | None = None,
    max_d_display: int = 60,
) -> Figure:
    """
    Bar/line plot of each regime's fitted duration PMF, for the "regime
    persistence diagnostics" deliverable. Requires the compiled `_core`
    module for `precompute_log_duration_pmf`.
    """
    from . import _core

    n_states = len(durations)
    fig, axes = plt.subplots(1, n_states, figsize=(5 * n_states, 4), squeeze=False)
    axes = axes[0]

    for k, dur_params in enumerate(durations):
        pmf = np.exp(_core.precompute_log_duration_pmf(dur_params))
        d_max = min(max_d_display, len(pmf))
        label = state_labels.get(k, f"Regime {k}") if state_labels else f"Regime {k}"
        axes[k].bar(np.arange(1, d_max + 1), pmf[:d_max], color=_color_for_state(k), alpha=0.8)
        expected_duration = np.sum(np.arange(1, len(pmf) + 1) * pmf)
        axes[k].set_title(f"{label}\nE[duration] \u2248 {expected_duration:.1f} days")
        axes[k].set_xlabel("Duration (trading days)")
        axes[k].set_ylabel("P(duration = d)")

    fig.tight_layout()
    return fig


def plot_transition_matrix(
    transition: np.ndarray,
    state_labels: dict[int, str] | None = None,
    ax: Axes | None = None,
) -> Figure:
    """
    Heatmap of the fitted (off-diagonal, semi-Markov) transition matrix:
    P(next regime = j | current regime = k, k != j).
    """
    n_states = transition.shape[0]
    if ax is None:
        fig, ax = plt.subplots(figsize=(1.2 * n_states + 2, 1.2 * n_states + 1))
    else:
        fig = ax.figure

    im = ax.imshow(transition, cmap="Blues", vmin=0, vmax=1)
    labels = [
        (state_labels.get(k, f"R{k}") if state_labels else f"R{k}") for k in range(n_states)
    ]
    ax.set_xticks(range(n_states))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(n_states))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Next regime")
    ax.set_ylabel("Current regime")

    for i in range(n_states):
        for j in range(n_states):
            val = transition[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                     color="white" if val > 0.5 else "black")

    fig.colorbar(im, ax=ax, label="P(transition)")
    ax.set_title("Regime Transition Matrix")
    fig.tight_layout()
    return fig


def plot_posterior_predictive_check(
    observed: np.ndarray,
    simulated_by_state: dict[int, np.ndarray],
    state_labels: dict[int, str] | None = None,
) -> Figure:
    """
    Overlay histograms of observed returns (for periods assigned to each
    regime) against samples drawn from that regime's fitted emission
    distribution, as a posterior predictive check on emission adequacy.

    `simulated_by_state[k]` should be an array of samples drawn from
    Regime k's fitted Student-t (see model.sample_emission).
    """
    n_states = len(simulated_by_state)
    fig, axes = plt.subplots(1, n_states, figsize=(5 * n_states, 4), squeeze=False)
    axes = axes[0]

    for k, sim in simulated_by_state.items():
        label = state_labels.get(k, f"Regime {k}") if state_labels else f"Regime {k}"
        axes[k].hist(observed, bins=40, density=True, alpha=0.5, label="Observed (all)",
                      color="gray")
        axes[k].hist(sim, bins=40, density=True, alpha=0.5, label="Simulated", color=_color_for_state(k))
        axes[k].set_title(f"{label}: Posterior Predictive Check")
        axes[k].legend()

    fig.tight_layout()
    return fig


def plot_impulse_responses(
    irfs_by_regime: dict[int, "np.ndarray"],
    variable_names: list[str],
    state_labels: dict[int, str] | None = None,
    shock_variable: int | None = None,
) -> Figure:
    """
    Plot impulse response functions across regimes, for the MS-VAR
    deliverable (see hsmm_regime.ms_var.compare_regime_dynamics).

    `irfs_by_regime[k]` has shape (horizon+1, n_vars, n_vars), as returned
    by `compute_impulse_responses`/`compare_regime_dynamics`.

    If `shock_variable` is given, plots only the responses to a shock in
    that one variable (one subplot per response variable, one line per
    regime) -- the typical "how does a macro shock propagate under each
    regime" view. If None, plots the full n_vars x n_vars grid for a
    single regime at a time isn't supported here; call once per
    shock_variable of interest instead, since overlaying regimes is the
    primary comparison this function is built for.
    """
    n_vars = len(variable_names)
    regimes = sorted(irfs_by_regime.keys())
    shocks_to_plot = [shock_variable] if shock_variable is not None else range(n_vars)

    fig, axes = plt.subplots(
        len(list(shocks_to_plot)), n_vars, figsize=(5 * n_vars, 3.5 * len(list(shocks_to_plot))),
        squeeze=False,
    )

    for row, j in enumerate(shocks_to_plot):
        for col, i in enumerate(range(n_vars)):
            ax = axes[row][col]
            for k in regimes:
                horizon = irfs_by_regime[k].shape[0]
                response = irfs_by_regime[k][:, i, j]
                label = state_labels.get(k, f"Regime {k}") if state_labels else f"Regime {k}"
                ax.plot(range(horizon), response, label=label, color=_color_for_state(k),
                         linewidth=1.8)
            ax.axhline(0.0, color="black", linewidth=0.5, linestyle="--")
            ax.set_title(f"Response of {variable_names[i]}\nto {variable_names[j]} shock")
            ax.set_xlabel("Horizon (days)")
            if col == 0:
                ax.legend(fontsize=8)

    fig.tight_layout()
    return fig


def plot_strategy_performance(
    equity_curves: dict[str, pd.Series],
    title: str = "Strategy Performance",
) -> Figure:
    """
    Cumulative-return comparison across strategies (e.g. regime-momentum
    vs buy-and-hold vs trend-following baseline), for the Phase 4
    backtest deliverable.
    """
    fig, ax = plt.subplots(figsize=(13, 5))
    for name, curve in equity_curves.items():
        ax.plot(curve.index, curve.values, label=name, linewidth=1.5)
    ax.set_title(title)
    ax.set_ylabel("Cumulative return (growth of $1)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left")
    ax.axhline(1.0, color="black", linewidth=0.5, linestyle="--")
    fig.tight_layout()
    return fig
