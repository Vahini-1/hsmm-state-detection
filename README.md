# Bayesian Changepoints & Hidden Semi-Markov Regime Models

A quantitative research framework for identifying structural market breaks and dynamically modeling financial time series using Bayesian inference. This repository extends standard regime-switching models by implementing explicit-duration Hidden Semi-Markov Models (HSMMs) and Markov-Switching Vector Autoregressions (MS-VAR), enabling robust real-time adaptation for algorithmic asset allocation.

## Core Methodology

This project implements both retrospective (offline) analysis and real-time (online) inference to model the joint dynamics of asset returns, volatility, and macroeconomic covariates.

### 1. Offline Changepoint Detection
* **Algorithms:** Reversible Jump MCMC (RJ-MCMC) and Dynamic Programming.
* **Purpose:** To identify historical structural breaks in financial time series and macroeconomic data, providing a robust prior for historical regime transitions and volatility shifts.

### 2. Online Changepoint Detection
* **Algorithms:** Sequential Monte Carlo (SMC) / Particle Filtering.
* **Purpose:** To adapt to new regimes in real-time. By recursively updating the state posteriors, the system detects regime shifts as live data arrives without look-ahead bias.

### 3. Hidden Semi-Markov Models (HSMMs)
* Standard HMMs assume memoryless geometric state durations, which systematically fails to capture the persistence of financial regimes. 
* **Enhancement:** Explicitly models regime duration distributions to improve persistence realism: $P(d | z_t = k)$.

### 4. Markov-Switching VAR (MS-VAR)
* **Purpose:** Captures the joint, multivariate dynamics of market returns and macroeconomic factors under different hidden states.
* **Output:** Generates regime-dependent Impulse Response Functions (IRFs) to analyze the impact of macro shocks across different market environments.

## 📂 Repository Structure
```text
hsmm-regime/
├── cpp/
│   ├── include/hsmm/                     
│   │   ├── types.hpp                      # Matrix/Vector, EmissionParams, DurationParams, HSMMParams
│   │   ├── emissions.hpp
│   │   ├── duration_distributions.hpp
│   │   ├── forward_backward.hpp
│   │   ├── em_engine.hpp
│   │   └── particle_filter.hpp
│   ├── src/
│   │   ├── emissions.cpp                  # Student-t log-density + weighted MLE (mu, sigma, nu)
│   │   ├── duration_distributions.cpp     # Negative-Binomial duration PMF/survivor + fit
│   │   ├── forward_backward.cpp           # explicit-duration HSMM forward-backward (Yu 2010)
│   │   ├── em_engine.cpp                  # Baum-Welch-style EM + segmental Viterbi decoding
│   │   ├── particle_filter.cpp            # SMC / particle filter, systematic resampling
│   │   └── bindings.cpp                   # pybind11 module -> hsmm_regime._core
│   └── tests/                             # Catch2 unit tests (see CMakeLists.txt's hsmm_tests target)
│       ├── test_forward_backward.cpp      # incl. brute-force log-likelihood cross-check
│       └── test_particle_filter.cpp       # tracking accuracy, ESS bounds, resampling behavior
├── python/
│   ├── hsmm_regime/                       # Python package (imports the compiled C++ extension)
│   │   ├── __init__.py                    # high-level API: fit_offline, decode_regimes, OnlineFilter, ...
│   │   ├── data.py                        # fetch_data / build_features (returns, realized vol, liquidity)
│   │   ├── strategy.py                    # regime-momentum & risk-parity sizing + no-look-ahead backtester
│   │   ├── plotting.py                    # regime overlays, IRF plots, duration/transition diagnostics
│   │   ├── ms_var.py                      # regime-conditioned VAR + orthogonalized impulse response functions
│   │   ├── gibbs.py                       # FFBS + Metropolis-within-Gibbs sampler (Bayesian alternative to EM)
│   │   └── tests/
│   │       ├── test_backtest.py           # no-look-ahead perturbation tests + metric sanity
│   │       └── test_bayesian_extensions.py # FFBS-vs-C++ cross-validation, Gibbs/MS-VAR recovery tests
│   └── scripts/                           # CLI entry points, one per pipeline phase
│       ├── fetch_data.py                  # Phase 1: pull OHLCV via yfinance
│       ├── build_features.py              # Phase 1: log returns, realized vol, liquidity proxy
│       ├── fit_offline_hsmm.py            # Phase 2: EM fit + Viterbi decode + validation plots
│       ├── fit_gibbs_hsmm.py              # Phase 2b: Bayesian fit via Gibbs sampling, credible intervals
│       ├── run_online_filter.py           # Phase 3: streaming particle filter over history
│       ├── backtest_strategy.py           # Phase 4: regime-conditioned strategy vs. baselines
│       └── fit_ms_var.py                  # Phase 5: regime-conditioned VAR + impulse response functions
├── data/
│   ├── raw/                               # untouched downloaded OHLCV CSVs
│   └── processed/                         # features, fitted models, online posteriors, backtest/MS-VAR results
├── docs/
│   ├── figures/                           # PNGs written by the scripts (regime overlays, IRFs, diagnostics...)
│   ├── RUNNING.md                         # exact step-by-step commands with measured timings
│   └── NEXT_STEPS.md                      # what's needed to move from synthetic-data demo to real analysis
├── cmake/                                 # (reserved for CMake helper modules; none needed currently)
├── CMakeLists.txt
├── pyproject.toml
└── environment.yml
