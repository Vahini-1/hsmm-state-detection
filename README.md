# Bayesian Changepoints & Hidden Semi-Markov Regime Models

A quantitative research framework for identifying structural market breaks and dynamically modeling financial time series using Bayesian inference. This repository extends standard regime-switching models by implementing explicit-duration Hidden Semi-Markov Models (HSMMs) and Markov-Switching Vector Autoregressions (MS-VAR), enabling robust real-time adaptation for algorithmic asset allocation.

## 🧠 Core Methodology

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
├── data/                      # Historical returns, vol surfaces, and macro covariates
├── src/
│   ├── offline/               # RJ-MCMC and DP changepoint algorithms
│   ├── online/                # SMC particle filter implementations
│   ├── models/                # HMM, HSMM, and MS-VAR architectures
│   └── strategy/              # Regime-conditioned backtesting engine
├── notebooks/
│   ├── 01_offline_breaks.ipynb     # Historical structural break analysis
│   ├── 02_hsmm_vs_hmm.ipynb        # Predictive log-likelihood comparisons
│   ├── 03_ms_var_dynamics.ipynb    # MS-VAR fitting and IRF generation
│   └── 04_regime_allocation.ipynb  # Momentum & Risk-Parity backtests
├── requirements.txt
└── README.md
