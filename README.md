# Longitudinal Panel Econometrics & Policy Engine
> **Multi-Country Longitudinal Panel Econometrics, Entity & Time Fixed Effects, Swamy-Arora FGLS & Spectrally-Decomposed Hausman Test**  
> *Monte Carlo Macroeconomic Panel DGP (120 Countries × 10 Years, 1,200 Observations)*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Econometrics](https://img.shields.io/badge/Econometrics-Panel%20FE%20%26%20RE-blueviolet.svg)]()
[![Tests](https://img.shields.io/badge/tests-5%20passed-brightgreen.svg)]()
[![Hausman Test](https://img.shields.io/badge/Hausman%20Test-chi2%20%3D%2024.63%20(p%20%3C%200.001)-brightgreen.svg)]()

---

## 🎯 Executive Overview & Econometric Architecture
Cross-country macroeconomic panels suffer from **unobserved country-level heterogeneity** ($lpha_i$, e.g., institutional quality, geographic advantages, legal systems) and **common time shocks** ($\lambda_t$, e.g., global commodity price shocks, worldwide recessions). 

If unobserved country effects correlate with observed macroeconomic regressors ($	ext{Cov}(lpha_i, X_{it}) 
eq 0$), standard **Pooled OLS** and **Random Effects** estimators are structurally biased and inconsistent.

This repository implements a **Longitudinal Panel Econometric Engine** built with `linearmodels` and `NumPy`, evaluating:
1. **Pooled Ordinary Least Squares (POLS)**: Naive baseline ignoring panel structure.
2. **Entity-Only Fixed Effects (Within Estimator)**: Eliminates unobserved time-invariant heterogeneity $lpha_i$ via entity demeaning.
3. **Two-Way Fixed Effects (TWFE)**: Simultaneously controls for country fixed effects $lpha_i$ and common macro shocks $\lambda_t$, eliminating omitted trend bias on trending regressors.
4. **Random Effects (Swamy-Arora FGLS)**: Quasi-demeaned GLS efficient under the orthogonality condition $	ext{Cov}(lpha_i, X_{it}) = 0$.
5. **Spectrally-Decomposed Hausman Specification Test**: Mathematically tests $	ext{Cov}(lpha_i, X_{it}) = 0$ via Moore-Penrose pseudo-inversion over the positive-definite subspace of $(V_{	ext{FE}} - V_{	ext{RE}})$.

```
   y_it = α_i + λ_t + β_1·TFI_it + β_2·log(GDP_it) + β_3·Tariff_it + β_4·Infra_it + β_5·FX_it + γ·Distance_i + ε_it
```

---

## 📊 Ground Truth Recovery & Econometric Specification Results

### Monte Carlo Calibration DGP (120 Countries × 10 Years, 1,200 Observations)

| Regressor Variable | Planted Truth $eta$ | Two-Way FE (TWFE) 🏆<br>*(Entity + Time Effects)* | Entity-Only FE<br>*(Time Effects Omitted)* | Random Effects<br>*(Swamy-Arora FGLS)* | Econometric Interpretation |
|---|:---:|:---:|:---:|:---:|---|
| **Trade Facilitation (TFI)** | **$1.4200$** | **$1.4668$** *(error: $+0.047$)* | $1.2584$ | $1.1752$ | Key trade policy elasticity |
| **log(GDP)** | **$0.8500$** | **$0.9076$** *(error: $+0.058$)* | **$0.5704$** *(error: $-0.280$)* | $0.8358$ | **Omitting time trend biased Entity-FE to 0.57; TWFE restores 0.91!** |
| **Tariff Rate (%)** | **$-0.0400$** | **$-0.0434$** *(error: $-0.003$)* | $-0.0342$ | $-0.0308$ | Negative trade friction effect |
| **Infrastructure Score** | **$0.3500$** | **$0.3237$** *(error: $-0.026$)* | $0.2139$ | $0.1835$ | Positive logistics capital elasticity |
| **Exchange Rate Volatility** | **$-0.8000$** | **$-0.9071$** *(error: $-0.107$)* | $-1.0818$ | $-1.0774$ | Currency uncertainty friction |
| **log(Distance)** *(Time-Invariant)* | **$-0.6500$** | **`[Absorbed / Dropped]`** | **`[Absorbed / Dropped]`** | **$-0.9783$** *(error: $-0.328$)* | **Absorbed by FE country demeaning; estimated with bias in RE** |

---

## 🔬 Key Econometric Findings

1. **Why Two-Way Fixed Effects (TWFE) is Required:**
   * In macroeconomic panels, variables like GDP exhibit common upward trends over time. Entity-only demeaning fails to disentangle the shared macroeconomic trend from country-specific GDP variation, biasing the estimated GDP elasticity downward to **$0.5704$**.
   * By adding **Time Effects** ($\lambda_t$), TWFE demeans across both entities and years, successfully isolating true country-specific variation and recovering the planted elasticity at **$0.9076$** (within $0.058$ of truth).

2. **Time-Invariant Regressor Absorption:**
   * Gravity variables like `log(Distance)` do not vary over time within a country pair ($x_{i,t} - ar{x}_i = 0$).
   * Fixed Effects absorbs distance by mathematical construction, whereas Random Effects estimates it (at $-0.9783$) but produces inconsistent estimates due to correlation with unobserved country traits $lpha_i$.

---

## 🧪 Spectrally-Decomposed Hausman Specification Test

```text
Hausman Test: Comparing Fixed Effects vs. Random Effects Covariance Matrices
  • Null Hypothesis (H0)   : Cov(alpha_i, X_it) = 0 (Random Effects is consistent and efficient)
  • Alternative (H1)       : Cov(alpha_i, X_it) != 0 (Random Effects is biased; Fixed Effects is required)
  • Test Statistic (χ²)    : 24.63
  • Degrees of Freedom (df): 2 (Rank of the positive-definite subspace of V_FE - V_RE)
  • Asymptotic p-value     : 4.48e-06 (p < 0.001)
  • Specification Verdict  : REJECT RE (p < 0.001) -> Fixed Effects unobserved heterogeneity correction is strictly required.
```

---

## ⚡ Quickstart (Under 1 Second Execution)

### 1. Installation
```bash
git clone https://github.com/SurajChouhan14/Cross-Country-Panel-Econometrics.git
cd Cross-Country-Panel-Econometrics
pip install -r requirements.txt
```

### 2. Run Panel Tournament Pipeline
```bash
python run_pipeline.py
```

### 3. Run Unit Test Suite
```bash
python test_panel_econometrics.py
```
