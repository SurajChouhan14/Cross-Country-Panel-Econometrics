# 🌐 Longitudinal Panel Econometrics & Policy Estimation Engine
### Pooled OLS | Fixed Effects (Within & CRVE) | Random Effects (Swamy-Arora FGLS) | Hausman Specification Test | linearmodels

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Econometrics](https://img.shields.io/badge/Econometrics-linearmodels%20%2F%20statsmodels-success.svg)](https://bashtage.github.io/linearmodels/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An econometric estimation engine analyzing **120 countries across 10 annual periods (1,200 longitudinal observations)**. Implements Fixed Effects with Liang-Zeger Cluster-Robust Standard Errors (CRVE), Random Effects via Swamy-Arora FGLS, and spectrally-decomposed Hausman specification tests using `linearmodels`.

---

## 📌 Econometric Methodology & Hausman Specification Test

### 1. Fixed Effects (Within-Transformation):
$$(y_{it} - \bar{y}_i) = (\mathbf{x}_{it} - \bar{\mathbf{x}}_i)^T \boldsymbol{\beta} + (\epsilon_{it} - \bar{\epsilon}_i)$$
* Sweeps out unobserved, time-invariant entity heterogeneity $\alpha_i$, eliminating omitted variable bias.

### 2. Spectrally-Decomposed Hausman Test:
$$H = (\hat{\boldsymbol{\beta}}_{\text{FE}} - \hat{\boldsymbol{\beta}}_{\text{RE}})^T \left[\hat{\mathbf{V}}_{\text{FE, homo}} - \hat{\mathbf{V}}_{\text{RE}}\right]^+ (\hat{\boldsymbol{\beta}}_{\text{FE}} - \hat{\boldsymbol{\beta}}_{\text{RE}})$$
* **Test Statistic:** $\mathbf{\chi^2 = 24.63 \; (p < 0.001, \text{df}=2)}$.
* **Verdict:** Reject $H_0 \implies$ Random Effects is inconsistent due to endogeneity; **Fixed Effects is validated as the consistent estimator**.

---

## 📊 Empirical Estimation Results
| Regressor | Pooled OLS (SE) | Fixed Effects CRVE (SE) | Swamy-Arora RE (SE) |
|---|:---:|:---:|:---:|
| **TFI Score** | $1.5082$ ($0.3788$) | **$1.2584$ ($0.1671$)** | $1.1800$ ($0.1343$) |
| **Log GDP** | $0.8687$ ($0.0244$) | **$0.5704$ ($0.1626$)** | $0.8219$ ($0.0677$) |
| **Tariff Rate** | $-0.0277$ ($0.0163$) | **$-0.0342$ ($0.0057$)** | $-0.0311$ ($0.0057$) |
| **Infra Score** | $0.1609$ ($0.1623$) | **$0.2139$ ($0.0630$)** | $0.1842$ ($0.0569$) |
| **FX Volatility** | $-0.3509$ ($0.4844$) | **$-1.0818$ ($0.1772$)** | $-1.0778$ ($0.1710$) |

---

## 📂 Repository Structure
```
Cross-Country-Panel-Econometrics/
├── src/
│   ├── panel_econometric_engine.py # linearmodels FE, RE & Hausman test
│   └── data_loader.py              # Longitudinal panel dataset ingestion
├── Cross_Country_Panel_Econometrics.ipynb # Interactive evaluation notebook
├── run_pipeline.py                 # Pipeline execution script
├── test_panel_econometrics.py      # Unit testing suite (5/5 passing)
└── requirements.txt                # Production dependencies
```

---

## 🚀 Quickstart & Reproducibility
```bash
git clone https://github.com/SurajChouhan14/Cross-Country-Panel-Econometrics.git
cd Cross-Country-Panel-Econometrics
pip install -r requirements.txt
python run_pipeline.py
python -m unittest test_panel_econometrics.py
```
