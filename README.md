# Longitudinal Panel Data Econometrics Engine (NumPy Linear Algebra)

A custom econometric estimation engine developed from scratch in **pure Python and NumPy matrix algebra**, implementing longitudinal panel estimators, cluster-robust standard error sandwich matrices, and a spectrally-decomposed Hausman specification test on a simulated **Monte Carlo Data Generating Process (DGP)**.

---

## 1. Mathematical Architecture

```
                       +---------------------------------------+
                       | Monte Carlo Panel DGP (N=120, T=10)   |
                       +-------------------+-------------------+
                                           |
         +---------------------------------+---------------------------------+
         |                                 |                                 |
         v                                 v                                 v
+------------------+             +-------------------+             +-------------------+
| Pooled OLS       |             | Fixed Effects     |             | Random Effects    |
| (Baseline OLS)   |             | (Within Demean)   |             | (Swamy-Arora FGLS)|
+------------------+             +---------+---------+             +---------+---------+
                                           |                                 |
                                           +----------------+----------------+
                                                            |
                                                            v
                                           +---------------------------------+
                                           | Spectrally Decomposed Hausman   |
                                           | H0 vs H1 (Pos-Def Projection)   |
                                           +---------------------------------+
```

---

## 2. Key Econometric Mechanics

* **Fixed Effects (Within Transformation):** Eliminates time-invariant unobserved individual heterogeneity ($lpha_i$) by subtracting entity-level group means:
  $$y_{it} - ar{y}_i = (x_{it} - ar{x}_i)' eta + (\epsilon_{it} - ar{\epsilon}_i)$$
* **Liang-Zeger / Arellano Cluster-Robust Standard Errors (CRVE):** Computes degree-of-freedom-adjusted cluster-robust sandwich covariance:
  $$\hat{V}_{	ext{CRVE}} = rac{N-1}{N-K-M}rac{M}{M-1} (X'X)^{-1} \left( \sum_{i=1}^M X_i' \hat{u}_i \hat{u}_i' X_i ight) (X'X)^{-1}$$
* **Swamy-Arora Feasible Generalized Least Squares (FGLS):** Dynamically estimates between-entity variance ($\hat{\sigma}_u^2 = 1.4445$) and within-entity variance ($\hat{\sigma}_e^2 = 0.1782$) to calculate the quasi-demeaning parameter ($	heta = 0.8896$):
  $$	heta = 1 - \sqrt{rac{\hat{\sigma}_e^2}{\hat{\sigma}_e^2 + T\hat{\sigma}_u^2}}$$
* **Spectrally-Decomposed Hausman Specification Test:** Projects $V_{	ext{FE, homo}} - V_{	ext{RE}}$ onto its positive semi-definite subspace via spectral decomposition (`np.linalg.eigh`), resolving finite-sample non-positive-definite matrix failures:
  $$H = (\hat{eta}_{	ext{FE}} - \hat{eta}_{	ext{RE}})' \left[ \hat{V}(\hat{eta}_{	ext{FE, homo}}) - \hat{V}(\hat{eta}_{	ext{RE}}) ight]^{+} (\hat{eta}_{	ext{FE}} - \hat{eta}_{	ext{RE}}) \sim \chi^2(K^+)$$

---

## 3. Exact Computed Benchmark Results (120 Entities, 10 Periods)

| Variable Name | Pooled OLS Coef (SE) | Fixed Effects (CRVE) | Swamy-Arora RE (SE) |
| :--- | :---: | :---: | :---: |
| **TFI Score** | 1.5082 (0.3788) | **1.2584 (0.1768)** | 1.1800 (0.1343) |
| **Log GDP** | 0.8687 (0.0244) | **0.5704 (0.1721)** | 0.8219 (0.0677) |
| **Tariff Rate** | -0.0277 (0.0163) | **-0.0342 (0.0061)** | -0.0311 (0.0057) |
| **Infra Score** | 0.1609 (0.1623) | **0.2139 (0.0667)** | 0.1842 (0.0569) |
| **FX Volatility** | -0.3509 (0.4844) | **-1.0818 (0.1875)** | -1.0778 (0.1710) |
| **Within $R^2$** | 0.5264 | **0.2802** | 0.3173 |
| **$\hat{\sigma}_e^2$ / $\hat{\sigma}_u^2$** | - | **0.1782 / -** | 0.1782 / 1.4445 |
| **$	heta$ (Quasi-Demean)** | - | **1.0000** | **0.8896** |

* **Hausman Specification Test:** $\chi^2(2) = 24.63, p < 0.0001 \implies$ **Reject $H_0$** (Statistically significant endogeneity detected; Fixed Effects model is consistent and preferred).

---

## 4. Quick Start & Execution

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run panel econometric estimation pipeline
python run_pipeline.py
```

---

## 5. Master Placement Resume Description

> **Panel Data Econometrics Engine (NumPy Linear Algebra)**
> * Developed a custom econometric estimation pipeline in pure Python and NumPy to estimate longitudinal panel data via Monte Carlo simulation.
> * Implemented Fixed Effects (Within Transformation) and Random Effects (Swamy-Arora FGLS) estimators from scratch using raw linear algebra.
> * Engineered Liang-Zeger Cluster-Robust Standard Errors (CRVE) to account for intra-entity error correlation, alongside a spectrally-decomposed Hausman Specification Test to validate model consistency and handle finite-sample non-positive-definite matrices.

---

## License
MIT License. Open for academic research and portfolio demonstration.
