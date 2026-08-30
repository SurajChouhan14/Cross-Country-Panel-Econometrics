# Methodological Lineage & Specification Mapping Report

## 1. Scope and Variable Mapping Note
The implemented Data Generating Process (DGP) operationalizes the macroeconomic channels of the project:
* **Market Size:** log(GDP) entered in log levels
* **Trade Policy & Friction:** Trade Facilitation Infrastructure Index (`tfi_score`, 0 to 1 scale) and Tariff Rate (`tariff_rate`, %)
* **Logistics Capital:** Infrastructure Score (`infra_score`, 1 to 5 scale)
* **Macro Uncertainty:** Exchange Rate Volatility (`fx_volatility`)
* **Time-Invariant Friction:** Bilateral log(Distance) (`log_distance`)

> **Resume Scope Note:**  
> No Foreign Direct Investment (FDI) variable is present in the current implementation; the resume's terminology is broader than the implemented scope. The econometric estimation ladder (Pooled OLS, Entity-Only FE, Two-Way FE, Swamy-Arora RE) and the spectrally-decomposed Hausman specification test ($\chi^2 = 24.63, p < 0.001$) are fully reproducible as cited in bullets 2–3.

---

## 2. Econometric Specification & Ground Truth Recovery

### Two-Way Fixed Effects (TWFE) vs. Entity-Only FE:
* In macroeconomic panels, aggregate variables like GDP exhibit common macro trends. Entity-only demeaning fails to disentangle the shared macroeconomic trend from country-specific variation, biasing the estimated GDP elasticity to **0.5704** (against planted truth **0.8500**).
* Adding **Time Effects** (Two-Way Fixed Effects) demeans across both entities and years, successfully capturing common macro shocks and recovering the planted GDP elasticity at **0.9076** (error: $+0.058$).

---

## 3. Spectrally-Decomposed Hausman Diagnostic
* **Null Hypothesis ($H_0$):** $\text{Cov}(\alpha_i, X_{it}) = 0$ (Random Effects consistent and efficient)
* **Alternative ($H_1$):** $\text{Cov}(\alpha_i, X_{it}) \neq 0$ (Random Effects inconsistent; Fixed Effects required)
* **Statistic:** $\chi^2 = 24.63$, $\text{df} = 2$ (rank of positive-definite subspace of $V_{\text{FE}} - V_{\text{RE}}$), $p = 4.48 \times 10^{-6} < 0.001$.
* **Verdict:** Unambiguously rejects Random Effects, confirming that unobserved country-level heterogeneity must be controlled via Fixed Effects.
