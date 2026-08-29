"""
Econometric Estimation Engine for Longitudinal Panel Data.

Implements industry-standard panel econometrics using linearmodels and statsmodels:
1. Pooled Ordinary Least Squares (linearmodels.panel.PooledOLS)
2. Fixed Effects with Clustered Standard Errors (linearmodels.panel.PanelOLS with EntityEffects and CRVE)
3. Random Effects via Swamy-Arora FGLS (linearmodels.panel.RandomEffects)
4. Classical Asymptotic Hausman Specification Test (comparing FE vs RE covariance structures)
"""

import numpy as np
import pandas as pd
from scipy import stats
from linearmodels.panel import PanelOLS, RandomEffects, PooledOLS


class PanelEconometricEngine:
    """
    Longitudinal Panel Econometric Estimation Engine built with linearmodels and statsmodels.
    """

    def __init__(self, df, entity_col='country_id', time_col='year', dep_var='log_exports', indep_vars=None):
        self.raw_df = df.copy()
        self.entity_col = entity_col
        self.time_col = time_col
        self.dep_var = dep_var
        self.indep_vars = indep_vars or ['tfi_score', 'log_gdp', 'tariff_rate', 'infra_score', 'fx_volatility']
        
        # Format MultiIndex Panel DataFrame for linearmodels
        self.pdata = self.raw_df.set_index([self.entity_col, self.time_col])
        self.formula_rhs = ' + '.join(self.indep_vars)

    def estimate_pooled_ols(self):
        """
        Estimates Pooled Ordinary Least Squares (POLS) using linearmodels.panel.PooledOLS.
        """
        formula = f"{self.dep_var} ~ 1 + {self.formula_rhs}"
        mod = PooledOLS.from_formula(formula, self.pdata)
        res = mod.fit()

        return {
            "model": "Pooled OLS",
            "coefficients": res.params[self.indep_vars].values,
            "intercept": float(res.params['Intercept']),
            "std_errors": res.std_errors[self.indep_vars].values,
            "cov_matrix": res.cov.loc[self.indep_vars, self.indep_vars].values,
            "sigma2": float(res.s2),
            "r_squared": float(res.rsquared),
            "summary": res
        }

    def estimate_fixed_effects(self):
        """
        Estimates Fixed Effects (Within Transformation) with Liang-Zeger / Arellano
        Cluster-Robust Standard Errors (CRVE) using linearmodels.panel.PanelOLS.
        """
        formula = f"{self.dep_var} ~ {self.formula_rhs} + EntityEffects"
        mod = PanelOLS.from_formula(formula, self.pdata)
        
        # Fit with Cluster-Robust Variance-Covariance Estimator (CRVE)
        res_clustered = mod.fit(cov_type='clustered', cluster_entity=True)
        # Fit with unadjusted homoskedastic covariance for Hausman test
        res_homo = mod.fit(cov_type='unadjusted')

        return {
            "model": "Fixed Effects (Within)",
            "coefficients": res_clustered.params.values,
            "std_errors": res_clustered.std_errors.values,
            "cov_matrix": res_clustered.cov.values,
            "cov_matrix_homo": res_homo.cov.values,
            "sigma_e2": float(res_clustered.s2),
            "residuals": res_clustered.resids.values,
            "r_squared": float(res_clustered.rsquared_within),
            "summary": res_clustered
        }

    def estimate_random_effects(self):
        """
        Estimates Random Effects using Swamy-Arora Feasible Generalized Least Squares (FGLS)
        via linearmodels.panel.RandomEffects.
        """
        formula = f"{self.dep_var} ~ 1 + {self.formula_rhs}"
        mod = RandomEffects.from_formula(formula, self.pdata)
        res = mod.fit()

        theta_val = float(res.theta.iloc[0, 0])
        fe_res = self.estimate_fixed_effects()
        sigma_e2 = fe_res["sigma_e2"]
        
        # Calculate sigma_u^2 from Swamy-Arora theta
        t_periods = self.raw_df[self.time_col].nunique()
        sigma_u2 = max(0.0, float((sigma_e2 / ((1.0 - theta_val)**2) - sigma_e2) / t_periods))

        return {
            "model": "Random Effects (Swamy-Arora FGLS)",
            "coefficients": res.params[self.indep_vars].values,
            "intercept": float(res.params['Intercept']),
            "std_errors": res.std_errors[self.indep_vars].values,
            "cov_matrix": res.cov.loc[self.indep_vars, self.indep_vars].values,
            "sigma_e2": float(sigma_e2),
            "sigma_u2": float(sigma_u2),
            "theta": float(theta_val),
            "r_squared": float(res.rsquared),
            "summary": res
        }

    def run_hausman_specification_test(self):
        """
        Executes classical asymptotic Hausman Specification Test:
        H0: Random Effects is consistent and efficient (cov(alpha_i, x_it) = 0)
        H1: Fixed Effects is consistent, Random Effects is inconsistent (endogeneity present)
        
        Uses spectral decomposition on (V_FE_homo - V_RE) to ensure positive semi-definiteness.
        """
        fe = self.estimate_fixed_effects()
        re = self.estimate_random_effects()

        diff_b = fe['coefficients'] - re['coefficients']
        diff_var = fe['cov_matrix_homo'] - re['cov_matrix']

        # Spectral decomposition to isolate positive eigenvalue subspace
        eigvals, eigvecs = np.linalg.eigh(diff_var)
        pos_mask = eigvals > 1e-8
        
        if np.any(pos_mask):
            inv_eig = np.zeros_like(eigvals)
            inv_eig[pos_mask] = 1.0 / eigvals[pos_mask]
            inv_diff_var = eigvecs @ np.diag(inv_eig) @ eigvecs.T
            df_stat = int(np.sum(pos_mask))
            hausman_stat = max(0.0, float(diff_b.T @ inv_diff_var @ diff_b))
        else:
            hausman_stat = 0.0
            df_stat = len(self.indep_vars)

        p_value = float(1.0 - stats.chi2.cdf(hausman_stat, df_stat))
        preferred = "Fixed Effects (Within)" if p_value < 0.05 else "Random Effects (Swamy-Arora FGLS)"

        return {
            "hausman_statistic": round(hausman_stat, 2),
            "degrees_of_freedom": df_stat,
            "p_value": round(p_value, 6),
            "preferred_model": preferred,
            "hypothesis_verdict": "Reject H0 (Presence of Endogeneity)" if p_value < 0.05 else "Fail to Reject H0 (Random Effects Consistent)"
        }
