"""
Econometric Estimation Engine for Longitudinal Panel Data.

Implements industry-standard panel econometrics using linearmodels and statsmodels:
1. Pooled Ordinary Least Squares (statsmodels.api.OLS & linearmodels.panel.PooledOLS)
2. Fixed Effects with Clustered Standard Errors (linearmodels.panel.PanelOLS with EntityEffects and CRVE)
3. Two-Way Fixed Effects (EntityEffects + TimeEffects)
4. Random Effects via Swamy-Arora FGLS (linearmodels.panel.RandomEffects)
5. Spectrally-Decomposed Hausman Specification Test (comparing FE vs RE covariance structures)
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
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
        self.all_vars = self.indep_vars + (['log_distance'] if 'log_distance' in df.columns and 'log_distance' not in self.indep_vars else [])
        
        # Format MultiIndex Panel DataFrame for linearmodels
        self.pdata = self.raw_df.set_index([self.entity_col, self.time_col])
        self.formula_rhs = ' + '.join(self.indep_vars)
        self.formula_all_rhs = ' + '.join(self.all_vars)

        # Cache estimation results to prevent redundant refits
        self._cached_fe_entity = None
        self._cached_fe_twoway = None
        self._cached_re = None
        self._cached_re_sub = None

    def estimate_pooled_ols(self):
        """
        Estimates Pooled Ordinary Least Squares (POLS) on all covariates including distance
        using both statsmodels.api.OLS and linearmodels.panel.PooledOLS.
        """
        formula = f"{self.dep_var} ~ 1 + {self.formula_all_rhs}"
        mod = PooledOLS.from_formula(formula, self.pdata)
        res = mod.fit()

        # Statsmodels baseline verification
        X_sm = sm.add_constant(self.raw_df[self.all_vars])
        y_sm = self.raw_df[self.dep_var]
        sm_model = sm.OLS(y_sm, X_sm).fit()

        return {
            "model": "Pooled OLS (Statsmodels & LinearModels)",
            "coefficients": res.params[self.all_vars].values,
            "intercept": float(res.params['Intercept']),
            "std_errors": res.std_errors[self.all_vars].values,
            "cov_matrix": res.cov.loc[self.all_vars, self.all_vars].values,
            "sigma2": float(res.s2),
            "r_squared": float(res.rsquared),
            "statsmodels_summary": sm_model,
            "summary": res
        }

    def estimate_fixed_effects(self):
        """
        Estimates Entity-Only Fixed Effects (Within Transformation) with
        Cluster-Robust Standard Errors (CRVE) clustered by country.
        """
        if self._cached_fe_entity is not None:
            return self._cached_fe_entity

        formula = f"{self.dep_var} ~ {self.formula_rhs} + EntityEffects"
        mod = PanelOLS.from_formula(formula, self.pdata)
        
        res_clustered = mod.fit(cov_type='clustered', cluster_entity=True)
        res_homo = mod.fit(cov_type='unadjusted')

        self._cached_fe_entity = {
            "model": "Fixed Effects (Entity-Only Within)",
            "coefficients": res_clustered.params.values,
            "std_errors": res_clustered.std_errors.values,
            "cov_matrix": res_clustered.cov.values,
            "cov_matrix_homo": res_homo.cov.values,
            "sigma_e2": float(res_clustered.s2),
            "residuals": res_clustered.resids.values,
            "r_squared": float(res_clustered.rsquared_within),
            "summary": res_clustered
        }
        return self._cached_fe_entity

    def estimate_twoway_fixed_effects(self):
        """
        Estimates Two-Way Fixed Effects (EntityEffects + TimeEffects) with CRVE standard errors.
        Captures common macro shocks and recovers true structural parameters without omitted trend bias.
        """
        if self._cached_fe_twoway is not None:
            return self._cached_fe_twoway

        formula = f"{self.dep_var} ~ {self.formula_rhs} + EntityEffects + TimeEffects"
        mod = PanelOLS.from_formula(formula, self.pdata)
        res = mod.fit(cov_type='clustered', cluster_entity=True)

        self._cached_fe_twoway = {
            "model": "Two-Way Fixed Effects (Entity + Time Effects)",
            "coefficients": res.params.values,
            "std_errors": res.std_errors.values,
            "cov_matrix": res.cov.values,
            "sigma_e2": float(res.s2),
            "r_squared": float(res.rsquared_within),
            "summary": res
        }
        return self._cached_fe_twoway

    def estimate_random_effects(self):
        """
        Estimates Random Effects using Swamy-Arora Feasible Generalized Least Squares (FGLS).
        """
        if self._cached_re is not None:
            return self._cached_re

        formula = f"{self.dep_var} ~ 1 + {self.formula_all_rhs}"
        mod = RandomEffects.from_formula(formula, self.pdata)
        res = mod.fit()

        theta_val = float(res.theta.iloc[0, 0])
        fe_res = self.estimate_fixed_effects()
        sigma_e2 = fe_res["sigma_e2"]
        
        t_periods = self.raw_df[self.time_col].nunique()
        sigma_u2 = max(0.0, float((sigma_e2 / ((1.0 - theta_val)**2) - sigma_e2) / t_periods))

        self._cached_re = {
            "model": "Random Effects (Swamy-Arora FGLS)",
            "coefficients": res.params[self.all_vars].values,
            "intercept": float(res.params['Intercept']),
            "std_errors": res.std_errors[self.all_vars].values,
            "cov_matrix": res.cov.loc[self.all_vars, self.all_vars].values,
            "sigma_e2": float(sigma_e2),
            "sigma_u2": float(sigma_u2),
            "theta": float(theta_val),
            "summary": res
        }
        return self._cached_re

    def run_hausman_specification_test(self):
        """
        Executes the Classical Spectrally-Decomposed Hausman Specification Test.
        Compares Entity-Only Fixed Effects vs Random Effects on the time-varying subspace.
        
        H0: Cov(alpha_i, X_it) = 0 (Random Effects is consistent and efficient)
        H1: Cov(alpha_i, X_it) != 0 (Random Effects is inconsistent; Fixed Effects is required)
        
        Degrees of freedom df = rank of the positive-definite subspace of (V_FE - V_RE).
        """
        fe_res = self.estimate_fixed_effects()
        
        if self._cached_re_sub is None:
            mod_re_sub = RandomEffects.from_formula(f"{self.dep_var} ~ 1 + {self.formula_rhs}", self.pdata)
            self._cached_re_sub = mod_re_sub.fit()
        re_sub = self._cached_re_sub

        b_fe = fe_res["coefficients"]
        b_re = re_sub.params[self.indep_vars].values

        v_fe = fe_res["cov_matrix_homo"]
        v_re = re_sub.cov.loc[self.indep_vars, self.indep_vars].values

        diff_b = b_fe - b_re
        diff_v = v_fe - v_re

        # Spectral Moore-Penrose pseudo-inversion over positive eigenvalues
        eigvals, eigvecs = np.linalg.eigh(diff_v)
        pos_mask = eigvals > 1e-8
        df_deg = int(np.sum(pos_mask))

        if df_deg == 0:
            stat = 0.0
            p_val = 1.0
        else:
            inv_diag = np.diag(1.0 / eigvals[pos_mask])
            proj = eigvecs[:, pos_mask]
            v_pinv = proj @ inv_diag @ proj.T
            stat = float(diff_b.T @ v_pinv @ diff_b)
            p_val = float(1.0 - stats.chi2.cdf(stat, df_deg))

        verdict = "REJECT RE (p < 0.001) -> Fixed Effects unobserved heterogeneity correction is strictly required" if p_val < 0.001 else "FAIL TO REJECT"

        return {
            "test_name": "Spectrally-Decomposed Hausman Specification Test",
            "null_hypothesis": "Cov(alpha_i, X_it) == 0 (Random Effects consistent)",
            "chi2_statistic": round(stat, 2),
            "degrees_of_freedom": df_deg,
            "p_value": p_val,
            "verdict": verdict,
            "fe_coefficients": b_fe,
            "re_coefficients": b_re
        }
