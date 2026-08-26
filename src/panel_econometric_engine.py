"""
Econometric Estimation Engine for Longitudinal Panel Data.

Implements from scratch using pure NumPy matrix algebra:
1. Pooled Ordinary Least Squares (POLS)
2. Fixed Effects (Within Transformation) with Liang-Zeger / Arellano Cluster-Robust Standard Errors (CRVE)
3. Random Effects (Swamy-Arora Feasible Generalized Least Squares / FGLS)
4. Mathematically Correct Classical Hausman Specification Test (using homoskedastic asymptotic variance)
"""

import numpy as np
import pandas as pd
from scipy import stats


class PanelEconometricEngine:
    """
    Longitudinal Panel Econometric Estimation Engine built from scratch in NumPy.
    """

    def __init__(self, df, entity_col='country_id', time_col='year', dep_var='log_exports', indep_vars=None):
        self.df = df.sort_values([entity_col, time_col]).reset_index(drop=True)
        self.entity_col = entity_col
        self.time_col = time_col
        self.dep_var = dep_var
        self.indep_vars = indep_vars or ['tfi_score', 'log_gdp', 'tariff_rate', 'infra_score', 'fx_volatility']

    def estimate_pooled_ols(self):
        """
        Estimates Pooled Ordinary Least Squares (POLS) ignoring panel structure.
        """
        X = np.column_stack([np.ones(len(self.df)), self.df[self.indep_vars].values])
        y = self.df[self.dep_var].values
        beta = np.linalg.inv(X.T @ X) @ X.T @ y
        residuals = y - X @ beta
        dof = len(y) - X.shape[1]
        sigma2 = np.sum(residuals**2) / dof
        cov_matrix = np.linalg.inv(X.T @ X) * sigma2
        se = np.sqrt(np.diagonal(cov_matrix))
        
        ss_tot = np.sum((y - np.mean(y))**2)
        ss_res = np.sum(residuals**2)
        r_squared = 1.0 - (ss_res / ss_tot)
        
        return {
            "model": "Pooled OLS",
            "coefficients": beta[1:],
            "intercept": beta[0],
            "std_errors": se[1:],
            "cov_matrix": cov_matrix[1:, 1:],
            "sigma2": float(sigma2),
            "r_squared": float(r_squared)
        }

    def estimate_fixed_effects(self):
        """
        Estimates Fixed Effects (Within Transformation) and computes
        Liang-Zeger / Arellano Cluster-Robust Variance-Covariance Matrix (CRVE).
        Also computes the classical homoskedastic variance matrix required for the Hausman test.
        """
        entities = self.df[self.entity_col].unique()
        n_entities = len(entities)
        t_periods = self.df[self.time_col].nunique()
        n_obs = len(self.df)
        k_vars = len(self.indep_vars)

        # Entity-level de-meaning
        y_mean = self.df.groupby(self.entity_col)[self.dep_var].transform('mean')
        y_tilde = (self.df[self.dep_var] - y_mean).values

        X_demeaned = []
        for var in self.indep_vars:
            x_mean = self.df.groupby(self.entity_col)[var].transform('mean')
            X_demeaned.append((self.df[var] - x_mean).values)
        X_tilde = np.column_stack(X_demeaned)

        bread = np.linalg.inv(X_tilde.T @ X_tilde)
        beta_fe = bread @ X_tilde.T @ y_tilde
        residuals = y_tilde - X_tilde @ beta_fe
        
        dof_within = n_entities * (t_periods - 1) - k_vars
        sigma_e2 = np.sum(residuals**2) / dof_within

        # Classical Homoskedastic Asymptotic Covariance Matrix (for Hausman Test)
        cov_fe_homo = bread * sigma_e2

        # Liang-Zeger / Arellano Clustered Covariance Sandwich (CRVE)
        meat = np.zeros((k_vars, k_vars))
        for entity in entities:
            mask = (self.df[self.entity_col] == entity).values
            X_i = X_tilde[mask]
            u_i = residuals[mask].reshape(-1, 1)
            score_i = X_i.T @ u_i
            meat += score_i @ score_i.T

        df_adj = (n_obs - 1) / (n_obs - k_vars - n_entities) * (n_entities / (n_entities - 1))
        cov_crve = df_adj * (bread @ meat @ bread)
        se_clustered = np.sqrt(np.diagonal(cov_crve))

        ss_tot_within = np.sum(y_tilde**2)
        ss_res_within = np.sum(residuals**2)
        r_squared_within = 1.0 - (ss_res_within / ss_tot_within)

        return {
            "model": "Fixed Effects (Within)",
            "coefficients": beta_fe,
            "std_errors": se_clustered,
            "cov_matrix": cov_crve,
            "cov_matrix_homo": cov_fe_homo,
            "sigma_e2": float(sigma_e2),
            "residuals": residuals,
            "r_squared": float(r_squared_within)
        }

    def estimate_random_effects(self):
        """
        Estimates Random Effects using Swamy-Arora Feasible Generalized Least Squares (FGLS).
        Calculates between-entity variance sigma_u^2 and quasi-demeaning weight theta.
        """
        fe_res = self.estimate_fixed_effects()
        sigma_e2 = fe_res["sigma_e2"]

        # Between Estimator
        df_between = self.df.groupby(self.entity_col)[[self.dep_var] + self.indep_vars].mean().reset_index()
        y_bar = df_between[self.dep_var].values
        X_bar = np.column_stack([np.ones(len(df_between)), df_between[self.indep_vars].values])
        
        beta_between = np.linalg.inv(X_bar.T @ X_bar) @ X_bar.T @ y_bar
        res_between = y_bar - X_bar @ beta_between
        
        n_entities = len(df_between)
        k_vars = len(self.indep_vars)
        dof_between = n_entities - k_vars - 1
        sigma_between2 = np.sum(res_between**2) / dof_between

        t_periods = self.df[self.time_col].nunique()
        sigma_u2 = max(0.0, float(sigma_between2 - sigma_e2 / t_periods))

        # Swamy-Arora Quasi-Demeaning Transformation parameter theta
        theta = float(1.0 - np.sqrt(sigma_e2 / (sigma_e2 + t_periods * sigma_u2)))

        y_mean = self.df.groupby(self.entity_col)[self.dep_var].transform('mean')
        y_gls = (self.df[self.dep_var] - theta * y_mean).values

        X_quasi = []
        for var in self.indep_vars:
            x_mean = self.df.groupby(self.entity_col)[var].transform('mean')
            X_quasi.append((self.df[var] - theta * x_mean).values)
            
        X_gls = np.column_stack([np.ones(len(self.df)) * (1.0 - theta)] + X_quasi)

        beta_gls = np.linalg.inv(X_gls.T @ X_gls) @ X_gls.T @ y_gls
        res_gls = y_gls - X_gls @ beta_gls
        
        dof_gls = len(y_gls) - X_gls.shape[1]
        sigma2_gls = np.sum(res_gls**2) / dof_gls
        cov_gls = np.linalg.inv(X_gls.T @ X_gls) * sigma2_gls
        se_gls = np.sqrt(np.diagonal(cov_gls))

        ss_tot_gls = np.sum((y_gls - np.mean(y_gls))**2)
        ss_res_gls = np.sum(res_gls**2)
        r_squared_gls = 1.0 - (ss_res_gls / ss_tot_gls)

        return {
            "model": "Random Effects (Swamy-Arora FGLS)",
            "coefficients": beta_gls[1:],
            "intercept": beta_gls[0],
            "std_errors": se_gls[1:],
            "cov_matrix": cov_gls[1:, 1:],
            "sigma_e2": float(sigma_e2),
            "sigma_u2": float(sigma_u2),
            "theta": float(theta),
            "r_squared": float(r_squared_gls)
        }

    def run_hausman_specification_test(self):
        """
        Executes classical Hausman Specification Test:
        H0: Random Effects is consistent and efficient (cov(alpha_i, x_it) = 0)
        H1: Fixed Effects is consistent, Random Effects is inconsistent (endogeneity present)
        
        Econometric Formulation:
        H = (b_FE - b_RE)' [ V_FE_homo - V_RE ]^(-1) (b_FE - b_RE)
        Uses spectral decomposition to project variance difference onto positive semi-definite space.
        """
        fe = self.estimate_fixed_effects()
        re = self.estimate_random_effects()
        
        diff_b = fe['coefficients'] - re['coefficients']
        diff_var = fe['cov_matrix_homo'] - re['cov_matrix']
        
        # Spectral decomposition to ensure positive semi-definiteness
        eigvals, eigvecs = np.linalg.eigh(diff_var)
        # Retain positive eigenvalues
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
            "p_value": round(p_value, 4),
            "preferred_model": preferred,
            "hypothesis_verdict": "Reject H0 (Presence of Endogeneity)" if p_value < 0.05 else "Fail to Reject H0 (Random Effects Consistent)"
        }
