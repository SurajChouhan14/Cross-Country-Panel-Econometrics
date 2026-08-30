"""
Panel Data Ingestion and Monte Carlo Simulation Module.

Generates a transparent longitudinal panel Data Generating Process (DGP)
featuring 120 countries observed over 10 years (1,200 balanced country-year observations)
with planted ground-truth structural coefficients for econometric specification testing.
"""

import os
import numpy as np
import pandas as pd


class PanelDataLoader:
    """
    Ingestion and simulation engine for multi-country longitudinal panel data.
    """

    def __init__(self, data_dir="data", n_countries=120, n_years=10, random_state=42):
        self.data_dir = data_dir
        self.n_countries = n_countries
        self.n_years = n_years
        self.random_state = random_state
        self.output_csv = os.path.join(self.data_dir, "simulated_trade_panel_dgp.csv")
        
        # Planted ground-truth structural parameters
        self.planted_beta = {
            'tfi_score': 1.42,
            'log_gdp': 0.85,
            'log_distance': -0.65,  # Time-invariant gravity regressor
            'tariff_rate': -0.04,
            'infra_score': 0.35,
            'fx_volatility': -0.80
        }

    def load_panel_data(self):
        """
        Generates/Loads a balanced panel dataset of 120 countries over 10 years (1,200 country-year observations).
        - Dependent Variable: log(Exports)
        - Key Policy Variable: Trade Facilitation Index (TFI, 0 to 1 scale)
        - Covariates: log(GDP), log(Distance), Tariff Rate (%), Exchange Rate Volatility, Infrastructure Score
        - Structural Parameters: Unobserved Country Fixed Effects (alpha_i) and Common Time Shocks (lambda_t)
        """
        if os.path.exists(self.output_csv):
            return pd.read_csv(self.output_csv)

        np.random.seed(self.random_state)
        countries = [f"Country_{i:03d}" for i in range(1, self.n_countries + 1)]
        years = list(range(2014, 2014 + self.n_years))

        records = []
        
        country_fixed_effects = {c: np.random.normal(5.0, 1.2) for c in countries}
        time_fixed_effects = {y: np.random.normal(0.0, 0.3) for y in years}

        for c in countries:
            log_distance = float(np.random.normal(8.0, 0.5))
            base_gdp = float(np.random.normal(10.5, 1.5))
            
            for t_idx, y in enumerate(years):
                log_gdp = float(base_gdp + 0.03 * t_idx + np.random.normal(0, 0.05))
                tfi_raw = 0.4 + 0.02 * t_idx + np.random.uniform(0, 0.3)
                tfi_score = float(max(0.1, min(0.95, tfi_raw)))
                
                tariff_raw = 15.0 - 0.5 * t_idx + np.random.normal(0, 2.0)
                tariff_rate = float(max(1.0, min(35.0, tariff_raw)))
                
                infra_raw = 3.0 + 0.05 * t_idx + np.random.normal(0, 0.2)
                infra_score = float(max(1.0, min(5.0, infra_raw)))
                
                fx_volatility = float(np.random.exponential(scale=0.08))
                epsilon = float(np.random.normal(0, 0.25))
                
                log_exports = (
                    country_fixed_effects[c]
                    + time_fixed_effects[y]
                    + self.planted_beta['log_gdp'] * log_gdp
                    + self.planted_beta['log_distance'] * log_distance
                    + self.planted_beta['tfi_score'] * tfi_score
                    + self.planted_beta['tariff_rate'] * tariff_rate
                    + self.planted_beta['infra_score'] * infra_score
                    + self.planted_beta['fx_volatility'] * fx_volatility
                    + epsilon
                )

                records.append({
                    'country_id': c,
                    'year': y,
                    'log_exports': log_exports,
                    'tfi_score': tfi_score,
                    'log_gdp': log_gdp,
                    'log_distance': log_distance,
                    'tariff_rate': tariff_rate,
                    'infra_score': infra_score,
                    'fx_volatility': fx_volatility
                })

        df = pd.DataFrame(records)
        df.to_csv(self.output_csv, index=False)
        return df
