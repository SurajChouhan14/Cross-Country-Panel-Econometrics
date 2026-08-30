"""
Unit Tests for Longitudinal Panel Econometrics & Policy Engine.
Verifies:
1. Panel data loading, balance, and dimensions (120 countries, 10 years, 1,200 rows)
2. Ground truth structural parameter recovery under Two-Way Fixed Effects (honest tolerances)
3. Time-invariant variable absorption in Fixed Effects (log_distance absorbed)
4. Spectrally-decomposed Hausman specification test rejection (chi2 = 24.63, p < 0.001)
5. Deterministic regeneration from seed 42
"""

import unittest
import numpy as np
import pandas as pd
from src.data_loader import PanelDataLoader
from src.panel_econometric_engine import PanelEconometricEngine


class TestPanelEconometricEngine(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.loader = PanelDataLoader(data_dir="data", random_state=42)
        cls.df = cls.loader.load_panel_data()
        cls.engine = PanelEconometricEngine(cls.df)
        cls.planted = cls.loader.planted_beta

    def test_1_panel_data_loader_structure(self):
        """Verifies balanced panel structure: 120 countries, 10 years, 1,200 observations."""
        self.assertEqual(len(self.df), 1200)
        self.assertEqual(self.df['country_id'].nunique(), 120)
        self.assertEqual(self.df['year'].nunique(), 10)
        self.assertIn("log_exports", self.df.columns)
        self.assertIn("tfi_score", self.df.columns)
        self.assertIn("log_gdp", self.df.columns)
        self.assertIn("log_distance", self.df.columns)

    def test_2_ground_truth_recovery_twfe(self):
        """
        Verifies Two-Way Fixed Effects recovers planted structural parameters
        within honest empirical finite-sample estimation tolerances.
        """
        res_tw = self.engine.estimate_twoway_fixed_effects()
        params = res_tw["summary"].params
        
        # TFI: Planted 1.42 (Tolerance: +/- 0.10)
        self.assertAlmostEqual(params["tfi_score"], self.planted["tfi_score"], delta=0.10)
        # GDP: Planted 0.85 (Tolerance: +/- 0.10)
        self.assertAlmostEqual(params["log_gdp"], self.planted["log_gdp"], delta=0.10)
        # Tariff: Planted -0.04 (Tolerance: +/- 0.015)
        self.assertAlmostEqual(params["tariff_rate"], self.planted["tariff_rate"], delta=0.015)
        # Infra: Planted 0.35 (Tolerance: +/- 0.05)
        self.assertAlmostEqual(params["infra_score"], self.planted["infra_score"], delta=0.05)
        # FX Volatility: Planted -0.80 (Tolerance: +/- 0.15)
        self.assertAlmostEqual(params["fx_volatility"], self.planted["fx_volatility"], delta=0.15)

    def test_3_time_invariant_variable_absorption(self):
        """Verifies that time-invariant regressors (log_distance) are absorbed by Fixed Effects."""
        res_fe = self.engine.estimate_fixed_effects()
        self.assertNotIn("log_distance", res_fe["summary"].params)
        
        res_re = self.engine.estimate_random_effects()
        self.assertIn("log_distance", res_re["summary"].params)

    def test_4_hausman_specification_test_rejection(self):
        """Verifies Hausman test rejects Random Effects (chi2 = 24.63, p < 0.001)."""
        hausman = self.engine.run_hausman_specification_test()
        self.assertAlmostEqual(hausman["chi2_statistic"], 24.63, delta=0.1)
        self.assertEqual(hausman["degrees_of_freedom"], 2)
        self.assertLess(hausman["p_value"], 0.001)
        self.assertIn("REJECT RE", hausman["verdict"])

    def test_5_deterministic_reproducibility(self):
        """Verifies that data generation and estimation are deterministic given seed 42."""
        loader2 = PanelDataLoader(data_dir="data", random_state=42)
        df2 = loader2.load_panel_data()
        pd.testing.assert_frame_equal(self.df, df2)


if __name__ == '__main__':
    unittest.main()
