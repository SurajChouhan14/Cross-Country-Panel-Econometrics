"""
Automated Unit Test Suite for Longitudinal Panel Data Econometrics Engine.
Verifies Pooled OLS, Fixed Effects with CRVE, Swamy-Arora RE FGLS, and Hausman Specification Test.
"""

import unittest
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import PanelDataLoader
from src.panel_econometric_engine import PanelEconometricEngine


class TestPanelEconometricsEngine(unittest.TestCase):
    """
    Unit test cases for panel econometric linear algebra estimation engine.
    """

    @classmethod
    def setUpClass(cls):
        cls.loader = PanelDataLoader(data_dir="data")
        cls.df = cls.loader.load_panel_data()
        cls.engine = PanelEconometricEngine(
            df=cls.df,
            entity_col='country_id',
            time_col='year',
            dep_var='log_exports',
            indep_vars=['tfi_score', 'log_gdp', 'tariff_rate', 'infra_score', 'fx_volatility']
        )

    def test_panel_data_loading(self):
        """Verify longitudinal panel structure."""
        self.assertGreaterEqual(len(self.df), 1000)
        self.assertEqual(self.df['country_id'].nunique(), 120)
        self.assertEqual(self.df['year'].nunique(), 10)
        self.assertIn('log_exports', self.df.columns)
        self.assertIn('tfi_score', self.df.columns)

    def test_pooled_ols_estimation(self):
        """Verify Pooled OLS estimation computes finite coefficients and positive R-squared."""
        pols = self.engine.estimate_pooled_ols()
        self.assertEqual(len(pols['coefficients']), 5)
        self.assertTrue(np.all(np.isfinite(pols['coefficients'])))
        self.assertTrue(np.all(pols['std_errors'] > 0.0))
        self.assertGreater(pols['r_squared'], 0.0)
        self.assertLess(pols['r_squared'], 1.0)

    def test_fixed_effects_estimation(self):
        """Verify Fixed Effects within estimator and cluster-robust standard errors."""
        fe = self.engine.estimate_fixed_effects()
        self.assertEqual(len(fe['coefficients']), 5)
        self.assertTrue(np.all(np.isfinite(fe['coefficients'])))
        self.assertTrue(np.all(fe['std_errors'] > 0.0))
        self.assertGreater(fe['sigma_e2'], 0.0)
        self.assertGreater(fe['r_squared'], 0.0)

    def test_random_effects_swamy_arora(self):
        """Verify Random Effects Swamy-Arora FGLS quasi-demeaning parameter theta is in (0, 1)."""
        re = self.engine.estimate_random_effects()
        self.assertEqual(len(re['coefficients']), 5)
        self.assertGreater(re['sigma_e2'], 0.0)
        self.assertGreaterEqual(re['sigma_u2'], 0.0)
        self.assertGreater(re['theta'], 0.0)
        self.assertLessEqual(re['theta'], 1.0)

    def test_hausman_specification_test(self):
        """Verify spectral projection Hausman test computes non-negative Chi-sq statistic."""
        hausman = self.engine.run_hausman_specification_test()
        self.assertIn("hausman_statistic", hausman)
        self.assertIn("p_value", hausman)
        self.assertIn("preferred_model", hausman)
        self.assertGreaterEqual(hausman['hausman_statistic'], 0.0)
        self.assertTrue(0.0 <= hausman['p_value'] <= 1.0)


if __name__ == '__main__':
    unittest.main()
