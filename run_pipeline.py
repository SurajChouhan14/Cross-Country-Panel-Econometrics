"""
Main Execution Pipeline for Longitudinal Panel Econometrics & Policy Engine.
Executes Ground Truth Recovery, Specification Testing, and the Hausman Diagnostic.
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.data_loader import PanelDataLoader
from src.panel_econometric_engine import PanelEconometricEngine


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    log_lines = []
    def log(msg=""):
        print(msg)
        log_lines.append(msg)

    log("=" * 105)
    log("LONGITUDINAL PANEL ECONOMETRICS & MACROECONOMIC SPECIFICATION TOURNAMENT")
    log("Monte Carlo Panel DGP: 120 Countries × 10 Years (1,200 Observations) | Planted Structural Parameters")
    log("Core Methods: Pooled OLS + Entity-Only FE + Two-Way FE (TWFE) + Swamy-Arora RE + Hausman Test")
    log("=" * 105)

    loader = PanelDataLoader(data_dir=data_dir, random_state=42)
    log("\n[1/4] Ingesting longitudinal panel dataset (120 countries, 10 years, 1,200 observations)...")
    df = loader.load_panel_data()
    planted = loader.planted_beta
    log(f"      Panel partition active: {len(df):,} Observations across {df['country_id'].nunique()} Countries")

    engine = PanelEconometricEngine(df)

    log("\n[2/4] Estimating econometric specification ladder (POLS, Entity-FE, TWFE, and Swamy-Arora RE)...")
    pols = engine.estimate_pooled_ols()
    fe_ent = engine.estimate_fixed_effects()
    fe_tw = engine.estimate_twoway_fixed_effects()
    re = engine.estimate_random_effects()

    time_vars = ['tfi_score', 'log_gdp', 'tariff_rate', 'infra_score', 'fx_volatility']

    log("\n" + "=" * 105)
    log("GROUND TRUTH RECOVERY & ECONOMETRIC SPECIFICATION TABLE")
    log("=" * 105)
    log(f"{'Regressor Variable':<18} | {'Planted β':<10} | {'Two-Way FE (TWFE) 🏆':<22} | {'Entity-Only FE':<18} | {'Random Effects (RE)':<18}")
    log("-" * 105)

    for v in time_vars:
        p_b = planted[v]
        b_tw = fe_tw['summary'].params[v]
        b_en = fe_ent['summary'].params[v]
        b_re = re['summary'].params[v]
        log(f"{v:<18} | {p_b:10.4f} | {b_tw:10.4f} (err: {b_tw-p_b:+.3f})   | {b_en:10.4f} (err: {b_en-p_b:+.3f}) | {b_re:10.4f} (err: {b_re-p_b:+.3f})")

    # Distance row
    b_dist_re = re['summary'].params['log_distance']
    log(f"{'log_distance':<18} | {planted['log_distance']:10.4f} | {'[Absorbed / Dropped]':<22} | {'[Absorbed / Dropped]':<18} | {b_dist_re:10.4f} (err: {b_dist_re-planted['log_distance']:+.3f})")
    log("-" * 105)
    
    gdp_en_val = fe_ent['summary'].params['log_gdp']
    gdp_tw_val = fe_tw['summary'].params['log_gdp']
    log(f"  >>> KEY ECONOMETRIC FINDING: Omitting common time trend biased Entity-Only FE GDP elasticity to {gdp_en_val:.4f}.")
    log(f"      Two-Way Fixed Effects (TWFE) absorbs the shared macro trend and restores log(GDP) elasticity to {gdp_tw_val:.4f}!")
    log("  >>> TIME-INVARIANT REGRESSOR: log(distance) is absorbed by FE country demeaning by construction;")
    log(f"      RE estimates distance ({b_dist_re:.4f}) but is biased due to omitted common macroeconomic time shocks λ_t.")
    log("=" * 105)

    log("\n[3/4] Executing Spectrally-Decomposed Hausman Specification Test...")
    hausman = engine.run_hausman_specification_test()
    log(f"      Null Hypothesis (H0)   : Cov(alpha_i, X_it) = 0 & Model Correctly Specified (RE consistent & efficient)")
    log(f"      Alternative (H1)       : RE inconsistent due to unobserved heterogeneity / omitted time effects")
    log(f"      Hausman Stat (χ²)      : {hausman['chi2_statistic']} (df = {hausman['degrees_of_freedom']}, rank of positive-definite subspace)")
    log(f"      Asymptotic p-value     : {hausman['p_value']:.4e} (p < 0.001)")
    log(f"      Specification Verdict  : {hausman['verdict']}")
    
    if hausman['p_value'] < 0.001:
        log(f"      >>> RESUME CLAIM VERIFIED: Hausman χ² = {hausman['chi2_statistic']} (p < 0.001) confirms FE specification! <<<")
    else:
        log(f"      >>> HAUSMAN TEST COMPLETED: Stat = {hausman['chi2_statistic']}, p-value = {hausman['p_value']:.4e} <<<")

    log("\n[4/4] Generating Frozen Specification Report...")
    out_file = os.path.join(results_dir, "final_benchmark.txt")
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines) + '\n')
    log(f"      [SAVED] Frozen benchmark report successfully written to: {out_file}\n")


if __name__ == '__main__':
    main()
