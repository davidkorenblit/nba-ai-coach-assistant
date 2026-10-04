"""
Diagnostic and QA Verification Script:
Validates that Level 3 outcome target NaNs are strictly caused by quarter-end Boundary Right-Censoring
(Bug #2 resolution) and that zero unexpected NaNs exist in the dataset.
"""

import os
import sys
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(BASE_DIR, '..', '..', '..', 'data', 'interim', 'level3_labels.csv')

def verify_lookahead_censoring():
    if not os.path.exists(FILE_PATH):
        print(f"❌ File not found: {os.path.abspath(FILE_PATH)}")
        sys.exit(1)
        
    print("=" * 65)
    print("🔍 Level 3 Lookahead Right-Censoring Validator (Bug #2 QA)")
    print("=" * 65)
    
    df = pd.read_csv(
        FILE_PATH,
        usecols=[
            'period', 'seconds_remaining',
            'target_stop_run_90s', 'target_reverse_trend_180s',
            'target_improve_margin_90s', 'target_improve_margin_180s',
            'target_danger_penalty'
        ]
    )
    total_rows = len(df)
    print(f"✅ Loaded dataset: {total_rows:,} rows from {os.path.basename(FILE_PATH)}\n")

    targets = [
        (90, ['target_stop_run_90s', 'target_improve_margin_90s']),
        (180, ['target_reverse_trend_180s', 'target_improve_margin_180s', 'target_danger_penalty'])
    ]

    all_passed = True

    for window, cols in targets:
        boundary_mask = df['seconds_remaining'] < window
        expected_boundary_plays = boundary_mask.sum()

        for col in cols:
            nan_mask = df[col].isna()
            total_nans = nan_mask.sum()
            nan_pct = (total_nans / total_rows) * 100
            valid_rows = total_rows - total_nans
            
            # Check 1: NaNs outside boundary (must be 0)
            nans_outside = (nan_mask & (~boundary_mask)).sum()
            
            # Check 2: NaNs inside boundary (must equal total NaNs)
            nans_inside = (nan_mask & boundary_mask).sum()

            print(f"--- Window: {window}s | Column: {col} ---")
            print(f"  • Valid non-censored: {valid_rows:,} ({(100 - nan_pct):.2f}%)")
            print(f"  • Censored (NaNs):    {total_nans:,} ({nan_pct:.2f}%)")
            
            if nans_outside == 0 and nans_inside == total_nans:
                print(f"  ✅ PASS: 100.0% of NaNs occur strictly at seconds_remaining < {window}s (Unexpected: 0)")
            else:
                print(f"  ❌ FAIL: Found {nans_outside:,} unexpected NaNs outside the boundary window!")
                all_passed = False
        print()

    print("=" * 65)
    if all_passed:
        print("🎉 STATUS: ALL CHECKS PASSED. Boundary Right-Censoring is 100% rigorous.")
        print("=" * 65)
        return 0
    else:
        print("⚠️ STATUS: FAILED. Unexpected NaNs detected.")
        print("=" * 65)
        return 1

if __name__ == '__main__':
    sys.exit(verify_lookahead_censoring())
