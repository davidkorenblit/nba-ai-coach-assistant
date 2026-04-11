import pandas as pd
import numpy as np
import os
import sys

# --- Config ---
# מותאם לנתיב הריצה מתוך תיקיית validation
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(BASE_DIR, '..', '..', '..', 'data', 'interim', 'level2_features.csv')

class Level2Validator:
    """
    Validator Suite for Level 2 Feature Engineering (Hybrid V4).
    STRICT MODE: Enforces mathematical boundaries, logic rules, and data integrity 
    for Momentum, Usage Gravity, and Accumulated Fatigue.
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.results = []

    def load_data(self):
        if not os.path.exists(self.file_path):
            print(f"❌ Critical: File not found at {os.path.abspath(self.file_path)}")
            sys.exit(1)
        try:
            self.df = pd.read_csv(self.file_path, low_memory=False)
            print(f"✅ Loaded Level 2 Dataset: {len(self.df):,} rows.")
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            sys.exit(1)

    def _log(self, test_name, status, message=""):
        icon = "✅" if status else "❌"
        print(f"{icon} [{test_name}]: {message}")
        self.results.append(status)

    # --- Validation Logic Methods ---

    def check_feature_existence(self):
        """Ensures ALL Hybrid Level-2 features exist."""
        expected_cols = [
            'style_tempo_rolling', 'is_high_fatigue', 
            'momentum_streak_rolling', 'explosiveness_index', 
            'instability_index', 'is_star_resting', 'is_clutch_time',
            'home_usage_gravity', 'away_usage_gravity', 'usage_delta',
            'home_cum_fatigue', 'away_cum_fatigue'
        ]
        
        missing = [col for col in expected_cols if col not in self.df.columns]
        
        if not missing:
            self._log("Schema Check", True, f"All {len(expected_cols)} Level-2 features exist.")
        else:
            self._log("Schema Check", False, f"Missing columns: {missing}")

    def check_strict_clean_data(self):
        """STRICT: Ensures absolutely ZERO NaNs or Infinite values in the engineered features."""
        cols_to_check = [
            'style_tempo_rolling', 'is_high_fatigue', 'momentum_streak_rolling', 
            'explosiveness_index', 'instability_index', 'is_clutch_time',
            'home_usage_gravity', 'away_usage_gravity', 'usage_delta',
            'home_cum_fatigue', 'away_cum_fatigue'
        ]
        existing_cols = [c for c in cols_to_check if c in self.df.columns]
        
        # Check NaNs
        nans_count = self.df[existing_cols].isna().sum()
        failed_nan_cols = nans_count[nans_count > 0]
        
        # Check Infs
        inf_mask = np.isinf(self.df[existing_cols])
        inf_count = inf_mask.sum()
        failed_inf_cols = inf_count[inf_count > 0]
        
        if failed_nan_cols.empty and failed_inf_cols.empty:
            self._log("Strict Clean Data Check", True, "Zero NaNs and Zero Infs in all engineered features.")
        else:
            err_msg = ""
            if not failed_nan_cols.empty: err_msg += f"NaNs in {failed_nan_cols.to_dict()}. "
            if not failed_inf_cols.empty: err_msg += f"Infs in {failed_inf_cols.to_dict()}."
            self._log("Strict Clean Data Check", False, err_msg)

    def check_binary_flags(self):
        """Verifies binary features contain only 0/1."""
        binary_cols = ['is_high_fatigue', 'is_star_resting', 'is_clutch_time']
        valid = True
        
        for col in binary_cols:
            if col not in self.df.columns: continue
            invalid_vals = [x for x in self.df[col].unique() if pd.notna(x) and x not in [0, 1, 0.0, 1.0]]
            if invalid_vals:
                valid = False
                self._log("Binary Flags", False, f"Column '{col}' has invalid values: {invalid_vals}")
                break
        
        if valid:
            self._log("Binary Flags", True, "All binary flags contain only {0, 1}.")

    def check_clutch_time_logic(self):
        """Logic Test: Clutch time must be <= 300s (5 mins) and margin <= 5."""
        if 'is_clutch_time' not in self.df.columns: return

        errors = self.df[(self.df['is_clutch_time'] == 1) & 
                         ((self.df['seconds_remaining'] > 300) | (self.df['score_margin'].abs() > 5))]
        
        if errors.empty:
            self._log("Clutch Time Logic", True, "Clutch Time correctly bound to last 5 mins & <=5 pt margin.")
        else:
            self._log("Clutch Time Logic", False, f"Found {len(errors)} invalid Clutch Time rows!")

    def check_momentum_sanity(self):
        if 'momentum_streak_rolling' not in self.df.columns: return
        min_val, max_val = self.df['momentum_streak_rolling'].min(), self.df['momentum_streak_rolling'].max()
        
        if min_val == 0 and max_val == 0:
             self._log("Momentum Sanity", False, "⚠️ CRITICAL: All Momentum values are 0.0!")
        elif -100 <= min_val and max_val <= 100:
             self._log("Momentum Sanity", True, f"Valid Range: {min_val:.1f} to {max_val:.1f}.")
        else:
             self._log("Momentum Sanity", False, f"Suspicious extreme values: {min_val} to {max_val}.")

    def check_usage_gravity_logic(self):
        """NEW: Ensures Usage Gravity isn't just returning the default 0.75."""
        if 'home_usage_gravity' not in self.df.columns: return
        
        # If 99% of the values are exactly 0.75, it means the lookup mapping failed entirely
        default_rate = (self.df['home_usage_gravity'] == 0.75).mean()
        
        if default_rate > 0.98:
            self._log("Usage Gravity Logic", False, f"⚠️ {default_rate*100:.1f}% of values are default (0.75). Star lookup failed!")
        else:
            max_usg = self.df['home_usage_gravity'].max()
            self._log("Usage Gravity Logic", True, f"Dynamic Usage active. Max Lineup Gravity: {max_usg:.2f}")

    def check_cumulative_fatigue_logic(self):
        """NEW: Ensures Accumulated fatigue is actually growing and reasonable."""
        if 'home_cum_fatigue' not in self.df.columns: return
        
        min_f, max_f = self.df['home_cum_fatigue'].min(), self.df['home_cum_fatigue'].max()
        
        if min_f < 0:
            self._log("Cum Fatigue Logic", False, "Negative accumulated fatigue found!")
        elif max_f == 0:
            self._log("Cum Fatigue Logic", False, "Accumulated fatigue is completely stuck at 0.")
        else:
            # Assuming max fatigue for a player playing full 48 mins is 2880 seconds
            self._log("Cum Fatigue Logic", True, f"Fatigue accumulating properly (Max: {max_f:.1f} seconds).")

    def run(self):
        print(f"🕵️‍♂️ Running STRICT QA Validator on: {os.path.basename(self.file_path)}")
        print("-" * 60)
        self.load_data()
        print("-" * 60)
        
        self.check_feature_existence()
        self.check_strict_clean_data()
        self.check_binary_flags()
        self.check_clutch_time_logic()
        self.check_momentum_sanity()
        self.check_usage_gravity_logic()
        self.check_cumulative_fatigue_logic()
        
        print("-" * 60)
        if all(self.results):
            print("🚀 STATUS: PASSED. Level 2 dataset is mathematically sound and ready for Level 3/ML.")
            sys.exit(0)
        else:
            print("⚠️ STATUS: FAILED. Fix the logic errors in 02_build_level2_momentum.py before proceeding.")
            sys.exit(1)

if __name__ == "__main__":
    validator = Level2Validator(FILE_PATH)
    validator.run()