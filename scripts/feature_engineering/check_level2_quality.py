import pandas as pd
import numpy as np
import os
import sys

# --- Config ---
# בודקים את הקובץ המאוחד של שלב 2
FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'interim', 'level2_features.csv')

class Level2Validator:
    """
    Validator Suite for Level 2 Feature Engineering.
    Checks consistency of Context, Momentum, and Dynamic features.
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.results = []

    def load_data(self):
        """Loads data efficiently."""
        if not os.path.exists(self.file_path):
            print(f"❌ Critical: File not found at {self.file_path}")
            sys.exit(1)
        
        try:
            self.df = pd.read_csv(self.file_path, low_memory=False)
            print(f"✅ Loaded Level 2 Dataset: {len(self.df):,} rows.")
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            sys.exit(1)

    def _log(self, test_name, status, message=""):
        """Internal helper to log results."""
        icon = "✅" if status else "❌"
        print(f"{icon} [{test_name}]: {message}")
        self.results.append(status)

    # --- Validation Logic Methods ---

    def check_feature_existence(self):
        """Ensures all 7 new features were created and saved."""
        expected_cols = [
            'style_tempo_rolling', 'is_high_fatigue', 
            'momentum_streak_rolling', 'explosiveness_index', 
            'instability_index', 'is_star_resting', 'is_crunch_time'
        ]
        missing = [col for col in expected_cols if col not in self.df.columns]
        
        if not missing:
            self._log("Schema Check", True, "All 7 Level-2 features exist.")
        else:
            self._log("Schema Check", False, f"Missing columns: {missing}")

    def check_binary_flags(self):
        """Verifies binary features contain only 0/1."""
        binary_cols = ['is_high_fatigue', 'is_star_resting', 'is_crunch_time']
        valid = True
        
        for col in binary_cols:
            if col not in self.df.columns: continue
            unique_vals = self.df[col].unique()
            # בודק אם יש ערכים שאינם 0 או 1
            invalid_vals = [x for x in unique_vals if x not in [0, 1, 0.0, 1.0]]
            
            if invalid_vals:
                valid = False
                self._log("Binary Flags", False, f"Column '{col}' has invalid values: {invalid_vals}")
                break
        
        if valid:
            self._log("Binary Flags", True, "All binary flags contain only valid {0, 1} values.")

    def check_crunch_time_logic(self):
        """Logic Test: If is_crunch_time=1, then time must be <= 300 (5 mins)."""
        if 'is_crunch_time' not in self.df.columns: return

        errors = self.df[(self.df['is_crunch_time'] == 1) & (self.df['seconds_remaining'] > 300)]
        
        if errors.empty:
            self._log("Crunch Time Logic", True, "No Crunch Time events detected outside last 5 minutes.")
        else:
            self._log("Crunch Time Logic", False, f"Found {len(errors)} rows marked as Crunch Time with > 300s left!")

    def check_momentum_sanity(self):
        """Verifies momentum scores are within reasonable mathematical bounds."""
        if 'momentum_streak_rolling' not in self.df.columns: return
        
        min_val = self.df['momentum_streak_rolling'].min()
        max_val = self.df['momentum_streak_rolling'].max()
        
        if -100 <= min_val and max_val <= 100:
             self._log("Momentum Sanity", True, f"Values within sane range ({min_val:.1f} to {max_val:.1f}).")
        else:
             self._log("Momentum Sanity", False, f"Suspiciously extreme values detected ({min_val} to {max_val}).")

    def check_explosiveness_index(self):
        """Verifies Explosiveness isn't purely zero."""
        if 'explosiveness_index' not in self.df.columns: return
        
        if (self.df['explosiveness_index'] == 0).all():
             self._log("Explosiveness", False, "All values are 0. Calculation likely failed.")
        else:
             max_abs = self.df['explosiveness_index'].abs().max()
             if max_abs < 50:
                 self._log("Explosiveness", True, f"Calculation active (Max swing: {max_abs} pts).")
             else:
                 self._log("Explosiveness", False, f"Suspicious spike detected: {max_abs} pts change.")

    def check_nulls_in_features(self):
        """Ensures no NaNs in the new features."""
        new_cols = [
            'style_tempo_rolling', 'momentum_streak_rolling', 
            'explosiveness_index', 'instability_index'
        ]
        has_nans = False
        for col in new_cols:
            if col in self.df.columns and self.df[col].isna().sum() > 0:
                self._log("Null Check", False, f"Column '{col}' still has NaN values!")
                has_nans = True
        
        if not has_nans:
            self._log("Null Check", True, "Clean dataset. No NaNs in numeric features.")

    # --- Runner ---
    def run(self):
        print(f"🕵️‍♂️ Running QA Validator on: {os.path.basename(self.file_path)}")
        print("-" * 60)
        self.load_data()
        print("-" * 60)
        
        self.check_feature_existence()
        self.check_binary_flags()
        self.check_crunch_time_logic()
        self.check_momentum_sanity()
        self.check_explosiveness_index()
        self.check_nulls_in_features()
        
        print("-" * 60)
        if all(self.results):
            print("🚀 STATUS: PASSED. Level 2 features are robust.")
        else:
            print("⚠️ STATUS: WARNINGS DETECTED. Review logs above.")

if __name__ == "__main__":
    validator = Level2Validator(FILE_PATH)
    validator.run()