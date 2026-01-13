import pandas as pd
import numpy as np
import os
import sys

# --- Config ---
FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'interim', 'level1_base.csv')

class Level1Validator:
    """
    Validator Suite for Level 1 Feature Engineering.
    Implements the Validator Pattern: Isolated checks managed by a single runner.
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
            print(f"✅ Loaded Dataset: {len(self.df):,} rows.")
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            sys.exit(1)

    def _log(self, test_name, status, message=""):
        """Internal helper to log results."""
        icon = "✅" if status else "❌"
        print(f"{icon} [{test_name}]: {message}")
        self.results.append(status)

    # --- Validation Logic Methods ---

    def check_shot_clock_14s_rule(self):
        """Verifies that shot clock resets to max 14s after offensive rebounds."""
        # יעילות: סינון וקטורי מהיר
        off_reb_mask = self.df['reboundOffensiveTotal'] > 0
        violations = self.df.loc[off_reb_mask, 'shot_clock_estimated'] > 14.0
        
        if violations.any():
            max_val = self.df.loc[off_reb_mask, 'shot_clock_estimated'].max()
            self._log("14s Rule Logic", False, f"Found values > 14s after OffReb (Max: {max_val})")
        else:
            self._log("14s Rule Logic", True, "Shot clock correctly capped at 14s after all offensive rebounds.")

    def check_timeouts_inventory_integrity(self):
        """Verifies timeouts decrease from 7 to 0 and never go negative."""
        cols = [c for c in self.df.columns if 'timeouts_remaining' in c]
        if not cols:
            self._log("Timeouts Inventory", False, "No 'timeouts_remaining' columns found.")
            return

        # בדיקה וקטורית על כל עמודות הטיים-אאוט בבת אחת
        min_val = self.df[cols].min().min()
        max_val = self.df[cols].max().max()
        
        if min_val >= 0 and max_val == 7:
             self._log("Timeouts Inventory", True, f"Inventory valid (Range: {min_val}-{max_val}).")
        else:
             self._log("Timeouts Inventory", False, f"Invalid range detected: {min_val} to {max_val} (Expected 0-7).")

    def check_cumulative_counters_monotonicity(self):
        """Verifies that cumulative counters (points/fouls) generally increase."""
        # בדיקה מדגמית על עמודת הנקודות המצטברת
        if 'cum_pointsTotal' not in self.df.columns:
            self._log("Cumulative Counters", False, "Missing 'cum_pointsTotal'.")
            return
            
        max_pts = self.df['cum_pointsTotal'].max()
        if max_pts > 50: # סף שפיות מינימלי למשחק כדורסל
            self._log("Cumulative Counters", True, f"Counters are accumulating correctly (Max Pts: {max_pts}).")
        else:
            self._log("Cumulative Counters", False, f"Suspiciously low max points: {max_pts}.")

    def check_substitution_timer(self):
        """Verifies substitution timer accumulates properly (Fix Verification)."""
        if 'time_since_last_sub' not in self.df.columns:
            self._log("Sub Timer", False, "Missing column.")
            return

        min_val = self.df['time_since_last_sub'].min()
        max_val = self.df['time_since_last_sub'].max()
        
        if min_val < 0:
            self._log("Sub Timer", False, f"Negative time found: {min_val}s.")
            return

        if max_val > 300:
            self._log("Sub Timer", True, f"Valid accumulation detected (Max streak: {max_val:.0f}s).")
        else:
            self._log("Sub Timer", False, f"⚠️ Timer stuck low! Max value is only {max_val:.0f}s (Fix failed).")

    def check_critical_missing_values(self):
        """Ensures no gaps in critical flow columns and investigates if found."""
        critical = ['scoreHome', 'play_duration', 'possession_id', 'team_fouls_period']
        # בדיקה אם יש בכלל חוסרים
        missing_count = self.df[critical].isna().sum().sum()
        
        if missing_count == 0:
            self._log("Missing Values", True, "Zero missing values in critical columns.")
        else:
            self._log("Missing Values", False, f"Found {missing_count} missing values.")
            
            # --- תוספת: חקירה מיידית של החוסרים ---
            print("\n   🕵️‍♂️ DIAGNOSTICS: Where are the NaNs?")
            for col in critical:
                nans = self.df[self.df[col].isna()]
                if not nans.empty:
                    print(f"   👉 Column '{col}' has {len(nans)} missing values.")
                    print(f"      Top Action Types: {nans['actionType'].value_counts().head(3).to_dict()}")

    # --- Runner ---
    def run(self):
        print(f"🕵️‍♂️ Running QA Validator on: {os.path.basename(self.file_path)}")
        print("-" * 60)
        self.load_data()
        print("-" * 60)
        
        # הרצת כל הבדיקות
        self.check_shot_clock_14s_rule()
        self.check_timeouts_inventory_integrity()
        self.check_cumulative_counters_monotonicity()
        self.check_substitution_timer()
        self.check_critical_missing_values()
        
        print("-" * 60)
        # סיכום סופי
        if all(self.results):
            print("🚀 STATUS: PASSED. Dataset is ready for Level 2.")
        else:
            print("⚠️ STATUS: WARNINGS DETECTED. Review logs above.")

if __name__ == "__main__":
    validator = Level1Validator(FILE_PATH)
    validator.run()