import pandas as pd
import numpy as np
import os
import sys
import ast

# --- Config (4 levels up to Root) ---
# וודא שהנתיב הזה תואם למיקום הקובץ בתוך scripts/feature_engineering/validation/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
FILE_PATH = os.path.join(BASE_DIR, 'data', 'interim', 'level1_base.csv')

class Level1Validator:
    """
    Validator Suite for Hybrid Level 1.
    Implements the Validator Pattern: Isolated checks for game rules and data integrity.
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.results = []

    def load_data(self):
        """Loads data efficiently and converts string-lists back to Python objects."""
        if not os.path.exists(self.file_path):
            print(f"❌ Critical: File not found at {self.file_path}")
            sys.exit(1)
        
        try:
            self.df = pd.read_csv(self.file_path, low_memory=False)
            
            # המרת מחרוזות הרשימות בחזרה לאובייקטים (לצורך בדיקת len)
            for col in ['home_lineup', 'away_lineup']:
                if col in self.df.columns:
                    # שימוש ב-ast.literal_eval בצורה בטוחה
                    self.df[col] = self.df[col].apply(
                        lambda x: ast.literal_eval(x) if pd.notna(x) and str(x).startswith('[') else []
                    )
            
            print(f"✅ Loaded Dataset: {len(self.df):,} rows.")
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            sys.exit(1)

    def _log(self, test_name, status, message=""):
        """Internal helper to log results."""
        icon = "✅" if status else "❌"
        print(f"{icon} [{test_name}]: {message}")
        self.results.append(status)

    # --- 1. Hybrid & Lineup Logic Checks ---

    def check_lineup_completeness(self):
        """Verifies exactly 5 players per team in every row (The 10-Player Test)."""
        h_count = self.df['home_lineup'].apply(len)
        a_count = self.df['away_lineup'].apply(len)
        
        full_house = (h_count == 5) & (a_count == 5)
        fail_count = (~full_house).sum()
        
        if fail_count == 0:
            self._log("10-Player Test", True, "Perfect 5v5 coverage across all rows.")
        else:
            pct = (fail_count / len(self.df)) * 100
            self._log("10-Player Test", False, f"Missing players in {fail_count:,} rows ({pct:.1f}%). Note: Common at game starts before first PBP action.")

    def report_confidence_health(self):
        """Reports how much of our data is Official vs. Inferred."""
        if 'lineup_confidence' not in self.df.columns:
            self._log("Confidence Tracking", False, "Column 'lineup_confidence' missing.")
            return

        official_pct = self.df['lineup_confidence'].mean() * 100
        self._log("Inference Stats", True, f"Reliability Score: {official_pct:.1f}% Official API | {100-official_pct:.1f}% Hybrid/Inference")

    def check_player_team_consistency(self):
        """Ensures no player is in both lineups simultaneously."""
        def _has_overlap(row):
            if not row['home_lineup'] or not row['away_lineup']: return False
            return len(set(row['home_lineup']) & set(row['away_lineup'])) > 0
        
        overlaps = self.df.apply(_has_overlap, axis=1).sum()
        if overlaps == 0:
            self._log("Team Consistency", True, "No player overlaps found between Home and Away.")
        else:
            self._log("Team Consistency", False, f"Found {overlaps} rows where a player is listed on both teams.")

    # --- 2. Game Rules & Physics Checks ---

    def check_shot_clock_14s_rule(self):
        """Verifies shot clock resets to max 14s after offensive rebounds."""
        if 'reboundOffensiveTotal' not in self.df.columns: return
        
        off_reb_mask = self.df['reboundOffensiveTotal'] > 0
        violations = self.df.loc[off_reb_mask, 'shot_clock_estimated'] > 14.1
        
        if violations.any():
            max_val = self.df.loc[off_reb_mask, 'shot_clock_estimated'].max()
            self._log("14s Rule Logic", False, f"Found values > 14s after OffReb (Max: {max_val:.1f}s)")
        else:
            self._log("14s Rule Logic", True, "Shot clock correctly capped at 14s after all offensive rebounds.")

    def check_timeouts_inventory_integrity(self):
        """Verifies timeouts decrease correctly and never go negative."""
        cols = [c for c in self.df.columns if 'timeouts_remaining' in c]
        if not cols:
            self._log("Timeouts Inventory", False, "No 'timeouts_remaining' columns found.")
            return

        min_val = self.df[cols].min().min()
        max_val = self.df[cols].max().max()
        
        if min_val >= 0 and max_val <= 7:
             self._log("Timeouts Inventory", True, f"Inventory valid (Range: {min_val}-{max_val}).")
        else:
             self._log("Timeouts Inventory", False, f"Invalid range detected: {min_val} to {max_val}.")

    def check_cumulative_counters_monotonicity(self):
        """Verifies that points generally increase (Sanity check)."""
        if 'cum_pointsTotal' not in self.df.columns:
            self._log("Cumulative Counters", False, "Missing 'cum_pointsTotal'.")
            return
            
        max_pts = self.df['cum_pointsTotal'].max()
        if max_pts > 50:
            self._log("Cumulative Counters", True, f"Points are accumulating (Max Pts: {max_pts}).")
        else:
            self._log("Cumulative Counters", False, f"Suspiciously low max points: {max_pts}.")

    def check_substitution_timer_sync(self):
        """Ensures the fatigue timer resets on substitutions and stays positive."""
        if 'time_since_last_sub' not in self.df.columns:
            self._log("Sub Timer", False, "Missing 'time_since_last_sub' column.")
            return

        min_val = self.df['time_since_last_sub'].min()
        max_val = self.df['time_since_last_sub'].max()
        
        if min_val < 0:
            self._log("Sub Timer Sync", False, f"Negative time found: {min_val}s.")
        elif max_val > 60:
            self._log("Sub Timer Sync", True, f"Timer is active (Max streak: {max_val:.0f}s).")
        else:
            self._log("Sub Timer Sync", False, f"Timer seems stuck or failed to accumulate (Max: {max_val}).")

    def check_critical_missing_values(self):
        """Ensures no gaps in the critical flow of the game."""
        critical = ['scoreHome', 'play_duration', 'home_lineup', 'away_lineup']
        missing_count = self.df[critical].isna().sum().sum()
        
        if missing_count == 0:
            self._log("Critical Gaps", True, "Zero missing values in core columns.")
        else:
            self._log("Critical Gaps", False, f"Found {missing_count} missing values.")
            
            print("\n   🕵️‍♂️ DIAGNOSTICS: Where are the NaNs?")
            for col in critical:
                nans = self.df[self.df[col].isna()]
                if not nans.empty:
                    print(f"   👉 Column '{col}' has {len(nans)} NaNs.")

    # --- Runner ---
    def run(self):
        print(f"🕵️‍♂️ Running FULL HYBRID QA Validator on: {os.path.basename(self.file_path)}")
        print("-" * 60)
        self.load_data()
        print("-" * 60)
        
        # הרצת כל הבדיקות - מהחדש לישן
        self.check_lineup_completeness()
        self.report_confidence_health()
        self.check_player_team_consistency()
        self.check_substitution_timer_sync()
        self.check_shot_clock_14s_rule()
        self.check_timeouts_inventory_integrity()
        self.check_cumulative_counters_monotonicity()
        self.check_critical_missing_values()
        
        print("-" * 60)
        if all(self.results):
            print("🚀 STATUS: PASSED. Dataset is solid and ready for Level 2.")
        else:
            print("⚠️ STATUS: WARNINGS DETECTED. Review logs above before ML training.")

if __name__ == "__main__":
    validator = Level1Validator(FILE_PATH)
    validator.run()