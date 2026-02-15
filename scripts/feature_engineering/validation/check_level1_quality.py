import pandas as pd
import numpy as np
import os
import sys
import ast

# --- Config ---
# הנתיב מעודכן לפי המבנה החדש (3 רמות למעלה ל-Root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FILE_PATH = os.path.join(BASE_DIR, 'data', 'interim', 'level1_base.csv')

class Level1Validator:
    """
    Validator Suite for Hybrid Level 1.
    Ensures 10:10 lineup coverage and tracks inference reliability.
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.results = []

    def load_data(self):
        if not os.path.exists(self.file_path):
            print(f"❌ Critical: File not found at {self.file_path}")
            sys.exit(1)
        
        try:
            self.df = pd.read_csv(self.file_path, low_memory=False)
            # המרת מחרוזות הרשימות בחזרה לאובייקטים של Python (למי שלא נטען אוטומטית)
            for col in ['home_lineup', 'away_lineup']:
                if col in self.df.columns and isinstance(self.df[col].iloc[0], str):
                    self.df[col] = self.df[col].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])
            
            print(f"✅ Loaded Dataset: {len(self.df):,} rows.")
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            sys.exit(1)

    def _log(self, test_name, status, message=""):
        icon = "✅" if status else "❌"
        print(f"{icon} [{test_name}]: {message}")
        self.results.append(status)

    # --- New & Updated Validation Logic ---

    def check_lineup_completeness(self):
        """NEW: Verifies exactly 5 players per team in every row (The 10-Player Test)."""
        h_count = self.df['home_lineup'].apply(len)
        a_count = self.df['away_lineup'].apply(len)
        
        full_house = (h_count == 5) & (a_count == 5)
        fail_count = (~full_house).sum()
        
        if fail_count == 0:
            self._log("10-Player Test", True, "Perfect 5v5 coverage across all rows.")
        else:
            pct = (fail_count / len(self.df)) * 100
            self._log("10-Player Test", False, f"Missing players in {fail_count} rows ({pct:.2f}%).")

    def report_confidence_health(self):
        """NEW: Reports how much of our data is Official vs. Inferred."""
        if 'lineup_confidence' not in self.df.columns:
            self._log("Confidence Tracking", False, "Column missing.")
            return

        official_pct = self.df['lineup_confidence'].mean() * 100
        self._log("Inference Stats", official_pct > 50, 
                  f"Reliability: {official_pct:.1f}% Official API | {100-official_pct:.1f}% Hybrid/Inference")

    def check_player_team_consistency(self):
        """NEW: Ensures no player is in both lineups simultaneously."""
        def _has_overlap(row):
            return len(set(row['home_lineup']) & set(row['away_lineup'])) > 0
        
        overlaps = self.df.apply(_has_overlap, axis=1).sum()
        if overlaps == 0:
            self._log("Team Consistency", True, "No player overlaps between Home and Away.")
        else:
            self._log("Team Consistency", False, f"Found {overlaps} rows where player is on both teams.")

    def check_substitution_timer_sync(self):
        """UPDATED: Ensures timer resets properly and shows activity."""
        min_val = self.df['time_since_last_sub'].min()
        max_val = self.df['time_since_last_sub'].max()
        zeros = (self.df['time_since_last_sub'] == 0).sum()

        if min_val == 0 and zeros > 0 and max_val > 60:
            self._log("Sub Timer Sync", True, f"Timer is active and resets (Max: {max_val:.0f}s, Resets: {zeros}).")
        else:
            self._log("Sub Timer Sync", False, f"Timer issues: Resets={zeros}, Max={max_val:.0f}s.")

    def check_critical_missing_values(self):
        """UPDATED: Added lineups to critical list."""
        critical = ['scoreHome', 'play_duration', 'home_lineup', 'away_lineup']
        missing_count = self.df[critical].isna().sum().sum()
        
        if missing_count == 0:
            self._log("Critical Gaps", True, "Zero NaNs in lineup and flow columns.")
        else:
            self._log("Critical Gaps", False, f"Found {missing_count} missing values in key columns.")

    # --- Preserved Logic ---

    def check_shot_clock_14s_rule(self):
        off_reb_mask = self.df['reboundOffensiveTotal'] > 0
        violations = self.df.loc[off_reb_mask, 'shot_clock_estimated'] > 14.1 # Small buffer
        if violations.any():
            self._log("14s Rule Logic", False, f"Shot clock > 14s after OffReb.")
        else:
            self._log("14s Rule Logic", True, "Shot clock capped correctly.")

    def run(self):
        print(f"🕵️‍♂️ Running HYBRID QA Validator on: {os.path.basename(self.file_path)}")
        print("-" * 60)
        self.load_data()
        print("-" * 60)
        
        self.check_lineup_completeness()
        self.check_confidence_health()
        self.check_player_team_consistency()
        self.check_substitution_timer_sync()
        self.check_critical_missing_values()
        self.check_shot_clock_14s_rule()
        
        print("-" * 60)
        if all(self.results):
            print("🚀 STATUS: PASSED. Hybrid Data is solid. Ready for Level 2 Fatigue!")
        else:
            print("⚠️ STATUS: WARNINGS. Check Inference/Lineup logic before proceeding.")

if __name__ == "__main__":
    validator = Level1Validator(FILE_PATH)
    validator.run()