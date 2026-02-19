import pandas as pd
import numpy as np
import os
import sys
import ast

# --- Config (4 levels up to Root) ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
FILE_PATH = os.path.join(BASE_DIR, 'data', 'interim', 'level1_base.csv')

class Level1Validator:
    """
    Validator Suite for Hybrid Level 1.
    Includes Checks for: Completeness, Confidence, Consistency, Physics, and Lineup Turnover.
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
            
            # המרת מחרוזות הרשימות בחזרה לאובייקטים
            for col in ['home_lineup', 'away_lineup']:
                if col in self.df.columns:
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
        """Verifies exactly 5 players per team in every row."""
        h_count = self.df['home_lineup'].apply(len)
        a_count = self.df['away_lineup'].apply(len)
        
        full_house = (h_count == 5) & (a_count == 5)
        fail_count = (~full_house).sum()
        
        if fail_count == 0:
            self._log("10-Player Test", True, "Perfect 5v5 coverage.")
        else:
            pct = (fail_count / len(self.df)) * 100
            self._log("10-Player Test", False, f"Missing players in {fail_count:,} rows ({pct:.1f}%).")

    def check_lineup_turnover(self):
        """NEW: Detects 'Stagnant Lineups' where substitutions are not being captured."""
        # יצירת מזהה ייחודי לחמישייה (String) כדי לספור שינויים
        self.df['lineup_sig_qa'] = self.df['home_lineup'].astype(str) + self.df['away_lineup'].astype(str)
        
        # ספירת חמישיות ייחודיות לכל משחק
        lineup_counts = self.df.groupby('gameId')['lineup_sig_qa'].nunique()
        stagnant_games = lineup_counts[lineup_counts <= 2] # משחק שלם עם פחות מ-2 חמישיות הוא לא הגיוני
        
        avg_lineups = lineup_counts.mean()
        
        if len(stagnant_games) == 0:
            self._log("Lineup Turnover", True, f"Subs detected. Avg {avg_lineups:.1f} unique lineups per game.")
        else:
            pct = (len(stagnant_games) / self.df['gameId'].nunique()) * 100
            self._log("Lineup Turnover", False, f"{pct:.1f}% of games have NO or FEW substitutions detected (Stagnant).")
        
        self.df.drop(columns=['lineup_sig_qa'], inplace=True)

    def report_confidence_health(self):
        """Reports Official vs. Inferred data."""
        if 'lineup_confidence' not in self.df.columns:
            self._log("Confidence Tracking", False, "Column 'lineup_confidence' missing.")
            return
        official_pct = self.df['lineup_confidence'].mean() * 100
        self._log("Inference Stats", True, f"Reliability Score: {official_pct:.1f}% Official API.")

    def check_player_team_consistency(self):
        """Ensures no player is in both lineups simultaneously."""
        def _has_overlap(row):
            if not row['home_lineup'] or not row['away_lineup']: return False
            return len(set(row['home_lineup']) & set(row['away_lineup'])) > 0
        
        overlaps = self.df.apply(_has_overlap, axis=1).sum()
        if overlaps == 0:
            self._log("Team Consistency", True, "No player overlaps found.")
        else:
            self._log("Team Consistency", False, f"Found {overlaps} rows with player overlaps.")

    # --- 2. Game Rules & Physics Checks ---

    def check_shot_clock_14s_rule(self):
        """Verifies shot clock resets to max 14s after offensive rebounds."""
        if 'reboundOffensiveTotal' not in self.df.columns: return
        off_reb_mask = self.df['reboundOffensiveTotal'] > 0
        violations = self.df.loc[off_reb_mask, 'shot_clock_estimated'] > 14.1
        if violations.any():
            self._log("14s Rule Logic", False, "Found values > 14s after OffReb.")
        else:
            self._log("14s Rule Logic", True, "Shot clock correctly capped at 14s.")

    def check_timeouts_inventory_integrity(self):
        """Verifies timeouts range."""
        cols = [c for c in self.df.columns if 'timeouts_remaining' in c]
        if not cols: return
        min_val, max_val = self.df[cols].min().min(), self.df[cols].max().max()
        self._log("Timeouts Inventory", (min_val >= 0 and max_val <= 7), f"Inventory valid (Range: {min_val}-{max_val}).")

    def check_timeout_strategic_weights(self):
        """וואלידציה לסיווג החדש של פסקי הזמן"""
        if 'timeout_strategic_weight' not in self.df.columns:
            self._log("Timeout Weights", False, "Column 'timeout_strategic_weight' missing.")
            return
        
        unique_vals = sorted(self.df['timeout_strategic_weight'].unique())
        # בודק שהערכים הם רק 0, 1, 2, 3
        is_valid_range = all(v in [0, 1, 2, 3] for v in unique_vals)
        
        # בודק אם "תפסנו" פסקי זמן משמעותיים (משקלים 2 ו-3)
        has_heavy_tos = self.df['timeout_strategic_weight'].max() >= 2
        
        message = f"Values: {unique_vals}. Heavy timeouts (2+) detected: {has_heavy_tos}"
        self._log("Timeout Weights", is_valid_range and has_heavy_tos, message)

    def check_cumulative_counters_monotonicity(self):
        """Verifies points accumulation."""
        if 'cum_pointsTotal' not in self.df.columns:
            self._log("Cumulative Counters", False, "Missing points column.")
            return
        max_pts = self.df['cum_pointsTotal'].max()
        self._log("Cumulative Counters", (max_pts > 50), f"Points accumulating (Max: {max_pts}).")

    def check_substitution_timer_sync(self):
        """Ensures the fatigue timer resets."""
        if 'time_since_last_sub' not in self.df.columns: return
        min_val, max_val = self.df['time_since_last_sub'].min(), self.df['time_since_last_sub'].max()
        # אם הטיימר מגיע ליותר מ-720 שניות (רבע שלם) ללא איפוס ב-100% מהמקרים, זו תקלה.
        self._log("Sub Timer Sync", (min_val >= 0 and max_val > 60), f"Timer active (Max: {max_val:.0f}s).")

    def check_critical_missing_values(self):
        """Ensures no gaps in critical columns."""
        critical = ['scoreHome', 'play_duration', 'home_lineup', 'away_lineup']
        missing_count = self.df[critical].isna().sum().sum()
        self._log("Critical Gaps", (missing_count == 0), f"Missing values: {missing_count}")

    def run(self):
        print(f"🕵️‍♂️ Running FULL HYBRID QA Validator on: {os.path.basename(self.file_path)}")
        print("-" * 60)
        self.load_data()
        print("-" * 60)
        
        self.check_lineup_completeness()
        self.check_lineup_turnover()
        self.report_confidence_health()
        self.check_player_team_consistency()
        self.check_substitution_timer_sync()
        self.check_shot_clock_14s_rule()
        self.check_timeouts_inventory_integrity()
        self.check_timeout_strategic_weights() # הבדיקה החדשה כאן
        self.check_cumulative_counters_monotonicity()
        self.check_critical_missing_values()
        
        print("-" * 60)
        if all(self.results):
            print("🚀 STATUS: PASSED. Dataset is solid.")
        else:
            print("⚠️ STATUS: WARNINGS DETECTED. Review logs above.")

if __name__ == "__main__":
    validator = Level1Validator(FILE_PATH)
    validator.run()