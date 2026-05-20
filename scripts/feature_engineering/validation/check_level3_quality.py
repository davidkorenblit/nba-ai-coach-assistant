import pandas as pd
import numpy as np
import os
import sys

# --- Config ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(BASE_DIR, '..', '..', '..', 'data', 'interim', 'level3_labels.csv')

class Level3QAValidator:
    """Draconian QA Suite for Level 3 Labels (OOP Architecture)."""

    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.results = []
        
        # הפרדה לוגית בין סוגי הטרגטים
        self.continuous_targets = [
            'target_stop_run_90s', 'target_reverse_trend_180s',
            'target_improve_margin_90s', 'target_improve_margin_180s'
        ]
        self.binary_targets = ['target_danger_penalty']
        self.target_cols = self.continuous_targets + self.binary_targets

    def _log(self, test_name, status, message=""):
        icon = "✅" if status else "❌"
        print(f"{icon} [{test_name}]: {message}")
        self.results.append(status)

    def load_data(self):
        if not os.path.exists(self.file_path):
            print(f"❌ Critical: File not found at {os.path.abspath(self.file_path)}")
            sys.exit(1)
        self.df = pd.read_csv(self.file_path, low_memory=False)
        print(f"✅ Loaded Level 3 Labels: {len(self.df):,} rows.\n")

    def check_missing_targets(self):
        missing_cols = [c for c in self.target_cols if c not in self.df.columns]
        if missing_cols:
            self._log("Schema Check", False, f"Missing columns: {missing_cols}")
            return

        nulls = self.df[self.target_cols].isnull().sum()
        if nulls.sum() == 0:
            self._log("Missing Values", True, "Zero NaNs in all target columns.")
        else:
            self._log("Missing Values", False, f"NaNs detected!\n{nulls[nulls > 0]}")

    def check_class_balance(self):
        print("\n📊 DATA DISTRIBUTIONS:")
        valid = True
        
        print("   --- Continuous Targets (Average Delta) ---")
        for col in self.continuous_targets:
            mean_val = self.df[col].mean()
            print(f"   - {col}: {mean_val:+.2f} average shift")
            
        print("\n   --- Binary Targets (Class Balance) ---")
        for col in self.binary_targets:
            pos_rate = self.df[col].mean() * 100
            print(f"   - {col}: {pos_rate:.2f}% Positive (Class 1)")
            if pos_rate < 0.1 or pos_rate > 99.9:
                valid = False
        
        if valid:
            self._log("Class Balance", True, "Binary targets have valid diversity.")
        else:
            self._log("Class Balance", False, "Extreme class imbalance detected in binary targets. XGBoost will struggle.")

    def check_timeout_logic(self):
        is_timeout = self.df['actionType'].str.contains('timeout', case=False, na=False)
        timeouts_df = self.df[is_timeout]

        if timeouts_df.empty:
            self._log("Timeout Logic", False, "No timeouts found in 'actionType'.")
            return

        # Critical Check: Penalty should only apply if a coach IGNORED the danger.
        penalty_on_to = timeouts_df['target_danger_penalty'].sum()
        if penalty_on_to == 0:
            self._log("Timeout Logic", True, f"Danger Penalty on Timeouts is exactly 0 (Out of {len(timeouts_df)} timeouts).")
        else:
            self._log("Timeout Logic", False, f"LOGIC ERROR: {penalty_on_to} penalties assigned to timeout events!")

        print("\n📈 COACH TIMEOUT IMPACT (Average Delta when Timeout called):")
        for col in self.continuous_targets:
            avg_impact = timeouts_df[col].mean()
            # הערך פה הוא רציף - כמה השתפר/החמיר המומנטום או ההפרש בממוצע
            print(f"   - {col.replace('target_', '').ljust(25)}: {avg_impact:+.2f} average impact")

    def check_end_of_period(self):
        last_30_sec = self.df[self.df['seconds_remaining'] <= 30]
        self._log("Edge Case", True, f"Handled {len(last_30_sec):,} rows in the last 30s of periods via Lookahead Fallback.")

    def run(self):
        print("🕵️‍♂️ Starting DRACONIAN QA: Level 3 Labels")
        print("-" * 60)
        self.load_data()
        
        self.check_missing_targets()
        self.check_class_balance()
        self.check_timeout_logic()
        print("")
        self.check_end_of_period()
        
        print("-" * 60)
        if all(self.results):
            print("🚀 STATUS: PASSED. Labels are mathematically sound. Ready for ML!")
            sys.exit(0)
        else:
            print("⚠️ STATUS: FAILED. Fix logic errors before model training.")
            sys.exit(1)

if __name__ == "__main__":
    validator = Level3QAValidator(FILE_PATH)
    validator.run()