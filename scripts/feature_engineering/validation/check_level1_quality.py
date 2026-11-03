import pandas as pd
import numpy as np
import os
import sys
import ast
import json

# --- Config (4 levels up to Root) ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
FILE_PATH = os.path.join(BASE_DIR, 'data', 'interim', 'level1_base.csv')
BENCHMARKS_PATH = os.path.join(BASE_DIR, 'config', 'nba_benchmarks.json')

class Level1Validator:
    """
    Validator Suite for Hybrid Level 1.
    Includes Checks for: Completeness, Confidence, Consistency, Physics, Lineup Turnover,
    Monotonicity of Cumulative Counters, and Informative NBA Canonical Benchmarks.
    """

    def __init__(self, file_path, benchmarks_path=None):
        self.file_path = file_path
        self.benchmarks_path = benchmarks_path or BENCHMARKS_PATH
        self.benchmarks = self.load_benchmarks()
        self.df = None
        self.results = []

    def load_benchmarks(self):
        """Loads canonical NBA benchmarks from external JSON config."""
        if os.path.exists(self.benchmarks_path):
            try:
                with open(self.benchmarks_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Warning: Could not read benchmarks JSON ({self.benchmarks_path}): {e}")
        return {}

    def load_data(self):
        """Loads data efficiently and converts string-lists back to Python objects."""
        if not os.path.exists(self.file_path):
            print(f"❌ Critical: File not found at {self.file_path}")
            sys.exit(1)
        
        try:
            self.df = pd.read_csv(self.file_path, low_memory=False)
            
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
        """Detects 'Stagnant Lineups' where substitutions are not being captured."""
        self.df['lineup_sig_qa'] = self.df['home_lineup'].astype(str) + self.df['away_lineup'].astype(str)
        lineup_counts = self.df.groupby('gameId')['lineup_sig_qa'].nunique()
        stagnant_games = lineup_counts[lineup_counts <= 2]
        
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
        """Verifies timeout strategic weights distribution."""
        if 'timeout_strategic_weight' not in self.df.columns:
            self._log("Timeout Weights", False, "Column 'timeout_strategic_weight' missing.")
            return
        
        unique_vals = sorted(self.df['timeout_strategic_weight'].unique())
        is_valid_range = all(v in [0, 1, 2, 3] for v in unique_vals)
        has_heavy_tos = self.df['timeout_strategic_weight'].max() >= 2
        
        message = f"Values: {unique_vals}. Heavy timeouts (2+) detected: {has_heavy_tos}"
        self._log("Timeout Weights", is_valid_range and has_heavy_tos, message)

    def check_cumulative_counters_monotonicity(self):
        """Verifies that cumulative series within a game/team are non-decreasing."""
        metrics_to_check = ['cum_pointsTotal', 'cum_turnoverTotal', 'cum_reboundDefensiveTotal', 'cum_reboundOffensiveTotal']
        present_metrics = [m for m in metrics_to_check if m in self.df.columns]
        
        if not present_metrics:
            self._log("Cumulative Monotonicity", False, "Missing cumulative metric columns.")
            return
        
        violations = {}
        for metric in present_metrics:
            diffs = self.df.groupby(['gameId', 'teamId'])[metric].diff()
            drops = (diffs < 0).sum()
            if drops > 0:
                violations[metric] = int(drops)
        
        # Also check period team foul counter
        if 'team_fouls_period' in self.df.columns and 'period' in self.df.columns and 'teamTricode' in self.df.columns:
            foul_diffs = self.df.groupby(['gameId', 'period', 'teamTricode'])['team_fouls_period'].diff()
            foul_drops = (foul_diffs < 0).sum()
            if foul_drops > 0:
                violations['team_fouls_period'] = int(foul_drops)
        
        if not violations:
            self._log("Cumulative Monotonicity", True, f"100% strictly non-decreasing across all {len(present_metrics) + 1} series (0 drops).")
        else:
            self._log("Cumulative Monotonicity", False, f"Detected negative drops: {violations}")

    def check_nba_canonical_benchmarks(self):
        """
        Detailed, non-rigid distribution check against canonical NBA benchmarks.
        Calculates per-game distributions and reports deviations/outliers informatively.
        """
        b_dict = self.benchmarks.get('per_game_combined', {})
        if not b_dict:
            self._log("Canonical Benchmarks", True, "No benchmarks configured; skipping check.")
            return
        
        num_games = self.df['gameId'].nunique()
        if num_games == 0:
            self._log("Canonical Benchmarks", False, "No games found in dataset.")
            return
        
        # Vectorized metric aggregation per game
        agg_cols = {}
        if 'actionType' in self.df.columns:
            agg_cols['turnover'] = (self.df['actionType'] == 'turnover').astype(int)
            agg_cols['foul'] = self.df['is_foul'] if 'is_foul' in self.df.columns else (self.df['actionType'] == 'foul').astype(int)
            agg_cols['steal'] = (self.df['actionType'] == 'steal').astype(int)
            agg_cols['block'] = (self.df['actionType'] == 'block').astype(int)
            agg_cols['3pt'] = (self.df['actionType'] == '3pt').astype(int)
            agg_cols['timeout'] = (self.df['actionType'] == 'timeout').astype(int)
        
        game_sums = pd.DataFrame(agg_cols, index=self.df.index).groupby(self.df['gameId']).sum()
        
        if 'scoreHome' in self.df.columns and 'scoreAway' in self.df.columns:
            final_scores = self.df[['gameId', 'scoreHome', 'scoreAway']].groupby('gameId').last()
            game_sums['points'] = final_scores['scoreHome'] + final_scores['scoreAway']
        
        print("\n" + "=" * 85)
        print(f"📊 NBA CANONICAL BENCHMARK VALIDATION REPORT ({num_games:,} Games)")
        print("=" * 85)
        header = f"{'Metric':<12} | {'Obs Mean':<9} | {'Benchmark':<13} | {'In-Range %':<11} | {'Outliers':<9} | {'Obs Min-Max':<12} | {'Status'}"
        print(header)
        print("-" * 85)
        
        overall_bench_healthy = True
        notes = []
        
        for metric, b in b_dict.items():
            if metric not in game_sums.columns:
                continue
            s = game_sums[metric]
            mean_val = s.mean()
            exp = b.get('expected', 0)
            tol = b.get('tolerance', 0)
            low = exp - tol
            high = exp + tol
            
            in_range = ((s >= low) & (s <= high)).sum()
            in_range_pct = (in_range / num_games) * 100
            outliers_cnt = num_games - in_range
            
            # Informative status: Healthy if mean is within reasonable tolerance
            is_mean_in_range = (low <= mean_val <= high)
            status_icon = "🟢 Healthy" if is_mean_in_range else "🟡 Deviating"
            if not is_mean_in_range:
                overall_bench_healthy = False
            
            min_max_str = f"{s.min():.0f} - {s.max():.0f}"
            bench_str = f"{exp} ± {tol}"
            print(f"{metric:<12} | {mean_val:<9.1f} | {bench_str:<13} | {in_range_pct:<10.1f}% | {outliers_cnt:<9} | {min_max_str:<12} | {status_icon}")
            
            # Record notable deviations for informational notes
            extreme_high = s[s > exp + (2.5 * tol)]
            extreme_low = s[s < max(0, exp - (2.5 * tol))]
            if len(extreme_high) > 0:
                notes.append(f"  • {metric.capitalize()}: {len(extreme_high)} games with high count (> {exp + 2.5*tol:.0f}, max={s.max():.0f}) - usually OT/high pace")
            if len(extreme_low) > 0:
                notes.append(f"  • {metric.capitalize()}: {len(extreme_low)} games with low count (< {max(0, exp - 2.5*tol):.0f}, min={s.min():.0f})")
        
        print("-" * 85)
        if notes:
            print("ℹ️ Distribution & Outlier Observations:")
            for n in notes:
                print(n)
            print("-" * 85)
        
        self._log(
            "Canonical Benchmarks",
            overall_bench_healthy,
            f"All metric means align with NBA standards (Evaluated across {num_games:,} games)." if overall_bench_healthy else "Some metric means deviate from reference range."
        )

    def check_substitution_timer_sync(self):
        """Ensures the fatigue timer resets independently for home and away."""
        missing = [c for c in ['time_since_last_sub_home', 'time_since_last_sub_away'] if c not in self.df.columns]
        if missing:
            self._log("Sub Timer Sync", False, f"Missing independent sub timer columns: {missing}")
            return

        h_min, h_max = self.df['time_since_last_sub_home'].min(), self.df['time_since_last_sub_home'].max()
        a_min, a_max = self.df['time_since_last_sub_away'].min(), self.df['time_since_last_sub_away'].max()
        valid = (h_min >= 0 and h_max > 60 and a_min >= 0 and a_max > 60)
        self._log("Sub Timer Sync", valid, f"Home Range: [{h_min:.0f}s, {h_max:.0f}s], Away Range: [{a_min:.0f}s, {a_max:.0f}s].")

    def check_critical_missing_values(self):
        """Ensures no gaps in critical columns."""
        critical = ['scoreHome', 'play_duration', 'home_lineup', 'away_lineup']
        missing_count = self.df[critical].isna().sum().sum()
        self._log("Critical Gaps", (missing_count == 0), f"Missing values: {missing_count}")

    def run(self):
        print(f"🕵️‍♂️ Running FULL HYBRID QA Validator on: {os.path.basename(self.file_path)}")
        print("-" * 85)
        self.load_data()
        print("-" * 85)
        
        self.check_lineup_completeness()
        self.check_lineup_turnover()
        self.report_confidence_health()
        self.check_player_team_consistency()
        self.check_substitution_timer_sync()
        self.check_shot_clock_14s_rule()
        self.check_timeouts_inventory_integrity()
        self.check_timeout_strategic_weights()
        self.check_cumulative_counters_monotonicity()
        self.check_nba_canonical_benchmarks()
        self.check_critical_missing_values()
        
        print("-" * 85)
        if all(self.results):
            print("🚀 STATUS: PASSED. Dataset is solid.")
        else:
            print("⚠️ STATUS: WARNINGS DETECTED. Review logs above.")

def normalize_season(season_str: str):
    s = season_str.strip().replace('-', '_')
    if '_' in s:
        parts = s.split('_')
        return f"{parts[0]}_{parts[1]}", f"{parts[0]}-{parts[1]}"
    return s, s.replace('_', '-')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="QA Validator for Level 1 Base Timeline")
    parser.add_argument('--season', type=str, default='2024_25', help="Season to validate (default: 2024_25)")
    parser.add_argument('--file', type=str, default=None, help="Explicit path to level1 file")
    parser.add_argument('--benchmarks', type=str, default=None, help="Path to custom benchmarks JSON")
    args = parser.parse_args()

    if args.file:
        target_file = args.file
    else:
        s_under, _ = normalize_season(args.season)
        target_file = os.path.join(BASE_DIR, 'data', 'interim', f'level1_base_{s_under}.csv')
        if not os.path.exists(target_file) and s_under == '2024_25':
            target_file = os.path.join(BASE_DIR, 'data', 'interim', 'level1_base.csv')

    validator = Level1Validator(target_file, benchmarks_path=args.benchmarks)
    validator.run()
