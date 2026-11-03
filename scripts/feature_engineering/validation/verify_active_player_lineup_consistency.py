import pandas as pd
import numpy as np
import os
import sys
import ast
import argparse

# --- Config ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def normalize_season(season_str: str):
    s = season_str.strip().replace('-', '_')
    if '_' in s:
        parts = s.split('_')
        return f"{parts[0]}_{parts[1]}", f"{parts[0]}-{parts[1]}"
    return s, s.replace('_', '-')

class ActivePlayerConsistencyValidator:
    """
    Validates Lineup Reconstruction by verifying that any player performing
    an active basketball event (shot, rebound, foul, turnover, steal, block, freethrow)
    is tracked within the active lineup.
    """

    def __init__(self, file_path: str, season_label: str = "Unknown"):
        self.file_path = file_path
        self.season_label = season_label
        self.df = None

    def load_data(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")
        
        cols_to_use = [
            'gameId', 'period', 'actionNumber', 'actionType', 'subType',
            'personId', 'teamId', 'home_lineup', 'away_lineup', 'lineup_confidence'
        ]
        
        # Read only required columns for speed and memory efficiency
        print(f"📂 Loading dataset for {self.season_label} from: {os.path.basename(self.file_path)}")
        self.df = pd.read_csv(
            self.file_path,
            usecols=lambda c: c in cols_to_use,
            low_memory=False
        )
        print(f"✅ Loaded {len(self.df):,} rows.")

    def run_verification(self):
        # 1. Filter for active player events (personId > 0, exclude non-player events)
        non_player_actions = {'period', 'game', 'timeout', 'stoppage', 'instantreplay', 'substitution'}
        active_mask = (
            self.df['personId'].notna() & 
            (self.df['personId'] > 0) & 
            (~self.df['actionType'].isin(non_player_actions))
        )
        
        active_events = self.df[active_mask].copy()
        total_active_events = len(active_events)
        
        if total_active_events == 0:
            print("❌ No active player events found to evaluate.")
            return {}

        # 2. Check if player is present in either home or away lineup
        # String search is vectorized and orders of magnitude faster than ast.literal_eval on millions of rows
        pids_str = active_events['personId'].astype(int).astype(str)
        home_str = active_events['home_lineup'].astype(str)
        away_str = active_events['away_lineup'].astype(str)
        
        in_home = [pid in h for pid, h in zip(pids_str, home_str)]
        in_away = [pid in a for pid, a in zip(pids_str, away_str)]
        active_events['in_lineup'] = np.array(in_home) | np.array(in_away)

        # 3. Classify: Already Present vs. Discovery Event (first action of period)
        # A discovery event occurs when an active player registers their very first action in that period
        active_events['is_first_action_in_period'] = (
            active_events.groupby(['gameId', 'period', 'personId']).cumcount() == 0
        )
        
        # Metrics
        overall_in_lineup = active_events['in_lineup'].mean() * 100
        
        # Segment by confidence (Official vs Fallback)
        is_fallback = active_events['lineup_confidence'] < 1.0 if 'lineup_confidence' in active_events.columns else pd.Series(False, index=active_events.index)
        has_fallback_data = is_fallback.any()
        
        # If player already discovered in period, what is presence rate?
        post_discovery_events = active_events[~active_events['is_first_action_in_period']]
        post_discovery_presence = (
            post_discovery_events['in_lineup'].mean() * 100 if len(post_discovery_events) > 0 else 100.0
        )
        
        # Fallback specific metrics
        fb_events = active_events[is_fallback]
        fb_post_disc = post_discovery_events[post_discovery_events['lineup_confidence'] < 1.0] if has_fallback_data else pd.DataFrame()
        
        fb_overall_presence = fb_events['in_lineup'].mean() * 100 if len(fb_events) > 0 else 0.0
        fb_post_disc_presence = fb_post_disc['in_lineup'].mean() * 100 if len(fb_post_disc) > 0 else 0.0

        print("\n" + "=" * 80)
        print(f"🔬 ACTIVE PLAYER LINEUP CONSISTENCY REPORT: {self.season_label}")
        print("=" * 80)
        print(f"Total Evaluated Active Actions: {total_active_events:,} plays")
        print(f"  • Pre-action Presence (All Plays):         {overall_in_lineup:.2f}%")
        print(f"  • Established Plays (Post-Discovery):      {post_discovery_presence:.2f}%")
        
        if has_fallback_data:
            print("-" * 80)
            print(f"📊 Fallback-Only Subset ({len(fb_events):,} actions | {len(fb_events)/total_active_events*100:.1f}% of season):")
            print(f"  • Pre-action Presence (Raw):               {fb_overall_presence:.2f}%")
            print(f"  • Established Plays (Post-Discovery):      {fb_post_disc_presence:.2f}%")
            print(f"  • Quarter Discovery Events (Expected Gap): {100.0 - fb_overall_presence:.2f}%")
        print("=" * 80)
        
        results = {
            'season': self.season_label,
            'total_actions': total_active_events,
            'overall_presence_pct': overall_in_lineup,
            'post_discovery_presence_pct': post_discovery_presence,
            'has_fallback': bool(has_fallback_data),
            'fb_overall_pct': fb_overall_presence,
            'fb_post_discovery_pct': fb_post_disc_presence
        }
        return results

def main():
    parser = argparse.ArgumentParser(description="Active Player Lineup Consistency Validator")
    parser.add_argument('--season', type=str, default='2021_22', help="Season to evaluate (e.g. 2021_22, 2024_25)")
    parser.add_argument('--file', type=str, default=None, help="Explicit path to level1 file")
    args = parser.parse_args()

    interim_dir = os.path.join(BASE_DIR, 'data', 'interim')
    
    if args.file:
        target_file = args.file
        season_label = os.path.basename(args.file)
    else:
        s_under, s_dash = normalize_season(args.season)
        season_label = s_dash
        target_file = os.path.join(interim_dir, f'level1_base_{s_under}.csv')
        if not os.path.exists(target_file) and s_under == '2024_25':
            target_file = os.path.join(interim_dir, 'level1_base.csv')

    validator = ActivePlayerConsistencyValidator(target_file, season_label)
    validator.load_data()
    validator.run_verification()

if __name__ == "__main__":
    main()
