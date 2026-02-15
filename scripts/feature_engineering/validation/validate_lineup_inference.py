import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import time

# --- Configuration ---
# גם כאן נדרשות 4 רמות (תיקנתי מ-3 ל-4 כדי שיתאים למיקום החדש)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RAW_PBP_PATH = os.path.join(BASE_DIR, 'data', 'pureData', 'season_2024_25.csv')
OUTPUT_REPORT_PATH = os.path.join(BASE_DIR, 'data', 'reports', 'lineup_inference_report.csv')
OUTPUT_PLOT_PATH = os.path.join(BASE_DIR, 'data', 'reports', 'lineup_coverage_plot.png')

class LineupInferenceEngine:
    """
    Simulates lineup tracking using ONLY Play-by-Play events.
    """
    def __init__(self, df_game):
        self.df = df_game.sort_values('orderNumber').reset_index(drop=True)
        # Identify teams ignoring NaNs
        teams = self.df['teamTricode'].dropna().unique()
        self.teams = [t for t in teams if t] 
        
        self.current_lineups = {t: set() for t in self.teams}

    def _update_from_sub(self, row):
        desc = str(row['description'])
        team = row['teamTricode']
        player_name = row['playerName']
        
        if not team or team not in self.current_lineups:
            return

        # Logic based on inspection
        if 'SUB out' in desc:
            self.current_lineups[team].discard(player_name)
        elif 'SUB in' in desc:
            self.current_lineups[team].add(player_name)

    def _update_from_action(self, row):
        team = row['teamTricode']
        player_name = row['playerName']
        
        if not team or team not in self.current_lineups or pd.isna(player_name):
            return

        # Lazy Loading: If active, they must be on court
        if player_name not in self.current_lineups[team]:
            self.current_lineups[team].add(player_name)

    def process_game(self):
        game_log = []
        
        for idx, row in self.df.iterrows():
            # Update State
            if 'SUB' in str(row['description']):
                self._update_from_sub(row)
            else:
                self._update_from_action(row)
            
            # Snapshot
            counts = {t: len(self.current_lineups[t]) for t in self.teams}
            
            # Safe access to counts
            home_c = list(counts.values())[0] if len(counts) > 0 else 0
            away_c = list(counts.values())[1] if len(counts) > 1 else 0

            game_log.append({
                'gameId': row['gameId'],
                'total_known': home_c + away_c
            })
            
        return pd.DataFrame(game_log)

def run_validation():
    print(f"🚀 Starting Lineup Inference Validation...")
    print(f"📂 Reading data from: {RAW_PBP_PATH}")
    
    if not os.path.exists(RAW_PBP_PATH):
        print("❌ Data file not found.")
        return

    # Load data
    cols = ['gameId', 'period', 'clock', 'orderNumber', 'teamTricode', 'playerName', 'description', 'actionType']
    try:
        df_all = pd.read_csv(RAW_PBP_PATH, usecols=lambda c: c in cols)
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return
    
    game_ids = df_all['gameId'].unique()
    total_games = len(game_ids)
    print(f"📊 Found {total_games} games. Processing...")

    results = []
    start_time = time.time()
    
    # Simple Loop instead of tqdm
    for i, gid in enumerate(game_ids):
        # Progress Print every 50 games
        if i % 50 == 0:
            elapsed = time.time() - start_time
            print(f"   Processing game {i}/{total_games} ({i/total_games:.1%}) - {elapsed:.1f}s elapsed")

        df_game = df_all[df_all['gameId'] == gid]
        engine = LineupInferenceEngine(df_game)
        log_df = engine.process_game()
        
        # Metrics: How much of the game do we have full info (10 players)?
        # And how much do we have decent info (>= 8 players)?
        if not log_df.empty:
            full_info = (log_df['total_known'] >= 10).mean()
            partial_info = (log_df['total_known'] >= 8).mean()
            avg_players = log_df['total_known'].mean()
        else:
            full_info, partial_info, avg_players = 0, 0, 0
        
        results.append({
            'gameId': gid,
            'full_coverage_pct': full_info,
            'partial_coverage_pct': partial_info,
            'avg_known_players': avg_players
        })

    # Save Results
    results_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(OUTPUT_REPORT_PATH), exist_ok=True)
    results_df.to_csv(OUTPUT_REPORT_PATH, index=False)
    
    print("\n📈 Summary Statistics (Inference Quality):")
    print(results_df.describe().to_string())

    # Check valid games (where we know 10 players for > 70% of the game)
    valid_games = len(results_df[results_df['full_coverage_pct'] > 0.7])
    print(f"\n✅ Games with GOOD inferred coverage (>70% time): {valid_games} / {total_games}")

    # Visualization
    try:
        plt.figure(figsize=(10, 6))
        sns.histplot(results_df['full_coverage_pct'], bins=20, kde=True, color='green')
        plt.title('Distribution of Lineup Knowledge (Inference Method)')
        plt.xlabel('% of Game Time with Full 5v5 Lineups Identified')
        plt.ylabel('Number of Games')
        plt.axvline(results_df['full_coverage_pct'].mean(), color='red', linestyle='--', label='Mean')
        plt.legend()
        plt.savefig(OUTPUT_PLOT_PATH)
        print(f"✅ Plot saved to {OUTPUT_PLOT_PATH}")
    except ImportError:
        print("⚠️ Could not create plot (matplotlib/seaborn missing or failed), but CSV is saved.")

if __name__ == "__main__":
    run_validation()