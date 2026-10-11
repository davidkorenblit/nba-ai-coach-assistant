import pandas as pd
import numpy as np
import os

def find_perfect_game_for_demo():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    data_path = os.path.join(base_dir, 'data', 'processed', 'test.parquet')
    
    if not os.path.exists(data_path):
        print(f"❌ Error: Test dataset not found at {data_path}")
        return

    print("🔍 Scanning test dataset for the ultimate presentation game...")
    df = pd.read_parquet(data_path)
    
    # Ensure mandatory columns exist
    required_cols = ['gameId', 'period', 'score_margin', 'timeout_strategic_weight', 'target_stop_run_90s']
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0 # Placeholder if missing in current schema to prevent crash
            
    # Group by game to analyze game-wide dynamics
    games = df['gameId'].unique()
    candidates = []

    for game_id in games:
        game_df = df[df['gameId'] == game_id].sort_index().copy() # Chronological order
        
        # Filter 1: Check for garbage time by avoiding massive blowouts across the whole game
        if game_df['score_margin'].abs().max() > 25:
            continue
            
        # Analyze each period (Quarter) individually
        for period in [1, 2, 3, 4]:
            q_df = game_df[game_df['period'] == period]
            if len(q_df) < 30: # Not enough possessions in this quarter
                continue
                
            # Track score runs (diff between consecutive score margins)
            margins = q_df['score_margin'].values
            
            # Find big scoring streaks where the opponent runs away with the game
            # A streak is measured by looking at shifts in the margin over rows
            max_opponent_run = 0
            for i in range(len(margins) - 10):
                window = margins[i:i+10]
                run_size = window[-1] - window[0] # Positive or negative depending on team direction
                if abs(run_size) > max_opponent_run:
                    max_opponent_run = abs(run_size)
            
            # Check if there is a coach error zone (high strategic weight/alert but NO timeout taken)
            ignored_alerts = q_df[(q_df['timeout_strategic_weight'] > 0) & (q_df['target_stop_run_90s'] == 1)]
            
            if max_opponent_run >= 8 and len(ignored_alerts) > 0:
                candidates.append({
                    'gameId': game_id,
                    'period': period,
                    'max_opponent_run': max_opponent_run,
                    'ignored_alert_count': len(ignored_alerts),
                    'total_possessions': len(q_df)
                })

    # Sort candidates to find the most dramatic scenarios
    candidates_df = pd.DataFrame(candidates)
    if not candidates_df.empty:
        candidates_df = candidates_df.sort_values(by=['max_opponent_run', 'ignored_alert_count'], ascending=False)
        print(f"\n🎯 Found {len(candidates_df)} potential demo matches in your real data!")
        print("\nTop 5 Recommended Game IDs & Quarters for your script:")
        print(candidates_df.head(5).to_string(index=False))
        return candidates_df.iloc[0]['gameId']
    else:
        print("⚠️ No perfect real match found matching all strict criteria. We will proceed with generating a scripted game dataset.")
        return None

if __name__ == "__main__":
    find_perfect_game_for_demo()