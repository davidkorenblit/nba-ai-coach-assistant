import pandas as pd
import os
import numpy as np
import argparse

def normalize_season(season_str: str):
    """
    Normalizes season input like '2024-25' or '2024_25' to both:
    (season_underscore, season_dash), e.g. ('2024_25', '2024-25').
    """
    s = season_str.strip().replace('-', '_')
    if '_' in s:
        parts = s.split('_')
        return f"{parts[0]}_{parts[1]}", f"{parts[0]}-{parts[1]}"
    return s, s.replace('_', '-')

# --- Configuration ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, '..', '..', 'data', 'pureData')
OUTPUT_DIR = os.path.join(CURRENT_DIR, '..', '..', 'docs', 'reports')
BAD_GAMES_FILE = os.path.join(OUTPUT_DIR, 'invalid_games_blacklist.csv')

# --- Threshold Rules ---
MIN_TIMEOUTS = 7       # Minimum timeouts for valid game
MAX_TIMEOUTS = 14      # Maximum timeouts allowed
MAX_SCORE_JUMP = 4     # Maximum points per possession

def validate_game(df, game_id):
    """
    Validates a single game DataFrame and returns list of invalidation reasons
    """
    reasons = []
    
    # 1. Period Integrity Check
    periods = df['period'].unique()
    if not {1, 2, 3, 4}.issubset(periods):
        reasons.append("Missing Quarters")

    # 2. Timeout Integrity Check
    timeouts = df[df['actionType'] == 'timeout']
    timeout_count = len(timeouts)
    
    if timeout_count < MIN_TIMEOUTS:
        reasons.append(f"Too Few Timeouts ({timeout_count})")
    elif timeout_count > MAX_TIMEOUTS:
        reasons.append(f"Too Many Timeouts ({timeout_count})")
        
    # Check team assignment for timeouts
    orphaned_timeouts = timeouts['teamTricode'].isnull().sum()
    if orphaned_timeouts > 0:
        reasons.append(f"Orphaned Timeouts (Missing TeamID: {orphaned_timeouts})")

    # 3. Momentum Trigger & Identity Check
    steals = df[df['actionType'] == 'steal']
    bad_steals = steals['playerName'].isnull().sum()
    if bad_steals > 0 and len(steals) > 0:
        if (bad_steals / len(steals)) > 0.5: 
            reasons.append(f"Broken Steal Data ({bad_steals} missing names)")

    blocks = df[df['actionType'] == 'block']
    bad_blocks = blocks['playerName'].isnull().sum()
    if bad_blocks > 0 and len(blocks) > 0:
        if (bad_blocks / len(blocks)) > 0.5:
            reasons.append(f"Broken Block Data ({bad_blocks} missing names)")

    # 4. Score Logic Check
    # Sort chronologically
    df = df.sort_values('orderNumber')
    
    # If columns exist
    if 'scoreHome' in df.columns and 'scoreAway' in df.columns:
        # Forward fill score column values
        df['scoreHome'] = df['scoreHome'].ffill().fillna(0)
        df['scoreAway'] = df['scoreAway'].ffill().fillna(0)
        
        # Calculate current total game points
        total_score = df['scoreHome'] + df['scoreAway']
        
        # Check score delta between rows
        score_diff = total_score.diff().fillna(0)
        
        # Verify no single possession score jump > 4
        impossible_jumps = score_diff[score_diff > MAX_SCORE_JUMP]
        if not impossible_jumps.empty:
            reasons.append(f"Impossible Score Jumps (Found {len(impossible_jumps)} events > 4pts)")
            
    return reasons

def parse_args():
    parser = argparse.ArgumentParser(description="Validate deep game logic (timeouts, periods, score jumps).")
    parser.add_argument("--season", type=str, default="2024_25", help="Season string (e.g. 2024_25, 2024-25, 2021_22)")
    parser.add_argument("--all", action="store_true", help="Validate all available seasons")
    parser.add_argument("--file", type=str, default=None, help="Explicit single CSV file to validate")
    return parser.parse_args()

def main():
    args = parse_args()
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    if args.file:
        files_to_check = [os.path.basename(args.file)]
        search_dir = os.path.dirname(os.path.abspath(args.file))
    elif args.all:
        search_dir = DATA_DIR
        candidate_seasons = ['2021_22', '2022_23', '2023_24', '2024_25']
        files_to_check = [f"season_{s}.csv" for s in candidate_seasons if os.path.exists(os.path.join(DATA_DIR, f"season_{s}.csv"))]
    else:
        search_dir = DATA_DIR
        season_underscore, _ = normalize_season(args.season)
        target_filename = f"season_{season_underscore}.csv"
        target_path = os.path.join(DATA_DIR, target_filename)
        if not os.path.exists(target_path):
            print(f"❌ Season raw file not found: {target_path}")
            return
        files_to_check = [target_filename]

    blacklist_records = []
    total_games_checked = 0
    valid_games = 0

    print(f"--- STARTING DEEP LOGIC VALIDATION ---")
    print(f"Scanning {len(files_to_check)} season file(s)...\n")

    for file in files_to_check:
        file_path = os.path.join(search_dir, file)
        print(f"Processing {file}...")
        
        try:
            df = pd.read_csv(file_path, low_memory=False)
            unique_games = df['gameId'].unique()
            
            for game_id in unique_games:
                total_games_checked += 1
                game_df = df[df['gameId'] == game_id].copy()
                
                # Execute validation logic
                issues = validate_game(game_df, game_id)
                
                if issues:
                    matchup = game_df['matchup'].iloc[0] if 'matchup' in game_df.columns else "Unknown"
                    for issue in issues:
                        blacklist_records.append({
                            'gameId': game_id,
                            'matchup': matchup,
                            'season': file, # Source file
                            'reason': issue
                        })
                else:
                    valid_games += 1
                    
        except Exception as e:
            print(f"Error reading {file}: {e}")

    # Save report
    print(f"\n--- VALIDATION COMPLETE ---")
    print(f"Total Games Checked: {total_games_checked}")
    print(f"Valid Games: {valid_games}")
    print(f"Invalid Games: {len(set([x['gameId'] for x in blacklist_records]))}") # Count unique games
    
    if blacklist_records:
        blacklist_df = pd.DataFrame(blacklist_records)
        blacklist_df.to_csv(BAD_GAMES_FILE, index=False)
        print(f"\n[X] Blacklist saved to: {BAD_GAMES_FILE}")
        if not args.all and not args.file:
            season_specific_blacklist = os.path.join(OUTPUT_DIR, f'invalid_games_blacklist_{season_underscore}.csv')
            blacklist_df.to_csv(season_specific_blacklist, index=False)
            print(f"[X] Season blacklist saved to: {season_specific_blacklist}")
        print("Preview of bad games:")
        print(blacklist_df.head())
    else:
        print("\n[V] AMAZING! No logic errors found in any game.")

if __name__ == "__main__":
    main()
