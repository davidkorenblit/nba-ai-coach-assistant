import pandas as pd
import numpy as np
import os
import re

# --- Config ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_FILE_PATH = os.path.join(CURRENT_DIR, '..', '..', 'data', 'pureData', 'season_2024_25.csv')
OUTPUT_FILE = os.path.join(CURRENT_DIR, '..', '..', 'data', 'interim', 'level1_base.csv')

# --- Helper Functions ---
def parse_clock(clock_str):
    if pd.isna(clock_str): return 0.0
    s = str(clock_str).strip()
    try:
        if 'M' in s: 
            mins, secs = s.replace('PT','').replace('S','').split('M')
            return float(mins) * 60 + float(secs)
        elif ':' in s: 
            mins, secs = s.split(':')
            return float(mins) * 60 + float(secs)
        return float(s)
    except: return 0.0

# --- Feature Modules ---

def process_base_timeline(df):
    # 1. Clean Team Codes (Critical Fix for Equality Checks)
    if 'teamTricode' in df.columns:
        df['teamTricode'] = df['teamTricode'].astype(str).str.strip()
    
    # 2. Time
    df['seconds_remaining'] = df['clock'].apply(parse_clock)
    
    # 3. Sort
    df.sort_values(by=['gameId', 'period', 'seconds_remaining', 'actionNumber'], 
                   ascending=[True, True, False, True], inplace=True)
    
    # 4. Score Fill
    for col in ['scoreHome', 'scoreAway']:
        df[col] = df.groupby('gameId')[col].ffill().fillna(0)
    df['score_margin'] = df['scoreHome'] - df['scoreAway']

    # 5. Fill Counters
    zero_fill_cols = ['reboundDefensiveTotal', 'reboundOffensiveTotal', 'turnoverTotal', 'foulPersonalTotal', 'pointsTotal']
    for c in zero_fill_cols:
        if c in df.columns: df[c] = df[c].fillna(0)
        
    return df

def enrich_state_counters_v4(df):
    """
    V4 FIX: Calculates Home/Away Timeouts instead of sparse Team columns.
    Solves the Dashboard issue permanently.
    """
    # 1. Identify Home/Away Teams per Game
    # We create a map of gameId -> {home: 'BOS', away: 'ATL'}
    # Helper: Find the tricode associated with teamId for Home/Away
    
    # מיפוי: gameId -> teamId של הבית
    home_id_map = df[df['scoreHome'].diff() > 0].groupby('gameId')['teamId'].first().to_dict()
    
    # מיפוי: teamId -> teamTricode
    id_to_code = df.dropna(subset=['teamTricode']).groupby('teamId')['teamTricode'].first().to_dict()
    
    # פונקציה לזיהוי מי הבית ומי החוץ בכל שורה
    def get_role(row):
        hid = home_id_map.get(row['gameId'])
        if not hid: return 'unknown'
        if row['teamId'] == hid: return 'home'
        if row['teamId'] != 0 and row['teamId'] != hid: return 'away'
        return 'neutral'

    # 2. Resolve Timeout Takers (Clean Logic)
    def _resolve_timeout_role(row):
        # זיהוי אם זה טיים אאוט
        if row['actionType'] == 9 or 'Timeout' in str(row['description']):
            # בדיקה האם הקבוצה שלקחה היא הבית או החוץ
            # נשתמש ב-teamTricode הקיים בשורה
            current_code = str(row['teamTricode']).strip()
            
            # מציאת קוד הבית למשחק הזה
            hid = home_id_map.get(row['gameId'])
            home_code = id_to_code.get(hid)
            
            if current_code == home_code: return 'home'
            if current_code != 'nan': return 'away' # אם זה לא הבית וזה לא ריק, זה החוץ
        return 'none'

    df['timeout_role'] = df.apply(_resolve_timeout_role, axis=1)
    
    # 3. Calculate Inventory (Home/Away)
    for side in ['home', 'away']:
        is_side_to = (df['timeout_role'] == side).astype(int)
        used = is_side_to.groupby(df['gameId']).cumsum()
        df[f'timeouts_remaining_{side}'] = (7 - used).clip(lower=0)

    # 4. Fouls & Counters
    df['is_foul'] = (df['foulPersonalTotal'] > 0).astype(int)
    # כאן נשתמש בטריקוד כדי לפצל לקבוצות בגרפים אם נרצה, אבל המדד המרכזי הוא ה-Timeouts
    df['team_fouls_period'] = df.groupby(['gameId', 'period', 'teamTricode'])['is_foul'].cumsum().fillna(0)

    cols_to_sum = ['pointsTotal', 'turnoverTotal', 'reboundDefensiveTotal']
    for metric in cols_to_sum:
        df[metric] = pd.to_numeric(df[metric], errors='coerce').fillna(0)
        df[f'cum_{metric}'] = df.groupby(['gameId', 'teamId'])[metric].cumsum().fillna(0)
        
    return df

def calculate_temporal_metrics(df):
    prev_time = df.groupby(['gameId', 'period'])['seconds_remaining'].shift(1)
    df['play_duration'] = (prev_time - df['seconds_remaining']).fillna(0).clip(lower=0)
    return df

def calculate_possession_flow(df):
    is_def_reb = df['reboundDefensiveTotal'] > 0
    is_turnover = df['turnoverTotal'] > 0
    
    if 'shotResult' in df.columns:
        is_made_shot = df['shotResult'] == 'Made'
    else:
        is_made_shot = (df.groupby('gameId')['scoreHome'].diff() + 
                        df.groupby('gameId')['scoreAway'].diff()) > 0
        
    df['is_poss_change'] = (is_def_reb | is_turnover | is_made_shot).astype(int)
    df['possession_id'] = df.groupby('gameId')['is_poss_change'].cumsum()
    return df

def apply_shot_clock_logic(df):
    elapsed = df.groupby(['gameId', 'possession_id'])['play_duration'].cumsum()
    df['shot_clock_estimated'] = (24.0 - elapsed).clip(lower=0)
    mask_off_reb = df['reboundOffensiveTotal'] > 0
    df.loc[mask_off_reb, 'shot_clock_estimated'] = 14.0
    return df

def process_lineups_logic(df):
    valid_players = df[df['personId'] > 0]
    player_map = valid_players.groupby('personId')['teamId'].agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else 0
    ).to_dict()
    
    scoring = df[df['scoreHome'].diff() > 0]
    home_map = {}
    if not scoring.empty:
        home_map = scoring.groupby('gameId')['teamId'].agg(lambda x: x.mode().iloc[0]).to_dict()

    def _parse(row):
        gid = row['gameId']
        raw = str(row['personIdsFilter'])
        if not raw or raw == '0': return [], []
        hid = home_map.get(gid)
        if not hid: return [], []
        
        ids = [int(x) for x in re.findall(r'\d+', raw)]
        ids.sort() # Sorting fix
        
        return ([p for p in ids if player_map.get(p) == hid], 
                [p for p in ids if player_map.get(p) and player_map.get(p) != hid])

    lineups = df.apply(_parse, axis=1, result_type='expand')
    df['home_lineup'] = lineups[0]
    df['away_lineup'] = lineups[1]

    df['lineup_signature'] = df['home_lineup'].astype(str) + "|" + df['away_lineup'].astype(str)
    df['is_sub'] = (df['lineup_signature'] != df.groupby('gameId')['lineup_signature'].shift(1)).astype(int)
    
    df['lineup_era'] = df.groupby('gameId')['is_sub'].cumsum()
    start_times = df.groupby(['gameId', 'lineup_era'])['seconds_remaining'].transform('max')
    df['time_since_last_sub'] = start_times - df['seconds_remaining']
    
    df.drop(columns=['is_sub', 'lineup_era', 'lineup_signature'], inplace=True)
    return df

def clean_sparse_columns(df):
    cols_to_drop = [
        'assistPlayerNameInitial', 'assistPersonId', 'assistTotal',
        'stealPlayerName', 'stealPersonId',
        'blockPlayerName', 'blockPersonId',
        'timeout_role' # Cleanup helper
    ]
    existing_cols = [c for c in cols_to_drop if c in df.columns]
    if existing_cols:
        df.drop(columns=existing_cols, inplace=True)
    return df

# --- Main ---

def main():
    print(f"🚀 Starting V4 FE (Fixing Whitespace & Home/Away Logic)...")
    if not os.path.exists(RAW_FILE_PATH):
        print(f"❌ File not found: {RAW_FILE_PATH}"); return

    df = pd.read_csv(RAW_FILE_PATH, low_memory=False)
    
    df = process_base_timeline(df) # Includes whitespace strip
    df = enrich_state_counters_v4(df) # Logic fix for timeouts
    df = calculate_temporal_metrics(df)
    df = calculate_possession_flow(df)
    df = apply_shot_clock_logic(df)
    df = process_lineups_logic(df)
    df = clean_sparse_columns(df)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ V4 DONE. Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()