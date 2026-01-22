import pandas as pd
import numpy as np
import os
import sys
from tqdm import tqdm

# --- Config & Settings ---
pd.set_option('future.no_silent_downcasting', True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PBP_PATH = os.path.join(BASE_DIR, 'data', 'pureData', 'season_2024_25.csv')
ROTATIONS_PATH = os.path.join(BASE_DIR, 'data', 'pureData', 'rotations_2024_25.csv')
OUTPUT_FILE = os.path.join(BASE_DIR, 'data', 'interim', 'level1_base.csv')

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

def get_absolute_time(row):
    """Calculates absolute seconds elapsed from start of game for rotation matching."""
    p = row['period']
    rem = row['seconds_remaining']
    # Regulation (4x12m) + OT (5m)
    if p <= 4:
        return (p - 1) * 720 + (720 - rem)
    else:
        return (4 * 720) + ((p - 5) * 300) + (300 - rem)

# --- Core Logic Steps ---

def load_and_prep_data():
    print("🔹 Loading datasets...")
    if not os.path.exists(RAW_PBP_PATH): raise FileNotFoundError(f"Missing PBP: {RAW_PBP_PATH}")
    if not os.path.exists(ROTATIONS_PATH): raise FileNotFoundError(f"Missing Rotations: {ROTATIONS_PATH}")
    
    df_pbp = pd.read_csv(RAW_PBP_PATH, low_memory=False)
    df_rot = pd.read_csv(ROTATIONS_PATH)
    
    # Ensure IDs match
    df_pbp['gameId'] = df_pbp['gameId'].astype(str).str.zfill(10)
    df_rot['gameId'] = df_rot['gameId'].astype(str).str.zfill(10)
    
    return df_pbp, df_rot

def process_base_timeline(df):
    print("🔹 Processing timeline and scores...")
    if 'teamTricode' in df.columns:
        df['teamTricode'] = df['teamTricode'].astype(str).str.strip()
    
    df['seconds_remaining'] = df['clock'].apply(parse_clock)
    df['time_elapsed'] = df.apply(get_absolute_time, axis=1) # NEW: For rotation sync
    
    df.sort_values(by=['gameId', 'period', 'seconds_remaining', 'actionNumber'], 
                   ascending=[True, True, False, True], inplace=True)
    
    for col in ['scoreHome', 'scoreAway']:
        df[col] = df.groupby('gameId')[col].ffill().fillna(0)
    df['score_margin'] = df['scoreHome'] - df['scoreAway']

    cols = ['reboundDefensiveTotal', 'reboundOffensiveTotal', 'turnoverTotal', 'foulPersonalTotal', 'pointsTotal']
    for c in cols:
        if c in df.columns: df[c] = df[c].fillna(0)
        
    return df

def enrich_state_counters(df):
    """Restored from Old Code: Timeouts, Fouls, Cumulatives."""
    print("🔹 Enriching State Counters (Timeouts & Fouls)...")
    
    # 1. Identify Home/Away
    home_id_map = df[df['scoreHome'].diff() > 0].groupby('gameId')['teamId'].first().to_dict()
    id_to_code = df.dropna(subset=['teamTricode']).groupby('teamId')['teamTricode'].first().to_dict()
    
    def _resolve_timeout_role(row):
        if row['actionType'] == 9 or 'Timeout' in str(row['description']):
            current_code = str(row['teamTricode']).strip()
            hid = home_id_map.get(row['gameId'])
            home_code = id_to_code.get(hid)
            if current_code == home_code: return 'home'
            if current_code != 'nan': return 'away'
        return 'none'

    df['timeout_role'] = df.apply(_resolve_timeout_role, axis=1)
    
    # 2. Inventory
    for side in ['home', 'away']:
        is_side_to = (df['timeout_role'] == side).astype(int)
        used = is_side_to.groupby(df['gameId']).cumsum()
        df[f'timeouts_remaining_{side}'] = (7 - used).clip(lower=0)

    # 3. Fouls & Counters
    df['is_foul'] = (df['foulPersonalTotal'] > 0).astype(int)
    df['team_fouls_period'] = df.groupby(['gameId', 'period', 'teamTricode'])['is_foul'].cumsum().fillna(0)

    cols_to_sum = ['pointsTotal', 'turnoverTotal', 'reboundDefensiveTotal']
    for metric in cols_to_sum:
        df[metric] = pd.to_numeric(df[metric], errors='coerce').fillna(0)
        df[f'cum_{metric}'] = df.groupby(['gameId', 'teamId'])[metric].cumsum().fillna(0)
        
    return df

def calculate_flow_metrics(df):
    """Restored from Old Code: Temporal & Possession."""
    print("🔹 Calculating Flow Metrics...")
    
    # Temporal
    prev_time = df.groupby(['gameId', 'period'])['seconds_remaining'].shift(1)
    df['play_duration'] = (prev_time - df['seconds_remaining']).fillna(0).clip(lower=0)
    
    # Possession
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
    """Restored from Old Code."""
    elapsed = df.groupby(['gameId', 'possession_id'])['play_duration'].cumsum()
    df['shot_clock_estimated'] = (24.0 - elapsed).clip(lower=0)
    mask_off_reb = df['reboundOffensiveTotal'] > 0
    df.loc[mask_off_reb, 'shot_clock_estimated'] = 14.0
    return df

def merge_real_rotations(df_pbp, df_rot):
    """NEW: Merges external rotation file instead of guessing."""
    print("🔹 Merging REAL Rotations (This takes time)...")
    
    rot_map = {}
    print("   -> Building Index...")
    for gid, group in tqdm(df_rot.groupby('gameId')):
        rot_map[gid] = {'home': [], 'away': []}
        for _, row in group.iterrows():
            entry = (float(row['IN_TIME_REAL']), float(row['OUT_TIME_REAL']), int(row['PERSON_ID']))
            rot_map[gid][row['team_side']].append(entry)

    result_home = []
    result_away = []
    
    print("   -> Mapping...")
    grouped_pbp = df_pbp.groupby('gameId')
    
    for gid, game_events in tqdm(grouped_pbp):
        if gid not in rot_map:
            result_home.extend([None] * len(game_events))
            result_away.extend([None] * len(game_events))
            continue
            
        times = game_events['time_elapsed'].values
        
        # Optimize: Pre-fetch lists
        h_intervals = rot_map[gid]['home']
        a_intervals = rot_map[gid]['away']
        
        # List comprehension is faster than apply here
        h_lineups = [sorted([pid for (s, e, pid) in h_intervals if s <= t + 0.1 < e]) for t in times]
        a_lineups = [sorted([pid for (s, e, pid) in a_intervals if s <= t + 0.1 < e]) for t in times]
            
        result_home.extend(h_lineups)
        result_away.extend(a_lineups)
        
    df_pbp['home_lineup'] = result_home
    df_pbp['away_lineup'] = result_away
    return df_pbp

def calculate_fatigue_and_subs(df):
    """Updated logic using the real lineups."""
    print("🔹 Calculating Fatigue...")
    
    df['home_sig'] = df['home_lineup'].astype(str)
    df['away_sig'] = df['away_lineup'].astype(str)
    df['lineup_signature'] = df['home_sig'] + "|" + df['away_sig']
    
    df['is_new_period'] = (df['period'] != df.groupby('gameId')['period'].shift(1)).astype(int)
    shift_sig = df.groupby('gameId')['lineup_signature'].shift(1)
    
    df['is_sub'] = np.where(
        (df['lineup_signature'] != shift_sig) & (df['is_new_period'] == 0) & (shift_sig.notna()), 
        1, 0
    )
    
    df['stint_id'] = df.groupby('gameId')['is_sub'].cumsum()
    grp = df.groupby(['gameId', 'period', 'stint_id'])
    start_times = grp['seconds_remaining'].transform('max')
    df['time_since_last_sub'] = start_times - df['seconds_remaining']
    
    df.drop(columns=['home_sig', 'away_sig', 'lineup_signature', 'is_new_period', 'stint_id'], inplace=True)
    return df

def clean_sparse_columns(df):
    cols_to_drop = [
        'assistPlayerNameInitial', 'assistPersonId', 'assistTotal',
        'stealPlayerName', 'stealPersonId', 'blockPlayerName', 'blockPersonId',
        'timeout_role'
    ]
    existing = [c for c in cols_to_drop if c in df.columns]
    if existing: df.drop(columns=existing, inplace=True)
    return df

# --- Main ---

def main():
    print(f"🚀 Starting Level 1 FE (V8 - COMPLETE INTEGRATION)...")
    try:
        # 1. Load
        df, df_rot = load_and_prep_data()
        
        # 2. Timeline & Scores
        df = process_base_timeline(df)
        
        # 3. State Counters (Restored!)
        df = enrich_state_counters(df)
        
        # 4. Flow Metrics (Restored!)
        df = calculate_flow_metrics(df)
        df = apply_shot_clock_logic(df)
        
        # 5. Real Rotations (New!)
        df = merge_real_rotations(df, df_rot)
        
        # 6. Fatigue (Updated!)
        df = calculate_fatigue_and_subs(df)
        
        # 7. Cleanup
        df = clean_sparse_columns(df)
        
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"✅ Level 1 DONE. Saved to {OUTPUT_FILE}")
        
    except Exception as e:
        print(f"❌ Critical Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()