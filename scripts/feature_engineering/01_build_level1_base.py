import pandas as pd
import numpy as np
import os
import re

# --- Config & Constants ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_FILE_PATH = os.path.join(CURRENT_DIR, '..', '..', 'data', 'pureData', 'season_2024_25.csv')
OUTPUT_FILE = os.path.join(CURRENT_DIR, '..', '..', 'data', 'interim', 'level1_base.csv')

# --- Helper Functions (Utils) ---
def parse_clock(clock_str):
    """Utility: Converts clock string to float seconds."""
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

# --- Feature Implementation Modules ---

def process_base_timeline(df):
    """
    Implements: Precise Game Clock, Score & Margin, sorting.
    """
    # 1. Time Normalization
    df['seconds_remaining'] = df['clock'].apply(parse_clock)
    
    # 2. Chronological Sort
    df.sort_values(by=['gameId', 'period', 'seconds_remaining', 'actionNumber'], 
                   ascending=[True, True, False, True], inplace=True)
    
    # 3. Score State (Fill Gaps)
    for col in ['scoreHome', 'scoreAway']:
        df[col] = df.groupby('gameId')[col].ffill().fillna(0)
    
    df['score_margin'] = df['scoreHome'] - df['scoreAway']
    
    # 4. Fill NaNs for numerical counters (Safety)
    zero_fill_cols = ['reboundDefensiveTotal', 'reboundOffensiveTotal', 'turnoverTotal', 'foulPersonalTotal', 'pointsTotal']
    for c in zero_fill_cols:
        if c in df.columns: df[c] = df[c].fillna(0)
        
    return df

def enrich_state_counters(df):
    """
    FIXED V3: Better Timeout detection logic & Clean counters.
    """
    # --- Timeouts Fix ---
    # לוגיקה משולבת: אם יש טריקוד, קח אותו. אם אין, נסה לחלץ מהתיאור.
    def _resolve_timeout_team(row):
        # האם זה אירוע טיים אאוט?
        if row['actionType'] == 9 or 'Timeout' in str(row['description']):
            # עדיפות 1: הקוד הקיים בעמודה
            if pd.notna(row['teamTricode']):
                return row['teamTricode']
            # עדיפות 2: חילוץ מהטקסט (גיבוי)
            parts = str(row['description']).split()
            return parts[0].strip() if parts else 'Unknown'
        return 'None'

    df['timeout_team'] = df.apply(_resolve_timeout_team, axis=1)
    
    # חישוב מלאי פסקי זמן
    teams = [t for t in df['teamTricode'].dropna().unique()]
    for team in teams:
        # סופרים רק אם זו הקבוצה הספציפית
        is_team_to = (df['timeout_team'] == team).astype(int)
        used = is_team_to.groupby(df['gameId']).cumsum()
        df[f'timeouts_remaining_{team}'] = (7 - used).clip(lower=0)

    # --- Fouls ---
    df['is_foul'] = (df['foulPersonalTotal'] > 0).astype(int)
    df['team_fouls_period'] = df.groupby(['gameId', 'period', 'teamTricode'])['is_foul'].cumsum().fillna(0)

    # --- Event Counters (Cumulative) ---
    cols_to_sum = ['pointsTotal', 'turnoverTotal', 'reboundDefensiveTotal']
    for metric in cols_to_sum:
        # המרה בטוחה למספרים
        df[metric] = pd.to_numeric(df[metric], errors='coerce').fillna(0)
        df[f'cum_{metric}'] = df.groupby(['gameId', 'teamId'])[metric].cumsum().fillna(0)
        
    return df

def calculate_temporal_metrics(df):
    """
    Implements: Play Duration, Shot Clock Remaining.
    """
    prev_time = df.groupby(['gameId', 'period'])['seconds_remaining'].shift(1)
    df['play_duration'] = (prev_time - df['seconds_remaining']).fillna(0).clip(lower=0)
    return df

def calculate_possession_flow(df):
    """
    Implements: Possession ID.
    """
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
    """
    Implements: Shot Clock (Final Calculation).
    """
    elapsed = df.groupby(['gameId', 'possession_id'])['play_duration'].cumsum()
    df['shot_clock_estimated'] = (24.0 - elapsed).clip(lower=0)
    
    mask_off_reb = df['reboundOffensiveTotal'] > 0
    df.loc[mask_off_reb, 'shot_clock_estimated'] = 14.0
    return df

def process_lineups_logic(df):
    """
    FIXED V3: Added Sorting to prevent false substitutions.
    """
    # 1. Map Players
    valid_players = df[df['personId'] > 0]
    player_map = valid_players.groupby('personId')['teamId'].agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else 0
    ).to_dict()
    
    scoring = df[df['scoreHome'].diff() > 0]
    home_map = {}
    if not scoring.empty:
        home_map = scoring.groupby('gameId')['teamId'].agg(lambda x: x.mode().iloc[0]).to_dict()

    # 2. Parse & SORT IDs
    def _parse(row):
        gid = row['gameId']
        raw = str(row['personIdsFilter'])
        if not raw or raw == '0': return [], []
        hid = home_map.get(gid)
        if not hid: return [], []
        
        ids = [int(x) for x in re.findall(r'\d+', raw)]
        ids.sort() # <--- THE FIX: Sorting ensures [1,2,3] == [3,2,1]
        
        return ([p for p in ids if player_map.get(p) == hid], 
                [p for p in ids if player_map.get(p) and player_map.get(p) != hid])

    lineups = df.apply(_parse, axis=1, result_type='expand')
    df['home_lineup'] = lineups[0]
    df['away_lineup'] = lineups[1]

    # 3. Last Sub Time (Detect changes on sorted lists)
    df['lineup_signature'] = df['home_lineup'].astype(str) + "|" + df['away_lineup'].astype(str)
    df['is_sub'] = (df['lineup_signature'] != df.groupby('gameId')['lineup_signature'].shift(1)).astype(int)
    
    df['lineup_era'] = df.groupby('gameId')['is_sub'].cumsum()
    start_times = df.groupby(['gameId', 'lineup_era'])['seconds_remaining'].transform('max')
    df['time_since_last_sub'] = start_times - df['seconds_remaining']
    
    df.drop(columns=['is_sub', 'lineup_era', 'lineup_signature'], inplace=True)
    return df

def clean_sparse_columns(df):
    """
    NEW V3: Drops columns confirmed as 'Dead' or 'Redundant' by QA.
    """
    cols_to_drop = [
        'assistPlayerNameInitial', 'assistPersonId', 'assistTotal',
        'stealPlayerName', 'stealPersonId',
        'blockPlayerName', 'blockPersonId'
    ]
    # מסננים רק מה שקיים בפועל
    existing_cols = [c for c in cols_to_drop if c in df.columns]
    
    if existing_cols:
        print(f"   🧹 Cleaning up {len(existing_cols)} dead/redundant columns...")
        df.drop(columns=existing_cols, inplace=True)
        
    return df

# --- Main Pipeline ---

def main():
    print(f"🚀 Starting Optimized Level 1 FE (V3) on: {os.path.basename(RAW_FILE_PATH)}")
    
    if not os.path.exists(RAW_FILE_PATH):
        print(f"❌ File not found: {RAW_FILE_PATH}"); return

    df = pd.read_csv(RAW_FILE_PATH, low_memory=False)
    print(f"   Loaded {len(df)} rows.")

    # --- Pipeline Execution ---
    print("   ⏱️  Phase 1: Timeline, Score & Base State...")
    df = process_base_timeline(df)
    
    print("   📊 Phase 2: Inventory (Timeouts Fix) & Counters...")
    df = enrich_state_counters(df)

    print("   ⏳ Phase 3: Play Duration Calculation...")
    df = calculate_temporal_metrics(df)

    print("   🏀 Phase 4: Deterministic Possession Logic...")
    df = calculate_possession_flow(df)

    print("   ⏲️  Phase 5: Shot Clock Estimation...")
    df = apply_shot_clock_logic(df)
    
    print("   👥 Phase 6: Lineups (Sorted) & Sub Timer...")
    df = process_lineups_logic(df)

    print("   🧹 Phase 7: Final Cleanup (Sparsity Removal)...")
    df = clean_sparse_columns(df)

    # Save
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"✅ DONE. File saved: {OUTPUT_FILE}")
    print(f"   Final Shape: {df.shape}")

if __name__ == "__main__":
    main()