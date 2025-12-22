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

def extract_team_from_desc(row):
    """Utility: Extracts team tricode from description for timeouts."""
    if row['actionType'] == 9 or 'Timeout' in str(row['description']):
        parts = str(row['description']).split()
        return parts[0].strip() if parts else 'General'
    return 'None'

# --- Feature Implementation Modules ---

def process_base_timeline(df):
    """
    Implements: Precise Game Clock, Score & Margin, sorting.
    Pattern: Pre-processing & Sorting.
    """
    # 1. Time Normalization
    df['seconds_remaining'] = df['clock'].apply(parse_clock)
    
    # 2. Chronological Sort (Critical for all next steps)
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
    Implements: Timeouts Remaining, Foul Count, Event Counters.
    Pattern: GroupBy + CumSum / Rolling logic.
    """
    # A. Timeouts Parsing & Inventory
    df['timeout_type'] = df.apply(extract_team_from_desc, axis=1)
    
    # חישוב מלאי פסקי זמן
    teams = [t for t in df['timeout_type'].unique() if t not in ['None', 'General']]
    for team in teams:
        is_team_to_series = (df['timeout_type'] == team).astype(int)
        used = is_team_to_series.groupby(df['gameId']).cumsum()
        df[f'timeouts_remaining_{team}'] = (7 - used).clip(lower=0)

    # B. Foul Counts (Per Quarter) - THE FIX IS HERE
    df['is_foul'] = (df['foulPersonalTotal'] > 0).astype(int)
    
    # חישוב מצטבר. שורות ללא קבוצה יקבלו NaN
    df['team_fouls_period'] = df.groupby(['gameId', 'period', 'teamTricode'])['is_foul'].cumsum()
    
    # --- התיקון: מילוי חורים לאירועים ללא קבוצה (כמו סוף רבע) ---
    df['team_fouls_period'] = df['team_fouls_period'].fillna(0)

    # C. Event Counters (Cumulative)
    cols_to_sum = ['pointsTotal', 'turnoverTotal', 'reboundDefensiveTotal']
    for metric in cols_to_sum:
        df[metric] = pd.to_numeric(df[metric], errors='coerce').fillna(0)
        df[f'cum_{metric}'] = df.groupby(['gameId', 'teamId'])[metric].cumsum()
        # גם כאן כדאי למלא אפסים ליתר ביטחון
        df[f'cum_{metric}'] = df[f'cum_{metric}'].fillna(0)
        
    return df

def calculate_temporal_metrics(df):
    """
    Implements: Play Duration, Shot Clock Remaining.
    Pattern: Shift & Diff.
    """
    # Play Duration
    # shift(1) מביא את הזמן של השורה הקודמת (שהוא גדול יותר)
    prev_time = df.groupby(['gameId', 'period'])['seconds_remaining'].shift(1)
    df['play_duration'] = (prev_time - df['seconds_remaining']).fillna(0).clip(lower=0)
    
    return df

def calculate_possession_flow(df):
    """
    Implements: Possession ID.
    Pattern: Boolean Masks & Cumulative Sum.
    """
    # Deterministic Triggers (No text parsing!)
    is_def_reb = df['reboundDefensiveTotal'] > 0
    is_turnover = df['turnoverTotal'] > 0
    
    # Shot Logic (Prefer 'shotResult', fallback to score change)
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
    Requires: play_duration, possession_id.
    """
    # Base: 24 - Time elapsed in possession
    elapsed = df.groupby(['gameId', 'possession_id'])['play_duration'].cumsum()
    df['shot_clock_estimated'] = (24.0 - elapsed).clip(lower=0)
    
    # Correction: Offensive Rebound -> Reset to 14 max
    mask_off_reb = df['reboundOffensiveTotal'] > 0
    df.loc[mask_off_reb, 'shot_clock_estimated'] = 14.0
    
    return df

def process_lineups_logic(df):
    """
    Implements: Lineups, Last Sub Time.
    Pattern: Regex Parsing & Change Detection.
    """
    # 1. Map Players to Teams (Mode based)
    valid_players = df[df['personId'] > 0]
    player_map = valid_players.groupby('personId')['teamId'].agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else 0
    ).to_dict()
    
    # 2. Identify Home Team
    scoring = df[df['scoreHome'].diff() > 0]
    home_map = {}
    if not scoring.empty:
        home_map = scoring.groupby('gameId')['teamId'].agg(lambda x: x.mode().iloc[0]).to_dict()

    # 3. Parse Strings (Expensive operation - apply per row)
    def _parse(row):
        gid = row['gameId']
        raw = str(row['personIdsFilter'])
        if not raw or raw == '0': return [], []
        hid = home_map.get(gid)
        if not hid: return [], []
        
        ids = [int(x) for x in re.findall(r'\d+', raw)]
        return ([p for p in ids if player_map.get(p) == hid], 
                [p for p in ids if player_map.get(p) and player_map.get(p) != hid])

    lineups = df.apply(_parse, axis=1, result_type='expand')
    df['home_lineup'] = lineups[0]
    df['away_lineup'] = lineups[1]

    # 4. Last Sub Time
    # זיהוי שינוי במחרוזת ההרכב
    lineup_str = df['personIdsFilter'].astype(str)
    df['is_sub'] = (lineup_str != df.groupby('gameId')['personIdsFilter'].shift(1)).astype(int)
    
    # קיבוץ לפי "עידן הרכב" (Lineup Era)
    df['lineup_era'] = df.groupby('gameId')['is_sub'].cumsum()
    
    # הזמן בתחילת העידן פחות הזמן עכשיו = כמה זמן עבר ללא חילוף
    start_times = df.groupby(['gameId', 'lineup_era'])['seconds_remaining'].transform('max')
    df['time_since_last_sub'] = start_times - df['seconds_remaining']
    
    df.drop(columns=['is_sub', 'lineup_era'], inplace=True)
    return df

# --- Main Pipeline ---

def main():
    print(f"🚀 Starting Optimized Level 1 FE on: {os.path.basename(RAW_FILE_PATH)}")
    
    if not os.path.exists(RAW_FILE_PATH):
        print(f"❌ File not found: {RAW_FILE_PATH}"); return

    # Load
    df = pd.read_csv(RAW_FILE_PATH, low_memory=False)
    print(f"   Loaded {len(df)} rows.")

    # --- Pipeline Execution ---
    
    # Phase 1: Base State & Timeline
    print("   ⏱️  Phase 1: Timeline, Score & Base State...")
    df = process_base_timeline(df)
    
    # Phase 2: Counters & Inventory (Timeouts, Fouls)
    print("   📊 Phase 2: Inventory (Timeouts, Fouls) & Counters...")
    df = enrich_state_counters(df)

    # Phase 3: Temporal Calculation (Duration)
    print("   ⏳ Phase 3: Play Duration Calculation...")
    df = calculate_temporal_metrics(df)

    # Phase 4: Flow Logic (Possession)
    print("   🏀 Phase 4: Deterministic Possession Logic...")
    df = calculate_possession_flow(df)

    # Phase 5: Advanced Time (Shot Clock - depends on Possession)
    print("   ⏲️  Phase 5: Shot Clock Estimation (14s Logic)...")
    df = apply_shot_clock_logic(df)
    
    # Phase 6: Lineups (Complex Parsing)
    print("   👥 Phase 6: Lineups & Substitution Timer...")
    df = process_lineups_logic(df)

    # Save
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"✅ DONE. File saved: {OUTPUT_FILE}")
    print(f"   Final Shape: {df.shape}")
    print("   Features Implemented: Possession, Clock, Margins, EventCounters, TimeoutsRem, FoulCounts, Lineups, SubTimer.")

if __name__ == "__main__":
    main()