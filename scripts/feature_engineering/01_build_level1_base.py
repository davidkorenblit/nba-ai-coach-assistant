import pandas as pd
import numpy as np
import os
import re

# --- Config & Settings ---
pd.set_option('future.no_silent_downcasting', True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_FILE_PATH = os.path.join(BASE_DIR, 'data', 'pureData', 'season_2024_25.csv')
ROTATIONS_FILE_PATH = os.path.join(BASE_DIR, 'data', 'pureData', 'rotations_2024_25.csv')
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

# --- Feature Modules (DO NOT TOUCH) ---

def process_base_timeline(df):
    if 'teamTricode' in df.columns:
        df['teamTricode'] = df['teamTricode'].astype(str).str.strip()
    df['seconds_remaining'] = df['clock'].apply(parse_clock)
    df.sort_values(by=['gameId', 'period', 'seconds_remaining', 'actionNumber'], 
                   ascending=[True, True, False, True], inplace=True)
    for col in ['scoreHome', 'scoreAway']:
        df[col] = df.groupby('gameId')[col].ffill().fillna(0)
    df['score_margin'] = df['scoreHome'] - df['scoreAway']
    zero_fill_cols = ['reboundDefensiveTotal', 'reboundOffensiveTotal', 'turnoverTotal', 'foulPersonalTotal', 'pointsTotal']
    for c in zero_fill_cols:
        if c in df.columns: df[c] = df[c].fillna(0)
    return df

def enrich_state_counters_v4(df):
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
    for side in ['home', 'away']:
        is_side_to = (df['timeout_role'] == side).astype(int)
        used = is_side_to.groupby(df['gameId']).cumsum()
        df[f'timeouts_remaining_{side}'] = (7 - used).clip(lower=0)
    df['is_foul'] = (df['foulPersonalTotal'] > 0).astype(int)
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

# --- REFACTORED HYBRID LINEUP LOGIC ---

def process_lineups_logic(df, df_rot):
    """
    Tier 1: Official Rotations (Fetch)
    Tier 2: PBP Filter (personIdsFilter)
    Tier 3: Inference Fallback (Forward Fill / Stability)
    """
    print("   🔧 Processing Hybrid Lineups (Fetch -> Filter -> Inference)...")
    
    valid_players = df[df['personId'] > 0]
    player_to_team = valid_players.groupby('personId')['teamId'].agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else 0
    ).to_dict()
    
    scoring = df[df['scoreHome'].diff() > 0]
    home_team_map = scoring.groupby('gameId')['teamId'].agg(lambda x: x.mode().iloc[0]).to_dict()

    # Prep elapsed time for rotation matching
    def get_elapsed(row):
        if row['period'] <= 4:
            return (row['period'] - 1) * 720 + (720 - row['seconds_remaining'])
        return 2880 + (row['period'] - 5) * 300 + (300 - row['seconds_remaining'])

    df['elapsed_sec'] = df.apply(get_elapsed, axis=1)
    
    rot_lookup = {}
    if df_rot is not None:
        df_rot['gameId_str'] = df_rot['gameId'].astype(str).str.zfill(10)
        for gid, g_grp in df_rot.groupby('gameId_str'):
            rot_lookup[gid] = {'Home': [], 'Away': []}
            for _, r in g_grp.iterrows():
                side = 'Home' if r['team_side'] == 'home' else 'Away'
                rot_lookup[gid][side].append((r['IN_TIME_REAL'], r['OUT_TIME_REAL'], int(r['PERSON_ID'])))

    def _get_hybrid_lineup(row):
        gid = str(row['gameId']).zfill(10)
        t_elapsed = row['elapsed_sec']
        hid = home_team_map.get(row['gameId'])
        
        # Tier 1: Fetch
        if gid in rot_lookup:
            h_cands = [p for (s, e, p) in rot_lookup[gid]['Home'] if s <= t_elapsed < e]
            a_cands = [p for (s, e, p) in rot_lookup[gid]['Away'] if s <= t_elapsed < e]
            if len(h_cands) == 5 and len(a_cands) == 5:
                return h_cands, a_cands, 1

        # Tier 2: PBP Filter
        raw_filter = str(row['personIdsFilter'])
        if raw_filter and raw_filter not in ['0', 'nan', 'None']:
            ids = [int(x) for x in re.findall(r'\d+', raw_filter)]
            if len(ids) >= 10:
                h_l = sorted([p for p in ids if player_to_team.get(p) == hid])
                a_l = sorted([p for p in ids if player_to_team.get(p) and player_to_team.get(p) != hid])
                if len(h_l) == 5 and len(a_l) == 5:
                    return h_l, a_l, 0
        
        # Tier 3: Inference (will be handled by ffill later)
        return None, None, 0

    results = df.apply(_get_hybrid_lineup, axis=1, result_type='expand')
    df['home_lineup'] = results[0]
    df['away_lineup'] = results[1]
    df['lineup_confidence'] = results[2]

    # --- FFILL & SUB DETECTION ---
    s_home = df['home_lineup'].apply(lambda x: str(x) if x is not None else "MISSING")
    s_away = df['away_lineup'].apply(lambda x: str(x) if x is not None else "MISSING")
    df['lineup_temp'] = s_home + "|" + s_away
    df['lineup_temp'] = df['lineup_temp'].replace(to_replace=r'.*MISSING.*', value=np.nan, regex=True)
    
    df['lineup_signature'] = df.groupby('gameId')['lineup_temp'].ffill()
    df['lineup_signature'] = df['lineup_signature'].fillna("START")

    df['is_new_period'] = (df['period'] != df.groupby('gameId')['period'].shift(1)).astype(int)
    shift_sig = df.groupby('gameId')['lineup_signature'].shift(1)
    
    df['is_sub'] = np.where(
        (df['lineup_signature'] != shift_sig) & (df['is_new_period'] == 0) & (shift_sig != "START"), 1, 0
    )
    
    df['lineup_era'] = df.groupby('gameId')['is_sub'].cumsum()
    grp = df.groupby(['gameId', 'period', 'lineup_era'])
    start_times = grp['seconds_remaining'].transform('max')
    df['time_since_last_sub'] = (start_times - df['seconds_remaining']).clip(lower=0)
    
    # Final Cleanup
    df.drop(columns=['lineup_temp', 'is_new_period', 'lineup_era', 'lineup_signature', 'is_sub', 'elapsed_sec'], inplace=True)
    return df

def clean_sparse_columns(df):
    cols_to_drop = ['assistPlayerNameInitial', 'assistPersonId', 'assistTotal',
                    'stealPlayerName', 'stealPersonId', 'blockPlayerName', 
                    'blockPersonId', 'timeout_role']
    existing_cols = [c for c in cols_to_drop if c in df.columns]
    if existing_cols:
        df.drop(columns=existing_cols, inplace=True)
    return df

# --- Main ---

def main():
    print(f"🚀 Starting Hybrid Level 1 FE (V7 - Recovery)...")
    
    if not os.path.exists(RAW_FILE_PATH):
        print(f"❌ PBP File not found: {RAW_FILE_PATH}"); return
    
    df_rot = None
    if os.path.exists(ROTATIONS_FILE_PATH):
        print(f"📂 Loading Official Rotations for Tier 1...")
        df_rot = pd.read_csv(ROTATIONS_FILE_PATH)
    else:
        print(f"⚠️ Warning: Rotations file missing. Skipping Tier 1.")

    df = pd.read_csv(RAW_FILE_PATH, low_memory=False)
    
    df = process_base_timeline(df)
    df = enrich_state_counters_v4(df)
    df = calculate_temporal_metrics(df)
    df = calculate_possession_flow(df)
    df = apply_shot_clock_logic(df)
    
    # THE HYBRID ENGINE
    df = process_lineups_logic(df, df_rot)
    
    df = clean_sparse_columns(df)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Level 1 DONE. Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()