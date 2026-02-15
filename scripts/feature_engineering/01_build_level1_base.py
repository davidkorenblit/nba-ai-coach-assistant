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

# --- Helper Functions (DO NOT TOUCH) ---
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

# --- Feature Modules (Core Logic - UNTOUCHED) ---

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
    is_made_shot = (df.groupby('gameId')['scoreHome'].diff() + df.groupby('gameId')['scoreAway'].diff()) > 0
    df['is_poss_change'] = ((df['reboundDefensiveTotal'] > 0) | (df['turnoverTotal'] > 0) | is_made_shot).astype(int)
    df['possession_id'] = df.groupby('gameId')['is_poss_change'].cumsum()
    return df

def apply_shot_clock_logic(df):
    elapsed = df.groupby(['gameId', 'possession_id'])['play_duration'].cumsum()
    df['shot_clock_estimated'] = (24.0 - elapsed).clip(lower=0)
    df.loc[df['reboundOffensiveTotal'] > 0, 'shot_clock_estimated'] = 14.0
    return df

# --- DYNAMIC LINEUP ENGINE (THE ONLY MODIFIED FUNCTION) ---

def process_lineups_logic(df, df_rot):
    print("    🔍 Activating Dynamic State Tracker (Starters + Real-time Subs)...")
    
    # Pre-calculate maps
    home_team_map = df[df['scoreHome'].diff() > 0].groupby('gameId')['teamId'].agg(lambda x: x.mode().iloc[0]).to_dict()

    # Pre-process Rotations (Tier 1)
    def get_elapsed(row):
        if row['period'] <= 4: return (row['period'] - 1) * 720 + (720 - row['seconds_remaining'])
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

    # Starter Discovery Logic
    def get_starters(p_df, gid, hid):
        gid_str = str(gid).zfill(10)
        if gid_str in rot_lookup:
            t_start = p_df['elapsed_sec'].min()
            h_s = [p for (s, e, p) in rot_lookup[gid_str]['Home'] if s <= t_start < e]
            a_s = [p for (s, e, p) in rot_lookup[gid_str]['Away'] if s <= t_start < e]
            if len(h_s) == 5 and len(a_s) == 5: return set(h_s), set(a_s), 1
        
        # Fallback Inference
        h_s, a_s = set(), set()
        for _, r in p_df.iterrows():
            if pd.notna(r['personId']) and r['personId'] != 0:
                if r['teamId'] == hid: h_s.add(int(r['personId']))
                else: a_s.add(int(r['personId']))
            if 'SUB out' in str(r['description']):
                if r['teamId'] == hid: h_s.add(int(r['personId']))
                else: a_s.add(int(r['personId']))
            if len(h_s) >= 5 and len(a_s) >= 5: break
        return set(list(h_s)[:5]), set(list(a_s)[:5]), 0

    # Main Row-by-Row Tracking
    final_dfs = []
    for gid, g_df in df.groupby('gameId'):
        hid = home_team_map.get(gid)
        for period, p_df in g_df.groupby('period'):
            curr_h, curr_a, conf = get_starters(p_df, gid, hid)
            h_list, a_list = [], []
            
            for _, row in p_df.iterrows():
                desc = str(row['description'])
                pid, tid = row['personId'], row['teamId']
                
                if 'SUB out' in desc and pd.notna(pid):
                    if tid == hid: curr_h.discard(int(pid))
                    else: curr_a.discard(int(pid))
                elif 'SUB in' in desc and pd.notna(pid):
                    if tid == hid: curr_h.add(int(pid))
                    else: curr_a.add(int(pid))
                
                h_list.append(sorted(list(curr_h))[:5])
                a_list.append(sorted(list(curr_a))[:5])
            
            p_df = p_df.assign(home_lineup=h_list, away_lineup=a_list, lineup_confidence=conf)
            final_dfs.append(p_df)

    df = pd.concat(final_dfs)

    # Re-calculate Sub Timer
    df['lineup_temp'] = df['home_lineup'].astype(str) + "|" + df['away_lineup'].astype(str)
    df['lineup_signature'] = df.groupby('gameId')['lineup_temp'].ffill()
    df['is_new_period'] = (df['period'] != df.groupby('gameId')['period'].shift(1)).astype(int)
    shift_sig = df.groupby('gameId')['lineup_signature'].shift(1)
    
    df['is_sub'] = np.where((df['lineup_signature'] != shift_sig) & (df['is_new_period'] == 0) & (shift_sig.notna()), 1, 0)
    df['lineup_era'] = df.groupby('gameId')['is_sub'].cumsum()
    df['time_since_last_sub'] = df.groupby(['gameId', 'period', 'lineup_era'])['seconds_remaining'].transform('max') - df['seconds_remaining']
    
    df.drop(columns=['lineup_temp', 'is_new_period', 'lineup_era', 'lineup_signature', 'is_sub', 'elapsed_sec'], inplace=True)
    return df

def clean_sparse_columns(df):
    cols_to_drop = ['assistPlayerNameInitial', 'assistPersonId', 'assistTotal', 'stealPlayerName', 'stealPersonId', 'blockPlayerName', 'blockPersonId', 'timeout_role']
    existing = [c for c in cols_to_drop if c in df.columns]
    if existing: df.drop(columns=existing, inplace=True)
    return df

# --- Main (UNTOUCHED Pipeline) ---
def main():
    print(f"🚀 Starting DYNAMIC Level 1 Build (V9)...")
    if not os.path.exists(RAW_FILE_PATH): return
    df_rot = pd.read_csv(ROTATIONS_FILE_PATH) if os.path.exists(ROTATIONS_FILE_PATH) else None
    df = pd.read_csv(RAW_FILE_PATH, low_memory=False)
    
    df = process_base_timeline(df)
    df = enrich_state_counters_v4(df)
    df = calculate_temporal_metrics(df)
    df = calculate_possession_flow(df)
    df = apply_shot_clock_logic(df)
    df = process_lineups_logic(df, df_rot)
    df = clean_sparse_columns(df)
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Level 1 DONE. Dynamic Substitutions Captured.")

if __name__ == "__main__":
    main()