import pandas as pd
import numpy as np
import os
import sys
import ast

# --- Config ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(BASE_DIR, '..', '..', 'data', 'interim', 'level1_base.csv')
OUTPUT_PATH = os.path.join(BASE_DIR, '..', '..', 'data', 'interim', 'level2_features.csv')
LOOKUP_PATH = os.path.join(BASE_DIR, '..', '..', 'data', 'lookup', 'high_usage_players_2024-25.csv')

def load_data():
    if not os.path.exists(INPUT_PATH): 
        raise FileNotFoundError(f"Missing: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH, low_memory=False)
    
    # המרת רשימות חמישייה מטקסט לאובייקטים של פייתון (קריטי ללוגיקה החדשה)
    print("🔹 Converting lineups from strings to lists...")
    for col in ['home_lineup', 'away_lineup']:
        df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        
    # מיון חובה לפי זמן יורד
    df.sort_values(by=['gameId', 'period', 'seconds_remaining'], ascending=[True, True, False], inplace=True)
    return df

def get_star_ids():
    """Loads dynamic list of stars IDs from the lookup file."""
    if not os.path.exists(LOOKUP_PATH):
        print(f"⚠️ Warning: Lookup file not found at {LOOKUP_PATH}.")
        return []
    try:
        stars_df = pd.read_csv(LOOKUP_PATH)
        return stars_df['PLAYER_ID'].tolist() # שימוש ב-ID הוא מדויק יותר משמות
    except Exception as e:
        print(f"❌ Error loading star lookup: {e}")
        return []

# --- NEW: Advanced Features (Hybrid Architecture) ---

def feature_usage_gravity(df):
    """
    מחשב את ה'כובד' ההתקפי של החמישיות על המגרש.
    """
    if not os.path.exists(LOOKUP_PATH): return df
    print("🔹 Running: Usage Gravity (Lineup Threat)...")
    
    stars_df = pd.read_csv(LOOKUP_PATH)
    usage_map = dict(zip(stars_df['PLAYER_ID'], stars_df['USG_PCT']))
    
    def calc_gravity(lineup):
        if not isinstance(lineup, list): return 0.75 # ברירת מחדל ל-5 שחקנים (15% USG כל אחד)
        # שחקן שלא ברשימה מקבל usage ממוצע של 15% (0.15)
        return sum([usage_map.get(pid, 0.15) for pid in lineup])

    df['home_usage_gravity'] = df['home_lineup'].apply(calc_gravity)
    df['away_usage_gravity'] = df['away_lineup'].apply(calc_gravity)
    df['usage_delta'] = df['home_usage_gravity'] - df['away_usage_gravity']
    return df

def feature_accumulated_fatigue(df):
    """
    מחשב עייפות מצטברת לכל שחקן לאורך כל המשחק.
    """
    print("🔹 Running: Accumulated Fatigue Tracking...")
    
    def process_game(game_df):
        player_mins = {}
        h_fatigue = []
        a_fatigue = []
        
        for _, row in game_df.iterrows():
            dur = row.get('play_duration', 0)
            # צבירת זמן לכל שחקן בחמישיות
            for pid in row['home_lineup']:
                player_mins[pid] = player_mins.get(pid, 0) + dur
            for pid in row['away_lineup']:
                player_mins[pid] = player_mins.get(pid, 0) + dur
            
            # חישוב עייפות ממוצעת לחמישייה (סך הדקות ששיחקו עד כה)
            h_fatigue.append(sum([player_mins.get(p, 0) for p in row['home_lineup']]) / 5)
            a_fatigue.append(sum([player_mins.get(p, 0) for p in row['away_lineup']]) / 5)
            
        game_df['home_cum_fatigue'] = h_fatigue
        game_df['away_cum_fatigue'] = a_fatigue
        return game_df

    return df.groupby('gameId', group_keys=False).apply(process_game)

# --- Original Context Features (Preserved) ---

def feature_style_shift(df):
    print("🔹 Running: Style Shift...")
    WINDOW_SIZE = 15
    df['style_tempo_rolling'] = df.groupby('gameId')['shot_clock_estimated'].transform(
        lambda x: x.rolling(window=WINDOW_SIZE, min_periods=1).mean()
    ).fillna(14.0)
    return df

def feature_shared_fatigue(df):
    print("🔹 Running: Shared Fatigue (Calibrated)...")
    FATIGUE_THRESHOLD = 550
    df['is_high_fatigue'] = np.where(df['time_since_last_sub'] > FATIGUE_THRESHOLD, 1, 0)
    return df

# --- Original Momentum Features (Preserved) ---

def feature_smart_streak(df):
    print("🔹 Running: Smart Momentum Streak (Vectorized)...")
    df['event_momentum_val'] = 0.0
    
    df.loc[(df['actionType'] == '3pt') & (df['shotResult'] == 'Made'), 'event_momentum_val'] += 1.5
    df.loc[(df['actionType'] == '2pt') & (df['shotResult'] == 'Made'), 'event_momentum_val'] += 1.0
    df.loc[df['actionType'] == 'steal', 'event_momentum_val'] += 2.0
    df.loc[df['actionType'] == 'block', 'event_momentum_val'] += 1.5
    
    if 'foulTechnicalTotal' in df.columns:
        df.loc[df['foulTechnicalTotal'] > 0, 'event_momentum_val'] += 2.5
    
    WINDOW_EVENTS = 10
    df['momentum_streak_rolling'] = df.groupby('gameId')['event_momentum_val'].transform(
        lambda x: x.rolling(window=WINDOW_EVENTS, min_periods=1).sum()
    ).fillna(0)
    return df

def feature_explosiveness(df):
    print("🔹 Running: Explosiveness...")
    LOOKBACK = 20
    df['score_diff_lag'] = df.groupby('gameId')['score_margin'].shift(LOOKBACK)
    df['explosiveness_index'] = (df['score_margin'] - df['score_diff_lag']).fillna(0)
    return df.drop(columns=['score_diff_lag'])

def feature_instability(df):
    print("🔹 Running: Instability Index...")
    LAG_EVENTS = 10
    df['time_lag'] = df.groupby(['gameId', 'period'])['seconds_remaining'].shift(LAG_EVENTS)
    df['instability_index'] = (df['time_lag'] - df['seconds_remaining']).fillna(60)
    return df.drop(columns=['time_lag'])

def feature_star_resting(df):
    print("🔹 Running: Star Resting (Dynamic Lookup)...")
    star_ids = get_star_ids()
    if not star_ids:
        df['is_star_resting'] = 0
        return df
    
    # בדיקה האם לפחות כוכב אחד מהרשימה נמצא על המגרש
    def is_star_on(row):
        combined = set(row['home_lineup']) | set(row['away_lineup'])
        return 1 if any(sid in combined for sid in star_ids) else 0

    df['has_star_on_court'] = df.apply(is_star_on, axis=1)
    df['is_star_resting'] = 1 - df['has_star_on_court']
    return df.drop(columns=['has_star_on_court'])

def feature_clutch_time(df):
    print("🔹 Running: Clutch Time...")
    df['is_clutch_time'] = np.where(
        (df['seconds_remaining'] <= 300) & (df['score_margin'].abs() <= 5), 1, 0
    )
    return df

# --- Runner ---

def main():
    print("🚀 Starting Level 2 Feature Engineering (Hybrid Architecture)...")
    try:
        df = load_data()
        
        # 1. New Hybrid Features
        df = feature_usage_gravity(df)
        df = feature_accumulated_fatigue(df)
        
        # 2. Preserved Original Features
        df = feature_style_shift(df)
        df = feature_shared_fatigue(df)
        df = feature_smart_streak(df)
        df = feature_explosiveness(df)
        df = feature_instability(df)
        df = feature_star_resting(df)
        df = feature_clutch_time(df)
        
        df.to_csv(OUTPUT_PATH, index=False)
        print(f"✅ Saved Upgraded Level 2 to: {OUTPUT_PATH}")
        print(f"📊 Final Dataset Shape: {df.shape}")
        
    except Exception as e:
        print(f"❌ Critical Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()