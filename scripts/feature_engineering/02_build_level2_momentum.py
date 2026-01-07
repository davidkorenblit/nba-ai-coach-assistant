import pandas as pd
import numpy as np
import os
import sys

# --- Config ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# חישוב נתיבים יחסיים לתיקיית הדאטה
INPUT_PATH = os.path.join(BASE_DIR, '..', '..', 'data', 'interim', 'level1_base.csv')
OUTPUT_PATH = os.path.join(BASE_DIR, '..', '..', 'data', 'interim', 'level2_features.csv')
LOOKUP_PATH = os.path.join(BASE_DIR, '..', '..', 'data', 'lookup', 'high_usage_players_2024-25.csv')

def load_data():
    if not os.path.exists(INPUT_PATH): 
        raise FileNotFoundError(f"Missing: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH, low_memory=False)
    # מיון חובה לפי זמן יורד
    df.sort_values(by=['gameId', 'period', 'seconds_remaining'], ascending=[True, True, False], inplace=True)
    return df

def get_star_ids():
    """Loads dynamic list of stars from the lookup file."""
    if not os.path.exists(LOOKUP_PATH):
        print(f"⚠️ Warning: Lookup file not found at {LOOKUP_PATH}. Star features will be empty.")
        return []
    try:
        stars_df = pd.read_csv(LOOKUP_PATH)
        # המרה למחרוזת לצורך חיפוש בטקסט הליינאפ
        return stars_df['PLAYER_ID'].astype(str).tolist()
    except Exception as e:
        print(f"❌ Error loading star lookup: {e}")
        return []

# --- Context Features ---

def feature_style_shift(df):
    print("🔹 Running: Style Shift...")
    WINDOW_SIZE = 15
    df['style_tempo_rolling'] = df.groupby('gameId')['shot_clock_estimated'].transform(
        lambda x: x.rolling(window=WINDOW_SIZE, min_periods=1).mean()
    ).fillna(14.0)
    return df

def feature_shared_fatigue(df):
    print("🔹 Running: Shared Fatigue (Calibrated)...")
    # UPDATED: סף עייפות מכויל ל-100 שניות
    FATIGUE_THRESHOLD = 80
    df['is_high_fatigue'] = np.where(df['time_since_last_sub'] > FATIGUE_THRESHOLD, 1, 0)
    return df

# --- Momentum Features ---

def feature_smart_streak(df):
    """
    FIXED: Uses 'actionType', 'shotResult', and 'foulTechnicalTotal'.
    """
    print("🔹 Running: Smart Momentum Streak (Vectorized)...")
    
    df['event_momentum_val'] = 0.0
    
    # שלשות
    mask_3pt = (df['actionType'] == '3pt') & (df['shotResult'] == 'Made')
    df.loc[mask_3pt, 'event_momentum_val'] += 1.5
    
    # סלים ל-2
    mask_2pt = (df['actionType'] == '2pt') & (df['shotResult'] == 'Made')
    df.loc[mask_2pt, 'event_momentum_val'] += 1.0
    
    # חטיפות
    df.loc[df['actionType'] == 'steal', 'event_momentum_val'] += 2.0
    
    # חסימות
    df.loc[df['actionType'] == 'block', 'event_momentum_val'] += 1.5
    
    # עבירות טכניות
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
    df['explosiveness_index'] = df['score_margin'] - df['score_diff_lag']
    df['explosiveness_index'] = df['explosiveness_index'].fillna(0)
    return df.drop(columns=['score_diff_lag'])

def feature_instability(df):
    print("🔹 Running: Instability Index (Quarter Fixed)...")
    LAG_EVENTS = 10
    
    # UPDATED: חישוב לפי רבע (Period) כדי למנוע קפיצות זמן לא הגיוניות
    df['time_lag'] = df.groupby(['gameId', 'period'])['seconds_remaining'].shift(LAG_EVENTS)
    
    df['instability_index'] = df['time_lag'] - df['seconds_remaining']
    df['instability_index'] = df['instability_index'].fillna(60)
    return df.drop(columns=['time_lag'])

def feature_star_resting(df):
    print("🔹 Running: Star Resting (Dynamic Lookup)...")
    
    star_ids = get_star_ids()
    
    if not star_ids:
        df['is_star_resting'] = 0
        return df

    # אופטימיזציה: יצירת Regex Pattern לחיפוש מהיר במקום לולאות
    # בודק האם *אחד* מה-IDs ברשימה קיים בליינאפ
    star_pattern = '|'.join(star_ids)
    
    # איחוד הליינאפים לחיפוש אחד
    combined_lineups = df['home_lineup'].astype(str) + " " + df['away_lineup'].astype(str)
    
    # אם יש התאמה -> הכוכב משחק (has_star=1). אנו רוצים לדעת מתי הוא נח (is_resting=1)
    has_star = combined_lineups.str.contains(star_pattern, regex=True, na=False).astype(int)
    df['is_star_resting'] = 1 - has_star
    
    return df

def feature_crunch_time(df):
    print("🔹 Running: Crunch Time...")
    df['is_crunch_time'] = np.where(
        (df['seconds_remaining'] <= 300) & (df['score_margin'].abs() <= 5), 1, 0
    )
    return df

def main():
    print("🚀 Starting Level 2 Feature Engineering (V3 - Dynamic & Calibrated)...")
    try:
        df = load_data()
        
        # Pipeline
        df = feature_style_shift(df)
        df = feature_shared_fatigue(df)
        df = feature_smart_streak(df)
        df = feature_explosiveness(df)
        df = feature_instability(df)
        df = feature_star_resting(df)
        df = feature_crunch_time(df)
        
        df.to_csv(OUTPUT_PATH, index=False)
        print(f"✅ Saved Upgraded Level 2 to: {OUTPUT_PATH}")
        print(f"📊 Features Shape: {df.shape}")
        
    except Exception as e:
        print(f"❌ Critical Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()