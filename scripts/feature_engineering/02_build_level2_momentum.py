import pandas as pd
import numpy as np
import os

# --- Config ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(BASE_DIR, '..', '..', 'data', 'interim', 'level1_base.csv')
OUTPUT_PATH = os.path.join(BASE_DIR, '..', '..', 'data', 'interim', 'level2_features.csv')

# רשימת כוכבים (Joel Embiid, Devin Booker)
STAR_PLAYERS = [203954, 1626164]

def load_data():
    if not os.path.exists(INPUT_PATH): raise FileNotFoundError(f"Missing: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH, low_memory=False)
    # מיון חובה לפי זמן יורד
    df.sort_values(by=['gameId', 'period', 'seconds_remaining'], ascending=[True, True, False], inplace=True)
    return df

# --- Context Features ---

def feature_style_shift(df):
    print("🔹 Running: Style Shift...")
    WINDOW_SIZE = 15
    # חישוב ממוצע נע של שעון הזריקות
    df['style_tempo_rolling'] = df.groupby('gameId')['shot_clock_estimated'].transform(
        lambda x: x.rolling(window=WINDOW_SIZE, min_periods=1).mean()
    ).fillna(14.0)
    return df

def feature_shared_fatigue(df):
    print("🔹 Running: Shared Fatigue...")
    FATIGUE_THRESHOLD = 300
    df['is_high_fatigue'] = np.where(df['time_since_last_sub'] > FATIGUE_THRESHOLD, 1, 0)
    return df

# --- Momentum Features (THE FIX) ---

def feature_smart_streak(df):
    """
    FIXED: Uses 'actionType', 'shotResult', and 'foulTechnicalTotal' 
    instead of string parsing description.
    """
    print("🔹 Running: Smart Momentum Streak (Fixed Logic)...")
    
    # 1. איפוס עמודת הניקוד
    df['event_momentum_val'] = 0.0
    
    # 2. לוגיקה וקטורית מדויקת על בסיס העמודות המובנות
    
    # שלשות (חובה שיהיה Made)
    mask_3pt = (df['actionType'] == '3pt') & (df['shotResult'] == 'Made')
    df.loc[mask_3pt, 'event_momentum_val'] += 1.5
    
    # סלים ל-2 (חובה שיהיה Made)
    mask_2pt = (df['actionType'] == '2pt') & (df['shotResult'] == 'Made')
    df.loc[mask_2pt, 'event_momentum_val'] += 1.0
    
    # חטיפות (אירוע הגנתי חזק)
    df.loc[df['actionType'] == 'steal', 'event_momentum_val'] += 2.0
    
    # חסימות
    df.loc[df['actionType'] == 'block', 'event_momentum_val'] += 1.5
    
    # עבירות טכניות (משתמשים בעמודת המונה הספציפית)
    if 'foulTechnicalTotal' in df.columns:
        df.loc[df['foulTechnicalTotal'] > 0, 'event_momentum_val'] += 2.5
    
    # 3. חישוב מצטבר בחלון (Rolling Sum)
    WINDOW_EVENTS = 10
    df['momentum_streak_rolling'] = df.groupby('gameId')['event_momentum_val'].transform(
        lambda x: x.rolling(window=WINDOW_EVENTS, min_periods=1).sum()
    ).fillna(0) # מילוי אפסים בהתחלה
    
    return df

def feature_explosiveness(df):
    print("🔹 Running: Explosiveness...")
    LOOKBACK = 20
    df['score_diff_lag'] = df.groupby('gameId')['score_margin'].shift(LOOKBACK)
    df['explosiveness_index'] = df['score_margin'] - df['score_diff_lag']
    df['explosiveness_index'] = df['explosiveness_index'].fillna(0)
    return df.drop(columns=['score_diff_lag'])

def feature_instability(df):
    print("🔹 Running: Instability Index...")
    LAG_EVENTS = 10
    df['time_lag'] = df.groupby('gameId')['seconds_remaining'].shift(LAG_EVENTS)
    df['instability_index'] = df['time_lag'] - df['seconds_remaining']
    df['instability_index'] = df['instability_index'].fillna(60)
    return df.drop(columns=['time_lag'])

def feature_star_resting(df):
    print("🔹 Running: Star Resting...")
    def check_star_on_bench(row):
        try:
            current_players = str(row['home_lineup']) + str(row['away_lineup'])
            for star_id in STAR_PLAYERS:
                if str(star_id) not in current_players:
                    return 1 
            return 0
        except:
            return 0
    df['is_star_resting'] = df.apply(check_star_on_bench, axis=1)
    return df

def feature_crunch_time(df):
    print("🔹 Running: Crunch Time...")
    df['is_crunch_time'] = np.where(
        (df['seconds_remaining'] <= 300) & (df['score_margin'].abs() <= 5), 1, 0
    )
    return df

def main():
    print("🚀 Starting Level 2 Feature Engineering (V2 Fixed)...")
    df = load_data()
    
    # Pipeline
    df = feature_style_shift(df)
    df = feature_shared_fatigue(df)
    df = feature_smart_streak(df)    # <-- המתוקן
    df = feature_explosiveness(df)
    df = feature_instability(df)
    df = feature_star_resting(df)
    df = feature_crunch_time(df)
    
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Saved Fixed Level 2 to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()