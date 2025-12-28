import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(BASE_DIR, '..', '..', 'data', 'interim', 'level1_base.csv')
OUTPUT_PATH = os.path.join(BASE_DIR, '..', '..', 'data', 'interim', 'level2_context.csv')

def load_data():
    if not os.path.exists(INPUT_PATH): raise FileNotFoundError(f"Missing: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH, low_memory=False)
    # Sort specifically for rolling calculations
    df.sort_values(by=['gameId', 'period', 'seconds_remaining'], ascending=[True, True, False], inplace=True)
    return df

def feature_style_shift(df):
    """Calculates rolling average of shot clock to detect game tempo."""
    
    WINDOW_SIZE = 15


    # Rolling mean per game
    df['style_tempo_rolling'] = df.groupby('gameId')['shot_clock_estimated'].transform(
        lambda x: x.rolling(window=WINDOW_SIZE, min_periods=1).mean()
    ).fillna(14.0)
    
    return df

def feature_shared_fatigue(df):
    """Flags high fatigue based on time since last substitution."""
    
    FATIGUE_THRESHOLD = 300
    # -----------------------------------------------

    if FATIGUE_THRESHOLD == 0: raise ValueError("⚠️ Set FATIGUE_THRESHOLD in feature_shared_fatigue first.")

    df['is_high_fatigue'] = np.where(df['time_since_last_sub'] > FATIGUE_THRESHOLD, 1, 0)
    return df

def main():
    print("🚀 Starting Level 2 (Context)...")
    df = load_data()
    
    df = feature_style_shift(df)
    df = feature_shared_fatigue(df)
    
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Saved: {OUTPUT_PATH}")
    print(df[['gameId', 'style_tempo_rolling', 'time_since_last_sub', 'is_high_fatigue']].head())

if __name__ == "__main__":
    main()