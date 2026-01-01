import pandas as pd
import numpy as np
import os

# --- Config ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(BASE_DIR, '..', '..', 'data', 'interim', 'level1_base.csv')
# שיניתי את השם ל-_features כי זה מכיל כעת את הכל
OUTPUT_PATH = os.path.join(BASE_DIR, '..', '..', 'data', 'interim', 'level2_features.csv')

# TODO 1: רשימת כוכבים (לפי ID) לפיצ'ר מנוחת כוכב
# כרגע: Joel Embiid (203954), Devin Booker (1626164)
STAR_PLAYERS = [203954, 1626164]

def load_data():
    if not os.path.exists(INPUT_PATH): raise FileNotFoundError(f"Missing: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH, low_memory=False)
    # מיון חובה לפי זמן יורד לחישובי חלונות
    df.sort_values(by=['gameId', 'period', 'seconds_remaining'], ascending=[True, True, False], inplace=True)
    return df

# --- קבוצה 1: הקשר (Context) ---

def feature_style_shift(df):
    """Calculates rolling average of shot clock to detect game tempo."""
    print("🔹 Running: Style Shift...")
    WINDOW_SIZE = 15
    df['style_tempo_rolling'] = df.groupby('gameId')['shot_clock_estimated'].transform(
        lambda x: x.rolling(window=WINDOW_SIZE, min_periods=1).mean()
    ).fillna(14.0)
    return df

def feature_shared_fatigue(df):
    """Flags high fatigue based on time since last substitution."""
    print("🔹 Running: Shared Fatigue...")
    FATIGUE_THRESHOLD = 300
    df['is_high_fatigue'] = np.where(df['time_since_last_sub'] > FATIGUE_THRESHOLD, 1, 0)
    return df

# --- קבוצה 2: מומנטום ודינמיקה (Momentum) ---

def feature_smart_streak(df):
    """
    מחשב 'מומנטום חכם' (Smart Streak) בחלון זמן.
    משקלל: נקודות, שלשות (בונוס), הגנה (חטיפות/חסימות), וטעויות יריב.
    """
    print("🔹 Running: Smart Momentum Streak...")
    
    # 1. ניקוד לכל אירוע (Event Scoring)
    df['event_momentum_val'] = 0.0
    
    # בונוסים למומנטום
    df.loc[df['description'].str.contains('3pt Shot: Made', na=False), 'event_momentum_val'] += 1.5
    df.loc[df['description'].str.contains('2pt Shot: Made', na=False), 'event_momentum_val'] += 1.0
    df.loc[df['description'].str.contains('Steal', na=False), 'event_momentum_val'] += 2.0
    df.loc[df['description'].str.contains('Block', na=False), 'event_momentum_val'] += 1.5
    df.loc[df['actionType'] == 'technical', 'event_momentum_val'] += 2.5 # עבירה טכנית

    # 2. סכום מצטבר בחלון (Rolling Sum)
    # TODO 2: קבע את גודל החלון (כמות אירועים אחרונים)
    WINDOW_EVENTS = 10 
    
    df['momentum_streak_rolling'] = df.groupby('gameId')['event_momentum_val'].transform(
        lambda x: x.rolling(window=WINDOW_EVENTS, min_periods=1).sum()
    )
    return df

def feature_explosiveness(df):
    """
    מחשב את שיפוע שינוי ההפרש (כמה מהר התוצאה השתנתה).
    """
    print("🔹 Running: Explosiveness...")
    LOOKBACK = 20 # שורות אחורה להשוואה
    
    df['score_diff_lag'] = df.groupby('gameId')['scoreMargin'].shift(LOOKBACK)
    # השיפוע = ההפרש עכשיו פחות ההפרש לפני 20 מהלכים
    df['explosiveness_index'] = df['scoreMargin'] - df['score_diff_lag']
    df['explosiveness_index'] = df['explosiveness_index'].fillna(0)
    
    return df.drop(columns=['score_diff_lag'])

def feature_instability(df):
    """
    מדד אי-יציבות: צפיפות אירועים (כמה זמן עבר ב-10 האירועים האחרונים).
    """
    print("🔹 Running: Instability Index...")
    LAG_EVENTS = 10
    
    # מתי קרה האירוע לפני 10 תורות?
    df['time_lag'] = df.groupby('gameId')['seconds_remaining'].shift(LAG_EVENTS)
    
    # ההפרש בשניות. מספר נמוך = משחק מהיר מאוד (בלגן). מספר גבוה = משחק איטי.
    df['instability_index'] = df['time_lag'] - df['seconds_remaining']
    df['instability_index'] = df['instability_index'].fillna(60) # ברירת מחדל
    
    return df.drop(columns=['time_lag'])

def feature_star_resting(df):
    """
    בודק האם כוכב (מרשימה) נמצא כרגע על הספסל.
    """
    print("🔹 Running: Star Resting...")
    
    def check_star_on_bench(row):
        try:
            # חיבור המחרוזות של הליינאפים
            current_players = str(row['home_lineup']) + str(row['away_lineup'])
            # האם יש כוכב שחסר?
            for star_id in STAR_PLAYERS:
                if str(star_id) not in current_players:
                    return 1 # כוכב נח!
            return 0
        except:
            return 0

    df['is_star_resting'] = df.apply(check_star_on_bench, axis=1)
    return df

def feature_crunch_time(df):
    """זמן < 5 דקות והפרש < 5 נקודות."""
    print("🔹 Running: Crunch Time...")
    df['is_crunch_time'] = np.where(
        (df['seconds_remaining'] <= 300) & (df['scoreMargin'].abs() <= 5), 1, 0
    )
    return df

def main():
    print("🚀 Starting Level 2 (Full Feature Engineering)...")
    df = load_data()
    
    # הרצת כל הפיצ'רים (Pipeline)
    df = feature_style_shift(df)      # 1. Style
    df = feature_shared_fatigue(df)   # 2. Fatigue
    df = feature_smart_streak(df)     # 3. Smart Streak
    df = feature_explosiveness(df)    # 4. Explosiveness
    df = feature_instability(df)      # 5. Instability
    df = feature_star_resting(df)     # 6. Star Resting
    df = feature_crunch_time(df)      # 7. Crunch Time
    
    # שמירה
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Saved Full Level 2 to: {OUTPUT_PATH}")
    
    # הצגת דוגמה של הפיצ'רים החדשים
    new_cols = ['seconds_remaining', 'scoreMargin', 'momentum_streak_rolling', 
                'explosiveness_index', 'is_star_resting', 'is_crunch_time']
    print(df[new_cols].tail(10))

if __name__ == "__main__":
    main()