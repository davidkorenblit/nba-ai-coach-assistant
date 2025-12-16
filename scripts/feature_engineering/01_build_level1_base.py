import pandas as pd
import numpy as np
import os

# --- הגדרות נתיבים (ממוקד לעונת 2024/25 בלבד) ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# מניחים שהסקריפט נמצא ב scripts/feature_engineering
RAW_FILE_PATH = os.path.join(CURRENT_DIR, '..', '..', 'data', 'pureData', 'season_2024_25.csv')
OUTPUT_DIR = os.path.join(CURRENT_DIR, '..', '..', 'data', 'interim')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'level1_base.csv')

# --- עמודות שנשמור (הסינון) ---
COLS_TO_KEEP = [
    'gameId', 'period', 'clock', 'seconds_remaining', 'actionNumber',
    'actionType', 'subType', 'description',
    'playerName', 'personId', 'teamId',
    'scoreHome', 'scoreAway', 'score_margin',
    'is_timeout', 'timeout_type',
    'foulPersonalTotal', 'pointsTotal', 'turnoverTotal',
    'personIdsFilter', 'x', 'y'
]

def parse_clock(clock_str):
    """ המרת שעון (PT12M00.00S או 12:00) לשניות (float) """
    if pd.isna(clock_str): return 0.0
    s = str(clock_str).strip()
    try:
        if 'M' in s: # Format: PT12M00.00S
            mins = float(s.split('M')[0].replace('PT', ''))
            secs = float(s.split('M')[1].replace('S', ''))
            return mins * 60 + secs
        elif ':' in s: # Format: 12:00
            parts = s.split(':')
            return float(parts[0]) * 60 + float(parts[1])
        return float(s)
    except:
        return 0.0

def extract_timeout_info(row):
    """ חילוץ שם הקבוצה שלקחה את הטיים-אאוט מתוך התיאור """
    desc = str(row['description'])
    action_type = row['actionType']
    
    # בדיקה אם זה טיים אאוט
    if action_type == 9 or 'Timeout' in desc:
        # התיאור הוא לרוב "GSW Timeout" או "IND Timeout"
        # ננסה לחלץ את המילה הראשונה שהיא שם הקבוצה
        parts = desc.split()
        if parts:
            return parts[0] # מחזיר: 'GSW', 'LAL', 'BKN' וכו'
        return 'General' # גיבוי למקרה מוזר
        
    return 'None'

def process_single_game(df_game):
    """ לוגיקה ברמת משחק בודד """
    
    # 1. המרת זמן
    df_game['seconds_remaining'] = df_game['clock'].apply(parse_clock)
    
    # 2. מיון כרונולוגי חובה (רבע -> זמן יורד -> מספר פעולה)
    df_game.sort_values(by=['period', 'seconds_remaining', 'actionNumber'], 
                        ascending=[True, False, True], inplace=True)
    
    # 3. מילוי תוצאה (Forward Fill) - כדי שלא יהיו חורים בגרף התוצאה
    df_game['scoreHome'] = df_game['scoreHome'].ffill().fillna(0)
    df_game['scoreAway'] = df_game['scoreAway'].ffill().fillna(0)
    
    # חישוב הפרש עדכני
    df_game['score_margin'] = df_game['scoreHome'] - df_game['scoreAway']
    
    # 4. חילוץ טיים-אאוטים
    df_game['timeout_type'] = df_game.apply(extract_timeout_info, axis=1)
    df_game['is_timeout'] = (df_game['timeout_type'] != 'None').astype(int)

    # 5. מילוי אפסים בנתוני שחקן אישיים (כדי למנוע NaN)
    # כאן אנחנו לא עושים Forward Fill כי זה מידע נקודתי לאירוע
    cols_to_zero = ['foulPersonalTotal', 'pointsTotal', 'turnoverTotal']
    for col in cols_to_zero:
        if col in df_game.columns:
            df_game[col] = df_game[col].fillna(0)
            
    return df_game

def main():
    print(f"🚀 Starting Level 1 FE on: {os.path.basename(RAW_FILE_PATH)}")
    
    if not os.path.exists(RAW_FILE_PATH):
        print(f"❌ File not found: {RAW_FILE_PATH}")
        return

    # טעינת הקובץ הבודד
    try:
        df = pd.read_csv(RAW_FILE_PATH, low_memory=False)
        print(f"   Loaded {len(df)} rows.")
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return

    # עיבוד (Group by GameId ליתר ביטחון, למקרה שיש כמה משחקים בקובץ העונתי)
    print("   Processing game logic (Time, Timeouts, Filling)...")
    df_processed = df.groupby('gameId', group_keys=False).apply(process_single_game)
    
    # סינון עמודות
    # מוודאים שכל העמודות שאנחנו רוצים קיימות (למניעת שגיאות אם משהו חסר במקור)
    available_cols = [c for c in COLS_TO_KEEP if c in df_processed.columns]
    df_final = df_processed[available_cols]

    # שמירה
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_final.to_csv(OUTPUT_FILE, index=False)
    
    print(f"✅ DONE. Saved to: {OUTPUT_FILE}")
    print(f"   Shape: {df_final.shape}")
    print(f"   New Columns Example: {['seconds_remaining', 'is_timeout', 'timeout_type']}")

if __name__ == "__main__":
    main()