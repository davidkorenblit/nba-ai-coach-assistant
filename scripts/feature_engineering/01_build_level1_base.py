import pandas as pd
import numpy as np
import os
import re

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

def calculate_play_duration(df):
    # חישוב משך כל פעולה (בדקות) על בסיס שינוי ב-seconds_remaining
    df=df.sort_values(by=['gameId','period', 'seconds_remaining'], ascending=[True, True, False])
    # חישוב הפרש בין השורות
    df['prev_seconds'] = df.groupby(['gameId', 'period'])['seconds_remaining'].shift(1)
    df['play_duration'] = df['prev_seconds'] - df['seconds_remaining']
    df['play_duration'] = df['play_duration'].fillna(0).clip(lower=0)
    df.drop(columns=['prev_seconds'], inplace=True)

    return df

def build_player_team_map(df):
    """ מיפוי מזהה שחקן -> מזהה קבוצה (הפונקציה שהייתה חסרה) """
    # אנו מניחים ששחקן משחק ברוב המקרים באותה קבוצה באותה עונה
    # במקרה של טרייד, זה יקח את ה-Mode (הקבוצה בה שיחק הכי הרבה מהלכים)
    # לדיוק מקסימלי, עדיף למפות פר משחק, אבל לרוב זה מספיק טוב
    valid_players = df[df['personId'] > 0]
    return valid_players.groupby('personId')['teamId'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else 0).to_dict()



def map_home_away_teams(df):
    scoring_plays = df[df['scoreHome'].diff()>0]
    if scoring_plays.empty:
        return {}
    home_teams_map = scoring_plays.groupby('gameId')['teamId'].agg(lambda x: x.mode().iloc[0]).to_dict()

    return home_teams_map



def parse_lineups(df, player_map, home_teams_map):
    """
    מפרק את 'personIdsFilter' לשתי עמודות: home_lineup, away_lineup.
    דורש player_map (שנבנה קודם) ו-home_teams_map.
    """
    
    def _parse_row(row):
        game_id = row['gameId']
        raw_str = str(row['personIdsFilter'])
        
        # אם אין נתונים או שזה 0
        if not raw_str or raw_str == '0':
            return [], []

        home_id = home_teams_map.get(game_id)
        if not home_id: return [], [] # מקרה קצה

        # חילוץ כל המספרים מהמחרוזת
        all_ids = [int(x) for x in re.findall(r'\d+', raw_str)]
        
        home_players = []
        away_players = []
        
        for pid in all_ids:
            tid = player_map.get(pid)
            if tid == home_id:
                home_players.append(pid)
            elif tid: # אם השחקן מוכר אבל לא מהבית, הוא בחוץ
                away_players.append(pid)
                
        return home_players, away_players

    # החלת הלוגיקה (זה יקח זמן)
    # שימוש ב-zip כדי להחזיר שתי עמודות
    lineups = df.apply(_parse_row, axis=1, result_type='expand')
    df['home_lineup'] = lineups[0]
    df['away_lineup'] = lineups[1]
    
    return df


def calculate_possession(df):
    """
    קובע מזהה פוזשן (Possession ID) רץ.
    מחליף פוזשן כאשר:
    1. יש ריבאונד הגנה.
    2. יש איבוד כדור.
    3. נקלע סל שדה (2pt/3pt) - מזוהה לפי שינוי בניקוד.
    4. נקלע סל עונשין אחרון (אופציונלי, כאן נתמקד בעיקר, אפשר להוסיף 1of1 וכו').
    """
    # וודא שהמיון נכון לפני חישוב הפרשים
    df = df.sort_values(by=['gameId', 'period', 'seconds_remaining'], ascending=[True, True, False])

    # 1. זיהוי שינוי ניקוד (האם היה סל בשורה הזו?)
    # משווים לשורה הקודמת באותו משחק
    df['score_total'] = df['scoreHome'] + df['scoreAway']
    df['score_diff'] = df.groupby('gameId')['score_total'].diff().fillna(0)
    is_score_change = df['score_diff'] > 0

    # 2. הגדרת התנאים להחלפת פוזשן
    # א. ריבאונד הגנה
    is_def_reb = (df['actionType'] == 'rebound') & (df['subType'] == 'defensive')
    
    # ב. איבוד כדור
    is_turnover = df['actionType'] == 'turnover'
    
    # ג. סל שדה שנכנס (זריקה + שינוי ניקוד)
    # לפי התמונה: actionType הוא '2pt' או '3pt'
    is_fg_made = df['actionType'].isin(['2pt', '3pt']) & is_score_change

    # ד. זריקת עונשין אחרונה שנכנסה (למשל 2 of 2) - משנה פוזשן
    # נזהה לפי הטקסט ב-subType ושינוי ניקוד
    is_last_ft_made = (
        (df['actionType'] == 'freethrow') & 
        (df['subType'].isin(['1 of 1', '2 of 2', '3 of 3'])) & 
        is_score_change
    )

    # 3. איחוד כל הטריגרים
    df['is_poss_change'] = (is_def_reb | is_turnover | is_fg_made | is_last_ft_made).astype(int)

    # 4. יצירת ID רץ (Cumulative Sum)
    df['possession_id'] = df.groupby(['gameId'])['is_poss_change'].cumsum()
    
    # ניקוי עמודות עזר
    df.drop(columns=['score_total', 'score_diff'], inplace=True)
    
    return df


def estimate_shot_clock(df):
    """
    מחשב כמה זמן נשאר לזרוק (24 שניות פחות הזמן שעבר בפוזשן).
    """
    # מחשבים זמן מצטבר בתוך כל פוזשן
    df['time_elapsed_in_poss'] = df.groupby(['gameId', 'possession_id'])['play_duration'].cumsum()
    
    # שעון זריקות = 24 פחות מה שעבר
    df['shot_clock_estimated'] = 24.0 - df['time_elapsed_in_poss']
    
    # תיקון: אם היה ריבאונד התקפה, זה מתאפס ל-14 (דורש לוגיקה נוספת)
    # תיקון: לא יכול להיות שלילי
    df['shot_clock_estimated'] = df['shot_clock_estimated'].clip(lower=0)
    
    return df





def main():
    print(f"🚀 Starting Level 1 FE (Full Enrichment) on: {os.path.basename(RAW_FILE_PATH)}")
    
    if not os.path.exists(RAW_FILE_PATH):
        print(f"❌ File not found: {RAW_FILE_PATH}")
        return

    # 1. טעינה
    try:
        df = pd.read_csv(RAW_FILE_PATH, low_memory=False)
        print(f"   Loaded {len(df)} rows.")
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return

    # 2. עיבוד בסיסי (נרמול זמנים, מילוי תוצאה, זיהוי סוגי Timeout)
    print("   🔨 Step 1: Basic Processing (Time, Scores, Timeouts)...")
    df_processed = df.groupby('gameId', group_keys=False).apply(process_single_game)
    
    # 3. חישוב משך מהלך (חייב להיות לפני שעון זריקות)
    print("   ⏱️ Step 2: Calculating Play Duration...")
    df_processed = calculate_play_duration(df_processed)

    # 4. פיצוח הרכבים וזיהוי בית/חוץ
    print("   👥 Step 3: Parsing Lineups & Homeliness (This might take a moment)...")
    # בניית מפות עזר
    player_map = build_player_team_map(df_processed) # פונקציית עזר שהגדרנו קודם
    home_teams_map = map_home_away_teams(df_processed)
    # הרצת הפיענוח
    df_processed = parse_lineups(df_processed, player_map, home_teams_map)

    # 5. לוגיקת פוזשן (תלויה בזיהוי סלים ואיבודים)
    print("   🏀 Step 4: Calculating Possession Logic...")
    df_processed = calculate_possession(df_processed)

    # 6. שעון זריקות משוער (תלוי בפוזשן ובמשך מהלך)
    print("   ⏳ Step 5: Estimating Shot Clock...")
    df_processed = estimate_shot_clock(df_processed)

    # 7. שמירה
    # הערה: אנחנו שומרים את כל העמודות החדשות, לכן לא נסנן בקשיחות עם COLS_TO_KEEP הישן
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_processed.to_csv(OUTPUT_FILE, index=False)
    
    print(f"✅ DONE. Full Level 1 Dataset saved to: {OUTPUT_FILE}")
    print(f"   Final Shape: {df_processed.shape}")
    print(f"   New Features: {['play_duration', 'possession_id', 'shot_clock_estimated', 'home_lineup']}")

if __name__ == "__main__":
    main()