import pandas as pd
import os
import numpy as np

# --- הגדרות ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, '..', 'data', 'pureData')
OUTPUT_DIR = os.path.join(CURRENT_DIR, '..', 'docs', 'reports') # דוחות טקסטואליים
BAD_GAMES_FILE = os.path.join(OUTPUT_DIR, 'invalid_games_blacklist.csv')

# --- חוקי הסף (Thresholds) ---
MIN_TIMEOUTS = 7       # מינימום פסקי זמן למשחק תקין
MAX_TIMEOUTS = 30      # מקסימום (למנוע כפילויות משוגעות)
MAX_SCORE_JUMP = 4     # מקסימום נקודות שאפשר לקלוע במהלך אחד

def validate_game(df, game_id):
    """
    מקבל דאטא-פריים של משחק בודד ומחזיר רשימה של סיבות למה הוא פסול (אם בכלל)
    """
    reasons = []
    
    # 1. בדיקת שלמות רבעים (Periods)
    periods = df['period'].unique()
    if not {1, 2, 3, 4}.issubset(periods):
        reasons.append("Missing Quarters")

    # 2. בדיקת פסקי זמן (Timeouts)
    timeouts = df[df['actionType'] == 'timeout']
    timeout_count = len(timeouts)
    
    if timeout_count < MIN_TIMEOUTS:
        reasons.append(f"Too Few Timeouts ({timeout_count})")
    elif timeout_count > MAX_TIMEOUTS:
        reasons.append(f"Too Many Timeouts ({timeout_count})")
        
    # בדיקת שיוך קבוצה לפסק זמן
    orphaned_timeouts = timeouts['teamTricode'].isnull().sum()
    if orphaned_timeouts > 0:
        reasons.append(f"Orphaned Timeouts (Missing TeamID: {orphaned_timeouts})")

    # 3. בדיקת טריגרים למומנטום (Identity Check)
    # חטיפות ללא שם שחקן
    steals = df[df['actionType'] == 'steal']
    bad_steals = steals['playerName'].isnull().sum()
    if bad_steals > 0 and len(steals) > 0:
        # נבדוק אם זה רוב החטיפות או סתם אחת
        if (bad_steals / len(steals)) > 0.5: 
            reasons.append(f"Broken Steal Data ({bad_steals} missing names)")

    # חסימות ללא שם שחקן
    blocks = df[df['actionType'] == 'block']
    bad_blocks = blocks['playerName'].isnull().sum()
    if bad_blocks > 0 and len(blocks) > 0:
        if (bad_blocks / len(blocks)) > 0.5:
            reasons.append(f"Broken Block Data ({bad_blocks} missing names)")

    # 4. בדיקת היגיון ניקוד (Score Logic)
    # ממיינים לפי סדר כרונולוגי
    df = df.sort_values('orderNumber')
    
    # אם העמודות קיימות
    if 'scoreHome' in df.columns and 'scoreAway' in df.columns:
        # ממלאים ערכים חסרים בתוצאה עם הערך הקודם (Forward Fill) כי לא כל שורה מעדכנת תוצאה
        df['scoreHome'] = df['scoreHome'].ffill().fillna(0)
        df['scoreAway'] = df['scoreAway'].ffill().fillna(0)
        
        # חישוב סך הנקודות במשחק כרגע
        total_score = df['scoreHome'] + df['scoreAway']
        
        # בדיקת הפרש בין שורה לשורה
        score_diff = total_score.diff().fillna(0)
        
        # האם יש קפיצה של יותר מ-4 נקודות במהלך אחד?
        impossible_jumps = score_diff[score_diff > MAX_SCORE_JUMP]
        if not impossible_jumps.empty:
            reasons.append(f"Impossible Score Jumps (Found {len(impossible_jumps)} events > 4pts)")
            
    return reasons

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    all_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    blacklist_records = []
    
    total_games_checked = 0
    valid_games = 0

    print(f"--- STARTING DEEP LOGIC VALIDATION ---")
    print(f"Scanning {len(all_files)} season files...\n")

    for file in all_files:
        file_path = os.path.join(DATA_DIR, file)
        print(f"Processing {file}...")
        
        try:
            df = pd.read_csv(file_path, low_memory=False)
            unique_games = df['gameId'].unique()
            
            for game_id in unique_games:
                total_games_checked += 1
                game_df = df[df['gameId'] == game_id].copy()
                
                # הרצת הלוגיקה
                issues = validate_game(game_df, game_id)
                
                if issues:
                    # אם נמצאו בעיות, מוסיפים לרשימה השחורה
                    # נשמור גם את ה-Matchup אם קיים לזיהוי קל
                    matchup = game_df['matchup'].iloc[0] if 'matchup' in game_df.columns else "Unknown"
                    
                    for issue in issues:
                        blacklist_records.append({
                            'gameId': game_id,
                            'matchup': matchup,
                            'season': file, # מאיזה קובץ זה הגיע
                            'reason': issue
                        })
                else:
                    valid_games += 1
                    
        except Exception as e:
            print(f"Error reading {file}: {e}")

    # שמירת הדוח
    print(f"\n--- VALIDATION COMPLETE ---")
    print(f"Total Games Checked: {total_games_checked}")
    print(f"Valid Games: {valid_games}")
    print(f"Invalid Games: {len(set([x['gameId'] for x in blacklist_records]))}") # סופרים משחקים ייחודיים
    
    if blacklist_records:
        blacklist_df = pd.DataFrame(blacklist_records)
        blacklist_df.to_csv(BAD_GAMES_FILE, index=False)
        print(f"\n[X] Blacklist saved to: {BAD_GAMES_FILE}")
        print("Preview of bad games:")
        print(blacklist_df.head())
    else:
        print("\n[V] AMAZING! No logic errors found in any game.")

if __name__ == "__main__":
    main()