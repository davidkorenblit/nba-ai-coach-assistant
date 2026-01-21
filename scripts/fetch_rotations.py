import pandas as pd
import os
import time
import random
import concurrent.futures
from nba_api.stats.endpoints import gamerotation

# --- Config ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PBP_PATH = os.path.join(BASE_DIR, 'data', 'pureData', 'season_2024_25.csv')
OUTPUT_PATH = os.path.join(BASE_DIR, 'data', 'pureData', 'rotations_2024_25.csv')

MAX_WORKERS = 4      # מספר תהליכונים שמרני
SAVE_INTERVAL = 20   # כל כמה משחקים שומרים לקובץ

def get_existing_game_ids():
    """בודק איזה משחקים כבר שמרנו כדי לא להוריד שוב."""
    if not os.path.exists(OUTPUT_PATH):
        return set()
    try:
        # קוראים רק את עמודת gameId כדי לחסוך זיכרון
        df = pd.read_csv(OUTPUT_PATH, usecols=['gameId'], dtype={'gameId': str})
        return set(df['gameId'].unique())
    except:
        return set()

def fetch_single_game_rotation(game_id):
    """משיכת משחק בודד."""
    try:
        # השהייה אקראית (Jitter)
        time.sleep(random.uniform(0.5, 1.2))
        
        rot = gamerotation.GameRotation(game_id=game_id, timeout=10)
        frames = []
        
        # Home
        if hasattr(rot, 'home_team'):
            df = rot.home_team.get_data_frame()
            if not df.empty:
                df['gameId'] = game_id
                df['team_side'] = 'home'
                frames.append(df)
        
        # Away
        if hasattr(rot, 'away_team'):
            df = rot.away_team.get_data_frame()
            if not df.empty:
                df['gameId'] = game_id
                df['team_side'] = 'away'
                frames.append(df)
        
        return frames if frames else None

    except Exception:
        return None

def fetch_rotations_robust():
    print(f"🚀 Starting ROBUST Rotation Fetcher...")
    
    # 1. טעינת רשימת המשחקים
    if not os.path.exists(RAW_PBP_PATH):
        print("❌ Source file missing."); return

    df_source = pd.read_csv(RAW_PBP_PATH, usecols=['gameId'], low_memory=False)
    all_game_ids = df_source['gameId'].astype(str).str.zfill(10).unique()
    
    # 2. סינון משחקים שכבר נעשו
    existing_ids = get_existing_game_ids()
    games_to_process = [gid for gid in all_game_ids if gid not in existing_ids]
    
    print(f"📊 Total Games: {len(all_game_ids)}")
    print(f"✅ Already Done: {len(existing_ids)}")
    print(f"🔄 Remaining:   {len(games_to_process)}")
    
    if not games_to_process:
        print("🎉 Nothing to do! All games are fetched.")
        return

    # 3. הרצה במקביל
    batch_data = []
    completed_in_session = 0
    errors_in_session = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_game = {executor.submit(fetch_single_game_rotation, gid): gid for gid in games_to_process}
        
        for future in concurrent.futures.as_completed(future_to_game):
            game_id = future_to_game[future]
            completed_in_session += 1
            
            result = future.result()
            if result:
                batch_data.extend(result)
            else:
                errors_in_session += 1
            
            # הדפסת סטטוס
            print(f"   ⏳ Session Progress: {completed_in_session}/{len(games_to_process)} | Errors: {errors_in_session}", end="\r")
            
            # 4. שמירה אינקרמנטלית (Batch Save)
            if len(batch_data) > 0 and completed_in_session % SAVE_INTERVAL == 0:
                save_batch_to_csv(batch_data)
                batch_data = [] # ריקון הזיכרון

    # שמירת שאריות בסוף הריצה
    if batch_data:
        save_batch_to_csv(batch_data)

    print("\n✅ Session Complete.")

def save_batch_to_csv(data_frames):
    """שומר רשימת דאטה-פריימים לקובץ CSV (מצב Append)."""
    if not data_frames: return
    
    df_batch = pd.concat(data_frames, ignore_index=True)
    
    # סידור עמודות
    cols_order = ['gameId', 'team_side', 'PERSON_ID', 'IN_TIME_REAL', 'OUT_TIME_REAL', 'USG_PCT']
    existing = [c for c in cols_order if c in df_batch.columns]
    others = [c for c in df_batch.columns if c not in existing]
    df_batch = df_batch[existing + others]
    
    # האם הקובץ קיים? אם כן, לא כותבים כותרות (header=False)
    file_exists = os.path.exists(OUTPUT_PATH)
    
    df_batch.to_csv(OUTPUT_PATH, mode='a', header=not file_exists, index=False)
    # print(f" [Saved {len(df_batch)} rows] ", end="") # אופציונלי לדיבוג

if __name__ == "__main__":
    fetch_rotations_robust()