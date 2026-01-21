import pandas as pd
import os
import time
import sys
from nba_api.stats.endpoints import gamerotation

# --- Config ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PBP_PATH = os.path.join(BASE_DIR, 'data', 'pureData', 'season_2024_25.csv')
OUTPUT_PATH = os.path.join(BASE_DIR, 'data', 'pureData', 'rotations_2024_25.csv')

def fetch_rotations():
    print(f"🚀 Starting Rotation Fetcher...")
    
    # 1. טעינת רשימת המשחקים שיש לנו כבר
    if not os.path.exists(RAW_PBP_PATH):
        print(f"❌ Error: Source file not found at {RAW_PBP_PATH}")
        return

    print(f"📂 Reading Game IDs from existing PBP data...")
    try:
        # קוראים רק את עמודת ה-gameId כדי לחסוך זיכרון
        df_source = pd.read_csv(RAW_PBP_PATH, usecols=['gameId'])
        unique_games = df_source['gameId'].unique()
        print(f"🏀 Found {len(unique_games)} unique games to process.")
    except Exception as e:
        print(f"❌ Error reading source CSV: {e}")
        return

    all_rotations = []
    
    # 2. ריצה על המשחקים ומשיכת נתונים
    for i, gid in enumerate(unique_games):
        try:
            # המרת ID לפורמט של NBA API (מחרוזת של 10 ספרות)
            game_id_str = str(gid).zfill(10)
            
            print(f"   🔄 Fetching {game_id_str} ({i+1}/{len(unique_games)})...", end="\r")
            
            # קריאה ל-API
            rot = gamerotation.GameRotation(game_id=game_id_str)
            
            # עיבוד בית/חוץ
            df_home = rot.home_team_rotation.get_data_frame()
            df_away = rot.away_team_rotation.get_data_frame()
            
            if not df_home.empty:
                df_home['gameId'] = gid
                df_home['team_side'] = 'home'
                all_rotations.append(df_home)
                
            if not df_away.empty:
                df_away['gameId'] = gid
                df_away['team_side'] = 'away'
                all_rotations.append(df_away)
            
            # Pause to be nice to the API
            time.sleep(0.6)

        except Exception as e:
            print(f"\n   ⚠️ Error fetching {gid}: {e}")
            continue

    print("\n✅ Fetching complete. Saving data...")

    # 3. שמירה לקובץ מאוחד
    if all_rotations:
        final_df = pd.concat(all_rotations, ignore_index=True)
        
        # וידוא תיקייה
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        
        final_df.to_csv(OUTPUT_PATH, index=False)
        print(f"💾 Saved {len(final_df)} rotation rows to: {OUTPUT_PATH}")
    else:
        print("❌ No data was fetched.")

if __name__ == "__main__":
    fetch_rotations()