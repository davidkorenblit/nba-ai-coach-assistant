import pandas as pd
import os
import time
import random
from nba_api.stats.endpoints import gamerotation

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MISSING_IDS_PATH = os.path.join(BASE_DIR, 'data', 'pureData', 'missing_ids.csv')
OUTPUT_PATH = os.path.join(BASE_DIR, 'data', 'pureData', 'rotations_2024_25.csv')

def fill_smart():
    print("🚑 Starting SMART Rescue...")
    if not os.path.exists(MISSING_IDS_PATH): return
    
    missing_ids = pd.read_csv(MISSING_IDS_PATH, header=None, dtype=str)[0].tolist()
    i = 0
    while i < len(missing_ids):
        gid = missing_ids[i]
        try:
            print(f"   💉 Fixing {gid} [{i+1}/{len(missing_ids)}]...", end="\r")
            time.sleep(random.uniform(2, 4)) # הפסקה בין בקשות
            
            rot = gamerotation.GameRotation(game_id=gid, timeout=30)
            frames = []
            if hasattr(rot, 'home_team'): frames.append(rot.home_team.get_data_frame())
            if hasattr(rot, 'away_team'): frames.append(rot.away_team.get_data_frame())
            
            # אם חזר מידע - שומרים
            if frames:
                df = pd.concat(frames, ignore_index=True)
                df['gameId'] = gid
                # סידור מהיר של עמודות כדי למנוע קריסה
                cols = ['gameId', 'PERSON_ID', 'IN_TIME_REAL', 'OUT_TIME_REAL']
                exist = [c for c in cols if c in df.columns]
                other = [c for c in df.columns if c not in exist]
                df[exist + other].to_csv(OUTPUT_PATH, mode='a', header=False, index=False)
            
            i += 1 # הצלחה -> ממשיכים
            
        except Exception as e:
            print(f"\n   🛑 Blocked on {gid}. Sleeping 3 mins...")
            time.sleep(180) # ישנים 3 דקות ומנסים שוב את אותו משחק

if __name__ == "__main__":
    fill_smart()