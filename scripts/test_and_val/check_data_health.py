import pandas as pd
import os

# --- Config ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROTATIONS_PATH = os.path.join(BASE_DIR, 'data', 'pureData', 'rotations_2024_25.csv')
RAW_PBP_PATH = os.path.join(BASE_DIR, 'data', 'pureData', 'season_2024_25.csv')

def check_health():
    print("🏥 Starting Data Health Check...")
    
    if not os.path.exists(ROTATIONS_PATH):
        print("❌ Rotations file not found."); return

    # 1. טעינת נתונים
    df_rot = pd.read_csv(ROTATIONS_PATH)
    unique_fetched = df_rot['gameId'].astype(str).str.zfill(10).unique()
    
    df_source = pd.read_csv(RAW_PBP_PATH, usecols=['gameId'])
    total_games = df_source['gameId'].astype(str).str.zfill(10).nunique()
    
    # 2. חישוב סטטיסטיקות
    success_rate = (len(unique_fetched) / total_games) * 100
    
    print(f"\n📊 Summary:")
    print(f"   Total Games in Season: {total_games}")
    print(f"   Successfully Fetched:  {len(unique_fetched)}")
    print(f"   Missing Games:         {total_games - len(unique_fetched)}")
    print(f"   ✅ Success Rate:       {success_rate:.1f}%")
    
    # 3. בדיקת איכות (האם יש גם בית וגם חוץ?)
    # בדיקה מדגמית: האם למשחקים יש נתונים לשני הצדדים?
    games_with_both_sides = 0
    grouped = df_rot.groupby('gameId')['team_side'].nunique()
    games_with_both_sides = (grouped == 2).sum()
    
    print(f"\n🔍 Quality Check:")
    print(f"   Games with BOTH Home/Away data: {games_with_both_sides}")
    print(f"   Games with Partial data:        {len(unique_fetched) - games_with_both_sides}")

    if success_rate > 85:
        print("\n✅ STATUS: HEALTHY (Ready for ML)")
    elif success_rate > 70:
        print("\n⚠️ STATUS: ACCEPTABLE (Might have some noise)")
    else:
        print("\n❌ STATUS: CRITICAL (Too much missing data)")

if __name__ == "__main__":
    check_health()