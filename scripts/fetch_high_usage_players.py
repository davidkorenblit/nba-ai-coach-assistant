import pandas as pd
import os
from nba_api.stats.endpoints import leaguedashplayerstats

# --- Config ---
SEASON = '2024-25'
MIN_GAMES_PLAYED = 45  # הורדתי ל-45 לביטחון, אפשר לשנות ל-50
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'lookup')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, f'high_usage_players_{SEASON}.csv')

def fetch_high_usage_players():
    print(f"🚀 Fetching High Usage Players for {SEASON} (Min GP: {MIN_GAMES_PLAYED})...")
    
    try:
        # 1. שליפת נתונים מתקדמים לכל הליגה
        stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season=SEASON,
            measure_type_detailed_defense='Advanced'
        )
        df = stats.get_data_frames()[0]

        # 2. סינון לפי כמות משחקים מינימלית
        df_filtered = df[df['GP'] >= MIN_GAMES_PLAYED].copy()

        if df_filtered.empty:
            print("⚠️ Warning: No players found with this filter. Try lowering MIN_GAMES_PLAYED.")
            return

        # 3. מציאת השחקן עם ה-USG הכי גבוה בכל קבוצה
        # ממיינים לפי קבוצה ו-USG יורד
        df_sorted = df_filtered.sort_values(by=['TEAM_ID', 'USG_PCT'], ascending=[True, False])
        
        # לוקחים את הראשון בכל קבוצה
        top_usage_per_team = df_sorted.drop_duplicates(subset=['TEAM_ID'], keep='first')

        # 4. בחירת עמודות רלוונטיות לשמירה
        cols_to_keep = ['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ABBREVIATION', 'USG_PCT', 'GP', 'MIN']
        final_df = top_usage_per_team[cols_to_keep]

        # 5. שמירה לקובץ
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        final_df.to_csv(OUTPUT_FILE, index=False)
        
        print(f"\n✅ Success! Saved {len(final_df)} star players to:")
        print(f"   {OUTPUT_FILE}")
        
        # הצגה למשתמש
        print("\n📊 Preview (Top 5 Highest Usage Stars):")
        print(final_df.sort_values('USG_PCT', ascending=False).head(5).to_string(index=False))

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    fetch_high_usage_players()