from nba_api.stats.endpoints import leaguedashplayerstats
import pandas as pd
import time

# הגדרות
TARGET_TEAMS = ['LAL', 'BOS', 'DEN', 'GSW'] # לייקרס, בוסטון, דנבר, גולדן סטייט
SEASON = '2024-25'

print(f"🔹 Fetching Advanced Stats for Season {SEASON}...")

try:
    # שליפת נתונים לכל הליגה (סוג מדד: Advanced בשביל USG%)
    stats = leaguedashplayerstats.LeagueDashPlayerStats(
        season=SEASON,
        measure_type_detailed_defense='Advanced' 
    )
    
    df = stats.get_data_frames()[0]
    
    # סינון: רק הקבוצות שבחרנו + שחקנים ששיחקו לפחות 30 משחקים (למנוע רעש)
    mask = (df['TEAM_ABBREVIATION'].isin(TARGET_TEAMS)) & (df['GP'] >= 30)
    df_filtered = df[mask].copy()

    # בחירת העמודות הרלוונטיות
    cols = ['TEAM_ABBREVIATION', 'PLAYER_NAME', 'USG_PCT', 'GP', 'MIN']
    df_clean = df_filtered[cols]

    # הדפסת התוצאות - טופ 3 שחקנים עם ה-Usage הכי גבוה בכל קבוצה
    print(f"\n📊 Top High Usage Players (Season {SEASON}):")
    
    for team in TARGET_TEAMS:
        print(f"\n--- {team} ---")
        top_players = df_clean[df_clean['TEAM_ABBREVIATION'] == team].sort_values(by='USG_PCT', ascending=False).head(3)
        print(top_players.to_string(index=False))

except Exception as e:
    print(f"❌ Error: {e}")