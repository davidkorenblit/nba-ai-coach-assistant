import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder, playbyplayv3
import time

# הגדרות תצוגה
pd.set_option('display.max_columns', None)

print("Fetching last 5 NBA games...")
# משיכת רשימת המשחקים הכי עדכניים
gamefinder = leaguegamefinder.LeagueGameFinder(season_nullable='2024-25', league_id_nullable='00', season_type_nullable='Regular Season')
games = gamefinder.get_data_frames()[0].head(5) # לוקחים 5 משחקים
game_ids = games['GAME_ID'].unique()

results = []

print(f"Testing {len(game_ids)} games across the league...\n")

for g_id in game_ids:
    # השהיה קטנה כדי לא להעמיס על ה-API
    time.sleep(1)
    
    # פרטי המשחק לתצוגה
    matchup = games[games['GAME_ID'] == g_id]['MATCHUP'].iloc[0]
    
    # משיכת ה-PBP
    try:
        df = playbyplayv3.PlayByPlayV3(game_id=g_id).get_data_frames()[0]
        
        # בדיקת הבאג: כמה שורות ריקות יש ב-actionType שיש בהן Steal/Block בתיאור?
        hidden_events = df[
            (df['actionType'] == '') & 
            (df['description'].str.contains('STEAL|BLOCK', case=False, regex=True, na=False))
        ]
        
        count = len(hidden_events)
        status = "BROKEN" if count > 0 else "OK"
        
        results.append({
            'GameID': g_id,
            'Matchup': matchup,
            'Hidden_Events_Found': count,
            'Status': status
        })
        print(f"Game {matchup}: {status} ({count} missing labels)")
        
    except Exception as e:
        print(f"Error fetching game {g_id}: {e}")

# סיכום
print("\n--- SUMMARY ---")
summary_df = pd.DataFrame(results)
print(summary_df)