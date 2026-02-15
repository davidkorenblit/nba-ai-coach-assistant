import pandas as pd
import time
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.live.nba.endpoints import playbyplay

# רשימת עונות לבדיקה (אחורה בזמן)
# כבר בדקנו את 23-24 וזה עבד, אז נתחיל מ-22
seasons_to_check = ['2022-23', '2021-22', '2020-21', '2019-20', '2018-19', '2017-18']

print("--- Checking Historical Availability ---")

for season in seasons_to_check:
    print(f"\nChecking Season {season}...")
    try:
        # 1. מציאת משחק מייצג
        gamefinder = leaguegamefinder.LeagueGameFinder(season_nullable=season, league_id_nullable='00')
        games = gamefinder.get_data_frames()[0]
        
        if games.empty:
            print(f"  No games found in archive for {season}. Skipping.")
            continue
            
        # לוקחים משחק אחד לבדיקה
        game_id = games.iloc[0]['GAME_ID']
        
        # 2. בדיקת Live Endpoint
        pbp = playbyplay.PlayByPlay(game_id=game_id)
        actions = pbp.actions.get_dict()
        
        if actions:
            print(f"  [V] SUCCESS: Live data available for {season}")
        else:
            print(f"  [X] FAILED: Live data is EMPTY for {season}")
            print("  --- STOPPING: Reached the limit of available history. ---")
            break # עוצרים כשמגיעים לגבול
            
    except Exception as e:
        print(f"  [X] ERROR for {season}: {e}")
        print("  --- STOPPING: Reached the limit. ---")
        break
        
    time.sleep(1) # נימוס