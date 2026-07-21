import pandas as pd
import time
import os
from nba_api.stats.endpoints import leaguegamefinder
from datetime import datetime

# --- הגדרות דינמיות ---
def get_recent_nba_seasons(num_seasons=4):
    now = datetime.now()
    current_start_year = now.year - 1 if now.month < 10 else now.year
    seasons = []
    for i in range(num_seasons):
        start_yr = current_start_year - i
        end_yr_str = str(start_yr + 1)[-2:]
        seasons.append(f"{start_yr}-{end_yr_str}")
    return seasons

SEASONS_TO_FETCH = get_recent_nba_seasons(1)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, '..', 'data', 'pureData')


HEADERS = {
    'Host': 'stats.nba.com',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/117.0',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.5',
    'Referer': 'https://www.nba.com/',
    'Connection': 'keep-alive',
}

def fetch_multi_season_data():
    print(f"--- STARTING DATA COLLECTION FOR {len(SEASONS_TO_FETCH)} SEASONS ---")
    
    # 0. וידוא תיקייה
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # לולאה ראשית: רצים עונה אחרי עונה
    for season in SEASONS_TO_FETCH:
        print(f"\n=== Processing Season: {season} ===")
        season_actions = []
        
        # 1. שליפת רשימת המשחקים לאותה עונה
        print(f"Fetching game list for {season}...")
        try:
            gamefinder = leaguegamefinder.LeagueGameFinder(
                season_nullable=season, 
                league_id_nullable='00', 
                season_type_nullable='Regular Season',
                headers=HEADERS,
                timeout=60
            )
            games_df = gamefinder.get_data_frames()[0]
        except Exception as e:
            print(f"⚠️ Warning: Could not fetch season {season} due to API timeout: {e}")
            continue

        
        # סינון משחקים ששוחקו
        games_played = games_df[games_df['WL'].notna()]
        unique_game_ids = games_played['GAME_ID'].unique().tolist()
        total_games = len(unique_game_ids)
        
        print(f"Found {total_games} games.")

        # 2. לולאה על המשחקים
        for i, game_id in enumerate(unique_game_ids):
            try:
                # משיכה מה-Live Endpoint
                pbp = playbyplay.PlayByPlay(game_id=game_id)
                actions = pbp.actions.get_dict()
                
                if actions:
                    # משיכת פרטי המשחק לטובת המטא-דאטא
                    game_info = games_played[games_played['GAME_ID'] == game_id].iloc[0]
                    
                    for action in actions:
                        action['gameId'] = game_id
                        action['gameDate'] = game_info['GAME_DATE']
                        action['matchup'] = game_info['MATCHUP']
                        action['season'] = season # חשוב לזיהוי מאוחר
                    
                    season_actions.extend(actions)
                
                # הדפסה כל 50 משחקים כדי לא להציף את המסך
                if (i+1) % 50 == 0:
                    print(f"  Processed {i+1}/{total_games} games...")
                
                time.sleep(0.4) # השהייה קצרה
                
            except Exception as e:
                print(f"  FAILED Game {game_id}: {e}")
                time.sleep(1)

        # 3. שמירה בסוף כל עונה (קובץ נפרד!)
        if season_actions:
            safe_season_name = season.replace("-", "_") # 2021-22 -> 2021_22
            filename = f"season_{safe_season_name}.csv"
            full_path = os.path.join(OUTPUT_DIR, filename)
            
            print(f"Saving {len(season_actions)} rows to {filename}...")
            df = pd.DataFrame(season_actions)
            df.to_csv(full_path, index=False)
            print("Done saving.")
        else:
            print(f"No data collected for {season}.")

    print("\n--- ALL SEASONS COMPLETED ---")

if __name__ == "__main__":
    fetch_multi_season_data()