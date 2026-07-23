import pandas as pd
import time
import os
from datetime import datetime


# --- CRITICAL FIX FOR NBA API TIMEOUTS (Akamai WAF Bypass) ---
from curl_cffi import requests as curl_requests
from nba_api.library.http import NBAHTTP
from nba_api.live.nba.library.http import NBALiveHTTP

# Create a Chrome-impersonating session with NBA headers
session = curl_requests.Session(impersonate='chrome120')
session.headers.update({
    'Referer': 'https://www.nba.com/',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
})

# Optional Proxy support for cloud runners (e.g. GitHub Actions)
proxy = os.environ.get("NBA_API_PROXY")
if proxy:
    print(f"🌐 Using Proxy for NBA API: {proxy}")
    session.proxies = {"http": proxy, "https": proxy}

try:
    session.get('https://www.nba.com', timeout=15)
except Exception as e:
    print(f"Warning: Cookie warmup failed: {e}")


# Monkeypatch both stats and live HTTP sessions in nba_api
NBAHTTP.get_session = lambda self: session
NBALiveHTTP.get_session = lambda self: session

from nba_api.stats.endpoints import leaguegamefinder
from nba_api.live.nba.endpoints import playbyplay

# --- הגדרות דינמיות ---
def get_recent_nba_seasons(num_seasons=1):
    now = datetime.now()
    # In off-season (July-September), the last played season is 2024-25
    if now.month >= 10:
        current_start_year = now.year
    else:
        current_start_year = now.year - 1
        
    # Ensure during off-season we target the completed 2024-25 season
    if now.month in [7, 8, 9] and current_start_year == 2025:
        current_start_year = 2024

    seasons = []
    for i in range(num_seasons):
        start_yr = current_start_year - i
        end_yr_str = str(start_yr + 1)[-2:]
        seasons.append(f"{start_yr}-{end_yr_str}")
    return seasons


SEASONS_TO_FETCH = get_recent_nba_seasons(1)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, '..', 'data', 'pureData')

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
                season_type_nullable='Regular Season'
            )
            games_df = gamefinder.get_data_frames()[0]
        except Exception as e:
            print(f"⚠️ Warning: Could not fetch season {season}: {e}")
            continue



        
        # סינון משחקים ששוחקו
        games_played = games_df[games_df['WL'].notna()]
        unique_game_ids = games_played['GAME_ID'].unique().tolist()
        total_games = len(unique_game_ids)
        
        print(f"Found {total_games} games.")

        # 2. לולאה על המשחקים
        for i, game_id in enumerate(unique_game_ids):
            try:
                # משיכה מה-Live Endpoint המקורי
                pbp = playbyplay.PlayByPlay(game_id=game_id)
                actions = pbp.actions.get_dict()
                
                if actions:
                    game_info = games_played[games_played['GAME_ID'] == game_id].iloc[0]
                    for action in actions:
                        action['gameId'] = game_id
                        action['gameDate'] = game_info['GAME_DATE']
                        action['matchup'] = game_info['MATCHUP']
                        action['season'] = season
                    
                    season_actions.extend(actions)
                else:
                    print(f"  ⚠️ Warning: Game {game_id} returned 0 actions.")
            except Exception as e:
                print(f"  ❌ FAILED Game {game_id}: {e}")

            
            if (i + 1) % 50 == 0:
                print(f"  Processed {i+1}/{total_games} games | Collected {len(season_actions)} play-by-play actions so far...")
            time.sleep(0.2)



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