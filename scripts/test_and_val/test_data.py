import os
import time
import argparse
import pandas as pd

def normalize_season(season_str: str):
    """
    Normalizes season input like '2024-25' or '2024_25' to both:
    (season_underscore, season_dash), e.g. ('2024_25', '2024-25').
    """
    s = season_str.strip().replace('-', '_')
    if '_' in s:
        parts = s.split('_')
        return f"{parts[0]}_{parts[1]}", f"{parts[0]}-{parts[1]}"
    return s, s.replace('_', '-')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def parse_args():
    parser = argparse.ArgumentParser(description="Test data availability for NBA seasons.")
    parser.add_argument("--season", type=str, default="2024_25", help="Season string (e.g. 2024_25, 2024-25, 2021_22)")
    parser.add_argument("--all", action="store_true", help="Check all historical seasons in list")
    return parser.parse_args()

def test_season_data(season: str = "2024_25", check_all: bool = False):
    from nba_api.stats.endpoints import leaguegamefinder
    from nba_api.live.nba.endpoints import playbyplay

    if check_all:
        seasons_to_check = ['2024-25', '2023-24', '2022-23', '2021-22', '2020-21', '2019-20', '2018-19', '2017-18']
    else:
        season_underscore, season_dash = normalize_season(season)
        seasons_to_check = [season_dash]
        local_pbp = os.path.join(BASE_DIR, 'data', 'pureData', f'season_{season_underscore}.csv')
        if os.path.exists(local_pbp):
            print(f"  [V] Local PBP file verified: {os.path.basename(local_pbp)}")

    print("--- Checking Historical Availability ---")

    for s_check in seasons_to_check:
        print(f"\nChecking Season {s_check}...")
        try:
            # 1. Find representative game
            gamefinder = leaguegamefinder.LeagueGameFinder(season_nullable=s_check, league_id_nullable='00')
            games = gamefinder.get_data_frames()[0]
            
            if games.empty:
                print(f"  No games found in archive for {s_check}. Skipping.")
                continue
                
            # Select test game
            game_id = games.iloc[0]['GAME_ID']
            
            # 2. Test Live Endpoint
            pbp = playbyplay.PlayByPlay(game_id=game_id)
            actions = pbp.actions.get_dict()
            
            if actions:
                print(f"  [V] SUCCESS: Live data available for {s_check}")
            else:
                print(f"  [X] FAILED: Live data is EMPTY for {s_check}")
                print("  --- STOPPING: Reached the limit of available history. ---")
                break
                
        except Exception as e:
            print(f"  [X] ERROR for {s_check}: {e}")
            print("  --- STOPPING: Reached the limit. ---")
            break
            
        time.sleep(1)

if __name__ == "__main__":
    args = parse_args()
    test_season_data(season=args.season, check_all=args.all)
