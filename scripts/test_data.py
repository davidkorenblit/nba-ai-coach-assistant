# NBA Timeout Data Fetcher - Using PlayByPlayV3
from nba_api.stats.endpoints import playbyplayv3, leaguegamefinder
import pandas as pd
import random
import time

print("Fetching games from 2023-24 season...")
# 1. Get all games from 2023-24 season
gamefinder = leaguegamefinder.LeagueGameFinder(season_nullable='2023-24')
games = gamefinder.get_data_frames()[0]

# 2. Pick 10 random game IDs
random_games = games.sample(10)['GAME_ID'].tolist()
print(f"Selected {len(random_games)} random games")

# 3. Fetch play-by-play for each game and filter timeouts
all_timeouts = []
for i, game_id in enumerate(random_games, 1):
    try:
        print(f"Fetching game {i}/10: {game_id}")
        pbp = playbyplayv3.PlayByPlayV3(game_id=game_id)
        df = pbp.get_data_frames()[0]
        
        # Filter for timeout events (EVENTMSGTYPE == 4)
        timeouts = df[df['actionType'] == 'Timeout']
        print(f"  Found {len(timeouts)} timeouts")
        
        all_timeouts.append(timeouts)
        time.sleep(1)  # Wait 1 second between requests
        
    except Exception as e:
        print(f"  Error: {e}")
        continue

# 4. Combine all timeouts and save
if all_timeouts:
    final_df = pd.concat(all_timeouts, ignore_index=True)
    output_path = 'C:/Users/david/finalPro/data/sample/timeouts_sample.csv'
    final_df.to_csv(output_path, index=False)
    print(f"\nSuccess! Saved {len(final_df)} timeouts to {output_path}")
else:
    print("\nNo timeouts found!")