import os
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
TARGET_TEAMS = ['LAL', 'BOS', 'DEN', 'GSW']

def parse_args():
    parser = argparse.ArgumentParser(description="Check high usage players for a given season.")
    parser.add_argument("--season", type=str, default="2024_25", help="Season string (e.g. 2024_25, 2024-25, 2021_22)")
    parser.add_argument("--fetch-api", action="store_true", help="Force fetching from live NBA API instead of local lookup file")
    return parser.parse_args()

def check_usage(season: str = "2024_25", force_api: bool = False):
    season_underscore, season_dash = normalize_season(season)
    lookup_file = os.path.join(BASE_DIR, 'data', 'lookup', f'high_usage_players_{season_dash}.csv')

    df = None
    if not force_api and os.path.exists(lookup_file):
        print(f"📁 Loading High Usage Players from local lookup ({lookup_file})...")
        try:
            df = pd.read_csv(lookup_file)
        except Exception as e:
            print(f"⚠️ Error reading local lookup file: {e}")

    if df is None:
        print(f"🔹 Fetching Advanced Stats for Season {season_dash} from NBA API...")
        try:
            from nba_api.stats.endpoints import leaguedashplayerstats
            stats = leaguedashplayerstats.LeagueDashPlayerStats(
                season=season_dash,
                measure_type_detailed_defense='Advanced'
            )
            df = stats.get_data_frames()[0]
        except Exception as e:
            print(f"❌ Error fetching from API: {e}")
            return

    try:
        mask = (df['TEAM_ABBREVIATION'].isin(TARGET_TEAMS)) & (df['GP'] >= 30)
        df_filtered = df[mask].copy()

        # Select relevant columns
        cols = ['TEAM_ABBREVIATION', 'PLAYER_NAME', 'USG_PCT', 'GP', 'MIN']
        df_clean = df_filtered[[c for c in cols if c in df_filtered.columns]]

        # Print results - Top 3 high usage players per team
        print(f"\n📊 Top High Usage Players (Season {season_dash}):")
        
        for team in TARGET_TEAMS:
            print(f"\n--- {team} ---")
            top_players = df_clean[df_clean['TEAM_ABBREVIATION'] == team].sort_values(by='USG_PCT', ascending=False).head(3)
            print(top_players.to_string(index=False))

    except Exception as e:
        print(f"❌ Error processing usage data: {e}")

if __name__ == "__main__":
    args = parse_args()
    check_usage(season=args.season, force_api=args.fetch_api)
