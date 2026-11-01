import pandas as pd
import os
import argparse

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

# --- Config ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_health(season: str = "2024_25"):
    season_underscore, _ = normalize_season(season)
    rotations_path = os.path.join(BASE_DIR, 'data', 'pureData', f'rotations_{season_underscore}.csv')
    raw_pbp_path = os.path.join(BASE_DIR, 'data', 'pureData', f'season_{season_underscore}.csv')

    print(f"🏥 Starting Data Health Check (Season {season_underscore})...")
    
    if not os.path.exists(rotations_path):
        print(f"❌ Rotations file not found: {rotations_path}")
        return

    if not os.path.exists(raw_pbp_path):
        print(f"❌ Raw PBP file not found: {raw_pbp_path}")
        return

    # 1. Load rotation and raw PBP game IDs
    df_rot = pd.read_csv(rotations_path)
    unique_fetched = df_rot['gameId'].astype(str).str.zfill(10).unique()
    
    df_source = pd.read_csv(raw_pbp_path, usecols=['gameId'])
    total_games = df_source['gameId'].astype(str).str.zfill(10).nunique()
    
    # 2. Coverage calculation
    success_rate = (len(unique_fetched) / total_games) * 100
    
    print(f"\n📊 Summary ({season_underscore}):")
    print(f"   Total Games in Season: {total_games}")
    print(f"   Successfully Fetched:  {len(unique_fetched)}")
    print(f"   Missing Games:         {total_games - len(unique_fetched)}")
    print(f"   ✅ Success Rate:       {success_rate:.1f}%")
    
    # 3. Quality Check
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

def parse_args():
    parser = argparse.ArgumentParser(description="Check data health and rotation coverage.")
    parser.add_argument("--season", type=str, default="2024_25", help="Season string (e.g. 2024_25, 2024-25, 2021_22)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    check_health(season=args.season)
