import os
import sys
import time
import argparse
import pandas as pd

# --- CRITICAL FIX FOR NBA API TIMEOUTS (Akamai WAF Bypass) ---
try:
    from curl_cffi import requests as curl_requests
    from nba_api.library.http import NBAHTTP
    from nba_api.stats.library.http import NBAStatsHTTP
    from nba_api.live.nba.library.http import NBALiveHTTP

    session = curl_requests.Session(impersonate='chrome120', verify=False)
    session.headers.update({
        'Referer': 'https://www.nba.com/',
        'Origin': 'https://www.nba.com',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.9',
        'x-nba-stats-origin': 'stats',
        'x-nba-stats-token': 'true',
    })

    # Update NBAStatsHTTP headers as well for consistency
    NBAStatsHTTP.headers.update({
        'Referer': 'https://www.nba.com/',
        'Origin': 'https://www.nba.com',
        'x-nba-stats-origin': 'stats',
        'x-nba-stats-token': 'true',
    })

    # Ensure all get requests bypass local SSL verification if inspection proxy is active
    orig_get = session.get
    def patched_get(*args, **kwargs):
        kwargs['verify'] = False
        return orig_get(*args, **kwargs)
    session.get = patched_get

    proxy = os.environ.get("NBA_API_PROXY")
    if proxy:
        print(f"🌐 Using Proxy for NBA API: {proxy}")
        session.proxies = {"http": proxy, "https": proxy}

    try:
        session.get('https://www.nba.com', timeout=15)
    except Exception as e:
        print(f"⚠️ Warning: NBA warmup request failed: {e}")

    NBAHTTP.get_session = lambda self: session
    NBALiveHTTP.get_session = lambda self: session
    print("✅ Successfully configured curl_cffi Chrome impersonation for NBA API.")
except Exception as e:
    print(f"⚠️ Warning: Could not configure curl_cffi: {e}")

from nba_api.stats.endpoints import leaguedashplayerstats

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'lookup')
DEFAULT_SEASONS = ['2021-22', '2022-23', '2023-24']
DEFAULT_MIN_GP = 45


def fetch_high_usage_for_season(season: str, min_games: int = DEFAULT_MIN_GP, max_retries: int = 3):
    """
    Fetches high usage star players for each NBA team in a given season.
    Saves the output to data/lookup/high_usage_players_{season}.csv.
    """
    print(f"\n=======================================================")
    print(f"🚀 Fetching High Usage Players for {season} (Min GP: {min_games})...")
    print(f"=======================================================")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, f'high_usage_players_{season}.csv')

    df = None
    for attempt in range(1, max_retries + 1):
        try:
            print(f"📡 Querying LeagueDashPlayerStats (Attempt {attempt}/{max_retries})...")
            stats = leaguedashplayerstats.LeagueDashPlayerStats(
                season=season,
                measure_type_detailed_defense='Advanced',
                timeout=30
            )
            data_frames = stats.get_data_frames()
            if data_frames and len(data_frames) > 0 and not data_frames[0].empty:
                df = data_frames[0]
                break
        except Exception as e:
            print(f"⚠️ Error on attempt {attempt}: {e}")
            if attempt < max_retries:
                sleep_time = attempt * 3
                print(f"⏳ Waiting {sleep_time}s before retrying...")
                time.sleep(sleep_time)

    if df is None or df.empty:
        print(f"❌ Failed to fetch player stats for season {season}.")
        return False

    # Filter by minimum games played with dynamic fallback
    current_min_gp = min_games
    df_filtered = df[df['GP'] >= current_min_gp].copy()
    
    while df_filtered['TEAM_ID'].nunique() < 30 and current_min_gp > 20:
        current_min_gp -= 5
        print(f"ℹ️ Fewer than 30 teams represented. Lowering min GP to {current_min_gp}...")
        df_filtered = df[df['GP'] >= current_min_gp].copy()

    # Sort by TEAM_ID and USG_PCT descending
    df_sorted = df_filtered.sort_values(by=['TEAM_ID', 'USG_PCT'], ascending=[True, False])
    
    # Pick top usage player per team
    top_usage_per_team = df_sorted.drop_duplicates(subset=['TEAM_ID'], keep='first')

    # Keep relevant columns
    cols_to_keep = ['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ABBREVIATION', 'USG_PCT', 'GP', 'MIN']
    final_df = top_usage_per_team[cols_to_keep].copy()

    # Save to CSV
    final_df.to_csv(output_file, index=False)
    print(f"✅ Success! Saved {len(final_df)} star players for {season} to:")
    print(f"   {output_file}")
    
    print("\n📊 Preview (Top 5 Highest Usage Stars):")
    print(final_df.sort_values('USG_PCT', ascending=False).head(5).to_string(index=False))
    return True


def main():
    parser = argparse.ArgumentParser(description="Fetch High Usage Players for NBA seasons")
    parser.add_argument(
        '--seasons',
        nargs='+',
        default=DEFAULT_SEASONS,
        help="List of seasons to fetch, e.g. 2021-22 2022-23 2023-24"
    )
    parser.add_argument(
        '--min-gp',
        type=int,
        default=DEFAULT_MIN_GP,
        help="Minimum games played to be considered (default: 45)"
    )
    args = parser.parse_args()

    results = {}
    for season in args.seasons:
        success = fetch_high_usage_for_season(season, min_games=args.min_gp)
        results[season] = success
        # Short polite pause between seasons
        time.sleep(2)

    print("\n=======================================================")
    print("📋 SUMMARY OF HIGH USAGE STARS COLLECTION")
    print("=======================================================")
    for season, success in results.items():
        status = "✅ SUCCEEDED" if success else "❌ FAILED"
        print(f"Season {season}: {status}")


if __name__ == "__main__":
    main()
