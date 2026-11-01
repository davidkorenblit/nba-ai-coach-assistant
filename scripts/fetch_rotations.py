"""
Multi-Season NBA Game Rotations Fetcher
Fetches game rotation datasets from stats.nba.com with Akamai WAF Bypass (curl_cffi chrome120),
robust checkpointing/resume capabilities, thread-local session management, and batch persistence.

Usage Examples:
    python scripts/fetch_rotations.py --season 2021_22
    python scripts/fetch_rotations.py --season 2022_23 --workers 2
    python scripts/fetch_rotations.py --all
    python scripts/fetch_rotations.py --season 2021_22 --limit 5 --save-interval 2
    python scripts/fetch_rotations.py --all --skip-failed

Data outputs:
    data/pureData/rotations_{season}.csv
    data/pureData/rotations_failed_{season}.csv
"""

import os
import sys
import time
import random
import argparse
import threading
import concurrent.futures
from datetime import datetime, timezone
import pandas as pd

# Akamai WAF Bypass & NBA API Session Configuration
from curl_cffi import requests as curl_requests
from nba_api.library.http import NBAHTTP
from nba_api.stats.library.http import NBAStatsHTTP
from nba_api.live.nba.library.http import NBALiveHTTP
from nba_api.stats.endpoints import gamerotation

# Canonical NBA headers
NBA_HEADERS = {
    'Referer': 'https://www.nba.com/',
    'Origin': 'https://www.nba.com',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.9',
    'x-nba-stats-origin': 'stats',
    'x-nba-stats-token': 'true',
    'Sec-Ch-Ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
    'Sec-Ch-Ua-Mobile': '?0',
    'Sec-Ch-Ua-Platform': '"Windows"',
    'Sec-Fetch-Dest': 'empty',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Site': 'same-site'
}

NBAStatsHTTP.headers.update(NBA_HEADERS)

# Thread-local storage for curl_cffi sessions ensuring concurrent thread safety
_thread_local = threading.local()

def get_thread_session():
    """Returns a thread-local curl_cffi session configured with Chrome-120 TLS fingerprint and NBA headers."""
    if not hasattr(_thread_local, 'session') or _thread_local.session is None:
        sess = curl_requests.Session(impersonate='chrome120')
        sess.headers.update(NBA_HEADERS)
        proxy = os.environ.get("NBA_API_PROXY")
        if proxy:
            sess.proxies = {"http": proxy, "https": proxy}

        try:
            sess.get('https://www.nba.com', timeout=15)
        except Exception:
            pass
        _thread_local.session = sess
    return _thread_local.session

# Monkeypatch nba_api HTTP sessions to route through our curl_cffi session
NBAHTTP.get_session = lambda self: get_thread_session()
NBALiveHTTP.get_session = lambda self: get_thread_session()

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'pureData')
DEFAULT_TARGET_SEASONS = ['2021_22', '2022_23', '2023_24']

# Exact schema matching rotations_2024_25.csv
REQUIRED_COLUMNS = [
    'gameId', 'team_side', 'PERSON_ID', 'IN_TIME_REAL', 'OUT_TIME_REAL', 'USG_PCT',
    'GAME_ID', 'TEAM_ID', 'TEAM_CITY', 'TEAM_NAME', 'PLAYER_FIRST', 'PLAYER_LAST',
    'PLAYER_PTS', 'PT_DIFF'
]

# Thread safety lock for disk writes
file_lock = threading.Lock()


def normalize_season(season_str: str) -> str:
    """Normalizes season string format: e.g. 2021-22 -> 2021_22."""
    return season_str.strip().replace('-', '_')


def get_existing_game_ids(output_path: str) -> set:
    """Reads already collected gameIds from the output CSV to enable resuming."""
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        return set()
    try:
        df = pd.read_csv(output_path, usecols=['gameId'], dtype={'gameId': str})
        return set(df['gameId'].astype(str).str.zfill(10).unique())
    except Exception as e:
        print(f"⚠️ Warning: Could not read existing games from {output_path}: {e}")
        return set()


def get_failed_game_ids(failed_log_path: str) -> set:
    """Reads previously failed gameIds from the failed CSV log."""
    if not os.path.exists(failed_log_path) or os.path.getsize(failed_log_path) == 0:
        return set()
    try:
        df = pd.read_csv(failed_log_path, usecols=['gameId'], dtype={'gameId': str})
        return set(df['gameId'].astype(str).str.zfill(10).unique())
    except Exception:
        return set()


def save_batch_to_csv(data_frames: list, output_path: str) -> int:
    """Appends collected batch of DataFrames to the output CSV ensuring exact column order."""
    if not data_frames:
        return 0

    df_batch = pd.concat(data_frames, ignore_index=True)

    # Ensure required columns are present and strictly ordered
    existing_cols = [c for c in REQUIRED_COLUMNS if c in df_batch.columns]
    other_cols = [c for c in df_batch.columns if c not in REQUIRED_COLUMNS]
    df_batch = df_batch[existing_cols + other_cols]

    with file_lock:
        file_exists = os.path.exists(output_path) and os.path.getsize(output_path) > 0
        df_batch.to_csv(output_path, mode='a', header=not file_exists, index=False)

    return len(df_batch)


def log_failed_game(game_id: str, season: str, error_msg: str, attempts: int, failed_log_path: str):
    """Records games that failed all retries into a dedicated CSV log for later retry/analysis."""
    with file_lock:
        file_exists = os.path.exists(failed_log_path) and os.path.getsize(failed_log_path) > 0
        df_err = pd.DataFrame([{
            'gameId': str(game_id).zfill(10),
            'season': season,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'error': str(error_msg),
            'attempts': attempts
        }])
        df_err.to_csv(failed_log_path, mode='a', header=not file_exists, index=False)


def fetch_single_game_rotation(game_id: str, max_retries: int = 3, base_delay: float = 1.0):
    """
    Fetches rotation data for a single game from stats.nba.com.
    Returns: (game_id, frames_list_or_None, error_or_None)
    """
    game_id_padded = str(game_id).zfill(10)
    last_err = None

    for attempt in range(1, max_retries + 1):
        try:
            # Jitter to prevent rate limit spikes
            time.sleep(random.uniform(base_delay, base_delay + 0.8))

            rot = gamerotation.GameRotation(game_id=game_id_padded, timeout=25)
            frames = []

            # Home team
            if hasattr(rot, 'home_team'):
                df_home = rot.home_team.get_data_frame()
                if not df_home.empty:
                    df_home['gameId'] = game_id_padded
                    df_home['team_side'] = 'home'
                    frames.append(df_home)

            # Away team
            if hasattr(rot, 'away_team'):
                df_away = rot.away_team.get_data_frame()
                if not df_away.empty:
                    df_away['gameId'] = game_id_padded
                    df_away['team_side'] = 'away'
                    frames.append(df_away)

            if frames:
                return game_id_padded, frames, None
            else:
                return game_id_padded, None, "Empty data frames returned from NBA API"

        except Exception as e:
            last_err = e
            err_str = str(e)

            # HTTP 500 with empty body means game has no rotation tracking in stats.nba.com
            if "Expecting value: line 1 column 1" in err_str or "500" in err_str:
                return game_id_padded, None, f"HTTP 500 / No rotation data on stats.nba.com: {err_str}"

            if attempt < max_retries:
                backoff = random.uniform(1.2, 2.5) * attempt
                time.sleep(backoff)

    return game_id_padded, None, f"Exceeded {max_retries} attempts. Last error: {last_err}"


def fetch_season_rotations(season: str, max_workers: int = 2, save_interval: int = 20,
                           max_retries: int = 3, limit: int = None, skip_failed: bool = False,
                           base_delay: float = 1.0):
    """
    Fetches rotations for a specific season with checkpointing and batch saving.
    """
    season_norm = normalize_season(season)
    raw_pbp_path = os.path.join(DATA_DIR, f'season_{season_norm}.csv')
    output_path = os.path.join(DATA_DIR, f'rotations_{season_norm}.csv')
    failed_log_path = os.path.join(DATA_DIR, f'rotations_failed_{season_norm}.csv')

    print(f"\n=======================================================")
    print(f"🏀 Fetching Rotations for Season: {season_norm}")
    print(f"📂 Source PBP: {raw_pbp_path}")
    print(f"💾 Output File: {output_path}")
    print(f"📋 Failed Log: {failed_log_path}")
    print(f"⚙️ Workers: {max_workers} | Save Interval: {save_interval} games | Retries: {max_retries}")
    print(f"=======================================================")

    if not os.path.exists(raw_pbp_path):
        print(f"❌ Error: Source file not found: {raw_pbp_path}")
        return False

    # 1. Read all unique game IDs from source PBP
    print("📖 Reading source game IDs...")
    df_source = pd.read_csv(raw_pbp_path, usecols=['gameId'], low_memory=False)
    all_game_ids = list(df_source['gameId'].astype(str).str.zfill(10).unique())

    # 2. Check existing game IDs for resume
    existing_ids = get_existing_game_ids(output_path)
    games_to_process = [gid for gid in all_game_ids if gid not in existing_ids]

    if skip_failed:
        failed_ids = get_failed_game_ids(failed_log_path)
        before_count = len(games_to_process)
        games_to_process = [gid for gid in games_to_process if gid not in failed_ids]
        print(f"⏭️ Skipped {before_count - len(games_to_process)} previously failed games (--skip-failed).")

    print(f"📊 Total Games in Season: {len(all_game_ids)}")
    print(f"✅ Already Processed:   {len(existing_ids)}")
    print(f"🔄 Remaining to Fetch:   {len(games_to_process)}")

    if limit and limit > 0:
        games_to_process = games_to_process[:limit]
        print(f"🔬 Limit applied: Fetching first {len(games_to_process)} remaining games")

    if not games_to_process:
        print(f"🎉 All games for season {season_norm} are already fetched!")
        return True

    # 3. Concurrent fetching with ThreadPoolExecutor & Batch Persistence
    batch_data = []
    completed_count = 0
    success_count = 0
    error_count = 0
    total_rows_saved = 0
    games_in_current_batch = 0

    t_start = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_game = {
            executor.submit(fetch_single_game_rotation, gid, max_retries, base_delay): gid
            for gid in games_to_process
        }

        for future in concurrent.futures.as_completed(future_to_game):
            game_id = future_to_game[future]
            completed_count += 1

            try:
                gid, result, err = future.result()
            except Exception as exc:
                gid, result, err = game_id, None, str(exc)

            if result:
                batch_data.extend(result)
                success_count += 1
            else:
                error_count += 1
                log_failed_game(game_id, season_norm, err, max_retries, failed_log_path)

            games_in_current_batch += 1

            # Batch save checkpoint every `save_interval` games
            if games_in_current_batch >= save_interval:
                saved = save_batch_to_csv(batch_data, output_path)
                total_rows_saved += saved
                batch_data = []
                games_in_current_batch = 0
                print(f" 💾 Checkpoint saved: +{saved} rows written to {os.path.basename(output_path)}")

            elapsed = time.time() - t_start
            rate = completed_count / elapsed if elapsed > 0 else 0
            eta_sec = (len(games_to_process) - completed_count) / rate if rate > 0 else 0
            print(f"   ⏳ Progress: {completed_count}/{len(games_to_process)} "
                  f"({completed_count/len(games_to_process)*100:.1f}%) | "
                  f"Success: {success_count} | Errors: {error_count} | "
                  f"Rate: {rate:.2f} g/s | ETA: {eta_sec/60:.1f}m", end="\r")

    # Final flush of remaining batch
    if batch_data:
        saved = save_batch_to_csv(batch_data, output_path)
        total_rows_saved += saved
        print(f"\n 💾 Final batch saved: +{saved} rows written to {os.path.basename(output_path)}")

    total_time = time.time() - t_start
    print(f"\n\n🏁 Completed Season {season_norm} in {total_time/60:.2f} minutes.")
    print(f"   - Processed: {completed_count}")
    print(f"   - Succeeded: {success_count}")
    print(f"   - Failed:    {error_count} (logged in {os.path.basename(failed_log_path)})")
    print(f"   - Total Rows Saved this run: {total_rows_saved}")
    if os.path.exists(output_path):
        current_existing = get_existing_game_ids(output_path)
        print(f"   - Total Games now in {os.path.basename(output_path)}: {len(current_existing)}/{len(all_game_ids)}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Multi-Season NBA Rotations Fetcher with Akamai WAF Bypass & Resume")
    parser.add_argument('--season', type=str, default=None,
                        help="Target season to fetch, e.g. 2021_22, 2022_23, 2023_24")
    parser.add_argument('--all', action='store_true',
                        help="Fetch all multi-season targets (2021_22, 2022_23, 2023_24)")
    parser.add_argument('--workers', type=int, default=2,
                        help="Concurrent thread workers (default: 2)")
    parser.add_argument('--save-interval', type=int, default=20,
                        help="Save batch to CSV every N games (default: 20)")
    parser.add_argument('--max-retries', type=int, default=3,
                        help="Max retry attempts per game (default: 3)")
    parser.add_argument('--delay', type=float, default=1.0,
                        help="Base delay between requests in seconds (default: 1.0)")
    parser.add_argument('--limit', type=int, default=None,
                        help="Limit number of games to fetch (for testing)")
    parser.add_argument('--skip-failed', action='store_true',
                        help="Skip games already logged as failed in previous attempts")

    args = parser.parse_args()

    if args.all:
        seasons = DEFAULT_TARGET_SEASONS
    elif args.season:
        seasons = [args.season]
    else:
        print("ℹ️ No season specified. Use --season <season> or --all. Defaulting to 2021_22...")
        seasons = ['2021_22']

    for season in seasons:
        fetch_season_rotations(
            season=season,
            max_workers=args.workers,
            save_interval=args.save_interval,
            max_retries=args.max_retries,
            limit=args.limit,
            skip_failed=args.skip_failed,
            base_delay=args.delay
        )


if __name__ == "__main__":
    main()
