import pandas as pd
import os
import glob
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

# --- Path Configuration ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, '..', '..', 'data', 'pureData')

# Critical column validation list
COLUMNS_TO_CHECK = [
    # 1. Identity & Sequence
    'gameId', 'period', 'clock', 'actionNumber', 
    
    # 2. Event Detail
    'actionType', 'subType', 'description', 'qualifiers',
    
    # 3. Players / Actors
    'playerName', 'personId', 'teamId', 
    
    # 4. Outcome
    'scoreHome', 'scoreAway',
    
    # 5. Context & Lineups
    'personIdsFilter', 'x', 'y', 'shotDistance',
    
    # 6. Counters
    'foulPersonalTotal', 'turnoverTotal', 'pointsTotal'
]

def check_completeness(season: str = "2024_25", file_path: str = None):
    season_underscore, _ = normalize_season(season)
    
    if file_path:
        csv_files = [file_path]
        print(f"--- Starting Data Completeness Check (Custom File: {file_path}) ---")
    else:
        target_file = os.path.join(DATA_DIR, f"season_{season_underscore}.csv")
        print(f"--- Starting Data Completeness Check (Season {season_underscore}) ---")
        print(f"Target CSV file: {os.path.abspath(target_file)}")
        if not os.path.exists(target_file):
            print(f"❌ Target CSV file not found: {target_file}")
            return
        csv_files = [target_file]

    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        print(f"\n==========================================")
        print(f"📂 Analyzing file: {file_name}")
        print(f"==========================================")
        
        try:
            df = pd.read_csv(file_path, low_memory=False)
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            continue
            
        total_rows = len(df)
        print(f"Total Rows: {total_rows}")
        
        # Results Summary Table
        print(f"{'Column Name':<25} | {'Missing':<10} | {'% Missing':<10} | {'Status'}")
        print("-" * 65)
        
        for col in COLUMNS_TO_CHECK:
            if col not in df.columns:
                print(f"{col:<25} | {'MISSING COL':<10} | {'100%':<10} | ❌ NOT FOUND")
                continue
            
            # Count missing values (NaN/Null)
            missing_count = df[col].isna().sum()
            missing_pct = (missing_count / total_rows) * 100
            
            # Determine visual status
            if missing_pct == 0:
                status = "✅ Perfect"
            elif missing_pct < 5:
                status = "⚠️ OK (Low)"
            elif missing_pct < 20:
                status = "Rx Warning"
            else:
                status = "❌ Critical"
            
            # In some cases missing data is expected (e.g. x,y coordinates on turnovers)
            # Note: Diagnostic report shows raw data health.
            
            print(f"{col:<25} | {missing_count:<10} | {missing_pct:>6.2f}%    | {status}")

    print("\n--- Check Complete ---")

def parse_args():
    parser = argparse.ArgumentParser(description="Check raw PBP data integrity before feature engineering.")
    parser.add_argument("--season", type=str, default="2024_25", help="Season string (e.g. 2024_25, 2024-25, 2021_22)")
    parser.add_argument("--file", type=str, default=None, help="Optional direct path to CSV file")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    check_completeness(season=args.season, file_path=args.file)
