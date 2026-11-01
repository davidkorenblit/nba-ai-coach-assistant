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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def inspect_substitution_data(file_path: str):
    print(f"🕵️‍♂️ Starting Feasibility Check on: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found at {file_path}")
        return

    try:
        # 1. Load a sample (first 50,000 rows is usually enough to catch full games)
        cols = [
            'gameId', 'period', 'clock', 
            'actionType', 'subType', 
            'description', 'playerName', 'teamTricode'
        ]
        
        df = pd.read_csv(file_path, nrows=50000, usecols=lambda c: c in cols)
        
        # 2. Filter for Substitution Events
        mask_sub = df['description'].str.contains('SUB', case=False, na=False)
        subs_df = df[mask_sub]

        if subs_df.empty:
            print("⚠️ No substitution events found in the first 50k rows.")
            print("Unique ActionTypes:", df['actionType'].unique())
            return

        print(f"✅ Found {len(subs_df)} substitution events.")
        
        # 3. Inspect the Logic
        print("\n🔍 --- LOGIC INSPECTION ---")
        print("Goal: Determine if 'playerName' refers to the player COMING IN or GOING OUT.")
        
        sample = subs_df[['clock', 'teamTricode', 'playerName', 'description']].head(10)
        print(sample.to_string(index=False))
        
        print("\n🧠 Analysis Helper:")
        first_row = subs_df.iloc[0]
        desc = first_row['description']
        p_name = first_row['playerName']
        
        print(f"Event: {desc}")
        print(f"Column 'playerName': {p_name}")
        
        if p_name in desc:
            print("✅ 'playerName' is explicitly inside the description.")
            # Check for 'FOR' structure
            if 'FOR' in desc:
                parts = desc.split('FOR')
                print(f"   -> Likely Structure: [Player IN] FOR [Player OUT]")
                print(f"   -> Left part: {parts[0].strip()}")
                print(f"   -> Right part: {parts[1].strip()}")
            else:
                print("⚠️ 'FOR' keyword not found. Parsing might need Regex.")
        else:
            print("⚠️ 'playerName' column does NOT match text in description. Check IDs.")

    except Exception as e:
        print(f"❌ Critical Error: {e}")

def parse_args():
    parser = argparse.ArgumentParser(description="Inspect substitution data for a given season.")
    parser.add_argument("--season", type=str, default="2024_25", help="Season string (e.g. 2024_25, 2024-25, 2021_22)")
    parser.add_argument("--file", type=str, default=None, help="Explicit path to raw season CSV")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.file:
        target_path = args.file
    else:
        season_underscore, _ = normalize_season(args.season)
        target_path = os.path.join(BASE_DIR, 'data', 'pureData', f'season_{season_underscore}.csv')
    inspect_substitution_data(target_path)
