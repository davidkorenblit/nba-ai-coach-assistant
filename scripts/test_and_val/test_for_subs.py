import pandas as pd
import os

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FILE_PATH = os.path.join(BASE_DIR, 'data', 'pureData', 'season_2024_25.csv')

def inspect_substitution_data():
    print(f"🕵️‍♂️ Starting Feasibility Check on: {FILE_PATH}")
    
    if not os.path.exists(FILE_PATH):
        print(f"❌ Error: File not found at {FILE_PATH}")
        return

    try:
        # 1. Load a sample (first 50,000 rows is usually enough to catch full games)
        # We load specific columns based on your documentation to check the mapping
        cols = [
            'gameId', 'period', 'clock', 
            'actionType', 'subType', 
            'description', 'playerName', 'teamTricode'
        ]
        
        # Using lambda in usecols to avoid error if a column is slightly named differently in CSV
        df = pd.read_csv(FILE_PATH, nrows=50000, usecols=lambda c: c in cols)
        
        # 2. Filter for Substitution Events
        # Usually identified by 'SUB' in description or specific action types
        mask_sub = df['description'].str.contains('SUB', case=False, na=False)
        subs_df = df[mask_sub]

        if subs_df.empty:
            print("⚠️ No substitution events found in the first 50k rows.")
            print("Unique ActionTypes:", df['actionType'].unique())
            return

        print(f"✅ Found {len(subs_df)} substitution events.")
        
        # 3. Inspect the Logic (The crucial part)
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

if __name__ == "__main__":
    inspect_substitution_data()