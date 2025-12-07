import pandas as pd
import os

# הגדרות נתיב
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, '..', 'data', 'pureData')
INPUT_FILE = os.path.join(DATA_DIR, "season_2024_25.csv")

def verify_defense_structure():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: File not found at {INPUT_FILE}")
        return

    print(f"Loading data from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    
    # --- בדיקה 1: האם העמודות קיימות בכלל? ---
    print("\n=== 1. COLUMN EXISTENCE CHECK ===")
    target_cols = ['stealPlayerName', 'blockPlayerName', 'stealPersonId', 'blockPersonId']
    
    missing_cols = []
    for col in target_cols:
        if col in df.columns:
            print(f"[V] Column '{col}' EXISTS in the dataset.")
        else:
            print(f"[X] Column '{col}' is MISSING.")
            missing_cols.append(col)
            
    # --- בדיקה 2: איפה המידע מסתתר? (Main vs Side) ---
    print("\n=== 2. DATA LOCATION CHECK ===")
    
    # חטיפות
    steals = df[df['actionType'] == 'steal']
    if not steals.empty:
        main_filled = steals['playerName'].notna().sum()
        side_filled = 0
        if 'stealPlayerName' in df.columns:
            side_filled = steals['stealPlayerName'].notna().sum()
            
        print(f"Steals (Total: {len(steals)}):")
        print(f"  -> 'playerName' (Main) filled count:      {main_filled}")
        print(f"  -> 'stealPlayerName' (Side) filled count: {side_filled}")
    else:
        print("No steals found in data.")

    # חסימות
    blocks = df[df['actionType'] == 'block']
    if not blocks.empty:
        main_filled = blocks['playerName'].notna().sum()
        side_filled = 0
        if 'blockPlayerName' in df.columns:
            side_filled = blocks['blockPlayerName'].notna().sum()
            
        print(f"Blocks (Total: {len(blocks)}):")
        print(f"  -> 'playerName' (Main) filled count:      {main_filled}")
        print(f"  -> 'blockPlayerName' (Side) filled count: {side_filled}")

    print("\n--- CONCLUSION ---")
    if missing_cols:
        print("Warning: Some columns are missing entirely from the CSV.")
    else:
        print("Structure looks correct (columns exist). Now check which one has the data.")

if __name__ == "__main__":
    verify_defense_structure()