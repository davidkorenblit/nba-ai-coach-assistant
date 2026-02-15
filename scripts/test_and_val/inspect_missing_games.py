import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_PBP_PATH = os.path.join(BASE_DIR, 'data', 'pureData', 'season_2024_25.csv')
ROTATIONS_PATH = os.path.join(BASE_DIR, 'data', 'pureData', 'rotations_2024_25.csv')
MISSING_IDS_PATH = os.path.join(BASE_DIR, 'data', 'pureData', 'missing_ids.csv')


def inspect():
    # 1. כל המשחקים
    df_source = pd.read_csv(RAW_PBP_PATH, usecols=['gameId'], dtype={'gameId': str})
    all_games = set(df_source['gameId'].str.zfill(10))
    
    # 2. מה שהשגנו
    fetched = set()
    if os.path.exists(ROTATIONS_PATH):
        df_fetched = pd.read_csv(ROTATIONS_PATH, usecols=['gameId'], dtype={'gameId': str})
        fetched = set(df_fetched['gameId'].str.zfill(10))
        
    # 3. חיתוך
    missing = list(all_games - fetched)
    real_missing = [g for g in missing if g.startswith('002')] # רק עונה סדירה
    
    print(f"Missing: {len(real_missing)}")
    pd.Series(real_missing).to_csv(MISSING_IDS_PATH, index=False, header=False)
    print("✅ Saved to missing_ids.csv")

if __name__ == "__main__":
    inspect()