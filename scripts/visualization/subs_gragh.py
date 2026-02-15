import pandas as pd
import matplotlib.pyplot as plt
import os
import random
import ast

# --- Config (3 levels up to Root) ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'interim', 'level1_base.csv')

def plot_rotation_map():
    if not os.path.exists(DATA_PATH):
        print("❌ Data not found."); return
    
    df = pd.read_csv(DATA_PATH, low_memory=False)
    
    # בחירת משחק רנדומלי
    gid = random.choice(df['gameId'].unique())
    gdf = df[df['gameId'] == gid].copy()
    gdf.sort_values(['period', 'seconds_remaining'], ascending=[True, False], inplace=True)
    
    # המרת מחרוזות לרשימות
    for col in ['home_lineup', 'away_lineup']:
        gdf[col] = gdf[col].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])

    # יצירת רשימת שחקנים ייחודית שהשתתפו
    all_players = set()
    for row in gdf['home_lineup'] + gdf['away_lineup']:
        all_players.update(row)
    all_players = sorted(list(all_players))
    player_idx = {pid: i for i, pid in enumerate(all_players)}

    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12), sharex=True, 
                                   gridspec_kw={'height_ratios': [3, 1]})
    
    x_axis = range(len(gdf))

    # 1. Rotation Map (Who is on the court?)
    for i, pid in enumerate(all_players):
        # האם השחקן נמצא באחת החמישיות בשורה הזו?
        is_on_court = gdf.apply(lambda r: pid in r['home_lineup'] or pid in r['away_lineup'], axis=1)
        # צובע רק מתי שהוא על המגרש
        ax1.scatter(np.where(is_on_court, x_axis, np.nan), [i]*len(gdf), 
                    marker='|', s=100, color='blue', alpha=0.7)

    ax1.set_yticks(range(len(all_players)))
    ax1.set_yticklabels(all_players, fontsize=8)
    ax1.set_title(f"Level 1: Player Rotation Map (Game {gid})", fontsize=16)
    ax1.set_ylabel("Player ID")
    ax1.grid(axis='x', alpha=0.3)

    # 2. Substitution Timer (How long has this lineup been playing?)
    ax2.plot(x_axis, gdf['time_since_last_sub'], color='red', lw=2)
    ax2.fill_between(x_axis, gdf['time_since_last_sub'], color='red', alpha=0.1)
    ax2.set_title("Lineup Continuity (Seconds since last Sub)", fontsize=14)
    ax2.set_ylabel("Seconds")
    ax2.set_xlabel("Game Timeline (Event sequence)")

    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    import numpy as np # הוספת numpy לחישובים
    plot_rotation_map()