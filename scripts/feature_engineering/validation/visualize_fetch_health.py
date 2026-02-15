import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- Config (4 levels up to Root) ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ROTATIONS_PATH = os.path.join(BASE_DIR, 'data', 'pureData', 'rotations_2024_25.csv')
OUTPUT_PLOT = os.path.join(BASE_DIR, 'data', 'reports', 'fetch_quality_heatmap.png')

def analyze_fetch_quality():
    print("🏥 Starting Deep Health Check on Official Rotations (Fetch)...")
    
    if not os.path.exists(ROTATIONS_PATH):
        print("❌ Rotations file not found."); return

    df = pd.read_csv(ROTATIONS_PATH)
    
    # חישוב: לכל משחק, כמה שחקנים רשומים בכל רגע?
    # אנחנו נדגום 5 נקודות זמן בכל משחק (תחילת רבעים וסוף משחק)
    check_points = [100, 800, 1500, 2200, 2800] # שניות מתחילת המשחק
    
    game_results = []
    game_ids = df['gameId'].unique()
    
    print(f"🧐 Analyzing internal structure of {len(game_ids)} fetched games...")

    for gid in game_ids:
        game_data = df[df['gameId'] == gid]
        snapshots = []
        
        for t in check_points:
            # סופרים כמה שחקנים "על המגרש" בזמן t
            active = game_data[(game_data['IN_TIME_REAL'] <= t) & (game_data['OUT_TIME_REAL'] > t)]
            snapshots.append(len(active))
        
        game_results.append(snapshots)

    # יצירת מטריצה לגרף
    quality_matrix = np.array(game_results)
    
    # --- ויזואליזציה ---
    plt.figure(figsize=(12, 8))
    # אנחנו מצפים לראות "10" (5 נגד 5). כל מה שונה מ-10 הוא תקלה.
    sns.heatmap(quality_matrix[:100], annot=False, cmap='RdYlGn', vmin=0, vmax=12) 
    
    plt.title('Official Fetch Quality (First 100 Games Sample)\nTarget: 10 Players (Green) | Errors (Red/Yellow)')
    plt.xlabel('Game Timeline (Snapshots 1-5)')
    plt.ylabel('Game Index')
    
    os.makedirs(os.path.dirname(OUTPUT_PLOT), exist_ok=True)
    plt.savefig(OUTPUT_PLOT)
    
    # סטטיסטיקה מסכמת
    perfect_snapshots = np.sum(quality_matrix == 10)
    total_snapshots = quality_matrix.size
    print(f"\n📊 Fetch Integrity Score: {(perfect_snapshots/total_snapshots):.1%}")
    print(f"💡 (This means in X% of the checked moments, the API gave us exactly 10 players)")
    print(f"✅ Plot saved to: {OUTPUT_PLOT}")

if __name__ == "__main__":
    analyze_fetch_quality()