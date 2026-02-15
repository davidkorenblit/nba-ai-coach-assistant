import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import time

# --- Configuration ---
# שינינו את המיקום, אז צריך לעלות 3 רמות כדי להגיע ל-Root
# File is at: scripts/feature_engineering/validation/generate_lineup_quality_report.py
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

RAW_PBP_PATH = os.path.join(BASE_DIR, 'data', 'pureData', 'season_2024_25.csv')
REPORT_DIR = os.path.join(BASE_DIR, 'data', 'reports')
OUTPUT_CSV_PATH = os.path.join(REPORT_DIR, 'game_quality_index.csv')
OUTPUT_PLOT_PATH = os.path.join(REPORT_DIR, 'quality_distribution.png')

# לוודא שתיקיית הדוחות קיימת
os.makedirs(REPORT_DIR, exist_ok=True)

class LineupInferenceEngine:
    """
    המנוע שמנסה להרכיב חמישיות על בסיס Play-by-Play בלבד.
    משמש אך ורק לבדיקת היתכנות ואיכות נתונים.
    """
    def __init__(self, df_game):
        self.df = df_game.sort_values('orderNumber').reset_index(drop=True)
        # זיהוי קבוצות (מתעלם מ-NaN)
        teams = self.df['teamTricode'].dropna().unique()
        self.teams = [t for t in teams if t]
        
        # המצב הנוכחי (מי על המגרש)
        self.current_lineups = {t: set() for t in self.teams}

    def _update_from_sub(self, row):
        """טיפול בחילופים"""
        desc = str(row['description'])
        team = row['teamTricode']
        player_name = row['playerName']
        
        if not team or team not in self.current_lineups:
            return

        if 'SUB out' in desc:
            self.current_lineups[team].discard(player_name)
        elif 'SUB in' in desc:
            self.current_lineups[team].add(player_name)

    def _update_from_action(self, row):
        """טיפול בפעולות שוטפות (Lazy Loading)"""
        team = row['teamTricode']
        player_name = row['playerName']
        
        if not team or team not in self.current_lineups or pd.isna(player_name):
            return

        # אם שחקן פעיל, הוא חייב להיות על המגרש
        if player_name not in self.current_lineups[team]:
            self.current_lineups[team].add(player_name)

    def process_game(self):
        """רצה על כל המשחק ומחזירה דוח כיסוי לכל שורה"""
        game_log = []
        
        for idx, row in self.df.iterrows():
            # עדכון מצב
            if 'SUB' in str(row['description']):
                self._update_from_sub(row)
            else:
                self._update_from_action(row)
            
            # בדיקת מצב נוכחי
            known_players = sum(len(self.current_lineups[t]) for t in self.teams)
            
            game_log.append({
                'known_players': known_players
            })
            
        return pd.DataFrame(game_log)

def generate_quality_report():
    print(f"🚀 Starting Quality Index Generation...")
    print(f"📂 Reading Raw Data: {RAW_PBP_PATH}")
    
    if not os.path.exists(RAW_PBP_PATH):
        print(f"❌ Error: Raw data file missing at {RAW_PBP_PATH}")
        print("   Please check the path logic in BASE_DIR.")
        return

    # טעינת עמודות רלוונטיות בלבד (Memory Efficient)
    cols = ['gameId', 'period', 'orderNumber', 'teamTricode', 'playerName', 'description']
    try:
        df_all = pd.read_csv(RAW_PBP_PATH, usecols=lambda c: c in cols)
    except Exception as e:
        print(f"❌ Critical Error reading CSV: {e}")
        return
    
    game_ids = df_all['gameId'].unique()
    total_games = len(game_ids)
    print(f"📊 Analyzing {total_games} games...")

    quality_data = []
    start_time = time.time()
    
    for i, gid in enumerate(game_ids):
        # Progress Log every 100 games
        if i > 0 and i % 100 == 0:
            elapsed = time.time() - start_time
            print(f"   Processed {i}/{total_games} games... ({elapsed:.1f}s)")

        # עיבוד משחק בודד
        df_game = df_all[df_all['gameId'] == gid]
        engine = LineupInferenceEngine(df_game)
        log_df = engine.process_game()
        
        # חישוב מדדי איכות
        if not log_df.empty:
            # כמה אחוז מהזמן ידענו בדיוק 10 שחקנים?
            perfect_coverage = (log_df['known_players'] == 10).mean()
            # כמה אחוז מהזמן ידענו לפחות 8 שחקנים? (סביר)
            decent_coverage = (log_df['known_players'] >= 8).mean()
        else:
            perfect_coverage = 0.0
            decent_coverage = 0.0
            
        # סיווג איכות המשחק
        status = 'TRASH'
        if perfect_coverage > 0.90:
            status = 'PLATINUM'
        elif perfect_coverage > 0.75:
            status = 'GOLD'
        elif decent_coverage > 0.90:
            status = 'SILVER' # קצת רועש אבל שמיש
        
        quality_data.append({
            'gameId': gid,
            'perfect_coverage_pct': round(perfect_coverage, 4),
            'decent_coverage_pct': round(decent_coverage, 4),
            'data_status': status
        })

    # יצירת DataFrame ושמירה
    results_df = pd.DataFrame(quality_data)
    results_df.to_csv(OUTPUT_CSV_PATH, index=False)
    
    print(f"\n✅ Report Saved: {OUTPUT_CSV_PATH}")
    print("\n📈 Quality Summary:")
    print(results_df['data_status'].value_counts())

    # --- ויזואליזציה ---
    try:
        plt.figure(figsize=(10, 6))
        
        # צבעים לפי סטטוס
        palette = {'PLATINUM': 'green', 'GOLD': 'blue', 'SILVER': 'orange', 'TRASH': 'red'}
        sns.histplot(data=results_df, x='perfect_coverage_pct', hue='data_status', 
                     multiple="stack", palette=palette, bins=20)
        
        plt.title('Distribution of Data Quality (Lineup Inference)')
        plt.xlabel('Coverage % (Time with 10 players known)')
        plt.ylabel('Number of Games')
        plt.axvline(0.75, color='black', linestyle='--', label='Acceptable Threshold')
        
        plt.savefig(OUTPUT_PLOT_PATH)
        print(f"✅ Visualization Saved: {OUTPUT_PLOT_PATH}")
    except Exception as e:
        print(f"⚠️ Visualization skipped: {e}")

if __name__ == "__main__":
    generate_quality_report()