import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random

# --- הגדרות ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # scripts/
DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'interim', 'level1_base.csv')
FIGURES_DIR = os.path.join(BASE_DIR, '..', 'reports', 'figures')

def plot_single_game_dashboard():
    # 1. טעינת נתונים
    if not os.path.exists(DATA_PATH):
        print("❌ Data file not found.")
        return
    
    df = pd.read_csv(DATA_PATH, low_memory=False)
    
    # 2. בחירת משחק אקראי (שיש בו נתונים מלאים)
    game_ids = df['gameId'].unique()
    selected_game_id = random.choice(game_ids)
    game_df = df[df['gameId'] == selected_game_id].copy()
    
    # סידור לפי זמן יורד (כמו במשחק)
    game_df.sort_values(by=['period', 'seconds_remaining'], ascending=[True, False], inplace=True)
    
    # זיהוי קבוצות
    teams = [c.replace('timeouts_remaining_', '') for c in game_df.columns if 'timeouts_remaining_' in c]
    team_a, team_b = teams[0], teams[1]
    
    print(f"🎨 Generating Dashboard for Game ID: {selected_game_id} ({team_a} vs {team_b})")

    # --- הגדרת הדאשבורד (2x2) ---
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f'Level 1 Features Audit: {team_a} vs {team_b} (Game {selected_game_id})', fontsize=16)
    
    # גרף 1: זרימת משחק ופוזשנים (Game Flow & Possessions)
    # נראה את ההפרש, ונצבע את הרקע לפי מי שמחזיק בכדור (פוזשן)
    ax1 = axes[0, 0]
    # ציר X ליניארי פשוט (מספר שורה במשחק)
    x_axis = range(len(game_df))
    ax1.plot(x_axis, game_df['score_margin'], color='black', linewidth=1.5, label='Score Margin')
    ax1.fill_between(x_axis, game_df['score_margin'], 0, where=(game_df['score_margin'] > 0), color='green', alpha=0.3)
    ax1.fill_between(x_axis, game_df['score_margin'], 0, where=(game_df['score_margin'] < 0), color='red', alpha=0.3)
    ax1.set_title(f'Score Margin & Momentum ({team_a} is Positive)', fontsize=12)
    ax1.set_ylabel('Home Lead')
    ax1.grid(True, alpha=0.3)

    # גרף 2: ניהול משאבים (Inventory: Timeouts & Fouls)
    ax2 = axes[0, 1]
    # מלאי פסקי זמן
    ax2.step(x_axis, game_df[f'timeouts_remaining_{team_a}'], label=f'{team_a} Timeouts', where='post', lw=2)
    ax2.step(x_axis, game_df[f'timeouts_remaining_{team_b}'], label=f'{team_b} Timeouts', where='post', lw=2)
    # עבירות (נקודות על הגרף)
    fouls = game_df[game_df['foulPersonalTotal'] > 0]
    if not fouls.empty:
        # ממירים את האינדקס של העבירות לציר ה-X שלנו
        foul_indices = [game_df.index.get_loc(idx) for idx in fouls.index]
        ax2.scatter(foul_indices, [0.5]*len(fouls), color='red', marker='x', label='Foul Committed')
    
    ax2.set_title('Resource Management: Timeouts Inventory', fontsize=12)
    ax2.set_ylim(-0.5, 7.5)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # גרף 3: עייפות וחילופים (Substitutions & Fatigue)
    ax3 = axes[1, 0]
    # נציג את "זמן מאז חילוף אחרון"
    ax3.plot(x_axis, game_df['time_since_last_sub'], color='purple', alpha=0.7)
    ax3.set_title('Lineup Fatigue: Time Since Last Substitution (Sec)', fontsize=12)
    ax3.set_ylabel('Seconds without Sub')
    ax3.set_xlabel('Game Timeline (Events)')
    ax3.grid(True, alpha=0.3)

    # גרף 4: לוגיקת שעון זריקות (Shot Clock Logic)
    ax4 = axes[1, 1]
    sns.histplot(game_df['shot_clock_estimated'], bins=25, kde=True, ax=ax4, color='orange')
    ax4.axvline(24, color='red', linestyle='--', label='24s Limit')
    ax4.axvline(14, color='blue', linestyle='--', label='14s Reset')
    ax4.set_title('Shot Clock Distribution (Logic Check)', fontsize=12)
    ax4.legend()

    # שמירה
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(FIGURES_DIR, exist_ok=True)
    output_path = os.path.join(FIGURES_DIR, f'dashboard_game_{selected_game_id}.png')
    plt.savefig(output_path, dpi=150)
    print(f"✅ Dashboard saved to: {output_path}")
    plt.show() # אם מריצים מתוך IDE

if __name__ == "__main__":
    plot_single_game_dashboard()