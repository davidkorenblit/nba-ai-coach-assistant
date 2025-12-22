import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random

# --- הגדרות ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'interim', 'level1_base.csv')
FIGURES_DIR = os.path.join(BASE_DIR, '..', 'reports', 'figures')

def plot_extended_dashboard():
    # 1. טעינה
    if not os.path.exists(DATA_PATH):
        print("❌ Data file not found."); return
    
    df = pd.read_csv(DATA_PATH, low_memory=False)
    
    # 2. בחירת משחק
    game_ids = df['gameId'].unique()
    selected_game_id = random.choice(game_ids)
    game_df = df[df['gameId'] == selected_game_id].copy()
    game_df.sort_values(by=['period', 'seconds_remaining'], ascending=[True, False], inplace=True)
    
    # זיהוי קבוצות
    try:
        teams = [c.replace('timeouts_remaining_', '') for c in game_df.columns if 'timeouts_remaining_' in c]
        team_a, team_b = teams[0], teams[1]
    except:
        print("Could not identify teams from columns."); return

    print(f"🎨 Generating Extended Dashboard for: {team_a} vs {team_b} (Game {selected_game_id})")

    # --- יצירת דאשבורד (3x2) ---
    fig, axes = plt.subplots(3, 2, figsize=(18, 16))
    fig.suptitle(f'Level 1 Full Feature Audit: {team_a} vs {team_b}', fontsize=16)
    x_axis = range(len(game_df))

    # 1. Score Margin (Momentum)
    ax1 = axes[0, 0]
    ax1.plot(x_axis, game_df['score_margin'], color='k', lw=1)
    ax1.fill_between(x_axis, game_df['score_margin'], 0, where=(game_df['score_margin']>0), color='green', alpha=0.3)
    ax1.fill_between(x_axis, game_df['score_margin'], 0, where=(game_df['score_margin']<0), color='red', alpha=0.3)
    ax1.set_title('1. Game Flow & Score Margin')
    ax1.grid(alpha=0.3)

    # 2. Possession Pace (Cumulative Possessions)
    ax2 = axes[0, 1]
    ax2.plot(x_axis, game_df['possession_id'], color='purple', lw=2)
    ax2.set_title('2. Game Pace (Cumulative Possessions)')
    ax2.set_ylabel('Possession Count')
    ax2.grid(alpha=0.3)

    # 3. Inventory (Timeouts)
    ax3 = axes[1, 0]
    ax3.step(x_axis, game_df[f'timeouts_remaining_{team_a}'], label=team_a, where='post', lw=2)
    ax3.step(x_axis, game_df[f'timeouts_remaining_{team_b}'], label=team_b, where='post', lw=2)
    ax3.set_title('3. Timeouts Inventory (Should decrease)')
    ax3.set_ylim(-0.5, 7.5)
    ax3.legend()
    ax3.grid(alpha=0.3)

    # 4. Cumulative Counters (Turnovers) - NEW
    ax4 = axes[1, 1]
    # אנו צריכים לסכום איבודים לפי קבוצה. נשתמש ב-cumsum שכבר קיים בדאטה אם יש, או נחשב לצורך הגרף
    # נניח ש-cum_turnoverTotal קיים אבל הוא פר שורה. ננסה להפריד:
    # בגלל שהמבנה שטוח, נציג פשוט את העמודה הקיימת (היא כבר מצטברת לקבוצה הרלוונטית בשורה)
    # כדי לצייר קו יפה, נצטרך לפצל:
    for team in [team_a, team_b]:
        # מסננים שורות שבהן הקבוצה הזו פעלה
        team_rows = game_df[game_df['teamTricode'] == team]
        # מציירים נקודות בזמן שהאירוע קרה
        if not team_rows.empty and 'cum_turnoverTotal' in game_df.columns:
             # שימוש באינדקס המקורי כציר X
             indices = [game_df.index.get_loc(i) for i in team_rows.index]
             ax4.scatter(indices, team_rows['cum_turnoverTotal'], label=team, s=10)
    
    ax4.set_title('4. Cumulative Turnovers (Event Triggered)')
    ax4.legend()
    ax4.grid(alpha=0.3)

    # 5. Fatigue (Sub Timer)
    ax5 = axes[2, 0]
    ax5.plot(x_axis, game_df['time_since_last_sub'], color='brown', alpha=0.6, lw=1)
    ax5.set_title('5. Lineup Fatigue (Time since substitution)')
    ax5.set_ylabel('Seconds')
    ax5.grid(alpha=0.3)

    # 6. Shot Clock Distribution
    ax6 = axes[2, 1]
    sns.histplot(game_df['shot_clock_estimated'], bins=24, kde=True, ax=ax6, color='orange')
    ax6.axvline(14, color='blue', linestyle='--')
    ax6.set_title('6. Shot Clock Logic Check')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(FIGURES_DIR, exist_ok=True)
    out_path = os.path.join(FIGURES_DIR, f'dashboard_v2_{selected_game_id}.png')
    plt.savefig(out_path, dpi=150)
    print(f"✅ Saved: {out_path}")
    plt.show()

if __name__ == "__main__":
    plot_extended_dashboard()