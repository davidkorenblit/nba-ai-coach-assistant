import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
import numpy as np

# --- Config ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'interim', 'level1_base.csv')
FIGURES_DIR = os.path.join(BASE_DIR, 'reports', 'figures')

def identify_home_away(df):
    home_score_rows = df[df['scoreHome'].diff() > 0]
    if not home_score_rows.empty:
        home_team = home_score_rows.iloc[0]['teamTricode']
    else:
        home_team = df['teamTricode'].dropna().unique()[0]
    all_teams = df['teamTricode'].dropna().unique()
    away_team = [t for t in all_teams if t != home_team][0]
    return home_team, away_team

def plot_extended_dashboard():
    if not os.path.exists(DATA_PATH):
        print(f"❌ Data file not found at {DATA_PATH}.")
        return
    
    df = pd.read_csv(DATA_PATH, low_memory=False)
    
    # בחירת משחק
    game_ids = df['gameId'].unique()
    selected_game_id = random.choice(game_ids)
    
    # סינון ומיון חובה כדי שהציר יהיה כרונולוגי
    game_df = df[df['gameId'] == selected_game_id].copy()
    game_df.sort_values(by=['period', 'seconds_remaining'], ascending=[True, False], inplace=True)
    game_df.reset_index(drop=True, inplace=True) # קריטי לסנכרון הגרף
    
    home_team, away_team = identify_home_away(game_df)
    conf_score = game_df['lineup_confidence'].mean() * 100

    print(f"🎨 Generating Dashboard v5: {home_team} vs {away_team} (Reliability: {conf_score:.1f}%)")

    fig, axes = plt.subplots(3, 2, figsize=(18, 16))
    fig.suptitle(f'Level 1 Tactical Audit: {home_team} vs {away_team}\nGame ID: {selected_game_id}', fontsize=18, weight='bold')
    x_axis = game_df.index

    # 1. Score Margin
    axes[0, 0].plot(x_axis, game_df['score_margin'], color='k', lw=1)
    axes[0, 0].fill_between(x_axis, game_df['score_margin'], 0, where=(game_df['score_margin']>0), color='blue', alpha=0.2, label=f'{home_team} Lead')
    axes[0, 0].fill_between(x_axis, game_df['score_margin'], 0, where=(game_df['score_margin']<0), color='red', alpha=0.2, label=f'{away_team} Lead')
    axes[0, 0].set_title('1. Score Margin Flow')
    axes[0, 0].legend()

    # 2. Possession Pace
    axes[0, 1].plot(x_axis, game_df['possession_id'], color='purple')
    axes[0, 1].set_title('2. Possession Count (Pace)')

    # 3. Strategic Timeouts (התיקון כאן)
    ax3 = axes[1, 0]
    ax3.step(x_axis, game_df['timeouts_remaining_home'], label=f'{home_team}', color='blue', where='post')
    ax3.step(x_axis, game_df['timeouts_remaining_away'], label=f'{away_team}', color='red', where='post')
    
    # סימון פסקי זמן אסטרטגיים
    to_events = game_df[game_df['timeout_strategic_weight'] > 0]
    if not to_events.empty:
        # גדלים לפי משקל: 1=40, 2=100, 3=250
        sizes = to_events['timeout_strategic_weight'] * 80
        ax3.scatter(to_events.index, game_df.loc[to_events.index, 'timeouts_remaining_home'], 
                    s=sizes, color='gold', edgecolors='black', label='Strategic TO', zorder=5)
    
    ax3.set_title('3. Timeouts Inventory (Gold Dot = Strategic Weight)')
    ax3.set_ylim(-0.5, 7.5)
    ax3.legend()

    # 4. Cumulative Turnovers
    axes[1, 1].plot(x_axis, game_df['cum_turnoverTotal'], color='orange')
    axes[1, 1].set_title('4. Cumulative Game Turnovers')

    # 5. Fatigue & Confidence
    ax5 = axes[2, 0]
    ax5.fill_between(x_axis, 0, 720, where=(game_df['lineup_confidence'] == 1), color='green', alpha=0.1, label='Official API')
    ax5.fill_between(x_axis, 0, 720, where=(game_df['lineup_confidence'] == 0), color='orange', alpha=0.1, label='Inferred Logic')
    ax5.plot(x_axis, game_df['time_since_last_sub'], color='brown', lw=1.5)
    ax5.set_title('5. Lineup Fatigue vs Data Source Reliability')
    ax5.set_ylabel('Seconds Since Last Sub')
    ax5.legend(loc='upper left')

    # 6. Shot Clock Dist
    sns.histplot(game_df['shot_clock_estimated'], bins=24, kde=True, ax=axes[2, 1], color='teal')
    axes[2, 1].axvline(14, color='red', linestyle='--')
    axes[2, 1].set_title('6. Shot Clock Distribution (14s Reset Check)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    os.makedirs(FIGURES_DIR, exist_ok=True)
    out_path = os.path.join(FIGURES_DIR, f'dashboard_v5_tactical_{selected_game_id}.png')
    plt.savefig(out_path, dpi=150)
    print(f"✅ Saved Fixed Dashboard: {out_path}")
    plt.show()

if __name__ == "__main__":
    plot_extended_dashboard()