import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random

# --- Config ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'interim', 'level1_base.csv')
FIGURES_DIR = os.path.join(BASE_DIR, '..', 'reports', 'figures')

def identify_home_away(df):
    """Simple heuristic to identify Home vs Away team codes based on scoring."""
    # Find a row where scoreHome increased
    home_score_rows = df[df['scoreHome'].diff() > 0]
    if not home_score_rows.empty:
        home_team = home_score_rows.iloc[0]['teamTricode']
    else:
        # Fallback: just take the first code found
        home_team = df['teamTricode'].dropna().unique()[0]
        
    all_teams = df['teamTricode'].dropna().unique()
    away_team = [t for t in all_teams if t != home_team][0]
    
    return home_team, away_team

def plot_extended_dashboard():
    # 1. Load
    if not os.path.exists(DATA_PATH):
        print("❌ Data file not found."); return
    
    df = pd.read_csv(DATA_PATH, low_memory=False)
    
    # 2. Select Game
    game_ids = df['gameId'].unique()
    selected_game_id = random.choice(game_ids)
    game_df = df[df['gameId'] == selected_game_id].copy()
    game_df.sort_values(by=['period', 'seconds_remaining'], ascending=[True, False], inplace=True)
    
    # 3. Identify Teams (V4 Compatible)
    try:
        home_team, away_team = identify_home_away(game_df)
    except:
        print("⚠️ Could not identify teams (game might be empty or single team). skipping.")
        return

    print(f"🎨 Generating Dashboard V4 for: {home_team} (Home) vs {away_team} (Away) - Game {selected_game_id}")

    # --- Plotting (3x2) ---
    fig, axes = plt.subplots(3, 2, figsize=(18, 16))
    fig.suptitle(f'Level 1 Audit (V4 Logic): {home_team} vs {away_team}', fontsize=16)
    x_axis = range(len(game_df))

    # 1. Score Margin
    ax1 = axes[0, 0]
    ax1.plot(x_axis, game_df['score_margin'], color='k', lw=1)
    ax1.fill_between(x_axis, game_df['score_margin'], 0, where=(game_df['score_margin']>0), color='green', alpha=0.3, label=f'{home_team} Leads')
    ax1.fill_between(x_axis, game_df['score_margin'], 0, where=(game_df['score_margin']<0), color='red', alpha=0.3, label=f'{away_team} Leads')
    ax1.set_title('1. Game Flow & Score Margin')
    ax1.legend(loc='upper left')
    ax1.grid(alpha=0.3)

    # 2. Pace
    ax2 = axes[0, 1]
    ax2.plot(x_axis, game_df['possession_id'], color='purple', lw=2)
    ax2.set_title('2. Game Pace (Cumulative Possessions)')
    ax2.grid(alpha=0.3)

    # 3. Timeouts (V4 Logic: Home/Away columns)
    ax3 = axes[1, 0]
    if 'timeouts_remaining_home' in game_df.columns:
        ax3.step(x_axis, game_df['timeouts_remaining_home'], label=f'{home_team} (Home)', where='post', lw=2)
        ax3.step(x_axis, game_df['timeouts_remaining_away'], label=f'{away_team} (Away)', where='post', lw=2)
    ax3.set_title('3. Timeouts Inventory')
    ax3.set_ylim(-0.5, 7.5)
    ax3.legend()
    ax3.grid(alpha=0.3)

    # 4. Cumulative Turnovers (Fix: Use real team codes)
    ax4 = axes[1, 1]
    for team, color in zip([home_team, away_team], ['blue', 'red']):
        # Filter by the REAL team code found in 'teamTricode'
        team_rows = game_df[game_df['teamTricode'] == team]
        
        if not team_rows.empty and 'cum_turnoverTotal' in game_df.columns:
             # Map original indices to x-axis
             indices = [game_df.index.get_loc(i) for i in team_rows.index]
             # Filter only rows where turnover happened (value changed) to avoid clutter, or just plot all points
             # Plotting all points where team was active on offense/defense essentially
             ax4.scatter(indices, team_rows['cum_turnoverTotal'], label=team, s=10, color=color)
    
    ax4.set_title('4. Cumulative Turnovers')
    ax4.legend()
    ax4.grid(alpha=0.3)

    # 5. Fatigue
    ax5 = axes[2, 0]
    ax5.plot(x_axis, game_df['time_since_last_sub'], color='brown', alpha=0.6, lw=1)
    ax5.set_title('5. Lineup Fatigue')
    ax5.grid(alpha=0.3)

    # 6. Shot Clock
    ax6 = axes[2, 1]
    sns.histplot(game_df['shot_clock_estimated'], bins=24, kde=True, ax=ax6, color='orange')
    ax6.axvline(14, color='blue', linestyle='--')
    ax6.set_title('6. Shot Clock Logic Check')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(FIGURES_DIR, exist_ok=True)
    out_path = os.path.join(FIGURES_DIR, f'dashboard_v4_{selected_game_id}.png')
    plt.savefig(out_path, dpi=150)
    print(f"✅ Saved Dashboard: {out_path}")
    plt.show()

if __name__ == "__main__":
    plot_extended_dashboard()