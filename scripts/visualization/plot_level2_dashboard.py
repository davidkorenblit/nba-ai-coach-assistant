import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random

# --- Config ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'interim', 'level2_features.csv')
FIGURES_DIR = os.path.join(BASE_DIR, '..', 'reports', 'figures')

def identify_home_away(df):
    """Basic heuristic to identify teams."""
    try:
        home_score_rows = df[df['scoreHome'].diff() > 0]
        if not home_score_rows.empty:
            home_team = home_score_rows.iloc[0]['teamTricode']
        else:
            home_team = df['teamTricode'].dropna().unique()[0]
        
        all_teams = df['teamTricode'].dropna().unique()
        away_teams = [t for t in all_teams if t != home_team]
        away_team = away_teams[0] if away_teams else "OPP"
        return home_team, away_team
    except:
        return "Home", "Away"

def plot_level2_dashboard():
    # 1. Load Data
    if not os.path.exists(DATA_PATH):
        print("❌ Data file not found."); return
    
    df = pd.read_csv(DATA_PATH, low_memory=False)
    
    # 2. Select Random Game
    game_ids = df['gameId'].unique()
    selected_game_id = random.choice(game_ids)
    
    game_df = df[df['gameId'] == selected_game_id].copy()
    # Ensure chronological order
    game_df.sort_values(by=['period', 'seconds_remaining'], ascending=[True, False], inplace=True)
    game_df.reset_index(drop=True, inplace=True)
    
    home_team, away_team = identify_home_away(game_df)
    print(f"🎨 Generating Level 2 Dashboard (Clean) for Game {selected_game_id}: {home_team} vs {away_team}")

    # --- Plotting (4x2 Grid - 7 Plots) ---
    fig, axes = plt.subplots(4, 2, figsize=(20, 24))
    fig.suptitle(f'Level 2 Advanced Metrics: {home_team} vs {away_team}', fontsize=22, weight='bold')
    
    x_axis = game_df.index

    # 1. Explosiveness (Score Slope)
    ax1 = axes[0, 0]
    ax1.plot(x_axis, game_df['explosiveness_index'], color='#e74c3c', lw=1.5)
    ax1.set_title('1. Explosiveness Index (Sudden Runs)', fontsize=14)
    ax1.set_ylabel('Points Delta Change')
    ax1.axhline(0, color='black', lw=0.5, linestyle='--')
    ax1.grid(alpha=0.3)

    # 2. Style / Tempo
    ax2 = axes[0, 1]
    ax2.plot(x_axis, game_df['style_tempo_rolling'], color='#3498db', lw=2)
    ax2.set_title('2. Tempo Shift (Avg Shot Clock Used)', fontsize=14)
    ax2.set_ylabel('Seconds')
    ax2.grid(alpha=0.3)

    # 3. Instability
    ax3 = axes[1, 0]
    ax3.plot(x_axis, game_df['instability_index'], color='#9b59b6', lw=1.5)
    ax3.set_title('3. Instability Index (Game Chaos)', fontsize=14)
    ax3.set_ylabel('Time per 10 Events')
    ax3.grid(alpha=0.3)

    # 4. Lineup Fatigue (UPDATED: Continuous with Threshold)
    ax4 = axes[1, 1]
    # Plot the raw seconds counter
    ax4.plot(x_axis, game_df['time_since_last_sub'], color='#e67e22', lw=1.5, label='Time w/o Sub')
    # Add Threshold Line
    ax4.axhline(350, color='red', linestyle='--', lw=2, label='Threshold (350s)')
    # Highlight danger zone
    ax4.fill_between(x_axis, game_df['time_since_last_sub'], 350,
                     where=(game_df['time_since_last_sub'] > 350),
                     color='red', alpha=0.3)
    
    ax4.set_title('4. Lineup Fatigue Accumulation', fontsize=14)
    ax4.set_ylabel('Seconds Without Substitution')
    ax4.legend(loc='upper left')
    ax4.grid(alpha=0.3)

    # 5. Star Resting
    ax5 = axes[2, 0]
    ax5.step(x_axis, game_df['is_star_resting'], color='#34495e', where='post', lw=2)
    ax5.fill_between(x_axis, game_df['is_star_resting'], step='post', color='#34495e', alpha=0.3)
    ax5.set_title('5. Star Player Resting', fontsize=14)
    ax5.set_yticks([0, 1])
    ax5.set_yticklabels(['Playing', 'Resting'])
    ax5.grid(alpha=0.3)

    # 6. Clutch Time (Renamed & Logic Updated)
    ax6 = axes[2, 1]
    # Background Margin
    ax6.plot(x_axis, game_df['score_margin'], color='gray', alpha=0.3, label='Score Margin')
    
    # Check for new column name, fallback to old if needed
    clutch_col = 'is_clutch_time' if 'is_clutch_time' in game_df.columns else 'is_crunch_time'
    
    if clutch_col in game_df.columns:
        clutch_mask = game_df[clutch_col] == 1
        if clutch_mask.any():
            clutch_series = game_df['score_margin'].copy()
            clutch_series[~clutch_mask] = float('nan')
            ax6.plot(x_axis, clutch_series, color='#c0392b', lw=3, label='Clutch Time')
    
    ax6.set_title('6. Clutch Time (Last 5 mins, +/- 5 pts)', fontsize=14)
    ax6.legend()
    ax6.grid(alpha=0.3)

    # 7. Correlations Heatmap
    ax7 = axes[3, 0]
    l2_cols = ['momentum_streak_rolling', 'explosiveness_index', 'style_tempo_rolling', 
               'instability_index', 'score_margin']
    # Filter only existing columns
    valid_cols = [c for c in l2_cols if c in game_df.columns]
    corr = game_df[valid_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax7, square=True)
    ax7.set_title('7. Feature Correlations', fontsize=14)

    # 8. Empty Slot (Clean up)
    axes[3, 1].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    os.makedirs(FIGURES_DIR, exist_ok=True)
    out_path = os.path.join(FIGURES_DIR, f'dashboard_level2_clean_{selected_game_id}.png')
    plt.savefig(out_path, dpi=150)
    print(f"✅ Saved Clean Dashboard: {out_path}")
    plt.show()

if __name__ == "__main__":
    plot_level2_dashboard()