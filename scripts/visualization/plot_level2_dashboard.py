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
    # Ensure chronological order for plotting
    game_df.sort_values(by=['period', 'seconds_remaining'], ascending=[True, False], inplace=True)
    game_df.reset_index(drop=True, inplace=True)
    
    home_team, away_team = identify_home_away(game_df)
    print(f"🎨 Generating Level 2 Dashboard for Game {selected_game_id}: {home_team} vs {away_team}")

    # --- Plotting (4x2 Grid) ---
    fig, axes = plt.subplots(4, 2, figsize=(20, 20))
    fig.suptitle(f'Level 2 Features Analysis: {home_team} vs {away_team}', fontsize=20, weight='bold')
    
    x_axis = game_df.index

    # 1. Smart Momentum (The Fix)
    ax1 = axes[0, 0]
    ax1.plot(x_axis, game_df['momentum_streak_rolling'], color='#2ecc71', lw=2)
    ax1.fill_between(x_axis, game_df['momentum_streak_rolling'], 0, color='#2ecc71', alpha=0.2)
    ax1.set_title('1. Smart Momentum Streak (Rolling 10 events)', fontsize=14)
    ax1.set_ylabel('Momentum Score')
    ax1.grid(alpha=0.3)

    # 2. Explosiveness
    ax2 = axes[0, 1]
    ax2.plot(x_axis, game_df['explosiveness_index'], color='#e74c3c', lw=1.5)
    ax2.set_title('2. Explosiveness Index (Score Slope)', fontsize=14)
    ax2.set_ylabel('Margin Change Rate')
    ax2.axhline(0, color='black', lw=0.5, linestyle='--')
    ax2.grid(alpha=0.3)

    # 3. Style / Tempo
    ax3 = axes[1, 0]
    ax3.plot(x_axis, game_df['style_tempo_rolling'], color='#3498db', lw=2)
    ax3.set_title('3. Style Shift (Avg Shot Clock Used)', fontsize=14)
    ax3.set_ylabel('Seconds')
    ax3.grid(alpha=0.3)

    # 4. Instability
    ax4 = axes[1, 1]
    ax4.plot(x_axis, game_df['instability_index'], color='#9b59b6', lw=1.5)
    ax4.set_title('4. Instability Index (Event Density)', fontsize=14)
    ax4.set_ylabel('Time per 10 Events')
    ax4.grid(alpha=0.3)

    # 5. Fatigue (Binary)
    ax5 = axes[2, 0]
    ax5.step(x_axis, game_df['is_high_fatigue'], color='#e67e22', where='post', lw=2)
    ax5.fill_between(x_axis, game_df['is_high_fatigue'], step='post', color='#e67e22', alpha=0.3)
    ax5.set_title('5. Shared Fatigue (>5 mins w/o sub)', fontsize=14)
    ax5.set_yticks([0, 1])
    ax5.set_yticklabels(['Fresh', 'Fatigued'])
    ax5.grid(alpha=0.3)

    # 6. Star Resting (Binary)
    ax6 = axes[2, 1]
    ax6.step(x_axis, game_df['is_star_resting'], color='#34495e', where='post', lw=2)
    ax6.fill_between(x_axis, game_df['is_star_resting'], step='post', color='#34495e', alpha=0.3)
    ax6.set_title('6. Star Player Resting', fontsize=14)
    ax6.set_yticks([0, 1])
    ax6.set_yticklabels(['Playing', 'Resting'])
    ax6.grid(alpha=0.3)

    # 7. Crunch Time + Score Margin context
    ax7 = axes[3, 0]
    # Plot Margin in background
    ax7.plot(x_axis, game_df['score_margin'], color='gray', alpha=0.3, label='Score Margin')
    # Highlight Crunch Time
    crunch_mask = game_df['is_crunch_time'] == 1
    if crunch_mask.any():
        # Create a dummy series for plotting that is NaN where not crunch time
        crunch_series = game_df['score_margin'].copy()
        crunch_series[~crunch_mask] = float('nan')
        ax7.plot(x_axis, crunch_series, color='#c0392b', lw=3, label='Crunch Time!')
    
    ax7.set_title('7. Crunch Time Zones (on Score Margin)', fontsize=14)
    ax7.legend()
    ax7.grid(alpha=0.3)

    # 8. Correlations Heatmap (Bonus)
    ax8 = axes[3, 1]
    # Select only numeric L2 features
    l2_cols = ['momentum_streak_rolling', 'explosiveness_index', 'style_tempo_rolling', 
               'instability_index', 'score_margin']
    corr = game_df[l2_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax8, square=True)
    ax8.set_title('8. Feature Correlations (In-Game)', fontsize=14)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    os.makedirs(FIGURES_DIR, exist_ok=True)
    out_path = os.path.join(FIGURES_DIR, f'dashboard_level2_{selected_game_id}.png')
    plt.savefig(out_path, dpi=150)
    print(f"✅ Saved Level 2 Dashboard: {out_path}")
    plt.show()

if __name__ == "__main__":
    plot_level2_dashboard()