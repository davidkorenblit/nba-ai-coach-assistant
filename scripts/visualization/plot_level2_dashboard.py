import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
import numpy as np

# --- Config ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'interim', 'level2_features.csv')
FIGURES_DIR = os.path.join(BASE_DIR, 'reports', 'figures')

def identify_home_away(df):
    """Heuristic to identify team names from the data."""
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

def create_diagnostic_dashboard(game_df, game_id, home_team, away_team):
    """Generates the original 8-plot technical dashboard."""
    fig, axes = plt.subplots(4, 2, figsize=(20, 24))
    fig.suptitle(f'Diagnostic Dashboard: {home_team} vs {away_team} (Game {game_id})', fontsize=22, weight='bold')
    x_axis = game_df.index

    # 1. Explosiveness
    axes[0, 0].plot(x_axis, game_df['explosiveness_index'], color='#e74c3c')
    axes[0, 0].set_title('1. Explosiveness Index', fontsize=14)
    axes[0, 0].axhline(0, color='black', lw=0.5, ls='--')

    # 2. Style / Tempo
    axes[0, 1].plot(x_axis, game_df['style_tempo_rolling'], color='#3498db')
    axes[0, 1].set_title('2. Tempo Shift (Avg Shot Clock Used)', fontsize=14)

    # 3. Instability
    axes[1, 0].plot(x_axis, game_df['instability_index'], color='#9b59b6')
    axes[1, 0].set_title('3. Instability Index (Game Chaos)', fontsize=14)

    # 4. Lineup Fatigue (Timer)
    axes[1, 1].plot(x_axis, game_df['time_since_last_sub'], color='#e67e22', label='Time w/o Sub')
    axes[1, 1].axhline(550, color='red', ls='--')
    axes[1, 1].fill_between(x_axis, game_df['time_since_last_sub'], 550, where=(game_df['time_since_last_sub'] > 550), color='red', alpha=0.3)
    axes[1, 1].set_title('4. Substitution Timer', fontsize=14)

    # 5. Star Resting (Original Binary)
    axes[2, 0].step(x_axis, game_df['is_star_resting'], color='#34495e', where='post')
    axes[2, 0].set_title('5. Star Player Resting (Binary)', fontsize=14)
    axes[2, 0].set_yticks([0, 1])
    axes[2, 0].set_yticklabels(['Playing', 'Resting'])

    # 6. Clutch Time
    clutch_col = 'is_clutch_time' if 'is_clutch_time' in game_df.columns else 'is_crunch_time'
    axes[2, 1].plot(x_axis, game_df['score_margin'], color='gray', alpha=0.3)
    if clutch_col in game_df.columns:
        clutch_series = game_df['score_margin'].copy()
        clutch_series[game_df[clutch_col] == 0] = np.nan
        axes[2, 1].plot(x_axis, clutch_series, color='#c0392b', lw=3, label='Clutch')
    axes[2, 1].set_title('6. Clutch Time Highlight', fontsize=14)

    # 7. Correlation Heatmap
    l2_cols = ['momentum_streak_rolling', 'explosiveness_index', 'style_tempo_rolling', 'instability_index', 'score_margin']
    valid_cols = [c for c in l2_cols if c in game_df.columns]
    sns.heatmap(game_df[valid_cols].corr(), annot=True, cmap='coolwarm', ax=axes[3, 0])
    axes[3, 0].set_title('7. Feature Correlations', fontsize=14)

    axes[3, 1].axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    out_path = os.path.join(FIGURES_DIR, f'dashboard_FULL_DIAGNOSTIC_{game_id}.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"✅ Saved FULL DIAGNOSTIC Dashboard: {out_path}")

def create_strategic_dashboard(game_df, game_id, home_team, away_team):
    """Generates the 3-plot intuitive/strategic summary."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(18, 16), sharex=True)
    fig.suptitle(f'Strategic Insights: {home_team} vs {away_team}', fontsize=20, weight='bold')
    x_axis = game_df.index

    # 1. Accumulated Fatigue (Area Chart)
    home_f = game_df['home_cum_fatigue'] / 60
    away_f = game_df['away_cum_fatigue'] / 60
    ax1.fill_between(x_axis, home_f, color='blue', alpha=0.3, label=f'{home_team} Workload')
    ax1.plot(x_axis, home_f, color='blue', lw=2)
    ax1.fill_between(x_axis, away_f, color='red', alpha=0.3, label=f'{away_team} Workload')
    ax1.plot(x_axis, away_f, color='red', lw=2)
    ax1.set_title('1. Team Workload: Accumulated Minutes on Court', fontsize=14)
    ax1.set_ylabel('Avg Cumulative Minutes')
    ax1.legend()

    # 2. Talent Gravity & Star Gaps
    ax2.step(x_axis, game_df['home_usage_gravity'], color='blue', lw=2, label=f'{home_team} Gravity', where='post')
    ax2.step(x_axis, game_df['away_usage_gravity'], color='red', lw=2, label=f'{away_team} Gravity', where='post')
    star_resting = game_df['is_star_resting'] == 1
    ax2.fill_between(x_axis, 0.6, 1.1, where=star_resting, color='gray', alpha=0.2, label='Star Resting', step='post')
    ax2.set_title('2. Lineup Gravity (Offensive Talent) & Star Rest Windows', fontsize=14)
    ax2.set_ylim(0.65, 1.05)
    ax2.legend()

    # 3. Outcome & Momentum
    ax3.fill_between(x_axis, game_df['score_margin'], 0, color='gray', alpha=0.2, label='Score Margin')
    ax3.plot(x_axis, game_df['momentum_streak_rolling'], color='purple', lw=2, label='Momentum Streak')
    clutch_col = 'is_clutch_time' if 'is_clutch_time' in game_df.columns else 'is_crunch_time'
    if clutch_col in game_df.columns:
        cl_idx = x_axis[game_df[clutch_col] == 1]
        if len(cl_idx) > 0:
            ax3.scatter(cl_idx, [0]*len(cl_idx), color='orange', s=10, label='Clutch Window')
    ax3.set_title('3. Momentum Impact on Score Margin', fontsize=14)
    ax3.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = os.path.join(FIGURES_DIR, f'dashboard_STRATEGIC_SUMMARY_{game_id}.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"✅ Saved STRATEGIC SUMMARY Dashboard: {out_path}")

def main():
    if not os.path.exists(DATA_PATH): return
    df = pd.read_csv(DATA_PATH, low_memory=False)
    
    gid = random.choice(df['gameId'].unique())
    game_df = df[df['gameId'] == gid].copy()
    game_df.sort_values(by=['period', 'seconds_remaining'], ascending=[True, False], inplace=True)
    game_df.reset_index(drop=True, inplace=True)
    
    h_team, a_team = identify_home_away(game_df)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    create_diagnostic_dashboard(game_df, gid, h_team, a_team)
    create_strategic_dashboard(game_df, gid, h_team, a_team)

if __name__ == "__main__":
    main()