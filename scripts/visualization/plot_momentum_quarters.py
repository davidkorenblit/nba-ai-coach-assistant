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
    """Identifies Home and Away team codes."""
    try:
        # Heuristic: Home team usually appears in scoreHome increments
        home_score_rows = df[df['scoreHome'].diff() > 0]
        if not home_score_rows.empty:
            home_team = home_score_rows.iloc[0]['teamTricode']
        else:
            home_team = df['teamTricode'].mode()[0]
        
        all_teams = df['teamTricode'].dropna().unique()
        away_teams = [t for t in all_teams if t != home_team]
        away_team = away_teams[0] if away_teams else "OPP"
        return home_team, away_team
    except:
        return "HOME", "AWAY"

def calculate_split_momentum(df, home_team, away_team):
    """
    Splits the unified 'event_momentum_val' into separate rolling momentum 
    tracks for Home and Away teams.
    """
    # 1. Split events by team
    df['home_event_val'] = df.apply(lambda x: x['event_momentum_val'] if x['teamTricode'] == home_team else 0, axis=1)
    df['away_event_val'] = df.apply(lambda x: x['event_momentum_val'] if x['teamTricode'] == away_team else 0, axis=1)
    
    # 2. Calculate Rolling Sum (Momentum Buildup) for each team separately
    # Window size 10 events (adjustable)
    WINDOW = 10
    df['home_momentum'] = df['home_event_val'].rolling(window=WINDOW, min_periods=1).sum().fillna(0)
    df['away_momentum'] = df['away_event_val'].rolling(window=WINDOW, min_periods=1).sum().fillna(0)
    
    return df

def plot_momentum_by_quarter():
    # 1. Load
    if not os.path.exists(DATA_PATH):
        print("❌ Data file not found."); return
    df = pd.read_csv(DATA_PATH, low_memory=False)
    
    # 2. Pick Game
    game_ids = df['gameId'].unique()
    selected_game_id = random.choice(game_ids)
    game_df = df[df['gameId'] == selected_game_id].copy()
    
    # Important: Ensure Chronological Order for rolling calc
    game_df.sort_values(by=['period', 'seconds_remaining'], ascending=[True, False], inplace=True)
    game_df.reset_index(drop=True, inplace=True)
    
    # 3. Prep Data
    home_team, away_team = identify_home_away(game_df)
    game_df = calculate_split_momentum(game_df, home_team, away_team)
    
    print(f"🎨 Generating Momentum Split Report for: {home_team} vs {away_team} (Game {selected_game_id})")

    # 4. Plot - 4 Quarters Layout
    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=False, sharey=True)
    fig.suptitle(f'Momentum Wars: {home_team} (Green) vs {away_team} (Blue) - Game {selected_game_id}', fontsize=16, weight='bold')

    periods = [1, 2, 3, 4]
    
    for i, p in enumerate(periods):
        ax = axes[i]
        period_df = game_df[game_df['period'] == p]
        
        if period_df.empty:
            ax.text(0.5, 0.5, "No Data for Period", ha='center')
            continue

        # X-Axis: We want 'Time Elapsed in Quarter' (0 to 720) instead of raw index for clarity
        # But for simplicity in PBP flow, we can use the index within the period
        x_axis = range(len(period_df))
        
        # Plot Home
        ax.plot(x_axis, period_df['home_momentum'], color='green', label=f'{home_team}', lw=2, alpha=0.8)
        ax.fill_between(x_axis, period_df['home_momentum'], 0, color='green', alpha=0.1)
        
        # Plot Away
        ax.plot(x_axis, period_df['away_momentum'], color='blue', label=f'{away_team}', lw=2, alpha=0.8)
        ax.fill_between(x_axis, period_df['away_momentum'], 0, color='blue', alpha=0.1)
        
        ax.set_title(f'Quarter {p}', loc='left', fontsize=12, weight='bold')
        ax.set_ylabel('Momentum Intensity')
        ax.grid(True, alpha=0.2)
        
        if i == 0:
            ax.legend(loc='upper right')

    plt.xlabel('Event Sequence (Play-by-Play Order)')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    os.makedirs(FIGURES_DIR, exist_ok=True)
    out_path = os.path.join(FIGURES_DIR, f'momentum_split_{selected_game_id}.png')
    plt.savefig(out_path, dpi=150)
    print(f"✅ Saved Graph: {out_path}")
    plt.show()

if __name__ == "__main__":
    plot_momentum_by_quarter()