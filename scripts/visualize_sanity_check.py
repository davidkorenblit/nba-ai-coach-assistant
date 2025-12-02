import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import random
import re

# --- הגדרות נתיבים יחסיים (הרבה יותר מקצועי) ---
# משיג את המיקום של הקובץ הנוכחי (תיקיית scripts)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# עולים תיקייה אחת למעלה (..) ואז נכנסים ל-data/pureData
DATA_DIR = os.path.join(CURRENT_DIR, '..', 'data', 'pureData')

# עולים תיקייה אחת למעלה (..) ואז נכנסים ל-docs/figures
OUTPUT_DIR = os.path.join(CURRENT_DIR, '..', 'docs', 'figures')

# שם הקובץ (תוודא שיש לך קובץ כזה בתיקייה)
INPUT_FILENAME = "season_2024_25.csv"
INPUT_FILE = os.path.join(DATA_DIR, INPUT_FILENAME)

def parse_time_to_seconds(time_str):
    """Convert PT12M00.00S to seconds remaining in period"""
    try:
        match = re.match(r'PT(\d+)M(\d+\.?\d*)S', str(time_str))
        if match:
            minutes = float(match.group(1))
            seconds = float(match.group(2))
            return minutes * 60 + seconds
        return 0
    except:
        return 0

def generate_sanity_report():
    # בדיקה שהקובץ קיים לפני שמתחילים
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Data file not found at {INPUT_FILE}")
        return

    # 0. הקמת תיקיית פלט בתוך docs
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")
        
    # 1. טעינת דאטא
    print(f"Loading data...")
    df = pd.read_csv(INPUT_FILE)
    
    # 2. בחירת משחק אקראי
    unique_games = df['gameId'].unique()
    game_id = random.choice(unique_games)
    game_df = df[df['gameId'] == game_id].copy()
    
    matchup = game_df['matchup'].iloc[0] if 'matchup' in game_df.columns else str(game_id)
    print(f"Generating report for Game: {matchup} (ID: {game_id})")

    # 3. עיבוד זמן
    game_df['seconds_left'] = game_df['clock'].apply(parse_time_to_seconds)
    game_df['game_time_elapsed'] = (game_df['period'] - 1) * 720 + (720 - game_df['seconds_left'])
    game_df = game_df.sort_values('orderNumber')

    # --- יצירת הגרפים ---

    # גרף 1: זרימת ניקוד
    plt.figure(figsize=(12, 6))
    if 'scoreHome' in game_df.columns and 'scoreAway' in game_df.columns:
        game_df['score_margin'] = game_df['scoreHome'] - game_df['scoreAway']
        sns.lineplot(data=game_df, x='game_time_elapsed', y='score_margin', drawstyle='steps-post')
        plt.title(f'Score Margin Flow - {matchup}')
        plt.xlabel('Time (Seconds)')
        plt.ylabel('Score Margin')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(OUTPUT_DIR, f"1_score_flow_{game_id}.png"))
        plt.close()

    # גרף 2: התפלגות טריגרים
    plt.figure(figsize=(10, 6))
    triggers = ['3pt', 'turnover', 'steal', 'block', 'foul']
    trigger_df = game_df[game_df['actionType'].isin(triggers)]
    if not trigger_df.empty:
        sns.countplot(data=trigger_df, x='actionType', hue='actionType', palette='viridis')
        plt.title(f'Key Event Counts - {matchup}')
        plt.savefig(os.path.join(OUTPUT_DIR, f"2_event_counts_{game_id}.png"))
        plt.close()

    # גרף 3: בדיקת רצף
    plt.figure(figsize=(12, 4))
    if 'teamTricode' in game_df.columns:
        sns.scatterplot(data=game_df, x='game_time_elapsed', y='teamTricode', hue='teamTricode', s=10, legend=False)
        plt.title(f'Possession Stream - {matchup}')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(OUTPUT_DIR, f"3_possession_stream_{game_id}.png"))
        plt.close()

    # גרף 4: חילופים (פעילות שחקנים)
    plt.figure(figsize=(14, 8))
    top_players = game_df['playerName'].value_counts().head(20).index
    stints_df = game_df[game_df['playerName'].isin(top_players)]
    
    if not stints_df.empty:
        sns.scatterplot(data=stints_df, x='game_time_elapsed', y='playerName', hue='teamTricode', s=20)
        plt.title(f'Player Activity - {matchup}')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(OUTPUT_DIR, f"4_player_stints_{game_id}.png"))
        plt.close()

    print(f"\nDone! Images saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_sanity_report()