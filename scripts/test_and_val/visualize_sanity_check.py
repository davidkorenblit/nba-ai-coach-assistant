import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import random
import re

# --- הגדרות נתיבים ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, '..', '..', 'data', 'pureData')
OUTPUT_DIR = os.path.join(CURRENT_DIR, '..', '..', 'docs', 'figures')
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
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Data file not found at {INPUT_FILE}")
        return

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print(f"Loading data...")
    df = pd.read_csv(INPUT_FILE)
    
    unique_games = df['gameId'].unique()
    game_id = random.choice(unique_games)
    game_df = df[df['gameId'] == game_id].copy()
    
    matchup = game_df['matchup'].iloc[0] if 'matchup' in game_df.columns else str(game_id)
    print(f"Generating report for Game: {matchup} (ID: {game_id})")

    # חישובי זמן
    game_df['seconds_left'] = game_df['clock'].apply(parse_time_to_seconds)
    # זמן מצטבר מתחילת המשחק (בשניות)
    game_df['game_time_elapsed'] = (game_df['period'] - 1) * 720 + (720 - game_df['seconds_left'])
    game_df = game_df.sort_values('orderNumber')

    # --- יצירת הגרפים ---

    # גרף 1: זרימת ניקוד
    plt.figure(figsize=(12, 6))
    if 'scoreHome' in game_df.columns and 'scoreAway' in game_df.columns:
        game_df['score_margin'] = game_df['scoreHome'] - game_df['scoreAway']
        sns.lineplot(data=game_df, x='game_time_elapsed', y='score_margin', drawstyle='steps-post')
        plt.title(f'1. Score Margin Flow - {matchup}')
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
        plt.title(f'2. Key Event Counts - {matchup}')
        plt.savefig(os.path.join(OUTPUT_DIR, f"2_event_counts_{game_id}.png"))
        plt.close()

    # גרף 3: בדיקת רצף (Possession)
    plt.figure(figsize=(12, 4))
    if 'teamTricode' in game_df.columns:
        sns.scatterplot(data=game_df, x='game_time_elapsed', y='teamTricode', hue='teamTricode', s=10, legend=False)
        plt.title(f'3. Possession Stream - {matchup}')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(OUTPUT_DIR, f"3_possession_stream_{game_id}.png"))
        plt.close()

    # גרף 4: חילופים - פעילות שחקנים (מלא + ממוין)
    plt.figure(figsize=(14, 12)) # גבוה יותר כדי להכיל את כל השחקנים
    # מיון שחקנים לפי קבוצה כדי שיהיה מסודר בעין
    players_sorted = game_df.sort_values(['teamTricode', 'playerName'])['playerName'].unique()
    stints_df = game_df[game_df['playerName'].isin(players_sorted)]
    
    if not stints_df.empty:
        sns.scatterplot(data=stints_df, x='game_time_elapsed', y='playerName', hue='teamTricode', s=30)
        plt.title(f'4. Player Activity (All Players) - {matchup}')
        plt.xlabel('Time (Seconds)')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(OUTPUT_DIR, f"4_player_stints_{game_id}.png"))
        plt.close()

    # גרף 5: בדיקת שעון (Clock Integrity) - חדש!
    plt.figure(figsize=(12, 6))
    # נצייר את הזמן שעבר מול מספר הפעולה. זה אמור להיות קו עולה ליניארי כמעט מושלם.
    sns.lineplot(data=game_df, x='orderNumber', y='game_time_elapsed')
    plt.title(f'5. Clock Integrity Check (Linearity) - {matchup}')
    plt.xlabel('Event Order Number')
    plt.ylabel('Game Time Elapsed (Seconds)')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, f"5_clock_check_{game_id}.png"))
    plt.close()

    # גרף 6: בדיקת מיקום זריקות (Shot Chart Validation) - חדש!
    plt.figure(figsize=(8, 8)) # מרובע כדי לדמות מגרש
    # נסנן רק זריקות שדה (2 ו-3 נקודות)
    shots_df = game_df[(game_df['isFieldGoal'] == 1) & (game_df['actionType'].isin(['2pt', '3pt']))]
    
    if not shots_df.empty and 'x' in shots_df.columns and 'y' in shots_df.columns:
        sns.scatterplot(data=shots_df, x='x', y='y', hue='actionType', style='actionType', palette='deep', s=60)
        plt.title(f'6. Shot Locations (2pt vs 3pt) - {matchup}')
        # הופכים את ציר Y כי במגרש ה-NBA בדרך כלל 0 זה הסל והמספרים עולים לכיוון החצי, או להפך
        # זה לא קריטי לכיוון, העיקר ההפרדה בין הצבעים
        plt.axis('equal') # כדי שהמגרש לא יימעך
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(OUTPUT_DIR, f"6_shot_locations_{game_id}.png"))
        plt.close()

    print(f"\nDone! 6 Images saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_sanity_report()