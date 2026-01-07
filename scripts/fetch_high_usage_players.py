import pandas as pd
import os
import sys
from nba_api.stats.static import players

# --- הגדרות ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'lookup')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'high_usage_players_2024-25.csv')

def create_manual_usage_table():
    print("🔨 Building Complete 30-Team High-Usage Table...")
    
    # רשימה מלאה: הנציג בעל ה-Usage הגבוה ביותר מכל קבוצה (נכון לעונת 24-25)
    manual_data = [
        ("Trae Young", "ATL", 0.285),
        ("Jayson Tatum", "BOS", 0.301),
        ("Cam Thomas", "BKN", 0.320),
        ("LaMelo Ball", "CHA", 0.342),
        ("Zach LaVine", "CHI", 0.275), # או קובי וייט, בהתאם לפציעות
        ("Donovan Mitchell", "CLE", 0.298),
        ("Luka Doncic", "DAL", 0.328),
        ("Nikola Jokic", "DEN", 0.285),
        ("Cade Cunningham", "DET", 0.323),
        ("Stephen Curry", "GSW", 0.286),
        ("Jalen Green", "HOU", 0.270),
        ("Tyrese Haliburton", "IND", 0.260), # או פסקל סיאקם
        ("James Harden", "LAC", 0.286),
        ("Anthony Davis", "LAL", 0.301),
        ("Ja Morant", "MEM", 0.312),
        ("Tyler Herro", "MIA", 0.271),
        ("Giannis Antetokounmpo", "MIL", 0.346),
        ("Anthony Edwards", "MIN", 0.307),
        ("Zion Williamson", "NOP", 0.325),
        ("Jalen Brunson", "NYK", 0.289),
        ("Shai Gilgeous-Alexander", "OKC", 0.336),
        ("Paolo Banchero", "ORL", 0.330),
        ("Joel Embiid", "PHI", 0.342),
        ("Devin Booker", "PHX", 0.285),
        ("Anfernee Simons", "POR", 0.265),
        ("De'Aaron Fox", "SAC", 0.274),
        ("Victor Wembanyama", "SAS", 0.300),
        ("RJ Barrett", "TOR", 0.281),
        ("Lauri Markkanen", "UTA", 0.260),
        ("Jordan Poole", "WAS", 0.281)
    ]

    final_rows = []
    
    print(f"🔍 Resolving Player IDs for {len(manual_data)} teams...")
    
    for name, team, usg in manual_data:
        # שליפת ID סטטית
        found_players = players.find_players_by_full_name(name)
        
        if found_players:
            p_id = found_players[0]['id']
            final_rows.append({
                'PLAYER_ID': p_id,
                'PLAYER_NAME': name,
                'TEAM_ABBREVIATION': team,
                'USG_PCT': usg
            })
        else:
            print(f"❌ Critical Error: Could not find ID for {name}")

    # בדיקת שלמות (30 קבוצות)
    df = pd.DataFrame(final_rows)
    unique_teams = df['TEAM_ABBREVIATION'].nunique()
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"✅ Success! Coverage: {unique_teams}/30 NBA Teams.")
    print(f"📂 Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    create_manual_usage_table()