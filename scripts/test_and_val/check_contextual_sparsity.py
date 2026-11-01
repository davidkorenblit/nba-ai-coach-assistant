import pandas as pd
import os
import argparse

def normalize_season(season_str: str):
    """
    Normalizes season input like '2024-25' or '2024_25' to both:
    (season_underscore, season_dash), e.g. ('2024_25', '2024-25').
    """
    s = season_str.strip().replace('-', '_')
    if '_' in s:
        parts = s.split('_')
        return f"{parts[0]}_{parts[1]}", f"{parts[0]}-{parts[1]}"
    return s, s.replace('_', '-')

# --- Configuration ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
INTERIM_DIR = os.path.join(CURRENT_DIR, '..', '..', 'data', 'interim')

def check_event_context(df, event_name, text_trigger, col_substring):
    print(f"\n🏀 Testing Event: {event_name.upper()}")
    
    # 1. Filter rows by description trigger
    event_rows = df[df['description'].str.contains(text_trigger, case=False, na=False)]
    
    if event_rows.empty:
        print(f"   ⚠️ No events found for '{text_trigger}'."); return

    print(f"   Found {len(event_rows)} rows containing '{text_trigger}'.")

    # --- PART A: Check specific columns ---
    relevant_cols = [c for c in df.columns if col_substring.lower() in c.lower() and 'remaining' not in c]
    print(f"   [A] Checking specific '{col_substring}' columns:")
    for col in relevant_cols:
        missing_pct = event_rows[col].isna().mean() * 100
        status = "✅ KEEP" if missing_pct < 20 else "🗑️  DROP CANDIDATE"
        if missing_pct > 99: status = "💀 DEAD (100% Empty)"
        print(f"     -> {col:<30} : {missing_pct:6.1f}% missing. {status}")

    # --- PART B: Primary identity columns ---
    print(f"   [B] Checking PRIMARY identity columns:")
    main_cols = ['personId', 'teamTricode']
    
    for col in main_cols:
        if col in df.columns:
            missing = event_rows[col].isna().mean() * 100
            zeros = (event_rows[col] == 0).mean() * 100 if pd.api.types.is_numeric_dtype(event_rows[col]) else 0
            
            status = "✅ PERFECT" if (missing + zeros) < 1 else "❌ BROKEN"
            print(f"     -> {col:<30} : {missing:6.1f}% NaN, {zeros:6.1f}% Zeros. {status}")
            
            if col == 'teamTricode':
                unique_vals = event_rows[col].unique()
                print(f"        🕵️‍♂️ VALUES FOUND: {unique_vals[:10]} {'...' if len(unique_vals)>10 else ''}")

def parse_args():
    parser = argparse.ArgumentParser(description="Check contextual column sparsity.")
    parser.add_argument("--season", type=str, default="2024_25", help="Season string (e.g. 2024_25, 2024-25, 2021_22)")
    parser.add_argument("--file", type=str, default=None, help="Explicit path to level1 base CSV")
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"🕵️‍♂️ Starting QA...")

    if args.file:
        file_path = args.file
    else:
        season_underscore, _ = normalize_season(args.season)
        season_specific = os.path.join(INTERIM_DIR, f'level1_base_{season_underscore}.csv')
        default_file = os.path.join(INTERIM_DIR, 'level1_base.csv')
        if os.path.exists(season_specific):
            file_path = season_specific
        elif season_underscore == '2024_25' and os.path.exists(default_file):
            file_path = default_file
        else:
            file_path = season_specific

    if not os.path.exists(file_path):
        print(f"❌ File not found at {file_path}.")
        return

    print(f"Analyzing file: {file_path}")
    df = pd.read_csv(file_path, low_memory=False)
    
    check_event_context(df, "Assists", "Assist", "assist")
    check_event_context(df, "Timeouts", "Timeout", "teamTricode") 
    check_event_context(df, "Turnovers", "Turnover", "teamTricode")

    print("\n🏁 Analysis Complete.")

if __name__ == "__main__":
    main()
