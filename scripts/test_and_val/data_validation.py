import pandas as pd
import matplotlib.pyplot as plt
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
DATA_DIR = os.path.join(CURRENT_DIR, '..', '..', 'data', 'pureData')
OUTPUT_DIR = os.path.join(CURRENT_DIR, '..', '..', 'docs', 'figures')

# --- Configuration (NBA Benchmarks) ---
# Numbers represent combined totals for both teams
NBA_BENCHMARKS = {
    'turnover': {'expected': 27, 'tolerance': 5},  # ~13.5 per team
    'steal':    {'expected': 15, 'tolerance': 4},  # ~7.5 per team
    'block':    {'expected': 10, 'tolerance': 3},  # ~5 per team
    'foul':     {'expected': 38, 'tolerance': 6},  # ~19 per team
    '3pt':      {'expected': 70, 'tolerance': 15}, # 35 attempts per team (made+missed)
    'timeout':  {'expected': 14, 'tolerance': 4}   # ~7 per team
}

def generate_context_report(df, season_str: str = "2024_25", input_filename: str = "season_2024_25.csv"):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. Calculate statistics from data
    num_games = df['gameId'].nunique()
    print(f"Analyzing {num_games} games for Season {season_str}...")

    # Count events per game for average calculation
    event_counts = df.groupby(['gameId', 'actionType']).size().unstack(fill_value=0)
    
    # Prepare table data
    table_data = []
    # Headers
    table_data.append(["Event Type", "Your Data (Avg)", "NBA Benchmark", "Status"])
    
    for event, bench in NBA_BENCHMARKS.items():
        if event in event_counts.columns:
            actual_avg = event_counts[event].mean()
            diff = abs(actual_avg - bench['expected'])
            
            # Set status
            if diff <= bench['tolerance']:
                status = "OK (Healthy)"
            elif actual_avg < bench['expected']:
                status = "LOW (Missing Data?)"
            else:
                status = "HIGH (Duplicates?)"
                
            row = [
                event.title(), 
                f"{actual_avg:.1f}", 
                f"{bench['expected']} (+/-{bench['tolerance']})", 
                status
            ]
            table_data.append(row)
        else:
            table_data.append([event.title(), "0.0", str(bench['expected']), "MISSING"])

    # 2. Smart Null checks for critical fields
    table_data.append(["", "", "", ""]) # Spacer
    table_data.append(["Field Name", "Missing %", "Expected Missing", "Integrity"])
    
    # Target fields for inspection
    fields_check = [
        ('stealPlayerName', 0.97, 0.02), # Expected missing in 97% of cases (steals are ~3% of events)
        ('blockPlayerName', 0.98, 0.02),
        ('shotDistance',    0.68, 0.05), # Expected missing in non-shot events
        ('gameId',          0.00, 0.00)  # Must never be missing
    ]
    
    total_rows = len(df)
    
    for col, expected_missing_pct, tolerance in fields_check:
        if col in df.columns:
            missing_pct = df[col].isnull().sum() / total_rows
            diff = abs(missing_pct - expected_missing_pct)
            
            if diff <= tolerance:
                status = "Valid"
            else:
                status = "Suspicious"
                
            table_data.append([
                col, 
                f"{missing_pct:.1%}", 
                f"~{expected_missing_pct:.0%}", 
                status
            ])

    # 3. Render Table
    fig, ax = plt.subplots(figsize=(10, len(table_data) * 0.6))
    ax.axis('tight')
    ax.axis('off')
    
    # Status row colors styling
    table = ax.table(cellText=table_data, loc='center', cellLoc='left')
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # Format header rows
    for (i, j), cell in table.get_celld().items():
        if i == 0 or i == len(NBA_BENCHMARKS) + 2: # Header rows
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#e6e6e6')

    # Save output
    output_path = os.path.join(OUTPUT_DIR, f"0_data_health_context_{season_str}.png")
    plt.title(f"Data Health & Integrity Report - {input_filename}", y=0.99)
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    if season_str == "2024_25":
        legacy_path = os.path.join(OUTPUT_DIR, "0_data_health_context.png")
        plt.savefig(legacy_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"[V] Contextual health report saved to: {output_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Validate data against NBA benchmarks and generate report.")
    parser.add_argument("--season", type=str, default="2024_25", help="Season string (e.g. 2024_25, 2024-25, 2021_22)")
    return parser.parse_args()

def main():
    args = parse_args()
    season_underscore, _ = normalize_season(args.season)
    input_filename = f"season_{season_underscore}.csv"
    input_file = os.path.join(DATA_DIR, input_filename)

    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        return
    
    print(f"Loading data for season {season_underscore} from {input_file}...")
    df = pd.read_csv(input_file, low_memory=False)
    generate_context_report(df, season_str=season_underscore, input_filename=input_filename)

if __name__ == "__main__":
    main()
