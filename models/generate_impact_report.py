import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def generate_presentation_graphs():
    # 1. Define base directories and target output folder
    base_dir = r"C:\Users\david\finalPro"
    # Create the new directory inside 'reports'
    output_dir = os.path.join(base_dir, 'reports', 'important_graphs')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Directory ready: {output_dir}")

    # 2. Hardcoded data extracted from the model's output (Top 5% Threshold)
    data = {
        'Target': ['Stop Run 90s', 'Reverse Trend 180s', 'Improve Margin 90s', 'Improve Margin 180s'],
        'AUC': [79.4, 79.4, 79.4, 79.4],
        'Top5_Impact_Diff': [0.79, -0.23, 0.43, -0.37],
        'Top5_Miss_Rate': [97.9, 99.0, 98.5, 99.3]
    }
    
    df = pd.DataFrame(data)

    # 3. Calculate Macro Impact (Seasonal Expected Wins)
    # Constants based on strategic ROI assumptions
    CRITICAL_CASES_PER_GAME = 1.5 
    GAMES_PER_SEASON = 82
    POINTS_PER_WIN = 30
    
    # Formula: (Impact * cases_per_game * games_per_season) / points_per_win
    df['Expected_Season_Wins'] = (df['Top5_Impact_Diff'] * CRITICAL_CASES_PER_GAME * GAMES_PER_SEASON) / POINTS_PER_WIN

    # Helper function to assign colors based on positive/negative values
    def get_colors(values):
        return ['#2ca02c' if v > 0 else '#d62728' for v in values]

    # --- GRAPH 1: Expected Season Wins (Macro Level) ---
    plt.figure(figsize=(10, 6))
    bars1 = plt.bar(df['Target'], df['Expected_Season_Wins'], color=get_colors(df['Expected_Season_Wins']), edgecolor='black')
    plt.title('Expected Season Wins per Strategy (Top 5% Sweet Spot)', fontsize=14, fontweight='bold')
    plt.ylabel('Extra Wins', fontsize=12)
    plt.axhline(0, color='black', linewidth=1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add data labels
    for bar in bars1:
        yval = bar.get_height()
        offset = 0.1 if yval >= 0 else -0.2
        plt.text(bar.get_x() + bar.get_width()/2, yval + offset, f'{yval:+.1f}', ha='center', va='bottom' if yval >= 0 else 'top', fontweight='bold')
    
    plt.savefig(os.path.join(output_dir, '1_macro_season_wins.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # --- GRAPH 2: Actual Impact Diff Per Possession (Micro Level) ---
    plt.figure(figsize=(10, 6))
    bars2 = plt.bar(df['Target'], df['Top5_Impact_Diff'], color=get_colors(df['Top5_Impact_Diff']), edgecolor='black')
    plt.title('Actual Impact Difference by Target (Points per Possession)', fontsize=14, fontweight='bold')
    plt.ylabel('Impact Difference (Points)', fontsize=12)
    plt.axhline(0, color='black', linewidth=1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars2:
        yval = bar.get_height()
        offset = 0.02 if yval >= 0 else -0.04
        plt.text(bar.get_x() + bar.get_width()/2, yval + offset, f'{yval:+.2f}', ha='center', va='bottom' if yval >= 0 else 'top', fontweight='bold')
        
    plt.savefig(os.path.join(output_dir, '2_micro_impact_diff.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # --- GRAPH 3: Coach Miss Rate (The Intuition Gap) ---
    plt.figure(figsize=(10, 6))
    bars3 = plt.bar(df['Target'], df['Top5_Miss_Rate'], color='#1f77b4', edgecolor='black')
    plt.title('Coach Miss Rate in Critical Spots (Top 5% threshold)', fontsize=14, fontweight='bold')
    plt.ylabel('Miss Rate (%)', fontsize=12)
    plt.ylim(90, 100) # Focusing on the extreme high miss rate to emphasize the gap
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars3:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f'{yval:.1f}%', ha='center', va='bottom', fontweight='bold')
        
    plt.savefig(os.path.join(output_dir, '3_coach_miss_rates.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print("✅ All 3 graphs generated and saved successfully!")

if __name__ == "__main__":
    generate_presentation_graphs()