import pandas as pd
import matplotlib.pyplot as plt
import os

class ExplainabilityDashboard:
    def __init__(self, reports_dir: str):
        self.reports_dir = reports_dir
        self.csv_path = os.path.join(reports_dir, 'timeout_recommendations_report.csv')

    def run_dashboard(self):
        print("\n" + "="*65)
        print("🏀 TACTICAL TIMEOUT SCOUT REPORT (TOP 10 EXTREME ALERTS)")
        print("="*65)

        if not os.path.exists(self.csv_path):
            print(f"❌ File not found: {self.csv_path}. Please run script 07 first.")
            return

        df = pd.read_csv(self.csv_path)

        if df.empty:
            print("⚠️ No alerts found in the report.")
            return

        # Ensure sorted by highest CATE (Impact)
        df = df.sort_values(by='predicted_cate', ascending=False).reset_index(drop=True)

        # --- PART 1: Top 10 Text Report (The Scout Report) ---
        top_10 = df.head(10)
        for idx, row in top_10.iterrows():
            cate = row['predicted_cate'] * 100
            period = int(row.get('period', 0))
            margin = row.get('score_margin', 0)
            streak = row.get('momentum_streak_rolling', 0)
            h_fatigue = row.get('home_cum_fatigue', 0)
            a_fatigue = row.get('away_cum_fatigue', 0)
            
            # Determine game context
            if margin < 0:
                status = f"Down by {abs(margin)}"
            elif margin > 0:
                status = f"Up by {margin}"
            else:
                status = "Tied game"

            print(f"🚨 ALERT #{idx+1} | EXPECTED IMPACT (CATE): +{cate:.1f}%")
            print(f"   Context: Quarter {period}. Team is {status}.")
            print(f"   Momentum: Opponent is on a {streak:.1f} run.")
            print(f"   Fatigue Index: Home {h_fatigue:.1f} | Away {a_fatigue:.1f}")
            print("-" * 65)

        # --- PART 2: Visual Case Studies (Top 3 Subplots) ---
        print("\n📊 Generating Visual Case Studies for Top 3 events...")
        top_3 = df.head(3)
        
        fig, axes = plt.subplots(1, 3, figsize=(16, 6))
        fig.suptitle('Top 3 Timeout Alerts: Tactical Feature Breakdown', fontsize=16, fontweight='bold')

        features_to_plot = ['score_margin', 'momentum_streak_rolling', 'home_cum_fatigue', 'away_cum_fatigue']
        feature_names = ['Score\nMargin', 'Opponent\nRun', 'Home\nFatigue', 'Away\nFatigue']

        for i, (idx, row) in enumerate(top_3.iterrows()):
            ax = axes[i]
            values = [row.get(f, 0) for f in features_to_plot]
            
            # Dynamic colors: Red for negative/stress factors, Blue for positive
            colors = ['#ff9999' if v < 0 else '#66b3ff' for v in values]
            
            bars = ax.barh(feature_names, values, color=colors, edgecolor='black', alpha=0.8)
            ax.set_title(f"Alert #{i+1} (Impact: {row['predicted_cate']*100:.1f}%)", fontweight='bold')
            ax.axvline(0, color='black', linewidth=1)
            
            # Add data labels
            for bar in bars:
                width = bar.get_width()
                label_x_pos = width + (2 if width >= 0 else -2)
                ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.1f}', 
                        va='center', ha='left' if width >= 0 else 'right', fontweight='bold')

            # Expand limits so text doesn't get cut off
            ax.set_xlim(min(values) - max(10, abs(min(values)*0.5)), max(values) + max(10, abs(max(values)*0.5)))

        plt.tight_layout()
        plot_path = os.path.join(self.reports_dir, 'top_3_alerts_breakdown.png')
        plt.savefig(plot_path)
        print(f"✅ Visual Case Studies saved to: {plot_path}")
        print("Ready for the grand finale when you are.")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    reports_dir = os.path.join(base_dir, 'reports')
    
    dashboard = ExplainabilityDashboard(reports_dir)
    dashboard.run_dashboard()