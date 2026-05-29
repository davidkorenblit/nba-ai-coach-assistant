import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def run_multi_target_sweep(reports_dir):
    targets = [
        'target_stop_run_90s', 
        'target_reverse_trend_180s', 
        'target_improve_margin_90s', 
        'target_improve_margin_180s'
    ]
    
    for target in targets:
        csv_path = os.path.join(reports_dir, f'timeout_recommendations_report_{target}.csv')
        
        # מוודא שהקובץ קיים כדי שלא יקרוס אם סקריפט 7 לא סיים לרוץ על כולם
        if not os.path.exists(csv_path):
            print(f"\n⚠️ File not found for {target}, skipping... ({csv_path})")
            continue
            
        print(f"\n" + "="*50)
        print(f"--- 🔍 Running Threshold Sweep for {target} ---")
        df = pd.read_csv(csv_path)
        
        # חילוץ העמודות הנכונות (כולל המרה בטוחה למספרים לעמודת העונש)
        cate_scores = df['predicted_cate'].values 
        treatments = df['actual_treatment'].values 
        penalties = pd.to_numeric(df['target_danger_penalty'], errors='coerce').fillna(0).values
        
        results = []
        
        # סריקה של כל האחוזונים מ-80 עד 99
        for p in range(80, 100):
            threshold = np.percentile(cate_scores, p)
            
            alerts = (cate_scores >= threshold)
            ignored_alerts = alerts & (treatments == 0)
            total_ignored = ignored_alerts.sum()
            
            # סופר קריסות (כל מה שמעל 0.8 ייחשב קריסה)
            punished = ignored_alerts & (penalties >= 0.8) 
            total_punished = punished.sum()
            
            hit_rate = (total_punished / total_ignored * 100) if total_ignored > 0 else 0
            
            results.append({
                'Percentile': p,
                'Alerts': alerts.sum(),
                'Ignored': total_ignored,
                'Crashes': total_punished,
                'Hit_Rate_%': round(hit_rate, 3)
            })
            
        df_results = pd.DataFrame(results)
        print(df_results.to_string(index=False))
        
        # ציור ושמירת הגרף לטרגט הספציפי
        plt.figure(figsize=(10, 5))
        plt.plot(df_results['Percentile'], df_results['Hit_Rate_%'], marker='o', color='crimson', linewidth=2)
        plt.title(f'Threshold Sweep: Hit Rate vs. Percentile\n{target}')
        plt.xlabel('Percentile Threshold')
        plt.ylabel('Hit Rate (Crashes / Ignored Alerts) %')
        plt.grid(True, linestyle='--', alpha=0.6)
        
        plot_path = os.path.join(reports_dir, f'threshold_sweep_{target}.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"📊 Sweep Graph saved to: {plot_path}")

if __name__ == "__main__":
    # נתיב תיקיית הדו"חות שלך
    REPORTS_DIR = r"C:\Users\david\finalPro\reports"
    
    run_multi_target_sweep(REPORTS_DIR)