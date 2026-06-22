import pandas as pd
import numpy as np
import joblib
import os
import json

def analyze_sweet_spot_all_targets():
    base_dir = r"C:\Users\david\finalPro"
    print("🎯 Extracting Presentation Metrics for ALL TARGETS...\n" + "="*60)
    
    data_path = os.path.join(base_dir, 'data', 'processed', 'test.parquet')
    models_dir = os.path.join(base_dir, 'models', 'saved_models')
    reports_dir = os.path.join(base_dir, 'reports')
    summary_path = os.path.join(reports_dir, 'causal_multi_target_summary.json')
    
    # 1. שליפת אחוזי הדיוק (AUC) לכל המטרות
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            summary = json.load(f)
            print("📊 1. Propensity AUC (Accuracy of recognizing Timeout spots):")
            for target, res in summary.items():
                print(f"   - {target}: {res.get('auc', 0)*100:.1f}%")
            print("\n" + "="*60 + "\n")
    
    targets = [
        'target_stop_run_90s', 
        'target_reverse_trend_180s', 
        'target_improve_margin_90s', 
        'target_improve_margin_180s'
    ]
    
    # בודקים שני מצבי קיצון: החמישון העליון (Top 10%) והמאיון ה-95 (Top 5%)
    percentiles_to_test = [90, 95] 
    
    try:
        df_full = pd.read_parquet(data_path)
    except Exception as e:
        print(f"❌ Error loading parquet: {e}")
        return

    for target_col in targets:
        print(f"🔬 Analyzing Sweet Spot for: {target_col}")
        
        try:
            # טעינת מודלים
            p_model = joblib.load(os.path.join(models_dir, f'propensity_{target_col}.joblib'))
            t0_model = joblib.load(os.path.join(models_dir, f'tau0_{target_col}.joblib'))
            t1_model = joblib.load(os.path.join(models_dir, f'tau1_{target_col}.joblib'))
            
            df = df_full.dropna(subset=[target_col, 'timeout_strategic_weight']).copy()
            
            expected_features = p_model.get_booster().feature_names
            X = df.reindex(columns=expected_features, fill_value=0)
            
            # חישוב CATE
            g_x = np.clip(p_model.predict_proba(X)[:, 1], 0.01, 0.99)
            cate = (1 - g_x) * t0_model.predict(X) + g_x * t1_model.predict(X)
            
            eval_df = pd.DataFrame({
                'cate_score': cate,
                'treatment': (df['timeout_strategic_weight'] > 0).astype(int),
                'outcome': df[target_col]
            })
            
            # ריצה על האחוזונים
            for p in percentiles_to_test:
                threshold = np.percentile(eval_df['cate_score'], p)
                top_cases = eval_df[eval_df['cate_score'] >= threshold]
                
                with_to = top_cases[top_cases['treatment'] == 1]['outcome'].mean()
                no_to = top_cases[top_cases['treatment'] == 0]['outcome'].mean()
                actual_uplift = with_to - no_to
                
                total_critical = len(top_cases)
                ignored_critical = len(top_cases[top_cases['treatment'] == 0])
                ignored_percentage = (ignored_critical / total_critical) * 100 if total_critical > 0 else 0
                
                print(f"   ▶ Top {100-p}% (Threshold > {p}th percentile):")
                print(f"     * Cases in this zone: {total_critical:,}")
                print(f"     * Actual Impact Diff: {actual_uplift:+.2f} points (Timeout vs No Timeout)")
                print(f"     * Coach Miss Rate: {ignored_percentage:.1f}% ({ignored_critical:,} out of {total_critical:,} ignored)")
            
        except Exception as e:
            print(f"   ❌ Error analyzing {target_col}: {e}")
        print("-" * 60)

if __name__ == "__main__":
    analyze_sweet_spot_all_targets()