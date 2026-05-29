import pandas as pd
import numpy as np
import joblib
import os
import json

class InferenceEngine:
    def __init__(self, data_path: str, models_dir: str):
        self.data_path = data_path
        self.models_dir = models_dir

    def run_inference(self, target_col):
        # 1. טעינת מודלים
        p_model = joblib.load(os.path.join(self.models_dir, f'propensity_{target_col}.joblib'))
        t0_model = joblib.load(os.path.join(self.models_dir, f'tau0_{target_col}.joblib'))
        t1_model = joblib.load(os.path.join(self.models_dir, f'tau1_{target_col}.joblib'))

        # 2. טעינת נתונים
        test_path = self.data_path.replace('train.parquet', 'test.parquet')
        df = pd.read_parquet(test_path)
        df = df.dropna(subset=[target_col, 'timeout_strategic_weight'])
        
        # --- כאן התיקון הקריטי: אכיפת סכמה ---
        # אנחנו שואבים את השמות שהמודל "זוכר" מהאימון
        expected_features = p_model.get_booster().feature_names
        
        # reindex מסדר את העמודות בדיוק לפי מה שהמודל מצפה. 
        # אם חסרה עמודה - הוא ישלים 0. אם יש עמודה מיותרת - הוא יתעלם.
        X = df.reindex(columns=expected_features, fill_value=0)
        
        # 3. הסקה
        g_x = np.clip(p_model.predict_proba(X)[:, 1], 0.01, 0.99)
        cate = (1 - g_x) * t0_model.predict(X) + g_x * t1_model.predict(X)
        
        # 4. ניתוח
        results = X.copy()
        results['cate'] = cate
        results['actual_treatment'] = (df['timeout_strategic_weight'] > 0).astype(int)
        results['outcome'] = df[target_col]
        
        threshold = np.percentile(cate, 95.0)
        alerts = results[results['cate'] >= threshold].copy()
        
        complied = alerts[alerts['actual_treatment'] == 1]
        ignored = alerts[alerts['actual_treatment'] == 0]
        
        cost_of_ignoring = ignored['outcome'].mean() - complied['outcome'].mean()
        
        print(f"\nTarget: {target_col} | Ignored Avg: {ignored['outcome'].mean():.4f} | Complied Avg: {complied['outcome'].mean():.4f}")
        print(f"👉 OPPORTUNITY COST: {cost_of_ignoring:+.4f}")
        
        return alerts

if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    engine = InferenceEngine(
        os.path.join(base, 'data', 'processed', 'train.parquet'),
        os.path.join(base, 'models', 'saved_models')
    )
    
    for t in ['target_stop_run_90s', 'target_reverse_trend_180s', 'target_improve_margin_90s', 'target_improve_margin_180s']:
        try:
            engine.run_inference(t)
        except Exception as e:
            print(f"Error on {t}: {e}")