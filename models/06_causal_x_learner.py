import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_squared_error
import matplotlib.pyplot as plt
import os
import json

class NBACausalLearner:
    def __init__(self, data_path: str, target_col: str = 'target_stop_run_90s', treatment_col: str = 'timeout_strategic_weight'):
        self.data_path = data_path
        self.target_col = target_col
        self.treatment_col = treatment_col
        
        self.X_train, self.X_test = None, None
        self.T_train, self.T_test = None, None
        self.Y_train, self.Y_test = None, None
        
        self.auc = None
        self.ate = None
        
        # מודל הנטייה נשאר קלסיפייר (כי הטיפול הוא כן/לא פסק זמן)
        self.propensity_model = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
        
        # מודלי התוצאה הפכו ל-Regressor כי הטרגט שלנו עכשיו רציף!
        self.mu0_model = xgb.XGBRegressor(eval_metric='rmse', random_state=42)
        self.mu1_model = xgb.XGBRegressor(eval_metric='rmse', random_state=42)
        
        self.tau0_model = xgb.XGBRegressor(eval_metric='rmse', random_state=42)
        self.tau1_model = xgb.XGBRegressor(eval_metric='rmse', random_state=42)

    def load_and_prepare_data(self):
        print(f"\n--- Processing Target: {self.target_col} ---")
        
        # טעינת Train ו-Test אמיתיים
        test_path = self.data_path.replace('train.parquet', 'test.parquet')
        print(f"Loading TRAIN from: {self.data_path}")
        print(f"Loading TEST from: {test_path}")
        
        train_df = pd.read_parquet(self.data_path)
        test_df = pd.read_parquet(test_path)
        
        # קריאת הפיצ'רים מה-JSON (שהוא כבר סטרילי מכל דליפה!)
        metadata_path = os.path.join(os.path.dirname(self.data_path), 'split_metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        feature_cols = [c for c in metadata['features'] if c in train_df.columns]
        
        # --- סינון גארבג' טיים ---
        if 'is_garbage_time' in train_df.columns:
            train_df = train_df[train_df['is_garbage_time'] == 0]
            test_df = test_df[test_df['is_garbage_time'] == 0]
        
        # פונקציית עזר להכנת הסטים
        def prepare_split(df):
            # מוודאים שאין NaN במטרות או בטיפול
            df = df.dropna(subset=[self.target_col, self.treatment_col])
            
            # בינאריזציה של הטיפול (פסק זמן נלקח או לא)
            T = (df[self.treatment_col] > 0).astype(int)
            Y = df[self.target_col]
            
            # לוקחים רק את הפיצ'רים הנקיים (ומוודאים שהטיפול לא בתוכם)
            X_cols = [c for c in feature_cols if c != self.treatment_col]
            X = df[X_cols]
            
            return X, T, Y

        self.X_train, self.T_train, self.Y_train = prepare_split(train_df)
        self.X_test, self.T_test, self.Y_test = prepare_split(test_df)
        
        print(f"Data ready. Clean Features: {len(self.X_train.columns)}. Train: {self.X_train.shape[0]} | Test: {self.X_test.shape[0]}")
        print("---------------------------------\n")

    def stage_1_propensity(self):
        print("Stage 1: Training Propensity Model...")
        self.propensity_model.fit(self.X_train, self.T_train)
        
        self.g_x_train = self.propensity_model.predict_proba(self.X_train)[:, 1]
        self.g_x_test = self.propensity_model.predict_proba(self.X_test)[:, 1]

        importance_series = pd.Series(self.propensity_model.feature_importances_, index=self.X_train.columns)
        importance_series = importance_series.sort_values(ascending=False)
        
        print("\n🚨 TOP 5 LEAKAGE SUSPECTS (Propensity Model) 🚨")
        print(importance_series.head(5))
        
        reports_dir = os.path.join(os.path.dirname(self.data_path), '..', 'reports')
        os.makedirs(reports_dir, exist_ok=True)
        report_path = os.path.join(reports_dir, f'propensity_features_{self.target_col}.json')
        
        with open(report_path, 'w') as f:
            json.dump(importance_series.to_dict(), f, indent=4)
        
        self.g_x_train = np.clip(self.g_x_train, 0.01, 0.99)
        self.g_x_test = np.clip(self.g_x_test, 0.01, 0.99)
        
        self.auc = roc_auc_score(self.T_test, self.g_x_test)
        print(f"Propensity AUC: {self.auc:.4f}")

    def stage_2_outcome_modeling(self):
        print("Stage 2: Training Outcome Models (mu0, mu1 as Regressors)...")
        X0, Y0 = self.X_train[self.T_train == 0], self.Y_train[self.T_train == 0]
        X1, Y1 = self.X_train[self.T_train == 1], self.Y_train[self.T_train == 1]
        
        self.mu0_model.fit(X0, Y0)
        self.mu1_model.fit(X1, Y1)

    def stage_3_x_learning(self):
        print("Stage 3: Cross-Learning Imputed Treatment Effects...")
        X0, Y0 = self.X_train[self.T_train == 0], self.Y_train[self.T_train == 0]
        X1, Y1 = self.X_train[self.T_train == 1], self.Y_train[self.T_train == 1]
        
        # התאמה ל-Regressor: משתמשים ב-predict() ולא ב-predict_proba()
        D0 = self.mu1_model.predict(X0) - Y0 
        D1 = Y1 - self.mu0_model.predict(X1) 
        
        self.tau0_model.fit(X0, D0)
        self.tau1_model.fit(X1, D1)

    def estimate_cate(self, X_eval):
        tau0_pred = self.tau0_model.predict(X_eval)
        tau1_pred = self.tau1_model.predict(X_eval)
        g_x_eval = np.clip(self.propensity_model.predict_proba(X_eval)[:, 1], 0.01, 0.99)
        
        # --- BUG-04: תיקון הנוסחה של ה-X-Learner (המשקולות עכשיו נכונות) ---
        cate = (1 - g_x_eval) * tau0_pred + g_x_eval * tau1_pred
        return cate

    def plot_uplift_validation(self, cate_test, output_dir):
        eval_df = pd.DataFrame({
            'cate_score': cate_test,
            'treatment': self.T_test.values,
            'outcome': self.Y_test.values
        })
        
        eval_df['cate_bucket'] = pd.qcut(eval_df['cate_score'], q=5, labels=['Lowest 20%', 'Low-Mid', 'Medium', 'Mid-High', 'Top 20%'])
        
        # ממוצע התוצאות הרציפות לכל דלי
        actual_rates = eval_df.groupby(['cate_bucket', 'treatment'], observed=True)['outcome'].mean().unstack()
        actual_rates.columns = ['No_TO', 'With_TO']
        
        actual_rates['Actual_Uplift'] = actual_rates['With_TO'] - actual_rates['No_TO']
        
        plt.figure(figsize=(10, 6))
        actual_rates['Actual_Uplift'].plot(kind='bar', color='mediumseagreen', edgecolor='black')
        plt.title(f'Actual Timeout Effectiveness by Model Prediction (Uplift)\n{self.target_col}')
        plt.ylabel('Actual Impact Improvement (With TO - No TO)')
        plt.xlabel('Model Prediction (CATE Score Buckets)')
        plt.axhline(0, color='red', linestyle='dashed', linewidth=1)
        plt.xticks(rotation=15)
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, f'uplift_validation_{self.target_col}.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"📈 Uplift Validation Graph saved to: {plot_path}")

    def run_pipeline(self):
        self.load_and_prepare_data()
        self.stage_1_propensity()
        self.stage_2_outcome_modeling()
        self.stage_3_x_learning()
        
        print("\nPipeline Complete. Evaluating CATE on Test Set...")
        cate_test = self.estimate_cate(self.X_test)
        
        avg_treatment_effect = np.mean(cate_test)
        self.ate = avg_treatment_effect
        print(f"Average Treatment Effect (ATE) for '{self.target_col}': {avg_treatment_effect:+.2f} points/impact")
        
        reports_dir = os.path.join(os.path.dirname(self.data_path), '..', 'reports')
        self.plot_uplift_validation(cate_test, reports_dir)
        
        return cate_test

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    DATA_PATH = os.path.join(base_dir, 'data', 'processed', 'train.parquet')
    REPORTS_DIR = os.path.join(base_dir, 'reports')
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    print(f"Working with absolute path: {DATA_PATH}")
    
    targets = [
        'target_stop_run_90s', 
        'target_reverse_trend_180s', 
        'target_improve_margin_90s', 
        'target_improve_margin_180s'
    ]
    
    summary_results = {}

    for target in targets:
        causal_learner = NBACausalLearner(
            data_path=DATA_PATH,
            target_col=target,
            treatment_col='timeout_strategic_weight'
        )
        cate_results = causal_learner.run_pipeline()
        
        summary_results[target] = {
            "ate": float(causal_learner.ate),
            "auc": float(causal_learner.auc)
        }

    print("\n" + "="*55)
    print(f"{'Target':<30} | {'ATE (Impact)':<12} | {'Propensity AUC':<10}")
    print("-" * 55)
    for t, res in summary_results.items():
        print(f"{t:<30} | {res['ate']:>+12.4f} | {res['auc']:>8.4f}")
    print("="*55)

    summary_path = os.path.join(REPORTS_DIR, 'causal_multi_target_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary_results, f, indent=4)

    names = [t.replace('target_', '') for t in targets]
    ates = [res['ate'] for res in summary_results.values()]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(names, ates, color='coral')
    plt.axhline(0, color='black', linewidth=0.8)
    plt.ylabel('Average Treatment Effect (ATE Impact)')
    plt.title('Timeout Impact Across Different Strategic Goals (Continuous)')
    plt.xticks(rotation=15)
    
    for bar in bars:
        yval = bar.get_height()
        offset = 0.05 if yval >= 0 else -0.1
        plt.text(bar.get_x() + bar.get_width()/2, yval + offset, f'{yval:+.2f}', ha='center', va='bottom')

    plt.tight_layout()
    plot_path = os.path.join(REPORTS_DIR, 'ate_comparison_profile.png')
    plt.savefig(plot_path)
    plt.close()
    
    print(f"\n✅ Summary saved to JSON: {summary_path}")
    print(f"✅ Plot generated: {plot_path}")