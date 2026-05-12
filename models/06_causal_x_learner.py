import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss     
import matplotlib.pyplot as plt
import os
import json

class NBACausalLearner:
    """
    Orchestrates the X-Learner architecture for NBA Timeout Causal Inference.
    Calculates the Conditional Average Treatment Effect (CATE) to isolate 
    the true impact of taking a timeout on stopping an opponent's run.
    """
    
    def __init__(self, data_path: str, target_col: str = 'target_stop_run_180s', treatment_col: str = 'timeout_strategic_weight'):
        self.data_path = data_path
        self.target_col = target_col
        self.treatment_col = treatment_col # The Treatment (T) - Subtask 3.1
        
        # Base Data
        self.X_train, self.X_test = None, None
        self.T_train, self.T_test = None, None
        self.Y_train, self.Y_test = None, None
        
        # Metrics to save for summary
        self.auc = None
        self.ate = None
        
        # M-Models Configuration
        # Stage 1: Propensity Model (Predicts Probability of Timeout)
        self.propensity_model = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
        
        # Stage 2: Outcome Models (Predicts Probability of Stopping Run - Binary)
        self.mu0_model = xgb.XGBClassifier(eval_metric='logloss', random_state=42) # Control (No TO)
        self.mu1_model = xgb.XGBClassifier(eval_metric='logloss', random_state=42) # Treatment (TO)
        
        # Stage 3: CATE Models (Predicts Difference in Probabilities - Continuous)
        self.tau0_model = xgb.XGBRegressor(random_state=42)
        self.tau1_model = xgb.XGBRegressor(random_state=42)

    def load_and_prepare_data(self):
        """Loads data and removes proxy clocks/categorical leakage."""
        print(f"\n--- Processing Target: {self.target_col} ---")
        print(f"Loading data from: {self.data_path}")
        df = pd.read_parquet(self.data_path)
        
        # Absolute purge list to ensure tactical signal only
        leakage_cols = [
            'gameId', 'seconds_remaining', 'orderNumber', 'actionNumber', 'actionId',
            'explosiveness_index', 'teamId', 'possession_id', 'cum_pointsTotal',
            'possession', 'scoreHome', 'scoreAway', 'reboundTotal', 
            'reboundDefensiveTotal', 'reboundOffensiveTotal', 'cum_reboundDefensiveTotal',
            'personId', 'shotActionNumber', 'officialId', 'x', 'y',
            'xLegacy', 'yLegacy', 'jumpBallRecoverdPersonId', 'jumpBallWonPersonId', 
            'jumpBallLostPersonId', 'foulDrawnPersonId', 'is_poss_change', 
            'play_duration', 'isTargetScoreLastPeriod', 'pointsTotal', 
            'turnoverTotal', 'foulPersonalTotal', 'foulTechnicalTotal',
            'shot_clock_estimated' # הוסף לחסימה
        ]
        
        # Identify target columns to drop (all except the current one being analyzed)
        targets_to_drop = [c for c in df.columns if c.startswith('target_') and c != self.target_col]
        cols_to_drop = targets_to_drop + [c for c in leakage_cols if c in df.columns]
        
        print(f"Sanitizing features. Dropping {len(cols_to_drop)} leakage/target columns...")
        df = df.drop(columns=cols_to_drop)
        
        # Ensure treatment is binary (0 or 1)
        df[self.treatment_col] = (df[self.treatment_col] > 0).astype(int)
        
        df = df.dropna(subset=[self.target_col, self.treatment_col])
        
        X = df.drop(columns=[self.target_col, self.treatment_col])
        T = df[self.treatment_col]
        Y = df[self.target_col]
        
        # Split into training and testing sets (80/20)
        self.X_train, self.X_test, self.T_train, self.T_test, self.Y_train, self.Y_test = train_test_split(
            X, T, Y, test_size=0.2, random_state=42, stratify=Y
        )
        print(f"Data ready. Training size: {self.X_train.shape[0]} possessions.")
        print("---------------------------------\n")

    def stage_1_propensity(self):
        
        """Sub-task 3.2: Train Propensity Score model e(x) = P(T=1|X)"""
        print("Stage 1: Training Propensity Model (Coach Psychology)...")
        self.propensity_model.fit(self.X_train, self.T_train)
        
        # Calculate propensity scores (g_x)
        self.g_x_train = self.propensity_model.predict_proba(self.X_train)[:, 1]
        self.g_x_test = self.propensity_model.predict_proba(self.X_test)[:, 1]

        # --- EXPORT FULL FEATURE IMPORTANCE TO JSON ---
        importance_series = pd.Series(self.propensity_model.feature_importances_, index=self.X_train.columns)
        importance_series = importance_series.sort_values(ascending=False)
        
        # Save full list to JSON (unique name per target)
        reports_dir = os.path.join(os.path.dirname(self.data_path), '..', 'reports')
        os.makedirs(reports_dir, exist_ok=True)
        report_path = os.path.join(reports_dir, f'propensity_features_{self.target_col}.json')
        
        with open(report_path, 'w') as f:
            json.dump(importance_series.to_dict(), f, indent=4)
        print(f"Full feature importance report saved to: {report_path}")
        # ----------------------------------------------
        
        # Clip to prevent division by zero in weighting
        self.g_x_train = np.clip(self.g_x_train, 0.01, 0.99)
        self.g_x_test = np.clip(self.g_x_test, 0.01, 0.99)
        
        self.auc = roc_auc_score(self.T_test, self.g_x_test)
        print(f"Propensity AUC: {self.auc:.4f}")

    def stage_2_outcome_modeling(self):
        """Trains models for Control (mu0) and Treatment (mu1)."""
        print("Stage 2: Training Outcome Models (mu0: Control, mu1: Treatment)...")
        # Split Data by Treatment status
        X0, Y0 = self.X_train[self.T_train == 0], self.Y_train[self.T_train == 0]
        X1, Y1 = self.X_train[self.T_train == 1], self.Y_train[self.T_train == 1]
        
        self.mu0_model.fit(X0, Y0)
        self.mu1_model.fit(X1, Y1)

    def stage_3_x_learning(self):
        """Sub-task 3.3: Calculate imputed effects and train CATE models."""
        print("Stage 3: Cross-Learning Imputed Treatment Effects...")
        
        X0, Y0 = self.X_train[self.T_train == 0], self.Y_train[self.T_train == 0]
        X1, Y1 = self.X_train[self.T_train == 1], self.Y_train[self.T_train == 1]
        
        # Imputed treatment effects (Differences in stopping probability)
        # What would have happened if the control group had taken a timeout?
        D0 = self.mu1_model.predict_proba(X0)[:, 1] - Y0 
        # What would have happened if the treated group had NOT taken a timeout?
        D1 = Y1 - self.mu0_model.predict_proba(X1)[:, 1] 
        
        self.tau0_model.fit(X0, D0)
        self.tau1_model.fit(X1, D1)

    def estimate_cate(self, X_eval):
        """Sub-task 3.4: Calculate final CATE using propensity weighting."""
        tau0_pred = self.tau0_model.predict(X_eval)
        tau1_pred = self.tau1_model.predict(X_eval)
        g_x_eval = np.clip(self.propensity_model.predict_proba(X_eval)[:, 1], 0.01, 0.99)
        
        # Final CATE calculation: Weighted average based on propensity score
        cate = g_x_eval * tau0_pred + (1 - g_x_eval) * tau1_pred
        return cate

    def run_pipeline(self):
        """Executes the full Causal Inference flow."""
        self.load_and_prepare_data()
        self.stage_1_propensity()
        self.stage_2_outcome_modeling()
        self.stage_3_x_learning()
        
        print("\nPipeline Complete. Evaluating CATE on Test Set...")
        cate_test = self.estimate_cate(self.X_test)
        
        avg_treatment_effect = np.mean(cate_test)
        self.ate = avg_treatment_effect
        print(f"Average Treatment Effect (ATE) for '{self.target_col}': {avg_treatment_effect * 100:.2f}%")
        print("Positive ATE means timeouts increase the probability of a successful outcome.")
        
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

    # Run pipeline for each target
    for target in targets:
        causal_learner = NBACausalLearner(
            data_path=DATA_PATH,
            target_col=target,
            treatment_col='timeout_strategic_weight'
        )
        cate_results = causal_learner.run_pipeline()
        
        # Save metrics for summary
        summary_results[target] = {
            "ate": float(causal_learner.ate),
            "auc": float(causal_learner.auc)
        }

    # --- Generate Final Reports ---
    print("\n" + "="*55)
    print(f"{'Target':<30} | {'ATE (%)':<10} | {'AUC':<10}")
    print("-" * 55)
    for t, res in summary_results.items():
        print(f"{t:<30} | {res['ate']*100:>8.2f}% | {res['auc']:>8.4f}")
    print("="*55)

    # 1. Save JSON Summary
    summary_path = os.path.join(REPORTS_DIR, 'causal_multi_target_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary_results, f, indent=4)

    # 2. Create Plot
    names = [t.replace('target_', '') for t in targets]
    ates = [res['ate'] * 100 for res in summary_results.values()]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(names, ates, color='coral')
    plt.axhline(0, color='black', linewidth=0.8)
    plt.ylabel('Average Treatment Effect (ATE %)')
    plt.title('Timeout Impact Across Different Strategic Goals')
    plt.xticks(rotation=15)
    
    for bar in bars:
        yval = bar.get_height()
        offset = 0.5 if yval >= 0 else -1.5
        plt.text(bar.get_x() + bar.get_width()/2, yval + offset, f'{yval:.2f}%', ha='center', va='bottom')

    plt.tight_layout()
    plot_path = os.path.join(REPORTS_DIR, 'ate_comparison_profile.png')
    plt.savefig(plot_path)
    
    print(f"\n✅ Summary saved to JSON: {summary_path}")
    print(f"✅ Plot generated: {plot_path}")