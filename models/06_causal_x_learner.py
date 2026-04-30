import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss     
import matplotlib.pyplot as plt
import os

class NBACausalLearner:
    """
    Orchestrates the X-Learner architecture for NBA Timeout Causal Inference.
    Calculates the Conditional Average Treatment Effect (CATE) to isolate 
    the true impact of taking a timeout on stopping an opponent's run.
    """
    
    def __init__(self, data_path: str, target_col: str = 'target_90s', treatment_col: str = 'is_timeout'):
        self.data_path = data_path
        self.target_col = target_col
        self.treatment_col = treatment_col # The Treatment (T) - Subtask 3.1
        
        # Base Data
        self.X_train, self.X_test = None, None
        self.T_train, self.T_test = None, None
        self.Y_train, self.Y_test = None, None
        
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
        print("Loading data and sanitizing features...")
        df = pd.read_parquet(self.data_path)
        
        # Absolute purge list from our Baseline Hardening
        leakage_cols = [
            'gameId', 'seconds_remaining', 'orderNumber', 'actionNumber', 'actionId',
            'explosiveness_index', 'teamId', 'possession_id', 'cum_pointsTotal',
            'possession', 'scoreHome', 'scoreAway', 'reboundTotal', 
            'reboundDefensiveTotal', 'reboundOffensiveTotal', 'cum_reboundDefensiveTotal',
            'personId'
        ]
        
        targets_to_drop = [c for c in df.columns if c.startswith('target_') and c != self.target_col]
        cols_to_drop = targets_to_drop + [c for c in leakage_cols if c in df.columns]
        
        df = df.drop(columns=cols_to_drop)
        df = df.dropna(subset=[self.target_col, self.treatment_col])
        
        X = df.drop(columns=[self.target_col, self.treatment_col])
        T = df[self.treatment_col]
        Y = df[self.target_col]
        
        self.X_train, self.X_test, self.T_train, self.T_test, self.Y_train, self.Y_test = train_test_split(
            X, T, Y, test_size=0.2, random_state=42, stratify=Y
        )
        print(f"Data ready. Training size: {self.X_train.shape[0]} possessions.")

    def stage_1_propensity(self):
        """Sub-task 3.2: Train Propensity Score model e(x) = P(T=1|X)"""
        print("Stage 1: Training Propensity Model...")
        self.propensity_model.fit(self.X_train, self.T_train)
        
        # Calculate propensity scores (g_x)
        self.g_x_train = self.propensity_model.predict_proba(self.X_train)[:, 1]
        self.g_x_test = self.propensity_model.predict_proba(self.X_test)[:, 1]
        
        # Clip to prevent division by zero in weighting
        self.g_x_train = np.clip(self.g_x_train, 0.01, 0.99)
        self.g_x_test = np.clip(self.g_x_test, 0.01, 0.99)
        print(f"Propensity AUC: {roc_auc_score(self.T_test, self.g_x_test):.4f}")

    def stage_2_outcome_modeling(self):
        """Trains models for Control (mu0) and Treatment (mu1)."""
        print("Stage 2: Training Outcome Models (mu0, mu1)...")
        # Split Data by Treatment
        X0, Y0 = self.X_train[self.T_train == 0], self.Y_train[self.T_train == 0]
        X1, Y1 = self.X_train[self.T_train == 1], self.Y_train[self.T_train == 1]
        
        self.mu0_model.fit(X0, Y0)
        self.mu1_model.fit(X1, Y1)

    def stage_3_x_learning(self):
        """Sub-task 3.3: Calculate imputed effects and train CATE models."""
        print("Stage 3: Cross-Learning Imputed Treatment Effects...")
        
        X0, Y0 = self.X_train[self.T_train == 0], self.Y_train[self.T_train == 0]
        X1, Y1 = self.X_train[self.T_train == 1], self.Y_train[self.T_train == 1]
        
        # Imputed treatment effects (Differences in probability)
        # What if the control group got treated?
        D0 = self.mu1_model.predict_proba(X0)[:, 1] - Y0 
        # What if the treated group wasn't treated?
        D1 = Y1 - self.mu0_model.predict_proba(X1)[:, 1] 
        
        self.tau0_model.fit(X0, D0)
        self.tau1_model.fit(X1, D1)

    def estimate_cate(self, X_eval):
        """Sub-task 3.4: Calculate final CATE using propensity weighting."""
        tau0_pred = self.tau0_model.predict(X_eval)
        tau1_pred = self.tau1_model.predict(X_eval)
        g_x_eval = np.clip(self.propensity_model.predict_proba(X_eval)[:, 1], 0.01, 0.99)
        
        # Final CATE calculation: Weighted average based on propensity
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
        print(f"Average Treatment Effect (ATE): {avg_treatment_effect * 100:.2f}%")
        print("Positive ATE means timeouts generally increase the probability of stopping a run.")
        
        return cate_test

if __name__ == "__main__":
    # Adjust paths as needed based on your project structure
    DATA_PATH = os.path.join('..', 'data', 'processed', 'train.parquet')
    
    # Initialize and run
    causal_learner = NBACausalLearner(data_path=DATA_PATH)
    cate_results = causal_learner.run_pipeline()