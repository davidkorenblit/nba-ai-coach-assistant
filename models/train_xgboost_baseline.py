import pandas as pd
import numpy as np
import xgboost as xgb
import os
import sys
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt

# --- Config ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, '..', 'data', 'processed')
TARGET_COL = 'target_stop_run_90s' 

class BaselineXGBoost:
    """Baseline XGBoost Model to validate Feature Engineering and establish benchmarks."""
    
    def __init__(self, data_dir: str, target: str):
        self.data_dir = data_dir
        self.target = target
        self.model = None
        self.feature_cols = []
        
    def load_splits(self):
        print(f"Loading Parquet splits from {self.data_dir}...")
        try:
            train_df = pd.read_parquet(os.path.join(self.data_dir, 'train.parquet'))
            val_df = pd.read_parquet(os.path.join(self.data_dir, 'val.parquet'))
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            sys.exit(1)
            
        print(f"Loaded Train: {len(train_df):,} rows | Val: {len(val_df):,} rows")
        return train_df, val_df

    def prepare_xy(self, df: pd.DataFrame):
        """Separates features (X) and target (y), ignoring other targets/metadata and removing leakage."""
        targets_to_drop = [c for c in df.columns if c.startswith('target_')]
        
        # Explicitly drop target leakage, non-ordinal categorical IDs, and proxy clocks
        leakage_cols = [
            'gameId', 'seconds_remaining', 'orderNumber', 'actionNumber', 'actionId',
            'explosiveness_index', 'teamId', 'possession_id', 'cum_pointsTotal',
            'possession', 'scoreHome', 'scoreAway', 'reboundTotal', 
            'reboundDefensiveTotal', 'reboundOffensiveTotal', 'cum_reboundDefensiveTotal', 'personId'
        ]
        
        cols_to_drop = targets_to_drop + [c for c in leakage_cols if c in df.columns]
        
        X = df.drop(columns=cols_to_drop)
        y = df[self.target]
        
        if not self.feature_cols:
            self.feature_cols = X.columns.tolist()
            
        return X, y

    def train(self, X_train, y_train, X_val, y_val):
        print("\nTraining Hardened Baseline XGBoost Model...")
        
        scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
        
        self.model = xgb.XGBClassifier(
            n_estimators=300,          
            learning_rate=0.05,        
            max_depth=4,               
            subsample=0.8,             
            colsample_bytree=0.8,      
            reg_alpha=1.0,             
            reg_lambda=5.0,            
            scale_pos_weight=scale_pos_weight,
            early_stopping_rounds=20,  
            eval_metric='auc',         
            random_state=42,
            n_jobs=-1                  
        )
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=20 
        )
        print("Training Complete.")

    def evaluate(self, X_val, y_val):
        print("\nEvaluating on Validation Set...")
        y_pred = self.model.predict(X_val)
        y_proba = self.model.predict_proba(X_val)[:, 1]
        
        auc = roc_auc_score(y_val, y_proba)
        print(f"ROC-AUC Score: {auc:.4f}")
        
        print("\n--- Classification Report ---")
        print(classification_report(y_val, y_pred))
        
        print("--- Confusion Matrix ---")
        print(confusion_matrix(y_val, y_pred))

    def plot_feature_importance(self):
        print("\nGenerating Feature Importance Plot...")
        importance = self.model.feature_importances_
        
        feature_df = pd.DataFrame({'Feature': self.feature_cols, 'Importance': importance})
        feature_df = feature_df.sort_values(by='Importance', ascending=True).tail(15) 
        
        plt.figure(figsize=(10, 8))
        plt.barh(feature_df['Feature'], feature_df['Importance'], color='coral') 
        plt.xlabel('XGBoost Feature Importance (Gain)')
        plt.title('Top 15 Most Important Features for Stopping a Run (Hardened Model)')
        plt.tight_layout()
        
        plot_path = os.path.join(BASE_DIR, 'feature_importance_baseline.png')
        plt.savefig(plot_path)
        print(f"Feature Importance plot saved to: {plot_path}")

def main():
    pipeline = BaselineXGBoost(PROCESSED_DIR, TARGET_COL)
    
    train_df, val_df = pipeline.load_splits()
    
    X_train, y_train = pipeline.prepare_xy(train_df)
    X_val, y_val = pipeline.prepare_xy(val_df)
    
    pipeline.train(X_train, y_train, X_val, y_val)
    
    pipeline.evaluate(X_val, y_val)
    pipeline.plot_feature_importance()

if __name__ == "__main__":
    main()