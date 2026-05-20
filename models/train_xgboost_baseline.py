import pandas as pd
import numpy as np
import xgboost as xgb
import os
import sys
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# --- Config ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, '..', 'data', 'processed')
TARGET_COL = 'target_stop_run_90s' 

class BaselineXGBoostRegressor:
    """Baseline XGBoost Model (Regression) to validate Feature Engineering."""
    
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
        """Separates features (X) and target (y), using the CLEAN JSON metadata."""
        
        metadata_path = os.path.join(self.data_dir, 'split_metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        clean_features = [c for c in metadata['features'] if c in df.columns]
        
        # סינון גארבג' טיים ליישור קו עם מודל ההסקה
        if 'is_garbage_time' in df.columns:
            df = df[df['is_garbage_time'] == 0]
            
        df = df.dropna(subset=[self.target])
        
        X = df[clean_features]
        y = df[self.target]
        
        if not self.feature_cols:
            self.feature_cols = clean_features
            
        return X, y

    def train(self, X_train, y_train, X_val, y_val):
        print("\nTraining Hardened Baseline XGBoost Regressor...")
        
        self.model = xgb.XGBRegressor(
            n_estimators=300,          
            learning_rate=0.05,        
            max_depth=4,               
            subsample=0.8,             
            colsample_bytree=0.8,      
            reg_alpha=1.0,             
            reg_lambda=5.0,            
            early_stopping_rounds=20,  
            eval_metric='rmse',        
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
        
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        
        print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")
        print(f"MAE (Mean Absolute Error): {mae:.4f}")
        print(f"R² Score: {r2:.4f}")

    def plot_feature_importance(self):
        print("\nGenerating Feature Importance Plot...")
        importance = self.model.feature_importances_
        
        feature_df = pd.DataFrame({'Feature': self.feature_cols, 'Importance': importance})
        feature_df = feature_df.sort_values(by='Importance', ascending=True).tail(15) 
        
        plt.figure(figsize=(10, 8))
        plt.barh(feature_df['Feature'], feature_df['Importance'], color='coral') 
        plt.xlabel('XGBoost Feature Importance (Gain)')
        plt.title(f'Top 15 Most Important Features for {self.target} (Regression)')
        plt.tight_layout()
        
        plot_path = os.path.join(BASE_DIR, 'feature_importance_baseline.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Feature Importance plot saved to: {plot_path}")

def main():
    pipeline = BaselineXGBoostRegressor(PROCESSED_DIR, TARGET_COL)
    
    train_df, val_df = pipeline.load_splits()
    
    X_train, y_train = pipeline.prepare_xy(train_df)
    X_val, y_val = pipeline.prepare_xy(val_df)
    
    pipeline.train(X_train, y_train, X_val, y_val)
    
    pipeline.evaluate(X_val, y_val)
    pipeline.plot_feature_importance()

if __name__ == "__main__":
    main()