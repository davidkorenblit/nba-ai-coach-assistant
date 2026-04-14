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
TARGET_COL = 'target_stop_run_90s' # המטרה הראשונה שלנו: האם הריצה נעצרה?

class BaselineXGBoost:
    """Baseline XGBoost Model to validate Feature Engineering and establish benchmarks."""
    
    def __init__(self, data_dir: str, target: str):
        self.data_dir = data_dir
        self.target = target
        self.model = None
        self.feature_cols = []
        
    def load_splits(self):
        print(f"⏳ Loading Parquet splits from {self.data_dir}...")
        try:
            train_df = pd.read_parquet(os.path.join(self.data_dir, 'train.parquet'))
            val_df = pd.read_parquet(os.path.join(self.data_dir, 'val.parquet'))
        except FileNotFoundError as e:
            print(f"❌ Error loading data: {e}. Did you run step 04?")
            sys.exit(1)
            
        print(f"✅ Loaded Train: {len(train_df):,} rows | Val: {len(val_df):,} rows")
        return train_df, val_df

    def prepare_xy(self, df: pd.DataFrame):
        """Separates features (X) and target (y), ignoring other targets/metadata."""
        # נסנן החוצה את כל עמודות המטרה האחרות והמזהים שלא שייכים לאימון
        cols_to_drop = [c for c in df.columns if c.startswith('target_')] + ['gameId']
        
        X = df.drop(columns=cols_to_drop)
        y = df[self.target]
        
        # שמירת שמות הפיצ'רים לטובת גרף החשיבות
        if not self.feature_cols:
            self.feature_cols = X.columns.tolist()
            
        return X, y

    def train(self, X_train, y_train, X_val, y_val):
        print("\n🚀 Training Baseline XGBoost Model...")
        
        # מכיוון שהתיוג שלנו הוא באזור ה-34%, כדאי לתת קצת משקל למחלקת המיעוט
        scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
        
        self.model = xgb.XGBClassifier(
            n_estimators=200,          # מספר העצים
            learning_rate=0.05,        # קצב למידה (שמרני כדי לא לעשות Overfit)
            max_depth=5,               # עומק עץ (מוגבל כדי ללמוד תבניות כלליות)
            scale_pos_weight=scale_pos_weight,
            early_stopping_rounds=20,  # עצירה אם אין שיפור
            eval_metric='auc',         # אופטימיזציה לפי השטח מתחת לעקומה
            random_state=42,
            n_jobs=-1                  # שימוש בכל ליבות המעבד
        )
        
        # אימון עם שילוב של קבוצת האימות (Validation) למניעת זליגה
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=20 # הדפסת התקדמות כל 20 עצים
        )
        print("✅ Training Complete.")

    def evaluate(self, X_val, y_val):
        print("\n📊 Evaluating on Validation Set (Unseen during training)...")
        y_pred = self.model.predict(X_val)
        y_proba = self.model.predict_proba(X_val)[:, 1]
        
        auc = roc_auc_score(y_val, y_proba)
        print(f"🌟 ROC-AUC Score: {auc:.4f} (0.5 is random guessing, 1.0 is perfect)")
        
        print("\n--- Classification Report ---")
        print(classification_report(y_val, y_pred))
        
        print("--- Confusion Matrix ---")
        print(confusion_matrix(y_val, y_pred))

    def plot_feature_importance(self):
        print("\n🎨 Generating Feature Importance Plot...")
        importance = self.model.feature_importances_
        
        # חיבור השמות לערכים ומיון
        feature_df = pd.DataFrame({'Feature': self.feature_cols, 'Importance': importance})
        feature_df = feature_df.sort_values(by='Importance', ascending=True).tail(15) # ניקח את ה-15 הכי חזקים
        
        plt.figure(figsize=(10, 8))
        plt.barh(feature_df['Feature'], feature_df['Importance'], color='skyblue')
        plt.xlabel('XGBoost Feature Importance (Gain)')
        plt.title('Top 15 Most Important Features for Stopping a Run (90s)')
        plt.tight_layout()
        
        # שמירת הגרף בתיקיית המודלים
        plot_path = os.path.join(BASE_DIR, 'feature_importance_baseline.png')
        plt.savefig(plot_path)
        print(f"✅ Feature Importance plot saved to: {plot_path}")

def main():
    pipeline = BaselineXGBoost(PROCESSED_DIR, TARGET_COL)
    
    # 1. טעינת נתונים
    train_df, val_df = pipeline.load_splits()
    
    # 2. הכנת X, Y
    X_train, y_train = pipeline.prepare_xy(train_df)
    X_val, y_val = pipeline.prepare_xy(val_df)
    
    # 3. אימון
    pipeline.train(X_train, y_train, X_val, y_val)
    
    # 4. הערכה והפקת תובנות
    pipeline.evaluate(X_val, y_val)
    pipeline.plot_feature_importance()

if __name__ == "__main__":
    main()