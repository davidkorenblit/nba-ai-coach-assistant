import pandas as pd
import numpy as np
import os
import sys

# --- Config ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(BASE_DIR, '..',  'data', 'interim', 'level3_labels.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, '..',  'data', 'processed')

class SplitValidator:
    """Draconian QA with Diagnostic Reporting for MLOps."""
    
    @staticmethod
    def validate(train_df, val_df, test_df, original_df):
        print("\n🛡️ Running Split Validation & Diagnostic Report...")
        
        train_games = set(train_df['gameId'])
        val_games = set(val_df['gameId'])
        test_games = set(test_df['gameId'])
        
        is_valid = True
        
        # 1. 🔍 זיהוי מדויק של מקור הכישלון (Data Leakage)
        train_val_leak = train_games.intersection(val_games)
        train_test_leak = train_games.intersection(test_games)
        val_test_leak = val_games.intersection(test_games)
        
        if train_val_leak:
            print(f"❌ [FAIL] Leakage Train/Val! Caused by Game IDs: {list(train_val_leak)[:5]}...")
            is_valid = False
        if train_test_leak:
            print(f"❌ [FAIL] Leakage Train/Test! Caused by Game IDs: {list(train_test_leak)[:5]}...")
            is_valid = False
        if val_test_leak:
            print(f"❌ [FAIL] Leakage Val/Test! Caused by Game IDs: {list(val_test_leak)[:5]}...")
            is_valid = False
            
        # 2. 📊 מיפוי דאטא חסר (Missing Data Heatmap)
        print("\n🔎 DIAGNOSTICS: Missing Data Report (NaNs expected by XGBoost)")
        nan_counts = original_df.isna().sum()
        missing_data = nan_counts[nan_counts > 0].sort_values(ascending=False)
        
        if not missing_data.empty:
            print("   ⚠️ Features with missing data (Model will use default tree paths):")
            for col, count in missing_data.items():
                percent = (count / len(original_df)) * 100
                print(f"      - {col}: {count:,} missing rows ({percent:.1f}%)")
        else:
            print("   ✨ Zero missing data. The dataset is incredibly dense.")

        # 3. הפלת הריצה רק אם יש Leakage (כי NaNs מותרים ב-XGBoost)
        if not is_valid:
            raise ValueError("🚨 CRITICAL: Pipeline halted due to Data Leakage. Check the diagnostic report above.")
            
        total_rows = len(train_df) + len(val_df) + len(test_df)
        print(f"\n✅ Diagnostic Passed: Sets are strictly disjoint. Total rows: {total_rows:,}\n")
        return True
    

class MLDataPreparer:
    """Prepares and splits Level 3 data for XGBoost modeling."""
    
    def __init__(self, input_path: str, output_dir: str):
        self.input_path = input_path
        self.output_dir = output_dir
        self.df = None
        
        # Create output dir if not exists
        os.makedirs(self.output_dir, exist_ok=True)

    def run_pipeline(self):
        print("Starting ML Data Preparation Pipeline...")
        
        # --- STEP 1:  (Loading Level 3 Data) ---
        print("STEP 1: Loading Level 3 Data...")
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"Missing: {self.input_path}")
        self.df = pd.read_csv(self.input_path, low_memory=False)
        
        # --- STEP 2: (Feature Selection) ---
        print(" STEP 2: Feature Selection (Dropping incompatible strings/objects)...")
        # עמודות שלא נכנסות לחישובי המודל (מחרוזות, מערכים, או מזהים טכניים)
        metadata_cols = [
            'actionType', 'actionSubtype', 'description', 'shotResult',
            'home_lineup', 'away_lineup', 'period_start_time', 'time_elapsed'
            'jumpBallRecoverdPersonId', 'jumpBallWonPersonId', 'jumpBallLostPersonId',
            'foulDrawnPersonId', 'foulTechnicalTotal', 'officialId', 
            'shotActionNumber', 'teamId'
        ]
        # we may do need those bcasue of fe
        # נשמור רק את העמודות שרלוונטיות או מזהים שחייבים לחיתוך
        self.df.drop(columns=[c for c in metadata_cols if c in self.df.columns], inplace=True)
        
        # וידוא שאין עמודות Object (String) שנשארו ויפילו את XGBoost
        object_cols = self.df.select_dtypes(include=['object']).columns
        if len(object_cols) > 0:
            print(f"⚠️ Warning: String columns detected and will be dropped: {list(object_cols)}")
            self.df.drop(columns=object_cols, inplace=True)

        # --- STEP 3: מיון וקיבוץ כרונולוגי (Sorting & Grouping) ---
        print("STEP 3: Chronological Sorting by Game ID...")
        # מיון כדי להבטיח שמשחקים בתחילת העונה יהיו ב-Train ומשחקי הסוף ב-Test
        self.df.sort_values(by=['gameId', 'period', 'seconds_remaining'], 
                            ascending=[True, True, False], inplace=True)
        
        unique_games = self.df['gameId'].unique()
        total_games = len(unique_games)
        print(f"   Found {total_games} unique games.")

        # --- STEP 4: חיתוך מתמטי לפי משחקים (Mathematical Split) ---
        print("STEP 4: Splitting into Train (70%), Val (15%), Test (15%)...")
        train_idx = int(total_games * 0.70)
        val_idx = int(total_games * 0.85)
        
        train_games = unique_games[:train_idx]
        val_games = unique_games[train_idx:val_idx]
        test_games = unique_games[val_idx:]
        
        train_df = self.df[self.df['gameId'].isin(train_games)].copy()
        val_df = self.df[self.df['gameId'].isin(val_games)].copy()
        test_df = self.df[self.df['gameId'].isin(test_games)].copy()
        
        # הפעלת ולידציה
        SplitValidator.validate(train_df, val_df, test_df, self.df)

        # --- STEP 5: ייצוא אופטימלי לזיכרון (Optimal Export to Parquet) ---
        print("STEP 5: Exporting splits to Parquet format...")
        train_path = os.path.join(self.output_dir, 'train.parquet')
        val_path = os.path.join(self.output_dir, 'val.parquet')
        test_path = os.path.join(self.output_dir, 'test.parquet')
        
        train_df.to_parquet(train_path, index=False)
        val_df.to_parquet(val_path, index=False)
        test_df.to_parquet(test_path, index=False)
        
        print(f"✅ Success! Data is ready for XGBoost at: {self.output_dir}")
        print(f"   Train: {len(train_df):,} rows")
        print(f"   Val:   {len(val_df):,} rows")
        print(f"   Test:  {len(test_df):,} rows")

def main():
    try:
        preparer = MLDataPreparer(INPUT_PATH, OUTPUT_DIR)
        preparer.run_pipeline()
    except Exception as e:
        print(f"❌ Critical Error in Splitting Pipeline: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()