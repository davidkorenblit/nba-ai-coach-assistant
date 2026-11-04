import pandas as pd
import numpy as np
import os
import sys
import json
# Import pipeline constants module
from pipeline_constants import get_blacklisted_features

# --- Config ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INTERIM_DIR = os.path.join(BASE_DIR, '..', 'data', 'interim')
OUTPUT_DIR = os.path.join(BASE_DIR, '..', 'data', 'processed')

TRAIN_SEASONS = ['2021_22', '2022_23', '2023_24']
TEST_SEASON = '2024_25'

class SplitValidator:
    """Draconian QA with Diagnostic Reporting for MLOps."""
    
    @staticmethod
    def validate(train_df, val_df, test_df, original_df):
        print("\n🛡️ Running Split Validation & Diagnostic Report...")
        
        train_games = set(train_df['gameId'])
        val_games = set(val_df['gameId'])
        test_games = set(test_df['gameId'])
        
        is_valid = True
        
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

        if not is_valid:
            raise ValueError("🚨 CRITICAL: Pipeline halted due to Data Leakage. Check the diagnostic report above.")
            
        total_rows = len(train_df) + len(val_df) + len(test_df)
        print(f"\n✅ Diagnostic Passed: Sets are strictly disjoint. Total rows: {total_rows:,}\n")
        return True
    
class MLDataPreparer:
    """Prepares and splits Multi-Season Level 3 data for Causal & ML modeling."""
    
    def __init__(self, interim_dir: str = None, output_dir: str = None):
        self.interim_dir = interim_dir or INTERIM_DIR
        self.output_dir = output_dir or OUTPUT_DIR
        self.df = None
        os.makedirs(self.output_dir, exist_ok=True)

    def run_pipeline(self):
        print("Starting ML Data Preparation Pipeline (Multi-Season Setup)...")
        
        print("STEP 1: Loading Multi-Season Level 3 Data...")
        historical_dfs = []
        historical_games = set()
        
        for s in TRAIN_SEASONS:
            p = os.path.join(self.interim_dir, f'level3_labels_{s}.csv')
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing training season file: {p}")
            print(f"   Loading train season {s}: {os.path.basename(p)}...")
            s_df = pd.read_csv(p, low_memory=False)
            historical_games.update(s_df['gameId'].unique())
            historical_dfs.append(s_df)
        
        # Load test season (2024-25)
        test_path = os.path.join(self.interim_dir, f'level3_labels_{TEST_SEASON}.csv')
        if not os.path.exists(test_path):
            test_path = os.path.join(self.interim_dir, 'level3_labels.csv')
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Missing test season file: {test_path}")
        print(f"   Loading holdout test season {TEST_SEASON}: {os.path.basename(test_path)}...")
        test_df_raw = pd.read_csv(test_path, low_memory=False)
        test_games = set(test_df_raw['gameId'].unique())
        
        self.df = pd.concat(historical_dfs + [test_df_raw], ignore_index=True)
        print(f"   Total combined raw rows: {len(self.df):,}")
        print(f"   Historical Train/Val games: {len(historical_games):,}, Blind Test games: {len(test_games):,}")
        
        print("STEP 2: Feature Selection (Dropping incompatible strings/objects)...")
        metadata_cols = [
            'actionType', 'actionSubtype', 'description', 'shotResult',
            'home_lineup', 'away_lineup', 'period_start_time', 'time_elapsed',
            'jumpBallRecoverdPersonId', 'jumpBallWonPersonId', 'jumpBallLostPersonId',
            'foulDrawnPersonId', 'foulTechnicalTotal', 'officialId', 
            'shotActionNumber', 'teamId'
        ]
        self.df.drop(columns=[c for c in metadata_cols if c in self.df.columns], inplace=True)
        
        object_cols = self.df.select_dtypes(include=['object']).columns
        if len(object_cols) > 0:
            print(f"⚠️ Warning: String columns detected and will be dropped: {list(object_cols)}")
            self.df.drop(columns=object_cols, inplace=True)

        print("STEP 3: Chronological Sorting by Game ID...")
        self.df.sort_values(by=['gameId', 'period', 'seconds_remaining'], 
                            ascending=[True, True, False], inplace=True)

        print("STEP 4: Splitting into Multi-Season Train (85% Historical), Val (15% Historical), Test (100% 2024/25)...")
        sorted_historical_games = [g for g in self.df['gameId'].unique() if g in historical_games]
        val_split_idx = int(len(sorted_historical_games) * 0.85)
        
        train_games = sorted_historical_games[:val_split_idx]
        val_games = sorted_historical_games[val_split_idx:]
        
        train_df = self.df[self.df['gameId'].isin(train_games)].copy()
        val_df = self.df[self.df['gameId'].isin(val_games)].copy()
        test_df = self.df[self.df['gameId'].isin(test_games)].copy()
        
        SplitValidator.validate(train_df, val_df, test_df, self.df)

        print("STEP 5: Sanitizing features and preventing Data Leakage...")
        all_targets = [c for c in train_df.columns if c.startswith('target_')]
        blacklisted = get_blacklisted_features()
        
        # Structural / experimental columns needed for split mapping and treatment
        structural_cols = [c for c in ['gameId', 'period', 'is_timeout', 'is_garbage_time'] if c in train_df.columns]
        
        # Clean causal features that exist in both train and test
        clean_features = [
            c for c in train_df.columns 
            if c not in all_targets 
            and c not in blacklisted 
            and c not in structural_cols
            and c in test_df.columns
        ]
        
        # All columns allowed in the exported Parquet splits
        export_cols = [c for c in train_df.columns if c in structural_cols or c in clean_features or c in all_targets]
        
        train_df = train_df[export_cols].copy()
        val_df = val_df[export_cols].copy()
        test_df = test_df[export_cols].copy()
        
        print(f"   Clean Features count: {len(clean_features)}")
        print(f"   Total exported columns in Parquet: {len(export_cols)} (Filtered out {len(self.df.columns) - len(export_cols)} leaked/metadata columns)")

        print("STEP 6: Exporting splits to Parquet format...")
        train_path = os.path.join(self.output_dir, 'train.parquet')
        val_path = os.path.join(self.output_dir, 'val.parquet')
        test_path = os.path.join(self.output_dir, 'test.parquet')
        
        train_df.to_parquet(train_path, index=False)
        val_df.to_parquet(val_path, index=False)
        test_df.to_parquet(test_path, index=False)

        print("STEP 7: Exporting Metadata JSON...")
        metadata = {
            "features": clean_features,
            "targets": [c for c in all_targets if c != 'target_danger_penalty'],
            "penalty_col": "target_danger_penalty" if 'target_danger_penalty' in train_df.columns else None
        }
        with open(os.path.join(self.output_dir, 'split_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print(f"✅ Success! Clean JSON and Parquets ready at: {self.output_dir}")

def main():
    try:
        preparer = MLDataPreparer(INTERIM_DIR, OUTPUT_DIR)
        preparer.run_pipeline()
    except Exception as e:
        print(f"❌ Critical Error in Splitting Pipeline: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
