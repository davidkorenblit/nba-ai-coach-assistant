import pandas as pd
import numpy as np
import os
import sys

# --- Config ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(BASE_DIR, '..', '..', 'data', 'interim', 'level2_features.csv')
OUTPUT_PATH = os.path.join(BASE_DIR, '..', '..', 'data', 'interim', 'level3_labels.csv')

class Level3Validator:
    """Quality Assurance for Level 3 Labels."""
    
    @staticmethod
    def validate(df: pd.DataFrame) -> bool:
        print("🛡️ Running Level 3 Label Validation...")
        target_cols = [
            'target_stop_run_90s', 'target_reverse_trend_180s', 
            'target_improve_margin_90s', 'target_improve_margin_180s', 
            'target_danger_penalty'
        ]
        
        # 1. Missing columns
        missing = [col for col in target_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Validator Error: Missing target columns {missing}")
            
        # 2. NaNs check
        nan_counts = df[target_cols].isna().sum()
        if nan_counts.sum() > 0:
            raise ValueError(f"Validator Error: Found NaNs in targets!\n{nan_counts[nan_counts > 0]}")
            
        # 3. Class Imbalance Check (Warning only, not fatal)
        for col in target_cols:
            positive_rate = df[col].mean()
            if positive_rate < 0.001 or positive_rate > 0.999:
                print(f"⚠️ Warning: Severe class imbalance in {col} ({positive_rate*100:.2f}% positive). XGBoost might struggle.")
            else:
                print(f"✅ {col} Class Balance: {positive_rate*100:.2f}% positive")
                
        print("✅ Validation Passed: Labels are clean and ready for ML.")
        return True

class Level3Labeler:
    """OOP implementation of Level 3 Target Generation (Lookahead)."""
    
    def __init__(self, input_path: str, output_path: str):
        self.input_path = input_path
        self.output_path = output_path
        self.col_margin = 'score_margin'
        self.col_mom = 'momentum_streak_rolling'
        self.col_exp = 'explosiveness_index'
        self.df = self._load_data()

    def _load_data(self) -> pd.DataFrame:
        if not os.path.exists(self.input_path): 
            raise FileNotFoundError(f"Missing: {self.input_path}")
        print(f"⏳ Loading Level 2 Data from {self.input_path}...")
        return pd.read_csv(self.input_path, low_memory=False)

    def build_lookahead_data(self):
        print("⏳ Creating time indices and merging future states (90s & 180s)...")
        
        # Time elapsed logic for forward lookup
        self.df['period_start_time'] = self.df.groupby(['gameId', 'period'])['seconds_remaining'].transform('max')
        self.df['time_elapsed'] = self.df['period_start_time'] - self.df['seconds_remaining']
        self.df = self.df.sort_values(by=['gameId', 'period', 'time_elapsed']).reset_index(drop=True)

        self.df['target_time_90'] = self.df['time_elapsed'] + 90
        self.df['target_time_180'] = self.df['time_elapsed'] + 180

        # Subset for future lookup
        df_future = self.df[['gameId', 'period', 'time_elapsed', self.col_margin, self.col_mom, self.col_exp]].copy()
        df_future.rename(columns={self.col_margin: 'fut_margin', self.col_mom: 'fut_mom', self.col_exp: 'fut_exp'}, inplace=True)

        # Merge 90s
        self.df = self.df.sort_values('target_time_90')
        df_future_90 = df_future.add_suffix('_90s').rename(columns={'gameId_90s': 'gameId', 'period_90s': 'period'}).sort_values('time_elapsed_90s')
        
        merged = pd.merge_asof(
            self.df, df_future_90,
            left_on='target_time_90', right_on='time_elapsed_90s',
            by=['gameId', 'period'], direction='forward'
        )

        # Merge 180s
        merged = merged.sort_values('target_time_180')
        df_future_180 = df_future.add_suffix('_180s').rename(columns={'gameId_180s': 'gameId', 'period_180s': 'period'}).sort_values('time_elapsed_180s')
        
        merged = pd.merge_asof(
            merged, df_future_180,
            left_on='target_time_180', right_on='time_elapsed_180s',
            by=['gameId', 'period'], direction='forward'
        )

        # Re-sort and fill edge cases (end of quarter/game)
        self.df = merged.sort_values(by=['gameId', 'period', 'time_elapsed']).reset_index(drop=True)
        
        for prefix in ['_90s', '_180s']:
            self.df['fut_margin'+prefix] = self.df['fut_margin'+prefix].fillna(self.df[self.col_margin])
            self.df['fut_mom'+prefix] = self.df['fut_mom'+prefix].fillna(self.df[self.col_mom])
            self.df['fut_exp'+prefix] = self.df['fut_exp'+prefix].fillna(self.df[self.col_exp])

    def build_targets(self):
        print("🎯 Generating Machine Learning Targets (Labels)...")

        # Target 1: Stop Run (Has the absolute explosiveness decreased?)
        self.df['delta_exp_abs_90s'] = self.df['fut_exp_90s'].abs() - self.df[self.col_exp].abs()
        self.df['target_stop_run_90s'] = (self.df['delta_exp_abs_90s'] < 0).astype(int)

        # Target 2: Reverse Trend 180s
        self.df['delta_mom_180s'] = self.df['fut_mom_180s'] - self.df[self.col_mom]
        self.df['target_reverse_trend_180s'] = (self.df['delta_mom_180s'] > 0).astype(int)

        # Target 3 & 4: Improve Margin (FIXED RELATIVITY BUG)
        # If score_margin > 0 (Home leading), Away is in pressure (interest_sign = -1)
        # If score_margin < 0 (Away leading), Home is in pressure (interest_sign = 1)
        self.df['interest_sign'] = np.where(self.df['score_margin'] > 0, -1, 1)
        
        self.df['delta_margin_90s'] = self.df['fut_margin_90s'] - self.df[self.col_margin]
        self.df['norm_delta_margin_90s'] = self.df['delta_margin_90s'] * self.df['interest_sign']
        self.df['target_improve_margin_90s'] = (self.df['norm_delta_margin_90s'] > 0).astype(int)

        self.df['delta_margin_180s'] = self.df['fut_margin_180s'] - self.df[self.col_margin]
        self.df['norm_delta_margin_180s'] = self.df['delta_margin_180s'] * self.df['interest_sign']
        self.df['target_improve_margin_180s'] = (self.df['norm_delta_margin_180s'] > 0).astype(int)

    def build_danger_penalty(self):
        # Target 5: Danger Penalty (FIXED DATA LEAKAGE - Domain Thresholds)
        FATIGUE_THRESHOLD = 1500 # Absolute ~25 mins instead of quantile
        EXP_THRESHOLD = 6.0      # Absolute 6-0 run instead of quantile
        
        self.df['max_fatigue'] = self.df[['home_cum_fatigue', 'away_cum_fatigue']].fillna(0).max(axis=1)
        is_danger = (self.df['max_fatigue'] > FATIGUE_THRESHOLD) & (self.df[self.col_exp].abs() > EXP_THRESHOLD)
        is_timeout = self.df['actionType'].str.contains('timeout', case=False, na=False)

        # Danger penalty: in danger, no timeout, and normalized margin got worse (negative)
        self.df['target_danger_penalty'] = (is_danger & ~is_timeout & (self.df['norm_delta_margin_180s'] < 0)).astype(int)

    def cleanup_and_save(self):
        print("🧹 Cleaning up temporary columns...")
        cols_to_drop = [
            'period_start_time', 'time_elapsed', 'target_time_90', 'target_time_180', 
            'time_elapsed_90s', 'time_elapsed_180s', 'fut_margin_90s', 'fut_mom_90s', 
            'fut_exp_90s', 'fut_margin_180s', 'fut_mom_180s', 'fut_exp_180s', 
            'max_fatigue', 'delta_exp_abs_90s', 'delta_mom_180s', 'interest_sign', 
            'delta_margin_90s', 'norm_delta_margin_90s', 'delta_margin_180s', 'norm_delta_margin_180s'
        ]
        self.df.drop(columns=[c for c in cols_to_drop if c in self.df.columns], inplace=True)

        self.df.to_csv(self.output_path, index=False)
        print(f"✅ Success! Level 3 Labels generated and saved to: {self.output_path}")

    def run_pipeline(self) -> pd.DataFrame:
        self.build_lookahead_data()
        self.build_targets()
        self.build_danger_penalty()
        self.cleanup_and_save()
        return self.df

# --- Main Execution ---
def main():
    print("🚀 Starting Level 3 Target Generation (OOP Architecture)...")
    try:
        labeler = Level3Labeler(INPUT_PATH, OUTPUT_PATH)
        df_labeled = labeler.run_pipeline()
        
        # Validate Labels
        Level3Validator.validate(df_labeled)
        
    except Exception as e:
        print(f"❌ Critical Error in Level 3: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()