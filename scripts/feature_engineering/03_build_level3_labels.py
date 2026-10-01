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
        
        # translated_comment translated_comment translated_comment translated_comment translated_comment translated_comment translated_comment translated_comment translated_comment translated_comment translated_comment translated_comment translated_comment
        continuous_targets = [
            'target_stop_run_90s', 'target_reverse_trend_180s', 
            'target_improve_margin_90s', 'target_improve_margin_180s'
        ]
        binary_targets = ['target_danger_penalty']
        target_cols = continuous_targets + binary_targets
        
        # 1. Missing columns
        missing = [col for col in target_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Validator Error: Missing target columns {missing}")
            
        # 2. Boundary-aware completeness check (Right-Censoring validation)
        # Non-boundary plays (>=90s or >=180s left in quarter) MUST have complete non-NaN targets
        valid_90s_mask = (df['seconds_remaining'] >= 90)
        nans_in_valid_90 = df.loc[valid_90s_mask, ['target_stop_run_90s', 'target_improve_margin_90s']].isna().sum()
        if nans_in_valid_90.sum() > 0:
            raise ValueError(f"Validator Error: Unexpected NaNs in non-boundary 90s targets!\n{nans_in_valid_90}")

        valid_180s_mask = (df['seconds_remaining'] >= 180)
        nans_in_valid_180 = df.loc[valid_180s_mask, ['target_reverse_trend_180s', 'target_improve_margin_180s', 'target_danger_penalty']].isna().sum()
        if nans_in_valid_180.sum() > 0:
            raise ValueError(f"Validator Error: Unexpected NaNs in non-boundary 180s targets!\n{nans_in_valid_180}")

        print(f"✅ Right-Censoring Validation Passed: 90s targets valid for {valid_90s_mask.sum():,} plays ({valid_90s_mask.mean()*100:.1f}%), 180s targets valid for {valid_180s_mask.sum():,} plays ({valid_180s_mask.mean()*100:.1f}%).")

        # 3. Class Imbalance Check (Only for valid binary penalty)
        penalty_valid = df['target_danger_penalty'].dropna()
        positive_rate = penalty_valid.mean()
        print(f"✅ target_danger_penalty Class Balance: {positive_rate*100:.2f}% positive (evaluated on {len(penalty_valid):,} valid non-boundary rows)")
                
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

        # Re-sort (Keep quarter-end boundary lookahead NaNs for proper Right-Censoring)
        self.df = merged.sort_values(by=['gameId', 'period', 'time_elapsed']).reset_index(drop=True)

    def build_targets(self):
        print("🎯 Generating Machine Learning Targets (Labels)...")

        self.df['is_garbage_time'] = (
            ((self.df['period'] == 4) & (self.df['seconds_remaining'] <= 180) & (self.df['score_margin'].abs() >= 15)) |
            ((self.df['period'] == 4) & (self.df['seconds_remaining'] > 180) & (self.df['score_margin'].abs() >= 30)) |
            ((self.df['period'] > 4) & (self.df['score_margin'].abs() >= 20))
        ).astype(int)

        # 1. Determine acting team perspective (Home: +1, Away: -1)
        home_team_series = self.df[self.df['scoreHome'].diff() > 0].groupby('gameId')['teamId'].agg(
            lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else np.nan
        )
        home_team_map = home_team_series.to_dict()
        for gid in self.df['gameId'].unique():
            if gid not in home_team_map or pd.isna(home_team_map[gid]):
                game_teams = self.df[self.df['gameId'] == gid]['teamId'].dropna().unique()
                if len(game_teams) > 0:
                    home_team_map[gid] = game_teams[0]

        home_team_id = self.df['gameId'].map(home_team_map)
        is_away = (self.df['teamId'].notna()) & (self.df['teamId'] != home_team_id)
        self.df['interest_sign'] = np.where(is_away, -1, 1)

        # Target 1: Stop Run (Reduction in absolute opponent explosiveness -> Positive is good)
        self.df['delta_exp_abs_90s'] = self.df['fut_exp_90s'].abs() - self.df[self.col_exp].abs()
        self.df['target_stop_run_90s'] = -self.df['delta_exp_abs_90s']

        # Target 2: Reverse Trend 180s (Momentum shift in favor of acting team)
        self.df['delta_mom_180s'] = self.df['fut_mom_180s'] - self.df[self.col_mom]
        self.df['target_reverse_trend_180s'] = self.df['delta_mom_180s'] * self.df['interest_sign']

        # Target 3 & 4: Improve Margin 90s & 180s (Normalized point gain from acting team perspective)
        self.df['delta_margin_90s'] = self.df['fut_margin_90s'] - self.df[self.col_margin]
        self.df['norm_delta_margin_90s'] = self.df['delta_margin_90s'] * self.df['interest_sign']
        self.df['target_improve_margin_90s'] = self.df['norm_delta_margin_90s']

        self.df['delta_margin_180s'] = self.df['fut_margin_180s'] - self.df[self.col_margin]
        self.df['norm_delta_margin_180s'] = self.df['delta_margin_180s'] * self.df['interest_sign']
        self.df['target_improve_margin_180s'] = self.df['norm_delta_margin_180s']

    def build_danger_penalty(self):
        # Target 5: Danger Penalty
        FATIGUE_THRESHOLD = 1500 
        EXP_THRESHOLD = 6.0      
        
        self.df['max_fatigue'] = self.df[['home_cum_fatigue', 'away_cum_fatigue']].fillna(0).max(axis=1)
        is_danger = (self.df['max_fatigue'] > FATIGUE_THRESHOLD) & (self.df[self.col_exp].abs() > EXP_THRESHOLD)
        is_timeout = self.df['actionType'].str.contains('timeout', case=False, na=False)

        # Danger penalty: in danger, no timeout, and normalized margin got worse (negative)
        self.df['target_danger_penalty'] = np.where(
            self.df['norm_delta_margin_180s'].isna(),
            np.nan,
            (is_danger & ~is_timeout & (self.df['norm_delta_margin_180s'] < 0)).astype(float)
        )

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
