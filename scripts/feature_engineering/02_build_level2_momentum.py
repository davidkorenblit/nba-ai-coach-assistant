import pandas as pd
import numpy as np
import os
import sys
import ast

if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# --- Config ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(BASE_DIR, '..', '..', 'data', 'interim', 'level1_base.csv')
OUTPUT_PATH = os.path.join(BASE_DIR, '..', '..', 'data', 'interim', 'level2_features.csv')
LOOKUP_PATH = os.path.join(BASE_DIR, '..', '..', 'data', 'lookup', 'high_usage_players_2024-25.csv')

class Level2Validator:
    """Quality Assurance for Level 2 Features."""
    
    @staticmethod
    def validate(df: pd.DataFrame) -> bool:
        print("🛡️ Running Level 2 Data Validation...")
        critical_cols = [
            'home_usage_gravity', 'usage_delta', 'home_cum_fatigue', 'away_cum_fatigue',
            'home_momentum_streak', 'away_momentum_streak', 'momentum_streak_rolling',
            'explosiveness_index', 'star_advantage', 'is_star_resting'
        ]
        
        for col in critical_cols:
            if col not in df.columns:
                raise ValueError(f"Validator Error: Missing column {col}")
            
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                raise ValueError(f"Validator Error: Found {nan_count} NaNs in {col}. Model will fail.")
                
        print("✅ Validation Passed: Zero NaNs and all feature columns present.")
        return True

class Level2FeatureEngineer:
    """OOP implementation of Level 2 Feature Engineering."""
    
    def __init__(self, input_path: str, lookup_path: str):
        self.input_path = input_path
        self.lookup_path = lookup_path
        self.stars_map = self._load_stars_lookup()
        self.df = self._load_data()

    def _load_stars_lookup(self) -> dict:
        if not os.path.exists(self.lookup_path):
            print(f"⚠️ Warning: Lookup not found at {self.lookup_path}.")
            return {}
        try:
            stars_df = pd.read_csv(self.lookup_path)
            return dict(zip(stars_df['PLAYER_ID'], stars_df['USG_PCT']))
        except Exception as e:
            print(f"❌ Error loading star lookup: {e}")
            return {}

    def _load_data(self) -> pd.DataFrame:
        if not os.path.exists(self.input_path): 
            raise FileNotFoundError(f"Missing: {self.input_path}")
        
        df = pd.read_csv(self.input_path, low_memory=False)
        print("🔹 Converting lineups from strings to lists...")
        for col in ['home_lineup', 'away_lineup']:
            df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
            
        df.sort_values(by=['gameId', 'period', 'seconds_remaining'], ascending=[True, True, False], inplace=True)
        return df

    def build_usage_gravity(self):
        print("🔹 Building: Usage Gravity (Vectorized Lookup)...")
        for prefix in ['home', 'away']:
            lineup_col = f'{prefix}_lineup'
            exploded = self.df[lineup_col].explode()
            usg_mapped = exploded.map(self.stars_map).fillna(0.15)
            usg_mapped[exploded.isna()] = 0.0
            
            gravity = usg_mapped.groupby(level=0).sum()
            gravity[gravity == 0] = 0.75 # Default threshold
            self.df[f'{prefix}_usage_gravity'] = gravity

        self.df['usage_delta'] = self.df['home_usage_gravity'] - self.df['away_usage_gravity']

    def build_accumulated_fatigue(self):
        print("🔹 Building: Accumulated Fatigue Track (Vectorized Explode/Cumsum)...")
        
        for prefix in ['home', 'away']:
            lineup_col = f'{prefix}_lineup'
            temp_exp = self.df[['gameId', 'play_duration', lineup_col]].explode(lineup_col)
            
            temp_exp['player_cum_dur'] = temp_exp.groupby(['gameId', lineup_col])['play_duration'].cumsum()
            
            self.df[f'{prefix}_cum_fatigue'] = temp_exp.groupby(level=0)['player_cum_dur'].mean().fillna(0)

    def build_smart_streak(self):
        print("🔹 Building: Smart Momentum Streak (Dual Streaks with Momentum Breakers)...")
        # 1. Team attribution
        home_ids = self.df[self.df['scoreHome'].diff() > 0].groupby('gameId')['teamId'].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else np.nan)
        is_home = (self.df['teamId'] == self.df['gameId'].map(home_ids))
        is_away = (self.df['teamId'].notna()) & (~is_home)

        # 2. Event weights
        weights = {'3pt': 1.5, '2pt': 1.0, 'steal': 2.0, 'block': 1.5}
        val = self.df['actionType'].map(weights).fillna(0.0) * (self.df['shotResult'].fillna('Made') == 'Made')
        if 'foulTechnicalTotal' in self.df.columns:
            val += (self.df['foulTechnicalTotal'] > 0).astype(float) * 2.5

        h_val, a_val = (val * is_home).to_numpy(), (val * is_away).to_numpy()

        # 3. Dual streak loop with Tug-of-War continuous gauge
        h_streaks, a_streaks = np.zeros(len(self.df)), np.zeros(len(self.df))
        cur_gid, h_s, a_s = None, 0.0, 0.0
        for i, gid in enumerate(self.df['gameId'].to_numpy()):
            if gid != cur_gid: cur_gid, h_s, a_s = gid, 0.0, 0.0
            if h_val[i] > 0:
                h_s += h_val[i]
                a_s = max(0.0, a_s - h_val[i])
            if a_val[i] > 0:
                a_s += a_val[i]
                h_s = max(0.0, h_s - a_val[i])
            h_streaks[i], a_streaks[i] = h_s, a_s

        self.df['home_momentum_streak'] = h_streaks
        self.df['away_momentum_streak'] = a_streaks
        self.df['momentum_delta'] = h_streaks - a_streaks
        self.df['momentum_streak_rolling'] = self.df['momentum_delta']

    def build_explosiveness(self):
        print("🔹 Building: Explosiveness Index...")
        LOOKBACK = 20
        self.df['score_diff_lag'] = self.df.groupby('gameId')['score_margin'].shift(LOOKBACK)
        self.df['explosiveness_index'] = (self.df['score_margin'] - self.df['score_diff_lag']).fillna(0)
        self.df.drop(columns=['score_diff_lag'], inplace=True)

    def build_context_features(self):
        print("🔹 Building: Contextual & Shift Features...")
        self.df['style_tempo_rolling'] = self.df.groupby('gameId')['shot_clock_estimated'].transform(
            lambda x: x.rolling(window=15, min_periods=1).mean()
        ).fillna(14.0)
        
        self.df['is_high_fatigue'] = np.where(self.df['time_since_last_sub'] > 550, 1, 0)
        
        self.df['time_lag'] = self.df.groupby(['gameId', 'period'])['seconds_remaining'].shift(10)
        self.df['instability_index'] = (self.df['time_lag'] - self.df['seconds_remaining']).fillna(60)
        self.df.drop(columns=['time_lag'], inplace=True)
        
        self.df['is_clutch_time'] = np.where(
            (self.df['seconds_remaining'] <= 300) & (self.df['score_margin'].abs() <= 5), 1, 0
        )

    def build_star_resting(self):
        print("🔹 Building: Star Resting & Advantage Differential...")
        star_ids = list(self.stars_map.keys())
        if not star_ids:
            self.df['star_advantage'] = 0
            self.df['is_star_resting'] = 0
            return

        home_has_star = self.df['home_lineup'].explode().isin(star_ids).groupby(level=0).any().reindex(self.df.index, fill_value=False)
        away_has_star = self.df['away_lineup'].explode().isin(star_ids).groupby(level=0).any().reindex(self.df.index, fill_value=False)
        
        self.df['star_advantage'] = (home_has_star.astype(int) - away_has_star.astype(int)).astype(int)
        self.df['is_star_resting'] = self.df['star_advantage']

    def run_pipeline(self) -> pd.DataFrame:
        self.build_usage_gravity()
        self.build_accumulated_fatigue()
        self.build_smart_streak()
        self.build_explosiveness()
        self.build_context_features()
        self.build_star_resting()
        return self.df

# --- Main Execution ---
def main():
    print("🚀 Starting Level 2 Feature Engineering (Optimized OOP Architecture)...")
    try:
        engineer = Level2FeatureEngineer(INPUT_PATH, LOOKUP_PATH)
        df_features = engineer.run_pipeline()
        
        Level2Validator.validate(df_features)
        
        df_features.to_csv(OUTPUT_PATH, index=False)
        print(f"✅ Saved Optimized Level 2 to: {OUTPUT_PATH}")
        print(f"📊 Final Dataset Shape: {df_features.shape}")
        
    except Exception as e:
        print(f"❌ Critical Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
