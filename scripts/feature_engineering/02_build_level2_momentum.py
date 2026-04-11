import pandas as pd
import numpy as np
import os
import sys
import ast

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
            'home_usage_gravity', 'usage_delta', 'home_cum_fatigue', 
            'momentum_streak_rolling', 'explosiveness_index', 'is_star_resting'
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
            # Returns a dict of PLAYER_ID: USG_PCT
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
        # Default usage is 0.15 (15%) for non-stars
        def calc_gravity(lineup):
            if not isinstance(lineup, list): return 0.75 
            return sum([self.stars_map.get(pid, 0.15) for pid in lineup])

        self.df['home_usage_gravity'] = self.df['home_lineup'].apply(calc_gravity)
        self.df['away_usage_gravity'] = self.df['away_lineup'].apply(calc_gravity)
        self.df['usage_delta'] = self.df['home_usage_gravity'] - self.df['away_usage_gravity']

    def build_accumulated_fatigue(self):
        print("🔹 Building: Accumulated Fatigue Track (Optimized State Machine)...")
        # Optimization: Avoid iterrows. Using a highly optimized list comprehension state tracker.
        
        def compute_fatigue(group):
            player_mins = {}
            h_fatigue = np.zeros(len(group))
            a_fatigue = np.zeros(len(group))
            
            durations = group['play_duration'].values
            h_lineups = group['home_lineup'].values
            a_lineups = group['away_lineup'].values
            
            for i in range(len(group)):
                dur = durations[i]
                h_l = h_lineups[i] if isinstance(h_lineups[i], list) else []
                a_l = a_lineups[i] if isinstance(a_lineups[i], list) else []
                
                # Accumulate
                for pid in h_l + a_l:
                    player_mins[pid] = player_mins.get(pid, 0) + dur
                
                # Calculate team average fatigue
                h_fatigue[i] = sum([player_mins.get(p, 0) for p in h_l]) / 5 if h_l else 0
                a_fatigue[i] = sum([player_mins.get(p, 0) for p in a_l]) / 5 if a_l else 0
                
            group['home_cum_fatigue'] = h_fatigue
            group['away_cum_fatigue'] = a_fatigue
            return group

        self.df = self.df.groupby('gameId', group_keys=False).apply(compute_fatigue)

    def build_smart_streak(self):
        print("🔹 Building: Smart Momentum Streak (Vectorized)...")
        self.df['event_momentum_val'] = 0.0
        
        # Vectorized assignments
        self.df.loc[(self.df['actionType'] == '3pt') & (self.df['shotResult'] == 'Made'), 'event_momentum_val'] += 1.5
        self.df.loc[(self.df['actionType'] == '2pt') & (self.df['shotResult'] == 'Made'), 'event_momentum_val'] += 1.0
        self.df.loc[self.df['actionType'] == 'steal', 'event_momentum_val'] += 2.0
        self.df.loc[self.df['actionType'] == 'block', 'event_momentum_val'] += 1.5
        
        if 'foulTechnicalTotal' in self.df.columns:
            self.df.loc[self.df['foulTechnicalTotal'] > 0, 'event_momentum_val'] += 2.5
        
        WINDOW_EVENTS = 10
        self.df['momentum_streak_rolling'] = self.df.groupby('gameId')['event_momentum_val'].transform(
            lambda x: x.rolling(window=WINDOW_EVENTS, min_periods=1).sum()
        ).fillna(0)

    def build_explosiveness(self):
        print("🔹 Building: Explosiveness Index...")
        LOOKBACK = 20
        self.df['score_diff_lag'] = self.df.groupby('gameId')['score_margin'].shift(LOOKBACK)
        self.df['explosiveness_index'] = (self.df['score_margin'] - self.df['score_diff_lag']).fillna(0)
        self.df.drop(columns=['score_diff_lag'], inplace=True)

    def build_context_features(self):
        print("🔹 Building: Contextual & Shift Features...")
        # Style Shift
        self.df['style_tempo_rolling'] = self.df.groupby('gameId')['shot_clock_estimated'].transform(
            lambda x: x.rolling(window=15, min_periods=1).mean()
        ).fillna(14.0)
        
        # Shared Fatigue Calibrated
        self.df['is_high_fatigue'] = np.where(self.df['time_since_last_sub'] > 550, 1, 0)
        
        # Instability Index
        self.df['time_lag'] = self.df.groupby(['gameId', 'period'])['seconds_remaining'].shift(10)
        self.df['instability_index'] = (self.df['time_lag'] - self.df['seconds_remaining']).fillna(60)
        self.df.drop(columns=['time_lag'], inplace=True)
        
        # Clutch Time
        self.df['is_clutch_time'] = np.where(
            (self.df['seconds_remaining'] <= 300) & (self.df['score_margin'].abs() <= 5), 1, 0
        )

    def build_star_resting(self):
        print("🔹 Building: Star Resting...")
        star_ids = set(self.stars_map.keys())
        if not star_ids:
            self.df['is_star_resting'] = 0
            return

        def is_star_on(row):
            combined = set(row.get('home_lineup', [])) | set(row.get('away_lineup', []))
            return 1 if not combined.isdisjoint(star_ids) else 0

        self.df['has_star_on_court'] = self.df.apply(is_star_on, axis=1)
        self.df['is_star_resting'] = 1 - self.df['has_star_on_court']
        self.df.drop(columns=['has_star_on_court'], inplace=True)

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
    print("🚀 Starting Level 2 Feature Engineering (OOP Architecture)...")
    try:
        engineer = Level2FeatureEngineer(INPUT_PATH, LOOKUP_PATH)
        df_features = engineer.run_pipeline()
        
        # Validate Data Integrity
        Level2Validator.validate(df_features)
        
        df_features.to_csv(OUTPUT_PATH, index=False)
        print(f"✅ Saved Upgraded Level 2 to: {OUTPUT_PATH}")
        print(f"📊 Final Dataset Shape: {df_features.shape}")
        
    except Exception as e:
        print(f"❌ Critical Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()