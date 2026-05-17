import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
import shap

class ExplainabilityDashboard:
    def __init__(self, reports_dir: str, models_dir: str, target_col: str):
        self.reports_dir = reports_dir
        self.models_dir = models_dir
        self.target_col = target_col
        self.csv_path = os.path.join(reports_dir, f'timeout_recommendations_report_{target_col}.csv')
        
        # טעינת המודלים לצורך SHAP
        self.tau0_model = joblib.load(os.path.join(models_dir, f'tau0_{target_col}.joblib'))
        self.tau1_model = joblib.load(os.path.join(models_dir, f'tau1_{target_col}.joblib'))
        self.propensity_model = joblib.load(os.path.join(models_dir, f'propensity_{target_col}.joblib'))

    def translate_to_basketball(self, feature_name):
        """מתרגם שמות עמודות לשפה של מאמנים."""
        translations = {
            'momentum_streak_rolling': "Opponent's Scoring Run",
            'home_cum_fatigue': "Home Team Fatigue Level",
            'away_cum_fatigue': "Away Team Fatigue Level",
            'score_margin': "Current Score Difference",
            'event_momentum_val': "Instant Momentum Shift",
            'period': "Game Period"
        }
        return translations.get(feature_name, feature_name)

    def run_shap_analysis(self, top_n=5):
        """מנתח את ה'למה' מאחורי ההמלצות המובילות."""
        print(f"\n--- 🧠 SHAP Tactical Analysis for {self.target_col} ---")
        
        df = pd.read_csv(self.csv_path)
        if df.empty: return

        # אנחנו מנתחים את מודל ה-Tau1 (השפעת פסק זמן בקבוצת הטיפול) כאינדיקטור מרכזי
        explainer = shap.TreeExplainer(self.tau1_model)
        
        # ניקח את ה-5 המקרים הכי קיצוניים
        top_cases = df.head(top_n)
        
        # הכנת הדאטה ל-SHAP (הסרת עמודות הציון)
        cols_to_drop = ['predicted_cate', 'actual_treatment', 'target_danger_penalty', 'gameId']
        X_explain = top_cases.drop(columns=[c for c in cols_to_drop if c in top_cases.columns])
        
        shap_values = explainer.shap_values(X_explain)

        for i in range(len(top_cases)):
            row = top_cases.iloc[i]
            print(f"\n🚨 ALERT #{i+1} (Impact: {row['predicted_cate']*100:.1f}%)")
            
            # מציאת הפיצ'ר הכי משפיע במקרה הזה
            idx = i
            feature_idx = np.argmax(np.abs(shap_values[idx]))
            feature_name = X_explain.columns[feature_idx]
            impact_direction = "increased" if shap_values[idx][feature_idx] > 0 else "decreased"
            
            scout_msg = f"   Scout Note: {self.translate_to_basketball(feature_name)} is the primary driver. " \
                        f"It {impact_direction} the recommendation urgency."
            print(scout_msg)

    def hunt_hero_case(self):
        """מוצא משחק ספציפי שבו המאמן טעה והמודל צדק."""
        df = pd.read_csv(self.csv_path)
        if 'gameId' not in df.columns:
            print("⚠️ Missing gameId in CSV. Cannot hunt for Hero Case. See fix below.")
            return

        # חיפוש משחק עם הכי הרבה התראות שהתעלמו מהן (actual_treatment == 0) 
        # ושבסוף הסתיים בענישה (target_danger_penalty == 1)
        bad_coaching_games = df[(df['actual_treatment'] == 0) & (df['target_danger_penalty'] == 1)]
        
        if bad_coaching_games.empty:
            print("No perfect Hero Case found where coach was punished.")
            return

        hero_game_id = bad_coaching_games['gameId'].value_counts().idxmax()
        num_alerts = bad_coaching_games['gameId'].value_counts().max()
        
        print(f"\n🏆 HERO CASE FOUND!")
        print(f"   Game ID: {hero_game_id}")
        print(f"   The coach ignored {num_alerts} critical alerts before a collapse.")
        return hero_game_id

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    REPORTS_DIR = os.path.join(base_dir, 'reports')
    MODELS_DIR = os.path.join(base_dir, 'models', 'saved_models')
    
    # הרצה על היעד המרכזי
    target = 'target_stop_run_90s'
    dashboard = ExplainabilityDashboard(REPORTS_DIR, MODELS_DIR, target)
    dashboard.run_shap_analysis()
    dashboard.hunt_hero_case()