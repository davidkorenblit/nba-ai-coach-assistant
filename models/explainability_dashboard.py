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

        # שליפת רשימת הפיצ'רים המדויקת שהמודל אומן עליהם
        # זה הפתרון לבעיית הממדים שקיבלת (35 vs 120)
        try:
            model_features = self.tau1_model.get_booster().feature_names
        except AttributeError:
            # גיבוי למקרה שהמודל עטוף ב-Scikit-Learn Wrapper בגרסה מסוימת
            model_features = self.tau1_model.feature_names_in_.tolist()

        # ניקח את ה-N המקרים הכי קיצוניים
        top_cases = df.head(top_n)
        
        # יצירת X_explain שמכיל אך ורק את הפיצ'רים שהמודל מכיר ובסדר הנכון
        X_explain = top_cases[model_features]
        
        # אנחנו מנתחים את מודל ה-Tau1 (השפעת פסק זמן) כאינדיקטור מרכזי
        explainer = shap.TreeExplainer(self.tau1_model)
        
        # חישוב ערכי SHAP
        shap_values = explainer.shap_values(X_explain)

        for i in range(len(top_cases)):
            row = top_cases.iloc[i]
            print(f"\n🚨 ALERT #{i+1} (Impact: {row['predicted_cate']*100:.1f}%)")
            
            # שליפת ערכי ה-SHAP עבור השורה הנוכחית
            # (בגרסאות SHAP חדשות זה עשוי להחזיר אובייקט Explanation, לכן נשתמש בגישה בטוחה)
            current_shap = shap_values.values[i] if hasattr(shap_values, "values") else shap_values[i]
            
            # מציאת הפיצ'ר הכי משפיע במקרה הספציפי הזה
            feature_idx = np.argmax(np.abs(current_shap))
            feature_name = X_explain.columns[feature_idx]
            impact_direction = "increased" if current_shap[feature_idx] > 0 else "decreased"
            
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