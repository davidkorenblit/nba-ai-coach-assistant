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
        
        # טעינת המודלים
        try:
            self.tau1_model = joblib.load(os.path.join(models_dir, f'tau1_{target_col}.joblib'))
        except:
            self.tau1_model = None

    def translate_to_basketball(self, feature_name):
        translations = {
            'momentum_streak_rolling': "Opponent's Scoring Run",
            'home_cum_fatigue': "Home Team Fatigue",
            'away_cum_fatigue': "Away Team Fatigue",
            'score_margin': "Score Margin",
            'usage_delta': "Usage Imbalance",
            'style_tempo_rolling': "Game Tempo Shift",
            'time_since_last_sub': "Stale Lineup (No Subs)"
        }
        return translations.get(feature_name, feature_name)

    def run_analysis(self):
        print(f"\n--- 🧠 Analyzing Target: {self.target_col} ---")
        if not os.path.exists(self.csv_path):
            print(f"⚠️ CSV not found for {self.target_col}")
            return

        df = pd.read_csv(self.csv_path).head(5) # ניתוח 5 המקרים הכי חזקים
        if df.empty or self.tau1_model is None: return

        # הכנת הדאטה ל-SHAP
        model_features = self.tau1_model.get_booster().feature_names
        X_explain = df[model_features]
        explainer = shap.TreeExplainer(self.tau1_model)
        shap_values = explainer.shap_values(X_explain)

        print("📊 Visual graph generation is currently disabled (commented out).")
        
        # --- לבקשתך, יצירת הגרפים מושהית כרגע בהערה כדי לא לעשות בלאגן בעין ---
        # fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        # fig.suptitle(f'Tactical Drivers: {self.target_col}', fontsize=16, fontweight='bold')
        # 
        # for i in range(min(3, len(df))):
        #     current_shap = shap_values[i] if not hasattr(shap_values, "values") else shap_values.values[i]
        #     
        #     # לקיחת 4 הפיצ'רים הכי משפיעים
        #     top_indices = np.argsort(np.abs(current_shap))[-4:]
        #     features = [self.translate_to_basketball(X_explain.columns[j]) for j in top_indices]
        #     impacts = [current_shap[j] for j in top_indices]
        # 
        #     axes[i].barh(features, impacts, color=['#ff9999' if x < 0 else '#66b3ff' for x in impacts])
        #     axes[i].set_title(f"Alert #{i+1} (Impact: {df.iloc[i]['predicted_cate']*100:.1f}%)")
        #     axes[i].axvline(0, color='black', linewidth=0.8)
        # 
        # plt.tight_layout()
        # plot_path = os.path.join(self.reports_dir, f'tactical_breakdown_{self.target_col}.png')
        # plt.savefig(plot_path)
        # plt.close()
        # print(f"📊 Graph saved to: {plot_path}")

    def explain_hero_case_text(self):
        print(f"\n" + "="*60)
        print(f"🎯 Hunting & Explaining Hero Case for {self.target_col}")
        
        if not os.path.exists(self.csv_path):
            print("⚠️ CSV not found.")
            return

        df = pd.read_csv(self.csv_path)
        if 'gameId' not in df.columns:
            return
            
        # סינון ה-Hero Case
        bad_coaching_games = df[(df['actual_treatment'] == 0) & (df['target_danger_penalty'] == 1)]
        if bad_coaching_games.empty: return

        hero_game_id = bad_coaching_games['gameId'].value_counts().idxmax()
        num_alerts = bad_coaching_games['gameId'].value_counts().max()
        
        print(f"🏆 HERO GAME FOUND: {hero_game_id} (Ignored {num_alerts} alerts before collapse)")
        
        # שולף רק את ההתראות של המשחק הספציפי הזה
        hero_alerts = bad_coaching_games[bad_coaching_games['gameId'] == hero_game_id].sort_values(by='predicted_cate', ascending=False)
        
        if self.tau1_model is None: return
        
        # חישוב SHAP
        model_features = self.tau1_model.get_booster().feature_names
        X_explain = hero_alerts[model_features]
        explainer = shap.TreeExplainer(self.tau1_model)
        shap_values = explainer.shap_values(X_explain)
        
        # הדפסת התוכן של הגרף כטקסט קריא
        for i in range(len(hero_alerts)):
            row = hero_alerts.iloc[i]
            current_shap = shap_values[i] if not hasattr(shap_values, "values") else shap_values.values[i]
            
            # מוצא את ה-4 פיצ'רים עם ההשפעה הכי גדולה (בערך מוחלט)
            top_indices = np.argsort(np.abs(current_shap))[-4:][::-1] 
            
            print(f"\n  🚨 ALERT #{i+1} | Risk (CATE): {row['predicted_cate']*100:.1f}%")
            print(f"      Context -> Period: {row.get('period', 'N/A')}, Score Margin: {row.get('score_margin', 'N/A')}")
            print(f"      Top Tactical Drivers (SHAP Values):")
            
            for j in top_indices:
                feature_name = self.translate_to_basketball(X_explain.columns[j])
                impact = current_shap[j]
                # אם ה-SHAP חיובי, הוא מגדיל את הסיכון (הצד האדום בגרף)
                direction = "🔴 DRIVES RISK UP" if impact > 0 else "🟢 Mitigates Risk"
                print(f"       - {feature_name}: {impact:+.4f} ({direction})")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    REPORTS_DIR = os.path.join(base_dir, 'reports')
    MODELS_DIR = os.path.join(base_dir, 'models', 'saved_models')
    
    targets = [
        'target_stop_run_90s', 
        'target_reverse_trend_180s', 
        'target_improve_margin_90s', 
        'target_improve_margin_180s'
    ]
    
    for t in targets:
        dashboard = ExplainabilityDashboard(REPORTS_DIR, MODELS_DIR, t)
        # dashboard.run_analysis() # שמתי בהערה כדי שנקבל פלט נקי
        dashboard.explain_hero_case_text()