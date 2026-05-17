import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

class RecommendationEngine:
    def __init__(self, data_path: str, target_col: str = 'target_stop_run_90s'):
        self.data_path = data_path
        self.target_col = target_col
        self.treatment_col = 'timeout_strategic_weight'
        self.penalty_col = 'target_danger_penalty'
        
        # Models
        self.propensity_model = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
        self.mu0_model = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
        self.mu1_model = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
        self.tau0_model = xgb.XGBRegressor(random_state=42)
        self.tau1_model = xgb.XGBRegressor(random_state=42)
        
        self.dynamic_threshold = None

    def load_and_train(self):
        """Quickly retrains the X-Learner so we can do inference on the Test set."""
        print(f"Loading data and training X-Learner for {self.target_col}...")
        df = pd.read_parquet(self.data_path)
        
        # --- 1. סינון זמן זבל (Garbage Time) ---
        if 'period' in df.columns and 'seconds_remaining' in df.columns and 'score_margin' in df.columns:
            garbage_mask = (
                ((df['period'] == 4) & (df['seconds_remaining'] <= 180) & (df['score_margin'].abs() >= 15)) |
                ((df['period'] == 4) & (df['seconds_remaining'] > 180) & (df['score_margin'].abs() >= 30))
            )
            original_len = len(df)
            df = df[~garbage_mask].copy()
            print(f"Filtered {original_len - len(df)} Garbage Time possessions.")

        leakage_cols = [
            'seconds_remaining', 'orderNumber', 'actionNumber', 'actionId',
            'explosiveness_index', 'teamId', 'possession_id', 'cum_pointsTotal',
            'possession', 'scoreHome', 'scoreAway', 'reboundTotal', 
            'reboundDefensiveTotal', 'reboundOffensiveTotal', 'cum_reboundDefensiveTotal',
            'personId', 'shotActionNumber', 'officialId', 'x', 'y',
            'xLegacy', 'yLegacy', 'jumpBallRecoverdPersonId', 'jumpBallWonPersonId', 
            'jumpBallLostPersonId', 'foulDrawnPersonId', 'is_poss_change', 
            'play_duration', 'isTargetScoreLastPeriod', 'pointsTotal', 
            'turnoverTotal', 'foulPersonalTotal', 'foulTechnicalTotal', 'shot_clock_estimated'
        ]
        
        # --- 3א. וידוא שהענישה וה-gameId נשארים לשימוש בהמשך ---
        targets_to_drop = [c for c in df.columns if c.startswith('target_') and c not in [self.target_col, self.penalty_col]]
        cols_to_drop = targets_to_drop + [c for c in leakage_cols if c in df.columns]
        df_clean = df.drop(columns=cols_to_drop).copy()
        
        df_clean[self.treatment_col] = (df_clean[self.treatment_col] > 0).astype(int)
        
        subset_drop = [self.target_col, self.treatment_col]
        if self.penalty_col in df_clean.columns:
            subset_drop.append(self.penalty_col)
        df_clean = df_clean.dropna(subset=subset_drop)
        
        # הפרדת נתונים למודל (X) לעומת מזהים טכניים (P, G)
        cols_to_drop_x = [self.target_col, self.treatment_col]
        if self.penalty_col in df_clean.columns:
            cols_to_drop_x.append(self.penalty_col)
            P = df_clean[self.penalty_col]
        else:
            P = pd.Series(0, index=df_clean.index) 

        if 'gameId' in df_clean.columns:
            G = df_clean['gameId']
            cols_to_drop_x.append('gameId')
        else:
            G = pd.Series(0, index=df_clean.index)
            
        X = df_clean.drop(columns=cols_to_drop_x)
        T = df_clean[self.treatment_col]
        Y = df_clean[self.target_col]
        
        # פיצול מרובע: כולל ענישה ו-gameId
        split_data = train_test_split(
            X, T, Y, P, G, test_size=0.2, random_state=42, stratify=Y
        )
        self.X_train, self.X_test, self.T_train, self.T_test, self.Y_train, self.Y_test, self.P_train, self.P_test, self.G_train, self.G_test = split_data
        
        # --- Fast X-Learner Training ---
        self.propensity_model.fit(self.X_train, self.T_train)
        
        X0, Y0 = self.X_train[self.T_train == 0], self.Y_train[self.T_train == 0]
        X1, Y1 = self.X_train[self.T_train == 1], self.Y_train[self.T_train == 1]
        self.mu0_model.fit(X0, Y0)
        self.mu1_model.fit(X1, Y1)
        
        D0 = self.mu1_model.predict_proba(X0)[:, 1] - Y0 
        D1 = Y1 - self.mu0_model.predict_proba(X1)[:, 1] 
        self.tau0_model.fit(X0, D0)
        self.tau1_model.fit(X1, D1)

    def generate_recommendations(self):
        print("\n--- 🧠 Running Inference Engine on Test Set ---")
        tau0_pred = self.tau0_model.predict(self.X_test)
        tau1_pred = self.tau1_model.predict(self.X_test)
        g_x_eval = np.clip(self.propensity_model.predict_proba(self.X_test)[:, 1], 0.01, 0.99)
        
        raw_cate_scores = g_x_eval * tau0_pred + (1 - g_x_eval) * tau1_pred
        
        # --- 2. חיתוך מתמטי (Clipping) ---
        cate_scores = np.clip(raw_cate_scores, 0.0, 1.0)
        
        results_df = self.X_test.copy()
        results_df['predicted_cate'] = cate_scores
        results_df['actual_treatment'] = self.T_test
        results_df[self.penalty_col] = self.P_test 
        results_df['gameId'] = self.G_test # הצמדת המזהה לתוצאות
        
        self.dynamic_threshold = np.percentile(cate_scores, 95)
        recommendations = results_df[results_df['predicted_cate'] >= self.dynamic_threshold].copy()
        
        print(f"Total Test Possessions: {len(results_df):,}")
        print(f"Dynamic Threshold (Top 5%): {self.dynamic_threshold*100:.2f}% Expected Impact")
        print(f"🚨 'TIMEOUT NOW!' Alerts Triggered: {len(recommendations):,}")
        
        ignored_alerts = recommendations[recommendations['actual_treatment'] == 0]
        punished_alerts = ignored_alerts[ignored_alerts[self.penalty_col] == 1]
        
        if len(ignored_alerts) > 0:
            penalty_rate = (len(punished_alerts) / len(ignored_alerts)) * 100
            print(f"⚠️ Validation: Coaches ignored the alert {len(ignored_alerts)} times.")
            print(f"   -> Out of those, they were punished (Penalty=1) {len(punished_alerts)} times ({penalty_rate:.1f}% hit rate).")

        return results_df, recommendations

    def generate_outputs(self, full_results, recommendations, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        plt.hist(full_results['predicted_cate'] * 100, bins=50, color='royalblue', edgecolor='black', alpha=0.7)
        plt.axvline(self.dynamic_threshold * 100, color='red', linestyle='dashed', linewidth=2, label=f'Top 5% Threshold ({self.dynamic_threshold*100:.1f}%)')
        plt.title(f'Distribution of Predicted Timeout Impact (CATE %)\nTest Set - {self.target_col}')
        plt.xlabel('Expected Improvement in Stopping Run (%)')
        plt.ylabel('Number of Possessions')
        plt.legend()
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, f'cate_threshold_distribution_{self.target_col}.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"📊 Distribution Graph saved to: {plot_path}")
        
        if len(recommendations) > 0:
            recommendations = recommendations.sort_values(by='predicted_cate', ascending=False)
            
            # --- הוספת gameId לרשימת העמודות ב-CSV ---
            explain_cols = [
                'gameId', 'predicted_cate', 'actual_treatment', self.penalty_col, 'period', 'score_margin', 
                'event_momentum_val', 'momentum_streak_rolling', 
                'home_cum_fatigue', 'away_cum_fatigue'
            ]
            
            final_cols = [c for c in explain_cols if c in recommendations.columns]
            
            report_path = os.path.join(output_dir, f'timeout_recommendations_report_{self.target_col}.csv')
            recommendations[final_cols].to_csv(report_path, index=False)
            print(f"📝 Explainability Report (Alerts) saved to: {report_path}")
        else:
            print("⚠️ No recommendations crossed the threshold.")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(base_dir, 'data', 'processed', 'train.parquet')
    REPORTS_DIR = os.path.join(base_dir, 'reports')
    
    targets = [
        'target_stop_run_90s', 
        'target_reverse_trend_180s', 
        'target_improve_margin_90s', 
        'target_improve_margin_180s'
    ]
    
    for target in targets:
        print("\n" + "="*60)
        engine = RecommendationEngine(DATA_PATH, target_col=target)
        engine.load_and_train()
        full_df, rec_df = engine.generate_recommendations()
        engine.generate_outputs(full_df, rec_df, REPORTS_DIR)
        print("="*60)