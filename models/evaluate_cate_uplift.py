import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure models directory is accessible for local imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    from pipeline_constants import get_blacklisted_features
except ImportError:
    from models.pipeline_constants import get_blacklisted_features

class CATEUpliftEvaluator:
    """
    Evaluates Out-of-Sample CATE Uplift by Deciles on the Sterile Test Set.
    Generates both an intuitive visual diagnostic plot and a structured JSON summary.
    """
    def __init__(self, data_dir: str = None, models_dir: str = None, target_col: str = 'target_stop_run_90s'):
        root_dir = os.path.dirname(BASE_DIR)
        self.data_dir = data_dir or os.path.join(root_dir, 'data', 'processed')
        self.models_dir = models_dir or os.path.join(BASE_DIR, 'saved_models')
        self.reports_dir = os.path.join(root_dir, 'reports', 'paper_preparation')
        os.makedirs(self.reports_dir, exist_ok=True)
        
        self.target_col = target_col
        self.treatment_col = 'is_timeout'
        
        self.propensity_model = None
        self.tau0_model = None
        self.tau1_model = None
        
        self.X_test = None
        self.T_test = None
        self.Y_test = None
        self.games_test = None

    def load_models(self):
        """Loads trained estimators from disk."""
        prop_path = os.path.join(self.models_dir, f'propensity_{self.target_col}.joblib')
        tau0_path = os.path.join(self.models_dir, f'tau0_{self.target_col}.joblib')
        tau1_path = os.path.join(self.models_dir, f'tau1_{self.target_col}.joblib')
        
        if not (os.path.exists(prop_path) and os.path.exists(tau0_path) and os.path.exists(tau1_path)):
            raise FileNotFoundError(
                f"Missing saved models for target '{self.target_col}'. "
                f"Please ensure models exist in: {self.models_dir}"
            )
            
        print(f"Loading trained models from: {self.models_dir}")
        self.propensity_model = joblib.load(prop_path)
        self.tau0_model = joblib.load(tau0_path)
        self.tau1_model = joblib.load(tau1_path)
        print("Models successfully loaded.")

    def load_test_data(self):
        """Loads sterile test set and filters blacklisted features."""
        test_path = os.path.join(self.data_dir, 'test.parquet')
        metadata_path = os.path.join(self.data_dir, 'split_metadata.json')
        
        print(f"Loading test set from: {test_path}")
        df = pd.read_parquet(test_path)
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        blacklist = get_blacklisted_features()
        feature_cols = [c for c in metadata['features'] if c in df.columns and c not in blacklist]
        
        if 'is_garbage_time' in df.columns:
            df = df[df['is_garbage_time'] == 0]
            
        # Treatment column resolution
        if self.treatment_col not in df.columns:
            if 'is_timeout' in df.columns:
                self.treatment_col = 'is_timeout'
            elif 'timeout_strategic_weight' in df.columns:
                self.treatment_col = 'timeout_strategic_weight'
                
        df = df.dropna(subset=[self.target_col, self.treatment_col])
        
        self.T_test = (df[self.treatment_col] > 0).astype(int).values
        self.Y_test = df[self.target_col].values
        self.games_test = df['gameId'].values if 'gameId' in df.columns else np.arange(len(df))
        
        X_cols = [c for c in feature_cols if c != self.treatment_col and c != self.target_col]
        self.X_test = df[X_cols]
        print(f"Test data ready. Features: {len(X_cols)} | Samples: {len(self.X_test):,}")

    def predict_cate(self):
        """Estimates CATE on the test set using X-Learner logic."""
        print("Predicting CATE scores on test set...")
        g_x = np.clip(self.propensity_model.predict_proba(self.X_test)[:, 1], 0.01, 0.99)
        tau0_pred = self.tau0_model.predict(self.X_test)
        tau1_pred = self.tau1_model.predict(self.X_test)
        
        cate = (1 - g_x) * tau0_pred + g_x * tau1_pred
        return cate, g_x

    def run_decile_evaluation(self, n_deciles=10):
        """Partitions test predictions into deciles and computes empirical uplifts."""
        cate, g_x = self.predict_cate()
        
        eval_df = pd.DataFrame({
            'cate': cate,
            'propensity': g_x,
            'treatment': self.T_test,
            'outcome': self.Y_test,
            'gameId': self.games_test
        })
        
        # Split into deciles based on predicted CATE
        eval_df['decile'] = pd.qcut(
            eval_df['cate'], 
            q=n_deciles, 
            labels=[f"D{i+1}" for i in range(n_deciles)]
        )
        
        decile_stats = []
        cumulative_treated_outcome = 0.0
        cumulative_control_outcome = 0.0
        cumulative_treated_count = 0
        cumulative_control_count = 0
        
        # Sort deciles descending (D10 = highest predicted priority)
        decile_order = [f"D{i+1}" for i in range(n_deciles)]
        
        for d in decile_order:
            sub = eval_df[eval_df['decile'] == d]
            n_tot = len(sub)
            
            ctrl = sub[sub['treatment'] == 0]
            treat = sub[sub['treatment'] == 1]
            
            mean_y_ctrl = float(ctrl['outcome'].mean()) if len(ctrl) > 0 else 0.0
            mean_y_treat = float(treat['outcome'].mean()) if len(treat) > 0 else 0.0
            actual_uplift = mean_y_treat - mean_y_ctrl
            
            decile_stats.append({
                "decile": d,
                "n_samples": n_tot,
                "n_control": len(ctrl),
                "n_treated": len(treat),
                "treatment_rate_pct": round((len(treat) / n_tot) * 100, 2) if n_tot > 0 else 0.0,
                "mean_predicted_cate": round(float(sub['cate'].mean()), 4),
                "mean_y_control": round(mean_y_ctrl, 4),
                "mean_y_treated": round(mean_y_treat, 4),
                "empirical_uplift": round(actual_uplift, 4)
            })
            
        # Convert to DataFrame
        stats_df = pd.DataFrame(decile_stats)
        
        # Calculate monotonic correlation (Spearman rank correlation between decile and actual uplift)
        spearman_corr = stats_df['mean_predicted_cate'].corr(stats_df['empirical_uplift'], method='spearman')
        is_monotonic = bool(spearman_corr > 0.5)
        
        top_decile_uplift = float(stats_df.iloc[-1]['empirical_uplift'])
        bottom_decile_uplift = float(stats_df.iloc[0]['empirical_uplift'])
        uplift_spread = top_decile_uplift - bottom_decile_uplift
        
        summary = {
            "target": self.target_col,
            "n_test_samples": len(eval_df),
            "spearman_rank_correlation": round(float(spearman_corr), 4),
            "is_monotonic_trend": is_monotonic,
            "bottom_decile_uplift_D1": bottom_decile_uplift,
            "top_decile_uplift_D10": top_decile_uplift,
            "uplift_spread_D10_minus_D1": round(uplift_spread, 4),
            "deciles": decile_stats
        }
        
        # Save JSON
        json_path = os.path.join(self.reports_dir, f'uplift_deciles_{self.target_col}.json')
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=4)
        print(f"✅ Saved JSON Summary to: {json_path}")
        
        # Plot Visual Graph
        self.plot_visual_uplift(stats_df, spearman_corr, json_path)
        
        return summary

    def plot_visual_uplift(self, stats_df, spearman_corr, json_path):
        """Generates an intuitive, high-resolution diagnostic plot."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 1. Bar Chart: Empirical Uplift per Decile
        deciles = stats_df['decile'].values
        uplifts = stats_df['empirical_uplift'].values
        colors = ['mediumseagreen' if u >= 0 else 'crimson' for u in uplifts]
        
        bars = ax1.bar(deciles, uplifts, color=colors, edgecolor='black', alpha=0.85, width=0.6)
        ax1.axhline(0, color='black', linewidth=1.2, linestyle='--')
        
        # Value labels above/below bars
        for bar in bars:
            height = bar.get_height()
            va_align = 'bottom' if height >= 0 else 'top'
            offset = 0.01 if height >= 0 else -0.01
            ax1.annotate(
                f'{height:+.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height + offset),
                xytext=(0, 2 if height >= 0 else -8),
                textcoords="offset points",
                ha='center', va=va_align,
                fontsize=10, fontweight='bold'
            )
            
        ax1.set_title(
            f'Empirical Timeout Uplift by Predicted CATE Decile\nTarget: {self.target_col}',
            fontsize=13, fontweight='bold'
        )
        ax1.set_xlabel('Predicted CATE Decile (D1 = Lowest Priority → D10 = Highest Priority)', fontsize=11)
        ax1.set_ylabel('Empirical Uplift (Treated - Control Outcome)', fontsize=11)
        ax1.grid(axis='y', linestyle=':', alpha=0.6)
        
        # 2. Line Chart: Predicted CATE vs Actual Empirical Uplift
        pred_cate = stats_df['mean_predicted_cate'].values
        
        ax2.plot(deciles, pred_cate, marker='o', linewidth=2.2, color='royalblue', label='Predicted CATE (Model)')
        ax2.plot(deciles, uplifts, marker='s', linewidth=2.2, color='darkorange', label='Actual Empirical Uplift (Test Data)')
        ax2.axhline(0, color='gray', linestyle=':', linewidth=1)
        
        ax2.set_title(
            f'Model Alignment: Predicted vs Actual Impact\nSpearman Rank Correlation: r = {spearman_corr:+.2f}',
            fontsize=13, fontweight='bold'
        )
        ax2.set_xlabel('Deciles', fontsize=11)
        ax2.set_ylabel('Treatment Effect Value', fontsize=11)
        ax2.legend(loc='best', fontsize=11)
        ax2.grid(True, linestyle=':', alpha=0.6)
        
        plt.tight_layout()
        plot_path = os.path.join(self.reports_dir, f'uplift_deciles_{self.target_col}.png')
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"📊 Saved Visual Uplift Graph to: {plot_path}")

def main():
    target = sys.argv[1] if len(sys.argv) > 1 else 'target_stop_run_90s'
    print(f"\n==================================================")
    print(f"Running CATE Decile Uplift Validation for: {target}")
    print(f"==================================================")
    
    evaluator = CATEUpliftEvaluator(target_col=target)
    evaluator.load_models()
    evaluator.load_test_data()
    evaluator.run_decile_evaluation()

if __name__ == '__main__':
    main()
