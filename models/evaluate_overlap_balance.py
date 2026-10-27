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

class OverlapBalanceEvaluator:
    """
    Evaluates Propensity Score Overlap, Common Support, and Covariate Balance (Love Plot).
    Generates academic diagnostics directly answering Dr. Shapira's Note #2.
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
        
        self.X = None
        self.T = None
        self.g_x = None

    def load_data_and_model(self):
        """Loads trained propensity model and dataset."""
        prop_path = os.path.join(self.models_dir, f'propensity_{self.target_col}.joblib')
        if not os.path.exists(prop_path):
            raise FileNotFoundError(f"Missing propensity model: {prop_path}")
            
        print(f"Loading trained propensity model from: {prop_path}")
        self.propensity_model = joblib.load(prop_path)
        
        # We evaluate balance on the training set where weights are fitted
        train_path = os.path.join(self.data_dir, 'train.parquet')
        metadata_path = os.path.join(self.data_dir, 'split_metadata.json')
        
        print(f"Loading dataset from: {train_path}")
        df = pd.read_parquet(train_path)
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        blacklist = get_blacklisted_features()
        feature_cols = [c for c in metadata['features'] if c in df.columns and c not in blacklist]
        
        if 'is_garbage_time' in df.columns:
            df = df[df['is_garbage_time'] == 0]
            
        # Resolve treatment column
        if self.treatment_col not in df.columns:
            if 'is_timeout' in df.columns:
                self.treatment_col = 'is_timeout'
            elif 'timeout_strategic_weight' in df.columns:
                self.treatment_col = 'timeout_strategic_weight'
                
        df = df.dropna(subset=[self.target_col, self.treatment_col])
        
        self.T = (df[self.treatment_col] > 0).astype(int).values
        X_cols = [c for c in feature_cols if c != self.treatment_col and c != self.target_col]
        self.X = df[X_cols]
        print(f"Data ready. Samples: {len(self.X):,} | Features: {len(X_cols)}")

    def compute_propensity_and_overlap(self):
        """Computes propensity scores and common support statistics."""
        print("Computing propensity scores e(X)...")
        # Raw predicted probability
        raw_prob = self.propensity_model.predict_proba(self.X)[:, 1]
        self.g_x = np.clip(raw_prob, 0.01, 0.99)
        
        g_treat = self.g_x[self.T == 1]
        g_ctrl = self.g_x[self.T == 0]
        
        # Empirical Common Support bounds
        cs_min = max(float(np.min(g_treat)), float(np.min(g_ctrl)))
        cs_max = min(float(np.max(g_treat)), float(np.max(g_ctrl)))
        
        in_cs = (self.g_x >= cs_min) & (self.g_x <= cs_max)
        cs_pct_total = float(np.mean(in_cs) * 100)
        cs_pct_treat = float(np.mean((g_treat >= cs_min) & (g_treat <= cs_max)) * 100)
        cs_pct_ctrl = float(np.mean((g_ctrl >= cs_min) & (g_ctrl <= cs_max)) * 100)
        
        # Crump et al. (2009) [0.05, 0.95]
        crump_in = (self.g_x >= 0.05) & (self.g_x <= 0.95)
        crump_pct = float(np.mean(crump_in) * 100)
        
        overlap_metrics = {
            "empirical_cs_min": round(cs_min, 4),
            "empirical_cs_max": round(cs_max, 4),
            "samples_in_common_support_pct": round(cs_pct_total, 2),
            "treated_in_common_support_pct": round(cs_pct_treat, 2),
            "control_in_common_support_pct": round(cs_pct_ctrl, 2),
            "crump_trimming_005_095_pct": round(crump_pct, 2),
            "positivity_assumption_verified": bool(cs_pct_total > 95.0),
            "n_total": len(self.X),
            "n_treated": int(np.sum(self.T == 1)),
            "n_control": int(np.sum(self.T == 0))
        }
        
        print("\n--- Overlap & Common Support Summary ---")
        print(f"Common Support Interval: [{cs_min:.4f}, {cs_max:.4f}]")
        print(f"Samples within Common Support: {cs_pct_total:.2f}% (Treated: {cs_pct_treat:.2f}%, Control: {cs_pct_ctrl:.2f}%)")
        print(f"Crump Trimmed [0.05, 0.95]: {crump_pct:.2f}%")
        print(f"Positivity Status: {'VERIFIED' if overlap_metrics['positivity_assumption_verified'] else 'VIOLATED'}")
        
        return overlap_metrics, g_treat, g_ctrl, cs_min, cs_max

    def compute_covariate_balance(self):
        """Computes Standardized Mean Differences (SMD) before and after IPW weighting."""
        print("\nComputing Standardized Mean Differences (SMD) across all features...")
        weights = np.where(
            self.T == 1,
            1.0 / self.g_x,
            1.0 / (1.0 - self.g_x)
        )
        
        smd_raw = {}
        smd_weighted = {}
        
        for col in self.X.columns:
            x_val = pd.to_numeric(self.X[col], errors='coerce').fillna(0).values
            x0, x1 = x_val[self.T == 0], x_val[self.T == 1]
            var0, var1 = np.var(x0), np.var(x1)
            pooled_std = np.sqrt((var0 + var1) / 2.0)
            if pooled_std < 1e-8:
                continue
                
            # Unweighted SMD
            raw_diff = np.abs(np.mean(x1) - np.mean(x0))
            smd_raw[col] = raw_diff / pooled_std
            
            # Weighted SMD (IPW)
            w0, w1 = weights[self.T == 0], weights[self.T == 1]
            mean0_w = np.sum(x0 * w0) / np.sum(w0)
            mean1_w = np.sum(x1 * w1) / np.sum(w1)
            weighted_diff = np.abs(mean1_w - mean0_w)
            smd_weighted[col] = weighted_diff / pooled_std
            
        unadjusted_mean_smd = float(np.mean(list(smd_raw.values())))
        weighted_mean_smd = float(np.mean(list(smd_weighted.values())))
        balanced_under_0_1_pct = float(np.mean([v < 0.1 for v in smd_weighted.values()]) * 100)
        
        print(f"Unadjusted Mean SMD: {unadjusted_mean_smd:.4f}")
        print(f"Weighted Mean SMD (IPW): {weighted_mean_smd:.4f}")
        print(f"Features Balanced under 0.1 threshold: {balanced_under_0_1_pct:.1f}%")
        
        sorted_features = sorted(smd_raw.keys(), key=lambda k: smd_raw[k], reverse=True)
        
        return smd_raw, smd_weighted, sorted_features, unadjusted_mean_smd, weighted_mean_smd, balanced_under_0_1_pct

    def plot_overlap(self, g_treat, g_ctrl, cs_min, cs_max, cs_pct_total):
        """Generates high-resolution Propensity Overlap density plot."""
        plt.figure(figsize=(11, 6))
        plt.hist(g_ctrl, bins=50, density=True, alpha=0.55, label='Control (No Timeout)', color='steelblue')
        plt.hist(g_treat, bins=50, density=True, alpha=0.55, label='Treated (Timeout Taken)', color='crimson')
        
        plt.axvline(cs_min, color='darkgreen', linestyle='--', linewidth=1.5, label=f'Common Support Min ({cs_min:.3f})')
        plt.axvline(cs_max, color='darkgreen', linestyle='--', linewidth=1.5, label=f'Common Support Max ({cs_max:.3f})')
        plt.axvline(0.01, color='gray', linestyle=':', label='Trimming Bounds [0.01, 0.99]')
        plt.axvline(0.99, color='gray', linestyle=':')
        
        plt.title(
            f'Propensity Score Overlap & Common Support\nTarget: {self.target_col} | Coverage: {cs_pct_total:.1f}%',
            fontsize=13, fontweight='bold'
        )
        plt.xlabel('Estimated Propensity Score e(X) = P(T=1 | X)', fontsize=11)
        plt.ylabel('Density', fontsize=11)
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        overlap_plot_path = os.path.join(self.reports_dir, f'overlap_propensity_{self.target_col}.png')
        plt.savefig(overlap_plot_path, dpi=300)
        plt.close()
        print(f"📊 Saved Overlap Histogram: {overlap_plot_path}")

    def plot_love_plot(self, smd_raw, smd_weighted, sorted_features):
        """Generates Love Plot showing covariate balance before and after IPW."""
        top_features = sorted_features[:25]
        plt.figure(figsize=(10, max(8, len(top_features) * 0.35)))
        y_pos = np.arange(len(top_features))
        raw_vals = [smd_raw[f] for f in top_features]
        adj_vals = [smd_weighted[f] for f in top_features]
        
        plt.plot(raw_vals, y_pos, 'o', color='crimson', label='Unadjusted (Raw)', markersize=7)
        plt.plot(adj_vals, y_pos, 'o', color='seagreen', label='Adjusted (IPW Weighted)', markersize=7)
        
        for i, (r, a) in enumerate(zip(raw_vals, adj_vals)):
            plt.hlines(y_pos[i], min(r, a), max(r, a), color='gray', linestyle=':', alpha=0.6)
            
        plt.axvline(0.1, color='black', linestyle='--', linewidth=1.2, label='Balance Threshold (SMD = 0.1)')
        plt.yticks(y_pos, top_features)
        plt.gca().invert_yaxis()
        plt.xlabel('Absolute Standardized Mean Difference (SMD)', fontsize=11)
        plt.title(
            f'Love Plot: Covariate Balance Before & After Weighting\nTarget: {self.target_col}',
            fontsize=13, fontweight='bold'
        )
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        love_plot_path = os.path.join(self.reports_dir, f'love_plot_{self.target_col}.png')
        plt.savefig(love_plot_path, dpi=300)
        plt.close()
        print(f"📊 Saved Love Plot: {love_plot_path}")

    def run_evaluation(self):
        """Executes full overlap and balance pipeline and exports JSON and Plots."""
        self.load_data_and_model()
        overlap_metrics, g_treat, g_ctrl, cs_min, cs_max = self.compute_propensity_and_overlap()
        smd_raw, smd_weighted, sorted_features, unadj_smd, w_smd, bal_pct = self.compute_covariate_balance()
        
        # Save plots
        self.plot_overlap(g_treat, g_ctrl, cs_min, cs_max, overlap_metrics['samples_in_common_support_pct'])
        self.plot_love_plot(smd_raw, smd_weighted, sorted_features)
        
        # Save full JSON
        summary = {
            "target": self.target_col,
            "total_features_evaluated": len(smd_raw),
            "unadjusted_mean_smd": round(unadj_smd, 4),
            "weighted_mean_smd": round(w_smd, 4),
            "features_balanced_under_0_1_pct": round(bal_pct, 2),
            "common_support": overlap_metrics,
            "top_imbalanced_features_before_weighting": {
                f: {
                    "raw_smd": round(float(smd_raw[f]), 4),
                    "weighted_smd": round(float(smd_weighted[f]), 4),
                    "is_balanced_under_0_1": bool(smd_weighted[f] < 0.1)
                }
                for f in sorted_features[:15]
            }
        }
        
        json_path = os.path.join(self.reports_dir, f'covariate_balance_{self.target_col}.json')
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=4)
        print(f"✅ Saved Comprehensive Covariate Balance & Overlap JSON to: {json_path}")
        return summary

def main():
    target = sys.argv[1] if len(sys.argv) > 1 else 'target_stop_run_90s'
    print(f"\n==================================================")
    print(f"Running Overlap & Covariate Balance Evaluation for: {target}")
    print(f"==================================================")
    
    evaluator = OverlapBalanceEvaluator(target_col=target)
    evaluator.run_evaluation()

if __name__ == '__main__':
    main()
