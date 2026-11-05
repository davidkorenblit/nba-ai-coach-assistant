"""
Academic Benchmark: DR-Learner (Doubly Robust) vs X-Learner across all 4 targets.
Evaluates model robustness (Triangulation) by comparing CATE predictions from EconML DRLearner and X-Learner.
"""
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from econml.dr import DRLearner
from scipy.stats import spearmanr, pearsonr

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    from pipeline_constants import get_blacklisted_features
except ImportError:
    from models.pipeline_constants import get_blacklisted_features


def load_dataset(data_dir: str):
    """Loads train and test datasets once into memory and filters invalid rows."""
    train_df = pd.read_parquet(os.path.join(data_dir, 'train.parquet'))
    test_df = pd.read_parquet(os.path.join(data_dir, 'test.parquet'))
    
    with open(os.path.join(data_dir, 'split_metadata.json'), 'r') as f:
        metadata = json.load(f)
        
    treat_col = 'is_timeout' if 'is_timeout' in train_df.columns else 'timeout_strategic_weight'
    
    if 'is_garbage_time' in train_df.columns:
        train_df = train_df[train_df['is_garbage_time'] == 0]
        test_df = test_df[test_df['is_garbage_time'] == 0]
        
    blacklist = set(get_blacklisted_features())
    feature_cols = [c for c in metadata['features'] if c in train_df.columns and c not in blacklist and c != treat_col]
    
    # Fill any NaNs in features (e.g. jerseyNumber for team events) to satisfy EconML strict finite check
    train_df[feature_cols] = train_df[feature_cols].fillna(-1.0)
    test_df[feature_cols] = test_df[feature_cols].fillna(-1.0)
    
    return train_df, test_df, feature_cols, treat_col


def predict_xlearner(X_test, target: str, models_dir: str):
    """Calculates CATE predictions from the saved X-Learner models."""
    prop_model = joblib.load(os.path.join(models_dir, f'propensity_{target}.joblib'))
    tau0_model = joblib.load(os.path.join(models_dir, f'tau0_{target}.joblib'))
    tau1_model = joblib.load(os.path.join(models_dir, f'tau1_{target}.joblib'))
    
    prop_score = np.clip(prop_model.predict_proba(X_test)[:, 1], 0.01, 0.99)
    tau0 = tau0_model.predict(X_test)
    tau1 = tau1_model.predict(X_test)
    return (1 - prop_score) * tau0 + prop_score * tau1


def plot_benchmark(plot_data, output_path: str):
    """Draws a 4-panel scatter comparison plot between X-Learner and DR-Learner."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flatten()
    
    for idx, (target, d) in enumerate(plot_data.items()):
        ax = axes[idx]
        ax.scatter(d['xl_sample'], d['dr_sample'], alpha=0.3, color='#2b5c8f', edgecolors='none', s=16)
        
        # Linear fit trendline
        m, b = np.polyfit(d['xl_sample'], d['dr_sample'], 1)
        x_grid = np.linspace(d['xl_sample'].min(), d['xl_sample'].max(), 50)
        ax.plot(x_grid, m * x_grid + b, color='#d9381e', lw=2, label=f"Fit (Spearman rho = {d['rho']:+.2f})")
        
        ax.axhline(0, color='gray', linestyle=':', lw=0.8)
        ax.axvline(0, color='gray', linestyle=':', lw=0.8)
        ax.set_title(f"{target}\nSpearman rho: {d['rho']:+.3f} | Pearson r: {d['r']:+.3f}", fontsize=11, fontweight='bold')
        ax.set_xlabel('X-Learner CATE', fontsize=10)
        ax.set_ylabel('DR-Learner CATE (EconML)', fontsize=10)
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.4)
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def run_benchmark():
    data_dir = os.path.join(ROOT_DIR, 'data', 'processed')
    models_dir = os.path.join(BASE_DIR, 'saved_models')
    reports_dir = os.path.join(ROOT_DIR, 'reports', 'paper_preparation')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    targets = [
        'target_stop_run_90s',
        'target_improve_margin_90s',
        'target_reverse_trend_180s',
        'target_improve_margin_180s'
    ]
    
    print("Loading datasets into memory...")
    train_df, test_df, feature_cols, treat_col = load_dataset(data_dir)
    print(f"Loaded {len(train_df):,} train rows, {len(test_df):,} test rows, {len(feature_cols)} features.")
    
    results = {}
    plot_data = {}
    
    for target in targets:
        print(f"\n[{target}] Fitting DR-Learner with EconML...")
        tr_sub = train_df.dropna(subset=[target, treat_col])
        te_sub = test_df.dropna(subset=[target, treat_col])
        
        X_train = tr_sub[feature_cols]
        T_train = (tr_sub[treat_col] > 0).astype(int).values
        Y_train = tr_sub[target].values
        
        X_test = te_sub[feature_cols]
        T_test = (te_sub[treat_col] > 0).astype(int).values
        Y_test = te_sub[target].values
        
        # Load or train EconML Doubly Robust Learner
        model_save_path = os.path.join(models_dir, f'dr_learner_{target}.joblib')
        retrain = '--retrain' in sys.argv or '--force' in sys.argv
        if os.path.exists(model_save_path) and not retrain:
            print(f" Loading existing model from: {model_save_path}")
            dr = joblib.load(model_save_path)
        else:
            print(f" Training EconML DR-Learner with multi-season data on {len(X_train):,} samples...")
            dr = DRLearner(
                model_propensity=xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, eval_metric='logloss', random_state=42, n_jobs=-1, tree_method='hist'),
                model_regression=xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, eval_metric='rmse', random_state=42, n_jobs=-1, tree_method='hist'),
                model_final=xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, eval_metric='rmse', random_state=42, n_jobs=-1, tree_method='hist'),
                cv=5
            )
            dr.fit(Y=Y_train, T=T_train, X=X_train)
            joblib.dump(dr, model_save_path)
            print(f" Saved model to: {model_save_path}")
        
        # Predictions & Comparisons
        cate_dr = dr.effect(X_test)
        cate_xl = predict_xlearner(X_test, target, models_dir)
        
        rho, _ = spearmanr(cate_xl, cate_dr)
        r, _ = pearsonr(cate_xl, cate_dr)
        ate_dr = float(np.mean(cate_dr))
        ate_xl = float(np.mean(cate_xl))
        
        # Decision Agreement in High-Priority Moments (Top Decile of X-Learner recommendations)
        top10_threshold_xl = np.percentile(cate_xl, 90)
        top10_mask_xl = cate_xl >= top10_threshold_xl
        
        # In moments where X-Learner recommends timeout (Top 10%), does DR-Learner agree the effect is positive?
        pct_dr_agrees_positive = float(np.mean(cate_dr[top10_mask_xl] > 0) * 100)
        # Average effect DR-Learner assigns to X-Learner's top 10% recommendations
        mean_dr_effect_on_xl_top10 = float(np.mean(cate_dr[top10_mask_xl]))
        
        print(f" -> ATE: X-Learner = {ate_xl:+.4f} | DR-Learner = {ate_dr:+.4f}")
        print(f" -> Spearman Rank Correlation: rho = {rho:+.4f} (Pearson r = {r:+.4f})")
        print(f" -> Recommendation Agreement (Top 10%): DR-Learner agrees positive in {pct_dr_agrees_positive:.1f}% of cases (Avg DR effect: {mean_dr_effect_on_xl_top10:+.4f})")
        
        results[target] = {
            "n_test_samples": int(len(X_test)),
            "ate_xlearner": round(ate_xl, 4),
            "ate_dr_learner": round(ate_dr, 4),
            "spearman_rho": round(float(rho), 4),
            "pearson_r": round(float(r), 4),
            "direction_agreement": bool(np.sign(ate_xl) == np.sign(ate_dr)),
            "top10_recommendation_positive_agreement_pct": round(pct_dr_agrees_positive, 2),
            "mean_dr_effect_on_xl_top10_recommendations": round(mean_dr_effect_on_xl_top10, 4)
        }
        
        # Sample for plotting
        sample_idx = np.random.RandomState(42).choice(len(cate_xl), size=min(2000, len(cate_xl)), replace=False)
        plot_data[target] = {
            "xl_sample": cate_xl[sample_idx],
            "dr_sample": cate_dr[sample_idx],
            "rho": float(rho),
            "r": float(r)
        }
        
    # Save benchmark summary JSON
    json_path = os.path.join(reports_dir, 'dr_learner_benchmark_summary.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\n Summary JSON saved: {json_path}")
    
    # Save 4-panel plot
    plot_path = os.path.join(reports_dir, 'dr_vs_xlearner_benchmark.png')
    plot_benchmark(plot_data, plot_path)
    print(f" 4-Panel Plot saved: {plot_path}")


if __name__ == '__main__':
    run_benchmark()
