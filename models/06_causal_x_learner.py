import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, mean_squared_error
import matplotlib.pyplot as plt
import os
import json
import sys
try:
    from pipeline_constants import get_blacklisted_features
except ImportError:
    from models.pipeline_constants import get_blacklisted_features

class NBACausalLearner:
    def __init__(self, data_path: str, target_col: str = 'target_stop_run_90s', treatment_col: str = 'is_timeout'):
        self.data_path = data_path
        self.target_col = target_col
        self.treatment_col = treatment_col
        
        self.X_train, self.X_test = None, None
        self.T_train, self.T_test = None, None
        self.Y_train, self.Y_test = None, None
        
        self.auc = None
        self.ate = None
        
        # Out-of-fold imputed effects
        self.oof_D0 = None
        self.oof_D1 = None
        
        # Propensity model is a classifier (Treatment is binary: timeout taken or not)
        self.propensity_model = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
        
        # Outcome models are regressors (Target is continuous impact)
        self.mu0_model = xgb.XGBRegressor(eval_metric='rmse', random_state=42)
        self.mu1_model = xgb.XGBRegressor(eval_metric='rmse', random_state=42)
        
        self.tau0_model = xgb.XGBRegressor(eval_metric='rmse', random_state=42)
        self.tau1_model = xgb.XGBRegressor(eval_metric='rmse', random_state=42)

    def save_models(self, save_dir=None):
        if save_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            save_dir = os.path.join(base_dir, 'models', 'saved_models')
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(self.propensity_model, os.path.join(save_dir, f'propensity_{self.target_col}.joblib'))
        joblib.dump(self.mu0_model, os.path.join(save_dir, f'mu0_{self.target_col}.joblib'))
        joblib.dump(self.mu1_model, os.path.join(save_dir, f'mu1_{self.target_col}.joblib'))
        joblib.dump(self.tau0_model, os.path.join(save_dir, f'tau0_{self.target_col}.joblib'))
        joblib.dump(self.tau1_model, os.path.join(save_dir, f'tau1_{self.target_col}.joblib'))
        print(f"💾 All 5 models for '{self.target_col}' successfully saved to: {save_dir}")

    def load_and_prepare_data(self):
        print(f"\n--- Processing Target: {self.target_col} ---")
        
        # Load genuine Train and Test sets
        test_path = self.data_path.replace('train.parquet', 'test.parquet')
        print(f"Loading TRAIN from: {self.data_path}")
        print(f"Loading TEST from: {test_path}")
        
        train_df = pd.read_parquet(self.data_path)
        test_df = pd.read_parquet(test_path)
        
        # Read clean features from JSON metadata
        metadata_path = os.path.join(os.path.dirname(self.data_path), 'split_metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        blacklist = get_blacklisted_features()   
        feature_cols = [c for c in metadata['features'] if c in train_df.columns and c not in blacklist]

        # Fallback for treatment col if default is missing
        if self.treatment_col not in train_df.columns:
            if 'is_timeout' in train_df.columns:
                self.treatment_col = 'is_timeout'
            elif 'timeout_strategic_weight' in train_df.columns:
                self.treatment_col = 'timeout_strategic_weight'

        # Filter Garbage Time
        if 'is_garbage_time' in train_df.columns:
            train_df = train_df[train_df['is_garbage_time'] == 0]
            test_df = test_df[test_df['is_garbage_time'] == 0]
        
        def prepare_split(df):
            # Drop NaN in specific target or treatment (handles right-censored boundary rows)
            df = df.dropna(subset=[self.target_col, self.treatment_col])
            
            # Binarize treatment
            T = (df[self.treatment_col] > 0).astype(int)
            Y = df[self.target_col]
            
            # Keep only clean features
            X_cols = [c for c in feature_cols if c != self.treatment_col and c != self.target_col]
            X = df[X_cols]
            
            games = df['gameId'].values if 'gameId' in df.columns else np.arange(len(df))
            return X, T, Y, games

        self.X_train, self.T_train, self.Y_train, self.games_train = prepare_split(train_df)
        self.X_test, self.T_test, self.Y_test, self.games_test = prepare_split(test_df)
        
        print(f"Data ready. Clean Features: {len(self.X_train.columns)}. Train: {self.X_train.shape[0]} | Test: {self.X_test.shape[0]}")
        print("---------------------------------\n")

    def stage_1_and_2_cross_fitting(self, n_splits=5):
        """
        K-Fold Cross-Fitting (Chernozhukov et al. / Double ML standard):
        Splits training data into K stratified folds. For each fold, trains propensity e(x) 
        and outcome models (mu0, mu1) on K-1 folds, and predicts strictly Out-of-Fold 
        to impute counterfactuals D0 and D1 without In-Sample Overfitting bias.
        """
        print(f"Stages 1 & 2: Stratified Group K-Fold Cross-Fitting (K={n_splits}, Grouped by gameId) for OOF Imputation...")
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # Out-of-fold containers
        oof_g_x = np.zeros(len(self.X_train))
        oof_D0 = np.full(len(self.X_train), np.nan)
        oof_D1 = np.full(len(self.X_train), np.nan)
        
        X_tr = self.X_train.values if isinstance(self.X_train, pd.DataFrame) else self.X_train
        T_tr = self.T_train.values if isinstance(self.T_train, pd.Series) else self.T_train
        Y_tr = self.Y_train.values if isinstance(self.Y_train, pd.Series) else self.Y_train
        
        for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_tr, T_tr, groups=self.games_train), 1):
            X_fold_tr, X_fold_val = X_tr[train_idx], X_tr[val_idx]
            T_fold_tr, T_fold_val = T_tr[train_idx], T_tr[val_idx]
            Y_fold_tr, Y_fold_val = Y_tr[train_idx], Y_tr[val_idx]
            
            # QA: Verify strictly zero game leakage between train and val fold
            train_games_set = set(self.games_train[train_idx])
            val_games_set = set(self.games_train[val_idx])
            overlap = train_games_set.intersection(val_games_set)
            assert len(overlap) == 0, f"Critical leakage in Fold {fold}: {len(overlap)} games leaked!"
            
            # 1. Propensity model on fold
            fold_prop = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
            fold_prop.fit(X_fold_tr, T_fold_tr)
            oof_g_x[val_idx] = fold_prop.predict_proba(X_fold_val)[:, 1]
            
            # 2. Outcome models on fold
            fold_mu0 = xgb.XGBRegressor(eval_metric='rmse', random_state=42)
            fold_mu1 = xgb.XGBRegressor(eval_metric='rmse', random_state=42)
            
            ctrl_mask_tr = (T_fold_tr == 0)
            treat_mask_tr = (T_fold_tr == 1)
            
            fold_mu0.fit(X_fold_tr[ctrl_mask_tr], Y_fold_tr[ctrl_mask_tr])
            fold_mu1.fit(X_fold_tr[treat_mask_tr], Y_fold_tr[treat_mask_tr])
            
            # Impute Out-of-Fold counterfactuals for val_idx
            ctrl_mask_val = (T_fold_val == 0)
            treat_mask_val = (T_fold_val == 1)
            
            # For control units in val: D0 = mu1(X_ctrl) - Y_ctrl
            if np.any(ctrl_mask_val):
                oof_D0[val_idx[ctrl_mask_val]] = fold_mu1.predict(X_fold_val[ctrl_mask_val]) - Y_fold_val[ctrl_mask_val]
            
            # For treated units in val: D1 = Y_treat - mu0(X_treat)
            if np.any(treat_mask_val):
                oof_D1[val_idx[treat_mask_val]] = Y_fold_val[treat_mask_val] - fold_mu0.predict(X_fold_val[treat_mask_val])
                
        # Clip OOF propensity
        oof_g_x = np.clip(oof_g_x, 0.01, 0.99)
        self.g_x_train = oof_g_x
        
        # Fit final full-sample models for test inference and artifact saving
        print("   Fitting final Propensity and Outcome models on full training set for inference & export...")
        self.propensity_model.fit(self.X_train, self.T_train)
        self.g_x_test = np.clip(self.propensity_model.predict_proba(self.X_test)[:, 1], 0.01, 0.99)
        self.auc = roc_auc_score(self.T_test, self.g_x_test)
        print(f"   Propensity AUC (Test Set): {self.auc:.4f}")
        
        # Save feature importances report
        importance_series = pd.Series(self.propensity_model.feature_importances_, index=self.X_train.columns)
        importance_series = importance_series.sort_values(ascending=False)
        reports_dir = os.path.join(os.path.dirname(self.data_path), '..', 'reports')
        os.makedirs(reports_dir, exist_ok=True)
        report_path = os.path.join(reports_dir, f'propensity_features_{self.target_col}.json')
        with open(report_path, 'w') as f:
            json.dump(importance_series.to_dict(), f, indent=4)
        
        self.mu0_model.fit(self.X_train[self.T_train == 0], self.Y_train[self.T_train == 0])
        self.mu1_model.fit(self.X_train[self.T_train == 1], self.Y_train[self.T_train == 1])
        
        # Store clean Out-of-Fold imputed effects
        self.oof_D0 = oof_D0[self.T_train == 0]
        self.oof_D1 = oof_D1[self.T_train == 1]

    def stage_3_x_learning(self):
        print("Stage 3: Cross-Learning Imputed Treatment Effects using Out-of-Fold Counterfactuals...")
        X0 = self.X_train[self.T_train == 0]
        X1 = self.X_train[self.T_train == 1]
        
        self.tau0_model.fit(X0, self.oof_D0)
        self.tau1_model.fit(X1, self.oof_D1)

    def estimate_cate(self, X_eval):
        tau0_pred = self.tau0_model.predict(X_eval)
        tau1_pred = self.tau1_model.predict(X_eval)
        g_x_eval = np.clip(self.propensity_model.predict_proba(X_eval)[:, 1], 0.01, 0.99)
        
        cate = (1 - g_x_eval) * tau0_pred + g_x_eval * tau1_pred
        return cate

    def plot_uplift_validation(self, cate_test, output_dir):
        eval_df = pd.DataFrame({
            'cate_score': cate_test,
            'treatment': self.T_test.values,
            'outcome': self.Y_test.values
        })
        
        eval_df['cate_bucket'] = pd.qcut(eval_df['cate_score'], q=5, labels=['Lowest 20%', 'Low-Mid', 'Medium', 'Mid-High', 'Top 20%'])
        
        actual_rates = eval_df.groupby(['cate_bucket', 'treatment'], observed=True)['outcome'].mean().unstack()
        actual_rates.columns = ['No_TO', 'With_TO']
        
        actual_rates['Actual_Uplift'] = actual_rates['With_TO'] - actual_rates['No_TO']
        
        plt.figure(figsize=(10, 6))
        actual_rates['Actual_Uplift'].plot(kind='bar', color='mediumseagreen', edgecolor='black')
        plt.title(f'Actual Timeout Effectiveness by Model Prediction (Uplift)\n{self.target_col}')
        plt.ylabel('Actual Impact Improvement (With TO - No TO)')
        plt.xlabel('Model Prediction (CATE Score Buckets)')
        plt.axhline(0, color='red', linestyle='dashed', linewidth=1)
        plt.xticks(rotation=15)
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, f'uplift_validation_{self.target_col}.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"📈 Uplift Validation Graph saved to: {plot_path}")

    def generate_overlap_and_balance_diagnostics(self, output_dir):
        """
        Generates Academic Covariate Balance and Overlap Diagnostics (Dr. Shapira Note #3):
        1. Propensity Score Overlap Histogram (Treated vs Control).
        2. Love Plot showing Standardized Mean Differences (SMD) before & after IPW weighting.
        3. Balance Metrics JSON summary.
        All saved to: <output_dir>/paper_preparation/
        """
        paper_dir = os.path.join(output_dir, 'paper_preparation')
        os.makedirs(paper_dir, exist_ok=True)
        print(f"📊 Generating Academic Overlap & Covariate Balance diagnostics in: {paper_dir}")
        
        # 1. Overlap Distribution
        g_x = self.g_x_train
        T = self.T_train.values if isinstance(self.T_train, pd.Series) else self.T_train
        
        plt.figure(figsize=(10, 6))
        plt.hist(g_x[T == 0], bins=50, density=True, alpha=0.5, label='Control (No Timeout)', color='steelblue')
        plt.hist(g_x[T == 1], bins=50, density=True, alpha=0.5, label='Treated (Timeout Taken)', color='crimson')
        plt.axvline(0.01, color='gray', linestyle=':', label='Clip Boundary [0.01, 0.99]')
        plt.axvline(0.99, color='gray', linestyle=':')
        plt.title(f'Propensity Score Overlap Distribution (Common Support)\n{self.target_col}', fontsize=12)
        plt.xlabel('Estimated Propensity Score P(T=1 | X)')
        plt.ylabel('Density')
        plt.legend(loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        overlap_plot_path = os.path.join(paper_dir, f'overlap_propensity_{self.target_col}.png')
        plt.savefig(overlap_plot_path)
        plt.close()
        print(f"   ✅ Saved Overlap Histogram: {overlap_plot_path}")
        
        # 2. Standardized Mean Difference (SMD) and Love Plot
        X = self.X_train.copy()
        weights = np.where(T == 1, 1.0 / np.clip(g_x, 0.01, 0.99), 1.0 / np.clip(1.0 - g_x, 0.01, 0.99))
        
        smd_raw = {}
        smd_weighted = {}
        
        for col in X.columns:
            x_val = pd.to_numeric(X[col], errors='coerce').fillna(0).values
            x0, x1 = x_val[T == 0], x_val[T == 1]
            var0, var1 = np.var(x0), np.var(x1)
            pooled_std = np.sqrt((var0 + var1) / 2.0)
            if pooled_std < 1e-8:
                continue
                
            # Unweighted SMD
            raw_diff = np.abs(np.mean(x1) - np.mean(x0))
            smd_raw[col] = raw_diff / pooled_std
            
            # Weighted SMD (IPW)
            w0, w1 = weights[T == 0], weights[T == 1]
            mean0_w = np.sum(x0 * w0) / np.sum(w0)
            mean1_w = np.sum(x1 * w1) / np.sum(w1)
            weighted_diff = np.abs(mean1_w - mean0_w)
            smd_weighted[col] = weighted_diff / pooled_std

        # Top 25 features by unadjusted SMD to keep Love Plot readable
        sorted_features = sorted(smd_raw.keys(), key=lambda k: smd_raw[k], reverse=True)[:25]
        
        plt.figure(figsize=(10, max(8, len(sorted_features) * 0.35)))
        y_pos = np.arange(len(sorted_features))
        raw_vals = [smd_raw[f] for f in sorted_features]
        adj_vals = [smd_weighted[f] for f in sorted_features]
        
        plt.plot(raw_vals, y_pos, 'o', color='crimson', label='Unadjusted (Raw)', markersize=7)
        plt.plot(adj_vals, y_pos, 'o', color='seagreen', label='Adjusted (IPW Weighted)', markersize=7)
        
        for i, (r, a) in enumerate(zip(raw_vals, adj_vals)):
            plt.hlines(y_pos[i], min(r, a), max(r, a), color='gray', linestyle=':', alpha=0.6)
            
        plt.axvline(0.1, color='black', linestyle='--', linewidth=1.2, label='Balance Threshold (SMD = 0.1)')
        plt.yticks(y_pos, sorted_features)
        plt.gca().invert_yaxis()
        plt.xlabel('Absolute Standardized Mean Difference (SMD)')
        plt.title(f'Love Plot: Covariate Balance Before & After Weighting\n{self.target_col}', fontsize=12)
        plt.legend(loc='lower right')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        love_plot_path = os.path.join(paper_dir, f'love_plot_{self.target_col}.png')
        plt.savefig(love_plot_path)
        plt.close()
        print(f"   ✅ Saved Love Plot: {love_plot_path}")
        
        # Save balance metrics JSON
        balance_summary = {
            "target": self.target_col,
            "total_features_evaluated": len(smd_raw),
            "unadjusted_mean_smd": float(np.mean(list(smd_raw.values()))),
            "weighted_mean_smd": float(np.mean(list(smd_weighted.values()))),
            "features_balanced_under_0_1_pct": float(np.mean([v < 0.1 for v in smd_weighted.values()]) * 100),
            "top_features": {
                f: {"raw_smd": float(smd_raw[f]), "weighted_smd": float(smd_weighted[f])}
                for f in sorted_features
            }
        }
        json_path = os.path.join(paper_dir, f'covariate_balance_{self.target_col}.json')
        with open(json_path, 'w') as f:
            json.dump(balance_summary, f, indent=4)
        print(f"   ✅ Saved Balance Metrics JSON: {json_path}")

    def run_cluster_bootstrap(self, output_dir, n_bootstraps=100):
        """
        Cluster Bootstrap at the Game Level (Dr. Shapira Note #5):
        Resamples entire games (gameId) with replacement to account for intra-game correlation,
        evaluates CATE and ATE for each resample, and computes 95% Confidence Intervals.
        """
        paper_dir = os.path.join(output_dir, 'paper_preparation')
        os.makedirs(paper_dir, exist_ok=True)
        print(f"🎲 Running Game-Level Cluster Bootstrap (B={n_bootstraps}) for '{self.target_col}'...")
        
        # Pre-calculate point predictions on test set
        test_cate = self.estimate_cate(self.X_test)
        
        unique_games = np.unique(self.games_test)
        n_games = len(unique_games)
        
        # Map game to indices for fast cluster sampling
        game_to_indices = {}
        for idx, gid in enumerate(self.games_test):
            game_to_indices.setdefault(gid, []).append(idx)
        for gid in game_to_indices:
            game_to_indices[gid] = np.array(game_to_indices[gid])
            
        rng = np.random.RandomState(42)
        bootstrap_ates = []
        
        for b in range(n_bootstraps):
            sampled_games = rng.choice(unique_games, size=n_games, replace=True)
            sampled_indices = np.concatenate([game_to_indices[gid] for gid in sampled_games])
            b_ate = np.mean(test_cate[sampled_indices])
            bootstrap_ates.append(b_ate)
            
        bootstrap_ates = np.array(bootstrap_ates)
        ci_lower = float(np.percentile(bootstrap_ates, 2.5))
        ci_upper = float(np.percentile(bootstrap_ates, 97.5))
        se = float(np.std(bootstrap_ates))
        
        print(f"   Point ATE: {self.ate:+.4f}")
        print(f"   95% CI: [{ci_lower:+.4f}, {ci_upper:+.4f}] (SE: {se:.4f})")
        
        # Plot Bootstrap Distribution
        plt.figure(figsize=(9, 6))
        plt.hist(bootstrap_ates, bins=25, color='royalblue', edgecolor='black', alpha=0.7, density=True)
        plt.axvline(self.ate, color='crimson', linestyle='-', linewidth=2, label=f'Point ATE ({self.ate:+.2f})')
        plt.axvline(ci_lower, color='black', linestyle='--', linewidth=1.5, label=f'95% CI Lower ({ci_lower:+.2f})')
        plt.axvline(ci_upper, color='black', linestyle='--', linewidth=1.5, label=f'95% CI Upper ({ci_upper:+.2f})')
        plt.axvline(0.0, color='gray', linestyle=':', linewidth=1)
        
        plt.title(f'Cluster Bootstrap Distribution (B={n_bootstraps} Games)\nTarget: {self.target_col}', fontsize=12)
        plt.xlabel('Estimated Average Treatment Effect (ATE)')
        plt.ylabel('Bootstrap Density')
        plt.legend(loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plot_path = os.path.join(paper_dir, f'bootstrap_ci_{self.target_col}.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"   ✅ Saved Bootstrap Plot: {plot_path}")
        
        # Save JSON
        res_json = {
            "target": self.target_col,
            "n_bootstraps": n_bootstraps,
            "unique_games_evaluated": int(n_games),
            "point_ate": float(self.ate),
            "ci_95_lower": ci_lower,
            "ci_95_upper": ci_upper,
            "standard_error": se,
            "statistically_significant": bool((ci_lower > 0 and ci_upper > 0) or (ci_lower < 0 and ci_upper < 0))
        }
        json_path = os.path.join(paper_dir, f'bootstrap_ci_{self.target_col}.json')
        with open(json_path, 'w') as f:
            json.dump(res_json, f, indent=4)
        print(f"   ✅ Saved Bootstrap CI JSON: {json_path}")

    def run_placebo_test(self, output_dir, n_placebo=20):
        """
        Placebo Treatment Test & Sensitivity E-Value (Dr. Shapira Note #6):
        1. Generates synthetic placebo treatments by random permutation to verify the estimated effect vanishes (ATE ~ 0.00).
        2. Calculates VanderWeele & Ding (2017) E-Value for robustness against unmeasured confounding.
        """
        paper_dir = os.path.join(output_dir, 'paper_preparation')
        os.makedirs(paper_dir, exist_ok=True)
        print(f"🧪 Running Placebo Treatment Test (M={n_placebo} iterations) for '{self.target_col}'...")
        
        rng = np.random.RandomState(42)
        T_real = self.T_train.values if isinstance(self.T_train, pd.Series) else self.T_train
        X_train_val = self.X_train.copy()
        
        placebo_ates = []
        
        for m in range(n_placebo):
            # Shuffle treatment randomly to break any true causal relationship
            T_placebo = rng.permutation(T_real)
            
            ctrl_mask = (T_placebo == 0)
            treat_mask = (T_placebo == 1)
            
            p_mu0 = xgb.XGBRegressor(n_estimators=30, max_depth=3, random_state=m)
            p_mu1 = xgb.XGBRegressor(n_estimators=30, max_depth=3, random_state=m)
            
            p_mu0.fit(X_train_val[ctrl_mask], self.Y_train[ctrl_mask])
            p_mu1.fit(X_train_val[treat_mask], self.Y_train[treat_mask])
            
            pred_mu0 = p_mu0.predict(self.X_test)
            pred_mu1 = p_mu1.predict(self.X_test)
            
            placebo_ate = np.mean(pred_mu1 - pred_mu0)
            placebo_ates.append(placebo_ate)
            
        placebo_ates = np.array(placebo_ates)
        mean_placebo_ate = float(np.mean(placebo_ates))
        std_placebo_ate = float(np.std(placebo_ates))
        
        # Calculate E-Value for Continuous Outcomes (VanderWeele & Ding 2017)
        y_std = float(np.std(self.Y_test))
        cohens_d = abs(self.ate) / (y_std if y_std > 1e-6 else 1.0)
        approx_rr = float(np.exp(0.91 * cohens_d))
        if approx_rr > 1.0:
            e_value = float(approx_rr + np.sqrt(approx_rr * (approx_rr - 1.0)))
        else:
            e_value = 1.0
            
        print(f"   Actual ATE: {self.ate:+.4f}")
        print(f"   Placebo Mean ATE: {mean_placebo_ate:+.4f} (Expected ~ 0.00)")
        print(f"   E-Value: {e_value:.2f}")
        
        # Plot Placebo vs Actual ATE
        plt.figure(figsize=(9, 6))
        plt.hist(placebo_ates, bins=15, color='darkgray', edgecolor='black', alpha=0.7, density=True, label='Placebo Treatment Distribution')
        plt.axvline(0.0, color='gray', linestyle=':', linewidth=1.5, label='Null Effect (0.00)')
        plt.axvline(mean_placebo_ate, color='blue', linestyle='--', linewidth=1.5, label=f'Mean Placebo ({mean_placebo_ate:+.2f})')
        plt.axvline(self.ate, color='crimson', linestyle='-', linewidth=2.5, label=f'Actual Timeout Effect ({self.ate:+.2f})')
        
        plt.title(f'Placebo Test: Synthetic Treatments vs Actual Timeout Effect\nTarget: {self.target_col}', fontsize=12)
        plt.xlabel('Estimated Average Treatment Effect (ATE)')
        plt.ylabel('Density')
        plt.legend(loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plot_path = os.path.join(paper_dir, f'placebo_test_{self.target_col}.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"   ✅ Saved Placebo Plot: {plot_path}")
        
        # Save JSON
        res_json = {
            "target": self.target_col,
            "n_placebo_iterations": n_placebo,
            "actual_ate": float(self.ate),
            "placebo_mean_ate": float(mean_placebo_ate),
            "placebo_std_ate": float(std_placebo_ate),
            "placebo_centered_near_zero": bool(abs(mean_placebo_ate) < 0.1),
            "cohens_d": float(cohens_d),
            "approx_relative_risk": float(approx_rr),
            "e_value_unmeasured_confounding": float(e_value)
        }
        json_path = os.path.join(paper_dir, f'placebo_and_evalue_{self.target_col}.json')
        with open(json_path, 'w') as f:
            json.dump(res_json, f, indent=4)
        print(f"   ✅ Saved Placebo & E-Value JSON: {json_path}")

    def run_pipeline(self):
        self.load_and_prepare_data()
        self.stage_1_and_2_cross_fitting()
        self.stage_3_x_learning()
        
        print("\nPipeline Complete. Evaluating CATE on Test Set...")
        cate_test = self.estimate_cate(self.X_test)
        
        avg_treatment_effect = np.mean(cate_test)
        self.ate = avg_treatment_effect
        print(f"Average Treatment Effect (ATE) for '{self.target_col}': {avg_treatment_effect:+.2f} points/impact")
        
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        reports_dir = os.path.join(base_dir, 'reports')
        self.plot_uplift_validation(cate_test, reports_dir)
        
        # Academic Validation Suite (Dr. Shapira Notes #3, #5, #6)
        self.generate_overlap_and_balance_diagnostics(reports_dir)
        self.run_cluster_bootstrap(reports_dir, n_bootstraps=100)
        self.run_placebo_test(reports_dir, n_placebo=20)
        
        # Persist trained model estimators to disk
        self.save_models()
        
        return cate_test

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    DATA_PATH = os.path.join(base_dir, 'data', 'processed', 'train.parquet')
    REPORTS_DIR = os.path.join(base_dir, 'reports')
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    print(f"Working with absolute path: {DATA_PATH}")
    
    targets = [
        'target_stop_run_90s', 
        'target_reverse_trend_180s', 
        'target_improve_margin_90s', 
        'target_improve_margin_180s'
    ]
    
    summary_results = {}

    for target in targets:
        causal_learner = NBACausalLearner(
            data_path=DATA_PATH,
            target_col=target,
            treatment_col='is_timeout'
        )
        cate_results = causal_learner.run_pipeline()
        
        summary_results[target] = {
            "ate": float(causal_learner.ate),
            "auc": float(causal_learner.auc)
        }

    print("\n" + "="*55)
    print(f"{'Target':<30} | {'ATE (Impact)':<12} | {'Propensity AUC':<10}")
    print("-" * 55)
    for t, res in summary_results.items():
        print(f"{t:<30} | {res['ate']:>+12.4f} | {res['auc']:>8.4f}")
    print("="*55)

    summary_path = os.path.join(REPORTS_DIR, 'causal_multi_target_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary_results, f, indent=4)

    names = [t.replace('target_', '') for t in targets]
    ates = [res['ate'] for res in summary_results.values()]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(names, ates, color='coral')
    plt.axhline(0, color='black', linewidth=0.8)
    plt.ylabel('Average Treatment Effect (ATE Impact)')
    plt.title('Timeout Impact Across Different Strategic Goals (Continuous)')
    plt.xticks(rotation=15)
    
    for bar in bars:
        yval = bar.get_height()
        offset = 0.05 if yval >= 0 else -0.1
        plt.text(bar.get_x() + bar.get_width()/2, yval + offset, f'{yval:+.2f}', ha='center', va='bottom')

    plt.tight_layout()
    plot_path = os.path.join(REPORTS_DIR, 'ate_comparison_profile.png')
    plt.savefig(plot_path)
    plt.close()
    
    print(f"\n✅ Summary saved to JSON: {summary_path}")
    print(f"✅ Plot generated: {plot_path}")
