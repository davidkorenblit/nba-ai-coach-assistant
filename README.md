# NBA-AI-Coach: Real-Time Tactical Timeout Optimization

## Project Executive Summary
A Machine Learning-driven Decision Support System (DSS) designed to assist NBA coaching staffs in making data-validated timeout calls. By transforming raw event-stream data into a multi-layered analytical framework, the system quantifies game momentum and player fatigue, utilizing predictive modeling and **Causal Inference** to isolate the true treatment effect of a timeout and predict the optimal moment to intervene.

## System Architecture: The Pipeline

### Layer 1: Base State Engine (ETL & Infrastructure)
* **Data Ingestion:** Automated processing of NBA Play-by-Play (PBP) JSON/CSV data.
* **Lineup Inference:** Hybrid heuristic engine (61% API / 39% Logic) for real-time 5-man unit tracking.
* **Temporal Normalization:** Mapping disparate period clocks to a unified, second-based game timeline.

### Layer 2: Feature Engineering (The "Brain")
Focusing on non-linear game dynamics to extract true basketball context:
* **Smart Momentum:** A weighted composite score of scoring efficiency, defensive stops, and high-value plays.
* **Explosiveness Index:** Calculated slope of score-margin variance to identify rapid momentum shifts.
* **Shared Fatigue & Gravity:** Vectorized tracking of on-court duration to identify "usage-gravity" drop-offs.
* **Style Shift Detection:** Identifying deviations in average shot-clock usage and pace.

### Layer 3: MLOps, Labeling & Baseline Prediction
* **Dynamic Target Labeling:** Object-Oriented labeling engine (`Level3Labeler`) resolving the "Margin Bug" (contextualizing score relativity) to define success across multiple temporal windows (e.g., 90s, 180s momentum stops).
* **Leakage-Proof Data Splitting:** Strict Game-Level chronological splitting (Train/Val/Test) combined with Parquet serialization to guarantee zero future-to-past data leakage.
* **Hardened Baseline Model:** An XGBoost Classifier heavily regularized (L1/L2 penalties, depth limits, 80% subsampling) and stripped of temporal "cheat" variables (e.g., `seconds_remaining`). 
* **Current Benchmark:** Established a robust Performance Floor of **0.86 ROC-AUC**, successfully validating the predictive power of Level 2 engineered features (Explosiveness, Fatigue).

### Layer 4: Causal Inference (Current Phase)
Transitioning from predictive to prescriptive analytics to answer: *"What happens if the coach intervenes?"*
* **Treatment Definition:** Isolating timeout events within possessions.
* **Propensity Scoring:** Modeling the probability of a coach calling a timeout in any given game state.
* **X-Learner Implementation:** Calculating the Conditional Average Treatment Effect (CATE) to provide actionable, situation-specific timeout recommendations.

## Tech Stack & Standards
* **Core:** Python (Pandas, NumPy, Scikit-Learn, XGBoost)
* **Design:** Object-Oriented Programming (OOP) with modular logic handlers.
* **Validation:** Integrated QA Validators (`Level3QAValidator`, `SplitValidator`) enforcing strict diagnostic checks prior to model training.
* **Efficiency:** 100% Vectorized data processing (no Python loops) for low-latency operations.