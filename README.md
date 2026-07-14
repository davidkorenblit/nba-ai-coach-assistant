# NBA-AI-Coach: Real-Time Tactical Timeout Optimization

> 🚀 **Live Demo Available:** Play with the interactive web simulator instantly here: [SimCast Arena Dashboard](https://davidkorenblit.github.io/nba-ai-coach-assistant/)

## Project Executive Summary
A Machine Learning-driven Decision Support System (DSS) designed to assist NBA coaching staffs in making data-validated timeout calls. By transforming raw event-stream data into a multi-layered analytical framework, the system quantifies game momentum and player fatigue, utilizing predictive modeling and **Causal Inference** to isolate the true treatment effect of a timeout and predict the optimal moment to intervene.

---

## 🚀 Interactive Live Demos

To showcase the decision-support engine in action, the project includes two interactive demo environments that simulate game scenarios and display real-time tactical recommendations.

> [!NOTE]
> **Demo Data Note:** The demos use curated, simulated game scenarios designed to highlight specific tactical moments (e.g., clutch collapses, momentum stops, and fatigue threshold breaches) while complying with proprietary NBA data license restrictions.

### 1. SimCast Arena Dashboard (`index.html`)
A premium, web-based visual dashboard mimicking a live game broadcast feed.

*   **🌐 Option A: Live Hosted Demo (No Installation)**
    *   Simply visit the hosted version: [SimCast Arena Dashboard](https://davidkorenblit.github.io/nba-ai-coach-assistant/)
*   **📂 Option B: Local File View (Zero-Setup)**
    *   Open [index.html](file:///c:/Users/david/finalPro/index.html) directly in any web browser by double-clicking it in your file explorer.
    *   *Note: Because the demo data is preloaded as a static module in `data/demo/demo_data.js`, this works instantly offline and is completely immune to browser CORS restriction issues.*
*   **🖥️ Option C: Local HTTP Server**
    *   Run a local server in the project directory:
        ```bash
        python -m http.server 8000
        ```
    *   Then visit: `http://localhost:8000`

### 2. Streamlit Live Simulation (`app.py`)
An interactive Python-based dashboard showcasing the data science, feature importance, and causal inference outputs.
*   **Features:** Dynamic playback control (Play/Pause/Reset), period-by-period breakdown, and real-time visualization of CATE (Conditional Average Treatment Effect) scores and propensity weights.
*   **How to Run:**
    1.  Install the required dependencies:
        ```bash
        pip install streamlit pandas numpy matplotlib pyarrow
        ```
    2.  Run the application:
        ```bash
        streamlit run app.py
        ```
    3.  Open `http://localhost:8501` in your browser.

---

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

### Layer 4: Causal Inference (Prescriptive Phase)
Transitioning from predictive to prescriptive analytics to answer: *"What happens if the coach intervenes?"*
* **Treatment Definition:** Isolating timeout events within possessions.
* **Propensity Scoring:** Modeling the probability of a coach calling a timeout in any given game state.
* **X-Learner Implementation:** Calculating the Conditional Average Treatment Effect (CATE) to provide actionable, situation-specific timeout recommendations.

---

## Tech Stack & Standards
* **Core:** Python (Pandas, NumPy, Scikit-Learn, XGBoost, Streamlit)
* **Frontend:** HTML5, TailwindCSS, JavaScript (Chart.js)
* **Design:** Object-Oriented Programming (OOP) with modular logic handlers.
* **Validation:** Integrated QA Validators (`Level3QAValidator`, `SplitValidator`) enforcing strict diagnostic checks prior to model training.
* **Efficiency:** 100% Vectorized data processing (no Python loops) for low-latency operations.