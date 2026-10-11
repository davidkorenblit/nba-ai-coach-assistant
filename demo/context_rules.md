# NBA SimCast Arena — Master Context & Testing Suite
## Objective: Validate & Align Generated Game Logs with Scripted Telemetry and Basketball Realism

This document serves as the absolute Source of Truth and Context Guide for the Local AI Agent to develop automated validation tests and diagnostic scripts for `demo_data.json` / `simulation_log.txt`.

---

## 1. Absolute Narrative Blueprint (The Golden Script)

The generator pipeline must produce two distinct game histories matching these exact chronological criteria. Any deviation constitutes a test failure.

### 🏀 Game 1: The Stubborn Coach (The Problem Scenario)
* **Quarter 1 (Start):** Score starts strictly at `0:0`.
* **Quarter 1 (~04:30 remaining / 60% into Q1 index):** `propensity_score` spikes to `>= 0.85` (**Green Light** turns ON). `play_description` contains the phrase: `"Propensity Alarm"`.
* **Quarter 1 (Exactly 2 possessions later):** Coach reacts late. `timeout_team` becomes `"INDIANA"`, `timeout_strategic_weight = 1`, and `target_stop_run_90s = 1`.
* **Quarter 1 (Next 3 possessions):** Game margin stabilizes temporarily (normal basketball scoring text, no massive delta drift).
* **Quarter 1 (~01:00 remaining / 85% into Q1 index):** `timeout_team` becomes `"INDIANA"`. `play_description` indicates a `"Tactical Timeout"` for end-of-quarter spacing. No system alerts/LED lights allowed here.
* **Quarter 2 (~Mid-quarter):** A False Positive occurs. `propensity_score` spikes to `>= 0.75` (**Green Light** pulses), but `timeout_team` remains `"NONE"`. The score and game flow must recover naturally without any score injection.
* **Quarter 3 (Last 2 minutes / 30% remaining in Q3):** The Crisis Zone. `propensity_score` remains persistently high (`>= 0.82`) across a 12-possession window, but the coach freezes (`timeout_team = "NONE"`).
* **Quarter 3 (Final 6 possessions):** The Catastrophe. `cate_score` spikes to `>= 0.95` (**Red Light** flashes violently). SHAP values (`shap_stale_lineup`, `shap_defensive_collapse`, etc.) must all cross `>= 0.80`. The opponent triggers a 15-0 scoring run. The final row of Q3 must assert a `score_margin` of exactly `-25`.
* **Quarter 4 (Entire Quarter):** The locked deficit corridor. The `-25` deficit must cascade and be maintained until the final buzzer. Final game output must pass: `assert score_margin == -25`.

### 🏀 Game 2: The Strategic Coach (The ROI Scenario)
* **Quarter 1:** Features a successful early defensive timeout based on standard propensity indicators.
* **Quarter 2 (~Mid-quarter):** The opponent initiates a run. A proactive defensive timeout is triggered (`timeout_team = "INDIANA"`). The telemetry must prove **Mean Reversion**—the game margin stabilizes and returns to the historical mean within 180 seconds (~12 possessions).
* **Quarter 3:** Smooth, highly competitive basketball text. No critical alerts or forced delta animations.
* **Quarter 4 (~05:00 remaining):** The Climax. The opponent creates a dangerous hidden run, opening a 7-point lead (`score_margin == -7`). System triggers a critical Red Alert (`cate_score >= 0.92`, `target_stop_run_90s = 1`).
* **Quarter 4 (Exactly 1 possession later):** The coach reacts immediately! `timeout_team = "INDIANA"`, `timeout_strategic_weight = 1`.
* **Quarter 4 (Post-Timeout Comeback):** Momentum shift. Injections add scoring deltas *exclusively* to Indiana (Home team). Indiana erases the 7-point deficit and takes the lead.
* **Quarter 4 (Final Minute):** Two tactical late-game timeouts appear (`"INDIANA"` and `"TORONTO"` to stop the clock).
* **Final Buzzer:** Indiana wins a nail-biter by exactly 2 points. Final row must pass: `assert score_margin == 2`.

---

## 2. Automated Test Suite Suite (The Rules for the Agent)

The local Agent must generate test scripts (or refactor the backend patcher) to enforce the following programmatic checks on the log streams:

### Rule 2.1: Mathematical Margin vs. System Alarm Consistency
* **The Logic Error:** System alarms cannot fire during positive scoring phases for the home team.
* **The Test:** Assert that whenever `target_stop_run_90s == 1` or `cate_score > 0.70` or `propensity_score > 0.75`, the immediate derivative of the score margin over the trailing 5 possessions must be negative (`Δmargin / Δpossessions < 0`). Alarms are mathematically restricted to opponent runs only.

### Rule 2.2: Possession Scoring Realism (Anti-Inflation Gate)
* **The Logic Error:** Continuous score increases on every single row create unrealistic final game outputs (e.g., scores exceeding 140–150 points).
* **The Test:** Assert that the maximum consecutive number of rows where a score increases cannot exceed 4. The pipeline must insert "empty possessions" (missed shots, defensive stops, turnovers) between points. Points injected to achieve a target margin must be distributed across wider windows, leaving empty rows where scores remain unchanged.

### Rule 2.3: The Quarter 4 Deficit Lock Formula (Game 1 Optimization)
* **The Logic Error:** In Game 1, forcing a `-25` margin across all of Q4 on top of highly volatile real data pushes the total points to absurd numbers because the script continuously adds points to match the fluctuating real template.
* **The Test/Solution:** For Game 1 Quarter 4, the original parquet scoring pace must be dampened. Instead of calculating offsets based on a volatile baseline, the pipeline should implement **Base Score Flattening** in Q4: lock the scores at the end of Q3 (`IND: X`, `TOR: X + 25`), and allow scores to increase only when a realistic scoring play happens in the template, applying identical increments to both teams to freeze the delta corridor at exactly `-25` without inflating total scores beyond 110 points.

---

## 3. High-Priority Diagnostic Directives for the Agent

Your immediate engineering tasks based on this testing suite are:

1. **Write `validate_logs.py`:** Create an automated python script that reads `data/demo/demo_data.json` and checks every chronological requirement listed in the Scripted Blueprints above, printing detailed `PASSED` or `FAILED` logs for each quarter frame.
2. **Refactor `_distribute_points` and `_apply_positive_margin_offset`:** Rewrite the point distribution algorithm to guarantee that point increments never exceed 2 or 3 per possession, enforce mandatory empty possessions between scores, and implement the Q4 Flattening technique to pull final score totals down below 115 points while passing the `-25` and `+2` final margin assertionsor `_distribute_points` and `_apply_positive_margin_offset`:** Rewrite the point distribution algorithm to guarantee that point increments never exceed 2 or 3 per possession, enforce mandatory empty possessions between scores, and pull final score totals down below 115 points while passing the `-25` and `+2` final margin assertions (Note: The Q4 Flattening technique is disabled for now)..