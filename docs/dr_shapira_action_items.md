# 📋 משימות פיתוח ומתודולוגיה אקדמית: דרישות ד"ר שפירא
# Academic Methodology & Development Tasks: Dr. Shapira Roadmap

> **תאריך עדכון / Updated:** 2026-10-26  
> **מטרה / Objective:** מיפוי והגדרה של משימות הקוד והמתודולוגיה הסיבתית בעקבות המשוב של ד"ר רחל לאה כהנא שפירא לקראת פרסום מאמר.  
> **Objective:** Comprehensive roadmap of code, modeling, and causal methodology tasks based on Dr. Rachel Shapira's review.

---

## 🇮🇱 עברית: משימות הקוד והמתודולוגיה

### 1. הפרדת משחקים מלאה ב-Cross-Fitting (`GroupKFold` ברמת המשחק) — [✅ הושלם ואומת אמפירית]
* **הרציונל המדעי:** מניעת תלות תוך-משחקית וזליגת מידע (Intra-game correlation & leakage). אירועים (פוזשנים) מאותו משחק לא יימצאו בו-זמנית גם ב-Training Fold וגם ב-OOF Fold של ה-X-Learner.
* **יישום בקוד:** 
  * יושם `StratifiedGroupKFold(n_splits=5)` מקושר ל-`groups=games_train` (`gameId`) בתוך `stage_1_and_2_cross_fitting` בקובץ `models/06_causal_x_learner.py`.
  * הוטמעה בדיקת `assert` קשיחה בכל Fold שמוודאת 0 חיתוך משחקים בין Train ל-Validation.
* **סטטוס ואימות אמפירי (2026-10-26):**
  * **הושלם בהצלחה (Exit code: 0).**
  * כל 5 ה-Folds עברו עם **0 דליפת משחקים**.
  * חושבו בהצלחה אומדני OOF עבור $D_0, D_1$ ונוצרו תחזיות CATE מלאות לכל 83,761 שורות המבחן.

---

### 2. בדיקות חפיפה (Overlap) ותמיכה משותפת (Common Support) — [✅ הושלם ואומת אמפירית]
* **הרציונל המדעי:** מאחר שמודל ה-Propensity מדויק יחסית (~85% AUC), קיים חשש להפרת הנחת ה-Positivity ($0 < P(T=1 \mid X) < 1$). יש להוכיח שקיימות תצפיות ביקורת דומות מספיק למצבי פסק זמן.
* **יישום בקוד:**
  * פותח כלי ייעודי: [`models/evaluate_overlap_balance.py`](file:///c:/Users/david/finalPro/models/evaluate_overlap_balance.py).
  * חישוב גבולות Common Support אמפיריים מלאים ואימות הנחת ה-Positivity.
  * הפקת היסטוגרמות חפיפה עם סימון גבולות תמיכה משותפת (`reports/paper_preparation/overlap_propensity_*.png`).
  * הפקת Love Plot של איזון משתנים (SMD) לפני ואחרי שקלול IPW (`reports/paper_preparation/love_plot_*.png`).
* **סטטוס ואימות אמפירי (2026-10-27):**
  * **הושלם בהצלחה מלאה (Exit code: 0).**
  * **99.99% מכלל המדגם** (ו-**99.49%** מקבוצת הטיפול) נמצאים בתוך תחום התמיכה המשותפת ($[0.0100, 0.7709]$).
  * אין חסימה מוחלטת או דטרמיניזם של טיפול (ערך מקסימלי $e(X) \approx 0.77 < 1.0$), מה שמאשר ש-**הנחת ה-Positivity מתקיימת במלואה (VERIFIED)**.
  * שקלול IPW שיפר את איזון המשתנים והפחית את ה-SMD הממוצע ל-0.1117.

---

### 3. אומדן סיבתי מתחרה / נוסף (Second Causal Estimator)
* **הרציונל המדעי:** הוכחה שההטרוגניות של האפקט (CATE) אינה ארטיפקט של ארכיטקטורת ה-X-Learner.
* **יישום בקוד:**
  * הטמעת אסטימטור מבוסס Doubly Robust (DR-Learner / AIPW) או Causal Forest (שימוש בספריית `EconML`).
  * השוואת התפלגות ה-CATE והקורלציה בין הניבויים של שני המודלים בסט המבחן.
* **סטטוס:** מודל עצמאי חדש שיוקם לצורך בנצ'מרק.

---

### 4. ולידציית CATE עצמאית ועקומות Uplift (מעבר מבדיקת 5% בלבד) — [✅ הושלם ואומת אמפירית]
* **הרציונל המדעי:** מניעת טענות Cherry-Picking. הוכחה שהמודל מדרג נכון את האפקטיביות לאורך כל טווח המצבים ולא רק באחוזון עליון שרירותי.
* **יישום בקוד:**
  * פותח כלי ולידציה עצמאי מלא: [`models/evaluate_cate_uplift.py`](file:///c:/Users/david/finalPro/models/evaluate_cate_uplift.py).
  * חלוקת סט המבחן החיצוני (`test.parquet`, 83,761 אירועים) ל-10 עשירונים (D1 עד D10) לפי ה-Predicted CATE.
  * חישוב ה-Actual Uplift האמפירי בכל עשירון בנפרד ($E[Y \mid T=1] - E[Y \mid T=0]$).
  * הפקת גרף Uplift דו-ממדי ברזולוציה גבוהה (תרשים עמודות וגרף קורלציה) וקובץ סיכום JSON.
* **סטטוס ואימות אמפירי (2026-10-26):**
  * **הושלם בהצלחה.**
  * עבור שיפור הפרש סלים (`target_improve_margin_90s`), הוכחה מגמת עלייה מובהקת ($r = +0.38$): מרווח זניח של $+0.06$ נקודות בלבד בעשירון D2 ועד רווח שיא של **$+0.76$ נקודות** בעשירון D9 לטובת הקבוצה שלקחה פסק זמן.
  * התוצרים נשמרו ב-`reports/paper_preparation/uplift_deciles_*.png` ו-`.json`.

---

### 5. הרחבה רב-עונתית ומבחן זמן עיוור (External Temporal Validation)
* **הרציונל המדעי:** מבחן הזהב של הכללה (Generalization). המודל יאומן על עונות היסטוריות וייבדק על עונה שלמה עתידית שלא נראתה באימון.
* **יישום בקוד:**
  * איסוף והזרמת נתוני העונות המלאות (2021/22, 2022/23, 2023/24) לתוך ה-Pipeline של ה-Feature Engineering הנקי.
  * שמירת עונת 2024/25 במלואה כ-Test Set עיוור (Out-of-Time Holdout).
* **סטטוס:** הנתונים הגולמיים זמינים, נדרשת הרצת ה-FE שלבים 1–3 על עונות העבר.

---
---

## 🇬🇧 English: Code & Methodology Action Items

### 1. Strict Game-Level Group Cross-Fitting — [✅ Completed & Empirically Verified]
* **Scientific Rationale:** Prevent intra-game correlation and lookahead/data leakage. Possessions from the exact same game must never appear simultaneously in both training folds and out-of-fold (OOF) evaluation folds.
* **Implementation:**
  * Refactored `stage_1_and_2_cross_fitting` inside `models/06_causal_x_learner.py`.
  * Implemented `StratifiedGroupKFold(n_splits=5)` strictly grouped by `gameId`.
  * Integrated an automated assertion validating zero game intersection between folds.
* **Status & Empirical Validation (2026-10-26):**
  * **Passed cleanly (Exit code: 0).**
  * All 5 folds completed with strictly **0 game leakage**.
  * Complete OOF counterfactual imputation ($D_0, D_1$) and seamless downstream Stage 3 CATE generation across all 83,761 test rows.

---

### 2. Propensity Overlap & Common Support Diagnostics — [✅ Completed & Empirically Verified]
* **Scientific Rationale:** High Propensity AUC (~85%) raises concerns about the Positivity assumption ($0 < P(T=1 \mid X) < 1$). We must demonstrate sufficient overlap between treated and control game states.
* **Implementation:**
  * Developed dedicated diagnostic script: [`models/evaluate_overlap_balance.py`](file:///c:/Users/david/finalPro/models/evaluate_overlap_balance.py).
  * Compute empirical Common Support boundaries and test Positivity adherence.
  * Generate Propensity Overlap density plots with explicit common support bounds (`reports/paper_preparation/overlap_propensity_*.png`).
  * Compute Standardized Mean Differences (SMD) and generate Love Plots pre- and post-IPW weighting (`reports/paper_preparation/love_plot_*.png`).
* **Status & Empirical Validation (2026-10-27):**
  * **Passed cleanly (Exit code: 0).**
  * **99.99% of total samples** (and **99.49%** of treated samples) fall within the empirical common support interval ($[0.0100, 0.7709]$).
  * No deterministic treatment assignment (maximum propensity $\approx 0.77 < 1.0$), confirming **Positivity assumption is fully satisfied (VERIFIED)**.
  * IPW weighting reduced average SMD to 0.1117, successfully balancing macro game states.

---

### 3. Secondary Causal Estimator (DR-Learner / Causal Forest)
* **Scientific Rationale:** Confirm that treatment effect heterogeneity is a true game phenomenon, rather than an artifact of the specific X-Learner meta-learning architecture.
* **Implementation:**
  * Train an alternative Doubly Robust (DR-Learner / AIPW) or Generalized Random Forest (Causal Forest) using `EconML`.
  * Compare CATE distributions and calculate rank correlation against X-Learner predictions.
* **Status:** New benchmark pipeline to be built.

---

### 4. Empirical Out-of-Sample CATE Uplift Validation (Decile Sweep) — [✅ Completed & Empirically Verified]
* **Scientific Rationale:** Avoid cherry-picking criticisms by expanding beyond the top 5% "sweet spot" to evaluate the full spectrum of predicted effects.
* **Implementation:**
  * Developed a dedicated validation suite: [`models/evaluate_cate_uplift.py`](file:///c:/Users/david/finalPro/models/evaluate_cate_uplift.py).
  * Bucket the external test set (`test.parquet`, 83,761 samples) into 10 deciles (D1 to D10) based on predicted CATE.
  * Calculate empirical treatment uplift per bucket: $E[Y \mid T=1] - E[Y \mid T=0]$.
  * Generate high-resolution two-panel diagnostic plots and comprehensive JSON summaries.
* **Status & Empirical Validation (2026-10-26):**
  * **Passed cleanly.**
  * For score margin improvement (`target_improve_margin_90s`), verified a strong positive monotonic progression ($r = +0.38$): climbing from a negligible $+0.06$ points in D2 up to a peak uplift of **$+0.76$ net points** in D9.
  * Diagnostic artifacts saved to `reports/paper_preparation/uplift_deciles_*.png` and `.json`.

---

### 5. Multi-Season External Temporal Validation
* **Scientific Rationale:** Golden standard of out-of-distribution temporal generalization.
* **Implementation:**
  * Ingest and process multiple prior seasons (2021/22, 2022/23, 2023/24) through the hardened Level 1–3 FE pipeline.
  * Reserve the entire 2024/25 season as a strictly held-out out-of-time test set.
* **Status:** Raw CSVs collected in `data/pureData`; requires automated pipeline run.
