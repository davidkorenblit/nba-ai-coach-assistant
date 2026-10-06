# 🔍 NBA AI Coach – דו"ח ביקורת קוד מלא + הצלבה עם הערות ד"ר שפירא

> **מטרה:** מיפוי מצב הקוד הקיים + הצלבה עם ההערות האקדמיות של ד"ר שפירא לקראת כתיבת מאמר אקדמי.  
> **סטטוס עדכני:** תוקנו כל באגי התשתית וה-Feature Engineering (שלב א' הושלם). המעבר לשכבת המודלים (שלב ב'-ג') מוכן.  
> **ענף פעיל:** `paper-preparation`

---

## TL;DR – תמונת מצב עדכנית (אוקטובר 2026)

1. **קוד הדמו (app.py, prepare_demo_data, context_rules):** בודד לחלוטין מהמחקר.
2. **ה-Pipeline של הנתונים (Feature Engineering):** **כל הבאגים הקריטיים והמתודולוגיים טופלו ואומתו אמפירית (Level 1, 2, 3 עברו QA מלא).**
3. **הבאגים שנוספו ונפתרו מעבר לרשימה המקורית:** תוקן מדד הכוכבים (`star_advantage`), תוקנה שחיתות הרוטציות מ-2024-25, הופרד טיימר החילופים (`time_since_last_sub_home/away`), ונבנה מערך ולידציה ל-Right-Censoring.
4. **השלב הבא (שכבת המודלים):** K-Fold Cross-Fitting ב-X-Learner, בדיקות Overlap, ו-Off-Policy Evaluation (OPE).

---

## 📋 סטטוס מרוכז של כלל הבאגים

| מס' | נושא הבאג | קובץ | סטטוס | פירוט הטיפול |
|:---:|:---|:---|:---:|:---|
| **#1** | מומנטום ללא כיוון (Directionless) | `02_build_level2_momentum.py` | ✅ **טופל** | הומר ל-Tug-of-War רציף: פעולת יריבה מכרסמת במומנטום; פוצל ל-`home_momentum_streak`, `away_momentum_streak` ו-`momentum_delta`. |
| **#2** | אפסים מזויפים בסופי רבעים (Boundary Truncation) | `03_build_level3_labels.py` | ✅ **טופל** | בוטל `fillna(0)`. הוטמע Right-Censoring (ערכי NaN מבוקרים). אומת אמפירית: 100% מה-NaNs בסופי רבעים בלבד (0 חריגות). |
| **#3** | היפוך קוטביות ב-Target (`interest_sign`) | `03_build_level3_labels.py` | ✅ **טופל** | סימן השינוי נקבע לפי הקבוצה הפועלת/הקבוצה שקראה לפסק זמן (+1 לבית, -1 לחוץ), ללא תלות במי מוביל בלוח. |
| **#4** | משתנים סיבתיים חסרים (Omitted Confounders) | `models/pipeline_constants.py` | ✅ **טופל** | שוחזר `seconds_remaining` והוגדרו קטגוריות מודולריות (`v1_standard_causal`), תוך שימור שעון משחק ו-Clutch. |
| **#5a** | זליגת שעון חילופים (Sub Timer Team Leakage) | `01_build_level1_base.py`, `02_build_level2_momentum.py` | ✅ **טופל** | הופרד לשני שעונים עצמאיים: `time_since_last_sub_home` ו-`away`. בלבל 2 נוספו `is_high_fatigue_home/away` ו-`stint_fatigue_diff`. |
| **#5b** | אין Cross-Fitting ב-X-Learner (In-Sample Bias) | `models/06_causal_x_learner.py` | ⏳ **ממתין לשלב המודלים** | דורש K-Fold Cross-Fitting לחישוב Counterfactuals ללא הטיה. |
| **#6** | Clutch Time מופעל בכל הרבעים | `02_build_level2_momentum.py` | ✅ **טופל** | נוסף תנאי קשיח `period >= 4`. |
| **#7** | Shot Clock לא מתאפס בריבאונד התקפי | `01_build_level1_base.py` | ⚪ **עקיף/זניח** | נבדק ונשמר במסגרת בדיקות חוק ה-14 שניות. |
| **#8** | ספירת מלאי פסקי זמן לא מדויקת | `01_build_level1_base.py` | ⚪ **עקיף** | מאומת ב-Level 1 QA inventory check. |
| **#9** | קבצי Parquet מכילים Features דולפים | `models/prepare_ml_splits.py` | ⏳ **ממתין** | יסונכרן יחד עם אימון המודלים החדש. |
| **#10** | שחיתות נתונים ב-Rotations CSV | `data/pureData/rotations_2024_25.csv` | ✅ **טופל** | תוקנו 31,232 שורות מוזזות, נוקתה עמודת `team_side` (100% תקין), נטרול `s1.py` שגרם לשיבוש. |
| **#11** | Recommendation Engine – השוואה אובסרבציונלית | `models/recommendation_engine.py` | ⏳ **ממתין לשלב ג'** | יימחק וייבנה מחדש באמצעות Off-Policy Evaluation (OPE). |
| **#12** | Hit Rate Sweep – ולידציה מעגלית | `models/hit_rate_sweep.py` | ⏳ **ממתין לשלב ג'** | יוחלף בעקומות Uplift / AUUC. |
| **+** | **באג א-סימטריה בכוכבים (`is_star_resting`)** *(נוסף מעבר לדו"ח)* | `02_build_level2_momentum.py` | ✅ **טופל** | הוחלף ב-`star_advantage` (+1 בית, 0 שוויון, -1 חוץ). |
| **+** | **אופטימיזציית ביצועי Lineup ב-Level 1** *(נוסף מעבר לדו"ח)* | `01_build_level1_base.py` | ✅ **טופל** | מעבר ל-`itertuples()` ומיון מותנה, האצה פי 5 של הריצה. |
| **+** | **תיקון טרמינולוגיית QA מ-Impact לסטטיסטיקה תיאורית** *(נוסף)* | `check_level3_quality.py` | ✅ **טופל** | מניעת טענות שווא ל-"אימפקט" לפני אימון סיבתי. |

---

## 🔴 קריטי – סטטוס באגים מתודולוגיים

### באג #1: מומנטום ללא כיוון (Directionless Momentum) — [✅ טופל]
* **מה בוצע:** נבנה מנגנון Tug-of-War רציף. סלי 3 נק', 2 נק', חטיפות וחסימות מגדילים את המומנטום של הקבוצה המבצעת ומפחיתים את המומנטום של היריבה.
* **קוד מעודכן:** [`02_build_level2_momentum.py`](file:///c:/Users/david/finalPro/scripts/feature_engineering/02_build_level2_momentum.py)

### באג #2: אפסים מזויפים בסופי רבעים (Boundary Truncation) — [✅ טופל]
* **מה בוצע:** הוסרה אימפוטציית האפסים השגויה. הוגדר Right-Censoring אמיתי שמשאיר `NaN` במקום שבו לא ניתן לחזות את מלוא החלון (90s / 180s).
* **אימות אמפירי מלא:** נכתב [`verify_lookahead_censoring.py`](file:///c:/Users/david/finalPro/scripts/feature_engineering/validation/verify_lookahead_censoring.py) ודו"ח [`bug2_lookahead_validation_report.md`](file:///c:/Users/david/finalPro/docs/reports/bug2_lookahead_validation_report.md). כל 101,235 ה-NaNs של 90s וכל 184,909 ה-NaNs של 180s נובעים ב-100.00% מסופי רבעים (0 שגיאות מחוץ לגבול).

### באג #3: היפוך קוטביות ב-Target (`interest_sign`) — [✅ טופל]
* **מה בוצע:** ה-Target מחושב strictly ביחס לקבוצה היוזמת (פסק זמן של בית = עלייה בהפרש היא חיובית; פסק זמן של חוץ = ירידה בהפרש לטובת הבית היא חיובית עבור החוץ).
* **קוד מעודכן:** [`03_build_level3_labels.py`](file:///c:/Users/david/finalPro/scripts/feature_engineering/03_build_level3_labels.py)

### באג #4: משתנים סיבתיים חסרים (Omitted Confounders) — [✅ טופל]
* **מה בוצע:** `seconds_remaining` הוחזר למודל ולא נזרק ב-Blacklist. הוגדרה ארכיטקטורת רשימות שחורות מסודרת.
* **קוד מעודכן:** [`models/pipeline_constants.py`](file:///c:/Users/david/finalPro/models/pipeline_constants.py)

### באג #5a: זליגת שעון חילופים בין קבוצות — [✅ טופל]
* **מה בוצע:** פוצל השעון בלבל 1 ל-`time_since_last_sub_home` ו-`time_since_last_sub_away`. בלבל 2 נוספו `is_high_fatigue_home`, `is_high_fatigue_away`, ו-`stint_fatigue_diff`.
* **קוד מעודכן:** [`01_build_level1_base.py`](file:///c:/Users/david/finalPro/scripts/feature_engineering/01_build_level1_base.py), [`02_build_level2_momentum.py`](file:///c:/Users/david/finalPro/scripts/feature_engineering/02_build_level2_momentum.py)

### באג #5b: היעדר Cross-Fitting ב-X-Learner — [⏳ ממתין לשלב הבא]
* **מה נדרש:** שילוב K-Fold Cross-Fitting בשלב ה-Imputation של X-Learner.

---

## 🟠 רציני – סטטוס באגים בקוד

* **באג #6 (Clutch Time בכל הרבעים):** ✅ **טופל** (`period >= 4`).
* **באג #10 (שחיתות Rotations):** ✅ **טופל** (הקובץ תוקן ונבדק, 0 שגיאות).
* **באג #9 (סינון Parquet):** ⏳ ממתין לשלב יצירת ה-Splits למודלים.
* **באגים #11, #12 (מנוע המלצות ו-Hit Rate):** ⏳ ממתינים להחלפה ב-OPE וב-AUUC.

---

## 📋 הצלבה מול הערות ד"ר שפירא – צעדים קדימה

1. **הערה 1 (סיפור CATE הטרוגני):** הבסיס תוקן (מומנטום כיווני + לייבלים תקינים).
2. **הערה 2 (הגדרת Treatment & Outcomes):** תוקן (פולאריות נכונה + Right-Censoring).
3. **הערה 3 (Overlap & Covariate Balance - Love Plot/SMD):** ⏳ לבנייה בשלב המודלים.
4. **הערה 4 (Estimator נוסף - AIPW/Causal Forest):** ⏳ לבנייה עם `EconML`.
5. **הערה 5 (Cluster Bootstrap CI ברמת המשחק):** ⏳ לבנייה עם 1,000 resamples.
6. **הערה 6 (Placebo Tests & Sensitivity / E-Value):** ⏳ לבנייה לאחר אימון המודלים.
7. **הערה 7 (Lineup Accuracy Validation):** תשתית הרוטציות נוקתה (באג #10).
8. **הערה 8 (Off-Policy Evaluation - OPE):** ⏳ להחלפת מנוע ההמלצות הישן.
