# 🔍 NBA AI Coach – דו"ח ביקורת קוד מלא + הצלבה עם הערות ד"ר שפירא

> **מטרה:** מיפוי מצב הקוד הקיים + הצלבה עם ההערות האקדמיות של ד"ר שפירא, לפני החלטה אם ואיך להתקדם לכיוון מאמר.  
> **גישה:** כנות מלאה, עם פרופורציה – לא הכול שבור, ולא הכול צריך תיקון.  
> **סטטוס עדכני:** שלב א' (תשתית ו-Feature Engineering) **הושלם ותוקן במלואו**. ענף פעיל: `paper-preparation`.

---

## TL;DR – סיכום ב-4 משפטים

1. **קוד הדמו (app.py, prepare_demo_data, context_rules) הוא 100% תיאטרון** – וזה בסדר גמור לפרויקט הנדסי. למאמר – הוא פשוט לא רלוונטי.
2. **ה-Pipeline של הנתונים (Feature Engineering) הכיל באגים מתודולוגיים אמיתיים** – **כל הבאגים הללו (#1, #2, #3, #4, #5a, #6, #10) תוקנו, אומתו אמפירית ועברו QA מלא.**
3. **המודל הסיבתי (X-Learner) בנוי נכון ברמת הארכיטקטורה**, אך דורש K-Fold Cross-Fitting ומבחני ולידציה אקדמיים (שלב העבודה הבא).
4. **ד"ר שפירא צודקת ב-100% של ההערות שלה**, וכעת לאחר שניקינו את בסיס הנתונים לחלוטין – ניתן לבנות את השכבות האקדמיות שהיא ביקשה על גבי תשתית יציבה.

---

## 🗑️ לא רלוונטי למאמר (דמו בלבד) – לא צריך לתקן

| קובץ | מה הבעיה | למה לא רלוונטי |
|:---|:---|:---|
| [app.py](file:///c:/Users/david/finalPro/app.py) | SHAP values הארדקודד, CATE ו-Propensity מיוצרים עם `np.random.uniform`, Win Probability = סיגמואיד נאיבי על margin | זה ה-Streamlit Demo UI – במאמר לא מציגים אפליקציה |
| [prepare_demo_data.py](file:///c:/Users/david/finalPro/models/prepare_demo_data.py) | שני משחקים מלאכותיים עם תסריטים קבועים מראש (הפסד ב-18, ניצחון ב-+2), CATE/SHAP/Propensity הארדקודד, Brute-force margin overrides | Mock data ל-UI בלבד |
| [context_rules.md](file:///c:/Users/david/finalPro/context_rules.md) | מפרט "Base Score Flattening" והזרקת ניקוד מלאכותי כדי לעמוד ב-margins ספציפיים | הוראות לגנרטור הדמו |
| [validate_logs.py](file:///c:/Users/david/finalPro/validate_logs.py) | בדיקות על תסריט הדמו התיאטרלי (Q3 חייב להגיע ל-exactly -18, Q4 ל-exactly +2) | ולידציה של הדמו, לא של המחקר |
| [export_to_supabase.py](file:///c:/Users/david/finalPro/scripts/export_to_supabase.py) | Fallback לנתונים סינתטיים כשקבצים חסרים, threshold שרירותי של 0.05, cap של 50 שורות | ייצוא ל-DB של הדמו |
| [generate_impact_report.py](file:///c:/Users/david/finalPro/models/generate_impact_report.py) | טבלת נתונים הארדקודד לגמרי (לא קורא שום קובץ), Y-axis חתוך (90-100%), AUC מועתק, נוסחת wins נאיבית | גרפים למצגת, לא לניתוח |

> [!TIP]
> **המסקנה:** כל קוד הדמו הוא בדיוק מה שהיה צריך להיות לפרויקט הנדסי. הוא פשוט לא נכנס למאמר. נקודה.

---

## 🔴 קריטי – באגים מתודולוגיים שחייבים תיקון לפני מאמר

---

### באג #1: מומנטום ללא כיוון (Directionless Momentum) — [✅ טופל בהצלחה]
**קובץ:** [02_build_level2_momentum.py](file:///c:/Users/david/finalPro/scripts/feature_engineering/02_build_level2_momentum.py) | **שורות:** 98–128

* **מה היה הליקוי:** `event_momentum_val` תמיד הוסיף ערך חיובי – בין אם Home ובין אם Away קלעו. המומנטום מדד "קצב אירועים" כללי ולא שליטה של קבוצה.
* **הפתרון שהוטמע:** הומר למנגנון **Tug-of-War רציף**:
  - סלים, חטיפות וחסימות של קבוצת הבית מגדילים את `home_momentum_streak` ומכרסמים/מפחיתים את המומנטום של קבוצת החוץ (`away_momentum_streak`).
  - פעולות חיוביות של החוץ מגדילות את `away_momentum_streak` ומפחיתות את מומנטום הבית.
  - נשמר הפרש מומנטום כיווני: `momentum_delta = home - away`.

---

### באג #2: Target Labels מאופסים בסוף רבעים (Horizon Boundary Distortion) — [✅ טופל בהצלחה]
**קובץ:** [03_build_level3_labels.py](file:///c:/Users/david/finalPro/scripts/feature_engineering/03_build_level3_labels.py) | **שורות:** 83–105

* **מה היה הליקוי:** `merge_asof` חיפש מהלכים עתידיים רק בתוך אותו רבע. ב-90/180 שניות האחרונות, `fillna(current)` כפה `delta = future - current = 0`, מה שהזריק ~25% אפסים מלאכותיים לתוויות התוצאה.
* **הפתרון שהוטמע:** יישום **Right-Censoring אמיתי** (השארת ערכי `NaN` מבוקרים בסופי רבעים ללא זיהום ב-0).
* **אימות אמפירי (686,008 מהלכים):**
  - בחלון 90s: בדיוק 101,235 חסרים (14.76%) — **100.00%** מהם נמצאים ב-$t < 90$s; בדיוק 0 חריגות מחוץ לגבול.
  - בחלון 180s: בדיוק 184,909 חסרים (26.95%) — **100.00%** מהם נמצאים ב-$t < 180$s; בדיוק 0 חריגות מחוץ לגבול.
  - תועד ב-[`docs/reports/bug2_lookahead_validation_report.md`](file:///c:/Users/david/finalPro/docs/reports/bug2_lookahead_validation_report.md) ונבדק ב-[`verify_lookahead_censoring.py`](file:///c:/Users/david/finalPro/scripts/feature_engineering/validation/verify_lookahead_censoring.py).

---

### באג #3: היפוך קוטביות ב-Target (Interest Sign Inversion) — [✅ טופל בהצלחה]
**קובץ:** [03_build_level3_labels.py](file:///c:/Users/david/finalPro/scripts/feature_engineering/03_build_level3_labels.py) | **שורה:** 128

* **מה היה הליקוי:** `interest_sign = np.where(score_margin > 0, -1, 1)` התבסס על מי שמוביל ברגע נתון ולא על מי שקרא לפסק זמן. קבוצה מובילה שהגדילה את יתרונה תויגה כתוצאה שלילית.
* **הפתרון שהוטמע:** קוטביות התוצאה מוגדרת תמיד ביחס לקבוצה היוזמת (`acting_team_id` / קבוצה שלקחה פסק זמן): `+1` לפעולת בית, `-1` לפעולת חוץ.

---

### באג #4: משתנים סיבתיים חסרים (Omitted Confounders) — [✅ טופל בהצלחה]
**קובץ:** [pipeline_constants.py](file:///c:/Users/david/finalPro/models/pipeline_constants.py) | **שורות:** 25–50

* **מה היה הליקוי:** `seconds_remaining` הוכנס ל-Blacklist והוסר מהמודל הסיבתי, מה שהפר את הנחת ה-Unconfoundedness ($Y(0), Y(1) \perp T \mid X$).
* **הפתרון שהוטמע:** הקובץ שוכתב לארכיטקטורה מודולרית (`v1_standard_causal`). משתני שעון המשחק (`seconds_remaining`, `is_clutch_time`) שוחזרו ונשמרים לכל שלבי האימון.

---

### באג #5a: זליגת שעון חילופים בין קבוצות (Sub Timer Leakage) — [✅ טופל בהצלחה] *(נוסף)*
**קבצים:** [01_build_level1_base.py](file:///c:/Users/david/finalPro/scripts/feature_engineering/01_build_level1_base.py) + [02_build_level2_momentum.py](file:///c:/Users/david/finalPro/scripts/feature_engineering/02_build_level2_momentum.py)

* **מה היה הליקוי:** `time_since_last_sub` שרשר את שתי החמישיות יחד. חילוף של היריבה איפס את שעון העייפות לשתי הקבוצות, מה שגרם לעיוות ב-`is_high_fatigue` בלבל 2.
* **הפתרון שהוטמע:**
  - בלבל 1: פוצל לשני שעונים עצמאיים: `time_since_last_sub_home` ו-`time_since_last_sub_away`.
  - בלבל 2: נוצרו אינדיקטורים כיווניים: `is_high_fatigue_home`, `is_high_fatigue_away`, והפרש רעננות `stint_fatigue_diff`.

---

### באג #5b: אין Cross-Fitting ב-X-Learner — [⏳ ממתין לשלב המודלים]
**קובץ:** [06_causal_x_learner.py](file:///c:/Users/david/finalPro/models/06_causal_x_learner.py)

* **מה קורה:** ה-Imputation של ה-Counterfactuals ($D_0, D_1$) נעשה In-Sample ללא Cross-Fitting. מודלי ה-Outcome ($\mu_0, \mu_1$) חוזים על אותו דאטה שעליו הם אומנו.
* **השלכה:** Overfitting מזהם את אמידת ה-CATE.
* **תיקון נדרש:** מימוש K-Fold Cross-Fitting מלא כסטנדרט ב-DML/AIPW.

---

## 🟠 רציני – באגים בקוד שדורשים תיקון

---

### באג #6: Clutch Time מופעל בכל הרבעים — [✅ טופל בהצלחה]
**קובץ:** [02_build_level2_momentum.py](file:///c:/Users/david/finalPro/scripts/feature_engineering/02_build_level2_momentum.py) | **שורות:** 133–135

* **מה היה הליקוי:** "Clutch Time" הופעל בכל רבע שבו השעון הראה $\le 300$ וההפרש $\le 5$, כולל ברבע הראשון.
* **הפתרון שהוטמע:** נוסף תנאי קשיח `& (self.df['period'] >= 4)`.

---

### באג #7: Shot Clock לא מתאפס אחרי ריבאונד התקפי — [⚪ עקיף/נבדק ב-QA]
**קובץ:** [01_build_level1_base.py](file:///c:/Users/david/finalPro/scripts/feature_engineering/01_build_level1_base.py)

* **סטטוס:** הלוגיקה נבדקה ב-Level 1 QA (חוק 14 שניות). שורות ריבאונד התקפי מוגבלות לתקרה של 14.1 שניות.

---

### באג #8: ספירת פסקי זמן לא מדויקת — [⚪ מאומת ב-QA]
**קובץ:** [01_build_level1_base.py](file:///c:/Users/david/finalPro/scripts/feature_engineering/01_build_level1_base.py)

* **סטטוס:** הטווח נבדק ומאומת ב-`check_timeouts_inventory_integrity()` ב-Level 1 QA.

---

### באג #9: קבצי Parquet מכילים Features שצריכים להיות מסוננים — [⏳ ממתין לשלב המודלים]
**קובץ:** [prepare_ml_splits.py](file:///c:/Users/david/finalPro/models/prepare_ml_splits.py) | **שורות:** 115–134

* **מה קורה:** ה-Blacklist מיושם ב-JSON אך קבצי ה-Parquet עצמם כוללים עמודות דולפות.
* **תיקון נדרש:** סינון ה-DataFrames לפני הייצוא לקבצי Parquet בעת בניית ה-Splits החדשים.

---

### באג #10: שחיתות נתונים ב-Rotations CSV — [✅ טופל בהצלחה]
**קבצים:** [fetch_rotations.py](file:///c:/Users/david/finalPro/scripts/fetch_rotations.py) + [rotations_2024_25.csv](file:///c:/Users/david/finalPro/data/pureData/rotations_2024_25.csv)

* **מה היה הליקוי:** הרצת סקריפט הצלה ישן (`s1.py`) הזיזה עמודות ב-31,232 שורות ב-CSV ויצרה שחיתות ב-`team_side`.
* **הפתרון שהוטמע:** הקובץ תוקן, כל 31,232 השורות המוזזות יושרו, נבדק ש-100% מעמודת `team_side` תקינה (`home`/`away`), והסקריפט המזיק נוטרל (`draft_s1_rescue.py`).

---

### באגים נוספים שטופלו מעבר לדו"ח המקורי:

* **באג א-סימטריה בכוכבים (`is_star_resting`) — [✅ טופל]:**  
  במקום מדד שמחזיר 1 רק כששתי הקבוצות מנוחות כוכב, הוטמע מדד יתרון יחסי:  
  `star_advantage = home_has_star - away_has_star` (`+1` יתרון לבית, `0` שוויון, `-1` יתרון לחוץ).
* **אופטימיזציית ביצועים ב-Level 1 (`process_lineups_logic`) — [✅ טופל]:**  
  מעבר מ-iterrows כבד ל-`itertuples()` עם מיון חמישיות רק בעת חילוף בפועל, מה שהאיץ את זמן הריצה ביותר מפי 5.
* **תיקון טרמינולוגיית QA מ-Impact לסטטיסטיקה תיאורית — [✅ טופל]:**  
  תוקנה ההדפסה ב-`check_level3_quality.py` שמיתגה ממוצע גולמי כ-"אימפקט של פסקי זמן", לסטטיסטיקה תיאורית בלבד ($E[Y \mid T=1]$).

---

### באג #11: Recommendation Engine – השוואה סיבתית שגויה — [⏳ ממתין לשלב ג']
**קובץ:** [recommendation_engine.py](file:///c:/Users/david/finalPro/models/recommendation_engine.py)

* **הליקוי:** השוואה אובסרבציונלית נאיבית של `ignored.mean() - complied.mean()`.
* **תוכנית:** יוחלף ב-Off-Policy Evaluation (OPE) מבוסס Doubly Robust / AIPW.

---

### באג #12: Hit Rate Sweep – ולידציה מעגלית — [⏳ ממתין לשלב ג']
**קובץ:** [hit_rate_sweep.py](file:///c:/Users/david/finalPro/models/hit_rate_sweep.py)

* **הליקוי:** ולידציה מעגלית מול ספים שרירותיים של אותה מערכת.
* **תוכנית:** יוחלף בעקומות Uplift / AUUC (Area Under Uplift Curve).

---

## 🟡 בינוני – Code Quality & Logging

| נושא | היכן | תיאור | סטטוס |
|:---|:---|:---|:---:|
| **אין Logging בכלל** | כל הפרויקט | הכול `print()`, אין `logging` module | ⏳ יטופל בהמשך |
| **מבחנים שתמיד עוברים** | [run_all_tests.py](file:///c:/Users/david/finalPro/scripts/test_and_val/run_all_tests.py) | עודכנו סקריפטי איכות קפדניים (`check_level1/2/3_quality.py`) | ✅ **נפתר ב-QA החדש** |
| **save_models() לא נקרא** | [06_causal_x_learner.py](file:///c:/Users/david/finalPro/models/06_causal_x_learner.py) | הפונקציה קיימת אך לא נקראת ב-`main` | ⏳ יסודר באימון המודלים |
| **Default Usage Rate** | [02_build_level2_momentum.py](file:///c:/Users/david/finalPro/scripts/feature_engineering/02_build_level2_momentum.py) | מבוסס על קובץ `high_usage_players_2024-25.csv` | ✅ מאומת ב-QA |
| **`is_star_resting` שגוי** | [02_build_level2_momentum.py](file:///c:/Users/david/finalPro/scripts/feature_engineering/02_build_level2_momentum.py) | תוקן ל-`star_advantage` דיפרנציאלי | ✅ **טופל** |

---

## ✅ מה עובד טוב ומאומת

| רכיב | סטטוס |
|:---|:---|
| **איסוף נתונים גולמיים (Play-by-Play)** | עובד – 4 עונות שלמות (~900MB) |
| **מבנה Pipeline 3 שלבים (Level 1→2→3)** | ✅ מאומת ונקי לחלוטין (עבר את כל ה-Validators) |
| **חלוקת Train/Val/Test לפי gameId** | נכון מתודולוגית (מונע דליפה בין החזקות באותו משחק) |
| **ארכיטקטורת X-Learner** | בסיס קיים, ממתין ל-Cross-Fitting |
| **Right-Censoring אמפירי** | 100.00% דיוק בסופי רבעים (נבדק על 686k שורות) |

---

## 📋 הצלבה: הערות ד"ר שפירא vs. מצב בקוד

---

### הערה 1: "החידוש – CATE, לא סתם האם timeout עובד"
* **סטטוס:** תשתית הנתונים (מומנטום כיווני, עייפות נפרדת, ללא היפוך סימנים) נקייה כעת ומאפשרת לימוד CATE אמיתי.

### הערה 2: "חדד Treatment, Outcomes, אין מידע עתידי"
* **סטטוס:** תוויות התוצאה נוקו לחלוטין מ-25% האפסים המזויפים והיפוך הסימנים (באגים #2 ו-#3 תוקנו במלואם).

### הערה 3: "בדיקות Overlap ו-Covariate Balance"
* **סטטוס:** ⏳ לבנייה בשלב המודלים (Love Plot, SMD, Propensity Histogram).

### הערה 4: "Estimator נוסף – DR-Learner / AIPW / Causal Forest"
* **סטטוס:** ⏳ לבנייה עם `EconML` לאחר שלב ה-X-Learner.

### הערה 5: "Confidence Intervals באמצעות Bootstrap ברמת משחק"
* **סטטוס:** ⏳ מתוכנן לשלב ג' (Cluster Bootstrap לפי `gameId`).

### הערה 6: "Placebo Tests ו-Sensitivity Analysis ל-Unmeasured Confounding"
* **סטטוס:** ⏳ מתוכנן לשלב ג' (בדיקת פסק זמן מדומה ו-E-Values).

### הערה 7: "בדיקת Accuracy של מנוע שחזור החמישיות"
* **סטטוס:** תוקנה שחיתות קובץ הרוטציות (באג #10), 97.6% התאמה לרוטציות הרשמיות.

### הערה 8: "הערכת מדיניות פורמלית – Off-Policy Evaluation"
* **סטטוס:** ⏳ מתוכנן לשלב ג' (החלפת מנוע ההמלצות הישן ב-Doubly Robust OPE).

---

## 🎯 Roadmap מעודכן – סדר עבודה

### שלב א' – תיקון בסיס ו-Feature Engineering (✅ הושלם במלואו)
- [x] תיקון באג #1 – מומנטום עם כיוון (Tug-of-War רציף)
- [x] תיקון באג #2 – Targets חוצי-רבעים (Right-Censoring אמפירי מאומת)
- [x] תיקון באג #3 – Interest Sign לפי קבוצה פועלת
- [x] תיקון באג #4 – החזרת `seconds_remaining` + תיקון pipeline_constants
- [x] תיקון באג #5a – הפרדת שעוני חילופים ועייפות לבית/חוץ
- [x] תיקון באג #6 – Clutch Time period check (`period >= 4`)
- [x] תיקון באג #10 – סכמת Rotations CSV (תיקון 31k שורות מוזזות)
- [x] תיקון מדד הכוכבים הנחים (`star_advantage`)
- [x] הרצה ואימות מלא של כל סוויטת ה-QA (Level 1, Level 2, Level 3 - כולם PASSED)

### שלב ב' – הכנת הנתונים למודלים (השלב הבא)
- [ ] באג #9 – סינון Features דולפים מקבצי ה-Parquet ב-`prepare_ml_splits.py`
- [ ] יצירת קבצי Train / Val / Test נקיים

### שלב ג' – מודלים סיבתיים ומרכיבים אקדמיים
- [ ] באג #5b – מימוש K-Fold Cross-Fitting ב-`06_causal_x_learner.py`
- [ ] שמירת מודלים מאומנים לדיסק (`save_models`)
- [ ] הערה #3 – Overlap & Covariate Balance (Love Plot, SMD, Propensity histogram)
- [ ] הערה #4 – AIPW / Causal Forest (EconML) + השוואה ל-X-Learner
- [ ] הערה #5 – Cluster Bootstrap CI ברמת gameId
- [ ] הערה #6 – Placebo Test + Sensitivity Analysis (E-Value)
- [ ] הערה #8 – Off-Policy Evaluation (OPE) מבוסס AIPW (במקום recommendation_engine הישן)

### שלב ד' – בדיקת תוצאות וכתיבת מאמר
- [ ] האם התוצאות יציבות אחרי כל התיקונים?
- [ ] האם CATE מראה הטרוגניות משמעותית (באילו מצבים פסק זמן באמת עוזר)?
- [ ] האם רווחי הסמך מובהקים?
