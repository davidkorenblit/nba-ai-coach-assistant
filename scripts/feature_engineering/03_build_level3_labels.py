import pandas as pd
import numpy as np
import os

print("🚀 Starting Level 3: Label Generation (90s & 180s Windows)...")

# 1. טעינת הנתונים
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
level2_path = os.path.join(base_dir, 'data', 'interim', 'level2_features.csv')
output_path = os.path.join(base_dir, 'data', 'interim', 'level3_labels.csv')

try:
    df = pd.read_csv(level2_path, low_memory=False)
    print(f"✅ Loaded {len(df)} rows from Level 2.")
except FileNotFoundError:
    print("❌ Error: level2_features.csv not found.")
    exit()

# הגדרת עמודות היעד לפי הרשימה שחילצנו
col_margin = 'score_margin'
col_mom = 'momentum_streak_rolling'
col_exp = 'explosiveness_index'
col_fatigue = 'is_high_fatigue'

print("⏳ Creating time indices for Lookahead...")
# כדי לנוע קדימה בזמן ביעילות, ניצור עמודת "זמן שחלף" בתוך הרבע (מתחיל ב-0 ועולה)
df['period_start_time'] = df.groupby(['gameId', 'period'])['seconds_remaining'].transform('max')
df['time_elapsed'] = df['period_start_time'] - df['seconds_remaining']

# מיון חובה עבור פונקציית המיזוג העתידי
df = df.sort_values(by=['gameId', 'period', 'time_elapsed']).reset_index(drop=True)

# מגדירים את חלונות הזמן לחיפוש העתיד
df['target_time_90'] = df['time_elapsed'] + 90
df['target_time_180'] = df['time_elapsed'] + 180

# 2. חילוץ נתוני העתיד (Lookahead)
df_future = df[['gameId', 'period', 'time_elapsed', col_margin, col_mom, col_exp]].copy()
df_future.rename(columns={col_margin: 'fut_margin', col_mom: 'fut_mom', col_exp: 'fut_exp'}, inplace=True)

print("⏳ Merging future states (90s & 180s)...")

# הכנה ומיזוג חלון 90 שניות
df = df.sort_values('target_time_90')
df_future_90 = df_future.add_suffix('_90s').rename(columns={'gameId_90s': 'gameId', 'period_90s': 'period'}).sort_values('time_elapsed_90s')

merged = pd.merge_asof(
    df, 
    df_future_90,
    left_on='target_time_90', 
    right_on='time_elapsed_90s',
    by=['gameId', 'period'],
    direction='forward'
)

# הכנה ומיזוג חלון 180 שניות
merged = merged.sort_values('target_time_180')
df_future_180 = df_future.add_suffix('_180s').rename(columns={'gameId_180s': 'gameId', 'period_180s': 'period'}).sort_values('time_elapsed_180s')

merged = pd.merge_asof(
    merged, 
    df_future_180,
    left_on='target_time_180', 
    right_on='time_elapsed_180s',
    by=['gameId', 'period'],
    direction='forward'
)

# החזרת המיון המקורי וההגיוני כדי שהכל יישאר מסודר
df = merged.sort_values(by=['gameId', 'period', 'time_elapsed']).reset_index(drop=True)

# אם המשחק/רבע נגמר לפני שעברו 90/180 שניות, נמלא את החסר במצב הנוכחי של אותו רגע
for prefix in ['_90s', '_180s']:
    df['fut_margin'+prefix] = df['fut_margin'+prefix].fillna(df[col_margin])
    df['fut_mom'+prefix] = df['fut_mom'+prefix].fillna(df[col_mom])
    df['fut_exp'+prefix] = df['fut_exp'+prefix].fillna(df[col_exp])

print("🎯 Generating Machine Learning Targets (Labels)...")

# 3. יצירת עמודות ה-Labels הבינאריות (0 או 1)

# Target 1: כיבוי שריפות מיידי - האם האקספלוסיביות נבלמה תוך 90 שניות?
df['delta_exp_90s'] = df['fut_exp_90s'] - df[col_exp]
df['target_stop_run_90s'] = (df['delta_exp_90s'] < 0).astype(int)

# Target 2: היפוך מגמה - האם מומנטום המשחק עבר אלינו תוך 180 שניות?
df['delta_mom_180s'] = df['fut_mom_180s'] - df[col_mom]
df['target_reverse_trend_180s'] = (df['delta_mom_180s'] > 0).astype(int)

# Target 3: שיפור תוצאה ישיר (הפרש נקודות)
df['delta_margin_90s'] = df['fut_margin_90s'] - df[col_margin]
df['delta_margin_180s'] = df['fut_margin_180s'] - df[col_margin]
df['target_improve_margin_90s'] = (df['delta_margin_90s'] > 0).astype(int)
df['target_improve_margin_180s'] = (df['delta_margin_180s'] > 0).astype(int)

# Target 4: מחיר ההתעלמות (Danger Zone Penalty)
# סכנה = עייפות קיצונית קיימת + ריצת אקספלוסיביות של היריבה ברבעון העליון (אחוזון 75)
threshold_exp = df[col_exp].quantile(0.50)
is_danger = (df[col_fatigue] == 1) & (df[col_exp] > threshold_exp)
is_timeout = df['actionType'].str.contains('timeout', case=False, na=False)

# עונש = היינו בסכנה, המאמן *לא* התערב, וההפרש שלנו המשיך להצטמצם תוך 3 דקות
df['target_danger_penalty'] = (is_danger & ~is_timeout & (df['delta_margin_180s'] < 0)).astype(int)

# 4. ניקוי וייצוא
print("🧹 Cleaning up temporary columns...")
cols_to_drop = [
    'period_start_time', 'time_elapsed', 'target_time_90', 'target_time_180', 
    'time_elapsed_90s', 'time_elapsed_180s', 'fut_margin_90s', 'fut_mom_90s', 
    'fut_exp_90s', 'fut_margin_180s', 'fut_mom_180s', 'fut_exp_180s'
]
df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

print("💾 Saving Level 3 Labels dataset...")
df.to_csv(output_path, index=False)
print(f"✅ Success! Data is labeled and ready for modeling at: {output_path}")