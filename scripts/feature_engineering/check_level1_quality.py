import pandas as pd
import os

# נתיב לקובץ החדש (המאוחד)
FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'interim', 'level1_base.csv')

def run_quality_check():
    print(f"🕵️‍♂️ Running EXTENDED Quality Control on: level1_base.csv")
    
    try:
        # טעינת הקובץ
        df = pd.read_csv(FILE_PATH, low_memory=False)
    except FileNotFoundError:
        print("❌ File not found! Run the build script first.")
        return

    print("-" * 60)
    print(f"📊 Total Rows: {len(df)}")
    print("-" * 60)

    # --- 1. בדיקת הפיצ'רים החדשים (הכי חשוב!) ---
    print("\n🆕 NEW FEATURES SANITY CHECK:")
    print("-------------------------------")
    
    # בדיקה סטטיסטית: האם משך מהלך הגיוני? האם שעון הזריקות לא שלילי?
    # אנו מצפים לראות Min=0 (ולא מספר שלילי) ו-Max שלא עולה על כ-24-30 במשך מהלך רגיל (אלא אם יש עצירות)
    stats = df[['play_duration', 'shot_clock_estimated']].describe().loc[['min', 'max', 'mean', '50%']]
    print(stats)

    # --- 2. בדיקת פוזשנים ---
    print("\n🔄 POSSESSION LOGIC CHECK:")
    # נבדוק מה המקסימום פוזשנים למשחק. במשחק NBA ממוצע יש כ-100 פוזשנים.
    # אם נקבל 5 או 2000 - יש בעיה בלוגיקה.
    max_poss = df.groupby('gameId')['possession_id'].max().mean()
    print(f"   Avg Possessions per Game: {max_poss:.1f} (Should be around 95-105)")

    # --- 3. בדיקת הרכבים ---
    print("\n👥 LINEUPS SAMPLE:")
    # נוודא שהעמודות לא ריקות ושיש בהן רשימות של ID
    print(df[['home_lineup', 'away_lineup']].sample(3).to_string(index=False))

    # --- 4. בדיקות קודמות (תקינות בסיסית) ---
    print("\n⏱️ TIME & SCORE CHECK (Random Sample):")
    cols_to_show = ['period', 'seconds_remaining', 'scoreHome', 'score_margin', 'play_duration']
    print(df[cols_to_show].sample(5).to_string(index=False))

    # --- 5. בדיקת טיים-אאוטים ---
    print("\nTIMEOUTS FOUND (Top 5 Teams):")
    timeouts_only = df[df['timeout_type'] != 'None']
    if not timeouts_only.empty:
        print(timeouts_only['timeout_type'].value_counts().head(5))
        print(f"   Total Timeouts: {len(timeouts_only)}")
    else:
        print("❌ No timeouts classified.")

    # --- 6. בדיקת נתונים חסרים (Missing Values) ---
    print("\n⚠️ MISSING VALUES CHECK (Critical Columns):")
    # נבדוק אם נוצרו חורים בנתונים החדשים
    critical_cols = ['scoreHome', 'play_duration', 'shot_clock_estimated', 'home_lineup', 'possession_id']
    missing = df[critical_cols].isna().sum()
    if missing.sum() == 0:
        print("✅ Perfect! No missing values in critical columns.")
    else:
        print(missing[missing > 0])

if __name__ == "__main__":
    run_quality_check()