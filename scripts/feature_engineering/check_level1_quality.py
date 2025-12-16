import pandas as pd
import os

# נתיב לקובץ החדש שיצרנו
FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'interim', 'level1_base.csv')

def run_quality_check():
    print(f"🕵️‍♂️ Running Quality Control on: level1_base.csv")
    
    try:
        # אם הקובץ כבד מידי, אפשר להוסיף nrows=10000 לבדיקה זריזה
        df = pd.read_csv(FILE_PATH, low_memory=False)
    except FileNotFoundError:
        print("❌ File not found! Run the build script first.")
        return

    print("-" * 50)
    print(f"📊 Total Rows: {len(df)}")
    print("-" * 50)

    # 1. בדיקת זמנים
    print("\n⏱️ Time Check (Seconds Remaining):")
    print(df[['clock', 'seconds_remaining']].sample(5).to_string(index=False))

    # 2. בדיקת טיים-אאוטים (האם שמות הקבוצות חולצו?)
    print("\nTIMEOUTS FOUND (Top 10 Teams):")
    # מסננים את 'None' כדי לראות רק מי לקח טיים אאוט בפועל
    timeouts_only = df[df['timeout_type'] != 'None']
    if not timeouts_only.empty:
        print(timeouts_only['timeout_type'].value_counts().head(10))
        print(f"\nTotal Timeouts Identified: {len(timeouts_only)}")
    else:
        print("❌ No timeouts classified (Check extraction logic).")

    # 3. בדיקת מילוי תוצאה
    print("\n🏀 Score Check (Random Sample):")
    sample = df[df['period'] > 1].sample(5)[['period', 'clock', 'scoreHome', 'scoreAway', 'score_margin']]
    print(sample.to_string(index=False))

    # 4. בדיקת נתונים חסרים קריטיים
    print("\n⚠️ Missing Values Check:")
    # timeout_type לא אמור להיות NaN אלא 'None', אז בודקים אם יש NaN אמיתי
    print(df[['scoreHome', 'personIdsFilter', 'timeout_type']].isna().sum())

if __name__ == "__main__":
    run_quality_check()