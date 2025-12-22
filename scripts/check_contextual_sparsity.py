import pandas as pd
import os

# --- הגדרות ---
# נתיב לקובץ ה-Interim
FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'interim', 'level1_base.csv')

def check_event_context(df, event_name, text_trigger, col_substring):
    """
    בודק דלילות של עמודות ספציפיות + בדיקת תקינות של ה-ID הראשי.
    """
    print(f"\n🏀 Testing Event: {event_name.upper()}")
    
    # 1. סינון השורות בהן האירוע התרחש
    event_rows = df[df['description'].str.contains(text_trigger, case=False, na=False)]
    
    if event_rows.empty:
        print(f"   ⚠️ No events found for '{text_trigger}'.")
        return

    print(f"   Found {len(event_rows)} rows containing '{text_trigger}'.")

    # --- PART A: בדיקת עמודות ייעודיות ---
    relevant_cols = [c for c in df.columns if col_substring.lower() in c.lower()]
    # מסננים עמודות שלא מעניינות אותנו בבדיקה הזו (כמו מלאי פסקי זמן)
    relevant_cols = [c for c in relevant_cols if 'remaining' not in c] 
    
    print(f"   [A] Checking specific '{col_substring}' columns:")
    for col in relevant_cols:
        missing_pct = event_rows[col].isna().mean() * 100
        status = "✅ KEEP" if missing_pct < 20 else "🗑️  DROP CANDIDATE"
        if missing_pct > 99: status = "💀 DEAD (100% Empty)"
        print(f"     -> {col:<30} : {missing_pct:6.1f}% missing. {status}")

    # --- PART B: בדיקת עמודות זהות ראשיות (הקריטי לגרפים) ---
    print(f"   [B] Checking PRIMARY identity columns (Crucial for V3):")
    main_cols = ['personId', 'teamTricode'] # הקטנו את הרשימה לעיקר
    
    for col in main_cols:
        if col in df.columns:
            # בדיקת NaN
            missing = event_rows[col].isna().mean() * 100
            # בדיקת אפסים
            zeros = 0
            if pd.api.types.is_numeric_dtype(event_rows[col]):
                zeros = (event_rows[col] == 0).mean() * 100
            
            # אם יש חוסר משמעותי - זה מסביר למה הגרף ריק!
            status = "✅ PERFECT" if (missing + zeros) < 1 else "❌ BROKEN (Causes Empty Graphs)"
            print(f"     -> {col:<30} : {missing:6.1f}% NaN, {zeros:6.1f}% Zeros. {status}")
        else:
            print(f"     -> {col:<30} : COLUMN MISSING")

def main():
    print(f"🕵️‍♂️ Starting Advanced Contextual QA...")
    if not os.path.exists(FILE_PATH):
        print(f"❌ File not found at: {FILE_PATH}"); return

    df = pd.read_csv(FILE_PATH, low_memory=False)
    
    # --- 1. בדיקות מקוריות (לוודא שמה שעבד עדיין עובד) ---
    check_event_context(df, "Assists", "Assist", "assist")
    check_event_context(df, "Blocks", "Block", "block")
    check_event_context(df, "Steals", "Steal", "steal")

    # --- 2. בדיקות חדשות (אבחון הבעיה בגרפים) ---
    # כאן אנחנו בודקים: כשיש Timeout/Turnover, האם יש teamTricode?
    check_event_context(df, "Timeouts", "Timeout", "teamTricode") 
    check_event_context(df, "Turnovers", "Turnover", "teamTricode")

    print("\n🏁 Analysis Complete.")

if __name__ == "__main__":
    main()