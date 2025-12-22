import pandas as pd
import os

# --- הגדרות ---
# נתיב לקובץ ה-Interim (V2)
FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'interim', 'level1_base.csv')

def check_event_context(df, event_name, text_trigger, col_substring):
    """
    בודק דלילות של עמודות ספציפיות רק בתוך השורות שבהן האירוע קרה.
    """
    print(f"\n🏀 Testing Event: {event_name.upper()}")
    
    # 1. סינון השורות בהן האירוע התרחש (לפי התיאור)
    event_rows = df[df['description'].str.contains(text_trigger, case=False, na=False)]
    
    if event_rows.empty:
        print(f"   ⚠️ No events found for '{text_trigger}'.")
        return

    print(f"   Found {len(event_rows)} rows containing '{text_trigger}'.")

    # 2. מציאת כל העמודות הקשורות לאירוע (לפי השם)
    relevant_cols = [c for c in df.columns if col_substring.lower() in c.lower()]
    
    if not relevant_cols:
        print(f"   ❌ No columns found matching substring '{col_substring}'.")
        return

    # 3. בדיקת חוסרים
    print(f"   Checking {len(relevant_cols)} related columns:")
    for col in relevant_cols:
        # אחוז החוסרים רק בשורות הרלוונטיות
        missing_pct = event_rows[col].isna().mean() * 100
        
        # החלטה: אם חסר ב-99% מהמקרים שבהם האירוע קרה - העמודה כנראה מיותרת
        status = "✅ KEEP" if missing_pct < 20 else "🗑️  DROP CANDIDATE"
        if missing_pct > 99: status = "💀 DEAD (100% Empty)"
            
        print(f"     -> {col:<30} : {missing_pct:6.1f}% missing. {status}")

def main():
    print(f"🕵️‍♂️ Starting Advanced Contextual QA...")
    if not os.path.exists(FILE_PATH):
        print("❌ File not found."); return

    df = pd.read_csv(FILE_PATH, low_memory=False)
    
    # --- בדיקות הקשריות ---
    # 1. Assists
    check_event_context(df, "Assists", "Assist", "assist")
    
    # 2. Blocks
    check_event_context(df, "Blocks", "Block", "block")
    
    # 3. Steals
    check_event_context(df, "Steals", "Steal", "steal")

    print("\n🏁 Analysis Complete.")

if __name__ == "__main__":
    main()