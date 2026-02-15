import pandas as pd
import os

# --- הגדרות ---
FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'interim', 'level1_base.csv')

def check_event_context(df, event_name, text_trigger, col_substring):
    print(f"\n🏀 Testing Event: {event_name.upper()}")
    
    # 1. סינון השורות
    event_rows = df[df['description'].str.contains(text_trigger, case=False, na=False)]
    
    if event_rows.empty:
        print(f"   ⚠️ No events found for '{text_trigger}'."); return

    print(f"   Found {len(event_rows)} rows containing '{text_trigger}'.")

    # --- PART A: בדיקת עמודות ייעודיות ---
    relevant_cols = [c for c in df.columns if col_substring.lower() in c.lower() and 'remaining' not in c]
    print(f"   [A] Checking specific '{col_substring}' columns:")
    for col in relevant_cols:
        missing_pct = event_rows[col].isna().mean() * 100
        status = "✅ KEEP" if missing_pct < 20 else "🗑️  DROP CANDIDATE"
        if missing_pct > 99: status = "💀 DEAD (100% Empty)"
        print(f"     -> {col:<30} : {missing_pct:6.1f}% missing. {status}")

    # --- PART B: בדיקת זהות ראשית (עם חשיפת ערכים) ---
    print(f"   [B] Checking PRIMARY identity columns:")
    main_cols = ['personId', 'teamTricode']
    
    for col in main_cols:
        if col in df.columns:
            missing = event_rows[col].isna().mean() * 100
            zeros = (event_rows[col] == 0).mean() * 100 if pd.api.types.is_numeric_dtype(event_rows[col]) else 0
            
            status = "✅ PERFECT" if (missing + zeros) < 1 else "❌ BROKEN"
            print(f"     -> {col:<30} : {missing:6.1f}% NaN, {zeros:6.1f}% Zeros. {status}")
            
            # --- תוספת: הדפסת הערכים כדי לפתור את התעלומה ---
            if col == 'teamTricode':
                unique_vals = event_rows[col].unique()
                print(f"        🕵️‍♂️ VALUES FOUND: {unique_vals[:10]} {'...' if len(unique_vals)>10 else ''}")

def main():
    print(f"🕵️‍♂️ Starting QA...")
    if not os.path.exists(FILE_PATH): print("❌ File not found."); return

    df = pd.read_csv(FILE_PATH, low_memory=False)
    
    check_event_context(df, "Assists", "Assist", "assist")
    check_event_context(df, "Timeouts", "Timeout", "teamTricode") 
    check_event_context(df, "Turnovers", "Turnover", "teamTricode")

    print("\n🏁 Analysis Complete.")

if __name__ == "__main__":
    main()