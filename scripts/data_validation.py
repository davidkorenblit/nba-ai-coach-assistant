import pandas as pd
import os

# --- הגדרות נתיבים יחסיים ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, '..', 'data', 'pureData')
INPUT_FILENAME = "season_2024_25.csv" 
INPUT_FILE = os.path.join(DATA_DIR, INPUT_FILENAME)

def check_general_health(df):
    """
    מבצע בדיקת בריאות בסיסית: גודל דאטא וערכים חסרים
    """
    print("\n=== DATA HEALTH REPORT ===")
    
    # 1. מימדים
    rows, cols = df.shape
    print(f"Total Rows:    {rows:,}")
    print(f"Total Columns: {cols}")
    
    # 2. ערכים חסרים (NaN)
    print("\n--- Missing Values Analysis ---")
    
    # סופרים nulls בכל עמודה
    null_counts = df.isnull().sum()
    
    # מסננים רק עמודות שיש בהן חוסרים, וממיינים מהכי חסר להכי פחות
    missing_data = null_counts[null_counts > 0].sort_values(ascending=False)
    
    if not missing_data.empty:
        print(f"{'Column Name':<30} | {'Missing':<10} | {'% of Total'}")
        print("-" * 55)
        for col, count in missing_data.items():
            percent = (count / rows) * 100
            print(f"{col:<30} | {count:<10} | {percent:.2f}%")
    else:
        print("[V] Perfect! No missing values found in any column.")

def main():
    # בדיקה שהקובץ קיים
    if not os.path.exists(INPUT_FILE):
        print(f"Error: File not found at {INPUT_FILE}")
        print("Please run the data collector script first.")
        return
    
    print(f"Loading data from: {INPUT_FILENAME}...")
    try:
        # טעינת הדאטא (low_memory=False מונע אזהרות בקבצים גדולים)
        df = pd.read_csv(INPUT_FILE, low_memory=False)
        check_general_health(df)
        
    except Exception as e:
        print(f"Failed to load CSV: {e}")

if __name__ == "__main__":
    main()