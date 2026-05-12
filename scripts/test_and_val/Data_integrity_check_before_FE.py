import pandas as pd
import os
import glob

# --- הגדרת נתיבים ---
# מניחים שהסקריפט רץ מתוך תיקיית scripts או תת-תיקייה שלה
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# ניסיון לאתר את תיקיית הדאטה (עולה למעלה עד שמוצא)
# מותאם למבנה: project/data/pureData ו-project/scripts/...
DATA_DIR = os.path.join(CURRENT_DIR, '..', '..','data', 'pureData')

# רשימת העמודות הקריטיות לבדיקה (לפי מה שסיכמנו)
COLUMNS_TO_CHECK = [
    # 1. זהות וסדר
    'gameId', 'period', 'clock', 'actionNumber', 
    
    # 2. מהות האירוע
    'actionType', 'subType', 'description', 'qualifiers',
    
    # 3. מבצעים
    'playerName', 'personId', 'teamId', 
    
    # 4. תוצאה
    'scoreHome', 'scoreAway',
    
    # 5. הקשר והרכבים
    'personIdsFilter', 'x', 'y', 'shotDistance',
    
    # 6. מונים
    'foulPersonalTotal', 'turnoverTotal', 'pointsTotal'
]

def check_completeness():
    print(f"--- Starting Data Completeness Check ---")
    print(f"Searching for CSV files in: {os.path.abspath(DATA_DIR)}")
    
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    
    if not csv_files:
        print("❌ No CSV files found! Check the path.")
        return

    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        print(f"\n==========================================")
        print(f"📂 Analyzing file: {file_name}")
        print(f"==========================================")
        
        try:
            df = pd.read_csv(file_path, low_memory=False)
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            continue
            
        total_rows = len(df)
        print(f"Total Rows: {total_rows}")
        
        # טבלה יפה של תוצאות
        print(f"{'Column Name':<25} | {'Missing':<10} | {'% Missing':<10} | {'Status'}")
        print("-" * 65)
        
        for col in COLUMNS_TO_CHECK:
            if col not in df.columns:
                print(f"{col:<25} | {'MISSING COL':<10} | {'100%':<10} | ❌ NOT FOUND")
                continue
            
            # ספירת ערכים חסרים (NaN/Null)
            missing_count = df[col].isna().sum()
            missing_pct = (missing_count / total_rows) * 100
            
            # קביעת סטטוס ויזואלי
            if missing_pct == 0:
                status = "✅ Perfect"
            elif missing_pct < 5:
                status = "⚠️ OK (Low)"
            elif missing_pct < 20:
                status = "Rx Warning"
            else:
                status = "❌ Critical"
            
            # במקרים מסוימים חוסר הוא תקין (למשל x,y לא קיימים באיבודי כדור וכו')
            # אבל הסקריפט הזה הוא "טיפש" - הוא רק מראה את המצב.
            
            print(f"{col:<25} | {missing_count:<10} | {missing_pct:>6.2f}%    | {status}")

    print("\n--- Check Complete ---")

if __name__ == "__main__":
    check_completeness()