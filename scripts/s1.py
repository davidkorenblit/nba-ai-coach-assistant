import pandas as pd
import os

# הגדרת נתיב: עולים למעלה לתיקיית data/interim
# ההנחה היא שהסקריפט רץ מתוך תיקיית scripts/feature_engineering או scripts
current_dir = os.path.dirname(os.path.abspath(__file__))

# ניסיון לאתר את הנתיב הנכון (תמיכה גם אם מריצים מתוך תת-תיקייה)
path = os.path.join(current_dir, '..', 'data', 'interim', 'level2_features.csv')
if not os.path.exists(path):
    # ניסיון לעלות שתי רמות למעלה (למקרה שהסקריפט בתוך feature_engineering)
    path = os.path.join(current_dir, '..', '..', 'data', 'interim', 'level2_features.csv')

print(f"📂 Looking for file at: {path}")

try:
    if not os.path.exists(path):
        print(f"❌ File not found at {path}. Check path manually.")
    else:
        # טעינת הקובץ
        df = pd.read_csv(path, low_memory=False)
        print(f"✅ Success! File loaded. Shape: {df.shape}")
        
        print("\n📋 Columns List:")
        print(df.columns.tolist())
        
        # --- בדיקה לתיקון הבאג: מהם סוגי הפעולות הקיימים? ---
        if 'actionType' in df.columns:
            print("\n🔍 Action Types Found (Top 20):")
            print(df['actionType'].value_counts().head(20))
        else:
            print("\n⚠️ Note: 'actionType' column is missing from Level 2 file.")

except Exception as e:
    print(f"❌ Error: {e}")