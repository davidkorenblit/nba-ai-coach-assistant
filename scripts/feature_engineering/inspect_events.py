import pandas as pd
import os

FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'interim', 'level1_base.csv')

def inspect_events():
    df = pd.read_csv(FILE_PATH, low_memory=False)
    
    print("🔍 Event Types Analysis:")
    
    # נבדוק אילו עמודות רלוונטיות קיימות
    cols = ['actionType', 'subType', 'eventType', 'shotResult'] # מנחשים שמות נפוצים
    existing_cols = [c for c in cols if c in df.columns]
    
    # מדפיסים דוגמאות ייחודיות כדי להבין איך נראה "סל"
    print(df[existing_cols].drop_duplicates().head(20))
    
    # בדיקה ספציפית: איך נראית שורה של שינוי ניקוד?
    print("\n🏀 Scoring Events Example:")
    # שורות שבהן הניקוד השתנה
    scoring = df[df['scoreHome'].diff() != 0].head(5)
    print(scoring[existing_cols])

if __name__ == "__main__":
    inspect_events()