import pandas as pd
import os

# נתיב לקובץ המעובד (יותר מהיר מלטעון את הכל מחדש)
FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'data', 'interim', 'level1_base.csv')

def inspect_timeout_descriptions():
    print(f"🕵️‍♂️ Inspecting 'Unknown' Timeouts in: {os.path.basename(FILE_PATH)}")
    
    try:
        df = pd.read_csv(FILE_PATH, low_memory=False)
    except FileNotFoundError:
        print("❌ File not found.")
        return

    # סינון: קח רק את השורות שזיהינו כטיים-אאוט אבל לא הצלחנו לסווג
    unknown_timeouts = df[df['timeout_type'] == 'Unknown']
    
    if unknown_timeouts.empty:
        print("No 'Unknown' timeouts found. Did the previous script run correctly?")
        return

    print(f"\nFound {len(unknown_timeouts)} unclassified timeouts.")
    print("-" * 50)
    print("TOP 20 DESCRIPTIONS:")
    print("-" * 50)
    
    # הדפסת התיאורים הנפוצים ביותר כדי שנזהה תבניות
    print(unknown_timeouts['description'].value_counts().head(20))

if __name__ == "__main__":
    inspect_timeout_descriptions()