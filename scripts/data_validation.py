import pandas as pd
import matplotlib.pyplot as plt
import os

# --- הגדרות ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, '..', 'data', 'pureData')
OUTPUT_DIR = os.path.join(CURRENT_DIR, '..', 'docs', 'figures')
INPUT_FILENAME = "season_2024_25.csv"
INPUT_FILE = os.path.join(DATA_DIR, INPUT_FILENAME)

# --- הגדרות בנצ'מארק (ממוצעים למשחק ב-NBA) ---
# המספרים מייצגים סכום לשתי הקבוצות יחד
NBA_BENCHMARKS = {
    'turnover': {'expected': 27, 'tolerance': 5},  # ~13.5 per team
    'steal':    {'expected': 15, 'tolerance': 4},  # ~7.5 per team
    'block':    {'expected': 10, 'tolerance': 3},  # ~5 per team
    'foul':     {'expected': 38, 'tolerance': 6},  # ~19 per team
    '3pt':      {'expected': 70, 'tolerance': 15}, # 35 attempts per team (made+missed)
    'timeout':  {'expected': 14, 'tolerance': 4}   # ~7 per team
}

def generate_context_report(df):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. חישוב סטטיסטיקות מהדאטא
    num_games = df['gameId'].nunique()
    print(f"Analyzing {num_games} games...")

    # ספירת אירועים לכל משחק כדי לקבל ממוצע
    event_counts = df.groupby(['gameId', 'actionType']).size().unstack(fill_value=0)
    
    # הכנת נתונים לטבלה
    table_data = []
    # כותרות
    table_data.append(["Event Type", "Your Data (Avg)", "NBA Benchmark", "Status"])
    
    for event, bench in NBA_BENCHMARKS.items():
        if event in event_counts.columns:
            actual_avg = event_counts[event].mean()
            diff = abs(actual_avg - bench['expected'])
            
            # קביעת סטטוס
            if diff <= bench['tolerance']:
                status = "OK (Healthy)"
            elif actual_avg < bench['expected']:
                status = "LOW (Missing Data?)"
            else:
                status = "HIGH (Duplicates?)"
                
            row = [
                event.title(), 
                f"{actual_avg:.1f}", 
                f"{bench['expected']} (+/-{bench['tolerance']})", 
                status
            ]
            table_data.append(row)
        else:
            table_data.append([event.title(), "0.0", str(bench['expected']), "MISSING"])

    # 2. הוספת בדיקת Nulls חכמה (לשדות קריטיים בלבד)
    table_data.append(["", "", "", ""]) # רווח
    table_data.append(["Field Name", "Missing %", "Expected Missing", "Integrity"])
    
    # שדות שנרצה לבדוק
    fields_check = [
        ('stealPlayerName', 0.97, 0.02), # צפוי להיות חסר ב-97% מהמקרים (כי רק 3% זה חטיפות)
        ('blockPlayerName', 0.98, 0.02),
        ('shotDistance',    0.68, 0.05), # צפוי להיות חסר ברוב האירועים שאינם זריקות
        ('gameId',          0.00, 0.00)  # אסור שיהיה חסר
    ]
    
    total_rows = len(df)
    
    for col, expected_missing_pct, tolerance in fields_check:
        if col in df.columns:
            missing_pct = df[col].isnull().sum() / total_rows
            diff = abs(missing_pct - expected_missing_pct)
            
            if diff <= tolerance:
                status = "Valid"
            else:
                status = "Suspicious"
                
            table_data.append([
                col, 
                f"{missing_pct:.1%}", 
                f"~{expected_missing_pct:.0%}", 
                status
            ])

    # 3. ציור הטבלה
    fig, ax = plt.subplots(figsize=(10, len(table_data) * 0.6))
    ax.axis('tight')
    ax.axis('off')
    
    # בחירת צבעים לשורות הסטטוס (אופציונלי למתקדמים, כרגע נשאיר נקי)
    table = ax.table(cellText=table_data, loc='center', cellLoc='left')
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # עיצוב שורות כותרת (מודגשות)
    for (i, j), cell in table.get_celld().items():
        if i == 0 or i == len(NBA_BENCHMARKS) + 2: # שורות כותרת
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#e6e6e6')

    # שמירה
    output_path = os.path.join(OUTPUT_DIR, "0_data_health_context.png")
    plt.title(f"Data Health & Integrity Report - {INPUT_FILENAME}", y=0.99)
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"[V] Contextual health report saved to: {output_path}")

def main():
    if not os.path.exists(INPUT_FILE):
        print("File not found.")
        return
    
    print("Loading data...")
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    generate_context_report(df)

if __name__ == "__main__":
    main()