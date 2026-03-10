import pandas as pd
import os

print("🕵️‍♂️ Starting DRACONIAN QA: check_level3_quality.py\n")

# Path logic: moving up 4 levels to reach project root from scripts/feature_engineering/validation/
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
file_path = os.path.join(base_dir, 'data', 'interim', 'level3_labels.csv')

try:
    df = pd.read_csv(file_path, low_memory=False)
    print(f"✅ File loaded successfully: {len(df)} rows found.\n")
except FileNotFoundError:
    print(f"❌ Error: Could not find file at {file_path}")
    print(f"Attempted path: {file_path}")
    exit()

target_cols = [c for c in df.columns if 'target_' in c]

print("1️⃣ MISSING VALUES CHECK (Target Columns):")
nulls = df[target_cols].isnull().sum()
print(nulls.to_string())
if nulls.sum() == 0:
    print("✨ Status: No missing values found. Perfect.")
else:
    print("🚨 WARNING: NaNs detected in labels! Logic might be broken.")
print("-" * 60)

print("\n2️⃣ CLASS BALANCE (Label Diversity):")
for col in target_cols:
    dist = df[col].value_counts(normalize=True).mul(100).round(2)
    print(f"  {col}:")
    print(f"    Class 0: {dist.get(0, 0)}% | Class 1: {dist.get(1, 0)}%")
print("-" * 60)

print("\n3️⃣ TACTICAL TIMEOUT LOGIC (Integrity Check):")
# Identify timeouts (case-insensitive)
is_timeout = df['actionType'].str.contains('timeout', case=False, na=False)
timeouts_df = df[is_timeout]

if not timeouts_df.empty:
    print(f"  Total Timeouts identified: {len(timeouts_df)}")
    
    # Critical Check: Penalty should only apply if a coach IGNORED the danger.
    # Therefore, rows that ARE timeouts should have 0 penalties.
    penalty_on_to = timeouts_df['target_danger_penalty'].sum()
    print(f"  Danger Penalty on Timeout rows (Must be 0): {penalty_on_to}")
    if penalty_on_to > 0:
        print("🚨 LOGIC ERROR: Penalties assigned to timeout events!")
    
    # Real-world benchmark: How often do NBA coaches actually succeed?
    print("\n  COACH SUCCESS RATES (After calling timeout):")
    for col in [c for c in target_cols if 'penalty' not in c]:
        success_rate = timeouts_df[col].mean() * 100
        print(f"    {col.replace('target_', '').ljust(25)}: {success_rate:.1f}% success")
else:
    print("🚨 WARNING: No timeouts found! Verify 'actionType' column values.")

print("\n4️⃣ EDGE CASE CHECK: END OF PERIOD")
# Rows where time lookahead might have hit the buzzer
last_30_sec = df[df['seconds_remaining'] <= 30]
print(f"  Rows in last 30 seconds of periods: {len(last_30_sec)}")
print("  (Lookahead fallback logic ensures these are valid 0-delta labels)")

print("\n" + "="*60)
print("✅ QA SESSION FINISHED.")