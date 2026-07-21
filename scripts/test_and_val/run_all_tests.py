import os
import sys
import subprocess
import glob

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# List of test/validation scripts to run automatically in the pipeline
TEST_SCRIPTS = [
    "Data_integrity_check_before_FE.py",
    "check_data_health.py",
    "data_validation.py",
    "validate_game_logic.py",
    "check_contextual_sparsity.py",
    "check_usage_test.py",
    "test_data.py",
    "test_for_subs.py"
]

def run_all_tests():
    print("="*60)
    print("🧪 RUNNING PRE-FEATURE ENGINEERING QA & VALIDATION SUITE")
    print("="*60)
    
    failed_tests = []
    passed_tests = []
    
    for script_name in TEST_SCRIPTS:
        script_path = os.path.join(CURRENT_DIR, script_name)
        if not os.path.exists(script_path):
            print(f"⚠️ Warning: Test script not found: {script_name}")
            continue
            
        print(f"\n▶️ Running Test: {script_name}...")
        result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ PASSED: {script_name}")
            passed_tests.append(script_name)
        else:
            print(f"❌ FAILED: {script_name}")
            print(f"   Error Details:\n{result.stderr or result.stdout}")
            failed_tests.append((script_name, result.stderr or result.stdout))

    print("\n" + "="*60)
    print(f"📊 QA SUITE SUMMARY: {len(passed_tests)} Passed | {len(failed_tests)} Failed")
    print("="*60)
    
    if failed_tests:
        print("\n🚨 CRITICAL: The following QA tests failed:")
        for name, err in failed_tests:
            print(f"  - {name}")
        sys.exit(1)
    else:
        print("✨ All QA & Validation tests passed successfully!")

if __name__ == "__main__":
    run_all_tests()
