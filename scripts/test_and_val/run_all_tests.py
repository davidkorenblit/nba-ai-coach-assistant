import os
import sys
import subprocess
import argparse
from datetime import datetime

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
REPORTS_DIR = os.path.join(PROJECT_ROOT, 'reports', 'qa_reports')

ALL_SEASONS = ['2021_22', '2022_23', '2023_24', '2024_25']

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

def normalize_season(season_str: str):
    """
    Normalizes season input like '2024-25' or '2024_25' to both:
    (season_underscore, season_dash), e.g. ('2024_25', '2024-25').
    """
    s = season_str.strip().replace('-', '_')
    if '_' in s:
        parts = s.split('_')
        return f"{parts[0]}_{parts[1]}", f"{parts[0]}-{parts[1]}"
    return s, s.replace('_', '-')

def run_tests_for_season(season: str, reports_dir: str = REPORTS_DIR):
    season_underscore, season_dash = normalize_season(season)
    os.makedirs(reports_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"qa_report_{season_underscore}_{timestamp}.txt"
    report_path = os.path.join(reports_dir, report_filename)

    print("=" * 65, flush=True)
    print(f"🧪 RUNNING PRE-FE QA & VALIDATION SUITE - SEASON: {season_underscore}", flush=True)
    print("=" * 65, flush=True)

    passed_tests = []
    failed_tests = []

    # Initialize report file
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("PRE-FEATURE ENGINEERING QA & VALIDATION REPORT\n")
        f.write(f"Season:    {season_underscore} ({season_dash})\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Python:    {sys.executable}\n")
        f.write("=" * 70 + "\n\n")

    for script_name in TEST_SCRIPTS:
        script_path = os.path.join(CURRENT_DIR, script_name)
        if not os.path.exists(script_path):
            warning_msg = f"⚠️ Warning: Test script not found: {script_name}"
            print(warning_msg, flush=True)
            with open(report_path, "a", encoding="utf-8") as f:
                f.write(warning_msg + "\n\n")
            continue

        print(f"\n▶️ Running Test: {script_name} (Season: {season_underscore})...", flush=True)
        cmd = [sys.executable, script_path, "--season", season_underscore]
        result = subprocess.run(cmd, capture_output=True, text=True)

        is_passed = (result.returncode == 0)
        status_label = "PASSED" if is_passed else "FAILED"

        # Append test details to report
        with open(report_path, "a", encoding="utf-8") as f:
            f.write("-" * 70 + "\n")
            f.write(f"TEST: {script_name}\n")
            f.write(f"COMMAND: {' '.join(cmd)}\n")
            f.write(f"STATUS: {status_label} (Exit Code: {result.returncode})\n")
            f.write("-" * 70 + "\n")
            f.write("[STDOUT]:\n")
            f.write((result.stdout.strip() if result.stdout.strip() else "<empty>") + "\n\n")
            f.write("[STDERR]:\n")
            f.write((result.stderr.strip() if result.stderr.strip() else "<empty>") + "\n")
            f.write("=" * 70 + "\n\n")

        if is_passed:
            print(f"✅ PASSED: {script_name}", flush=True)
            passed_tests.append(script_name)
        else:
            print(f"❌ FAILED: {script_name}", flush=True)
            err_output = result.stderr.strip() or result.stdout.strip()
            print(f"   Error Details:\n{err_output}", flush=True)
            failed_tests.append((script_name, err_output))

    # Summary
    summary_banner = [
        "=" * 70,
        f"📊 QA SUITE SUMMARY FOR {season_underscore}: {len(passed_tests)} Passed | {len(failed_tests)} Failed",
        "=" * 70
    ]
    with open(report_path, "a", encoding="utf-8") as f:
        for line in summary_banner:
            print(line, flush=True)
            f.write(line + "\n")

        if failed_tests:
            fail_header = "\n🚨 Failed Tests:"
            print(fail_header, flush=True)
            f.write(fail_header + "\n")
            for name, err in failed_tests:
                fail_item = f"  - {name}"
                print(fail_item, flush=True)
                f.write(fail_item + "\n")
        else:
            success_msg = f"\n✨ All QA & Validation tests passed successfully for Season {season_underscore}!"
            print(success_msg, flush=True)
            f.write(success_msg + "\n")

    print(f"\n📄 Full QA Report saved to: {report_path}\n", flush=True)

    return passed_tests, failed_tests, report_path

def parse_args():
    parser = argparse.ArgumentParser(description="Run Pre-FE QA & Validation Suite across seasons.")
    parser.add_argument("--season", type=str, default="2024_25",
                        help="Specific season to run (e.g. 2021_22, 2022_23, 2023_24, 2024_25). Default: 2024_25")
    parser.add_argument("--all", action="store_true",
                        help="Run QA suite on all supported seasons: 2021_22, 2022_23, 2023_24, 2024_25")
    parser.add_argument("--reports-dir", type=str, default=REPORTS_DIR,
                        help="Directory to save QA execution reports.")
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.all:
        seasons_to_run = ALL_SEASONS
    else:
        season_underscore, _ = normalize_season(args.season)
        seasons_to_run = [season_underscore]

    total_passed = 0
    total_failed = 0
    all_season_failures = {}

    for season in seasons_to_run:
        passed, failed, rpath = run_tests_for_season(season, reports_dir=args.reports_dir)
        total_passed += len(passed)
        total_failed += len(failed)
        if failed:
            all_season_failures[season] = failed

    if len(seasons_to_run) > 1:
        print("\n" + "#" * 65, flush=True)
        print(f"🏁 GLOBAL MULTI-SEASON QA SUMMARY: {total_passed} Passed | {total_failed} Failed across {len(seasons_to_run)} seasons", flush=True)
        print("#" * 65, flush=True)
        for season, failed in all_season_failures.items():
            print(f"  ❌ Season {season}: {len(failed)} failures: {', '.join([f[0] for f in failed])}", flush=True)

    if total_failed > 0:
        sys.exit(1)
    else:
        print("🎉 Multi-Season QA run complete with zero failures!", flush=True)
        sys.exit(0)

if __name__ == "__main__":
    main()
