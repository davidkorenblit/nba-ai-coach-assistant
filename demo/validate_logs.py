import json
import os
import re
import pandas as pd
import glob

def find_latest_log(game_name):
    log_dir = 'data/demo/logs'
    pattern = os.path.join(log_dir, f'simulation_log_{game_name}_*.txt')
    files = glob.glob(pattern)
    if not files:
        # Try fallback matching any txt files starting with simulation_log_game_X
        pattern_fallback = os.path.join(log_dir, f'simulation_log_{game_name}*')
        files = glob.glob(pattern_fallback)
        if not files:
            raise FileNotFoundError(f"No log file matching pattern {pattern} in {log_dir}")
    # Sort files by modification time, newest first
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]

def load_json_data():
    json_path = 'data/demo/demo_data.json'
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON data file not found at {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def parse_log_file(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Log file not found at {file_path}")
    
    log_entries = {}
    # Matches: [Q1 - P:2] IND 2 : TOR 0 | IND: Beautiful driving layup scored in transition!
    # Or: [Q4 - P:2] BOS 2 : MIA 0 | BOS: Beautiful driving layup scored in transition!
    pattern = re.compile(r'^\[Q(\d+) - P:(\d+)\] ([A-Z]{3}) (\d+) : ([A-Z]{3}) (\d+) \| (.*)$')
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            match = pattern.match(line)
            if match:
                period = int(match.group(1))
                poss_idx = int(match.group(2))
                home_score = int(match.group(4))
                away_score = int(match.group(6))
                desc = match.group(7)
                log_entries[poss_idx] = {
                    'period': period,
                    'possession_index': poss_idx,
                    'home_score': home_score,
                    'away_score': away_score,
                    'score_margin': home_score - away_score,
                    'play_description': desc
                }
    return log_entries

def merge_data(json_game_data, log_game_data):
    # Align JSON data and Parsed Log text into a single Pandas DataFrame
    merged_list = []
    for idx, json_row in enumerate(json_game_data):
        row = json_row.copy()
        row['possession_index'] = idx
        log_row = log_game_data.get(idx, {})
        
        # Override with log scores and description if present, to validate actual output
        if log_row:
            row['home_score'] = log_row['home_score']
            row['away_score'] = log_row['away_score']
            row['score_margin'] = log_row['score_margin']
            row['play_description_log'] = log_row['play_description']
        else:
            row['play_description_log'] = row.get('play_description', '')
            
        merged_list.append(row)
        
    return pd.DataFrame(merged_list)

def validate_game_1(df):
    results = []
    
    # ----------------------------------------------------
    # Q1: Event 1 -> Event 2 within 2 possessions.
    # Tactical Event 2 in last minute without Event 1.
    # ----------------------------------------------------
    q1_df = df[df['period'] == 1]
    
    # Check start score
    first_row = q1_df.iloc[0]
    if first_row['home_score'] == 0 and first_row['away_score'] == 0:
        results.append({'status': 'PASS', 'category': 'Narrative State Machine', 'message': 'Q1 starts strictly at 0:0.'})
    else:
        results.append({'status': 'FAIL', 'category': 'Narrative State Machine', 'message': f"Q1 started at {first_row['home_score']}:{first_row['away_score']}"})

    # Event 1 in Q1 (propensity score >= 0.85)
    e1_q1 = q1_df[q1_df['propensity_score'] >= 0.85]
    if not e1_q1.empty:
        e1_idx = e1_q1.index[0]
        results.append({'status': 'PASS', 'category': 'Narrative State Machine', 'message': f"Q1 Event 1 (Green Light) triggered at P:{e1_idx}."})
        
        # Event 2 within 2 possessions
        e2_candidates = q1_df.loc[e1_idx:e1_idx+2]
        e2_row = e2_candidates[e2_candidates['timeout_team'] == 'INDIANA']
        if not e2_row.empty:
            results.append({'status': 'PASS', 'category': 'Narrative State Machine', 'message': f"Q1 Event 2 (Actual TO) occurred within 2 possessions at P:{e2_row.index[0]}."})
        else:
            results.append({'status': 'FAIL', 'category': 'Narrative State Machine', 'message': 'Q1 Event 2 did not occur within 2 possessions of Event 1.'})
    else:
        results.append({'status': 'FAIL', 'category': 'Narrative State Machine', 'message': 'Q1 Event 1 not found.'})

    # Tactical timeout in last minute of Q1 (end of quarter spacing) without preceding Event 1
    # We define last minute of Q1 as index in the last 20% of Q1
    q1_len = len(q1_df)
    last_min_q1 = q1_df.iloc[int(q1_len * 0.80):]
    tactical_to = last_min_q1[last_min_q1['play_description_log'].str.contains('Tactical Timeout', na=False)]
    if not tactical_to.empty:
        tactical_idx = tactical_to.index[0]
        # Check preceding 5 possessions for Event 1
        preceding_5 = df.loc[max(0, tactical_idx-5):tactical_idx-1]
        has_e1 = any(preceding_5['propensity_score'] >= 0.75)
        if not has_e1:
            results.append({'status': 'PASS', 'category': 'Narrative State Machine', 'message': f"Tactical Event 2 in Q1 occurred at P:{tactical_idx} without preceding Event 1."})
        else:
            results.append({'status': 'FAIL', 'category': 'Narrative State Machine', 'message': f"Tactical Event 2 at P:{tactical_idx} had active Event 1 indicators in trailing window."})
    else:
        results.append({'status': 'FAIL', 'category': 'Narrative State Machine', 'message': 'Q1 Tactical Event 2 not found.'})

    # ----------------------------------------------------
    # Q2: Event 1 triggers -> Event 2 occurs within 2 possessions.
    # ----------------------------------------------------
    q2_df = df[df['period'] == 2]
    e1_q2 = q2_df[q2_df['propensity_score'] >= 0.85]
    if not e1_q2.empty:
        e1_idx = e1_q2.index[0]
        results.append({'status': 'PASS', 'category': 'Narrative State Machine', 'message': f"Q2 Event 1 (Propensity Alarm) triggered at P:{e1_idx}."})
        
        # Event 2 within 2 possessions
        e2_candidates = q2_df.loc[e1_idx:e1_idx+2]
        e2_row = e2_candidates[e2_candidates['timeout_team'] == 'INDIANA']
        if not e2_row.empty:
            results.append({'status': 'PASS', 'category': 'Narrative State Machine', 'message': f"Q2 Event 2 (Actual TO) occurred within 2 possessions at P:{e2_row.index[0]}."})
        else:
            results.append({'status': 'FAIL', 'category': 'Narrative State Machine', 'message': 'Q2 Event 2 did not occur within 2 possessions of Event 1.'})
    else:
        results.append({'status': 'FAIL', 'category': 'Narrative State Machine', 'message': 'Q2 Event 1 not found.'})

    # ----------------------------------------------------
    # Q3: Event 1 active for ~2 minutes (12 poss) -> Event 2 does NOT happen -> Event 3 triggers -> Immediately followed by a 6-possession opponent run ending at -18 margin.
    # ----------------------------------------------------
    q3_df = df[df['period'] == 3]
    
    # Track rolling window of 12 possessions where propensity >= 0.80 and no timeout
    q3_df = q3_df.copy()
    q3_df['e1_active'] = (q3_df['propensity_score'] >= 0.80) & (q3_df['timeout_team'] == 'NONE')
    q3_df['rolling_e1_sum'] = q3_df['e1_active'].rolling(window=12).sum()
    
    streak_rows = q3_df[q3_df['rolling_e1_sum'] == 12]
    if not streak_rows.empty:
        end_idx = streak_rows.index[0]
        start_idx = end_idx - 11
        results.append({'status': 'PASS', 'category': 'Narrative State Machine', 'message': f"Q3 Event 1 active for 12+ consecutive possessions (P:{start_idx}-{end_idx}) without Event 2."})
        
        # Verify Event 3 triggers immediately after the ignore window
        e3_idx = end_idx + 1
        if e3_idx in q3_df.index:
            e3_row = q3_df.loc[e3_idx]
            if e3_row['target_stop_run_90s'] == 1 or e3_row['cate_score'] >= 0.95:
                results.append({'status': 'PASS', 'category': 'Narrative State Machine', 'message': f"Q3 Event 3 (Red Alert) triggered immediately after ignore window at P:{e3_idx}."})
                
                # Check immediate 6-possession opponent run ending at -18 margin
                run_rows = df.loc[e3_idx:e3_idx+5]
                if len(run_rows) == 6:
                     margin_start = df.loc[e3_idx - 1]['score_margin']
                     margin_end = run_rows.iloc[-1]['score_margin']
                     # Deficit grows (meaning margin gets more negative)
                     if margin_end < margin_start and margin_end == -18:
                         results.append({'status': 'PASS', 'category': 'Narrative State Machine', 'message': f"Q3 Catastrophe verified: 6-possession opponent run ending at exactly score_margin of {margin_end}."})
                     else:
                         results.append({'status': 'FAIL', 'category': 'Narrative State Machine', 'message': f"Q3 Opponent run check failed. Margin start: {margin_start}, end: {margin_end} (expected: -18)."})
                else:
                    results.append({'status': 'FAIL', 'category': 'Narrative State Machine', 'message': 'Not enough possessions for 6-possession opponent run.'})
            else:
                results.append({'status': 'FAIL', 'category': 'Narrative State Machine', 'message': f"Q3 Event 3 did not trigger at P:{e3_idx}."})
        else:
            results.append({'status': 'FAIL', 'category': 'Narrative State Machine', 'message': 'Dataset ended immediately after ignore window.'})
    else:
        results.append({'status': 'FAIL', 'category': 'Narrative State Machine', 'message': 'Q3 ignore window of 12 consecutive possessions not found.'})

    # ----------------------------------------------------
    # Category A: Logical Alarms vs. Margin Consistency
    # ----------------------------------------------------
    alarms_ok = True
    for idx in range(5, len(df)):
        row = df.iloc[idx]
        desc = row['play_description_log']
        # Alarm check: CATE/Red alert triggers only
        is_cate_alarm = "CRITICAL ALARM" in desc or row['cate_score'] >= 0.92 or row['target_stop_run_90s'] == 1
        if is_cate_alarm:
            margin_now = row['score_margin']
            margin_past = df.iloc[idx - 5]['score_margin']
            delta = margin_now - margin_past
            # Margin delta must be negative (worse for home team)
            if delta >= 0:
                results.append({
                    'status': 'FAIL',
                    'category': 'Logical Alarms vs. Margin Consistency',
                    'message': f"Red Alert active at P:{row['possession_index']} during positive/stable momentum (Delta margin over past 5 poss: {delta} >= 0). Margin was {margin_past} -> {margin_now}."
                })
                alarms_ok = False
    if alarms_ok:
        results.append({
            'status': 'PASS',
            'category': 'Logical Alarms vs. Margin Consistency',
            'message': 'All Red Alarms (Event 3) occurred strictly during negative momentum phases.'
        })

    return results

def validate_game_2(df):
    results = []
    
    # ----------------------------------------------------
    # Q4: Boston vs Miami, trail by 4 at start, Miami run, Red Alarm, timeout, comeback, buzzer victory
    # ----------------------------------------------------
    q4_df = df[df['period'] == 4]
    if q4_df.empty:
        results.append({'status': 'FAIL', 'category': 'Narrative State Machine', 'message': 'No Q4 data found in Game 2.'})
        return results

    # 1. Start of Q4: keep margin around -4 (trail by 4)
    first_row = q4_df.iloc[0]
    if -6 <= first_row['score_margin'] <= -2:
        results.append({'status': 'PASS', 'category': 'Narrative State Machine', 'message': f"Q4 start margin is around -4 (actually {first_row['score_margin']})."})
    else:
        results.append({'status': 'FAIL', 'category': 'Narrative State Machine', 'message': f"Q4 start margin is {first_row['score_margin']} (expected around -4)."})

    # Q4 mid-quarter stable margin (possessions 10 to 25)
    mid_q4 = q4_df.iloc[10:25]
    if not mid_q4.empty:
        margin_mean = mid_q4['score_margin'].mean()
        if -6 <= margin_mean <= -2:
            results.append({'status': 'PASS', 'category': 'Narrative State Machine', 'message': f"Q4 mid-quarter stable margin is around -4 (mean: {margin_mean:.1f})."})
        else:
            results.append({'status': 'FAIL', 'category': 'Narrative State Machine', 'message': f"Q4 mid-quarter margin was not stable around -4 (mean: {margin_mean:.1f})."})
    else:
        results.append({'status': 'FAIL', 'category': 'Narrative State Machine', 'message': "Not enough possessions to verify mid-quarter stable margin."})

    # Propensity Alarm at -11 margin
    prop_alarm = q4_df[q4_df['propensity_score'] >= 0.85]
    if not prop_alarm.empty:
        p_idx = prop_alarm.index[0]
        p_margin = prop_alarm.iloc[0]['score_margin']
        if p_margin == -11:
            results.append({'status': 'PASS', 'category': 'Narrative State Machine', 'message': f"Propensity Alarm triggered at exactly -11 margin (P:{p_idx})."})
        else:
            results.append({'status': 'FAIL', 'category': 'Narrative State Machine', 'message': f"Propensity Alarm triggered at {p_margin} margin instead of -11 (P:{p_idx})."})
            
        # Critical Alarm (CATE Red Alert) at -13 margin, 1 possession later
        cate_idx = p_idx + 1
        if cate_idx in q4_df.index:
            cate_row = q4_df.loc[cate_idx]
            if (cate_row['cate_score'] >= 0.92 or cate_row['target_stop_run_90s'] == 1) and cate_row['score_margin'] == -13:
                results.append({'status': 'PASS', 'category': 'Narrative State Machine', 'message': f"Critical Alarm (CATE Red Alert) triggered 1 possession later at exactly -13 margin (P:{cate_idx})."})
            else:
                results.append({'status': 'FAIL', 'category': 'Narrative State Machine', 'message': f"Critical Alarm did not trigger as expected at P:{cate_idx} (margin: {cate_row['score_margin']}, cate_score: {cate_row['cate_score']})."})
        else:
            results.append({'status': 'FAIL', 'category': 'Narrative State Machine', 'message': "Dataset ended immediately after Propensity Alarm."})
            
        # Strategic timeout by Boston at -15 margin, 1 possession after Critical Alarm
        to_idx = p_idx + 2
        if to_idx in q4_df.index:
            to_row = q4_df.loc[to_idx]
            if to_row['timeout_team'] == 'BOSTON' and to_row['timeout_strategic_weight'] == 1 and to_row['score_margin'] == -15:
                results.append({'status': 'PASS', 'category': 'Narrative State Machine', 'message': f"Boston called a strategic timeout 1 possession after Critical Alarm at exactly -15 margin (P:{to_idx})."})
            else:
                results.append({'status': 'FAIL', 'category': 'Narrative State Machine', 'message': f"Boston strategic timeout check failed at P:{to_idx} (team: {to_row['timeout_team']}, margin: {to_row['score_margin']})."})
        else:
            results.append({'status': 'FAIL', 'category': 'Narrative State Machine', 'message': "Dataset ended too early to verify Boston strategic timeout."})
            
        # Boston fast comeback run reducing deficit to -10 within 5 possessions
        cb_idx = to_idx + 5
        if cb_idx in q4_df.index:
            cb_row = q4_df.loc[cb_idx]
            if cb_row['score_margin'] == -10:
                results.append({'status': 'PASS', 'category': 'Narrative State Machine', 'message': f"Boston comeback run successfully reduced deficit to -10 at P:{cb_idx}."})
            else:
                results.append({'status': 'FAIL', 'category': 'Narrative State Machine', 'message': f"Boston comeback run failed to reach exactly -10 deficit at P:{cb_idx} (margin: {cb_row['score_margin']})."})
        else:
            results.append({'status': 'FAIL', 'category': 'Narrative State Machine', 'message': "Dataset ended too early to verify Boston comeback run."})
            
        # Miami immediately calls timeout at possession to_idx + 6 to stop the run
        mia_to_idx = to_idx + 6
        if mia_to_idx in q4_df.index:
            mia_row = q4_df.loc[mia_to_idx]
            if mia_row['timeout_team'] == 'MIAMI':
                results.append({'status': 'PASS', 'category': 'Narrative State Machine', 'message': f"Miami immediately called a timeout to stop the run at P:{mia_to_idx}."})
            else:
                results.append({'status': 'FAIL', 'category': 'Narrative State Machine', 'message': f"Miami timeout check failed at P:{mia_to_idx} (team: {mia_row['timeout_team']})."})
        else:
            results.append({'status': 'FAIL', 'category': 'Narrative State Machine', 'message': "Dataset ended too early to verify Miami timeout."})
            
        # Score stabilizes around -5
        stable_start = to_idx + 7
        stable_end = len(q4_df) - 15
        if stable_start < stable_end:
            stable_slice = q4_df.iloc[stable_start:stable_end]
            stable_mean = stable_slice['score_margin'].mean()
            if -7 <= stable_mean <= -3:
                results.append({'status': 'PASS', 'category': 'Narrative State Machine', 'message': f"Score stabilized around -5 post Miami timeout (mean: {stable_mean:.1f})."})
            else:
                results.append({'status': 'FAIL', 'category': 'Narrative State Machine', 'message': f"Score did not stabilize around -5 post Miami timeout (mean: {stable_mean:.1f})."})
    else:
        results.append({'status': 'FAIL', 'category': 'Narrative State Machine', 'message': "Propensity Alarm (Event 1) not found in Q4."})

    # Tactical timeouts in the final minute (last 15 possessions)
    last_15 = q4_df.iloc[-15:]
    bos_tac = last_15[last_15['timeout_team'] == 'BOSTON']
    mia_tac = last_15[last_15['timeout_team'] == 'MIAMI']
    if not bos_tac.empty and not mia_tac.empty:
        results.append({'status': 'PASS', 'category': 'Narrative State Machine', 'message': f"Tactical timeouts called in final minute by Boston (P:{bos_tac.index[0]}) and Miami (P:{mia_tac.index[0]})."})
    else:
        results.append({'status': 'FAIL', 'category': 'Narrative State Machine', 'message': f"Missing tactical timeouts in final minute (Boston TOs: {len(bos_tac)}, Miami TOs: {len(mia_tac)})."})

    # Final buzzer victory by Boston of exactly +2 points
    final_row = q4_df.iloc[-1]
    if final_row['score_margin'] == 2:
        results.append({'status': 'PASS', 'category': 'Narrative State Machine', 'message': f"Final buzzer check: Boston wins by exactly {final_row['score_margin']}."})
    else:
        results.append({'status': 'FAIL', 'category': 'Narrative State Machine', 'message': f"Final buzzer check failed: score margin is {final_row['score_margin']} (expected: 2)."})

    # ----------------------------------------------------
    # Logical Alarms vs. Margin Consistency
    # ----------------------------------------------------
    alarms_ok = True
    for idx in range(5, len(df)):
        row = df.iloc[idx]
        desc = row['play_description_log']
        is_cate_alarm = "CRITICAL ALARM" in desc or row['cate_score'] >= 0.92 or row['target_stop_run_90s'] == 1
        if is_cate_alarm:
            margin_now = row['score_margin']
            margin_past = df.iloc[idx - 5]['score_margin']
            delta = margin_now - margin_past
            if delta >= 0:
                results.append({
                    'status': 'FAIL',
                    'category': 'Logical Alarms vs. Margin Consistency',
                    'message': f"Red Alert active at P:{row['possession_index']} during positive/stable momentum (Delta margin: {delta} >= 0). Margin was {margin_past} -> {margin_now}."
                })
                alarms_ok = False
    if alarms_ok:
        results.append({
            'status': 'PASS',
            'category': 'Logical Alarms vs. Margin Consistency',
            'message': 'All Red Alarms (Event 3) occurred strictly during negative momentum phases.'
        })
    return results

def print_category_results(category_name, check_list):
    print(f"\n[CATEGORY: {category_name}]")
    for check in check_list:
        status = check['status']
        msg = check['message']
        if status == 'PASS':
            print(f"  \033[92m[✓ PASS]\033[0m {msg}")
        elif status == 'FAIL':
            print(f"  \033[91m[✗ FAIL]\033[0m {msg}")
        elif status == 'WARNING':
            print(f"  \033[93m[! WARN]\033[0m {msg}")
        elif status == 'SKIPPED':
            print(f"  \033[94m[- SKIP]\033[0m {msg}")

def main():
    print("=" * 60)
    print("      NBA SIMCAST DSS - STATE MACHINE QA VALIDATION SUITE    ")
    print("=" * 60)
    
    try:
        json_data = load_json_data()
    except Exception as e:
        print(f"Error loading JSON data: {e}")
        return
        
    try:
        game_1_log_path = find_latest_log('game_1')
        print(f"Found latest Game 1 log: {game_1_log_path}")
        game_1_logs = parse_log_file(game_1_log_path)
    except Exception as e:
        print(f"Error loading Game 1 logs: {e}")
        return

    try:
        game_2_log_path = find_latest_log('game_2')
        print(f"Found latest Game 2 log: {game_2_log_path}")
        game_2_logs = parse_log_file(game_2_log_path)
    except Exception as e:
        print(f"Error loading Game 2 logs: {e}")
        return
    
    # ----------------------------------------------------
    # GAME 1
    # ----------------------------------------------------
    print("\n" + "=" * 50)
    print("GAME 1: The Stubborn Coach (The Problem Scenario)")
    print("=" * 50)
    df_1 = merge_data(json_data.get('game_1', []), game_1_logs)
    g1_results = validate_game_1(df_1)
    
    logical_g1 = [r for r in g1_results if r['category'] == 'Logical Alarms vs. Margin Consistency']
    timeline_g1 = [r for r in g1_results if r['category'] == 'Narrative State Machine']
    
    print_category_results("Logical Alarms vs. Margin Consistency", logical_g1)
    print_category_results("Narrative State Machine Transitions", timeline_g1)
    
    # ----------------------------------------------------
    # GAME 2
    # ----------------------------------------------------
    print("\n" + "=" * 50)
    print("GAME 2: The Strategic Coach (The ROI Scenario)")
    print("=" * 50)
    df_2 = merge_data(json_data.get('game_2', []), game_2_logs)
    g2_results = validate_game_2(df_2)
    
    logical_g2 = [r for r in g2_results if r['category'] == 'Logical Alarms vs. Margin Consistency']
    timeline_g2 = [r for r in g2_results if r['category'] == 'Narrative State Machine']
    
    print_category_results("Logical Alarms vs. Margin Consistency", logical_g2)
    print_category_results("Narrative State Machine Transitions", timeline_g2)
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
