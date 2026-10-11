import json

json_file_path = "data/demo/demo_data.json"

with open(json_file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

game_1 = data.get("game_1", [])
home_team = "IND"
away_team = "TOR"

print(f"Total possessions in game_1: {len(game_1)}")
print(f"Home: {home_team} (Indiana Pacers), Away: {away_team} (Toronto Raptors)")
print("-" * 80)

for idx, row in enumerate(game_1):
    timeout_called = row.get("timeout_team", "NONE") != "NONE"
    target_stop = row.get("target_stop_run_90s", 0) == 1
    
    if timeout_called or target_stop:
        print(f"=== Event at Row {idx} ===")
        print(f"Type: {'TIMEOUT CALLED' if timeout_called else ''} {'TARGET STOP RUN' if target_stop else ''}")
        print(f"Period: {row.get('period')}")
        print(f"Timeout called by: {row.get('timeout_team')}")
        print(f"Play Description: {row.get('play_description')}")
        print("\n--- Play History (Last 5 possessions) ---")
        
        start_history = max(0, idx - 5)
        for h_idx in range(start_history, idx + 1):
            h_row = game_1[h_idx]
            prefix = ">>> CURRENT" if h_idx == idx else f"    [{h_idx}]"
            print(f"{prefix} Score: {h_row.get('home_score')}({home_team}) : {h_row.get('away_score')}({away_team}) | Margin: {h_row.get('score_margin')} | Play: {h_row.get('play_description')}")
            
        print("=" * 80)
