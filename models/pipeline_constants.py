# pipeline_constants.py
# Centralized configuration for NBA Causal Inference Feature Engineering and Leakage Prevention.

FEATURE_EXPERIMENTS = {
    # Baseline with everything included (Except the main target/treatment)
    "v1_all_surviving": [
        'actionNumber', 'period', 'personId', 'x', 'y', 'possession', 'scoreHome', 
        'scoreAway', 'orderNumber', 'isTargetScoreLastPeriod', 'xLegacy', 'yLegacy', 
        'isFieldGoal', 'gameId', 'shotDistance', 'reboundTotal', 'reboundDefensiveTotal', 
        'reboundOffensiveTotal', 'pointsTotal', 'turnoverTotal', 'foulPersonalTotal', 
        'seconds_remaining', 'score_margin', 'timeouts_remaining_home', 
        'timeouts_remaining_away', 'is_foul', 'team_fouls_period', 'cum_pointsTotal', 
        'cum_turnoverTotal', 'cum_reboundDefensiveTotal', 'play_duration', 'is_poss_change', 
        'possession_id', 'shot_clock_estimated', 'lineup_confidence', 'time_since_last_sub', 
        'home_usage_gravity', 'away_usage_gravity', 'usage_delta', 'home_cum_fatigue', 
        'away_cum_fatigue', 'event_momentum_val', 'momentum_streak_rolling', 
        'explosiveness_index', 'style_tempo_rolling', 'is_high_fatigue', 'instability_index', 
        'is_clutch_time', 'is_star_resting',
    ],
    
    # Aggressive mitigation based on the Frozen Time and Dead Ball logical criteria
    "v2_aggressive_clean": [
        # Technical/Event IDs (77%+ Leakage culprits)
        'possession_id', 'personId', 'gameId', 'actionNumber', 'orderNumber', 'possession',
        
        # Dead Ball Triggers & Direct Action Types (0.97 AUC Leakage culprits)
        'is_foul', 'isFieldGoal', 'is_poss_change', 'event_momentum_val', 'shotDistance',
        
        # Proxy Clocks & Absolute Stop-Clocks
        'seconds_remaining', 'shot_clock_estimated', 'play_duration', 
        'x', 'y', 'xLegacy', 'yLegacy',
        
        # Historical / Absolute Counters (Replaced by tactical margins/rolling metrics)
        'scoreHome', 'scoreAway', 'pointsTotal', 'cum_pointsTotal',
        'reboundTotal', 'reboundDefensiveTotal', 'reboundOffensiveTotal', 'cum_reboundDefensiveTotal',
        'turnoverTotal', 'cum_turnoverTotal', 'foulPersonalTotal',
         'turnoverTotal', 'cum_turnoverTotal', 'foulPersonalTotal',
        'explosiveness_index'
        
    ]
}

# The active experiment configuration to be consumed by the pipeline splits and models
CURRENT_EXPERIMENT = "v2_aggressive_clean"

def get_blacklisted_features():
    """Returns the list of features to drop for the active experiment."""
    return FEATURE_EXPERIMENTS.get(CURRENT_EXPERIMENT, [])
