# pipeline_constants.py
# Centralized configuration for NBA Causal Inference Feature Engineering and Leakage Prevention.

# --- 1. Modular Category Definitions ---

# Technical / Event Identifiers (Pure metadata, zero statistical meaning)
TECHNICAL_METADATA_COLS = [
    'gameId', 'personId', 'actionNumber', 'orderNumber', 'possession', 
    'possession_id', 'officialId', 'jumpBallRecoverdPersonId', 
    'jumpBallWonPersonId', 'jumpBallLostPersonId', 'foulDrawnPersonId'
]

# Dead-Ball & Direct Event Artifacts (Leads to artificial 0.97+ AUC without representing macro game state)
DEAD_BALL_LEAKAGE_COLS = [
    'is_foul', 'isFieldGoal', 'is_poss_change', 'event_momentum_val', 
    'shotDistance', 'x', 'y', 'xLegacy', 'yLegacy', 'play_duration'
]

# Raw Absolute Accumulators (Superseded by normalized differentials and rolling metrics)
RAW_TOTALS_COLS = [
    'scoreHome', 'scoreAway', 'pointsTotal', 'cum_pointsTotal',
    'reboundTotal', 'reboundDefensiveTotal', 'reboundOffensiveTotal', 
    'cum_reboundDefensiveTotal', 'turnoverTotal', 'cum_turnoverTotal', 
    'foulPersonalTotal'
]

# --- 2. Centralized Blacklists by Experiment ---

FEATURE_BLACKLISTS = {
    # Baseline: Drop only technical IDs and direct event leakage, preserving tactical clock & game state confounders
    "v1_standard_causal": (
        TECHNICAL_METADATA_COLS + 
        DEAD_BALL_LEAKAGE_COLS + 
        RAW_TOTALS_COLS
    ),
    
    # Aggressive: Additional removal of high-variance short-term dynamics
    "v2_conservative_tactical": (
        TECHNICAL_METADATA_COLS + 
        DEAD_BALL_LEAKAGE_COLS + 
        RAW_TOTALS_COLS + 
        ['instability_index', 'style_tempo_rolling']
    )
}

# Active experiment configuration to be consumed by splits and models
CURRENT_EXPERIMENT = "v1_standard_causal"

def get_blacklisted_features(experiment_name: str = None) -> list:
    """Returns the list of features to drop for the specified or active experiment."""
    exp = experiment_name or CURRENT_EXPERIMENT
    return list(set(FEATURE_BLACKLISTS.get(exp, [])))
