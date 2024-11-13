from pandaskill.experiments.performance_score.visualization import visualize_performance_scores
from pandaskill.experiments.performance_score.training_testing_cv import compute_performance_scores_cv_loop
from pandaskill.experiments.general.utils import ARTIFACTS_DIR, load_data
from pandaskill.libs.performance_score.playerank_model import PlayerankModel
from pandaskill.libs.performance_score.pscore_model import PScoreModel
from pandaskill.libs.performance_score.perf_index_model import PerformanceIndexModel
import logging
import os
from os.path import join
import pandas as pd
import yaml

def _get_model_class_from_name(model_name: str) -> callable:
    if model_name == "pscore":
        return PScoreModel
    elif model_name == "playerank":
        return PlayerankModel
    elif model_name == "perf_index":
        return PerformanceIndexModel
    else:
        raise ValueError(f"Model name `{model_name}` is not supported")

playerank_model_config = {
    "name": "playerank",
    "parameters": {
        "C": 0.1, 
        "kernel": "linear",
        "max_iter": 50000,
    }
}
performance_index_model_config = {
    "name": "perf_index",
    "parameters": {
        "n_estimators": 500,
    }
}
pscore_model_config = {
    "name": "pscore",
    "parameters": {
        "n_estimators": 2000,
        "learning_rate": 0.01,
        "monotone_constraints": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, -1],
    }
}


if __name__ == "__main__":
    config = {
        "experiment_name": "pscore",
        "features": [
            "gold_per_minute",
            "cs_per_minute",
            "xp_per_minute",    
            "damage_dealt_per_total_kills",
            "damage_dealt_per_total_kills_per_gold",
            "damage_taken_per_total_kills",
            "damage_taken_per_total_kills_per_gold",
            "kla",
            "largest_multi_kill",
            "largest_killing_spree_per_total_kills",
            "wards_placed_per_minute",
            'objective_contest_loserate',
            'objective_contest_winrate',
            "free_kill_ratio",
            "worthless_death_ratio",
        ],
        "model":  pscore_model_config,
        "training": {
            "n_splits": 5,
            "random_state": 42,
            "one_model_per_role": True,
        },
        "visualization": {
            "visualize_shap_values": False, # activating this will significantly slow down the computation
            "specific_games_analysis": [
                36348, # close game - LCK 2024
            ]
        }
    }

    logging.info(f"Starting performance score experiment `{config['experiment_name']}`")
    experiment_dir = join(ARTIFACTS_DIR, "experiments", config["experiment_name"], "performance_score")
    os.makedirs(experiment_dir, exist_ok=True)
    with open(join(experiment_dir, "config.yaml"), "w") as file:
        yaml.dump(config, file, default_flow_style=False)

    logging.info(f"Loading data...")
    data = load_data(
        load_features=True,
        drop_na=True
    )

    logging.info(f"Computing performance scores...")
    performance_scores_df, metrics = compute_performance_scores_cv_loop(
        data, 
        config["features"], 
        _get_model_class_from_name(config["model"]["name"]),
        config["model"]["parameters"], 
        config["training"],
        experiment_dir,
        config["visualization"]
    )

    logging.info(f"Saving performance scores and metrics to `{experiment_dir}`")
    visualize_performance_scores(data, performance_scores_df, metrics, experiment_dir)
    performance_scores_df.to_csv(join(experiment_dir, "performance_scores.csv"))
    with open(join(experiment_dir, f"performance_scores_metrics.yaml"), "w") as file:
        yaml.dump(metrics, file, default_flow_style=False)
