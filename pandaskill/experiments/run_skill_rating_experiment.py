
from pandaskill.experiments.general.metrics import *
from pandaskill.experiments.general.utils import *
from pandaskill.experiments.skill_rating.evaluation import evaluate_skill_ratings
from pandaskill.experiments.skill_rating.ranking import create_rankings, evaluate_ranking
from pandaskill.experiments.skill_rating.visualization import visualize_ratings
from pandaskill.libs.skill_rating.ewma import compute_ewma_ratings
from pandaskill.libs.skill_rating.bayesian import compute_bayesian_ratings
import os 
from os.path import join
import pandas as pd
import logging

def compute_skill_ratings(data: pd.DataFrame, func: callable, parameters: dict) -> pd.DataFrame:
    data_with_ratings = func(data, **parameters)
    return data_with_ratings

def save_ratings(skill_ratings: pd.DataFrame, skill_rating_experiment_dir: str) -> None:
    skill_ratings.to_csv(join(skill_rating_experiment_dir, f"skill_ratings.csv"))

def get_method_from_method_name(method_name: str) -> callable:
    if method_name == "bayesian":
        func = compute_bayesian_ratings
    elif method_name == "ewma":
        func = compute_ewma_ratings
    else:
        raise ValueError(f"Method `{method_name}` not supported")
    return func

openskill_config = {
    "name": "bayesian",
    "parameters": {
        "rater_model": "openskill",
        "use_ffa_setting": False,
        "use_meta_ratings": False,
    },
}
ffa_openskill_config = {
    "name": "bayesian",
    "parameters": {
        "rater_model": "openskill",
        "use_ffa_setting": True,
        "use_meta_ratings": False,
    },
}
meta_openskill_config = {
    "name": "bayesian",
    "parameters": {
        "rater_model": "openskill",
        "use_ffa_setting": False,
        "use_meta_ratings": True,
    },
}
meta_ffa_openskill_config = {
    "name": "bayesian",
    "parameters": {
        "rater_model": "openskill",
        "use_ffa_setting": True,
        "use_meta_ratings": True,
    },
}
meta_ffa_trueskill_config = {
    "name": "bayesian",
    "parameters": {
        "rater_model": "trueskill",
        "use_ffa_setting": True,
        "use_meta_ratings": True,
    },
}
ewma_config = {
    "name": "ewma",
    "parameters": {
        "alpha": 0.05
    }
}

if __name__ == "__main__":    
    config = {
        "experiment": "meta_ffa_openskill",
        "performance_score_experiment": "pscore",
        "method": meta_ffa_openskill_config,
        "evaluation": {
            "start_warmup_date": "2019-09-15",
            "end_warmup_date": "2020-09-15",
            "C": 1.0
        },
        "visualization": {
            "min_nb_games": 20,
            "since": "2023-09-15"
        },
        "ranking":{
            "min_nb_games": 10,
            "since": "2024-03-15"
        }
    }
    logging.info(f"Starting skill rating experiment `{config['experiment']}`")
    
    experiment_dir = join(
        ARTIFACTS_DIR, "experiments", config["performance_score_experiment"], 
        "skill_rating", config["experiment"]
    )
    os.makedirs(experiment_dir, exist_ok=True)    
    save_yaml(config, experiment_dir, "config.yaml")
    
    logging.info(f"Loading data from `{experiment_dir}`")
    data = load_data(
        load_features=True, 
        performance_score_path=join(ARTIFACTS_DIR, "experiments", config["performance_score_experiment"], "performance_score", "performance_scores.csv"),
        drop_na=True
    )

    logging.info(f"Computing skill ratings using method `{config["method"]["name"]}`")
    method = get_method_from_method_name(config["method"]["name"])
    skill_ratings = compute_skill_ratings(data, method, config["method"]["parameters"])
    save_ratings(skill_ratings, experiment_dir)

    logging.info(f"Evaluating skill ratings")
    data_with_ratings = data.join(skill_ratings)
    evaluate_skill_ratings(data_with_ratings, experiment_dir, config["evaluation"])
    visualize_ratings(data_with_ratings, experiment_dir, config["visualization"])
    
    logging.info(f"Creating and evaluating player rankings")
    ranking = create_rankings(data_with_ratings, experiment_dir, config["ranking"])
    evaluate_ranking(ranking, experiment_dir)

    logging.info(f"skill rating experiment `{config['experiment']}` finished")