
from pandaskill.experiments.general.metrics import *
from pandaskill.experiments.general.utils import *
from pandaskill.experiments.skill_rating.evaluation import evaluate_player_ratings
from pandaskill.experiments.skill_rating.ranking import create_rankings, evaluate_ranking
from pandaskill.experiments.skill_rating.visualization import visualize_ratings
from pandaskill.libs.skill_rating.ewma import compute_ewma_ratings
from pandaskill.libs.skill_rating.bayesian import compute_bayesian_ratings
import os 
from os.path import join
import pandas as pd
import logging

def load_data(performance_score_experiment: str) -> pd.DataFrame:
    performance_scores_path = join(ARTIFACTS_DIR, "experiments", performance_score_experiment, "performance_score", "performance_scores.csv")
    
    raw_data_folder = join(ARTIFACTS_DIR, "data", "raw")
    game_metadata_df = pd.read_csv(join(raw_data_folder, "game_metadata.csv"), index_col=0)
    game_players_stats_df = pd.read_csv(join(raw_data_folder, "game_players_stats.csv"), index_col=(0,1))
    game_features_df = pd.read_csv(join(ARTIFACTS_DIR, "data", "preprocessing", "game_features.csv"), index_col=(0,1))

    data = game_players_stats_df.join(game_metadata_df, on="game_id", how="left")
    data = pd.concat([data, game_features_df], axis=1)
    
    performance_scores_df = pd.read_csv(performance_scores_path, index_col=(0,1))
    data = pd.concat([data, performance_scores_df], axis=1)

    data = data.sort_values(by=["date", "win"], ascending=[True, False])
    data = data.dropna()
    
    return data

def compute_player_ratings(data: pd.DataFrame, func: callable, parameters: dict) -> pd.DataFrame:
    data_with_ratings = func(data, **parameters)
    return data_with_ratings

def save_ratings(player_ratings: pd.DataFrame, player_rating_experiment_dir: str) -> None:
    player_ratings.to_csv(join(player_rating_experiment_dir, f"player_ratings.csv"))

def prepare_ratings_data_for_evaluation(
    data: pd.DataFrame, player_ratings:pd.DataFrame, method_name: str
) -> pd.DataFrame:
    if method_name in ["bayesian"]:
        player_ratings.drop(columns=["region"], inplace=True)
        player_ratings["contextual_rating_before"] = player_ratings["contextual_rating_before"].apply(lambda x: x["lower_bound"])
        player_ratings["contextual_rating_after_mu"] = player_ratings["contextual_rating_after"].apply(lambda x: x["mu"])
        player_ratings["contextual_rating_after_sigma"] = player_ratings["contextual_rating_after"].apply(lambda x: x["sigma"])
        player_ratings["contextual_rating_after"] = player_ratings["contextual_rating_after"].apply(lambda x: x["lower_bound"])
        
        player_ratings["meta_rating_before"] = player_ratings["meta_rating_before"].apply(lambda x: x["lower_bound"])
        player_ratings["meta_rating_after_mu"] = player_ratings["meta_rating_after"].apply(lambda x: x["mu"])
        player_ratings["meta_rating_after_sigma"] = player_ratings["meta_rating_after"].apply(lambda x: x["sigma"])
        player_ratings["meta_rating_after"] = player_ratings["meta_rating_after"].apply(lambda x: x["lower_bound"])

        player_ratings["player_rating_before"] = player_ratings["player_rating_before"].apply(lambda x: x["lower_bound"])        
        player_ratings["player_rating_after_mu"] = player_ratings["player_rating_after"].apply(lambda x: x["mu"])
        player_ratings["player_rating_after_sigma"] = player_ratings["player_rating_after"].apply(lambda x: x["sigma"])
        player_ratings["player_rating_after"] = player_ratings["player_rating_after"].apply(lambda x: x["lower_bound"])

    data_with_ratings = data.join(player_ratings)

    return data_with_ratings

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
        "experiment": "meta_ffa_openskill2",
        "performance_score_experiment": "playerank_test",
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
    logging.info(f"Starting player rating experiment `{config['experiment']}`")
    
    experiment_dir = join(
        ARTIFACTS_DIR, "experiments", config["performance_score_experiment"], 
        "player_rating", config["experiment"]
    )
    os.makedirs(experiment_dir, exist_ok=True)    
    save_yaml(config, experiment_dir, "config.yaml")
    
    logging.info(f"Loading data from `{experiment_dir}`")
    data = load_data(config["performance_score_experiment"])

    logging.info(f"Computing player ratings using method `{config["method"]["name"]}`")
    method = get_method_from_method_name(config["method"]["name"])
    player_ratings = compute_player_ratings(data, method, config["method"]["parameters"])
    save_ratings(player_ratings, experiment_dir)

    logging.info(f"Evaluating player ratings")
    data_with_ratings = prepare_ratings_data_for_evaluation(data, player_ratings, config["method"]["name"])
    evaluate_player_ratings(data_with_ratings, experiment_dir, config["evaluation"])
    visualize_ratings(data_with_ratings, experiment_dir, config["visualization"])
    
    logging.info(f"Creating and evaluating player rankings")
    ranking = create_rankings(data_with_ratings, experiment_dir, config["ranking"])
    evaluate_ranking(ranking, experiment_dir)

    logging.info(f"Player rating experiment `{config['experiment']}` finished")