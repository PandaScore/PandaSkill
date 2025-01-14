from pandaskill.experiments.general.metrics import *
from pandaskill.experiments.general.utils import ROLES, ALL_REGIONS, ARTIFACTS_DIR, save_yaml
from pandaskill.experiments.general.visualization import plot_model_calibration
from pandaskill.libs.skill_rating.bayesian import lower_bound_rating
import os
from os.path import join
import pandas as pd
import numpy as np
from scipy.stats import norm

def create_rankings(
    data_with_ratings: pd.DataFrame, 
    experiment_dir: str, 
    parameters: dict
) -> pd.DataFrame:
    saving_dir = join(experiment_dir, "rankings")
    os.makedirs(saving_dir, exist_ok=True)

    ranking = create_global_player_ranking(data_with_ratings, parameters)
    
    ranking.to_csv(join(saving_dir, "global_player_ranking.csv"), index=False)

    _save_player_ranking_per_region(ranking, saving_dir)

    _save_player_ranking_per_role(ranking, saving_dir)

    _save_region_average_ranking(ranking, saving_dir)

    _save_region_top_ranking(ranking, 10, saving_dir)

    _save_team_ranking(ranking, saving_dir)

    return ranking

def create_global_player_ranking(
    data_with_ratings: pd.DataFrame, 
    parameters: dict
) -> pd.DataFrame:
    data_with_ratings = data_with_ratings[data_with_ratings.date > parameters["since"]]

    most_played_recent_league_by_player = data_with_ratings.groupby("player_id")["league_name"].agg(
        lambda x: x.mode()[0]
    )

    ranking = data_with_ratings.groupby("player_id").last()
    nb_games_per_player = data_with_ratings.groupby("player_id").count()["date"]
    ranking["nb_games"] = nb_games_per_player
    ranking["league_name"] = most_played_recent_league_by_player

    if "meta_rating_after" in ranking.columns:
        ranking = _update_meta_ratings_with_latest_known_values(ranking)

    ranking = ranking[ranking["nb_games"] >= parameters["min_nb_games"]]
    ranking = ranking.sort_values("skill_rating_after", ascending=False)
    ranking = ranking.reset_index()  

    columns_renaming_dict = {
        "skill_rating_after": "skill_rating",
        "date": "last_game_date"
    }
    if "skill_rating_after_mu" in ranking.columns:
        columns_renaming_dict = {
            **columns_renaming_dict,
            "skill_rating_after_mu": "skill_rating_mu",
            "skill_rating_after_sigma": "skill_rating_sigma",
        }


    ranking = ranking.rename(columns=columns_renaming_dict)
    ranking["rank"] = ranking.index + 1 

    columns = [
        "rank", "player_id", "player_name", "team_name", "region", "league_name", "role", "nb_games", 
        "last_game_date", "skill_rating"]
    if "skill_rating_mu" in ranking.columns:
        columns += ["skill_rating_mu", "skill_rating_sigma"]

    ranking = ranking.loc[:, columns]

    return ranking

def _update_meta_ratings_with_latest_known_values(ranking: pd.DataFrame) -> pd.DataFrame:
    ranking = ranking.copy()
    last_meta_games = ranking.sort_values("date").groupby("region").last()
    last_meta_ratings_mu = last_meta_games["meta_rating_after_mu"].to_dict()
    last_meta_ratings_sigma = last_meta_games["meta_rating_after_sigma"].to_dict()

    for region in ALL_REGIONS:
        region_players_index = ranking[ranking["region"] == region].index
        ranking.loc[region_players_index, "meta_rating_after_mu"] = last_meta_ratings_mu[region]
        ranking.loc[region_players_index, "meta_rating_after_sigma"] = last_meta_ratings_sigma[region]

    ranking.loc[:, "meta_rating_after"] = ranking.apply(
        lambda row: lower_bound_rating(row["meta_rating_after_mu"], row["meta_rating_after_sigma"]), axis=1
    )
    ranking.loc[:, "skill_rating_after_mu"] = (
        ranking.loc[:, "meta_rating_after_mu"] 
        + ranking.loc[:, "contextual_rating_after_mu"]
    )
    ranking.loc[:, "skill_rating_after_sigma"] = np.sqrt(
        (ranking.loc[:, "contextual_rating_after_sigma"] ** 2)
        + (ranking.loc[:, "meta_rating_after_sigma"] ** 2)
    )
    ranking.loc[:, "skill_rating_after"] = ranking.apply(
        lambda row: lower_bound_rating(row["skill_rating_after_mu"], row["skill_rating_after_sigma"]), axis=1
    )
    return ranking
    

def _save_player_ranking_per_region(
    ranking: pd.DataFrame,
    saving_dir: str
) -> None:    
    region_saving_dir = join(saving_dir, "region_rankings")
    os.makedirs(region_saving_dir, exist_ok=True)
    for region in ALL_REGIONS:
        region_ranking = ranking[ranking["region"] == region]
        region_ranking = region_ranking.reset_index(drop=True)
        region_ranking["rank"] = region_ranking.index + 1
        region_ranking.to_csv(join(region_saving_dir, f"{region}_player_ranking.csv"), index=False)

def _save_player_ranking_per_role(
    ranking: pd.DataFrame,
    saving_dir: str
) -> None:     
    role_saving_dir = join(saving_dir, "role_rankings")
    os.makedirs(role_saving_dir, exist_ok=True)
    for role in ROLES:
        role_ranking = ranking[ranking["role"] == role]
        role_ranking = role_ranking.reset_index(drop=True)
        role_ranking["rank"] = role_ranking.index + 1
        role_ranking.to_csv(join(role_saving_dir, f"{role}_player_ranking.csv"), index=False)

def _save_region_average_ranking(
    ranking: pd.DataFrame, 
    saving_dir: str
) -> None:
    region_average_ranking = ranking.groupby("region")["skill_rating"].mean().reset_index()
    region_average_ranking = region_average_ranking.sort_values("skill_rating", ascending=False)
    region_average_ranking["rank"] = region_average_ranking.index + 1
    region_average_ranking = region_average_ranking.loc[:, ["rank", "region", "skill_rating"]]
    region_average_ranking.to_csv(join(saving_dir, "region_average_ranking.csv"), index=False)

def _save_region_top_ranking(
    ranking: pd.DataFrame, 
    top: int, 
    saving_dir: str
) -> None:
    region_top10_ranking = ranking.groupby("region").head(top).reset_index()
    region_top10_ranking = region_top10_ranking.groupby("region")["skill_rating"].mean().reset_index()
    region_top10_ranking = region_top10_ranking.sort_values("skill_rating", ascending=False)
    region_top10_ranking["rank"] = region_top10_ranking.index + 1
    region_top10_ranking = region_top10_ranking.loc[:, ["rank", "region", "skill_rating"]]
    region_top10_ranking.to_csv(join(saving_dir, "region_top10_ranking.csv"), index=False)

def _save_team_ranking(
    ranking: pd.DataFrame, 
    saving_dir: str
) -> None:
    team_ranking = ranking.groupby("team_name").agg(
        {
            "region": "first",
            "nb_games": "mean",
            "last_game_date": "first",
            "skill_rating": "mean",
        }
    ).reset_index()
    team_ranking = team_ranking.sort_values("skill_rating", ascending=False)
    team_ranking["rank"] = team_ranking.index + 1
    team_ranking = team_ranking.loc[:, ["rank", "team_name", "region", "nb_games", "last_game_date", "skill_rating"]]
    team_ranking.to_csv(join(saving_dir, "team_ranking.csv"), index=False)

def evaluate_ranking(ranking: pd.DataFrame, experiment_dir: str) -> None:
    data_dir = join(ARTIFACTS_DIR, "data", "survey")
    
    player_id_to_rating_mapping = ranking[["player_id", "skill_rating"]].set_index("player_id").to_dict()["skill_rating"]

    metrics = {}
    for ranking_kind in ["global", "europe", "north_america", "china", "korea"]:


        player_comparison_df = pd.read_csv(join(data_dir, "questions", f"{ranking_kind}_survey_questions.csv"))

        player_comparison_df["player_0_rating"] = player_comparison_df.player_id_0.apply(lambda x: player_id_to_rating_mapping[x])
        player_comparison_df["player_1_rating"] = player_comparison_df.player_id_1.apply(lambda x: player_id_to_rating_mapping[x])
        player_comparison_df["model_player_0_is_better"] = player_comparison_df["player_0_rating"] > player_comparison_df["player_1_rating"]

        survey_results_df = pd.read_csv(join(data_dir, "answers", f"{ranking_kind}_survey_answers.csv"))

        expert_names = survey_results_df.columns.tolist()
        survey_results_df = survey_results_df.apply(lambda col: col == player_comparison_df["player_name_0"])
        nb_experts = len(survey_results_df.columns)
        survey_results_df.columns = [f"expert_{i}_player_0_is_better" for i in range(0, nb_experts)]

        player_comparison_df = pd.concat([player_comparison_df, survey_results_df], axis=1)

        nb_unique_players = len(
            set(
                player_comparison_df["player_id_0"].unique().tolist() + 
                player_comparison_df["player_id_1"].unique().tolist()
            )
        )

        metrics[ranking_kind] = {
            **_compute_concordance_metrics(player_comparison_df, expert_names),
            "nb_unique_players": int(nb_unique_players),
            "nb_experts": int(nb_experts),
        }

        if "skill_rating_mu" in ranking.columns:
            openskill_metrics = _openskill_ranking_evaluation(ranking, player_comparison_df, expert_names, ranking_kind, experiment_dir)
            metrics[ranking_kind]["openskill_metrics"] = openskill_metrics
    
    save_yaml(metrics, experiment_dir, "ranking_experts_evaluation.yaml")


def _compute_concordance_metrics(player_comparison_df: pd.DataFrame, expert_names: list[str]) -> dict:
    nb_experts = len(expert_names)

    def majority_voting(row):
        expert_opinions = [row[f"expert_{i}_player_0_is_better"] for i in range(nb_experts)]
        if sum(expert_opinions) == nb_experts / 2:
            return None
        else:
            return sum(expert_opinions) > nb_experts / 2

    player_comparison_df[f"expert_majority_player_0_is_better"] = player_comparison_df.apply(
        majority_voting, axis=1
    )

    def check_unanimous(row):
        expert_opinions = [row[f"expert_{i}_player_0_is_better"] for i in range(nb_experts)]
        return all(expert_opinions) or all(x == False for x in expert_opinions)
    player_comparison_df["experts_are_unanimous"] = player_comparison_df.apply(check_unanimous, axis=1)

    player_comparison_df["model_expert_concordance"] = player_comparison_df["model_player_0_is_better"] == (player_comparison_df["expert_majority_player_0_is_better"])
    player_comparison_df = player_comparison_df[~player_comparison_df.loc[:, "expert_majority_player_0_is_better"].isna()]

    unanimous_experts_ratio = player_comparison_df["experts_are_unanimous"].mean()

    majority_concordance = player_comparison_df["model_expert_concordance"].mean()
    unanimous_concordance = player_comparison_df[player_comparison_df["experts_are_unanimous"]]["model_expert_concordance"].mean()
    partial_concordance = player_comparison_df[~player_comparison_df["experts_are_unanimous"]]["model_expert_concordance"].mean()
    per_expert_concordance = {
        expert_name: float((
            player_comparison_df[f"expert_{i}_player_0_is_better"]
            == player_comparison_df["model_player_0_is_better"]
        ).mean())
        for i, expert_name in enumerate(expert_names)
    }
    return {
        "majority_concordance": float(majority_concordance),
        "unanimous_concordance": float(unanimous_concordance),
        "partial_concordance": float(partial_concordance),
        "per_expert_concordance": per_expert_concordance,
        "unanimous_experts_ratio": float(unanimous_experts_ratio),
    }

def _openskill_ranking_evaluation(ranking: pd.DataFrame, player_comparison_df: pd.DataFrame, expert_names: list, ranking_kind: str, experiment_dir: str) -> None:
    player_comparison_df = player_comparison_df.copy()
    player_id_to_rating_mapping = ranking[["player_id", "skill_rating_mu"]].set_index("player_id").to_dict()["skill_rating_mu"]
    player_id_to_rating_sigma_mapping = ranking[["player_id", "skill_rating_sigma"]].set_index("player_id").to_dict()["skill_rating_sigma"]

    player_comparison_df["player_0_rating_mu"] = player_comparison_df.player_id_0.apply(lambda x: player_id_to_rating_mapping[x])
    player_comparison_df["player_1_rating_mu"] = player_comparison_df.player_id_1.apply(lambda x: player_id_to_rating_mapping[x])
    player_comparison_df["player_0_rating_sigma"] = player_comparison_df.player_id_0.apply(lambda x: player_id_to_rating_sigma_mapping[x])
    player_comparison_df["player_1_rating_sigma"] = player_comparison_df.player_id_1.apply(lambda x: player_id_to_rating_sigma_mapping[x])

    mu_diff = player_comparison_df['player_0_rating_mu'] - player_comparison_df['player_1_rating_mu']
    sigma_combined = np.sqrt(player_comparison_df['player_0_rating_sigma']**2 + player_comparison_df['player_1_rating_sigma']**2)
    z_score = mu_diff / sigma_combined
    player_comparison_df['model_player_0_is_better'] = norm.cdf(z_score)

    player_comparison_df = player_comparison_df.dropna() # nan values come from strict equality in majority voting

    y_prob_list = player_comparison_df['model_player_0_is_better'].values.tolist()
    y_true_list = player_comparison_df['expert_majority_player_0_is_better'].values.tolist()
    nbins = int(np.sqrt(len(y_true_list)))
    ece = compute_ece(y_true_list, y_prob_list, nbins)

    experiment_dir = join(experiment_dir, "rankings")
    plot_model_calibration(
        y_true_list, 
        y_prob_list, 
        nbins, 
        f"Calibration plot for predicting which player is better from skill rating - {ranking_kind}", 
        experiment_dir, 
        f"better_player_prediction_calibration_from_rating_{ranking_kind}.png"
    )

    expert_unanimous_mask = player_comparison_df[player_comparison_df["experts_are_unanimous"]].index.tolist()
    y_prob_list = player_comparison_df.loc[expert_unanimous_mask, 'model_player_0_is_better'].values.tolist()
    y_true_list = player_comparison_df.loc[expert_unanimous_mask, 'expert_majority_player_0_is_better'].values.tolist()
    ece_unanimous = compute_ece(y_true_list, y_prob_list, nbins)
    plot_model_calibration(
        y_true_list, 
        y_prob_list, 
        nbins, 
        f"Calibration plot for predicting which player is better from skill rating - {ranking_kind} - Experts unanimous", 
        experiment_dir, 
        f"better_player_prediction_calibration_from_rating_experts_unanimous_{ranking_kind}.png"
    )

    player_comparison_df['model_player_0_is_better'] = player_comparison_df['model_player_0_is_better'] > 0.5
    concordance_metrics = _compute_concordance_metrics(player_comparison_df, expert_names)

    metrics = {
        "ece": float(ece),
        "ece_unanimous": float(ece_unanimous),
        **concordance_metrics
    }

    return metrics
