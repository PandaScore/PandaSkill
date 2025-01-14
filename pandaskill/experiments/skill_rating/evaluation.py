from dateutil.relativedelta import relativedelta
from datetime import datetime
import logging
from pandaskill.experiments.general.metrics import *
from pandaskill.experiments.general.utils import *
from pandaskill.experiments.general.visualization import plot_model_calibration
import itertools
import numpy as np
import pandas as pd
from progressbar.progressbar import ProgressBar
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def evaluate_skill_ratings(
    data_with_ratings: pd.DataFrame, 
    experiment_dir: str, 
    evaluation_config: dict
) -> None:
    game_id_test_list, y_test_list, y_prob_list, coefs_list, start_test_date_list = _rolling_forecast_game_outcome(
        data_with_ratings, evaluation_config
    )

    metrics = {
        "coefs": {
            "mean": np.mean(coefs_list, axis=0).tolist(), 
            "std": np.std(coefs_list, axis=0).tolist()
        },
        "coefs_rolling_window": {
            str(date): coefs.tolist()
            for date, coefs in zip(start_test_date_list, coefs_list)
        },
    }

    inter_region_metrics = _compute_intra_inter_region_metrics(
        data_with_ratings, 
        game_id_test_list, 
        y_test_list, 
        y_prob_list,
        "inter",
        experiment_dir
    )
    intra_region_metrics = _compute_intra_inter_region_metrics(
        data_with_ratings, 
        game_id_test_list, 
        y_test_list, 
        y_prob_list,
        "intra",
        experiment_dir
    )
    region_change_metrics = _compute_region_change_metrics(
        data_with_ratings, 
        game_id_test_list, 
        y_test_list, 
        y_prob_list,
        experiment_dir
    )
    role_ratings_distribution_metric = _compute_role_ratings_distribution_metrics(
        data_with_ratings
    )

    metrics = {
        **metrics,
        **_compute_metrics(y_test_list, y_prob_list),
        "interregion_metrics": inter_region_metrics,
        "intra_region_metrics": intra_region_metrics,
        "region_change_metrics": region_change_metrics,
        "role_ratings_distributions_metrics": role_ratings_distribution_metric
    }    

    plot_model_calibration(
        y_test_list, 
        y_prob_list, 
        25, 
        f"Calibration plot for game outcome forecasting from skill rating", 
        experiment_dir, 
        f"skill_rating_outcome_forecasting_calibration"
    )

    save_yaml(metrics, experiment_dir, "skill_ratings_metrics.yaml")

def _rolling_forecast_game_outcome(
    data_with_ratings: pd.DataFrame, evaluation_config: dict
) -> tuple[list[int], list[int], list[float], list[float], list[datetime]]:
    game_eval_df = _format_data_for_rolling_game_forecast(data_with_ratings, evaluation_config)

    last_date = game_eval_df['date'].max()
    start_testing_date = datetime.strptime(evaluation_config["end_warmup_date"], "%Y-%m-%d") + relativedelta(years=1)
    window_size = relativedelta(months=1)
    rolling_window_date_range = pd.date_range(start=start_testing_date, end=last_date, freq='MS')

    game_id_test_list = []
    y_test_list = []
    y_prob_list = []
    coefs_list = []
    start_test_date_list = []    

    for start_test_date in ProgressBar()(rolling_window_date_range):
        end_test_date = start_test_date + window_size
        start_training_date = start_test_date - relativedelta(years=1)
        
        train_data = game_eval_df[(game_eval_df['date'] >= start_training_date) & (game_eval_df['date'] < start_test_date)]
        test_data = game_eval_df[(game_eval_df['date'] >= start_test_date) & (game_eval_df['date'] < end_test_date)]
        
        if len(train_data) == 0 or len(test_data) == 0:
            logging.warning(f"Skipping {start_test_date} due to lack of data")
            continue
        
        X_train = train_data.drop('date', axis=1).values
        X_test = test_data.drop('date', axis=1).values
        
        # mirror team sides and outcomes (assuming blue/red side balance)
        y_train = [0] * len(X_train) + [1] * len(X_train)
        y_test = [0] * len(X_test) + [1] * len(X_test)    
        X_train = np.concatenate([X_train, np.concatenate([X_train[:,5:], X_train[:, :5]], axis=1)], axis=0)
        X_test = np.concatenate([X_test, np.concatenate([X_test[:,5:], X_test[:, :5]], axis=1)], axis=0)
        game_id_test = np.concatenate([test_data.index.values, test_data.index.values], axis=0)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = LogisticRegression(C=evaluation_config["C"]).fit(X_train_scaled, y_train)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

        game_id_test_list.extend(game_id_test)
        y_test_list.extend(y_test)
        y_prob_list.extend(y_prob)
        coefs_list.append(model.coef_[0])
        start_test_date_list.append(start_test_date)

    return game_id_test_list, y_test_list, y_prob_list, coefs_list, start_test_date_list
    
def _format_data_for_rolling_game_forecast(
    data_with_ratings: pd.DataFrame, evaluation_config: dict
) -> pd.DataFrame:
    role_per_team_columns = [f"{role}_{team}" for team in [0, 1] for role in ROLES]

    def compute_data_for_game(game):
        game = game.sort_values(["win", "role_order"])
        return pd.DataFrame(data=[[
            game.date.values[0],
            *game.skill_rating_before.values,
        ]], columns=["date", *role_per_team_columns])

    end_warmup_date = evaluation_config["end_warmup_date"]

    eval_df = data_with_ratings.copy()
    eval_df = eval_df[eval_df.date > end_warmup_date]

    eval_df.loc[:, "role_order"] = eval_df.role.map({
        role: i
        for i, role in enumerate(ROLES)
    })

    game_eval_df = eval_df.groupby("game_id").apply(compute_data_for_game)
    game_eval_df = game_eval_df.reset_index(level=1, drop=True)
    game_eval_df['date'] = pd.to_datetime(game_eval_df['date'])
    game_eval_df = game_eval_df.sort_values('date')

    return game_eval_df

def _compute_intra_inter_region_metrics(
    data_with_ratings: pd.DataFrame, 
    game_id_test_list: list[int], 
    y_test_list: list[int], 
    y_prob_list: list[float],
    kind: str,
    experiment_dir: str
) -> dict:
    if kind == "inter":
        region_agg_rule = lambda x: len(set(x)) > 1
    elif kind == "intra":
        region_agg_rule = lambda x: len(set(x)) == 1
    else:
        raise ValueError(f"`kind` can only be one of 'inter' or 'intra', got {kind} instead.")
    
    region_agg_games_df = data_with_ratings.groupby("game_id").agg({"region": region_agg_rule})
    region_agg_game_ids = region_agg_games_df[region_agg_games_df.values].index.values
    region_agg_game_ids_test_list_index = np.where(np.isin(game_id_test_list, region_agg_game_ids))[0]
    nb_region_agg_games_per_series = data_with_ratings.loc[region_agg_games_df[region_agg_games_df.values].index].series_name.value_counts() // 10
    
    y_test_list_region_agg = np.array(y_test_list)[region_agg_game_ids_test_list_index]
    y_prob_list_region_agg = np.array(y_prob_list)[region_agg_game_ids_test_list_index]

    region_aggmetrics = _compute_metrics(y_test_list_region_agg, y_prob_list_region_agg)
    region_aggmetrics["nb_games"] = len(region_agg_game_ids)

    plot_model_calibration(
        y_test_list_region_agg, 
        y_prob_list_region_agg, 
        25, 
        f"Calibration plot for {kind}-region game outcome forecasting from skill rating", 
        experiment_dir, 
        f"skill_rating_{kind}_region_outcome_forecasting_calibration"
    )

    return region_aggmetrics

def _compute_region_change_metrics(
    data_with_ratings: pd.DataFrame,
    game_id_test_list: list[int],
    y_test_list: list[int],
    y_prob_list: list[float],
    experiment_dir: str
) -> dict:
    game_ids_with_region_change = data_with_ratings.groupby("game_id")["region_change"].agg(lambda x: any(x)).index.values
    region_change_game_ids_test_list_index = np.where(np.isin(game_id_test_list, game_ids_with_region_change))[0]
    
    y_test_list_region_change = np.array(y_test_list)[region_change_game_ids_test_list_index]
    y_prob_list_region_change = np.array(y_prob_list)[region_change_game_ids_test_list_index]

    region_change_metrics = _compute_metrics(y_test_list_region_change, y_prob_list_region_change)
    region_change_metrics["nb_games"] = len(game_ids_with_region_change)

    plot_model_calibration(
        y_test_list_region_change, 
        y_prob_list_region_change, 
        25, 
        f"Calibration plot for game outcome forecasting from skill rating - game with region change", 
        experiment_dir, 
        f"skill_rating_region_change_outcome_forecasting_calibration"
    )

    return region_change_metrics

def _compute_metrics(y_test_list: list[int], y_prob_list: list[float]) -> dict:
    accuracy = sum(np.array(np.array(y_prob_list) > 0.5) == np.array(y_test_list)) / len(y_test_list)
    ece = compute_ece(y_test_list, y_prob_list, 25)

    metrics = {
        'accuracy': float(accuracy),
        'ece': float(ece),
    }

    return metrics 

def _compute_role_ratings_distribution_metrics(data_with_ratings: pd.DataFrame) -> dict:
    role_pairs = list(itertools.combinations(ROLES, 2))
    wasserstein_disance_list = []
    for role_1, role_2 in role_pairs:
        wasserstein_disance = stats.wasserstein_distance(
            data_with_ratings[data_with_ratings.role == role_1].skill_rating_after,
            data_with_ratings[data_with_ratings.role == role_2].skill_rating_after
        )
        wasserstein_disance_list.append(wasserstein_disance)
        
    wasserstein_disance_mean = np.mean(wasserstein_disance_list)
    wasserstein_disance_std = np.std(wasserstein_disance_list)

    return {
        "wasserstein_disance": {
            "mean": float(wasserstein_disance_mean),
            "std": float(wasserstein_disance_std)
        }
    }